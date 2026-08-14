// v1: shared memory tiling —— 在 v0 基础上加一层 smem 复用。
//
// v0 的问题：每个 warp 都独自从 global 读自己那份 A/B 的 fragment，没有复用。
// 同一行 A 的 fragment 会被沿 N 方向排开的每个 warp 各读一遍，global 流量被放大。
//
// 这版的做法：一个 block 协作把 A 的 BM×BK、B 的 BK×BN 搬进 shared memory，
// block 内所有 warp 都从 smem 读 fragment，global 读取次数除以 tile 边长。
// 每个 warp 仍然只负责一个 16×16 的输出 tile，和 v0 一样——
// 区别只在于 fragment 从 smem 取，而不是从 global 取。
//
// 实测 24 TFLOP/s —— 比 v0 的 27 还慢一点。这个结果值得看清楚，它说明"加了 smem
// 复用"本身不等于变快。ncu 对比两版（被测 kernel，不含 cuBLAS 参考）：
//
//                      v0          v1        说明
//   global load      1.34 亿     1678 万    smem 复用生效，指令数降到 1/8
//   L2 读 sector     3300 万     1390 万    L2 流量降到 2.4 分之一
//   DRAM 读           16.8 MB    16.8 MB    完全一样，都是 A/B 各读一遍
//   occupancy          94%         49%      ← 代价在这里
//
// 访存该省的都省了，但 occupancy 掉了一半：512 线程 + 4 KB smem，每个 SM 只能放 6 个
// block；v0 没有 smem 限制，能放 32 个。而 2048³ 的工作集在 L2 里命中率很高，v0 那些
// 冗余的 global 请求几乎全被 L2 吃掉了，压力落在 L2 带宽而不是 DRAM 上，没那么疼。
// 于是"高 occupancy 掩盖延迟"的收益盖过了"少读一点 L2"的收益。
//
// 换句话说，tiling 的价值不在这一版体现，要等 tile 足够大、warp 内复用足够多之后
// 才兑现——v2 只是把 warp tile 放大，一步就到 76 TFLOP/s。这一版是那个结构的地基。
//
// 三个瓶颈留给后面：
//   v2: 一个 warp 只算 16×16 太少，搬进 smem 的数据在寄存器里没有复用
//   v3: As 的 leading dimension = BK = 16 half = 32B，wmma 读 smem 有 bank conflict
//   v5: 搬运和计算被两个 __syncthreads 完全串起来，谁都盖不住谁
#include <cuda_fp16.h>
#include <mma.h>

#include "common.h"

using namespace nvcuda;

// block 级 tile：每个 block 算 BM×BN 的输出，K 方向每次推进 BK
constexpr int BM = 64, BN = 64, BK = 16;

// 每个 warp 负责一个 16×16 的 wmma tile，block 内排成 WARPS_M × WARPS_N 的网格
constexpr int WARPS_M = BM / WMMA_M;          // 4
constexpr int WARPS_N = BN / WMMA_N;          // 4
constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;  // 16 warp = 512 线程

__global__ void matmul_v1(const half* __restrict__ A, const half* __restrict__ B,
                          float* __restrict__ C, int m, int n, int k) {
    __shared__ half As[BM * BK];  // 静态 smem，编译期已知大小，一共 4 KB
    __shared__ half Bs[BK * BN];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int warp_m = warp / WARPS_N;  // 本 warp 的 tile 在 block 内的行号
    const int warp_n = warp % WARPS_N;  // 列号
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // accumulator 全程留在寄存器里，K 循环结束才写回 global
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < k; k0 += BK) {
        // ---- global -> smem ----
        // 每个线程一次搬 8 个 half（float4，16B），这是单条指令能搬的最大宽度。
        // 于是 A 的 tile 被切成 BM×(BK/8) 个小块，512 个线程 grid-stride 领完。
        // 从一维块号 i 还原二维坐标：行 = i / (BK/8)，列 = (i % (BK/8)) * 8。
        for (int i = tid; i < BM * (BK / 8); i += NUM_THREADS) {
            const int r = i / (BK / 8), c = (i % (BK / 8)) * 8;
            *(float4*)&As[r * BK + c] = *(const float4*)&A[(size_t)(block_row + r) * k + k0 + c];
        }
        for (int i = tid; i < BK * (BN / 8); i += NUM_THREADS) {
            const int r = i / (BN / 8), c = (i % (BN / 8)) * 8;
            *(float4*)&Bs[r * BN + c] = *(const float4*)&B[(size_t)(k0 + r) * n + block_col + c];
        }
        __syncthreads();  // 等 tile 搬完，才能开始读

        // ---- smem -> 寄存器 -> tensor core ----
        // BK == WMMA_K，所以每个 K 步恰好一次 mma，不需要内层 K 循环。
        // load_matrix_sync 的最后一个参数是源矩阵的 leading dimension：
        // A 的 tile 在 smem 里按行存，行距 BK；B 的行距 BN。
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, &As[warp_m * WMMA_M * BK], BK);
        wmma::load_matrix_sync(b_frag, &Bs[warp_n * WMMA_N], BN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();  // 下一轮要覆盖 smem，得等所有 warp 都读完
    }

    const size_t row = block_row + warp_m * WMMA_M;
    const size_t col = block_col + warp_n * WMMA_N;
    wmma::store_matrix_sync(&C[row * n + col], c_frag, n, wmma::mem_row_major);
}

int main() {
    const size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;

    half* h_A = (half*)malloc(a_elems * sizeof(half));
    half* h_B = (half*)malloc(b_elems * sizeof(half));
    float* h_C = (float*)malloc(c_elems * sizeof(float));
    float* h_ref = (float*)malloc(c_elems * sizeof(float));
    fill_inputs(h_A, a_elems, h_B, b_elems);

    printf("v1 smem tiling  (M=%d, N=%d, K=%d)\n", M, N, K);
    printf("  block tile %dx%dx%d, %d warp, 每 warp 1 个 %dx%d tile, smem %zu B\n", BM, BN, BK,
           NUM_THREADS / 32, WMMA_M, WMMA_N, (BM * BK + BK * BN) * sizeof(half));

    half *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, a_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, b_elems * sizeof(half), cudaMemcpyHostToDevice));

    // 先借 d_C 让 cuBLAS 算一遍参考结果，拷回主机后这块显存就可以覆盖了
    cublas_reference(d_A, d_B, d_C, h_ref);
    // 填 sentinel 而不是 0：漏算的 tile 会以 3.4e38 暴露，而不是伪装成合理的小数
    CUDA_CHECK(cudaMemset(d_C, 0x7F, c_elems * sizeof(float)));

    // 一个 block 出 BM×BN 的输出，所以 grid 铺满 C。grid.x 沿 N，grid.y 沿 M。
    dim3 block(NUM_THREADS);
    dim3 grid(CEIL(N, BN), CEIL(M, BM));

    // CUDA 12 默认 lazy module loading：首次 launch 才把 kernel 模块加载进来，
    // 这笔一次性开销会整个落在单次计时里（实测能把 v3 从 130 TFLOP/s 压到 87）。
    // 先查一次 kernel 属性触发加载，把它挤出计时区间。这不是预热——kernel 本身没执行过，
    // cache 和 TLB 仍然是冷的，和 ncu 看到的那一次一致。
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, matmul_v1));

    // 只启动一次、不预热，打印的就是 ncu 会 profile 到的那一次
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    matmul_v1<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    report_perf(ms);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
    const bool pass = check_result(h_C, h_ref);

    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return pass ? 0 : 1;
}
