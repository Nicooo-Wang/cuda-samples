// v4: tile 尺寸调优 —— 只把 BK 从 32 提到 64，其余和 v3 一样。
//
// 实测 130 -> 223 TFLOP/s。为什么加深 K 方向有用：
//
// 1) 同步次数减半。K 循环每一轮有两个 __syncthreads()，BK 翻倍就意味着轮数减半，
//    2048 / 32 = 64 轮变成 32 轮。这一版还没做流水线，搬运和计算完全串行，
//    每一轮的同步开销都是纯浪费，轮数少一半就直接省一半。
//
// 2) 每次搬运的粒度更大，global 访问更连续。A 的 tile 一行从 32 个 half（64 字节）
//    变成 64 个 half（128 字节），正好是一个 L2 cache line 的宽度。
//
// 3) fragment 在寄存器里被复用的次数不变，但装载和 mma 的比例更好摊开：
//    BK=64 时 block tile 内部沿 K 走 4 步（BK / WMMA_K），指令级并行的余地更大。
//
// 注意 global 流量并没有变。每个输出元素分摊的 global 读是 (1/BM + 1/BN)，
// 只跟 block tile 的 M/N 有关，和 BK 无关。BK 变大只是把同样的读取次数
// 用更少、更大的批次完成。
//
// 这一版实测的几个方向（改本文件顶部的常量就能复现，M=N=K=2048 单次冷跑）：
//   BM   BN   BK   warp tile   线程   smem      TFLOP/s
//   128  128  16   64x32        256   10 KB      125
//   128  128  32   64x32        256   19 KB      129    <- v3
//   128  128  64   64x32        256   35 KB      224    <- 本版
//   128  128  64   64x64        128   35 KB      184
//   128  128  32   64x64        128   19 KB      170
//   128  128  64   32x32        512   35 KB      118
//    64   64  64   32x32        128   18 KB       98
//
// 三个值得注意的地方：
//
// BK 从 32 到 64 是 1.7 倍，但从 16 到 32 只有 125 -> 129。收益不是线性的：
// BK 太小时同步开销占比高，加深立刻见效；到了 64 已经接近"搬运批次足够大"的拐点。
// 再往上 BK=128 需要 68 KB smem，超过静态 __shared__ 的 48 KB 上限，编译期就被
// ptxas 拒掉（"uses too much shared data"）。要用满 Hopper 每 SM 的 228 KB，
// 必须改成 extern __shared__ 动态申请并调 cudaFuncSetAttribute 显式 opt-in，v5 会用到。
//
// warp tile 往两个方向调都更差。缩到 32x32 是 118：512 线程里每个 warp 只有 2x2 = 4 个
// accumulator，寄存器复用不足，退回 v2 的老问题。放大到 64x64 是 184：block 只剩
// 4 个 warp = 128 线程，2048 规模下 grid 只有 16x16 = 256 个 block 铺 132 个 SM，
// 波次本来就浅，occupancy 再降就盖不住访存延迟。
//
// 但 64x64 在 v5 里会翻正（249 > 230）——那时 cp.async 已经把访存延迟藏起来了，
// occupancy 不再是关键，寄存器复用更充分的一侧就赢了。所以 tile 参数没有绝对最优，
// 只有"在当前流水线结构和当前矩阵规模下最优"。调参必须和结构一起调。
#include <cuda_fp16.h>
#include <mma.h>

#include "common.h"

using namespace nvcuda;

constexpr int BM = 128, BN = 128, BK = 64;  // 这一版唯一的改动：BK 32 -> 64
constexpr int WARP_M = 64, WARP_N = 32;
constexpr int WTILE_M = WARP_M / WMMA_M;  // 4
constexpr int WTILE_N = WARP_N / WMMA_N;  // 2
constexpr int WARPS_M = BM / WARP_M;      // 2
constexpr int WARPS_N = BN / WARP_N;      // 4
constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;  // 256

constexpr int PAD = 8;
constexpr int LDA = BK + PAD;  // 72
constexpr int LDB = BN + PAD;  // 136

__global__ void matmul_v4(const half* __restrict__ A, const half* __restrict__ B,
                          float* __restrict__ C, int m, int n, int k) {
    // 18 KB + 17 KB = 35 KB，还在静态 48 KB 上限内。再往上就要动态申请了。
    __shared__ half As[BM * LDA];
    __shared__ half Bs[BK * LDB];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WTILE_M][WTILE_N];
#pragma unroll
    for (int i = 0; i < WTILE_M; ++i)
#pragma unroll
        for (int j = 0; j < WTILE_N; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    // 2048 / 64 = 32 轮，v3 是 64 轮
    for (int k0 = 0; k0 < k; k0 += BK) {
        // ---- global -> smem ----
        // 代码和 v3 一字不差，只是 BK 变了，所以每个线程分到的搬运量翻倍：
        // As 有 128 x (64/8) = 1024 个 float4 块，256 个线程各领 4 个。
        for (int i = tid; i < BM * (BK / 8); i += NUM_THREADS) {
            const int r = i / (BK / 8), c = (i % (BK / 8)) * 8;
            *(float4*)&As[r * LDA + c] = *(const float4*)&A[(size_t)(block_row + r) * k + k0 + c];
        }
        for (int i = tid; i < BK * (BN / 8); i += NUM_THREADS) {
            const int r = i / (BN / 8), c = (i % (BN / 8)) * 8;
            *(float4*)&Bs[r * LDB + c] = *(const float4*)&B[(size_t)(k0 + r) * n + block_col + c];
        }
        __syncthreads();

        // ---- smem -> 寄存器 -> tensor core ----
        // BK / WMMA_K = 4，block tile 内部沿 K 走 4 步（v3 是 2 步）
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WTILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WTILE_N];
#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
#pragma unroll
            for (int i = 0; i < WTILE_M; ++i)
                wmma::load_matrix_sync(a_frag[i], &As[(warp_m * WARP_M + i * WMMA_M) * LDA + kk],
                                       LDA);
#pragma unroll
            for (int j = 0; j < WTILE_N; ++j)
                wmma::load_matrix_sync(b_frag[j], &Bs[kk * LDB + warp_n * WARP_N + j * WMMA_N], LDB);
#pragma unroll
            for (int i = 0; i < WTILE_M; ++i)
#pragma unroll
                for (int j = 0; j < WTILE_N; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < WTILE_M; ++i)
#pragma unroll
        for (int j = 0; j < WTILE_N; ++j) {
            const size_t row = block_row + warp_m * WARP_M + i * WMMA_M;
            const size_t col = block_col + warp_n * WARP_N + j * WMMA_N;
            wmma::store_matrix_sync(&C[row * n + col], acc[i][j], n, wmma::mem_row_major);
        }
}

int main() {
    const size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;

    half* h_A = (half*)malloc(a_elems * sizeof(half));
    half* h_B = (half*)malloc(b_elems * sizeof(half));
    float* h_C = (float*)malloc(c_elems * sizeof(float));
    float* h_ref = (float*)malloc(c_elems * sizeof(float));
    fill_inputs(h_A, a_elems, h_B, b_elems);

    printf("v4 tile tuning  (M=%d, N=%d, K=%d)\n", M, N, K);
    printf("  block tile %dx%dx%d, %d warp, K 循环 %d 轮, smem %zu B\n", BM, BN, BK,
           NUM_THREADS / 32, K / BK, (BM * LDA + BK * LDB) * sizeof(half));

    half *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, a_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, b_elems * sizeof(half), cudaMemcpyHostToDevice));

    cublas_reference(d_A, d_B, d_C, h_ref);
    CUDA_CHECK(cudaMemset(d_C, 0x7F, c_elems * sizeof(float)));

    dim3 block(NUM_THREADS);
    dim3 grid(CEIL(N, BN), CEIL(M, BM));

    // CUDA 12 默认 lazy module loading：首次 launch 才把 kernel 模块加载进来，
    // 这笔一次性开销会整个落在单次计时里（实测能把 v3 从 130 TFLOP/s 压到 87）。
    // 先查一次 kernel 属性触发加载，把它挤出计时区间。这不是预热——kernel 本身没执行过，
    // cache 和 TLB 仍然是冷的，和 ncu 看到的那一次一致。
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, matmul_v4));

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    matmul_v4<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
