// v2: warp tile / 寄存器分块 —— 让一个 warp 算多个 wmma tile。
//
// v1 的问题：一个 warp 只算一个 16×16 的输出 tile，读进来的 a_frag 和 b_frag 各用一次
// 就扔了。tensor core 的一次 mma 只要几个周期，而从 smem 装两个 fragment 要几十个周期，
// 于是绝大部分时间花在等 smem，tensor core 在空转。
//
// 这版的做法：一个 warp 负责 WARP_M×WARP_N = 64×32 的输出，也就是 4×2 = 8 个 wmma tile。
// 装进来的 4 个 a_frag 和 2 个 b_frag 在寄存器里两两配对，做 8 次 mma：
//
//        b_frag[0] b_frag[1]
//   a[0]   acc00     acc01     6 次 smem 装载 -> 8 次 mma
//   a[1]   acc10     acc11     v1 是 2 次装载 -> 1 次 mma
//   a[2]   acc20     acc21     计算/访存比从 0.5 提到 1.33
//   a[3]   acc30     acc31
//
// 这才是 block tile 能放大到 128×128 的前提。如果还是"一个 warp 一个 16×16 tile"，
// 128×128 需要 64 个 warp = 2048 线程，超过 block 上限 1024；现在只要 8 个 warp。
//
// 顺带一个好处：block tile 变大，同样的 C 元素数量下 global 读的 A/B 更少。
// 一个 BM×BN 的 block 每步读 BM×BK + BK×BN 个元素，算 BM×BN 个输出的 BK 步累加，
// 所以每个输出元素分摊到的 global 读是 (1/BN + 1/BM)。64×64 是 1/32，128×128 是 1/64。
//
// 实测 79 TFLOP/s（v1 是 22）。还差得远，因为 smem 有严重的 bank conflict，见 v3。
#include <cuda_fp16.h>
#include <mma.h>

#include "common.h"

using namespace nvcuda;

// block 级 tile：每个 block 算 128×128 的输出，K 方向每次推进 32
constexpr int BM = 128, BN = 128, BK = 32;

// warp 级 tile：每个 warp 算 64×32 的输出
constexpr int WARP_M = 64, WARP_N = 32;
constexpr int WTILE_M = WARP_M / WMMA_M;  // 4 —— 每个 warp 在 M 方向持有几个 wmma tile
constexpr int WTILE_N = WARP_N / WMMA_N;  // 2 —— N 方向

// block 内 warp 排成 2×4 的网格，一共 8 个 warp
constexpr int WARPS_M = BM / WARP_M;  // 2
constexpr int WARPS_N = BN / WARP_N;  // 4
constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;  // 8 warp = 256 线程

__global__ void matmul_v2(const half* __restrict__ A, const half* __restrict__ B,
                          float* __restrict__ C, int m, int n, int k) {
    __shared__ half As[BM * BK];  // 8 KB
    __shared__ half Bs[BK * BN];  // 8 KB

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // 8 个 accumulator 全程留在寄存器里。每个 fragment 是 8 个 float，
    // 所以这里占 8×8 = 64 个寄存器，是这一版寄存器压力的主要来源。
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WTILE_M][WTILE_N];
#pragma unroll
    for (int i = 0; i < WTILE_M; ++i)
#pragma unroll
        for (int j = 0; j < WTILE_N; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < k; k0 += BK) {
        // ---- global -> smem，和 v1 完全一样，只是 tile 变大了 ----
        for (int i = tid; i < BM * (BK / 8); i += NUM_THREADS) {
            const int r = i / (BK / 8), c = (i % (BK / 8)) * 8;
            *(float4*)&As[r * BK + c] = *(const float4*)&A[(size_t)(block_row + r) * k + k0 + c];
        }
        for (int i = tid; i < BK * (BN / 8); i += NUM_THREADS) {
            const int r = i / (BN / 8), c = (i % (BN / 8)) * 8;
            *(float4*)&Bs[r * BN + c] = *(const float4*)&B[(size_t)(k0 + r) * n + block_col + c];
        }
        __syncthreads();

        // ---- smem -> 寄存器 -> tensor core ----
        // BK = 32 > WMMA_K = 16，所以 block tile 内部还要沿 K 走 2 步。
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WTILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WTILE_N];
#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            // 先把这一步要用的 fragment 全部装进寄存器
#pragma unroll
            for (int i = 0; i < WTILE_M; ++i)
                wmma::load_matrix_sync(a_frag[i], &As[(warp_m * WARP_M + i * WMMA_M) * BK + kk], BK);
#pragma unroll
            for (int j = 0; j < WTILE_N; ++j)
                wmma::load_matrix_sync(b_frag[j], &Bs[kk * BN + warp_n * WARP_N + j * WMMA_N], BN);

            // 再做 WTILE_M × WTILE_N 次 mma，每个 fragment 被复用 WTILE_N / WTILE_M 次
#pragma unroll
            for (int i = 0; i < WTILE_M; ++i)
#pragma unroll
                for (int j = 0; j < WTILE_N; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        __syncthreads();
    }

    // 8 个 accumulator 逐个写回它在 C 里的位置
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

    printf("v2 warp tile  (M=%d, N=%d, K=%d)\n", M, N, K);
    printf("  block tile %dx%dx%d, %d warp, 每 warp %dx%d = %d 个 %dx%d tile, smem %zu B\n", BM, BN,
           BK, NUM_THREADS / 32, WTILE_M, WTILE_N, WTILE_M * WTILE_N, WMMA_M, WMMA_N,
           (BM * BK + BK * BN) * sizeof(half));

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
    CUDA_CHECK(cudaFuncGetAttributes(&attr, matmul_v2));

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    matmul_v2<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
