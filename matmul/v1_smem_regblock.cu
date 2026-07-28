// v1: shared memory tiling + register blocking
//
// v0 的问题：A 的每一行 tile 会被 N/16 个 warp 各自从 global 读一遍，
// 实测 L1 流量约 4.29 GB（理论只需读 A+B = 16.7 MB），放大 ~256 倍。
// ncu 上表现为 l1tex__throughput ~100%，而 tensor pipe 只有 ~4%：
// tensor core 在等数据，瓶颈是 L1/LSU 数据通路，不是显存带宽。
//
// 两级复用把流量降下来：
//   1. shared memory tiling：一个 block 协作把 A 的 BM×BK、B 的 BK×BN 搬进 smem，
//      block 内所有 warp 共享，global 读取次数除以 tile 边长。
//   2. register blocking：每个 warp 算 WM×WN 的输出（多个 wmma tile），
//      载入的 a_frag / b_frag 在寄存器里被复用 TN / TM 次，
//      TM+TN 次 smem load 换来 TM*TN 次 mma。这是 tensor 利用率能提上去的关键。
//
// 注意 smem 没有做 padding：As 的 ld = BK = 32 halfs = 64B，
// wmma 从 smem 取数时会有 bank conflict。消除它是下一步的事，
// 这一版先把注意力放在"两级复用"本身。
//
// 实测 global load sectors 从 134M 降到 8.4M（÷16），tensor pipe 从 4.4% 升到 29.5%。
#include <cuda_fp16.h>
#include <mma.h>

#include "common.h"

using namespace nvcuda;

constexpr int M = 2048, N = 2048, K = 2048;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

// block 级 tile：每个 block 算 BM×BN 的输出，K 方向每次推进 BK
constexpr int BM = 128, BN = 128, BK = 32;
// warp 级 tile：每个 warp 算 WM×WN 的输出
constexpr int WM = 64, WN = 64;
constexpr int WARPS_M = BM / WM;              // 2
constexpr int WARPS_N = BN / WN;              // 2
constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;  // 128
// 每个 warp 在 M / N 方向各持有几个 wmma tile 的 accumulator
constexpr int TM = WM / WMMA_M;  // 4
constexpr int TN = WN / WMMA_N;  // 4

constexpr int ELEMS_PER_LOAD = 8;  // 每次 float4 搬 8 个 half（16B），global load 的最大宽度

__global__ __launch_bounds__(NUM_THREADS) void matmul_v1(const half* __restrict__ A,
                                                         const half* __restrict__ B,
                                                         float* __restrict__ C, int m, int n,
                                                         int k) {
    extern __shared__ half smem[];
    half* As = smem;             // BM x BK
    half* Bs = smem + BM * BK;   // BK x BN

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // 每个 warp 的 TM×TN 个 accumulator 全程留在寄存器里，K 循环结束才写回 global
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < k; k0 += BK) {
        // ---- global -> smem，128 个线程 grid-stride 协作搬运，float4 向量化 ----
        for (int idx = tid; idx < BM * BK / ELEMS_PER_LOAD; idx += NUM_THREADS) {
            const int r = idx / (BK / ELEMS_PER_LOAD);
            const int c = (idx % (BK / ELEMS_PER_LOAD)) * ELEMS_PER_LOAD;
            *(float4*)&As[r * BK + c] = *(const float4*)&A[(size_t)(block_row + r) * k + k0 + c];
        }
        for (int idx = tid; idx < BK * BN / ELEMS_PER_LOAD; idx += NUM_THREADS) {
            const int r = idx / (BN / ELEMS_PER_LOAD);
            const int c = (idx % (BN / ELEMS_PER_LOAD)) * ELEMS_PER_LOAD;
            *(float4*)&Bs[r * BN + c] = *(const float4*)&B[(size_t)(k0 + r) * n + block_col + c];
        }
        __syncthreads();

        // ---- smem -> 寄存器 -> tensor core ----
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            // 先把这一小步 K 需要的 fragment 全部载入寄存器，再做 TM*TN 次 mma 复用它们
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[TM];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[TN];

            for (int i = 0; i < TM; ++i)
                wmma::load_matrix_sync(a_frag[i], &As[(warp_m * WM + i * WMMA_M) * BK + kk], BK);
            for (int j = 0; j < TN; ++j)
                wmma::load_matrix_sync(b_frag[j], &Bs[kk * BN + warp_n * WN + j * WMMA_N], BN);

            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();  // 下一轮要覆盖 smem，得等所有 warp 都读完
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            const size_t row = block_row + warp_m * WM + i * WMMA_M;
            const size_t col = block_col + warp_n * WN + j * WMMA_N;
            wmma::store_matrix_sync(&C[row * n + col], acc[i][j], n, wmma::mem_row_major);
        }
    }
}

int main() {
    const size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;

    half* h_A = (half*)malloc(a_elems * sizeof(half));
    half* h_B = (half*)malloc(b_elems * sizeof(half));
    float* h_C = (float*)malloc(c_elems * sizeof(float));
    float* h_ref = (float*)malloc(c_elems * sizeof(float));

    fill_inputs(h_A, a_elems, h_B, b_elems);

    printf("v1 smem tiling + regblock  (M=%d, N=%d, K=%d)\n", M, N, K);

    // ---- 1. 先算 CPU 参考 ----
    printf("  [1/2] CPU reference ...\n");
    fflush(stdout);
    cpu_matmul(h_A, h_B, h_ref, M, N, K);

    // ---- 2. 再跑 GPU，单轮 ----
    printf("  [2/2] GPU kernel ...\n");
    fflush(stdout);

    half *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, a_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, b_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0x7F, c_elems * sizeof(float)));

    // As (BM×BK) + Bs (BK×BN) = 16KB，在默认的 48KB 静态上限内
    const size_t smem_bytes = (size_t)(BM * BK + BK * BN) * sizeof(half);

    dim3 block(NUM_THREADS);
    dim3 grid(CEIL(N, BN), CEIL(M, BM));
    matmul_v1<<<grid, block, smem_bytes>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 3. 精度验证 ----
    const bool pass = check_result(h_C, h_ref, M, N);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return pass ? 0 : 1;
}
