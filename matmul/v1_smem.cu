// v1: shared memory tiling —— 在 v0 基础上加一层 smem 复用。
//
// v0 的问题：每个 warp 都独自从 global 读自己那份 A/B 的 fragment，没有复用。
// 同一行 A 的 fragment 会被沿 N 方向排开的每个 warp 各读一遍，global 流量被放大。
//
// 这版的做法：一个 block 协作把 A 的 BM×BK、B 的 BK×BN 搬进 shared memory，
// block 内的所有 warp 都从 smem 读 fragment，global 读取次数除以 tile 边长。
// 每个 warp 仍然只负责一个 16×16 的输出 tile，和 v0 一样——
// 区别只在于 fragment 从 smem 取，而不是从 global 取。
//
// 有意没做两件事，留给后续：
//   - 寄存器分块（register blocking）：让一个 warp 算多个 16×16 tile，
//     载入的 fragment 在寄存器里复用，进一步提升 tensor core 利用率。
//   - shared memory padding：现在 As 的 ld = BK = 16，wmma 从 smem 取数有 bank conflict。
#include <cuda_fp16.h>
#include <mma.h>

#include "common.h"

using namespace nvcuda;

constexpr int M = 2048, N = 2048, K = 2048;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

// block 级 tile：每个 block 算 BM×BN 的输出，K 方向每次推进 BK
constexpr int BM = 64, BN = 64, BK = 16;
// 每个 warp 负责一个 16×16 的 wmma tile，block 内排成 WARPS_M × WARPS_N 的网格
constexpr int WARPS_M = BM / WMMA_M;              // 4
constexpr int WARPS_N = BN / WMMA_N;              // 4
constexpr int NUM_WARPS = WARPS_M * WARPS_N;      // 16
constexpr int NUM_THREADS = NUM_WARPS * 32;       // 512

constexpr int ELEMS_PER_LOAD = 8;  // 每次 float4 搬 8 个 half（16B），global load 的最大宽度

__global__ void matmul_v1(const half* __restrict__ A, const half* __restrict__ B,
                          float* __restrict__ C, int m, int n, int k) {
    __shared__ half As[BM * BK];   // 静态 smem，编译期已知大小，无需动态配额
    __shared__ half Bs[BK * BN];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int warp_m = warp / WARPS_N;  // 本 warp 的 tile 在 block 内的行号
    const int warp_n = warp % WARPS_N;  // 列号
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // 单个 accumulator 全程留在寄存器里，K 循环结束才写回 global
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < k; k0 += BK) {
        // ---- global -> smem，NUM_THREADS 个线程 grid-stride 协作搬运，float4 向量化 ----
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
        // BK == WMMA_K，所以每个 K 步恰好一次 mma，不需要内层循环。
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

    dim3 block(NUM_THREADS);
    dim3 grid(CEIL(N, BN), CEIL(M, BM));
    matmul_v1<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
