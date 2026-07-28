// v0: 最朴素的 wmma 版本 —— 每个 warp 负责一个 16x16 的输出 tile，
// A / B 的 fragment 每次都直接从 global memory 载入，没有任何数据复用。
//
// 来源：Section011/tensor.cu，这里修掉了原版的两个问题：
//   1. warpN 的映射写成了 blockIdx.y * blockDim.y / warpSize + threadIdx.y，
//      按运算优先级是 (blockIdx.y * 32) / 32 + threadIdx.y = blockIdx.y + threadIdx.y，
//      blockIdx.y 的步长被整除掉了，只覆盖了一小块 C，其余 72% 从未被写入。
//   2. accumulator 用 half：K 大了会失精甚至溢出（fp16 最大有限值 65504）。
//      在这个 kernel 里换成 float 是免费的，因为瓶颈根本不在 tensor core。
#include <cuda_fp16.h>
#include <mma.h>

#include "common.h"

using namespace nvcuda;

constexpr int M = 2048, N = 2048, K = 2048;  // C[M,N] = A[M,K] * B[K,N]，全部 row-major
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

__global__ void matmul_v0(const half* A, const half* B, float* C, int m, int n, int k) {
    const int warp_m = blockIdx.x;                             // M 方向 tile 下标
    const int warp_n = blockIdx.y * blockDim.y + threadIdx.y;  // N 方向 tile 下标

    if (warp_m * WMMA_M >= m || warp_n * WMMA_N >= n) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // 沿 K 方向串行归约。用 size_t 做偏移：int 在 M*K 超过 2^31 时会静默溢出成负地址。
    for (int k0 = 0; k0 < k; k0 += WMMA_K) {
        const half* a_tile = A + (size_t)warp_m * WMMA_M * k + k0;
        const half* b_tile = B + (size_t)k0 * n + warp_n * WMMA_N;
        wmma::load_matrix_sync(a_frag, a_tile, k);
        wmma::load_matrix_sync(b_frag, b_tile, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_tile = C + (size_t)warp_m * WMMA_M * n + warp_n * WMMA_N;
    wmma::store_matrix_sync(c_tile, c_frag, n, wmma::mem_row_major);
}

int main() {
    const size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;

    half* h_A = (half*)malloc(a_elems * sizeof(half));
    half* h_B = (half*)malloc(b_elems * sizeof(half));
    float* h_C = (float*)malloc(c_elems * sizeof(float));
    float* h_ref = (float*)malloc(c_elems * sizeof(float));

    fill_inputs(h_A, a_elems, h_B, b_elems);

    printf("v0 wmma naive  (M=%d, N=%d, K=%d)\n", M, N, K);

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
    // 填 sentinel 而不是 0：漏算的 tile 会以 3.4e38 的形式直接暴露，而不是伪装成合理的小数
    CUDA_CHECK(cudaMemset(d_C, 0x7F, c_elems * sizeof(float)));

    dim3 block(32, 8);  // block 内 8 个 warp，全部排在 N 方向
    dim3 grid(CEIL(M, WMMA_M), CEIL(CEIL(N, WMMA_N), block.y));
    matmul_v0<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
