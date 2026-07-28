// v0: 最朴素的 wmma 版本 —— 每个 warp 负责一个 16x16 的输出 tile，
// A / B 的 fragment 每次都直接从 global memory 载入，没有任何数据复用。
//
// 来源：Section011/tensor.cu，这里修掉了原版的两个问题：
//   1. warpN 的映射写成了 blockIdx.y * blockDim.y / warpSize + threadIdx.y，
//      按运算优先级是 (blockIdx.y * 32) / 32 + threadIdx.y = blockIdx.y + threadIdx.y，
//      blockIdx.y 的步长被整除掉了，只覆盖了一小块 C，其余 72% 从未被写入。
//   2. accumulator 用 half：K 大了会失精甚至溢出（fp16 最大有限值 65504）。
//      在这个 kernel 里换成 float 是免费的，因为瓶颈根本不在 tensor core。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

#define CEIL(a, b) (((a) + (b) - 1) / (b))

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

    srand(0);
    for (size_t i = 0; i < a_elems; ++i) h_A[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (size_t i = 0; i < b_elems; ++i) h_B[i] = __float2half((float)rand() / RAND_MAX - 0.5f);

    printf("v0 wmma naive  (M=%d, N=%d, K=%d)\n", M, N, K);

    // ---- 1. 先算 CPU 参考（ikj 顺序对 cache 友好，fp32 累加和 GPU 对齐）----
    printf("  [1/2] CPU reference ...\n");
    fflush(stdout);
    for (size_t i = 0; i < c_elems; ++i) h_ref[i] = 0.0f;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int kk = 0; kk < K; ++kk) {
            const float a = __half2float(h_A[(size_t)i * K + kk]);
            const half* b_row = h_B + (size_t)kk * N;
            float* c_row = h_ref + (size_t)i * N;
            for (int j = 0; j < N; ++j) c_row[j] += a * __half2float(b_row[j]);
        }
    }

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
    // 输入零均值，C 里约 0.02% 的元素会正负相消到接近 0，对这些算纯相对误差会炸
    // （分母趋近 0），所以用 numpy allclose 式判据：|diff| <= atol + rtol * |ref|
    double scale = 0.0;
    for (size_t i = 0; i < c_elems; ++i) scale = fmax(scale, fabs((double)h_ref[i]));
    const double rtol = 1e-2, atol = 1e-4 * scale;

    double max_abs = 0.0;
    size_t bad = 0, first_bad = 0;
    for (size_t i = 0; i < c_elems; ++i) {
        const double diff = fabs((double)h_C[i] - (double)h_ref[i]);
        max_abs = fmax(max_abs, diff);
        if (!(diff <= atol + rtol * fabs((double)h_ref[i]))) {  // NaN 也会走到这里
            if (bad == 0) first_bad = i;
            ++bad;
        }
    }

    printf("  max|C| = %.4f, max abs diff = %.3e (atol %.3e, rtol %.0e)  ->  %s\n", scale, max_abs,
           atol, rtol, bad == 0 ? "PASS" : "FAIL");
    if (bad != 0) {
        printf("  %zu / %zu off, first at %zu (row %zu, col %zu): gpu = %f, cpu = %f\n", bad,
               c_elems, first_bad, first_bad / N, first_bad % N, h_C[first_bad], h_ref[first_bad]);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return bad == 0 ? 0 : 1;
}
