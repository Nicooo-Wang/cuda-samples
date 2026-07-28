// matmul 各版本共用的测试脚手架：分配内存、cuBLAS 参考、精度验证、计时
#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

// 固定 shape：C[M,N] = A[M,K] * B[K,N]，全部 row-major
constexpr int M = 2048;
constexpr int N = 2048;
constexpr int K = 2048;

constexpr int WMMA_M = 16;  // wmma 基本 tile：16x16x16
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

#define CEIL(a, b) (((a) + (b) - 1) / (b))

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st_ = (call);                                           \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                    \
            printf("cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__, (int)st_); \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// 参考实现用 cuBLAS：既是"标准答案"，也是性能上界。
// cuBLAS 是 column-major，row-major 的 C = A*B 等价于 col-major 的 C^T = B^T * A^T，
// 所以把 B 当第一个参数、A 当第二个参数传进去，op 都用 N 即可。
inline void cublas_reference(cublasHandle_t handle, const half* dA, const half* dB, float* dC,
                             int m, int n, int k) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dB, CUDA_R_16F, n,
                              dA, CUDA_R_16F, k, &beta, dC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// 所有版本共用的 launcher。Launcher 签名 (const half* A, const half* B, float* C, int m,int n,int k)。
// fp16 输入 + fp32 累加，和 cuBLAS 的差别只来自 K 方向累加顺序不同，用相对误差判定。
template <typename Launcher>
int run_matmul(const char* name, Launcher launch) {
    const size_t a_elems = (size_t)M * K;
    const size_t b_elems = (size_t)K * N;
    const size_t c_elems = (size_t)M * N;

    half* h_A = (half*)malloc(a_elems * sizeof(half));
    half* h_B = (half*)malloc(b_elems * sizeof(half));
    float* h_C = (float*)malloc(c_elems * sizeof(float));
    float* h_ref = (float*)malloc(c_elems * sizeof(float));

    srand(0);
    for (size_t i = 0; i < a_elems; ++i) h_A[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (size_t i = 0; i < b_elems; ++i) h_B[i] = __float2half((float)rand() / RAND_MAX - 0.5f);

    half *d_A, *d_B;
    float *d_C, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, a_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, b_elems * sizeof(half), cudaMemcpyHostToDevice));
    // 填 sentinel 而不是 0：如果 kernel 漏算了某些 tile，验证阶段能直接暴露出来
    CUDA_CHECK(cudaMemset(d_C, 0x7F, c_elems * sizeof(float)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cublas_reference(handle, d_A, d_B, d_ref, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_ref, d_ref, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    launch(d_A, d_B, d_C, M, N, K);  // warmup（也避免把 JIT / 首次 launch 开销算进计时）
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    const int iters = 20;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    // 精度：相对误差 + 统计未写入的元素个数
    const double tol = 1e-2;
    double max_rel = 0.0;
    size_t bad = 0, first_bad = 0;
    for (size_t i = 0; i < c_elems; ++i) {
        double ref = h_ref[i], gpu = h_C[i];
        double rel = fabs(gpu - ref) / (fabs(ref) + 1e-6);
        if (rel > max_rel) max_rel = rel;
        if (!(rel < tol)) {  // NaN 也会走到这里
            if (bad == 0) first_bad = i;
            ++bad;
        }
    }

    const double tflop = 2.0 * M * N * K / 1e12;
    printf("%-28s  %dx%dx%d  time = %8.3f ms  %8.1f TFLOP/s  max_rel = %.2e  %s\n", name, M, N, K,
           ms, tflop / (ms / 1e3), max_rel, bad == 0 ? "PASS" : "FAIL");
    if (bad != 0) {
        printf("  %zu / %zu elements off (tol %.0e), first at %zu (row %zu, col %zu): "
               "gpu = %f, ref = %f\n",
               bad, c_elems, tol, first_bad, first_bad / N, first_bad % N, h_C[first_bad],
               h_ref[first_bad]);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);
    return bad == 0 ? 0 : 1;
}
