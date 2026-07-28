// v2: cuBLAS 参考实现，用来对照自己写的 kernel 是否正确。
//
// cuBLAS 是 column-major，而这里所有矩阵都是 row-major。
// row-major 的 C = A*B 等价于 col-major 的 C^T = B^T * A^T，
// 所以把 B 当第一个参数、A 当第二个参数传进去，op 都用 N 即可，不需要真的转置数据。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                           \
    do {                                                                             \
        cublasStatus_t st_ = (call);                                                 \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                          \
            printf("cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__, (int)st_); \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    } while (0)

constexpr int M = 2048, N = 2048, K = 2048;

int main() {
    const size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;

    half* h_A = (half*)malloc(a_elems * sizeof(half));
    half* h_B = (half*)malloc(b_elems * sizeof(half));
    float* h_C = (float*)malloc(c_elems * sizeof(float));
    float* h_ref = (float*)malloc(c_elems * sizeof(float));

    srand(0);
    for (size_t i = 0; i < a_elems; ++i) h_A[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (size_t i = 0; i < b_elems; ++i) h_B[i] = __float2half((float)rand() / RAND_MAX - 0.5f);

    printf("v2 cuBLAS GemmEx TC  (M=%d, N=%d, K=%d)\n", M, N, K);

    // ---- 1. 先算 CPU 参考 ----
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
    CUDA_CHECK(cudaMemset(d_C, 0x7F, c_elems * sizeof(float)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_16F, N,
                              d_A, CUDA_R_16F, K, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 3. 精度验证（判据同 v0）----
    double scale = 0.0;
    for (size_t i = 0; i < c_elems; ++i) scale = fmax(scale, fabs((double)h_ref[i]));
    const double rtol = 1e-2, atol = 1e-4 * scale;

    double max_abs = 0.0;
    size_t bad = 0, first_bad = 0;
    for (size_t i = 0; i < c_elems; ++i) {
        const double diff = fabs((double)h_C[i] - (double)h_ref[i]);
        max_abs = fmax(max_abs, diff);
        if (!(diff <= atol + rtol * fabs((double)h_ref[i]))) {
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

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return bad == 0 ? 0 : 1;
}
