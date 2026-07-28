// v2: cuBLAS 参考实现，用来对照自己写的 kernel 是否正确。
//
// cuBLAS 是 column-major，而这里所有矩阵都是 row-major。
// row-major 的 C = A*B 等价于 col-major 的 C^T = B^T * A^T，
// 所以把 B 当第一个参数、A 当第二个参数传进去，op 都用 N 即可，不需要真的转置数据。
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "common.h"

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

    fill_inputs(h_A, a_elems, h_B, b_elems);

    printf("v2 cuBLAS GemmEx TC  (M=%d, N=%d, K=%d)\n", M, N, K);

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

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_16F, N,
                              d_A, CUDA_R_16F, K, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 3. 精度验证 ----
    const bool pass = check_result(h_C, h_ref, M, N);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return pass ? 0 : 1;
}
