// matmul 各版本共用的样板：矩阵规模、错误检查、随机填充、参考结果、精度校验。
// 只放这些"每个版本都一样"的东西；tile 形状、grid/block、显存申请、kernel 启动和计时
// 都留在各自的 .cu 里，因为那些正是每一版要讲的内容。
#pragma once

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

#define CUBLAS_CHECK(call)                                                            \
    do {                                                                              \
        cublasStatus_t st_ = (call);                                                  \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                           \
            printf("cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__, (int)st_);  \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// C[M,N] = A[M,K] * B[K,N]，全部 row-major。各版本跑同一规模，TFLOP/s 可以直接横向比。
constexpr int M = 2048, N = 2048, K = 2048;

// wmma 的原子形状，各版本都用同一档：sm_70+ 对 fp16 最通用的 16x16x16
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

// 用固定种子填 A / B，保证各版本跑在同一组输入上
inline void fill_inputs(half* A, size_t a_elems, half* B, size_t b_elems) {
    srand(0);
    for (size_t i = 0; i < a_elems; ++i) A[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (size_t i = 0; i < b_elems; ++i) B[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
}

// 参考结果用 cuBLAS 在 GPU 上算：它和被测 kernel 一样走 tensor core、一样 fp32 累加，
// 实测逐元素完全相等，而且比 CPU 快几个数量级。d_scratch 借来放结果，算完就可以覆盖。
//
// cuBLAS 是 column-major，而这里全是 row-major。row-major 的 C = A*B 等价于
// col-major 的 C^T = B^T * A^T，所以把 B 当第一个参数、A 当第二个传进去，op 都用 N，
// 不需要真的转置数据。
inline void cublas_reference(const half* d_A, const half* d_B, float* d_scratch, float* h_ref) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_16F, N,
                              d_A, CUDA_R_16F, K, &beta, d_scratch, CUDA_R_32F, N,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_ref, d_scratch, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUBLAS_CHECK(cublasDestroy(handle));
}

// 完全独立于 GPU 的 CPU 参考。默认没人调用，怀疑 cuBLAS 时把 cublas_reference 换成它。
// 2048³ 要几秒。循环顺序取 i-k-j 而不是教科书的 i-j-k：最内层沿 j 走时 B 和 C 都按行连续
// 访问，对 cache 友好。累加用 float，和 GPU 的 fp32 accumulator 对齐。
inline void cpu_matmul(const half* A, const half* B, float* C) {
    for (size_t i = 0; i < (size_t)M * N; ++i) C[i] = 0.0f;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int kk = 0; kk < K; ++kk) {
            const float a = __half2float(A[(size_t)i * K + kk]);
            const half* b_row = B + (size_t)kk * N;
            float* c_row = C + (size_t)i * N;
            for (int j = 0; j < N; ++j) c_row[j] += a * __half2float(b_row[j]);
        }
    }
}

// 一次 GEMM 是 M*N*K 次乘加 = 2*M*N*K 次浮点运算。
//
// 各版本都只启动一次、不预热：这样打印的就是 ncu profile 到的那一次，两边数字对得上。
// 代价是把 kernel 模块加载等一次性开销也算进去了（实测版本间抖动 3% 以内，不影响比较）。
inline void report_perf(float ms) {
    printf("  %.3f ms, %.1f TFLOP/s\n", ms, 2.0 * M * N * K / (ms * 1e-3) / 1e12);
}

// 精度校验：打印结果并返回是否通过。
// 输入零均值，C 里约 0.02% 的元素会正负相消到接近 0，对这些算纯相对误差会炸（分母趋近 0），
// 所以用 numpy allclose 式判据：|diff| <= atol + rtol * |ref|
inline bool check_result(const float* gpu, const float* ref) {
    const size_t n = (size_t)M * N;

    double scale = 0.0;
    for (size_t i = 0; i < n; ++i) scale = fmax(scale, fabs((double)ref[i]));
    const double rtol = 1e-2, atol = 1e-4 * scale;

    double max_abs = 0.0;
    size_t bad = 0, first_bad = 0;
    for (size_t i = 0; i < n; ++i) {
        const double diff = fabs((double)gpu[i] - (double)ref[i]);
        max_abs = fmax(max_abs, diff);
        if (!(diff <= atol + rtol * fabs((double)ref[i]))) {  // NaN 也会走到这里
            if (bad == 0) first_bad = i;
            ++bad;
        }
    }

    printf("  max|C| = %.4f, max abs diff = %.3e (atol %.3e, rtol %.0e)  ->  %s\n", scale, max_abs,
           atol, rtol, bad == 0 ? "PASS" : "FAIL");
    if (bad != 0) {
        printf("  %zu / %zu off, first at %zu (row %zu, col %zu): gpu = %f, ref = %f\n", bad, n,
               first_bad, first_bad / N, first_bad % N, gpu[first_bad], ref[first_bad]);
    }
    return bad == 0;
}
