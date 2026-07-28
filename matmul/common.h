// matmul 三个版本共用的 CPU 参考实现和精度校验。
// 抽出来是为了让每个 .cu 只剩下"这一版 kernel 到底做了什么"这一件事。
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// 用固定种子填 A / B，保证三个版本跑在同一组输入上，结果可以横向对比
inline void fill_inputs(half* A, size_t a_elems, half* B, size_t b_elems) {
    srand(0);
    for (size_t i = 0; i < a_elems; ++i) A[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    for (size_t i = 0; i < b_elems; ++i) B[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
}

// CPU 参考：C = A * B，全部 row-major。
// 循环顺序取 i-k-j 而不是教科书的 i-j-k：最内层沿 j 走时 B 和 C 都是按行连续访问，
// 对 cache 友好。累加用 float，和 GPU 的 fp32 accumulator 对齐。
inline void cpu_matmul(const half* A, const half* B, float* C, int M, int N, int K) {
    for (size_t i = 0; i < (size_t)M * N; ++i) C[i] = 0.0f;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const float a = __half2float(A[(size_t)i * K + k]);
            const half* b_row = B + (size_t)k * N;
            float* c_row = C + (size_t)i * N;
            for (int j = 0; j < N; ++j) c_row[j] += a * __half2float(b_row[j]);
        }
    }
}

// 精度校验：打印结果并返回是否通过。
// 输入零均值，C 里约 0.02% 的元素会正负相消到接近 0，对这些算纯相对误差会炸
// （分母趋近 0），所以用 numpy allclose 式判据：|diff| <= atol + rtol * |ref|
inline bool check_result(const float* gpu, const float* ref, int M, int N) {
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
        printf("  %zu / %zu off, first at %zu (row %zu, col %zu): gpu = %f, cpu = %f\n", bad, n,
               first_bad, first_bad / N, first_bad % N, gpu[first_bad], ref[first_bad]);
    }
    return bad == 0;
}
