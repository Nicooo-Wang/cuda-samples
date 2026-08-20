// cute_05 通用工具
//
// 和 cute_04 的 common.h 相比, 这里去掉了 bank 统计 (那是搬运章的事),
// 换成了矩阵乘需要的东西: 填矩阵、CPU 参考实现、误差比对、TFLOP/s。
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

inline void print_separator(const char* title) {
    printf("\n");
    printf("===============================================================\n");
    printf("  %s\n", title);
    printf("===============================================================\n");
}

// ---------------------------------------------------------------------------
// 填矩阵
//
// 用 ±1 而不是 i%1024: fp16 的整数只精确到 2048, 累加 K 次之后
// 用大数会累积舍入误差, 让"结果对不对"变成"误差多大"的判断题。
// ±1 时 C 的每个元素都是小整数, fp16/fp32 都能精确表示 -> 可以要求逐点相等。
// (这一条是 cute_04 踩过的坑, 见 README §1.4)
// ---------------------------------------------------------------------------
inline void fill_pm1(cute::half_t* p, size_t n, unsigned seed = 1234) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) p[i] = cute::half_t((rand() % 2) ? 1.f : -1.f);
}

// ---------------------------------------------------------------------------
// CPU 参考 GEMM: C = A * B
//   A 是 M x K, row-major (stride = (K,1))
//   B 是 N x K, row-major (stride = (K,1))  <- 注意 B 存成 N x K, 即 B^T
//   C 是 M x N, row-major (stride = (N,1))
//
// 全程用 float 累加, 避免 CPU 侧自己先失去精度。
// 这个"A(M,K) 行主 + B(N,K) 行主"的摆法就是 BLAS 里的 TN 布局, 也是
// Tensor Core 最自然的摆法: 两个矩阵的 K 方向都连续。README §2.1 有图。
// ---------------------------------------------------------------------------
inline void gemm_cpu(const cute::half_t* A, const cute::half_t* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += float(A[size_t(m) * K + k]) * float(B[size_t(n) * K + k]);
            C[size_t(m) * N + n] = acc;
        }
}

// ---------------------------------------------------------------------------
// 比对: 返回最大绝对误差, 并统计有多少个元素超过 tol
// ±1 输入下正确的结果应该是 bad == 0 且 maxerr == 0
// ---------------------------------------------------------------------------
struct CheckResult {
    int bad;
    float maxerr;
    bool ok() const { return bad == 0; }
};

inline CheckResult check_close(const float* got, const float* ref, size_t n, float tol = 1e-3f) {
    CheckResult r{0, 0.f};
    for (size_t i = 0; i < n; ++i) {
        float e = fabsf(got[i] - ref[i]);
        if (e > r.maxerr) r.maxerr = e;
        if (e > tol) ++r.bad;
    }
    return r;
}

// half 版本的比对 (输出是 half 时用)
inline CheckResult check_close(const cute::half_t* got, const float* ref, size_t n,
                               float tol = 1e-2f) {
    CheckResult r{0, 0.f};
    for (size_t i = 0; i < n; ++i) {
        float e = fabsf(float(got[i]) - ref[i]);
        if (e > r.maxerr) r.maxerr = e;
        if (e > tol) ++r.bad;
    }
    return r;
}

// ---------------------------------------------------------------------------
// 计时: 返回单次 kernel 的平均毫秒数
// ---------------------------------------------------------------------------
template <class F>
inline float time_kernel(F&& fn, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms / iters;
}

// GEMM 的算力: 2*M*N*K 次浮点运算
inline double gemm_tflops(int M, int N, int K, float ms) {
    return 2.0 * M * N * K / 1e12 / (double(ms) * 1e-3);
}
