// transpose 六个版本共用的样板：错误检查、随机填充、CPU 参考、bit 级校验。
// 抽出来是为了让每个 .cu 只剩下"这一版 kernel 到底做了什么"这一件事。
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

#define CEIL(a, b) (((a) + (b) - 1) / (b))

constexpr int M = 4096;  // 输入是 M 行 N 列，转置成 N 行 M 列
constexpr int N = 4096;

// 用固定种子填 [-1, 1) 随机数，保证六个版本在同一组输入上跑，结果可横向对比
inline void fill_random(float* in, size_t count) {
    srand(0);
    for (size_t i = 0; i < count; ++i) in[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// CPU 参考：逐元素搬运，output[col][row] = input[row][col]
inline void cpu_transpose(const float* in, float* out, int M_, int N_) {
    for (int row = 0; row < M_; ++row)
        for (int col = 0; col < N_; ++col)
            out[(size_t)col * M_ + row] = in[(size_t)row * N_ + col];
}

// 转置是纯搬运，要求逐元素 bit 级相等。打印结果并返回是否通过。
inline bool verify_transpose(const float* gpu, const float* ref, int M_, int N_) {
    const size_t count = (size_t)M_ * N_;
    size_t errors = 0, first_bad = 0;
    for (size_t i = 0; i < count; ++i) {
        if (gpu[i] != ref[i]) {
            if (errors == 0) first_bad = i;
            ++errors;
        }
    }
    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    if (errors != 0)
        printf("  %zu mismatches, first at %zu: gpu = %f, cpu = %f\n", errors, first_bad,
               gpu[first_bad], ref[first_bad]);
    return errors == 0;
}
