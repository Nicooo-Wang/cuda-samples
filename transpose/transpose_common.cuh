// transpose 各版本共用的测试脚手架：分配内存、CPU 参考、精度验证、计时
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// 固定 shape：M 行 N 列的输入，转置成 N 行 M 列的输出
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int TILE_DIM = 32;  // 共享内存版本的 tile 边长

#define CEIL(a, b) (((a) + (b) - 1) / (b))

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

// CPU 参考：output[col * M + row] = input[row * N + col]
inline void cpu_transpose(const float* input, float* output, int m, int n) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            output[col * m + row] = input[row * n + col];
        }
    }
}

// 所有版本共用的 launcher。Launcher 是个可调用对象，签名 (const float*, float*, int, int)。
// 转置是纯搬运，逐元素 bit 级相等即可，不需要容差。
template <typename Launcher>
int run_transpose(const char* name, Launcher launch) {
    const size_t in_bytes = (size_t)M * N * sizeof(float);
    const size_t out_bytes = (size_t)N * M * sizeof(float);

    float* h_input = (float*)malloc(in_bytes);
    float* h_output = (float*)malloc(out_bytes);
    float* h_ref = (float*)malloc(out_bytes);

    srand(0);
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        h_input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, out_bytes));

    launch(d_input, d_output, M, N);  // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    const int iters = 20;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch(d_input, d_output, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost));
    cpu_transpose(h_input, h_ref, M, N);

    size_t errors = 0;
    size_t first_bad = 0;
    for (size_t i = 0; i < (size_t)N * M; ++i) {
        if (h_output[i] != h_ref[i]) {
            if (errors == 0) first_bad = i;
            ++errors;
        }
    }

    // 有效带宽：读 + 写各一遍
    double gb = 2.0 * in_bytes / 1e9;
    printf("%-24s  %dx%d  time = %8.3f ms  bandwidth = %7.1f GB/s  %s\n", name, M, N, ms,
           gb / (ms / 1e3), errors == 0 ? "PASS" : "FAIL");
    if (errors != 0) {
        printf("  %zu mismatches, first at %zu: gpu = %f, cpu = %f\n", errors, first_bad,
               h_output[first_bad], h_ref[first_bad]);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);
    free(h_ref);
    return errors == 0 ? 0 : 1;
}
