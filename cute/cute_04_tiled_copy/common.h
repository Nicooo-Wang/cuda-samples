// cute_04 通用工具
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

inline void print_separator(const char* title) {
    printf("\n");
    printf("===============================================================\n");
    printf("  %s\n", title);
    printf("===============================================================\n");
}

// 32 个 bank，每 bank 4 字节
inline __host__ __device__ int bank_of(int byte_offset) { return (byte_offset / 4) % 32; }

// 统计一次访存里"最热的 bank 被请求了几次" —— 即 N-way conflict 的 N
template <class F>
inline int max_bank_requests(int n_lanes, F&& byte_offset_of_lane) {
    int hist[32] = {0};
    for (int lane = 0; lane < n_lanes; ++lane) ++hist[bank_of(byte_offset_of_lane(lane))];
    int worst = 0;
    for (int b = 0; b < 32; ++b)
        if (hist[b] > worst) worst = hist[b];
    return worst;
}

// 计时：返回单次 kernel 的平均毫秒数
template <class F>
inline float time_kernel(F&& fn, int warmup = 10, int iters = 200) {
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

// 转置的有效带宽：读一遍 + 写一遍
inline double transpose_bandwidth_gbs(size_t elems, size_t elem_bytes, float ms) {
    return double(2 * elems * elem_bytes) / 1e9 / (double(ms) * 1e-3);
}
