// cute_03 Copy Atom 教程的通用辅助函数
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

// 打印分隔线
inline void print_separator(const char* title = nullptr) {
    if (title) {
        printf("\n========== %s ==========\n", title);
    } else {
        printf("========================================\n");
    }
}

// 打印一个 Copy_Atom 的内部三件套：ThrID / ValLayoutSrc / ValLayoutDst
// 这三个决定了"一次原子操作要几个线程、每个线程搬几个值"
template <typename Atom>
void print_atom(const char* name, Atom const& = {}) {
    printf("%s\n", name);
    printf("  ThrID        = ");
    cute::print(typename Atom::ThrID{});
    printf("   (参与线程数 = %d)\n", int(cute::size(typename Atom::ThrID{})));
    printf("  ValLayoutSrc = ");
    cute::print(typename Atom::ValLayoutSrc{});
    printf("\n");
    printf("  ValLayoutDst = ");
    cute::print(typename Atom::ValLayoutDst{});
    printf("\n");
    printf("  NumValSrc = %d   NumValDst = %d\n", Atom::NumValSrc, Atom::NumValDst);
}

// 计时：返回 kernel 的平均耗时（毫秒）
template <typename Fn>
float time_kernel(Fn&& fn, int warmup = 5, int iters = 20) {
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

// memcpy 类 kernel 的有效带宽：读 + 写 各算一遍
inline double copy_bandwidth_gbs(size_t bytes, float ms) {
    return 2.0 * double(bytes) / (double(ms) * 1e-3) / 1e9;
}
