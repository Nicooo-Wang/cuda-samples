// cute_02 Tensor 基础教程的通用辅助函数
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

// 在 device 上打印（用于 kernel 内部）
__device__ inline void device_print_separator(const char* title) {
    printf("\n========== %s ==========\n", title);
}
