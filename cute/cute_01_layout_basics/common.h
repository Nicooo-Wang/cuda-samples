// cute_01 Layout 基础教程的通用辅助函数
#pragma once

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

// CuTe 的 print 系列函数会输出到 stdout，这里提供一个简单的分隔线工具
inline void print_separator(const char* title = nullptr) {
    if (title) {
        printf("\n========== %s ==========\n", title);
    } else {
        printf("========================================\n");
    }
}
