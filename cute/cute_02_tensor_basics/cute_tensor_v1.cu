// v1: GPU 上的 Tensor (Global 和 Shared Memory)
//
// 学习目标：
//   1. 在 GPU kernel 中使用 CuTe Tensor
//   2. 创建 global memory 和 shared memory 的 Tensor
//   3. 理解 make_smem_ptr 的作用
//   4. 在 device 上打印 Tensor 信息（调试技巧）
//
// 核心概念：
//   GPU 上的 Tensor 和 CPU 上完全一样的 API，但指针指向不同的内存空间

#include <cute/tensor.hpp>
#include "common.h"

using namespace cute;

// ========== Kernel 1: Global Memory Tensor ==========
__global__ void kernel_gmem_tensor() {
    // 每个线程只运行一次（用于演示）
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    device_print_separator("Kernel 1: Global Memory Tensor");

    // 注意：这里为了演示，直接在 kernel 里声明
    // 实际使用时，gmem 指针通常作为参数传入
    printf("在 GPU kernel 中，Tensor 的 API 和 CPU 上完全相同\n");
    printf("只是底层指针指向 device memory\n");
}

// ========== Kernel 2: Shared Memory Tensor ==========
template <int M, int N>
__global__ void kernel_smem_tensor() {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    device_print_separator("Kernel 2: Shared Memory Tensor");

    // 声明 shared memory
    __shared__ float smem[M * N];

    // 初始化 shared memory
    for (int i = 0; i < M * N; ++i) {
        smem[i] = i + 100.0f;
    }

    // 创建 Layout
    auto layout = make_layout(make_shape(Int<M>{}, Int<N>{}),
                              make_stride(Int<N>{}, Int<1>{}));

    // 关键：使用 make_smem_ptr 包装 shared memory 指针
    auto tensor = make_tensor(make_smem_ptr(smem), layout);

    printf("\nShared memory tensor shape: ");
    print(shape(tensor));
    printf("\n");

    printf("\n前 2 行数据:\n");
    for (int i = 0; i < 2; ++i) {
        printf("  row %d: ", i);
        for (int j = 0; j < N; ++j) {
            printf("%.0f ", tensor(i, j));
        }
        printf("\n");
    }

    // 修改数据
    tensor(0, 0) = 999.0f;
    printf("\n修改 tensor(0, 0) = 999.0\n");
    printf("验证: tensor(0, 0) = %.0f\n", tensor(0, 0));
}

// ========== Kernel 3: 多个线程协作访问 Shared Memory ==========
template <int M, int N>
__global__ void kernel_cooperative_smem(float* output) {
    constexpr int THREADS = 32;
    __shared__ float smem[M * N];

    int tid = threadIdx.x;

    // 创建 shared memory tensor
    auto layout = make_layout(make_shape(Int<M>{}, Int<N>{}),
                              make_stride(Int<N>{}, Int<1>{}));
    auto tensor = make_tensor(make_smem_ptr(smem), layout);

    // 每个线程负责初始化一部分数据
    for (int i = tid; i < M * N; i += THREADS) {
        int row = i / N;
        int col = i % N;
        tensor(row, col) = i * 2.0f;
    }

    __syncthreads();

    // Thread 0 打印结果
    if (tid == 0) {
        device_print_separator("Kernel 3: 多线程协作");
        printf("用 %d 个线程初始化 shared memory\n\n", THREADS);
        printf("前 2 行:\n");
        for (int i = 0; i < 2; ++i) {
            printf("  ");
            for (int j = 0; j < N; ++j) {
                printf("%.0f ", tensor(i, j));
            }
            printf("\n");
        }
    }

    __syncthreads();

    // 每个线程将一部分数据写回 global memory
    for (int i = tid; i < M * N; i += THREADS) {
        int row = i / N;
        int col = i % N;
        output[i] = tensor(row, col);
    }
}

// ========== Kernel 4: Global → Shared → Global ==========
template <int M, int N>
__global__ void kernel_gmem_to_smem(const float* input, float* output) {
    __shared__ float smem[M * N];

    int tid = threadIdx.x;
    constexpr int THREADS = 256;

    auto layout = make_layout(make_shape(Int<M>{}, Int<N>{}),
                              make_stride(Int<N>{}, Int<1>{}));

    // Global memory tensors
    auto gmem_in = make_tensor(input, layout);
    auto gmem_out = make_tensor(output, layout);

    // Shared memory tensor
    auto smem_tensor = make_tensor(make_smem_ptr(smem), layout);

    // Step 1: Load from global to shared
    for (int i = tid; i < M * N; i += THREADS) {
        int row = i / N;
        int col = i % N;
        smem_tensor(row, col) = gmem_in(row, col);
    }

    __syncthreads();

    // Step 2: 在 shared memory 中做计算（这里简单地乘以 2）
    for (int i = tid; i < M * N; i += THREADS) {
        int row = i / N;
        int col = i % N;
        smem_tensor(row, col) *= 2.0f;
    }

    __syncthreads();

    // Step 3: Store from shared to global
    for (int i = tid; i < M * N; i += THREADS) {
        int row = i / N;
        int col = i % N;
        gmem_out(row, col) = smem_tensor(row, col);
    }

    if (tid == 0) {
        device_print_separator("Kernel 4: Gmem -> Smem -> Gmem");
        printf("完成数据流水线: Global -> Shared (计算) -> Global\n");
    }
}

int main() {
    print_separator("CuTe Tensor on GPU");

    // ========== 1. 基础 Global Memory Tensor ==========
    print_separator("1. 启动 Kernel 1");
    kernel_gmem_tensor<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========== 2. Shared Memory Tensor ==========
    print_separator("2. 启动 Kernel 2");
    constexpr int M = 4, N = 8;
    kernel_smem_tensor<M, N><<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========== 3. 多线程协作 ==========
    print_separator("3. 启动 Kernel 3 (多线程)");
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, M * N * sizeof(float)));

    kernel_cooperative_smem<M, N><<<1, 32>>>(d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 验证结果
    float h_output[M * N];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n验证从 GPU 拷贝回的数据 (前 8 个):\n  ");
    for (int i = 0; i < 8; ++i) {
        printf("%.0f ", h_output[i]);
    }
    printf("\n");

    // ========== 4. 完整的数据流水线 ==========
    print_separator("4. 启动 Kernel 4 (数据流水线)");

    float* h_input = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = i;
    }

    float *d_input, *d_output2;
    CUDA_CHECK(cudaMalloc(&d_input, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output2, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, M * N * sizeof(float), cudaMemcpyHostToDevice));

    kernel_gmem_to_smem<M, N><<<1, 256>>>(d_input, d_output2);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_output2[M * N];
    CUDA_CHECK(cudaMemcpy(h_output2, d_output2, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n验证计算结果 (每个元素 × 2):\n");
    printf("输入前 8 个: ");
    for (int i = 0; i < 8; ++i) printf("%.0f ", h_input[i]);
    printf("\n输出前 8 个: ");
    for (int i = 0; i < 8; ++i) printf("%.0f ", h_output2[i]);
    printf("\n");

    bool pass = true;
    for (int i = 0; i < M * N; ++i) {
        if (h_output2[i] != h_input[i] * 2.0f) {
            pass = false;
            break;
        }
    }
    printf("正确性检查: %s\n", pass ? "PASS ✓" : "FAIL ✗");

    // ========== 总结 ==========
    print_separator("总结");
    printf("关键要点:\n");
    printf("  • GPU 上的 Tensor API 和 CPU 完全相同\n");
    printf("  • Shared memory 用 make_smem_ptr() 包装\n");
    printf("  • 多线程可以协作访问同一个 Tensor\n");
    printf("  • Tensor 抽象简化了 gmem ↔ smem 的数据搬运\n");
    printf("\n下一步: v2 会介绍 local_partition 和线程映射\n");

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output2));
    free(h_input);

    return 0;
}
