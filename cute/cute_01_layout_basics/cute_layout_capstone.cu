// capstone: Layout 综合应用 - GPU 上的矩阵转置
//
// 综合目标：
//   1. 在 GPU 上使用 CuTe Layout
//   2. 实现一个高效的矩阵转置 kernel
//   3. 理解 shared memory bank conflict 和 Layout 的关系
//   4. 对比不同 Layout 策略的性能
//
// 实现策略：
//   - Naive: 直接全局内存转置（bank conflict + 非合并访问）
//   - Tiled: 使用 shared memory，用 Layout 描述 tile 的加载和存储
//   - Optimized: 通过 padding 避免 bank conflict

#include <cute/tensor.hpp>
#include "common.h"
#include <cstdlib>
#include <cuda_runtime.h>

using namespace cute;

// ========== Kernel 1: Naive 转置（全局内存直接转置） ==========
template <typename Element>
__global__ void transpose_naive(const Element* __restrict__ input,
                                Element* __restrict__ output,
                                int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        // 读: row-major (i, j) -> i * N + j
        // 写: col-major (i, j) -> 转置后的 (j, i) -> j * M + i
        output[j * M + i] = input[i * N + j];
    }
}

// ========== Kernel 2: Tiled 转置（使用 shared memory + CuTe Layout） ==========
template <int TILE_M, int TILE_N, typename Element>
__global__ void transpose_tiled_cute(const Element* __restrict__ input,
                                     Element* __restrict__ output,
                                     int M, int N) {
    // 在 shared memory 中创建一个 tile
    __shared__ Element smem[TILE_M * TILE_N];

    // 用 CuTe Layout 描述 shared memory 的布局
    auto smem_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                   make_stride(Int<TILE_N>{}, Int<1>{}));  // row-major

    // 创建 CuTe Tensor (将 smem 指针和 Layout 绑定)
    auto tile = make_tensor(make_smem_ptr(smem), smem_layout);

    // 计算当前 block 负责的 tile 位置
    int tile_i = blockIdx.y * TILE_M;
    int tile_j = blockIdx.x * TILE_N;

    int local_i = threadIdx.y;
    int local_j = threadIdx.x;

    // ---- 阶段 1: 从 global memory 加载到 shared memory ----
    int global_i = tile_i + local_i;
    int global_j = tile_j + local_j;

    if (global_i < M && global_j < N) {
        // 使用 CuTe Tensor 的索引
        tile(local_i, local_j) = input[global_i * N + global_j];
    }

    __syncthreads();

    // ---- 阶段 2: 从 shared memory 转置写入到 global memory ----
    // 转置：读 (i, j) 写到 (j, i)
    int out_i = tile_j + local_i;  // 注意交换
    int out_j = tile_i + local_j;

    if (out_i < N && out_j < M) {
        // 从 shared memory 读取时也转置索引
        output[out_i * M + out_j] = tile(local_j, local_i);  // 注意交换
    }
}

// ========== Kernel 3: 优化版（添加 padding 避免 bank conflict） ==========
template <int TILE_M, int TILE_N, typename Element>
__global__ void transpose_tiled_padded(const Element* __restrict__ input,
                                       Element* __restrict__ output,
                                       int M, int N) {
    // 添加 padding: 每行多 1 个元素，避免 bank conflict
    constexpr int PADDING = 1;
    __shared__ Element smem[TILE_M * (TILE_N + PADDING)];

    // Layout 的 stride 包含 padding
    auto smem_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                   make_stride(Int<TILE_N + PADDING>{}, Int<1>{}));

    auto tile = make_tensor(make_smem_ptr(smem), smem_layout);

    int tile_i = blockIdx.y * TILE_M;
    int tile_j = blockIdx.x * TILE_N;
    int local_i = threadIdx.y;
    int local_j = threadIdx.x;

    // 加载
    int global_i = tile_i + local_i;
    int global_j = tile_j + local_j;
    if (global_i < M && global_j < N) {
        tile(local_i, local_j) = input[global_i * N + global_j];
    }

    __syncthreads();

    // 转置写回
    int out_i = tile_j + local_i;
    int out_j = tile_i + local_j;
    if (out_i < N && out_j < M) {
        output[out_i * M + out_j] = tile(local_j, local_i);
    }
}

// ========== 验证函数 ==========
template <typename T>
bool verify_transpose(const T* output, const T* input, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T expected = input[i * N + j];
            T actual = output[j * M + i];
            if (expected != actual) {
                printf("  Mismatch at (%d, %d): expected %d, got %d\n",
                       i, j, (int)expected, (int)actual);
                return false;
            }
        }
    }
    return true;
}

// ========== 性能测试 ==========
template <typename Kernel>
float benchmark_kernel(Kernel kernel, dim3 grid, dim3 block,
                      const int* d_input, int* d_output,
                      int M, int N, int warmup, int iters) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        kernel<<<grid, block>>>(d_input, d_output, M, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        kernel<<<grid, block>>>(d_input, d_output, M, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

int main() {
    print_separator("CuTe Layout Capstone: GPU 矩阵转置");

    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;

    printf("矩阵大小: %d x %d\n", M, N);
    printf("Tile 大小: %d x %d\n", TILE_M, TILE_N);

    // ========== 1. 准备数据 ==========
    print_separator("1. 准备数据");

    size_t elems = (size_t)M * N;
    size_t bytes = elems * sizeof(int);

    int* h_input = (int*)malloc(bytes);
    int* h_output = (int*)malloc(bytes);

    // 初始化
    for (size_t i = 0; i < elems; ++i) {
        h_input[i] = i % 1000;  // 小数字便于调试
    }

    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    printf("分配内存: %.2f MB\n", bytes / 1024.0 / 1024.0);

    // ========== 2. Naive 转置 ==========
    print_separator("2. Naive 转置");

    dim3 block_naive(16, 16);
    dim3 grid_naive((N + block_naive.x - 1) / block_naive.x,
                    (M + block_naive.y - 1) / block_naive.y);

    transpose_naive<<<grid_naive, block_naive>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool pass = verify_transpose(h_output, h_input, M, N);
    printf("正确性验证: %s\n", pass ? "PASS" : "FAIL");

    float time_naive = benchmark_kernel(transpose_naive<int>, grid_naive, block_naive,
                                        d_input, d_output, M, N, 5, 100);
    float bandwidth_naive = 2.0f * bytes / (time_naive * 1e-3) / 1e9;  // GB/s
    printf("性能: %.3f ms, %.2f GB/s\n", time_naive, bandwidth_naive);

    // ========== 3. Tiled 转置（CuTe） ==========
    print_separator("3. Tiled 转置 (CuTe)");

    dim3 block_tiled(TILE_N, TILE_M);
    dim3 grid_tiled((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    transpose_tiled_cute<TILE_M, TILE_N><<<grid_tiled, block_tiled>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    pass = verify_transpose(h_output, h_input, M, N);
    printf("正确性验证: %s\n", pass ? "PASS" : "FAIL");

    float time_tiled = benchmark_kernel(transpose_tiled_cute<TILE_M, TILE_N, int>,
                                        grid_tiled, block_tiled,
                                        d_input, d_output, M, N, 5, 100);
    float bandwidth_tiled = 2.0f * bytes / (time_tiled * 1e-3) / 1e9;
    printf("性能: %.3f ms, %.2f GB/s (提升 %.1fx)\n",
           time_tiled, bandwidth_tiled, time_naive / time_tiled);

    // ========== 4. Padded 转置（避免 bank conflict） ==========
    print_separator("4. Padded 转置 (优化)");

    transpose_tiled_padded<TILE_M, TILE_N><<<grid_tiled, block_tiled>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    pass = verify_transpose(h_output, h_input, M, N);
    printf("正确性验证: %s\n", pass ? "PASS" : "FAIL");

    float time_padded = benchmark_kernel(transpose_tiled_padded<TILE_M, TILE_N, int>,
                                         grid_tiled, block_tiled,
                                         d_input, d_output, M, N, 5, 100);
    float bandwidth_padded = 2.0f * bytes / (time_padded * 1e-3) / 1e9;
    printf("性能: %.3f ms, %.2f GB/s (提升 %.1fx vs naive, %.1fx vs tiled)\n",
           time_padded, bandwidth_padded,
           time_naive / time_padded, time_tiled / time_padded);

    // ========== 5. 总结 ==========
    print_separator("总结");
    printf("性能对比:\n");
    printf("  Naive:  %.2f GB/s (基准)\n", bandwidth_naive);
    printf("  Tiled:  %.2f GB/s (%.1fx)\n", bandwidth_tiled, bandwidth_tiled / bandwidth_naive);
    printf("  Padded: %.2f GB/s (%.1fx)\n", bandwidth_padded, bandwidth_padded / bandwidth_naive);
    printf("\n");
    printf("关键收获:\n");
    printf("  • CuTe Layout 可以直接用于 GPU kernel\n");
    printf("  • make_tensor(ptr, layout) 创建 GPU Tensor\n");
    printf("  • Padding stride 避免 bank conflict\n");
    printf("  • Layout 的抽象让优化变得清晰和可组合\n");
    printf("\n恭喜! 你已经掌握了 CuTe Layout 的核心概念\n");
    printf("下一个教程: cute_02_tensor_basics 会深入 Tensor 的操作\n");

    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
