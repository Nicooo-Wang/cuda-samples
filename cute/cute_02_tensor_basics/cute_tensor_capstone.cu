// capstone: 矩阵-向量乘法 (GEMV) 使用 CuTe Tensor
//
// 综合应用：
//   1. 使用 Tensor 描述矩阵和向量
//   2. 使用 local_partition 分配线程工作
//   3. 使用 shared memory 优化访问
//   4. 对比不同实现的性能
//
// 问题：y = A * x
//   A: M × N 矩阵 (row-major)
//   x: N × 1 向量
//   y: M × 1 向量
//
// 每个元素: y[i] = sum(A[i][j] * x[j]) for j in [0, N)

#include <cute/tensor.hpp>
#include "common.h"

using namespace cute;

// ========== Baseline: 传统 CUDA 实现 ==========
__global__ void gemv_baseline(const float* A, const float* x, float* y, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += A[i * N + j] * x[j];
    }
    y[i] = sum;
}

// ========== Version 1: CuTe Tensor (naive) ==========
template <int M, int N>
__global__ void gemv_cute_naive(const float* A, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    // 创建 Tensors
    auto layout_A = make_layout(make_shape(Int<M>{}, Int<N>{}),
                                make_stride(Int<N>{}, Int<1>{}));
    auto layout_x = make_layout(Int<N>{});
    auto layout_y = make_layout(Int<M>{});

    auto tA = make_tensor(A, layout_A);
    auto tx = make_tensor(x, layout_x);
    auto ty = make_tensor(y, layout_y);

    // 计算 y[i]
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += tA(i, j) * tx(j);
    }
    ty(i) = sum;
}

// ========== Version 2: CuTe with Shared Memory ==========
template <int M, int N, int THREADS>
__global__ void gemv_cute_smem(const float* A, const float* x, float* y) {
    __shared__ float smem_x[N];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 创建 global memory tensors
    auto layout_A = make_layout(make_shape(Int<M>{}, Int<N>{}),
                                make_stride(Int<N>{}, Int<1>{}));
    auto tA = make_tensor(A, layout_A);
    auto tx_gmem = make_tensor(x, make_layout(Int<N>{}));

    // 创建 shared memory tensor
    auto tx_smem = make_tensor(make_smem_ptr(smem_x), make_layout(Int<N>{}));

    // 使用 partition 协作加载 x 到 shared memory
    auto thr_layout = make_layout(Int<THREADS>{});
    auto my_x = local_partition(tx_smem, thr_layout, tid);
    auto my_x_gmem = local_partition(tx_gmem, thr_layout, tid);

    for (int j = 0; j < size(my_x); ++j) {
        my_x(j) = my_x_gmem(j);
    }

    __syncthreads();

    // 计算 y[i] (从 shared memory 读取 x)
    if (i < M) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += tA(i, j) * tx_smem(j);
        }
        y[i] = sum;
    }
}

// ========== Version 3: CuTe with Partition for A ==========
template <int M, int N, int BLOCK_SIZE>
__global__ void gemv_cute_partition(const float* A, const float* x, float* y) {
    __shared__ float smem_x[N];

    int tid = threadIdx.x;

    // Global tensors
    auto layout_A = make_layout(make_shape(Int<M>{}, Int<N>{}),
                                make_stride(Int<N>{}, Int<1>{}));
    auto tA = make_tensor(A, layout_A);
    auto tx_gmem = make_tensor(x, make_layout(Int<N>{}));
    auto ty = make_tensor(y, make_layout(Int<M>{}));

    // Shared tensor for x
    auto tx_smem = make_tensor(make_smem_ptr(smem_x), make_layout(Int<N>{}));

    // Load x to smem using partition
    auto thr_layout = make_layout(Int<BLOCK_SIZE>{});
    auto my_x = local_partition(tx_smem, thr_layout, tid);
    auto my_x_gmem = local_partition(tx_gmem, thr_layout, tid);

    for (int j = 0; j < size(my_x); ++j) {
        my_x(j) = my_x_gmem(j);
    }

    __syncthreads();

    // Partition A by rows
    auto my_rows = local_partition(tA, thr_layout, tid);
    auto my_y = local_partition(ty, thr_layout, tid);

    // 每个线程处理若干行
    for (int row_idx = 0; row_idx < size(my_y); ++row_idx) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += my_rows(row_idx, j) * tx_smem(j);
        }
        my_y(row_idx) = sum;
    }
}

// ========== 性能测试 ==========
template <typename Kernel>
float benchmark(Kernel kernel, dim3 grid, dim3 block, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        kernel<<<grid, block>>>();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        kernel<<<grid, block>>>();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

// 验证函数
void verify(const float* y, const float* y_ref, int M) {
    bool pass = true;
    for (int i = 0; i < M; ++i) {
        if (fabs(y[i] - y_ref[i]) > 1e-3) {
            printf("Mismatch at %d: %.3f vs %.3f\n", i, y[i], y_ref[i]);
            pass = false;
            break;
        }
    }
    printf("验证: %s\n", pass ? "PASS ✓" : "FAIL ✗");
}

int main() {
    print_separator("CuTe Tensor Capstone: GEMV");

    constexpr int M = 4096;
    constexpr int N = 512;
    constexpr int BLOCK_SIZE = 256;

    printf("问题规模: y = A * x\n");
    printf("  A: %d × %d\n", M, N);
    printf("  x: %d × 1\n", N);
    printf("  y: %d × 1\n", M);

    // 分配和初始化
    float *h_A = (float*)malloc(M * N * sizeof(float));
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(M * sizeof(float));
    float *h_y_ref = (float*)malloc(M * sizeof(float));

    for (int i = 0; i < M * N; ++i) h_A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < N; ++i) h_x[i] = (rand() % 100) / 100.0f;

    // CPU 参考实现
    print_separator("CPU 参考实现");
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += h_A[i * N + j] * h_x[j];
        }
        h_y_ref[i] = sum;
    }
    printf("CPU 计算完成\n");

    // GPU 内存分配
    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ========== Version 0: Baseline ==========
    print_separator("Version 0: Baseline");
    gemv_baseline<<<grid, block>>>(d_A, d_x, d_y, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
    verify(h_y, h_y_ref, M);

    auto time0 = benchmark([=]() { gemv_baseline<<<grid, block>>>(d_A, d_x, d_y, M, N); },
                           grid, block, 5, 100);
    printf("时间: %.3f ms\n", time0);

    // ========== Version 1: CuTe Naive ==========
    print_separator("Version 1: CuTe Naive");
    gemv_cute_naive<M, N><<<grid, block>>>(d_A, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
    verify(h_y, h_y_ref, M);

    auto time1 = benchmark([=]() { gemv_cute_naive<M, N><<<grid, block>>>(d_A, d_x, d_y); },
                           grid, block, 5, 100);
    printf("时间: %.3f ms (%.1fx vs baseline)\n", time1, time0/time1);

    // ========== Version 2: CuTe + Shared Memory ==========
    print_separator("Version 2: CuTe + Shared Memory");
    gemv_cute_smem<M, N, BLOCK_SIZE><<<grid, block>>>(d_A, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
    verify(h_y, h_y_ref, M);

    auto time2 = benchmark([=]() { gemv_cute_smem<M, N, BLOCK_SIZE><<<grid, block>>>(d_A, d_x, d_y); },
                           grid, block, 5, 100);
    printf("时间: %.3f ms (%.1fx vs baseline)\n", time2, time0/time2);

    // ========== Version 3: CuTe + Partition ==========
    print_separator("Version 3: CuTe + Partition");
    gemv_cute_partition<M, N, BLOCK_SIZE><<<grid, block>>>(d_A, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
    verify(h_y, h_y_ref, M);

    auto time3 = benchmark([=]() { gemv_cute_partition<M, N, BLOCK_SIZE><<<grid, block>>>(d_A, d_x, d_y); },
                           grid, block, 5, 100);
    printf("时间: %.3f ms (%.1fx vs baseline)\n", time3, time0/time3);

    // ========== 总结 ==========
    print_separator("性能总结");
    printf("实现           时间(ms)   相对性能\n");
    printf("Baseline       %.3f     1.00x\n", time0);
    printf("CuTe Naive     %.3f     %.2fx\n", time1, time0/time1);
    printf("CuTe + Smem    %.3f     %.2fx\n", time2, time0/time2);
    printf("CuTe + Part    %.3f     %.2fx\n", time3, time0/time3);

    print_separator("关键收获");
    printf("✓ CuTe Tensor 提供了清晰的抽象，代码更易读\n");
    printf("✓ local_partition 简化了线程到数据的映射\n");
    printf("✓ 零开销抽象：性能和手写 CUDA 相当或更好\n");
    printf("✓ Shared memory 优化在 CuTe 中依然有效\n");
    printf("\n恭喜！你已经掌握了 CuTe Tensor 的核心用法\n");
    printf("下一个教程: cute_03_copy_atom 会介绍数据搬运的抽象\n");

    free(h_A); free(h_x); free(h_y); free(h_y_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return 0;
}
