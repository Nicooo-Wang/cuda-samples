// v2: local_partition - 线程到数据的映射
//
// ========== 什么是 Partition？ ==========
//
// 在 GPU 编程中，我们经常需要回答这个问题：
//   "每个线程应该处理哪些数据？"
//
// CuTe 的 local_partition 函数就是用来回答这个问题的！
//
// local_partition(tensor, thread_layout, thread_id)
//   → 返回属于 thread_id 的数据子集（一个新的 Tensor view）
//
// ========== Thread Layout 的概念 ==========
//
// Thread Layout 描述了线程的组织方式：
//   - 有多少个线程？（Shape）
//   - 线程如何编号？（Stride）
//
// 例如：
//   auto thr_layout = make_layout(Int<32>{});  // 32 个线程，线性编号
//
// ========== Partition 的工作原理 ==========
//
// 给定：
//   - 数据 Tensor: shape = (M, N)
//   - 线程数量: 32
//
// local_partition 会：
//   1. 将数据平均分配给每个线程
//   2. 返回每个线程的"视图"（view）
//   3. 不同线程的 view 访问不同的数据（无重叠）
//
// 关键：partition 后的 Tensor 仍然指向原始数据，只是改变了索引方式！
//
// ========== 为什么需要 Partition？ ==========
//
// 传统方式：
//   int tid = threadIdx.x;
//   for (int i = tid; i < N; i += blockDim.x) {
//       data[i] = ...;  // 手动计算线程到数据的映射
//   }
//
// CuTe 方式：
//   auto my_data = local_partition(tensor, thr_layout, tid);
//   for (int i = 0; i < size(my_data); ++i) {
//       my_data(i) = ...;  // 自动映射，简洁清晰
//   }

#include <cute/tensor.hpp>
#include "common.h"

using namespace cute;

// ========== Kernel 1: 基础 Partition ==========
__global__ void kernel_basic_partition() {
    constexpr int N = 64;
    constexpr int THREADS = 8;
    __shared__ float smem[N];

    int tid = threadIdx.x;
    if (tid >= THREADS) return;

    // 创建一个 1D Tensor
    auto layout = make_layout(Int<N>{});
    auto tensor = make_tensor(make_smem_ptr(smem), layout);

    // 创建线程布局：8 个线程
    auto thr_layout = make_layout(Int<THREADS>{});

    // Partition: 将 tensor 分配给当前线程
    auto my_partition = local_partition(tensor, thr_layout, tid);

    if (tid == 0) {
        device_print_separator("Kernel 1: 基础 Partition");
        printf("原始 tensor: size = %d\n", int(size(tensor)));
        printf("线程数: %d\n", THREADS);
        printf("每个线程的 partition size = %d\n\n", int(size(my_partition)));
    }

    // 每个线程初始化自己的数据
    for (int i = 0; i < size(my_partition); ++i) {
        my_partition(i) = tid * 100.0f + i;
    }

    __syncthreads();

    // Thread 0 打印所有数据
    if (tid == 0) {
        printf("结果（每个线程负责 %d 个元素）:\n", int(size(my_partition)));
        for (int t = 0; t < THREADS; ++t) {
            printf("  Thread %d: ", t);
            for (int i = 0; i < 8; ++i) {
                printf("%.0f ", tensor(t * int(size(my_partition)) + i));
            }
            printf("\n");
        }
    }
}

// ========== Kernel 2: 2D Partition ==========
template <int M, int N, int THREADS>
__global__ void kernel_2d_partition() {
    __shared__ float smem[M * N];
    int tid = threadIdx.x;
    if (tid >= THREADS) return;

    // 创建 2D Tensor
    auto layout = make_layout(make_shape(Int<M>{}, Int<N>{}),
                              make_stride(Int<N>{}, Int<1>{}));
    auto tensor = make_tensor(make_smem_ptr(smem), layout);

    // 线程布局
    auto thr_layout = make_layout(Int<THREADS>{});

    // Partition
    auto my_data = local_partition(tensor, thr_layout, tid);

    if (tid == 0) {
        device_print_separator("Kernel 2: 2D Partition");
        printf("2D Tensor: %d × %d = %d 元素\n", M, N, M*N);
        printf("线程数: %d\n", THREADS);
        printf("每个线程: %d 元素\n\n", int(size(my_data)));
    }

    // 每个线程初始化自己的部分
    for (int i = 0; i < size(my_data); ++i) {
        my_data(i) = tid * 10.0f + i;
    }

    __syncthreads();

    if (tid == 0) {
        printf("前 4 行结果:\n");
        for (int i = 0; i < 4; ++i) {
            printf("  ");
            for (int j = 0; j < N; ++j) {
                printf("%4.0f ", tensor(i, j));
            }
            printf("\n");
        }
    }
}

// ========== Kernel 3: 理解 Partition 的映射关系 ==========
__global__ void kernel_partition_mapping() {
    constexpr int N = 32;
    constexpr int THREADS = 4;
    __shared__ float smem[N];

    int tid = threadIdx.x;
    if (tid >= THREADS) return;

    auto tensor = make_tensor(make_smem_ptr(smem), make_layout(Int<N>{}));
    auto thr_layout = make_layout(Int<THREADS>{});
    auto my_part = local_partition(tensor, thr_layout, tid);

    if (tid == 0) {
        device_print_separator("Kernel 3: Partition 映射关系");
        printf("演示：partition 如何将数据分配给线程\n\n");
    }

    __syncthreads();

    // 每个线程打印自己负责的索引
    printf("Thread %d 负责的全局索引: ", tid);
    for (int i = 0; i < size(my_part); ++i) {
        // 计算在原始 tensor 中的索引
        // 这里我们通过写入唯一值来验证
        my_part(i) = tid * 100.0f + i;
    }
    printf("\n");

    __syncthreads();

    if (tid == 0) {
        printf("\n完整数据 (按 thread_id 区分):\n");
        for (int i = 0; i < N; ++i) {
            if (i % 8 == 0) printf("  ");
            printf("%5.0f ", smem[i]);
            if ((i+1) % 8 == 0) printf("\n");
        }
    }
}

// ========== Kernel 4: 实际应用 - 向量加法 ==========
template <int N, int THREADS>
__global__ void vector_add_with_partition(const float* A, const float* B, float* C) {
    int tid = threadIdx.x;
    if (tid >= THREADS) return;

    // 创建 global memory tensors
    auto layout = make_layout(Int<N>{});
    auto tA = make_tensor(A, layout);
    auto tB = make_tensor(B, layout);
    auto tC = make_tensor(C, layout);

    // 线程布局
    auto thr_layout = make_layout(Int<THREADS>{});

    // Partition: 每个线程得到自己的数据视图
    auto my_A = local_partition(tA, thr_layout, tid);
    auto my_B = local_partition(tB, thr_layout, tid);
    auto my_C = local_partition(tC, thr_layout, tid);

    // 每个线程只处理自己的数据
    for (int i = 0; i < size(my_A); ++i) {
        my_C(i) = my_A(i) + my_B(i);
    }

    if (tid == 0 && blockIdx.x == 0) {
        device_print_separator("Kernel 4: 向量加法");
        printf("使用 partition 简化线程到数据的映射\n");
        printf("每个线程处理 %d 个元素\n", int(size(my_A)));
    }
}

int main() {
    print_separator("CuTe local_partition");

    // ========== 1. 基础 Partition ==========
    print_separator("1. 基础 1D Partition");
    kernel_basic_partition<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========== 2. 2D Partition ==========
    print_separator("2. 2D Tensor Partition");
    kernel_2d_partition<8, 8, 16><<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========== 3. 映射关系 ==========
    print_separator("3. Partition 映射关系");
    kernel_partition_mapping<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========== 4. 实际应用：向量加法 ==========
    print_separator("4. 向量加法应用");

    constexpr int N = 256;
    constexpr int THREADS = 32;

    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    vector_add_with_partition<N, THREADS><<<1, THREADS>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证
    bool pass = true;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            pass = false;
            printf("Error at %d: %.1f + %.1f = %.1f (expected %.1f)\n",
                   i, h_A[i], h_B[i], h_C[i], h_A[i] + h_B[i]);
            break;
        }
    }

    printf("\n验证前 8 个结果:\n");
    for (int i = 0; i < 8; ++i) {
        printf("  %.1f + %.1f = %.1f\n", h_A[i], h_B[i], h_C[i]);
    }
    printf("正确性: %s\n", pass ? "PASS ✓" : "FAIL ✗");

    // ========== 总结 ==========
    print_separator("总结");
    printf("关键概念:\n");
    printf("  • local_partition(tensor, thr_layout, tid)\n");
    printf("  • 自动将数据分配给每个线程\n");
    printf("  • 返回的是 view，不拷贝数据\n");
    printf("  • 简化了线程到数据的映射逻辑\n");
    printf("\n下一步:\n");
    printf("  capstone 会用 partition 实现完整的矩阵-向量乘法\n");

    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
