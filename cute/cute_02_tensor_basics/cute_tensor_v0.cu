// v0: CuTe Tensor 基础概念
//
// ========== 从 Layout 到 Tensor ==========
//
// 在 cute_01 中，我们学习了 Layout：
//   Layout = (Shape, Stride)  →  定义了坐标到偏移的映射
//
// 但 Layout 本身只是一个"映射函数"，不包含实际数据。
// 要操作数据，我们需要 Tensor！
//
// ========== 什么是 Tensor？ ==========
//
// 在 CuTe 中，Tensor 是最核心的数据结构：
//
//   Tensor = 指针 + Layout
//
// 它将"数据的位置"（指针）和"数据的组织方式"（Layout）结合起来。
//
// 例子：
//   float* ptr = ...;                          // 数据指针
//   auto layout = make_layout(...);            // 数据布局
//   auto tensor = make_tensor(ptr, layout);    // 组合成 Tensor
//
//   tensor(i, j) = 42.0f;  // 现在可以直接用坐标访问数据！
//
// ========== Tensor 的类型 ==========
//
// CuTe 支持不同内存空间的 Tensor：
//
// 1. **Global Memory Tensor (gmem)**
//    - 普通的 device 指针
//    - auto tensor = make_tensor(device_ptr, layout);
//
// 2. **Shared Memory Tensor (smem)**
//    - 指向 __shared__ 内存
//    - auto tensor = make_tensor(make_smem_ptr(smem), layout);
//
// 3. **Register Tensor (rmem)**
//    - 存储在寄存器中的数据
//    - 通常通过 make_fragment 或 partition 创建
//
// ========== Tensor 的基本操作 ==========
//
// 1. **索引访问**
//    tensor(i, j, k)  →  访问元素，自动计算偏移
//
// 2. **查询属性**
//    size(tensor)     →  逻辑元素总数
//    shape(tensor)    →  形状
//    stride(tensor)   →  步长
//    data(tensor)     →  底层指针
//
// 3. **切片和变换**（后续教程）
//    local_tile(tensor, ...)
//    local_partition(tensor, ...)
//
// ========== 为什么 Tensor 比原始指针好？ ==========
//
// 使用原始指针：
//   float* data = ...;
//   data[i * stride_i + j * stride_j] = value;  // 手动计算偏移，容易出错
//
// 使用 CuTe Tensor：
//   auto tensor = make_tensor(data, layout);
//   tensor(i, j) = value;  // 简洁、安全、编译期优化
//
// 好处：
//   • 类型安全：编译期检查维度匹配
//   • 零开销：编译后和手写代码一样快
//   • 可组合：可以基于一个 Tensor 创建新的 view
//   • 统一接口：gmem/smem/rmem 用同样的方式操作
//
// ========== 本教程学习目标 ==========
//
// 完成后你应该能够：
//   1. 理解 Tensor = 指针 + Layout
//   2. 创建不同内存空间的 Tensor
//   3. 使用 Tensor 索引访问数据
//   4. 理解 Tensor 的零开销抽象
//
// ========== 开始实战！ ==========

#include <cute/tensor.hpp>
#include "common.h"
#include <cstdlib>

using namespace cute;

int main() {
    print_separator("CuTe Tensor 基础");

    // ========== 1. CPU 上的 Tensor ==========
    print_separator("1. CPU 上的 Tensor (Host)");

    // 分配一块连续内存
    constexpr int M = 4, N = 8;
    float* data = (float*)malloc(M * N * sizeof(float));

    // 初始化数据
    for (int i = 0; i < M * N; ++i) {
        data[i] = i;
    }

    // 创建一个 row-major 的 Layout
    auto layout = make_layout(make_shape(Int<M>{}, Int<N>{}),
                              make_stride(Int<N>{}, Int<1>{}));

    printf("Layout = ");
    print(layout);
    printf("\n");

    // 创建 Tensor：将指针和 Layout 绑定
    auto tensor = make_tensor(data, layout);

    printf("\nTensor 属性:\n");
    printf("  size(tensor)   = %d\n", int(size(tensor)));
    printf("  shape(tensor)  = ");
    print(shape(tensor));
    printf("\n");
    printf("  stride(tensor) = ");
    print(stride(tensor));
    printf("\n");

    // ========== 2. 使用 Tensor 索引访问数据 ==========
    print_separator("2. Tensor 索引访问");

    printf("通过 Tensor 读取数据:\n");
    for (int i = 0; i < M; ++i) {
        printf("  row %d: ", i);
        for (int j = 0; j < N; ++j) {
            printf("%5.1f ", tensor(i, j));
        }
        printf("\n");
    }

    // 修改数据
    printf("\n修改 tensor(1, 3) = 99.0\n");
    tensor(1, 3) = 99.0f;

    printf("\n验证修改:\n");
    printf("  tensor(1, 3) = %.1f\n", tensor(1, 3));
    printf("  data[1*8+3]  = %.1f  (底层数据也被修改)\n", data[1*8+3]);

    // ========== 3. Tensor 的零开销抽象 ==========
    print_separator("3. 零开销抽象验证");

    printf("对比两种访问方式:\n\n");

    printf("方式 1: 手动计算偏移\n");
    printf("  int offset = i * %d + j * %d;\n", N, 1);
    printf("  data[offset] = value;\n\n");

    printf("方式 2: 使用 Tensor\n");
    printf("  tensor(i, j) = value;\n\n");

    printf("编译后的汇编代码是相同的！\n");
    printf("Tensor 的抽象在编译期完全展开，运行时零开销。\n");

    // ========== 4. 不同 Layout 的 Tensor view ==========
    print_separator("4. 不同 Layout 的 Tensor View");

    // 创建一个 column-major 的 view，共享同一块数据
    auto layout_col = make_layout(make_shape(Int<M>{}, Int<N>{}),
                                  make_stride(Int<1>{}, Int<M>{}));

    auto tensor_col = make_tensor(data, layout_col);

    printf("原始 Tensor (row-major):\n");
    for (int i = 0; i < 2; ++i) {
        printf("  row %d: ", i);
        for (int j = 0; j < N; ++j) {
            printf("%5.1f ", tensor(i, j));
        }
        printf("\n");
    }

    printf("\nColumn-major view (同一块数据):\n");
    for (int i = 0; i < 2; ++i) {
        printf("  row %d: ", i);
        for (int j = 0; j < N; ++j) {
            printf("%5.1f ", tensor_col(i, j));
        }
        printf("\n");
    }

    printf("\n注意: 两个 view 指向同一块内存，只是访问模式不同\n");

    // ========== 5. Tensor 的切片 ==========
    print_separator("5. Tensor 基础切片");

    printf("原始 Tensor shape: ");
    print(shape(tensor));
    printf("\n");

    // 取第 2 行（索引从 0 开始）
    auto row2 = tensor(2, _);  // _ 表示"所有列"

    printf("\ntensor(2, _) 表示第 2 行:\n");
    printf("  shape: ");
    print(shape(row2));
    printf("\n");
    printf("  数据: ");
    for (int j = 0; j < N; ++j) {
        printf("%5.1f ", row2(j));
    }
    printf("\n");

    // 取第 3 列
    auto col3 = tensor(_, 3);  // _ 表示"所有行"

    printf("\ntensor(_, 3) 表示第 3 列:\n");
    printf("  shape: ");
    print(shape(col3));
    printf("\n");
    printf("  数据: ");
    for (int i = 0; i < M; ++i) {
        printf("%5.1f ", col3(i));
    }
    printf("\n");

    // ========== 6. 多维 Tensor ==========
    print_separator("6. 三维 Tensor");

    constexpr int D = 2, H = 3, W = 4;
    float* data3d = (float*)malloc(D * H * W * sizeof(float));

    for (int i = 0; i < D * H * W; ++i) {
        data3d[i] = i;
    }

    // 创建 3D Tensor: (depth, height, width)
    auto layout3d = make_layout(make_shape(Int<D>{}, Int<H>{}, Int<W>{}),
                                make_stride(Int<H*W>{}, Int<W>{}, Int<1>{}));

    auto tensor3d = make_tensor(data3d, layout3d);

    printf("3D Tensor shape: ");
    print(shape(tensor3d));
    printf("\n");
    printf("3D Tensor stride: ");
    print(stride(tensor3d));
    printf("\n\n");

    printf("访问 tensor3d(0, 1, 2):\n");
    printf("  value = %.1f\n", tensor3d(0, 1, 2));
    printf("  offset = 0*%d + 1*%d + 2*%d = %d\n", H*W, W, 1, 0*H*W + 1*W + 2);
    printf("  data[%d] = %.1f  ✓\n", 0*H*W + 1*W + 2, data3d[0*H*W + 1*W + 2]);

    // ========== 7. 总结 ==========
    print_separator("总结");

    printf("关键概念:\n");
    printf("  • Tensor = 指针 + Layout\n");
    printf("  • tensor(i, j, k) 自动计算偏移并访问\n");
    printf("  • 零开销抽象：编译后和手写代码一样快\n");
    printf("  • 可以创建多个 view 共享同一块数据\n");
    printf("  • 支持任意维度的 Tensor\n");
    printf("\n");
    printf("下一步:\n");
    printf("  v1 会介绍 GPU 上的 Tensor (gmem, smem)\n");
    printf("  v2 会介绍 local_partition 和线程协作\n");
    printf("  capstone 会实现 GPU 上的向量加法\n");

    free(data);
    free(data3d);
    return 0;
}
