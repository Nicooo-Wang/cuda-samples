// v0: CuTe Layout 最基础概念
//
// ========== CuTe 是什么？ ==========
//
// CuTe (CUDA Templates) 是 CUTLASS 3.x 的核心抽象层，提供了一套编译期的
// 张量（Tensor）和布局（Layout）抽象，让 GPU 编程更简洁、更高效。
//
// 核心理念：
//   • 用类型系统（模板）描述数据的形状和访问模式
//   • 编译期计算：让编译器在编译时完成大量计算，而不是运行时
//   • 零开销抽象：高层抽象不牺牲性能，生成的代码和手写一样快
//
// ========== Int<N> vs int：为什么要用编译期常量？ ==========
//
// 在 CuTe 中，你会经常看到 Int<8>{}、Int<4>{} 这样的写法，而不是普通的 int。
//
// Int<N> 是编译期常量，它的好处是：
//   1. 编译器知道确切的值，可以：
//      - 展开循环（loop unrolling）
//      - 优化寄存器分配
//      - 消除分支判断
//      - 内联所有计算
//
//   2. 类型系统可以携带更多信息：
//      - Int<8> 和 Int<16> 是不同的类型
//      - 编译期就能检测到形状不匹配
//
//   3. 生成更高效的代码：
//      - PTX 汇编中直接是立即数，不需要从内存/寄存器读取
//
// 例子：
//   int n = 8;              // 运行时变量，编译器不知道具体值
//   Int<8> n_compile{};     // 编译期常量，编译器可以做激进优化
//
// ========== Shape 和 Stride：描述数据的形状和步长 ==========
//
// **Shape（形状）**：描述每个维度有多少个元素
//   - 一维：Shape = (8)           → 8 个元素
//   - 二维：Shape = (4, 8)        → 4 行 8 列
//   - 三维：Shape = (2, 4, 8)     → 2×4×8 的立方体
//
//   类比：Shape 就像房子的"户型图" —— 告诉你有几间房，每间房多大
//
// **Stride（步长）**：描述每个维度步进一个单位时，线性地址跳多远
//   - 一维：Stride = (1)          → 连续存储
//   - 二维 row-major：(4, 8) with Stride = (8, 1)
//       换一行（维度 0）：跳 8 个元素
//       换一列（维度 1）：跳 1 个元素
//   - 二维 column-major：(4, 8) with Stride = (1, 4)
//       换一行：跳 1 个元素
//       换一列：跳 4 个元素
//
//   类比：Stride 就像"门牌号的编号规则" —— 告诉你从一个房间到另一个房间，
//        门牌号要加多少
//
// **Layout（布局）**：Shape + Stride 的组合，完整描述如何访问数据
//
//   Layout = (Shape, Stride)
//
//   它定义了一个映射函数：
//       坐标 (i, j, k, ...) → 线性偏移 offset
//
//   计算公式：
//       offset = i * stride[0] + j * stride[1] + k * stride[2] + ...
//
// ========== 为什么需要 Layout？ ==========
//
// 在 GPU 编程中，我们经常需要：
//   1. 将多维数组映射到一维内存（flatten）
//   2. 转置、重排、切片数据
//   3. 描述 shared memory 的 bank conflict 优化（padding）
//   4. 表达 warp/block/thread 的层次化结构
//
// 如果没有 Layout 抽象：
//   - 每个操作都要手写偏移计算：i * N + j, (i * stride_i + j * stride_j) ...
//   - 容易出错，难以维护
//   - 编译器难以优化（因为是运行时计算）
//
// 有了 Layout 抽象：
//   - layout(i, j) 自动计算偏移
//   - 改变访问模式只需改变 Layout，不需要改代码逻辑
//   - 编译期计算，零运行时开销
//
// ========== 学习目标 ==========
//
// 完成本教程后，你应该能够：
//   1. 理解 Int<N> 编译期常量的作用
//   2. 理解 Shape 和 Stride 的含义
//   3. 学会创建和打印 Layout
//   4. 理解 Layout 如何将逻辑坐标映射到线性偏移
//
// ========== 开始实战！ ==========

#include <cute/tensor.hpp>
#include "common.h"

using namespace cute;

// ========== 可视化示例：理解 Shape 和 Stride ==========
//
// 假设有一个 4x8 的矩阵，在内存中是一维数组：
//
//   内存: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ..., 31]
//
// **Row-major 布局：**
//   Shape = (4, 8), Stride = (8, 1)
//
//   逻辑视图（4 行 8 列）：
//      列:  0   1   2   3   4   5   6   7
//   行 0: [ 0,  1,  2,  3,  4,  5,  6,  7]
//   行 1: [ 8,  9, 10, 11, 12, 13, 14, 15]
//   行 2: [16, 17, 18, 19, 20, 21, 22, 23]
//   行 3: [24, 25, 26, 27, 28, 29, 30, 31]
//
//   坐标 (1, 3) 的偏移 = 1 * 8 + 3 * 1 = 11  ✓
//   坐标 (2, 5) 的偏移 = 2 * 8 + 5 * 1 = 21  ✓
//
// **Column-major 布局：**
//   Shape = (4, 8), Stride = (1, 4)
//
//   逻辑视图（同样是 4 行 8 列，但内存排布不同）：
//      列:  0   1   2   3   4   5   6   7
//   行 0: [ 0,  4,  8, 12, 16, 20, 24, 28]
//   行 1: [ 1,  5,  9, 13, 17, 21, 25, 29]
//   行 2: [ 2,  6, 10, 14, 18, 22, 26, 30]
//   行 3: [ 3,  7, 11, 15, 19, 23, 27, 31]
//
//   坐标 (1, 3) 的偏移 = 1 * 1 + 3 * 4 = 13  ✓
//   坐标 (2, 5) 的偏移 = 2 * 1 + 5 * 4 = 22  ✓
//
// 关键洞察：
//   • 同样的逻辑坐标 (i, j)，不同的 Stride 会映射到不同的内存位置
//   • Layout 抽象让我们可以用统一的方式处理不同的内存排布
//   • 这就是为什么 CuTe 如此强大！

int main() {
    print_separator("CuTe Layout 基础：Shape 和 Stride");

    // ========== 1. 一维 Layout ==========
    print_separator("1. 一维 Layout");

    // 最简单的一维 Layout：8 个元素，步长为 1（连续存储）
    auto layout_1d = make_layout(Int<8>{});

    printf("layout_1d = ");
    print(layout_1d);
    printf("\n");

    // 验证坐标映射：逻辑坐标 -> 线性偏移
    printf("坐标映射示例:\n");
    for (int i = 0; i < 8; ++i) {
        int offset = layout_1d(i);
        printf("  layout_1d(%d) = %d\n", i, offset);
    }

    // ========== 2. 二维 Layout (row-major) ==========
    print_separator("2. 二维 Layout (row-major)");

    // 4 行 8 列的 row-major 布局
    // Shape:  (4, 8)  - 4 行，每行 8 列
    // Stride: (8, 1)  - 行步长 8，列步长 1
    auto layout_row_major = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                        make_stride(Int<8>{}, Int<1>{}));

    printf("layout_row_major = ");
    print(layout_row_major);
    printf("\n");

    printf("坐标映射示例 (row-major):\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            int offset = layout_row_major(i, j);
            printf("  (%d, %d) -> %2d", i, j, offset);
            if (j == 7) printf("\n");
        }
    }

    // ========== 3. 二维 Layout (column-major) ==========
    print_separator("3. 二维 Layout (column-major)");

    // 同样是 4 行 8 列，但改成 column-major
    // Shape:  (4, 8)  - 4 行 8 列
    // Stride: (1, 4)  - 行步长 1，列步长 4
    auto layout_col_major = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                        make_stride(Int<1>{}, Int<4>{}));

    printf("layout_col_major = ");
    print(layout_col_major);
    printf("\n");

    printf("坐标映射示例 (column-major):\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            int offset = layout_col_major(i, j);
            printf("  (%d, %d) -> %2d", i, j, offset);
            if (j == 7) printf("\n");
        }
    }

    // ========== 4. 理解 Layout 的 size 和 cosize ==========
    print_separator("4. Layout 的 size 和 cosize");

    // size:   逻辑元素总数（Shape 的乘积）
    // cosize: 线性地址空间需要的大小（最大偏移 + 1）
    printf("layout_row_major:\n");
    printf("  size()   = %d (逻辑元素数: 4 * 8)\n", int(size(layout_row_major)));
    printf("  cosize() = %d (地址空间大小)\n", int(cosize(layout_row_major)));

    printf("\nlayout_col_major:\n");
    printf("  size()   = %d\n", int(size(layout_col_major)));
    printf("  cosize() = %d\n", int(cosize(layout_col_major)));

    // ========== 5. 带步长的特殊 Layout ==========
    print_separator("5. 自定义步长的 Layout");

    // 创建一个 4x8 的 Layout，但行步长是 16 (而不是 8)
    // 这在实际中对应于：有一个更大的矩阵，我们只取了前 8 列
    auto layout_strided = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                      make_stride(Int<16>{}, Int<1>{}));

    printf("layout_strided = ");
    print(layout_strided);
    printf("\n");

    printf("坐标映射示例 (行步长 16):\n");
    printf("  size()   = %d (逻辑元素数)\n", int(size(layout_strided)));
    printf("  cosize() = %d (地址空间需要 4*16 = 64)\n", int(cosize(layout_strided)));

    // 显示前两行的映射
    for (int i = 0; i < 2; ++i) {
        printf("  第 %d 行: ", i);
        for (int j = 0; j < 8; ++j) {
            printf("%2d ", layout_strided(i, j));
        }
        printf("\n");
    }

    // ========== 6. 总结 ==========
    print_separator("总结");
    printf("关键概念:\n");
    printf("  • Layout = (Shape, Stride)\n");
    printf("  • Shape:  逻辑维度大小\n");
    printf("  • Stride: 每个维度的步长\n");
    printf("  • layout(i, j, ...) 计算线性偏移\n");
    printf("  • size():   逻辑元素总数\n");
    printf("  • cosize(): 实际占用的地址空间\n");
    printf("\n下一步: v1 会介绍 Layout 的代数运算\n");

    return 0;
}
