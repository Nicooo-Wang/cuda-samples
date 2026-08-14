// v2: 用 Layout 操作实际数据 - 转置示例
//
// 学习目标：
//   1. 使用 Layout 在 CPU 上操作实际数据
//   2. 理解 Layout 如何改变数据的访问模式
//   3. 通过矩阵转置理解 row-major 和 column-major 的转换
//
// 实战场景：
//   给定一个 row-major 存储的矩阵，如何用 Layout 实现转置？
//   关键：转置就是改变访问模式，而不需要移动数据

#include <cute/tensor.hpp>
#include "common.h"
#include <cstdlib>
#include <cstring>

using namespace cute;

// 打印一个小矩阵（用于可视化）
template <typename T>
void print_matrix(const T* data, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        printf("  ");
        for (int j = 0; j < cols; ++j) {
            printf("%3d ", (int)data[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    print_separator("CuTe Layout 实战: 矩阵转置");

    constexpr int M = 4;
    constexpr int N = 8;

    // ========== 1. 创建测试数据 ==========
    print_separator("1. 创建测试数据");

    int* data = (int*)malloc(M * N * sizeof(int));

    // 初始化：row-major 填充 0, 1, 2, ..., 31
    for (int i = 0; i < M * N; ++i) {
        data[i] = i;
    }

    print_matrix(data, M, N, "原始矩阵 (row-major)");

    // ========== 2. 用 Layout 描述 row-major 访问 ==========
    print_separator("2. Row-major Layout");

    // row-major: 行步长 N，列步长 1
    auto layout_row = make_layout(make_shape(Int<M>{}, Int<N>{}),
                                  make_stride(Int<N>{}, Int<1>{}));

    printf("layout_row = ");
    print(layout_row);
    printf("\n\n");

    // 验证：用 layout 访问数据
    printf("通过 layout_row 访问前 2 行:\n");
    for (int i = 0; i < 2; ++i) {
        printf("  row %d: ", i);
        for (int j = 0; j < N; ++j) {
            int offset = layout_row(i, j);
            printf("%3d ", data[offset]);
        }
        printf("\n");
    }

    // ========== 3. 转置：只需改变 Layout! ==========
    print_separator("3. 转置 - 交换 Shape 和 Stride");

    // 转置的 Layout: 交换 shape 和 stride 的对应关系
    // 原来: (M, N) with stride (N, 1)
    // 转置: (N, M) with stride (1, N)  <- 注意顺序
    auto layout_transposed = make_layout(make_shape(Int<N>{}, Int<M>{}),
                                        make_stride(Int<1>{}, Int<N>{}));

    printf("layout_transposed = ");
    print(layout_transposed);
    printf("\n\n");

    // 用转置后的 layout 访问同一块数据
    printf("通过 layout_transposed 访问（无需移动数据）:\n");
    printf("转置后矩阵 (%dx%d):\n", N, M);
    for (int i = 0; i < N; ++i) {
        printf("  ");
        for (int j = 0; j < M; ++j) {
            int offset = layout_transposed(i, j);
            printf("%3d ", data[offset]);
        }
        printf("\n");
    }

    // ========== 4. 验证转置的正确性 ==========
    print_separator("4. 验证转置");

    printf("对比原始和转置:\n");
    printf("  原始 A[1][3] = data[1*8+3] = data[11] = %d\n", data[11]);
    printf("  转置 A'[3][1] 应该等于原始 A[1][3]\n");
    int transposed_offset = layout_transposed(3, 1);
    printf("  转置 A'[3][1] = data[%d] = %d ✓\n", transposed_offset, data[transposed_offset]);

    // ========== 5. 物理转置 vs 逻辑转置 ==========
    print_separator("5. 物理转置 vs 逻辑转置");

    printf("逻辑转置（改变 Layout）:\n");
    printf("  • 优点: 零拷贝，立即完成\n");
    printf("  • 缺点: 访问模式可能变成非连续（影响性能）\n");
    printf("  • 适用: 转置后立即用于另一个操作，可以 fuse\n\n");

    printf("物理转置（重排数据）:\n");
    printf("  • 优点: 转置后的访问是连续的\n");
    printf("  • 缺点: 需要额外内存和时间\n");
    printf("  • 适用: 需要多次访问转置后的数据\n\n");

    // 演示物理转置
    int* data_transposed = (int*)malloc(M * N * sizeof(int));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            int src_offset = layout_transposed(i, j);
            int dst_offset = i * M + j;  // 转置后按 row-major 存储
            data_transposed[dst_offset] = data[src_offset];
        }
    }

    print_matrix(data_transposed, N, M, "物理转置后的矩阵 (新内存)");

    // ========== 6. Layout 的强大之处 ==========
    print_separator("6. Layout 的抽象能力");

    printf("关键洞察:\n");
    printf("  • Layout 将\"数据布局\"和\"访问模式\"分离\n");
    printf("  • 同一块数据可以有多种解释（多个 Layout）\n");
    printf("  • 很多操作可以通过改变 Layout 而非移动数据来完成\n\n");

    printf("在 GPU 编程中的应用:\n");
    printf("  • Shared memory 的 bank conflict 优化: 改变 stride\n");
    printf("  • Tensor Core 的数据排布: 用 Layout 描述 WMMA 的 fragment\n");
    printf("  • CUTLASS 的核心: 所有 Tensor 都是 (指针 + Layout)\n");

    // ========== 7. 练习：创建一个 strided 访问 ==========
    print_separator("7. 练习: 每隔一列采样");

    // 创建一个 Layout：只访问偶数列
    auto layout_strided_cols = make_layout(make_shape(Int<M>{}, Int<N/2>{}),
                                           make_stride(Int<N>{}, Int<2>{}));  // 列步长 2

    printf("layout_strided_cols = ");
    print(layout_strided_cols);
    printf("\n\n");

    printf("只访问偶数列 (0, 2, 4, 6):\n");
    for (int i = 0; i < M; ++i) {
        printf("  ");
        for (int j = 0; j < N/2; ++j) {
            int offset = layout_strided_cols(i, j);
            printf("%3d ", data[offset]);
        }
        printf("\n");
    }

    // ========== 8. 总结 ==========
    print_separator("总结");
    printf("核心要点:\n");
    printf("  • Layout 是数据访问模式的数学描述\n");
    printf("  • 改变 Layout = 改变访问模式，无需移动数据\n");
    printf("  • 转置、采样、重排都可以通过 Layout 表达\n");
    printf("  • CuTe Tensor = 指针 + Layout（下一个教程）\n");
    printf("\n下一步: cute_02 会介绍 Tensor 的创建和使用\n");

    free(data);
    free(data_transposed);
    return 0;
}
