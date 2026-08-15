// v0: Tensor = 指针 + Layout
//
// 概念讲解见 ../README.md §1 §2。本文件用可运行的例子逐条验证：
//   - Tensor 的创建与查询
//   - 指针为什么必须用 make_gmem_ptr 包装
//   - 切片 T(i,_) / T(_,j) 的 rank 与起点变化
//   - view 语义：写 Tensor 会穿透到底层数组

#include <cute/tensor.hpp>

#include "common.h"

using namespace cute;

int main() {
    print_separator("Tensor = 指针 + Layout");

    // ========== 1. 创建 ==========
    print_separator("1. 创建 Tensor");

    constexpr int M = 4, N = 8;
    float data[M * N];
    for (int i = 0; i < M * N; ++i) data[i] = i;

    auto layout = make_layout(make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));

    // 关键: 指针必须用 make_gmem_ptr 包装。
    // 直接写 make_tensor(data, layout) 会编译失败 —— CuTe 要在类型里
    // 知道这块内存住在哪 (gmem/smem/rmem)，好在后面自动选搬运指令。
    auto T = make_tensor(make_gmem_ptr(data), layout);

    printf("T = ");
    print(T);
    printf("\n  ^ 打印格式是 '指针 o Layout'，o 就是 composition 的记号\n");

    printf("\n查询:\n");
    printf("  size(T)  = %d\n", int(size(T)));
    printf("  rank(T)  = %d\n", int(rank(T)));
    printf("  shape(T) = ");
    print(shape(T));
    printf("\n  stride(T)= ");
    print(stride(T));
    printf("\n");

    printf("\nprint_tensor(T) 打印实际数值:\n");
    print_tensor(T);

    // ========== 2. 索引访问 ==========
    print_separator("2. 索引访问");

    printf("T(i,j) 自动完成 layout(i,j) + 解引用:\n");
    printf("  T(1,3) = %.0f   (layout(1,3)=%d, data[%d]=%.0f)\n", float(T(1, 3)),
           int(layout(1, 3)), int(layout(1, 3)), data[layout(1, 3)]);
    printf("  T(3,7) = %.0f\n", float(T(3, 7)));

    // ========== 3. 切片 ==========
    print_separator("3. 切片: _ 通配符");

    auto row2 = T(2, _);  // 第 2 行
    auto col3 = T(_, 3);  // 第 3 列

    printf("T      = ");
    print(T);
    printf("\n");
    printf("T(2,_) = ");
    print(row2);
    printf("\n");
    printf("T(_,3) = ");
    print(col3);
    printf("\n");

    printf("\n两点变化:\n");
    printf("  1) rank 降低: rank(T)=%d -> rank(T(2,_))=%d  (固定的那维消失)\n", int(rank(T)),
           int(rank(row2)));
    printf("  2) 起点记在指针上, 不在 layout 里:\n");
    printf("     T      基址偏移 0\n");
    printf("     T(2,_) 基址偏移 %d 个元素 (= layout(2,0))\n", int(layout(2, 0)));

    printf("\n内容验证:\n");
    printf("  row2 = ");
    for (int j = 0; j < N; ++j) printf("%.0f ", float(row2(j)));
    printf("   (原矩阵第 2 行)\n");
    printf("  col3 = ");
    for (int i = 0; i < M; ++i) printf("%.0f ", float(col3(i)));
    printf("   (原矩阵第 3 列)\n");

    // ========== 4. view 语义 ==========
    print_separator("4. Tensor 是 view, 不拥有数据");

    printf("修改前: data[11] = %.0f\n", data[11]);
    T(1, 3) = 999.0f;
    printf("执行 T(1,3) = 999 后:\n");
    printf("  T(1,3)   = %.0f\n", float(T(1, 3)));
    printf("  data[11] = %.0f   <- 底层数组被直接改写\n", data[11]);

    printf("\n切片也是 view (不是拷贝):\n");
    row2(0) = -1.0f;
    printf("  执行 row2(0) = -1 后, data[%d] = %.0f\n", int(layout(2, 0)),
           data[layout(2, 0)]);

    printf("\n=> 所有切分操作(切片/local_tile/local_partition)都是零拷贝的 view\n");
    printf("   在 kernel 里可以放心层层切分, 不会产生任何内存搬运\n");

    // ========== 5. 同一块数据的多个 view ==========
    print_separator("5. 同一块数据的多个 view");

    // 复原数据, 便于观察
    for (int i = 0; i < M * N; ++i) data[i] = i;

    auto layout_col = make_layout(make_shape(Int<M>{}, Int<N>{}), make_stride(Int<1>{}, Int<M>{}));
    auto T_col = make_tensor(make_gmem_ptr(data), layout_col);

    printf("同一个 data 指针, 两个不同 Layout:\n");
    printf("  T     (row-major) = ");
    print(T.layout());
    printf("\n  T_col (col-major) = ");
    print(T_col.layout());
    printf("\n");

    printf("\n前两行对照:\n");
    for (int i = 0; i < 2; ++i) {
        printf("  T    row %d: ", i);
        for (int j = 0; j < N; ++j) printf("%2.0f ", float(T(i, j)));
        printf("\n  T_col row %d: ", i);
        for (int j = 0; j < N; ++j) printf("%2.0f ", float(T_col(i, j)));
        printf("\n");
    }
    printf("  <- 数据没动, 只是解释方式不同 (Section 01 §1 的逻辑转置)\n");

    // ========== 6. Layout 运算依然可用 ==========
    print_separator("6. Tensor 里的 Layout 可以继续做代数运算");

    printf("T.layout()            = ");
    print(T.layout());
    printf("\n");
    printf("coalesce(T.layout())  = ");
    print(coalesce(T.layout()));
    printf("   (row-major 合不了, 见 Section 01 §5)\n");
    printf("coalesce(T_col.layout()) = ");
    print(coalesce(T_col.layout()));
    printf("   (col-major 可以合并)\n");

    print_separator("小结");
    printf("  - Tensor = make_tensor(make_gmem_ptr(p), layout)\n");
    printf("  - T(i,j) 自动算偏移并访问\n");
    printf("  - T(i,_) / T(_,j) 切片: rank 降低, 起点进指针\n");
    printf("  - Tensor 是 view, 写它会穿透到底层数组\n");
    printf("\n下一步: v1 演示 GPU 上的 gmem / smem Tensor\n");

    return 0;
}
