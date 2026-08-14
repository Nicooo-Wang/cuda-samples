// v1: CuTe Layout 的代数运算和组合
//
// 学习目标：
//   1. Layout 的 composition (组合)
//   2. Layout 的 complement (补集)
//   3. Layout 的 coalesce (合并维度)
//   4. 嵌套 Layout (hierarchical layout)
//
// 核心概念：
//   - composition(L1, L2): 复合映射，先用 L2 映射，再用 L1 映射
//   - complement(L, size): L 的补集，填充未覆盖的地址空间
//   - coalesce(L): 合并可以合并的连续维度
//   - 嵌套 Layout: 用 Layout 构建层次化的索引结构

#include <cute/tensor.hpp>
#include "common.h"

using namespace cute;

int main() {
    print_separator("CuTe Layout 代数运算");

    // ========== 1. Layout composition (组合) ==========
    print_separator("1. Layout Composition");

    // 假设我们有两个映射：
    //   L1: 将 (i, j) 映射到一维偏移
    //   L2: 将 k 映射到 (i, j)
    // composition(L1, L2) 得到：k -> offset 的直接映射

    auto L1 = make_layout(make_shape(Int<4>{}, Int<8>{}),
                          make_stride(Int<8>{}, Int<1>{}));  // 4x8 row-major

    auto L2 = make_layout(make_shape(Int<32>{}),
                          make_stride(Int<1>{}));  // 32 个元素，连续

    auto L_composed = composition(L1, L2);

    printf("L1 (4x8 row-major) = ");
    print(L1);
    printf("\n");

    printf("L2 (32 连续) = ");
    print(L2);
    printf("\n");

    printf("composition(L1, L2) = ");
    print(L_composed);
    printf("\n");

    printf("验证: composition(L1, L2)(k) = L1(L2(k))\n");
    for (int k = 0; k < 8; ++k) {
        int via_composition = L_composed(k);
        int via_manual = L1(k / 8, k % 8);  // L2 把 k 映射到 (k/8, k%8)
        printf("  k=%d: composed=%2d, manual=%2d\n", k, via_composition, via_manual);
    }

    // ========== 2. Layout complement (补集) ==========
    print_separator("2. Layout Complement");

    // complement 用于找到 Layout 未覆盖的地址空间
    // 例如：一个 warp 的 32 个线程如何覆盖一个 64 元素的向量？

    // 场景：32 个线程，每个线程负责连续的元素
    auto thread_layout = make_layout(Int<32>{});  // 32 个线程，步长 1

    printf("thread_layout (32 threads) = ");
    print(thread_layout);
    printf("\n");
    printf("  size()   = %d\n", int(size(thread_layout)));
    printf("  cosize() = %d\n", int(cosize(thread_layout)));

    // 如果目标是 64 个元素，每个线程需要处理多少个？
    // complement 会生成"剩余维度"的 Layout
    auto value_layout = complement(thread_layout, Int<64>{});

    printf("\nvalue_layout = complement(thread_layout, 64) = ");
    print(value_layout);
    printf("\n");
    printf("  size()   = %d (每个线程处理 2 个元素)\n", int(size(value_layout)));

    // 完整的 Layout：(thread_id, value_id) -> offset
    auto full_layout = make_layout(thread_layout, value_layout);
    printf("\nfull_layout = ");
    print(full_layout);
    printf("\n");

    printf("示例: 前 4 个线程的元素分配\n");
    for (int tid = 0; tid < 4; ++tid) {
        printf("  thread %d: ", tid);
        for (int vid = 0; vid < int(size(value_layout)); ++vid) {
            printf("elem[%2d] ", full_layout(tid, vid));
        }
        printf("\n");
    }

    // ========== 3. Coalesce (维度合并) ==========
    print_separator("3. Layout Coalesce");

    // coalesce 会合并可以合并的连续维度
    // 规则：如果 shape[i] * stride[i] == stride[i+1]，则可以合并

    // 例子：(2, 3, 4) 的 shape，stride 为 (12, 4, 1)
    auto layout_3d = make_layout(make_shape(Int<2>{}, Int<3>{}, Int<4>{}),
                                 make_stride(Int<12>{}, Int<4>{}, Int<1>{}));

    printf("layout_3d = ");
    print(layout_3d);
    printf("\n");
    printf("  size()   = %d\n", int(size(layout_3d)));
    printf("  cosize() = %d\n", int(cosize(layout_3d)));

    // 因为 3*4 = 12, 4*1 = 4，最后两个维度可以合并
    auto layout_coalesced = coalesce(layout_3d);

    printf("\ncoalesce(layout_3d) = ");
    print(layout_coalesced);
    printf("\n");
    printf("  维度从 3D 变成了更简单的形式\n");

    // ========== 4. 嵌套 Layout ==========
    print_separator("4. 嵌套 Layout (Hierarchical)");

    // CuTe 支持嵌套的 Shape/Stride，用于表达层次化的结构
    // 例如：一个 block 有 4 个 warp，每个 warp 有 32 个线程

    auto warp_layout = make_layout(Int<32>{});  // 一个 warp 内 32 个线程
    auto block_layout = make_layout(make_shape(Int<4>{}, Int<32>{}),
                                    make_stride(Int<32>{}, Int<1>{}));  // 4 个 warp

    printf("block_layout (4 warps × 32 threads) = ");
    print(block_layout);
    printf("\n");

    printf("示例: 每个 warp 的第一个线程\n");
    for (int warp_id = 0; warp_id < 4; ++warp_id) {
        printf("  warp %d, thread 0: global_tid = %d\n",
               warp_id, block_layout(warp_id, 0));
    }

    // 嵌套表示：((warp_id, thread_in_warp), ...)
    // 这在 CUTLASS 中大量使用，用于表达 block -> warp -> thread 的层次
    auto nested_layout = make_layout(make_shape(make_shape(Int<4>{}, Int<32>{})),
                                     make_stride(make_stride(Int<32>{}, Int<1>{})));

    printf("\nnested_layout = ");
    print(nested_layout);
    printf("\n");

    // ========== 5. 实际应用：Tile 一个矩阵 ==========
    print_separator("5. 实际应用: Tile 一个矩阵");

    // 场景：一个 128x128 的矩阵，分成 16x16 的 tile
    // 外层：8x8 个 tile
    // 内层：每个 tile 内 16x16 个元素

    auto tile_shape = make_shape(Int<16>{}, Int<16>{});
    auto tile_layout = make_layout(tile_shape,
                                   make_stride(Int<128>{}, Int<1>{}));  // row-major

    printf("单个 tile 的 layout (16x16, 在 128x128 矩阵中) = ");
    print(tile_layout);
    printf("\n");

    // 8x8 个 tile 的布局
    auto tile_grid = make_layout(make_shape(Int<8>{}, Int<8>{}),
                                 make_stride(Int<16*128>{}, Int<16>{}));  // tile 步长

    printf("\ntile_grid (8x8 tiles) = ");
    print(tile_grid);
    printf("\n");

    printf("示例: tile (1, 2) 的起始偏移 = %d\n", tile_grid(1, 2));
    printf("       应该是 1*16*128 + 2*16 = %d\n", 1*16*128 + 2*16);

    // ========== 6. 总结 ==========
    print_separator("总结");
    printf("高级 Layout 操作:\n");
    printf("  • composition(L1, L2): 复合映射 k -> L1(L2(k))\n");
    printf("  • complement(L, size): 计算补集维度\n");
    printf("  • coalesce(L): 合并连续维度\n");
    printf("  • 嵌套 Layout: 支持层次化的索引结构\n");
    printf("\n这些操作是 CuTe 强大表达能力的基础\n");
    printf("下一步: v2 会介绍如何用 Layout 创建 Tensor\n");

    return 0;
}
