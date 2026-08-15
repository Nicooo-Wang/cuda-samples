// v2: local_tile 与 local_partition —— 两级切分范式
//
// 概念讲解见 ../README.md §3 §4 §5。本文件验证：
//   - local_tile 取块：各块 layout 相同、起点在指针上
//   - local_tile 底下就是 zipped_divide (composition 的封装)
//   - local_partition 分线程：thr_layout 决定分配模式
//   - thr_layout 写错会静默出错 (每个线程拿到相同数据)
//   - tile -> partition 两级范式

#include <cute/tensor.hpp>

#include "common.h"

using namespace cute;

int main() {
    print_separator("local_tile 与 local_partition");

    // ========== 1. local_tile 取块 ==========
    print_separator("1. local_tile: 把大 Tensor 切成块");

    constexpr int BM = 8, BN = 8;
    float big[BM * BN];
    for (int i = 0; i < BM * BN; ++i) big[i] = i;

    auto BT = make_tensor(make_gmem_ptr(big),
                          make_layout(make_shape(Int<BM>{}, Int<BN>{}),
                                      make_stride(Int<BN>{}, Int<1>{})));
    printf("BT = ");
    print(BT);
    printf("   (8x8, 值 0-63)\n");

    auto tile_shape = make_shape(Int<4>{}, Int<4>{});
    auto t00 = local_tile(BT, tile_shape, make_coord(0, 0));
    auto t01 = local_tile(BT, tile_shape, make_coord(0, 1));
    auto t11 = local_tile(BT, tile_shape, make_coord(1, 1));

    printf("\n切成 4x4 的块:\n");
    printf("  块(0,0) layout=");
    print(t00.layout());
    printf("  首元素=%.0f\n", float(t00(0, 0)));
    printf("  块(0,1) layout=");
    print(t01.layout());
    printf("  首元素=%.0f\n", float(t01(0, 0)));
    printf("  块(1,1) layout=");
    print(t11.layout());
    printf("  首元素=%.0f\n", float(t11(0, 0)));

    printf("\n关键观察: 三个块的 layout 完全相同, 只有首元素不同\n");
    printf("  => 块的位置记在指针上, 不在 Layout 里\n");
    printf("  => 好处: 处理每个块的代码可以完全一致\n");

    printf("\n块(1,1) 的内容 (原矩阵 row 4-7 x col 4-7):\n");
    print_tensor(t11);

    // ========== 2. local_tile 背后是 composition ==========
    print_separator("2. local_tile 底下就是 zipped_divide");

    auto zipped = zipped_divide(BT, tile_shape);
    printf("zipped_divide(BT, (4,4)) = ");
    print(zipped.layout());
    printf("\n");
    printf("  嵌套结构: ((块内坐标), (块号))\n");
    printf("    mode0 = ");
    print(get<0>(zipped.layout()));
    printf("   <- 块内怎么走\n");
    printf("    mode1 = ");
    print(get<1>(zipped.layout()));
    printf("   <- 块与块之间怎么走\n");
    printf("       row 方向跳 32 = 4 行 x 行距 8 ; col 方向跳 4\n");
    printf("\nlocal_tile(BT, tile, coord) 就是: 把块号固定成 coord, 留下块内部分\n");
    printf("这正是 Section 01 §3.4 讲的嵌套 layout 的实际用途\n");

    printf("\ndivide 家族的三种打包方式:\n");
    printf("  zipped_divide = ");
    print(zipped_divide(BT, tile_shape).layout());
    printf("\n  tiled_divide  = ");
    print(tiled_divide(BT, tile_shape).layout());
    printf("\n  flat_divide   = ");
    print(flat_divide(BT, tile_shape).layout());
    printf("\n");

    printf("\ntiler 也可以只切一部分维度:\n");
    auto strip = local_tile(BT, make_shape(Int<2>{}, Int<8>{}), make_coord(1, 0));
    printf("  local_tile(BT, (2,8), (1,0)) = ");
    print(strip.layout());
    printf("  首元素=%.0f\n", float(strip(0, 0)));
    printf("  <- 按行条带切分, 取 row 2-3\n");

    // ========== 3. local_partition 分线程 ==========
    print_separator("3. local_partition: 把数据分给线程");

    constexpr int VN = 16;
    float v[VN];
    for (int i = 0; i < VN; ++i) v[i] = i;
    auto VT = make_tensor(make_gmem_ptr(v), make_layout(Int<VN>{}, Int<1>{}));

    printf("16 个元素 (值 0-15) 分给 4 个线程, thr_layout = 4:1\n\n");
    auto thr4 = make_layout(Int<4>{}, Int<1>{});
    for (int tid = 0; tid < 4; ++tid) {
        auto p = local_partition(VT, thr4, tid);
        printf("  tid=%d 拿到: ", tid);
        for (int i = 0; i < size(p); ++i) printf("%2.0f ", float(p(i)));
        printf("\n");
    }
    printf("\n注意: 每个线程拿到的是**跨步的** 4 个, 不是连续的 4 个\n");
    printf("  => 同一轮里 4 个线程访问下标 0,1,2,3 —— 跨线程连续 = 合并访存\n");
    printf("  => 这正是 GPU 想要的默认行为\n");

    printf("\npartition 的 layout 就是 thr_layout 的 complement:\n");
    auto p0 = local_partition(VT, thr4, 0);
    printf("  local_partition(VT, 4:1, 0).layout = ");
    print(p0.layout());
    printf("\n  complement(4:1, 16)                = ");
    print(complement(thr4, Int<VN>{}));
    printf("\n  <- 印证 Section 01 §4: complement 不是理论, 它就是 partition 的实现\n");

    // ========== 4. thr_layout 陷阱 ==========
    print_separator("4. 陷阱: thr_layout 写错会静默出错");

    printf("把 thr_layout 从 4:1 换成 4:4:\n\n");
    auto thr_bad = make_layout(Int<4>{}, Int<4>{});
    for (int tid = 0; tid < 4; ++tid) {
        auto p = local_partition(VT, thr_bad, tid);
        printf("  tid=%d 拿到: ", tid);
        for (int i = 0; i < size(p); ++i) printf("%2.0f ", float(p(i)));
        printf("\n");
    }
    printf("\n四个线程拿到完全相同的元素!\n");
    printf("  - 12 个元素从没被碰过\n");
    printf("  - 4 个元素被重复写 4 次 (race condition)\n");
    printf("  - 不报错、不警告, 跑起来只是结果不对\n");
    printf("\n=> 记住: thr_layout 的 stride 通常应该是 1 (线程连续编号)\n");
    printf("   这是 partition 最常见的错误来源, 且极难调试\n");

    // ========== 5. 2D partition ==========
    print_separator("5. 2D 线程布局");

    auto thr2d = make_layout(make_shape(Int<2>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
    printf("thr_layout = ");
    print(thr2d);
    printf("   (2x4 = 8 个线程)\n\n");
    for (int tid = 0; tid < 4; ++tid) {
        auto p = local_partition(BT, thr2d, tid);
        printf("  tid=%d layout=", tid);
        print(p.layout());
        printf("  首元素=%.0f\n", float(p(0, 0)));
    }
    printf("\n同样是 layout 相同、首元素不同的模式\n");
    printf("相邻 tid 首元素差 1 => 沿最后一维连续 => 合并访存友好\n");

    // ========== 6. 两级范式 ==========
    print_separator("6. 标准范式: tile -> partition");

    printf("真实 kernel 里这两个算子几乎总是连用:\n\n");
    printf("  1) CTA 层: local_tile 取出本 block 负责的块\n");
    printf("  2) Thread 层: local_partition 取出本线程负责的元素\n");
    printf("  3) 直接对结果读写, 下标都算好了\n\n");

    auto blk = local_tile(BT, tile_shape, make_coord(1, 0));
    printf("1) local_tile(BT, (4,4), (1,0)) = ");
    print(blk.layout());
    printf("  首元素=%.0f\n", float(blk(0, 0)));

    auto thr22 = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<2>{}, Int<1>{}));
    printf("2) 再 partition 给 2x2 = 4 个线程:\n");
    for (int tid = 0; tid < 4; ++tid) {
        auto mine = local_partition(blk, thr22, tid);
        printf("   tid=%d layout=", tid);
        print(mine.layout());
        printf("  值: ");
        for (int j = 0; j < size<1>(mine); ++j)
            for (int i = 0; i < size<0>(mine); ++i) printf("%.0f ", float(mine(i, j)));
        printf("\n");
    }

    printf("\n这个两级结构对应 GPU 的两级并行 (grid 里的 block, block 里的 thread)\n");
    printf("而两级都只是 view 变换, 没有任何数据搬运\n");

    // ========== 7. make_fragment_like ==========
    print_separator("7. 寄存器 Tensor: make_fragment_like");

    auto mine = local_partition(blk, thr22, 0);
    printf("mine 的 layout          = ");
    print(mine.layout());
    printf("   (指向大数组, stride 跨得远)\n");

    auto frag_like = make_tensor_like(mine);
    auto frag = make_fragment_like(mine);
    printf("make_tensor_like(mine)   = ");
    print(frag_like.layout());
    printf("\n");
    printf("make_fragment_like(mine) = ");
    print(frag.layout());
    printf("\n");
    printf("\n两者都丢掉了原来的跨步 stride, 换成紧密排布\n");
    printf("  => 新数据是独立的一小块, 没必要跨步\n");
    printf("  => make_fragment_like 保证第一维 stride=1, 最利于向量化和寄存器分配\n");
    printf("  => 用途: 累加器等临时缓冲 (Section 05 的 MMA 会用到)\n");

    print_separator("小结");
    printf("  - local_tile(T, tile, coord): 取第 coord 块, 各块 layout 相同\n");
    printf("  - local_partition(T, thr, tid): 分给线程, thr 的 stride 通常取 1\n");
    printf("  - 两者都是 view, 零拷贝\n");
    printf("  - tile -> partition 是 CuTe kernel 的标准骨架\n");
    printf("\n下一步: capstone 用这套工具实现 GEMV\n");

    return 0;
}
