// v3: local_partition 拆开看 —— thr_layout 的 shape 和 stride 各管什么
//
// v2 只演示了"thr_layout 写错会静默出错"。本文件把 local_partition
// 拆成它真正的两个组成部分，解释为什么会错、错在哪一步。
//
// 源码 (cutlass/include/cute/tensor_impl.hpp:1084):
//
//   local_partition(tensor, thr, tid) {
//     return outer_partition(tensor,
//                            product_each(shape(thr)),   // <- 只用 shape
//                            thr.get_flat_coord(tid));   // <- 只用来反解坐标
//   }
//
// 两个参数来自 thr_layout 的不同部分，作用完全不同：
//   product_each(shape(thr))  决定"切成多少片、每片多大"  —— 只看 shape
//   thr.get_flat_coord(tid)   决定"tid 该拿第几片"        —— shape+stride 都看
//
// 所以 stride 不参与切分，只参与"tid -> 片号"这个查表。stride 一旦不是 1，
// 这个查表就不再是恒等映射，于是出现多个 tid 撞进同一片。

#include <cute/tensor.hpp>

#include "common.h"

using namespace cute;

static float g_data[32];

// 打印一个 rank-1 thr_layout 下的完整分配情况
template <class Thr>
static void show_1d(const char* tag, Thr const& thr) {
    auto T = make_tensor(make_gmem_ptr(g_data), make_layout(Int<32>{}, Int<1>{}));

    printf("\n%s  thr_layout = ", tag);
    print(thr);
    printf("   size=%d cosize=%d\n", int(size(thr)), int(cosize(thr)));

    printf("  第一步 tiler = product_each(shape) = ");
    print(product_each(shape(thr)));
    printf("      (stride 在这一步被丢掉)\n");

    printf("  第二步 tid -> get_flat_coord(tid) -> 片号:\n    ");
    for (int t = 0; t < size(thr); ++t) {
        printf("tid%d->", t);
        print(thr.get_flat_coord(t));
        printf("  ");
    }
    printf("\n");

    printf("  实际分配:\n");
    for (int t = 0; t < size(thr); ++t) {
        auto p = local_partition(T, thr, t);
        printf("    tid%d: ", t);
        for (int i = 0; i < size(p); ++i) printf("%2.0f ", float(p(i)));
        printf("\n");
    }
}

int main() {
    for (int i = 0; i < 32; ++i) g_data[i] = float(i);

    print_separator("local_partition = tiler(只看 shape) + 片号查表(看 stride)");

    // ================================================================
    // 1. local_partition 等价于 zipped_divide + 切片
    // ================================================================
    print_separator("1. 底层就是 zipped_divide 后取一片");

    auto T = make_tensor(make_gmem_ptr(g_data), make_layout(Int<32>{}, Int<1>{}));
    printf("T = 32 个元素, 值 0-31\n");

    auto zd = zipped_divide(T, make_shape(Int<4>{}));
    printf("\nzipped_divide(T, (4)) = ");
    print(zd.layout());
    printf("\n  mode0 = 片内 (4 片), mode1 = 每片的 8 个元素\n");
    for (int t = 0; t < 4; ++t) {
        printf("  zd(%d,_) = ", t);
        auto s = zd(t, _);
        for (int i = 0; i < size(s); ++i) printf("%2.0f ", float(s(i)));
        printf("\n");
    }

    auto lp1 = local_partition(T, make_layout(Int<4>{}, Int<1>{}), 1);
    printf("\nlocal_partition(T, 4:1, tid=1) = ");
    for (int i = 0; i < size(lp1); ++i) printf("%2.0f ", float(lp1(i)));
    printf("\n  => 和 zd(1,_) 完全一样。local_partition 就是这两步的封装\n");

    // ================================================================
    // 2. 无洞: stride=1，tid 直接就是片号
    // ================================================================
    print_separator("2. 无洞的 thr_layout: stride=1");

    show_1d("[A]", make_layout(Int<4>{}, Int<1>{}));
    printf("  分析: 4:1 的 image = {0,1,2,3}，覆盖 [0,4) 无洞。\n");
    printf("        get_flat_coord 是恒等映射, tid == 片号, 4 个线程各拿一片。\n");
    printf("        32 个元素不重不漏。这是唯一正确的用法。\n");

    show_1d("[B]", make_layout(Int<8>{}, Int<1>{}));
    printf("  分析: 换成 8 个线程, tiler 变 8, 每片 4 个元素。同样正确。\n");
    printf("        => shape 决定分几片, 线程数就是 size(thr)。\n");

    // ================================================================
    // 3. 有洞: stride>1，image 出现空隙
    // ================================================================
    print_separator("3. 有洞的 thr_layout: stride=2, image={0,2,4,6}");

    show_1d("[C]", make_layout(Int<4>{}, Int<2>{}));
    printf("  分析: 4:2 的 image = {0,2,4,6}，奇数位置是洞。\n");
    printf("        tiler 仍是 4 (shape 没变), 所以还是切 4 片、每片 8 个。\n");
    printf("        但 get_flat_coord 要满足 crd2idx(结果) == 输入:\n");
    printf("          tid=0 -> 0*2=0 ✓ 片号 0\n");
    printf("          tid=1 -> 落在洞里(1 不是 2 的倍数), 向下取整 -> 片号 0\n");
    printf("          tid=2 -> 1*2=2 ✓ 片号 1\n");
    printf("          tid=3 -> 洞 -> 片号 1\n");
    printf("        结果: tid0/tid1 撞车, tid2/tid3 撞车。片 2、片 3 没人要。\n");
    printf("        => 16 个元素被重复读, 16 个元素没人碰。静默错误。\n");

    show_1d("[D]", make_layout(Int<4>{}, Int<8>{}));
    printf("  分析: 洞更大, image={0,8,16,24}。tid 0-3 全部落进第一个洞区间,\n");
    printf("        4 个线程拿到完全相同的数据。\n");

    show_1d("[E]", make_layout(Int<8>{}, Int<4>{}));
    printf("  分析: 这就是 exercises/ex.cu 练习 4 的第 (2) 题。\n");
    printf("        image={0,4,8,...,28}, 每 4 个连续 tid 挤进同一片。\n");
    printf("        tid0-3 -> 片0, tid4-7 -> 片1。只用到 2 片, 剩 6 片闲置。\n");

    // ================================================================
    // 4. 2D: 同样的规律，外加 stride 决定"谁挨着谁"
    // ================================================================
    print_separator("4. 2D thr_layout: stride 还决定线程的排列顺序");

    auto T2 = make_tensor(make_gmem_ptr(g_data),
                          make_layout(make_shape(Int<4>{}, Int<8>{}),
                                      make_stride(Int<8>{}, Int<1>{})));
    printf("T2 = 4x8 row-major, 值 0-31\n");

    {
        auto thr = make_layout(make_shape(Int<2>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
        printf("\n[F] thr = (2,4):(4,1)  行主序, 8 个线程\n");
        printf("  tiler = ");
        print(product_each(shape(thr)));
        printf("  => 把 4x8 切成 2x4 的网格, 每片 2x2\n");
        for (int t = 0; t < 8; ++t) {
            auto p = local_partition(T2, thr, t);
            printf("    tid%d 片号=", t);
            print(thr.get_flat_coord(t));
            printf(" -> ");
            for (int i = 0; i < size(p); ++i) printf("%2.0f ", float(p(i)));
            printf("\n");
        }
        printf("  分析: stride=(4,1) 无洞, 8 个 tid 一对一映射到 8 个片号。\n");
        printf("        tid 递增时先走列(mode1), 因为 mode1 的 stride 是 1。\n");
    }

    {
        auto thr = make_layout(make_shape(Int<2>{}, Int<4>{}), make_stride(Int<1>{}, Int<2>{}));
        printf("\n[G] thr = (2,4):(1,2)  列主序, 同样 8 个线程、同样无洞\n");
        for (int t = 0; t < 8; ++t) {
            auto p = local_partition(T2, thr, t);
            printf("    tid%d 片号=", t);
            print(thr.get_flat_coord(t));
            printf(" -> ");
            for (int i = 0; i < size(p); ++i) printf("%2.0f ", float(p(i)));
            printf("\n");
        }
        printf("  分析: 和 [F] 切法完全相同(tiler 一样), 覆盖也不重不漏。\n");
        printf("        差别只在 tid 到片的对应关系: 这里先走行(mode0)。\n");
        printf("        => 无洞时 stride 只影响 '谁拿哪片', 不影响正确性。\n");
        printf("        这决定了访存合并: 相邻 tid 是否读相邻地址。\n");
    }

    {
        auto thr = make_layout(make_shape(Int<2>{}, Int<4>{}), make_stride(Int<8>{}, Int<1>{}));
        printf("\n[H] thr = (2,4):(8,1)  mode0 有洞 (cosize=%d > size=%d)\n",
               int(cosize(thr)), int(size(thr)));
        for (int t = 0; t < 8; ++t) {
            auto p = local_partition(T2, thr, t);
            printf("    tid%d 片号=", t);
            print(thr.get_flat_coord(t));
            printf(" -> ");
            for (int i = 0; i < size(p); ++i) printf("%2.0f ", float(p(i)));
            printf("\n");
        }
        printf("  分析: mode0 stride=8 但 mode1 只占 4 个位置, 4-7 是洞。\n");
        printf("        tid4-7 本该映射到 mode0=1, 却落进洞里退回 mode0=0。\n");
        printf("        => tid4-7 和 tid0-3 完全重复, 下半个矩阵没人处理。\n");
    }

    // ================================================================
    // 5. 判据
    // ================================================================
    print_separator("5. 怎么判断 thr_layout 写对了");

    printf("充要条件: size(thr) == cosize(thr)\n");
    printf("  含义 = thr 的 image 恰好铺满 [0, size)，没有洞。\n");
    printf("  等价 = get_flat_coord 在 [0, size) 上是双射, tid 一对一拿到片号。\n\n");

    struct { const char* name; int s, c; } tbl[] = {
        {"4:1",         4,  4},
        {"8:1",         8,  8},
        {"4:2",         4,  7},
        {"4:8",         4, 25},
        {"8:4",         8, 29},
        {"(2,4):(4,1)", 8,  8},
        {"(2,4):(1,2)", 8,  8},
        {"(2,4):(8,1)", 8, 12},
    };
    printf("  %-14s %6s %8s   %s\n", "thr_layout", "size", "cosize", "判定");
    for (auto& e : tbl) {
        printf("  %-14s %6d %8d   %s\n", e.name, e.s, e.c,
               e.s == e.c ? "OK" : "有洞 -> 线程重复, 数据漏算");
    }

    printf("\n实用规则:\n");
    printf("  1) thr_layout 用 make_layout(shape) 让 CuTe 自动生成紧凑 stride,\n");
    printf("     绝不手写 stride —— 手写就有踩洞的风险。\n");
    printf("  2) 想改变线程的排列顺序(为了访存合并), 改 shape 的维度顺序\n");
    printf("     或用 LayoutLeft/LayoutRight, 不要靠调 stride。\n");
    printf("  3) 出错时没有任何报错, 只能靠 size==cosize 自查, 或者\n");
    printf("     像本文件这样把每个 tid 的数据打出来看。\n");

    printf("\n注意: stride=0 的 thr_layout (如 (2,4):(1,0)) 会让 get_flat_coord\n");
    printf("  在运行时崩溃 —— 实测 signal 8。stride-0 表示广播, 无法反解出\n");
    printf("  唯一片号。不要把 stride-0 的 layout 交给 local_partition。\n");

    return 0;
}
