// cute_04 v0 —— bank conflict 与 Swizzle
//
// 对应 README §1 §2 §3。
//
// 本文件只做一件事：把"smem 的摆放方式决定访存效率"这句话变成可以看见的数字。
// 全部在 host 上算 layout —— bank 归属只取决于偏移，不需要真的跑 kernel。

#include <cute/tensor.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// §1  32 个 bank 的模型
// ---------------------------------------------------------------------------
void section1_banks() {
    print_separator("§1  smem 的 32 个 bank");

    printf("smem 按 4 字节为单位轮流分配给 32 个 bank:\n\n");
    printf("  float 下标 :  0   1   2  ...  31 | 32  33 ...\n");
    printf("  bank      :  0   1   2  ...  31 |  0   1 ...\n\n");
    printf("规则: 一个 warp 的 32 个 lane 如果落在 32 个不同 bank -> 一次完成;\n");
    printf("      如果 N 个 lane 落在同一个 bank -> 硬件拆成 N 次, 即 N-way conflict。\n");

    printf("\n一个 32x32 float 的 tile, row-major (32,32):(32,1):\n");
    auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));

    printf("\n  按行读 s(0, 0..7): 偏移 ");
    for (int c = 0; c < 8; ++c) printf("%d ", int(plain(0, c)));
    printf("\n  对应 bank         ");
    for (int c = 0; c < 8; ++c) printf("%d ", bank_of(int(plain(0, c)) * 4));
    printf("   <- 32 个 lane 落在 32 个不同 bank, 无冲突\n");

    printf("\n  按列读 s(0..7, 0): 偏移 ");
    for (int r = 0; r < 8; ++r) printf("%d ", int(plain(r, 0)));
    printf("\n  对应 bank         ");
    for (int r = 0; r < 8; ++r) printf("%d ", bank_of(int(plain(r, 0)) * 4));
    printf("   <- 全部是 bank 0!\n");

    int worst_row = max_bank_requests(32, [&](int l) { return int(plain(0, l)) * 4; });
    int worst_col = max_bank_requests(32, [&](int l) { return int(plain(l, 0)) * 4; });
    printf("\n  一个 warp 读一行: 最热 bank 被请求 %2d 次\n", worst_row);
    printf("  一个 warp 读一列: 最热 bank 被请求 %2d 次   <- 32-way conflict\n", worst_col);

    printf("\n为什么列方向全撞: 行 stride = 32, 而 bank 数也是 32,\n");
    printf("所以下一行的同一列 = 偏移 +32 = 绕回同一个 bank。\n");
}

// ---------------------------------------------------------------------------
// §2  传统解法: padding
// ---------------------------------------------------------------------------
void section2_padding() {
    print_separator("§2  padding: 把行 stride 改成 33");

    auto pad = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<33>{}, Int<1>{}));
    printf("layout = ");
    print(pad);
    printf("\n");

    printf("\n  按列读 s(0..7, 0): 偏移 ");
    for (int r = 0; r < 8; ++r) printf("%d ", int(pad(r, 0)));
    printf("\n  对应 bank         ");
    for (int r = 0; r < 8; ++r) printf("%d ", bank_of(int(pad(r, 0)) * 4));
    printf("   <- 每行错开 1 个 bank\n");

    int worst = max_bank_requests(32, [&](int l) { return int(pad(l, 0)) * 4; });
    printf("\n  一个 warp 读一列: 最热 bank 被请求 %d 次  -> 无冲突\n", worst);

    printf("\n代价:\n");
    printf("  1) 浪费 smem: size = %d, 但要占 cosize = %d 个 float (+%.1f%%)\n",
           int(size(pad)), int(cosize(pad)), 100.0 * (cosize(pad) - size(pad)) / size(pad));
    printf("  2) 行方向不再是 32 的整数倍 -> 破坏 128B 对齐, 宽向量指令用不了\n");
    printf("  3) TMA 和 WGMMA 不接受这种 layout (见 §4 和 Section 05)\n");
}

// ---------------------------------------------------------------------------
// §3  Swizzle: 对偏移的比特做异或
// ---------------------------------------------------------------------------
void section3_swizzle() {
    print_separator("§3  Swizzle<B,M,S>");

    auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto swz = composition(Swizzle<5, 0, 5>{}, plain);

    printf("plain    = ");
    print(plain);
    printf("\n");
    printf("swizzled = ");
    print(swz);
    printf("\n");

    printf("\nSwizzle<B,M,S> 不改 shape, 也不改 size —— 它只改\"逻辑坐标 -> 偏移\"这个映射,\n");
    printf("做法是把偏移的某几个比特异或到另几个比特上。\n");
    printf("  B=5: 参与异或的比特宽度   M=0: 每个 bank 单元的比特偏移   S=5: 异或的距离\n");

    printf("\n  按列读, plain vs swizzled:\n");
    printf("    row |  plain_off  bank |  swz_off  bank\n");
    for (int r = 0; r < 8; ++r) {
        int po = int(plain(r, 0)), so = int(swz(r, 0));
        printf("    %3d | %9d  %4d | %8d  %4d\n", r, po, bank_of(po * 4), so, bank_of(so * 4));
    }

    int wp = max_bank_requests(32, [&](int l) { return int(plain(l, 0)) * 4; });
    int ws = max_bank_requests(32, [&](int l) { return int(swz(l, 0)) * 4; });
    printf("\n  一个 warp 读一列: plain 最热 bank %2d 次, swizzled %d 次\n", wp, ws);

    printf("\n关键: 行方向必须仍然连续, 否则合并访存就没了。\n");
    printf("  按行读 swz(0, 0..7): 偏移 ");
    for (int c = 0; c < 8; ++c) printf("%d ", int(swz(0, c)));
    printf("  <- 仍然连续\n");

    printf("\nsmem 用量: plain cosize = %d, swizzled cosize = %d  <- 一个字节都没多用\n",
           int(cosize(plain)), int(cosize(swz)));

    // 双射检查：swizzle 必须是排列，不能把两个坐标映到同一个偏移
    bool ok = true;
    static int seen[1024];
    for (int i = 0; i < 1024; ++i) seen[i] = 0;
    for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 32; ++c) {
            int o = int(swz(r, c));
            if (o < 0 || o >= 1024 || seen[o]++) ok = false;
        }
    printf("swizzle 是 32x32 上的双射: %s  <- 不丢数据、不重叠\n", ok ? "是" : "否");

    auto pad = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<33>{}, Int<1>{}));
    int wpad = max_bank_requests(32, [&](int l) { return int(pad(l, 0)) * 4; });

    printf("\n三者对比 (ASCII 表头, 便于对齐):\n");
    printf("  %-22s %7s %9s %s\n", "layout", "cosize", "col-conf", "row-contig");
    printf("  %-22s %7d %6d-way %s\n", "plain  (32,32):(32,1)", int(cosize(plain)), wp, "yes");
    printf("  %-22s %7d %6d-way %s\n", "padded (32,32):(33,1)", int(cosize(pad)), wpad,
           "yes (misaligned)");
    printf("  %-22s %7d %6d-way %s\n", "Swizzle<5,0,5>", int(cosize(swz)), ws, "yes");
}

// ---------------------------------------------------------------------------
// §4  GMMA 要求的 swizzle 原子
// ---------------------------------------------------------------------------
void section4_gmma_atoms() {
    print_separator("§4  GMMA::Layout_K_SW*_Atom —— Hopper 指定的 swizzle");

    printf("Section 05 的 WGMMA 不接受任意 smem layout, 只接受这几种 swizzle 原子:\n\n");
    printf("  %-26s %-34s %s\n", "atom (half_t)", "layout", "要求 K 能被整除");
    printf("  %-26s %-34s %s\n", "Layout_K_SW128_Atom", "Sw<3,4,3> o (8,64):(64,1)", "64");
    printf("  %-26s %-34s %s\n", "Layout_K_SW64_Atom", "Sw<2,4,3> o (8,32):(32,1)", "32");
    printf("  %-26s %-34s %s\n", "Layout_K_SW32_Atom", "Sw<1,4,3> o (8,16):(16,1)", "16");
    printf("  %-26s %-34s %s\n", "Layout_K_INTER_Atom", "Sw<0,4,3> o (8,8):(8,1)", "8");

    printf("\n实际打印:\n");
    printf("  K_SW128 = ");
    print(GMMA::Layout_K_SW128_Atom<half_t>{});
    printf("\n");
    printf("  K_SW64  = ");
    print(GMMA::Layout_K_SW64_Atom<half_t>{});
    printf("\n");
    printf("  K_INTER = ");
    print(GMMA::Layout_K_INTER_Atom<half_t>{});
    printf("\n");

    printf("\n用 tile_to_shape 把原子铺到需要的大小:\n");
    auto s64 = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{}, make_shape(Int<64>{}, Int<64>{}));
    printf("  tile_to_shape(K_SW128, (64,64)) = ");
    print(s64);
    printf("\n");

    auto s3 = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                            make_shape(Int<64>{}, Int<64>{}, Int<3>{}));
    printf("  加一个 PIPE=3 的 stage 维                = ");
    print(s3);
    printf("\n");
    printf("  (Section 05/06 的多 stage 就靠这第三维, 这里先见一面)\n");

    printf("\n注意: K 不被原子的 K 长度整除会编译失败, 报\n");
    printf("  \"tile_to_shape: block shape does not divide the target shape\"\n");
    printf("  例如 K_SW128 (K 长 64) 配 K=16 -> 失败, 要用 K_SW32。\n");

    printf("\n这就是 §3 结论的落地: Hopper 不给你选 padding, 只能用 swizzle。\n");
}

int main() {
    printf("cute_04 v0 —— bank conflict 与 Swizzle\n");
    printf("对应 README §1 §2 §3 §4\n");

    section1_banks();
    section2_padding();
    section3_swizzle();
    section4_gmma_atoms();

    printf("\n");
    print_separator("小结");
    printf("  §1  行 stride = 32 时, 按列读 = 32-way bank conflict\n");
    printf("  §2  padding 能解决, 但浪费 smem、破坏对齐、TMA/WGMMA 不接受\n");
    printf("  §3  Swizzle 改映射不改 shape: 零额外 smem, 行方向仍连续, 是双射\n");
    printf("  §4  Hopper 的 WGMMA 只接受 GMMA::Layout_K_SW*_Atom 这几种\n");
    printf("\nv0 OK\n");
    return 0;
}
