// cute_04 v0 —— bank conflict 与 Swizzle 的映射机制
//
// 对应 README §1 §2 §3。
//
// 本文件全部在 host 上跑: bank 归属只取决于偏移, 不需要 GPU。
// 目标是把"Swizzle 到底怎么算偏移"这件事变成可以逐比特看见的东西。
//
// 五个小节:
//   §1  32 个 bank 的模型, 为什么按列读慢 32 倍
//   §2  padding 能修, 代价是什么
//   §3.2 Swizzle<5,0,5> 逐比特手算一遍
//   §3.3 打印整张映射表 (plain vs swizzled 并排)
//   §3.4 M 参数: 连续性和消冲突的权衡  <- 最容易被漏掉的一节

#include <cute/tensor.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 小工具: 把一个整数按二进制打印成 nbits 位
// ---------------------------------------------------------------------------
static void print_bits(int v, int nbits) {
    for (int b = nbits - 1; b >= 0; --b) printf("%d", (v >> b) & 1);
}

// 全坐标扫描: 不只看第 0 列, 而是 32 列都扫一遍取最坏
// —— 只看第 0 列会得出错误结论, 这个函数是 §3.4 那张表的依据
template <class Lay>
static int worst_conflict_all_cols(Lay lay, int elem_bytes) {
    int worst = 0;
    for (int c = 0; c < 32; ++c) {
        int w = max_bank_requests(32, [&](int l) { return int(lay(l, c)) * elem_bytes; });
        if (w > worst) worst = w;
    }
    return worst;
}

template <class Lay>
static int worst_conflict_all_rows(Lay lay, int elem_bytes) {
    int worst = 0;
    for (int r = 0; r < 32; ++r) {
        int w = max_bank_requests(32, [&](int l) { return int(lay(r, l)) * elem_bytes; });
        if (w > worst) worst = w;
    }
    return worst;
}

// 行内最短连续段: 决定能不能用宽向量指令 (128-bit 需要 >= 4 个 float)
template <class Lay>
static int min_contiguous_run(Lay lay) {
    int global_min = 1 << 30;
    for (int r = 0; r < 32; ++r) {
        int run = 1;
        for (int c = 1; c < 32; ++c) {
            if (int(lay(r, c)) == int(lay(r, c - 1)) + 1) {
                ++run;
            } else {
                if (run < global_min) global_min = run;
                run = 1;
            }
        }
        if (run < global_min) global_min = run;
    }
    return global_min;
}

// ---------------------------------------------------------------------------
// §1  32 个 bank 的模型
// ---------------------------------------------------------------------------
static void section1_banks() {
    print_separator("§1  smem 的 32 个 bank: 为什么按列读慢 32 倍");

    printf("smem 按 4 字节轮流分配给 32 个 bank:\n\n");
    printf("  float 下标 :   0    1    2  ...   31 |  32   33  ...\n");
    printf("  bank      :   0    1    2  ...   31 |   0    1  ...\n");
    printf("                └────── 一轮 32 个 ──────┘   └ 绕回来 ┘\n\n");
    printf("规则: 32 个 lane 落在 32 个不同 bank -> 一个周期完成;\n");
    printf("      N 个 lane 落在同一个 bank     -> 拆成 N 次, 即 N-way conflict。\n");

    auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    printf("\n一个 32x32 float tile, row-major: ");
    print(plain);
    printf("\n偏移公式 off(r,c) = r*32 + c\n");

    printf("\n  按行读 s(0, 0..7):\n    偏移 ");
    for (int c = 0; c < 8; ++c) printf("%4d", int(plain(0, c)));
    printf("\n    bank ");
    for (int c = 0; c < 8; ++c) printf("%4d", bank_of(int(plain(0, c)) * 4));
    printf("     <- 32 个 lane 落在 32 个不同 bank\n");

    printf("\n  按列读 s(0..7, 0):\n    偏移 ");
    for (int r = 0; r < 8; ++r) printf("%4d", int(plain(r, 0)));
    printf("\n    bank ");
    for (int r = 0; r < 8; ++r) printf("%4d", bank_of(int(plain(r, 0)) * 4));
    printf("     <- 全部是 bank 0!\n");

    printf("\n  一个 warp 读一行: 最热 bank %2d 次\n",
           max_bank_requests(32, [&](int l) { return int(plain(0, l)) * 4; }));
    printf("  一个 warp 读一列: 最热 bank %2d 次   <- 32-way conflict\n",
           max_bank_requests(32, [&](int l) { return int(plain(l, 0)) * 4; }));

    printf("\n根源: 行 stride = 32, bank 数也 = 32。\n");
    printf("      下一行同一列 = 偏移 +32, 而 32 %% 32 = 0 -> bank 号纹丝不动。\n");
    printf("      转置这个操作必然要按列读 —— 冲突就是这么来的。\n");
}

// ---------------------------------------------------------------------------
// §2  padding: 能修, 但 SM90 上是死路
// ---------------------------------------------------------------------------
static void section2_padding() {
    print_separator("§2  padding: 行 stride 改成 33");

    auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto pad = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<33>{}, Int<1>{}));

    printf("plain = ");
    print(plain);
    printf("\npad   = ");
    print(pad);
    printf("      <- 只改了 stride 的第一个数\n");

    printf("\n  按列读 s(0..7, 0):\n    偏移 ");
    for (int r = 0; r < 8; ++r) printf("%4d", int(pad(r, 0)));
    printf("\n    bank ");
    for (int r = 0; r < 8; ++r) printf("%4d", bank_of(int(pad(r, 0)) * 4));
    printf("     <- 每行错开 1 个 bank (33 %% 32 = 1)\n");

    printf("\n  一个 warp 读一列: %d-way -> %d-way\n",
           max_bank_requests(32, [&](int l) { return int(plain(l, 0)) * 4; }),
           max_bank_requests(32, [&](int l) { return int(pad(l, 0)) * 4; }));

    printf("\n代价三条:\n");
    printf("  1) 多占 smem: size = %d, 但 cosize = %d (+%.1f%%)\n", int(size(pad)),
           int(cosize(pad)), 100.0 * (cosize(pad) - size(pad)) / size(pad));
    printf("  2) 行首不再 128B 对齐\n");
    printf("  3) SM90 的 WGMMA 编译期拒绝这种 layout  <- 这条是硬墙\n");
    printf("\n     实测 (见 v2): padded layout 配 SM90_64x64x16_F32F16F16_SS 会编译失败,\n");
    printf("     报 \"Not a canonical GMMA_K Layout\"。注意是编译期, 不是算错。\n");
    printf("     所以 SM90 上必须用 Swizzle。\n");
}

// ---------------------------------------------------------------------------
// §3.2  Swizzle<B,M,S> 逐比特手算
// ---------------------------------------------------------------------------
static void section3_bits() {
    print_separator("§3.2  Swizzle<5,0,5> 逐比特手算");

    printf("32x32 float tile 的偏移是 10 位 (0..1023):\n\n");
    printf("     bit:   9   8   7   6   5 |  4   3   2   1   0\n");
    printf("            \\____ r 的 5 位 ___/  \\____ c 的 5 位 ___/\n");
    printf("            (off = r*32 + c, 所以高 5 位是 r, 低 5 位是 c)\n\n");

    printf("Swizzle<B,M,S> 的三个参数:\n");
    printf("  B = 参与异或的比特数     (这里 5)\n");
    printf("  M = 最低几位不动         (这里 0, 即不保护任何连续性)\n");
    printf("  S = 异或的距离           (这里 5, 让高 5 位去改低 5 位)\n\n");
    printf("映射公式:\n");
    printf("  swz_off = plain_off XOR ( ((plain_off >> S) & mask_B) << M )\n\n");

    auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto swz = composition(Swizzle<5, 0, 5>{}, plain);

    printf("拿 (r,c) = (3,5) 走一遍, plain_off = 3*32 + 5 = 101:\n\n");
    int r = 3, c = 5, po = r * 32 + c;
    printf("  plain_off = %3d = 0b", po);
    print_bits(po, 10);
    printf("\n                      ");
    printf("\\__r=%d__/\\__c=%d__/\n", r, c);
    printf("  第 1 步: 取出高 5 位 (r)      = 0b");
    print_bits((po >> 5) & 31, 5);
    printf(" = %d\n", (po >> 5) & 31);
    printf("  第 2 步: M=0, 不左移          = %d\n", (po >> 5) & 31);
    printf("  第 3 步: 和低 5 位异或  c^r   = 0b");
    print_bits(po & 31, 5);
    printf(" XOR 0b");
    print_bits((po >> 5) & 31, 5);
    printf(" = 0b");
    print_bits((po & 31) ^ ((po >> 5) & 31), 5);
    printf(" = %d\n", (po & 31) ^ ((po >> 5) & 31));
    printf("  第 4 步: 拼回去               = %d*32 + %d = %d\n", r,
           (po & 31) ^ ((po >> 5) & 31), r * 32 + ((po & 31) ^ ((po >> 5) & 31)));
    printf("\n  CuTe 实际算出来               = %d   %s\n", int(swz(r, c)),
           int(swz(r, c)) == r * 32 + ((po & 31) ^ ((po >> 5) & 31)) ? "<- 一致" : "<- 不一致!");

    printf("\n一句话: 用行号去打乱列号, c_new = c XOR r。\n");

    printf("\n多验几个 (公式 vs CuTe):\n");
    printf("  (r, c)   plain_off  二进制      r_bits  c_bits  c^r   手算  CuTe\n");
    int rs[] = {0, 1, 2, 3, 5, 7, 6}, cs[] = {0, 1, 3, 5, 6, 9, 3};
    for (int i = 0; i < 7; ++i) {
        int rr = rs[i], cc = cs[i], p = rr * 32 + cc;
        int xr = (p >> 5) & 31, xc = p & 31, x = xc ^ xr;
        printf("  (%d, %d)   %8d  0b", rr, cc, p);
        print_bits(p, 10);
        printf("  %5d  %5d  %4d  %5d  %4d\n", xr, xc, x, rr * 32 + x, int(swz(rr, cc)));
    }
}

// ---------------------------------------------------------------------------
// §3.3  整张映射表: plain vs swizzled 并排
// ---------------------------------------------------------------------------
static void section3_table() {
    print_separator("§3.3  整张映射表 (理解 swizzle 最快的方式)");

    auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto swz = composition(Swizzle<5, 0, 5>{}, plain);

    printf("左边 plain, 右边 Swizzle<5,0,5>, 前 8 行 x 8 列:\n\n");
    printf("        plain 偏移                    |        swizzled 偏移\n");
    printf("     c= 0   1   2   3   4   5   6   7 |     c= 0   1   2   3   4   5   6   7\n");
    for (int r = 0; r < 8; ++r) {
        printf(" r=%d ", r);
        for (int c = 0; c < 8; ++c) printf("%4d", int(plain(r, c)));
        printf("  | r=%d ", r);
        for (int c = 0; c < 8; ++c) printf("%4d", int(swz(r, c)));
        if (r == 0) printf("   <- r=0: XOR 0, 不变");
        if (r == 1) printf("   <- 两两交换");
        if (r == 2) printf("   <- 每 2 个一组交换");
        if (r == 3) printf("   <- 每 4 个一组倒转");
        printf("\n");
    }

    printf("\n竖着读第 0 列, 这是关键:\n");
    printf("  r        :");
    for (int r = 0; r < 8; ++r) printf("%5d", r);
    printf("\n  plain    :");
    for (int r = 0; r < 8; ++r) printf("%5d", int(plain(r, 0)));
    printf("\n  -> bank  :");
    for (int r = 0; r < 8; ++r) printf("%5d", bank_of(int(plain(r, 0)) * 4));
    printf("   <- 全 0, 32-way conflict\n");
    printf("  swizzled :");
    for (int r = 0; r < 8; ++r) printf("%5d", int(swz(r, 0)));
    printf("\n  -> bank  :");
    for (int r = 0; r < 8; ++r) printf("%5d", bank_of(int(swz(r, 0)) * 4));
    printf("   <- 0..7 各不相同, 无冲突\n");

    printf("\n注意: swizzled 的列偏移 0, 33, 66, 99 ... 和 padding 一模一样,\n");
    printf("      但 cosize 还是 %d (padding 要 1055)。\n", int(cosize(swz)));
    printf("      padding 靠多占空间把行推开, swizzle 靠原地重排达到同样效果。\n");

    // 三个不变量
    printf("\n三个不变量:\n");
    printf("  1) cosize 不变: plain %d, swizzled %d\n", int(cosize(plain)), int(cosize(swz)));

    static int seen[1024];
    for (int i = 0; i < 1024; ++i) seen[i] = 0;
    bool bij = true;
    for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 32; ++c) {
            int o = int(swz(r, c));
            if (o < 0 || o >= 1024 || seen[o]++) bij = false;
        }
    printf("  2) 是双射: %s (扫全 1024 个坐标, 无重复无越界)\n", bij ? "是" : "否");
    printf("  3) 行读仍无冲突: 全 32 行扫描, 最坏 %d-way\n", worst_conflict_all_rows(swz, 4));

    printf("\n第 3 条要说清楚: \"行读无冲突\" != \"行内偏移连续\"。\n");
    printf("  swz 第 1 行的偏移是 ");
    for (int c = 0; c < 6; ++c) printf("%d ", int(swz(1, c)));
    printf("... 不连续 (所以不能向量化),\n");
    printf("  但 32 个 lane 仍落在 32 个不同 bank (所以不冲突)。下一节就讲这个区别。\n");
}

// ---------------------------------------------------------------------------
// §3.4  M 参数: 连续性和消冲突的权衡  <- 本文件最重要的一节
// ---------------------------------------------------------------------------
static void section3_m_param() {
    print_separator("§3.4  M 参数: 保住向量化 vs 消掉冲突");

    printf("M 保护最低 M 位不参与异或, 即 2^M 个相邻元素保持连续。\n");
    printf("这直接决定了「还能不能用宽向量指令搬它」。\n\n");

    auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto pad = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<33>{}, Int<1>{}));

    printf("  %-14s %10s %8s %14s %s\n", "layout", "列读最坏", "cosize", "行内最短连续", "128-bit atom");
    auto row = [&](const char* tag, auto lay, const char* vec) {
        printf("  %-14s %7d-way %8d %11d 个 float  %s\n", tag, worst_conflict_all_cols(lay, 4),
               int(cosize(lay)), min_contiguous_run(lay), vec);
    };
    row("plain", plain, "可用");
    row("pad 33", pad, "可用 (但 WGMMA 拒绝)");
    row("Sw<5,0,5>", composition(Swizzle<5, 0, 5>{}, plain), "编译失败");
    row("Sw<4,1,4>", composition(Swizzle<4, 1, 4>{}, plain), "编译失败");
    row("Sw<3,2,3>", composition(Swizzle<3, 2, 3>{}, plain), "可用");

    printf("\n注意上表的\"列读最坏\"是扫过全部 32 列取最坏, 不是只看第 0 列 ——\n");
    printf("只看第 0 列会把 Sw<3,2,3> 误判成 1-way。\n");

    printf("\n看 M 怎么起作用, 打印每行前 8 个偏移:\n");
    auto show_run = [&](const char* tag, auto lay) {
        printf("  %-12s r=1: ", tag);
        for (int c = 0; c < 8; ++c) printf("%4d", int(lay(1, c)));
        printf("   最短连续段 = %d\n", min_contiguous_run(lay));
    };
    show_run("Sw<5,0,5>", composition(Swizzle<5, 0, 5>{}, plain));
    show_run("Sw<4,1,4>", composition(Swizzle<4, 1, 4>{}, plain));
    show_run("Sw<3,2,3>", composition(Swizzle<3, 2, 3>{}, plain));
    printf("       M=0 -> 每个元素单独打乱; M=1 -> 2 个一组; M=2 -> 4 个一组\n");

    printf("\nSw<5,0,5> 配 128-bit atom 的实际报错 (已实测):\n");
    printf("  static assertion failed:\n");
    printf("  \"Copy_Traits: dst failed to vectorize into registers.\n");
    printf("   Layout is incompatible with this CopyOp.\"\n");

    printf("\n选参数的规则:\n");
    printf("  先定 M = 你要的向量宽度   (128-bit float -> 4 个 -> M=2)\n");
    printf("                            (16B half      -> 8 个 -> M=3)\n");
    printf("  再让 B, S 去消冲突。\n");
    printf("  **先保住向量化, 再谈消冲突** —— 这就是 GMMA 官方原子全是 M=4 的原因 (见 v2)。\n");
}

int main() {
    printf("cute_04 v0 —— bank conflict 与 Swizzle 的映射机制\n");
    printf("对应 README §1 §2 §3\n");

    section1_banks();
    section2_padding();
    section3_bits();
    section3_table();
    section3_m_param();

    print_separator("小结");
    printf("  §1   行 stride = 32 时按列读 = 32-way conflict, 转置必然撞上\n");
    printf("  §2   padding 能修, 但 SM90 的 WGMMA 编译期就拒绝它\n");
    printf("  §3.2 Swizzle<5,0,5> 就是 c_new = c XOR r, 四步可以手算\n");
    printf("  §3.3 cosize 不变 / 是双射 / 行读仍无冲突\n");
    printf("  §3.4 M 决定 2^M 个元素保持连续 -> 决定能否向量化, 这是真正的权衡点\n");
    printf("\n下一步: v1 看搬运代码要不要改, v2 看 SM90 的 TMA 和 WGMMA。\n");
    printf("\nv0 OK\n");
    return 0;
}
