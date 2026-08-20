// cute_04 v3 —— 搬进 smem 之后怎么摆: bank conflict 与 swizzle
//
// 对应 README §4。
//
// v1/v2 已经会用 TMA 把一块搬进 smem、再搬出去。但一直没管过一件事:
// **搬进去之后在 smem 里怎么摆**。这一版回答它。
//
//   §4.1  32 个 bank: 为什么按列读慢 32 倍     <- 纯 host 计算, 不需要 GPU
//   §4.2  padding 能修, 但在 SM90 上是死路      <- 为什么不能用这条捷径
//   §4.3  swizzle 怎么映射                     <- 逐比特手算 + 整张映射表
//   §4.4  M 参数: 消冲突和向量化的权衡          <- 最容易选错的参数
//   §4.5  交给 TMA 时只有四种模式可选           <- descriptor 只有几个比特存摆法
//   §4.6  改 host 一行, kernel 不动             <- 落点: swizzle 藏在 layout 里
//
// ---------------------------------------------------------------------------
// 为什么在这一章讲 swizzle
//
// 因为 **smem 的摆法是写进 TMA descriptor 的**。host 侧那行
// make_tma_copy(SM90_TMA_LOAD{}, mIn, slay) 里的 slay 决定了硬件搬进来之后
// 怎么摆; kernel 里一个字都不用改。
//
// 而"该摆成什么样"由**下游**决定 —— 谁读这块 smem, 就按谁的口味摆。
// 这一章的下游是"按列读的线程"(§4.1); cute_05 之后的下游是 WGMMA。
//
// ---------------------------------------------------------------------------
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_v3

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 尺寸: 和 v1/v2 一样
// ---------------------------------------------------------------------------
constexpr int M = 256, N = 128;
constexpr int CM = 32, CN = 32;
constexpr int NTHR = 128;

static dim3 grid() { return dim3(M / CM, N / CN); }

struct Buffers {
    static constexpr size_t elems = size_t(M) * N;
    static constexpr size_t bytes = elems * sizeof(float);

    float* d_in;
    float* d_out;
    float* h_in;
    float* h_out;

    Buffers() {
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        h_in = new float[elems];
        h_out = new float[elems];
        for (size_t i = 0; i < elems; ++i) h_in[i] = float(i);
        CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    }

    ~Buffers() {
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        delete[] h_in;
        delete[] h_out;
    }

    bool check() {
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < elems; ++i)
            if (h_out[i] != h_in[i]) return false;
        return true;
    }
};

// ---------------------------------------------------------------------------
// 分析工具 (纯 host): 全列/全行扫描取最坏, 以及行内最短连续段
// 只看第 0 列会得出错误结论, 所以一律扫全部
// ---------------------------------------------------------------------------
template <class Lay>
static int worst_col_conflict(Lay lay) {
    int worst = 0;
    for (int c = 0; c < CN; ++c) {
        int w = max_bank_requests(32, [&](int l) { return int(lay(l, c)) * 4; });
        if (w > worst) worst = w;
    }
    return worst;
}

template <class Lay>
static int worst_row_conflict(Lay lay) {
    int worst = 0;
    for (int r = 0; r < CM; ++r) {
        int w = max_bank_requests(32, [&](int l) { return int(lay(r, l)) * 4; });
        if (w > worst) worst = w;
    }
    return worst;
}

// 行内最短连续段: 决定能不能用宽向量指令 (128-bit 需要连续 4 个 float)
template <class Lay>
static int min_contiguous_run(Lay lay) {
    int g = 1 << 30;
    for (int r = 0; r < CM; ++r) {
        int run = 1;
        for (int c = 1; c < CN; ++c) {
            if (int(lay(r, c)) == int(lay(r, c - 1)) + 1) {
                ++run;
            } else {
                if (run < g) g = run;
                run = 1;
            }
        }
        if (run < g) g = run;
    }
    return g;
}

// ===========================================================================
// §4.1  32 个 bank: 为什么按列读慢 32 倍
// ===========================================================================
static void section41_banks() {
    print_separator("§4.1  32 个 bank: 为什么按列读慢 32 倍");

    printf("  smem 硬件按 4 字节轮流分给 32 个 bank:\n\n");
    printf("    float 下标 :   0    1    2  ...   31 |  32   33  ...\n");
    printf("    bank      :   0    1    2  ...   31 |   0    1  ...\n");
    printf("                  +------ 一轮 32 个 ------+   +- 绕回来 -+\n\n");
    printf("  规则只有一条:\n");
    printf("    32 个 lane 落在 32 个不同 bank -> 一个周期完成\n");
    printf("    N 个 lane 落在同一个 bank      -> 硬件拆成 N 次, 即 N-way conflict\n\n");

    auto plain = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    printf("  v1/v2 一直在用的 smem layout: ");
    print(plain);
    printf("\n  偏移公式 off(r,c) = r*%d + c\n\n", CN);

    printf("    按行读 sT(0, 0..7):\n      偏移 ");
    for (int c = 0; c < 8; ++c) printf("%4d", int(plain(0, c)));
    printf("\n      bank ");
    for (int c = 0; c < 8; ++c) printf("%4d", bank_of(int(plain(0, c)) * 4));
    printf("     <- 32 个 lane 落 32 个不同 bank\n");

    printf("\n    按列读 sT(0..7, 0):\n      偏移 ");
    for (int r = 0; r < 8; ++r) printf("%4d", int(plain(r, 0)));
    printf("\n      bank ");
    for (int r = 0; r < 8; ++r) printf("%4d", bank_of(int(plain(r, 0)) * 4));
    printf("     <- 全部撞 bank 0!\n");

    printf("\n    一个 warp 读一行: 最热 bank %2d 次\n", worst_row_conflict(plain));
    printf("    一个 warp 读一列: 最热 bank %2d 次   <- %d-way conflict\n",
           worst_col_conflict(plain), worst_col_conflict(plain));

    printf("\n  根源: 行 stride = %d, bank 数也 = 32。下一行的同一列 = 偏移 +%d,\n", CN, CN);
    printf("  而 %d %% 32 = 0 -> bank 号纹丝不动。\n", CN);
    printf("\n  什么时候会按列读? 转置、reduction 沿列方向、以及某些 MMA 的取数模式。\n");
    printf("  只要 consumer 会按列读, 这个 32-way 就是实打实的 32 倍代价。\n");
}

// ===========================================================================
// §4.2  padding: 能修, 但在 SM90 上是死路
// ===========================================================================
static void section42_padding() {
    print_separator("§4.2  padding: 能修, 但在 SM90 上是死路");

    auto plain = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto pad = make_layout(make_shape(Int<CM>{}, Int<CN>{}),
                           make_stride(Int<CN + 1>{}, Int<1>{}));  // 只改行 stride

    printf("  最直观的修法: 行 stride 从 %d 改成 %d, 每行多占一个 float。\n\n", CN, CN + 1);
    printf("    plain = ");
    print(plain);
    printf("\n    pad   = ");
    print(pad);
    printf("\n                                    ^^^ 只改了这一个数\n\n");

    printf("    列方向偏移 (前 8 行):\n      plain ");
    for (int r = 0; r < 8; ++r) printf("%5d", int(plain(r, 0)));
    printf("\n      bank  ");
    for (int r = 0; r < 8; ++r) printf("%5d", bank_of(int(plain(r, 0)) * 4));
    printf("\n      pad   ");
    for (int r = 0; r < 8; ++r) printf("%5d", int(pad(r, 0)));
    printf("\n      bank  ");
    for (int r = 0; r < 8; ++r) printf("%5d", bank_of(int(pad(r, 0)) * 4));
    printf("   <- 每行错开 1 个 bank\n");

    printf("\n    列读冲突   plain %2d-way   pad %2d-way\n", worst_col_conflict(plain),
           worst_col_conflict(pad));
    printf("    smem 占用  plain size=%d cosize=%d   pad size=%d cosize=%d (+%.1f%%)\n",
           int(size(plain)), int(cosize(plain)), int(size(pad)), int(cosize(pad)),
           100.0 * (int(cosize(pad)) - int(cosize(plain))) / int(cosize(plain)));

    printf("\n  冲突确实消掉了。但 padding 有三条代价:\n\n");
    printf("    1) 多占 smem      cosize %d > size %d —— 开数组要按 cosize\n", int(cosize(pad)),
           int(size(pad)));
    printf("    2) 破坏对齐      行首不再 128B 对齐, 而 TMA 要求 128B 对齐\n");
    printf("    3) WGMMA 拒绝    SM90 的 Tensor Core **编译期**就不接受这种 layout\n");

    printf("\n  第三条是硬墙。SM90 的 WGMMA 不经过寄存器, 而是把 smem 地址和摆法\n");
    printf("  编码成一个 descriptor 让硬件直读 —— 那个 descriptor 里只有几个比特\n");
    printf("  存摆法, 能表达的只有 §4.6 那四种规范形式, padding 不在其中。\n");
    printf("  真去编译会得到:\n");
    printf("      static assertion failed:\n");
    printf("      \"Not a canonical GMMA_K Layout: Expected stride failure.\"\n");
    printf("  (这是 cute_05 讲 WGMMA 时会亲手撞的墙, 这里先记住结论。)\n");

    printf("\n  所以在 SM90 上, 消 bank conflict 只有一条路: **swizzle**。\n");
}

// ===========================================================================
// §4.3  swizzle 怎么映射
//
// Swizzle<B,M,S> 不改 shape、不多占一个字节, 只改"逻辑坐标 -> 偏移"这个
// 映射函数, 做法是把偏移的某几个比特异或到另几个比特上。
// ===========================================================================
static void print_bits(int v, int nbits) {
    for (int b = nbits - 1; b >= 0; --b) printf("%d", (v >> b) & 1);
}

static void section43_swizzle_mapping() {
    print_separator("§4.3  swizzle 怎么映射");

    auto plain = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto swz = composition(Swizzle<5, 0, 5>{}, plain);

    printf("  三个参数的含义 (以 %dx%d float tile 为例, 偏移 0..%d 共 10 位):\n\n", CM, CN,
           CM * CN - 1);
    printf("      bit:   9   8   7   6   5 |  4   3   2   1   0\n");
    printf("             +---- r 的 5 位 ---+  +---- c 的 5 位 ---+\n");
    printf("             (off = r*32 + c, 所以高 5 位是 r, 低 5 位是 c)\n\n");
    printf("      B = 参与异或的比特数\n");
    printf("      M = 最低几位不动 (保护 2^M 个元素保持连续)\n");
    printf("      S = 异或的距离 (源比特段和目标比特段相隔几位)\n\n");
    printf("  映射公式:\n");
    printf("      swz(off) = off XOR ( ((off >> S) & mask_B) << M )\n\n");

    // 逐比特手算一个例子
    const int r = 3, c = 5;
    int off = r * CN + c;
    int hi = (off >> 5) & 0x1F;
    int res = off ^ (hi << 0);
    printf("  Swizzle<5,0,5> 手算 (r=%d, c=%d):\n\n", r, c);
    printf("      plain off = %d*%d + %d = %-4d = 0b", r, CN, c, off);
    print_bits(off, 10);
    printf("\n                                        +-r=%d-++-c=%d-+\n", r, c);
    printf("      第 1 步  取出高 5 位 (r)   = 0b");
    print_bits(hi, 5);
    printf(" = %d\n", hi);
    printf("      第 2 步  M=0, 不左移        = %d\n", hi);
    printf("      第 3 步  和低 5 位异或      c XOR r = 0b");
    print_bits(c, 5);
    printf(" XOR 0b");
    print_bits(r, 5);
    printf(" = 0b");
    print_bits(c ^ r, 5);
    printf(" = %d\n", c ^ r);
    printf("      第 4 步  拼回去             = %d*%d + %d = %d\n", r, CN, c ^ r, res);
    printf("\n      CuTe 实算 swz(%d,%d) = %d   -> %s\n", r, c, int(swz(r, c)),
           int(swz(r, c)) == res ? "一致" : "不一致");
    printf("\n  一句话: **用行号去打乱列号**, c_new = c XOR r。\n");

    // 整张映射表
    printf("\n  整张映射表 (前 6 行 x 8 列):\n\n");
    printf("      plain 偏移                        swizzled Sw<5,0,5>\n");
    printf("        c=  0   1   2   3   4   5   6   7      c=  0   1   2   3   4   5   6   7\n");
    for (int rr = 0; rr < 6; ++rr) {
        printf("   r=%d   ", rr);
        for (int cc = 0; cc < 8; ++cc) printf("%4d", int(plain(rr, cc)));
        printf("     r=%d   ", rr);
        for (int cc = 0; cc < 8; ++cc) printf("%4d", int(swz(rr, cc)));
        if (rr == 0) printf("   <- r=0: XOR 0, 不变");
        if (rr == 1) printf("   <- 两两交换");
        if (rr == 3) printf("   <- 每 4 个一组倒转");
        printf("\n");
    }

    printf("\n  第 0 列的偏移序列 (这就是列读会撞不撞的关键):\n");
    printf("      r:        ");
    for (int rr = 0; rr < 8; ++rr) printf("%5d", rr);
    printf("\n      plain:    ");
    for (int rr = 0; rr < 8; ++rr) printf("%5d", int(plain(rr, 0)));
    printf("\n      bank:     ");
    for (int rr = 0; rr < 8; ++rr) printf("%5d", bank_of(int(plain(rr, 0)) * 4));
    printf("   <- 全 0, 撞死\n");
    printf("      swizzled: ");
    for (int rr = 0; rr < 8; ++rr) printf("%5d", int(swz(rr, 0)));
    printf("\n      bank:     ");
    for (int rr = 0; rr < 8; ++rr) printf("%5d", bank_of(int(swz(rr, 0)) * 4));
    printf("   <- 全不同\n");

    printf("\n  三个不变量 (实测):\n");
    printf("    cosize 不变    plain %d = swizzled %d       -> 不多占一个字节\n",
           int(cosize(plain)), int(cosize(swz)));
    // 双射检查
    {
        bool seen[CM * CN] = {false};
        bool bijective = true;
        for (int rr = 0; rr < CM; ++rr)
            for (int cc = 0; cc < CN; ++cc) {
                int o = int(swz(rr, cc));
                if (o < 0 || o >= CM * CN || seen[o]) bijective = false;
                else seen[o] = true;
            }
        printf("    是双射        全 %d 个坐标扫一遍无重复: %s   -> 不丢数据\n", CM * CN,
               bijective ? "是" : "否");
    }
    printf("    行读仍不冲突  plain %d-way, swizzled %d-way   -> 写 smem 那步没变慢\n",
           worst_row_conflict(plain), worst_row_conflict(swz));

    printf("\n  注意第三条是行读**不冲突**, 不是行内偏移**连续**。Sw<5,0,5> 的\n");
    printf("  第 1 行是 33 32 35 34 —— 不连续 (不能向量化), 但 32 个 lane 仍落\n");
    printf("  32 个不同 bank (不冲突)。这个区别就是 §4.4 的主题。\n");
}

// ===========================================================================
// §4.4  M 参数: 消冲突和向量化的权衡
// ===========================================================================
static void section44_m_param() {
    print_separator("§4.4  M 参数: 消冲突和向量化的权衡");

    auto plain = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto s505 = composition(Swizzle<5, 0, 5>{}, plain);
    auto s414 = composition(Swizzle<4, 1, 4>{}, plain);
    auto s323 = composition(Swizzle<3, 2, 3>{}, plain);

    printf("  M 保护最低 M 位不参与异或, 即 **2^M 个相邻元素保持连续**。\n\n");
    printf("    layout        列读最坏   行内最短连续   128-bit 向量 (需连续>=4)\n");
    printf("    ------------  --------   ------------   -----------------------\n");
    printf("    plain         %2d-way      %2d 个 float      可用\n", worst_col_conflict(plain),
           min_contiguous_run(plain));
    printf("    Sw<5,0,5>     %2d-way      %2d 个 float      **不可用**\n",
           worst_col_conflict(s505), min_contiguous_run(s505));
    printf("    Sw<4,1,4>     %2d-way      %2d 个 float      **不可用**\n",
           worst_col_conflict(s414), min_contiguous_run(s414));
    printf("    Sw<3,2,3>     %2d-way      %2d 个 float      可用\n", worst_col_conflict(s323),
           min_contiguous_run(s323));

    printf("\n  读这张表最容易犯的错: 挑冲突最少的 Sw<5,0,5>。\n");
    printf("  它把冲突消得最干净 (1-way), 但 M=0 意味着每个元素都被单独打乱,\n");
    printf("  行内一个连续对都不剩。拿 128-bit 指令 (一次搬 4 个连续 float) 去搬它,\n");
    printf("  **编译期**就挂:\n");
    printf("      static assertion failed: \"Copy_Traits: dst failed to vectorize\n");
    printf("      into registers. Layout is incompatible with this CopyOp.\"\n");

    printf("\n  Sw<3,2,3> 保住 4 个 float 连续 (M=2 -> 2^2=4), 128-bit 能用,\n");
    printf("  代价是 4-way 冲突。\n");

    printf("\n  选参数的规则:\n");
    printf("    **先定 M = 你要用的向量宽度, 再让 B、S 去消冲突。**\n");
    printf("    先保住向量化, 再谈消冲突 —— 这是 §4.6 里官方四个原子 M 全 = 4\n");
    printf("    的原因。\n");
}

// ===========================================================================
// §4.5  交给 TMA 时只有四种模式可选
//
// §4.3/§4.4 里的 Swizzle<B,M,S> 是完全自由的 —— 只要你自己写搬运代码,
// B/M/S 随便配。但一旦把这块 smem 交给 TMA, 自由度就没了:
//
//     descriptor 里存 swizzle 的只有 **几个比特**。
//
// CuTe 在编译期就替你挡住: 传一个不被支持的 Swizzle 给 make_tma_copy,
// 直接 static_assert 失败。把下面的开关打开就能亲眼看到。
// ===========================================================================

// 打开这个开关会**编译失败** —— 这正是本节要演示的
#define SHOW_UNSUPPORTED_SWIZZLE_ERROR 0

template <class SLay, class TmaLoad>
__global__ static void tma_swizzle_kernel(__grid_constant__ const TmaLoad tma, SLay slay,
                                          float* __restrict__ out) {
    constexpr int tx_bytes = CM * CN * sizeof(float);
    extern __shared__ __align__(128) char raw[];
    __shared__ uint64_t bar;

    auto sT = make_tensor(make_smem_ptr(reinterpret_cast<float*>(raw)), slay);

    auto gc = tma.get_tma_tensor(make_shape(Int<M>{}, Int<N>{}));
    auto gt = local_tile(gc, Shape<Int<CM>, Int<CN>>{}, make_coord(blockIdx.x, blockIdx.y));

    if (threadIdx.x == 0) initialize_barrier(bar, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        set_barrier_transaction_bytes(bar, tx_bytes);
        auto per = tma.get_slice(0);
        copy(tma.with(bar), per.partition_S(gt), per.partition_D(sT));
    }
    __syncthreads();
    wait_barrier(bar, 0);

    // 读的时候用**逻辑坐标** (r,c) —— swizzle 藏在 layout 里, 这里看不见
    auto mOut = make_tensor(make_gmem_ptr(out),
                            make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
    auto gOut = local_tile(mOut, Shape<Int<CM>, Int<CN>>{}, make_coord(blockIdx.x, blockIdx.y));
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) gOut(i / CN, i % CN) = sT(i / CN, i % CN);
}

template <class SLay>
static bool run_with_layout(SLay slay, Buffers& buf) {
    auto mIn = make_tensor(make_gmem_ptr(buf.d_in),
                           make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, mIn, slay);
    CUDA_CHECK(cudaMemset(buf.d_out, 0, Buffers::bytes));
    tma_swizzle_kernel<<<grid(), NTHR, cosize_v<SLay> * sizeof(float)>>>(tma, slay, buf.d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    return buf.check();
}

template <class Atom>
static void show_mode(const char* name, Atom atom, int row_bytes, Buffers& buf) {
    auto slay = tile_to_shape(atom, make_shape(Int<CM>{}, Int<CN>{}));
    bool ok = run_with_layout(slay, buf);
    printf("    %-7s 一行 %3d 字节   TMA %-4s   列读 %2d-way   行内连续 %2d   cosize %d\n", name,
           row_bytes, ok ? "正确" : "错误", worst_col_conflict(slay), min_contiguous_run(slay),
           int(cosize(slay)));
}

static void section45_only_four_modes() {
    print_separator("§4.5  交给 TMA 时只有四种模式可选");

    printf("  CuTe 把 descriptor 支持的四种摆法封成了四个 layout 原子。\n");
    printf("  名字里的数字是**一行占多少字节**:\n\n");
    printf("    float SW128 = ");
    print(GMMA::Layout_K_SW128_Atom<float>{});
    printf("\n    float SW64  = ");
    print(GMMA::Layout_K_SW64_Atom<float>{});
    printf("\n    float SW32  = ");
    print(GMMA::Layout_K_SW32_Atom<float>{});
    printf("\n    float INTER = ");
    print(GMMA::Layout_K_INTER_Atom<float>{});
    printf("\n\n");
    printf("  看它们的 Swizzle 参数: 全是 Sw<B,4,3>, 只有 B 在变 (3/2/1/0)。\n");
    printf("  **M 全 = 4, S 全 = 3** —— 这不是巧合, 是 descriptor 只支持这些。\n\n");

    printf("  TMA 接受的 Swizzle<B,M,S> 全集 (CuTe 源码 copy_traits_sm90_tma_swizzle.hpp):\n\n");
    printf("      M = 4, S = 3, B = 0..3     <- 就是上面四个原子\n");
    printf("      M = 5, S = 2, B = 2\n");
    printf("      M = 6,        B = 2\n\n");
    printf("  §4.3/§4.4 手写的那些 (Sw<5,0,5> M=0, Sw<3,2,3> M=2) **一个都不在里面**。\n");
    printf("  传给 make_tma_copy 会编译期失败:\n\n");
    printf("      static assertion failed: \"Unsupported layout swizzle.\"\n");
    printf("      static assertion failed: \"Expected 128b=16B=(2^4)B to 512b=64B=(2^6)B\n");
    printf("                                base swizzle.\"\n");

    printf("\n  为什么 M 必须 >= 4: M=4 表示\"最低 16 个元素不参与异或\"。float 是 4 字节,\n");
    printf("  16 个 float = 64 字节; half 是 2 字节, 16 个 half = 32 字节。TMA 一次访存\n");
    printf("  的最小粒度就在这个量级 —— 比它更细的打乱, 硬件表达不了。\n");
    printf("  换句话说 §4.4 那条\"先保住向量宽度\"的规则, 在 TMA 这里是**强制**的。\n");
}

// ===========================================================================
// §4.6  用 TMA: 官方原子 + "逻辑连续 vs 物理字节序"
//
// 和 §4.3/§4.4 不同, TMA 官方原子里的 swizzle 是**写进 descriptor 的**。
// 于是出现一个必须先说清的事实:
//
//     TMA 搬完数据后, smem 的**物理字节序**被硬件 XOR 过了。
//
// 硬件在写 smem 的时候把 swizzle 的 XOR 应用在**物理地址**上: 逻辑坐标
// (r,c) 的数据落在物理偏移 swz(r*CN+c), 不是 r*CN+c。
//
// 但如果你用**逻辑坐标** sT(r,c) 去读, 拿到的仍是正确数据 —— 因为
// layout 里带着同一份 swizzle 映射, 读取时自动做逆映射。
//
// 验证方式: 打印 TMA load 之后 smem 的裸字节序, 看到第 2 行开始乱序
// (SW128 是 4 个一组重排: 132 133 134 135 128 ...), 而逻辑坐标读取全对。
// ===========================================================================
template <class TmaLoad>
__global__ static void inspect_kernel(__grid_constant__ const TmaLoad tma, float* __restrict__ dump,
                                      int SM) {
    constexpr int tx_bytes = CM * CN * sizeof(float);
    __shared__ __align__(128) float smem[CM * CN];
    __shared__ uint64_t bar;

    auto sT = make_tensor(make_smem_ptr(smem), make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{}));
    auto gc = tma.get_tma_tensor(make_shape(Int<M>{}, Int<N>{}));
    auto gt = local_tile(gc, Shape<Int<CM>, Int<CN>>{}, make_coord(0, 0));

    if (threadIdx.x == 0) initialize_barrier(bar, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        set_barrier_transaction_bytes(bar, tx_bytes);
        auto per = tma.get_slice(0);
        copy(tma.with(bar), per.partition_S(gt), per.partition_D(sT));
    }
    __syncthreads();
    wait_barrier(bar, 0);

    for (int i = threadIdx.x; i < SM; i += blockDim.x) dump[i] = smem[i];
}

static void section46_tma_usage() {
    print_separator("§4.6  用 TMA: 官方原子 + 逻辑连续 vs 物理字节序");

    Buffers buf;
    auto mA = make_tensor(make_gmem_ptr(buf.d_in),
                          make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));

    auto swz = tile_to_shape(GMMA::Layout_K_SW128_Atom<float>{}, make_shape(Int<CM>{}, Int<CN>{}));
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, mA, swz);
    printf("  smem layout (交给 TMA 的): ");
    print(swz);
    printf("\n");

    // 把前 64 个 float 倒出来看物理字节序
    float* d_dump;
    CUDA_CHECK(cudaMalloc(&d_dump, sizeof(float) * CM * CN));
    inspect_kernel<<<1, NTHR>>>(tma, d_dump, 64);
    CUDA_CHECK(cudaDeviceSynchronize());
    float* h_dump = new float[CM * CN];
    CUDA_CHECK(cudaMemcpy(h_dump, d_dump, sizeof(float) * CM * CN, cudaMemcpyDeviceToHost));

    printf("\n  TMA load 之后, smem 的**物理字节序** (直接按地址读):\n");
    printf("    smem[0..31]  = ");
    for (int i = 0; i < 32; ++i) printf("%.0f ", h_dump[i]);
    printf("\n    smem[32..63] = ");
    for (int i = 32; i < 64; ++i) printf("%.0f ", h_dump[i]);
    printf("\n\n  前 32 个恰好是 0..31 (逻辑坐标=物理偏移的巧合), 但第 2 行开始:");
    printf("\n  132 133 134 135 128 129 ... —— SW128 的 XOR 在物理地址上生效, 4 个一组重排。\n");
    printf("  用逻辑坐标 sT(r,c) 读, 拿到的仍是正确数据 (layout 自动逆映射)。\n");

    printf("\n  那这个 XOR 到底保护什么? 和 §4.3/§4.4 的动机一样: **bank conflict**。\n");
    printf("  两个场景:\n\n");
    printf("    1) 如果 smem 按**物理地址**被读 (比如手写代码直接读裸数组, 或某些\n");
    printf("       consumer), 物理字节序的 XOR 让相邻 4 元素组错开 bank —— 和 §4.3\n");
    printf("       的逻辑重排是同一件事, 只是发生在物理层;\n");
    printf("    2) 当 TMA 直接喂 WGMMA 时 (cute_05/06), descriptor 里的 swizzle 字段\n");
    printf("       必须和 WGMMA 期望的规范形式一致 —— 那时它是一份**合同**。\n");

    printf("\n  对比 §4.3/§4.4: 手写搬运时 swizzle 是逻辑层的重排 (改 s(r,c) 的\n");
    printf("  偏移); TMA 搬运时 swizzle 是硬件层的重排 (物理字节序变, 逻辑坐标不变)。\n");
    printf("  **两条路的 swizzle 语义不同**, 这也是为什么 TMA 只接受那四种 ——\n");
    printf("  能写进硬件的比特就那么几个。\n");

    // 五个 layout 跑同一个 kernel, 验证"都能搬对"
    printf("\n  同一个 TMA kernel, 五种 smem layout 都能搬对 (tile %dx%d float):\n", CM, CN);
    CUDA_CHECK(cudaMemset(buf.d_out, 0, Buffers::bytes));
    run_with_layout(make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{}), buf);
    show_mode("SW128", GMMA::Layout_K_SW128_Atom<float>{}, 128, buf);
    show_mode("SW64", GMMA::Layout_K_SW64_Atom<float>{}, 64, buf);
    show_mode("SW32", GMMA::Layout_K_SW32_Atom<float>{}, 32, buf);
    show_mode("INTER", GMMA::Layout_K_INTER_Atom<float>{}, 16, buf);

    printf("\n  怎么选: 唯一的硬约束是**内层维度必须被原子的内层长度整除**。\n");
    printf("  float 的四个原子内层分别是 32 / 16 / 8 / 4 个元素:\n\n");
    printf("      CN (float) | SW128(32) | SW64(16) | SW32(8) | INTER(4)\n");
    printf("      -----------+-----------+----------+---------+---------\n");
    printf("          32     |    可     |    可    |   可    |   可\n");
    printf("          16     |   不可    |    可    |   可    |   可\n");
    printf("           8     |   不可    |   不可   |   可    |   可\n");
    printf("  违反了是**编译期**报错:\n");
    printf("      \"tile_to_shape: block shape does not divide the target shape\"\n");
    printf("\n  实用规则: 选能用的里面一行字节数最大的 (对齐越大访存越宽)。\n");
    printf("  本例 CN=%d float = %d 字节 -> 选 SW128。\n", CN, CN * 4);
}

int main() {
    printf("cute_04 v3 —— 搬进 smem 之后怎么摆: bank conflict 与 swizzle\n");
    printf("对应 README §4    需要 -arch=sm_90a\n");

    section41_banks();
    section42_padding();
    section43_swizzle_mapping();
    section44_m_param();
    section45_only_four_modes();
    section46_tma_usage();

    print_separator("小结");
    printf("  §4.1  行 stride = bank 数 -> 按列读 32-way conflict\n");
    printf("  §4.2  padding 能消冲突, 但多占 smem、破坏对齐、被 WGMMA 编译期拒绝\n");
    printf("  §4.3  swizzle = 用高位比特异或低位比特, cosize 不变、是双射\n");
    printf("  §4.4  M 定向量宽度, B/S 消冲突 —— 先保向量化再谈冲突\n");
    printf("  §4.5  descriptor 只认四种模式 (Sw<B,4,3>), 手写 swizzle 编译期被拒\n");
    printf("  §4.6  TMA 官方原子: 逻辑坐标不变, swizzle 是硬件层的重排\n");

    printf("\n下一步 (§5): 到这里一块 tile 的搬运已经完全会了。但搬和算目前是\n");
    printf("**串行**的 —— 搬的时候计算单元闲着。v4 用多个 smem buffer 让它们重叠。\n");
    printf("\nv3 OK\n");
    return 0;
}
