// v1: CuTe Layout 的代数运算和组合
//
// 讲解见本目录 README.md 的 §2 ~ §5。本文件只是那些内容的可执行版本：
// 每个小节的编号（§1a, §1b, ...）都在 README 里有对应的图和推导。
//
// 学习目标：
//   1. colex 展开顺序                         (README §2)
//   2. composition: 采样 / 三步法 / 反向规则  (README §3)
//   3. 嵌套 mode 的由来与用法                 (README §3.4 §3.5 §3.6)
//   4. complement: 补齐剩余空间               (README §4)
//   5. coalesce: 找等价的最简表示             (README §5)
//
// 核心概念：
//   - composition(A, B): 复合映射 (A∘B)(x) = A(B(x))；B 选形状，A 给地址
//   - complement(L, N):  L 的补集，与 L 一起不重不漏地铺满 N 个位置
//   - coalesce(L):       合并可以合并的相邻维度，不改变映射
//   - 嵌套 mode:         跨 mode 结构的无损表示，也用来主动分块

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

    // ---- 1a. 先立规矩：一维下标如何展开成坐标（colex 序） ----
    // 这是理解 composition 的前置卡点。A(k) 里的单个整数 k，CuTe 按
    // colex（column-major / 第一个 mode 变化最快）展开成坐标：
    //     shape=(4,8):  k -> (k%4, k/4)      注意不是 (k/8, k%8)!
    // 逻辑下标的展开永远是列主序，与 stride 是 row- 还是 col-major 无关。
    // stride 只负责最后一步：坐标 -> 内存偏移。
    auto A = make_layout(make_shape(Int<4>{}, Int<8>{}),
                         make_stride(Int<8>{}, Int<1>{}));  // 4x8 row-major

    printf("A (4x8 row-major) = ");
    print(A);
    printf("\n");
    print_layout(A);

    printf("\n一维下标 k 按 colex 展开 -> (k%%4, k/4):\n");
    for (int k = 0; k < 6; ++k) {
        auto crd = idx2crd(k, shape(A));
        printf("  k=%d -> crd=(%d,%d)  A(k)=%2d   A(k%%4,k/4)=%2d  %s\n", k,
               int(get<0>(crd)), int(get<1>(crd)), int(A(k)), int(A(k % 4, k / 4)),
               A(k) == A(k % 4, k / 4) ? "一致" : "不一致");
    }

    // ---- 1b. 最小例子：一维 composition 就是"采样" ----
    // P 提供偏移序列，Q 说"我要第几个"。stride 直接相乘。
    auto P = make_layout(Int<20>{}, Int<2>{});  // 偏移 0,2,4,6,...
    auto Q = make_layout(Int<4>{}, Int<5>{});   // 取第 0,5,10,15 个
    auto PQ = composition(P, Q);

    printf("\n--- 一维 composition = 采样 ---\n");
    printf("P = ");
    print(P);
    printf("   偏移序列: ");
    for (int i = 0; i < 8; ++i) printf("%d ", int(P(i)));
    printf("...\n");
    printf("Q = ");
    print(Q);
    printf("   取值: ");
    for (int i = 0; i < 4; ++i) printf("%d ", int(Q(i)));
    printf("\n");
    printf("composition(P, Q) = ");
    print(PQ);
    printf("   (stride 相乘: 2 x 5 = 10)\n");
    for (int i = 0; i < 4; ++i) {
        printf("  i=%d: (P∘Q)(i)=%2d   P(Q(i))=P(%2d)=%2d\n", i, int(PQ(i)), int(Q(i)),
               int(P(Q(i))));
    }

    // ---- 1b'. 用公式算 composition: 点乘 -> 展开 -> 点乘 (README §3.0 "用公式怎么算") ----
    // 求值公式:  L(k) = sum_m ( floor(k / L_m) mod s_m ) * d_m,  L_m = s_0*...*s_{m-1}
    // "stride 相乘"只是一维专属的捷径; 多维必须走三段式。
    printf("\n--- 用公式算, 而不是背捷径 ---\n");
    printf("Q = 4:5    Q(x) = (floor(x/1) mod 4) * 5 = 5x        (x<4, mod 不回绕)\n");
    printf("P = 20:2   P(k) = (floor(k/1) mod 20) * 2 = 2k       (k<20, mod 不回绕)\n");
    printf("=> (P∘Q)(x) = 2*(5x) = 10x  ->  4:10\n");
    printf("回绕条件: cosize(Q)=%d <= size(P)=%d，成立\n", int(cosize(Q)), int(size(P)));

    // 二维: ι 不再是恒等，捷径失效，必须三段式
    printf("\n二维 A = (4,8):(8,1) 上的三段式 (L_0=1, L_1=4):\n");
    printf("%-10s %-14s %-26s %s\n", "j=(i,j)", "1)点乘 B", "2)展开 A: (i_0,i_1)", "3)点乘 A");
    auto Bf = make_layout(make_shape(Int<2>{}, Int<4>{}), make_stride(Int<1>{}, Int<4>{}));
    auto ABf = composition(A, Bf);
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 2; ++i) {
            int k = i * 1 + j * 4;              // 1) B 的内积
            int i0 = (k / 1) % 4, i1 = (k / 4) % 8;  // 2) 按 A 的 shape colex 展开
            int addr = i0 * 8 + i1 * 1;         // 3) A 的内积
            printf("  (%d,%d)     k=%-12d (i_0,i_1)=(%d,%d)%-14s %2d   (A∘B)(i,j)=%2d %s\n", i, j,
                   k, i0, i1, "", addr, int(ABf(i, j)), addr == int(ABf(i, j)) ? "" : "<-- MISMATCH");
        }
    }

    // ---- 1c. 逻辑基：设计 B 的刻度尺 (README §3.1) ----
    // 逻辑基 L_m = shape_0 * ... * shape_{m-1}，表示"第 m 维前进 1 格，k 加多少"。
    // 只由 Shape 决定，与 Stride 无关。
    printf("\n--- 逻辑基 L_m (只看 Shape) ---\n");
    printf("A 的 shape = (4,8)  ->  L_0 = 1, L_1 = shape_0 = 4\n");
    printf("含义: row 前进 1 格 -> k 加 1 ; col 前进 1 格 -> k 加 4\n");

    // ---- 1d. 反向算 A∘B: 单 mode 规则 (README §3.3) ----
    // B = s:d  ->  定位 d = delta * L_m，则结果 = s : delta * A_stride[m]
    printf("\n--- 单 mode B = s:d 的闭式规则 ---\n");
    printf("规则: d = delta * L_m  =>  结果 stride = delta * A_stride[m]\n");
    printf("%-8s %-34s %s\n", "B", "定位 (d = delta * L_m)", "A∘B");
    struct {
        const char* b;
        const char* how;
    } single[] = {
        {"4:1", "d=1=1*L_0 -> m=0,delta=1 -> 1*8"},
        {"2:2", "d=2=2*L_0 -> m=0,delta=2 -> 2*8"},
        {"4:4", "d=4=1*L_1 -> m=1,delta=1 -> 1*1"},
        {"4:8", "d=8=2*L_1 -> m=1,delta=2 -> 2*1"},
        {"2:16", "d=16=4*L_1 -> m=1,delta=4 -> 4*1"},
    };
    printf("%-8s %-34s ", single[0].b, single[0].how);
    print(composition(A, make_layout(Int<4>{}, Int<1>{})));
    printf("\n");
    printf("%-8s %-34s ", single[1].b, single[1].how);
    print(composition(A, make_layout(Int<2>{}, Int<2>{})));
    printf("\n");
    printf("%-8s %-34s ", single[2].b, single[2].how);
    print(composition(A, make_layout(Int<4>{}, Int<4>{})));
    printf("\n");
    printf("%-8s %-34s ", single[3].b, single[3].how);
    print(composition(A, make_layout(Int<4>{}, Int<8>{})));
    printf("\n");
    printf("%-8s %-34s ", single[4].b, single[4].how);
    print(composition(A, make_layout(Int<2>{}, Int<16>{})));
    printf("\n");

    // ---- 1e. 正向设计 B: 三步法取子块 (README §3.2 ④⑤⑥) ----
    // 目标: A 的左上角 2x4 子块
    //   step1  L = (1, 4)
    //   step2  shape=(2,4)  stride=(row步进1 * L_0, col步进1 * L_1)=(1,4)
    //   step3  起点 (0,0) -> base = 0
    auto B = make_layout(make_shape(Int<2>{}, Int<4>{}), make_stride(Int<1>{}, Int<4>{}));
    auto AB = composition(A, B);

    printf("\n--- 三步法设计 B: 取左上角 2x4 子块 ---\n");
    printf("step1: L = (1, 4)\n");
    printf("step2: shape=(2,4), stride=(1*L_0, 1*L_1)=(1,4)\n");
    printf("step3: 起点 (0,0) -> base = 0\n");
    printf("B    = ");
    print(B);
    printf("\nA∘B  = ");
    print(AB);
    printf("\n");
    print_layout(AB);

    printf("\n逐点验证 (A∘B)(i,j) == A(B(i,j)):\n");
    bool all_match = true;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 2; ++i) {
            int lhs = AB(i, j);
            int rhs = A(B(i, j));  // B 的输出交给 A 当一维逻辑下标
            if (lhs != rhs) all_match = false;
            printf("  (%d,%d): A∘B=%2d   B=%2d -> A(B)=%2d  %s\n", i, j, lhs, int(B(i, j)), rhs,
                   lhs == rhs ? "" : "<-- MISMATCH");
        }
    }
    printf("全部一致: %s\n", all_match ? "YES" : "NO");

    // ---- 1f. 换 stride 就变成隔列采样 (README §3.2 ⑦) ----
    // B 的列步长从 1*L_1=4 改成 2*L_1=8，就是"跳一列取一列"
    auto Samp = make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<1>{}, Int<8>{}));
    auto ASamp = composition(A, Samp);

    printf("\n--- 列步长改成 8 -> 隔列采样 (取 A 的第 0,2,4,6 列) ---\n");
    printf("Samp   = ");
    print(Samp);
    printf("\n");
    printf("A∘Samp = ");
    print(ASamp);
    printf("\n");
    print_layout(ASamp);

    // ---- 1g. 三步法的完整形态：带起点的任意子区域 (README §3.2 ⑧) ----
    // 目标: 取 row{1,3} x col{2,3,4}  —— 起点不在 (0,0)，行还要跳步
    //   step1  L = (1, 4)
    //   step2  row 步进 2 -> 2*L_0 = 2 ; col 步进 1 -> 1*L_1 = 4
    //          shape=(2,3)  stride=(2,4)
    //   step3  base = A(1,2)
    printf("\n--- 三步法完整形态: 取 row{1,3} x col{2,3,4} ---\n");
    auto Bd = make_layout(make_shape(Int<2>{}, Int<3>{}), make_stride(Int<2>{}, Int<4>{}));
    auto ABd = composition(A, Bd);
    const int base = A(1, 2);  // 起点偏移由 base 承担，composition 只表达形状

    printf("step2: shape=(2,3), stride=(2*L_0, 1*L_1)=(2,4)\n");
    printf("step3: base = A(1,2) = %d\n", base);
    printf("B    = ");
    print(Bd);
    printf("\nA∘B  = ");
    print(ABd);
    printf("\n验证 base + (A∘B)(i,j) == A(1+2i, 2+j):\n");
    bool sub_ok = true;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 2; ++i) {
            int got = base + int(ABd(i, j));
            int want = int(A(1 + 2 * i, 2 + j));
            if (got != want) sub_ok = false;
            printf("  (%d,%d): base+A∘B=%2d   A(%d,%d)=%2d  %s\n", i, j, got, 1 + 2 * i, 2 + j,
                   want, got == want ? "" : "<-- MISMATCH");
        }
    }
    printf("全部一致: %s\n", sub_ok ? "YES" : "NO");

    // ---- 1h. 跨界 -> 嵌套 mode 的由来 (README §3.4) ----
    // 单 mode 规则要求"取的个数不超出这一维"。超出时 CuTe 自动分裂。
    printf("\n--- s 超出 mode 容量 -> 自动分裂 ---\n");
    printf("A 的 mode0 只有 4 格，连续取更多就会溢进 mode1:\n");
    printf("  A∘(8:1)  = ");
    print(composition(A, make_layout(Int<8>{}, Int<1>{})));
    printf("   <- 8 = mode0 全部 4 个 x mode1 的 2 个\n");
    printf("  A∘(16:1) = ");
    print(composition(A, make_layout(Int<16>{}, Int<1>{})));
    printf("\n");
    printf("  A∘(32:1) = ");
    print(composition(A, make_layout(Int<32>{}, Int<1>{})));
    printf("   <- 取满 32 个 = A 本身\n");

    // 多 mode B 里发生分裂时，两截必须留在同一个 mode 位置 -> 套括号
    auto R = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<1>{}, Int<8>{}));
    auto AR = composition(A, R);

    printf("\n--- 多 mode B 中的分裂 -> 嵌套 mode ---\n");
    printf("R    = ");
    print(R);
    printf("\n");
    printf("  mode0: 8:1 -> 跨界分裂 -> (4,2):(8,1)\n");
    printf("  mode1: 4:8 -> 4:2\n");
    printf("A∘R  = ");
    print(AR);
    printf("   <- 外层括号位置 = B 的第几个 mode\n");

    // ---- 1i. 嵌套 mode 怎么用 (README §3.5) ----
    printf("\n--- 嵌套 layout 的使用方式 ---\n");
    printf("rank(AR)    = %d   <- 仍是 2 个 mode，不是 3\n", int(rank(AR)));
    printf("size(AR)    = %d\n", int(size(AR)));
    printf("size<0>(AR) = %d   <- 第 0 个 mode 的整体大小 (4*2)\n", int(size<0>(AR)));
    printf("get<0>(AR)  = ");
    print(get<0>(AR));
    printf("   <- 子 layout，本身可继续参与运算\n");

    printf("\n扁平坐标与嵌套坐标等价 (mode0 内部按 colex: flat = i0 + 4*i1):\n");
    bool nest_ok = true;
    for (int i1 = 0; i1 < 2; ++i1) {
        for (int i0 = 0; i0 < 4; ++i0) {
            int flat = i0 + 4 * i1;
            int a = AR(flat, 2);
            int b = AR(make_coord(i0, i1), 2);
            if (a != b) nest_ok = false;
            printf("  AR(%d,2)=%2d  ==  AR((%d,%d),2)=%2d  %s\n", flat, a, i0, i1, b,
                   a == b ? "" : "<-- MISMATCH");
        }
    }
    printf("全部相等: %s\n", nest_ok ? "YES" : "NO");

    printf("\n摊平与塌缩:\n");
    printf("  flatten(AR)  = ");
    print(flatten(AR));
    printf("   <- 去掉括号，rank 2 -> 3\n");
    printf("  coalesce(AR) = ");
    print(coalesce(AR));
    printf("   <- 等价最简形式，正是 A 本身!\n");
    printf("  => 嵌套只是表示形式，映射本身没变\n");

    // ---- 1j. 嵌套的正面用法: 主动分块 (README §3.6) ----
    // 上面的嵌套是"跨界"被动产生的。更常见的是主动造一个嵌套 mode 表达分块。
    // 拿 §1e 取出的 2x4 子块，把 4 列切成 2 块、每块 2 列。
    printf("\n--- 主动分块: 把 2x4 子块的 4 列切成 2 块 ---\n");
    auto blk = make_layout(make_shape(Int<2>{}, Int<4>{}), make_stride(Int<8>{}, Int<1>{}));
    auto tiled = composition(
        blk, make_tile(make_layout(Int<2>{}, Int<1>{}),  // 行不切
                       make_layout(make_shape(Int<2>{}, Int<2>{}),
                                   make_stride(Int<1>{}, Int<2>{}))));  // 列切成 (2,2)

    printf("blk   = ");
    print(blk);
    printf("   <- README §3.2 取出的左上角 2x4 子块\n");
    printf("tiled = ");
    print(tiled);
    printf("   <- 列坐标变成两层 (j0=块内第几列, j1=第几块)\n");
    print_layout(tiled);

    printf("\n同一个格子的两种叫法 (分块只是换坐标系，映射没变):\n");
    for (int j1 = 0; j1 < 2; ++j1) {
        for (int j0 = 0; j0 < 2; ++j0) {
            int j = j0 + 2 * j1;
            printf("  blk(1,%d)=%2d   ==   tiled(1,(j0=%d,j1=%d))=%2d\n", j, int(blk(1, j)), j0,
                   j1, int(tiled(1, make_coord(j0, j1))));
        }
    }

    // 一维的同款操作有专门的函数
    printf("\n一维分块用 logical_divide:\n");
    printf("  logical_divide(24:1, 4) = ");
    print(logical_divide(make_layout(Int<24>{}, Int<1>{}), Int<4>{}));
    printf("   <- 24 个切成\"每块 4 个，共 6 块\"\n");
    printf("  mode0 = 块内 (步进 1, 取 4 个 -> 1*1=1)\n");
    printf("  mode1 = 块号 (步进 4, 取 6 个 -> 4*1=4)\n");

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

    // 换个输入体会"补集"的含义：L 跨步时，空洞在组内，补集的 stride 就是 1
    printf("\n--- 换个输入: 补集在补什么 ---\n");
    printf("  complement(32:1, 64) = ");
    print(complement(make_layout(Int<32>{}, Int<1>{}), Int<64>{}));
    printf("   <- L 铺满 0-31，补集负责跨到 32\n");
    printf("  complement(8:4,  32) = ");
    print(complement(make_layout(Int<8>{}, Int<4>{}), Int<32>{}));
    printf("   <- L 只碰 0,4,...,28，补集负责填组内的 0-3\n");

    // ========== 3. Coalesce (维度合并) ==========
    print_separator("3. Layout Coalesce");

    // coalesce 合并相邻维度的判据：
    //     stride[i+1] == stride[i] * shape[i]
    // 关键：因为逻辑下标按 colex 展开（第一个 mode 变化最快），
    // 能合并的是 **column-major 相邻**；row-major 反而合不了。
    // 这一点很反直觉，下面用五个例子逐个验证。

    struct {
        const char* desc;
        const char* why;
    } notes[] = {
        {"(2,3,4):(12,4,1)", "12*2=24 != 4  -> 合不了"},
        {"(4,3,2):(1,4,12)", "1*4=4 OK, 4*3=12 OK -> 全合并成 24:1"},
        {"(4,8):(1,4)  col-major", "1*4=4 OK -> 合并成 32:1"},
        {"(4,8):(8,1)  row-major", "8*4=32 != 1 -> 合不了"},
        {"(4,8):(1,16) 有空洞", "1*4=4 != 16 -> 合不了"},
    };

    auto c1 = make_layout(make_shape(Int<2>{}, Int<3>{}, Int<4>{}),
                          make_stride(Int<12>{}, Int<4>{}, Int<1>{}));
    auto c2 = make_layout(make_shape(Int<4>{}, Int<3>{}, Int<2>{}),
                          make_stride(Int<1>{}, Int<4>{}, Int<12>{}));
    auto c3 = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<1>{}, Int<4>{}));
    auto c4 = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
    auto c5 = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<1>{}, Int<16>{}));

    printf("%-26s  %s\n", "输入 layout", "coalesce 结果 / 判据");
    printf("%-26s  ", notes[0].desc);
    print(coalesce(c1));
    printf("   %s\n", notes[0].why);
    printf("%-26s  ", notes[1].desc);
    print(coalesce(c2));
    printf("   %s\n", notes[1].why);
    printf("%-26s  ", notes[2].desc);
    print(coalesce(c3));
    printf("   %s\n", notes[2].why);
    printf("%-26s  ", notes[3].desc);
    print(coalesce(c4));
    printf("   %s\n", notes[3].why);
    printf("%-26s  ", notes[4].desc);
    print(coalesce(c5));
    printf("   %s\n", notes[4].why);

    printf("\n注意: coalesce 不改变映射本身，只是找一个更简洁的等价表示。\n");
    printf("      合不了并不是失败 —— 说明这个 layout 本来就有跨步或空洞。\n");

    // ========== 4. 总结 ==========
    print_separator("总结");
    printf("本程序演示的 Layout 代数 (讲解见 README):\n");
    printf("  §1a  colex 展开顺序                      -> README §2\n");
    printf("  §1b  一维 composition = 采样             -> README §3.0\n");
    printf("  §1c  逻辑基 L_m                          -> README §3.1\n");
    printf("  §1d  单 mode B 的闭式规则                -> README §3.3\n");
    printf("  §1e  三步法: 取左上角 2x4 子块           -> README §3.2 ④⑤⑥\n");
    printf("  §1f  换 stride -> 隔列采样               -> README §3.2 ⑦\n");
    printf("  §1g  三步法完整形态: 带起点的子区域      -> README §3.2 ⑧\n");
    printf("  §1h  跨界分裂 -> 嵌套 mode 的由来        -> README §3.4\n");
    printf("  §1i  嵌套 mode 怎么读、怎么用            -> README §3.5\n");
    printf("  §1j  主动分块                            -> README §3.6\n");
    printf("  §2   complement: 把剩下的补齐            -> README §4\n");
    printf("  §3   coalesce:   找等价的更简洁表示      -> README §5\n");
    printf("\n下一步: v2 会用 Layout 操作真实数据 (README §6)\n");

    return 0;
}
