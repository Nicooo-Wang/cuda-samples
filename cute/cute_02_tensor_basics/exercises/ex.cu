// Section 02 练习框架
//
// 用法:
//   make ex && ./ex     跑全部
//   ./ex 3              只跑第 3 题
//
// 每题只需填 TODO 处，框架自动检查。题目见 ../README.md 的"练习"一节。

#include <unistd.h>  // dup/dup2/close，用于把 print() 抓成字符串

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cute/tensor.hpp>

using namespace cute;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, fmt, ...)                              \
    do {                                                   \
        if (cond) {                                        \
            ++g_pass;                                      \
            printf("    [PASS] " fmt "\n", ##__VA_ARGS__); \
        } else {                                           \
            ++g_fail;                                      \
            printf("    [FAIL] " fmt "\n", ##__VA_ARGS__); \
        }                                                  \
    } while (0)

#define TODO_INT (-999999)
#define TODO_F (-999999.0f)

static void banner(const char* s) { printf("\n===== %s =====\n", s); }

template <class T>
static void to_str(T const& obj, char* buf, size_t n) {
    buf[0] = '\0';
    FILE* tmp = tmpfile();
    if (!tmp) return;
    int saved = dup(fileno(stdout));
    fflush(stdout);
    dup2(fileno(tmp), fileno(stdout));
    print(obj);
    fflush(stdout);
    dup2(saved, fileno(stdout));
    close(saved);
    rewind(tmp);
    size_t got = fread(buf, 1, n - 1, tmp);
    buf[got] = '\0';
    fclose(tmp);
}

// 24 个元素的公共数据
static float g_v[24];
static void reset_v() {
    for (int i = 0; i < 24; ++i) g_v[i] = i;
}

// ============================================================
// 练习 1 — Tensor 基础                                   ★☆☆
// ============================================================
void ex1() {
    banner("练习 1: Tensor 基础");
    reset_v();

    // 创建 (4,6) row-major
    auto T = make_tensor(make_gmem_ptr(g_v),
                         make_layout(make_shape(Int<4>{}, Int<6>{}),
                                     make_stride(Int<6>{}, Int<1>{})));
    printf("  T = ");
    print(T);
    printf("\n");

    // ---- TODO ----
    const float my_T_2_3 = 15;   // T(2,3) 的值
    const int my_size = 24;    // size(T)
    const int my_rank = 2;    // rank(T)
    const char* my_row1 = "(_6):(_1)";        // T(1,_) 的 layout，格式如 "(_6):(_1)"
    const float my_row1_first = 6;  // T(1,_) 的第一个元素
    const char* my_col2 = "(_4):(_6)";        // T(_,2) 的 layout
    const float my_col2_first = 2;  // T(_,2) 的第一个元素
    // --------------

    if (my_T_2_3 == TODO_F) {
        printf("    (未填写，跳过)\n");
        printf("    参考: T(2,3)=%.0f size=%d rank=%d\n", float(T(2, 3)), int(size(T)),
               int(rank(T)));
        return;
    }

    CHECK(my_T_2_3 == float(T(2, 3)), "T(2,3): 你的 %.0f, 实际 %.0f", my_T_2_3, float(T(2, 3)));
    CHECK(my_size == int(size(T)), "size: 你的 %d, 实际 %d", my_size, int(size(T)));
    CHECK(my_rank == int(rank(T)), "rank: 你的 %d, 实际 %d", my_rank, int(rank(T)));

    auto r1 = T(1, _);
    auto c2 = T(_, 2);
    char b1[64], b2[64];
    to_str(r1.layout(), b1, sizeof(b1));
    to_str(c2.layout(), b2, sizeof(b2));

    if (my_row1[0]) CHECK(strcmp(my_row1, b1) == 0, "T(1,_) layout: 预测 %s 实际 %s", my_row1, b1);
    if (my_row1_first != TODO_F)
        CHECK(my_row1_first == float(r1(0)), "T(1,_) 首元素: 你的 %.0f 实际 %.0f", my_row1_first,
              float(r1(0)));
    if (my_col2[0]) CHECK(strcmp(my_col2, b2) == 0, "T(_,2) layout: 预测 %s 实际 %s", my_col2, b2);
    if (my_col2_first != TODO_F)
        CHECK(my_col2_first == float(c2(0)), "T(_,2) 首元素: 你的 %.0f 实际 %.0f", my_col2_first,
              float(c2(0)));
}

// ============================================================
// 练习 2 — view 语义                                     ★☆☆
// ============================================================
void ex2() {
    banner("练习 2: view 语义");
    reset_v();

    auto T = make_tensor(make_gmem_ptr(g_v),
                         make_layout(make_shape(Int<4>{}, Int<6>{}),
                                     make_stride(Int<6>{}, Int<1>{})));

    // ---- TODO ----
    const float my_v23_after_T = -1.0f;  // 执行 T(3,5) = -1.0f 后, g_v[23] = ?
    const float my_v23_after_r = -2.0f;  // 再执行 r(5) = -2.0f (r = T(3,_)) 后, g_v[23] = ?
    // --------------

    if (my_v23_after_T == TODO_F) {
        printf("    (未填写，跳过)\n");
        T(3, 5) = -1.0f;
        printf("    参考: T(3,5)=-1 之后 g_v[23] = %.0f\n", g_v[23]);
        return;
    }

    T(3, 5) = -1.0f;
    CHECK(my_v23_after_T == g_v[23], "T(3,5)=-1 后 g_v[23]: 你的 %.0f 实际 %.0f", my_v23_after_T,
          g_v[23]);

    auto r = T(3, _);
    r(5) = -2.0f;
    CHECK(my_v23_after_r == g_v[23], "r(5)=-2 后 g_v[23]: 你的 %.0f 实际 %.0f", my_v23_after_r,
          g_v[23]);
    printf("    => 切片同样是 view，写它一样穿透到底层\n");
}

// ============================================================
// 练习 3 — local_tile                                    ★★☆
// ============================================================
static float g_big[96];

void ex3() {
    banner("练习 3: local_tile");
    for (int i = 0; i < 96; ++i) g_big[i] = i;

    auto BT = make_tensor(make_gmem_ptr(g_big),
                          make_layout(make_shape(Int<8>{}, Int<12>{}),
                                      make_stride(Int<12>{}, Int<1>{})));
    printf("  BT = ");
    print(BT);
    printf("   (8x12, 值 0-95)\n");

    auto tile = make_shape(Int<4>{}, Int<4>{});

    // ---- TODO ----
    const int my_num_tiles = 6;   // 一共几块? (8/4) x (12/4)
    const float my_t12_first = 56;   // local_tile(BT, (4,4), (1,2)) 的首元素
    const char* my_t12_layout = "(_4,_4):(_12,_1)";      // 它的 layout
    const int same_as_t00 = 1;    // 和 (0,0) 块 layout 相同吗? 1=是 0=否
    // --------------

    auto t00 = local_tile(BT, tile, make_coord(0, 0));
    auto t12 = local_tile(BT, tile, make_coord(1, 2));

    if (my_num_tiles == TODO_INT) {
        printf("    (未填写，跳过)\n");
        printf("    参考: t12 首元素=%.0f  layout=", float(t12(0, 0)));
        print(t12.layout());
        printf("\n    zipped_divide = ");
        print(zipped_divide(BT, tile).layout());
        printf("\n");
        return;
    }

    CHECK(my_num_tiles == 6, "块数: 你的 %d, 实际 (8/4)x(12/4)=6", my_num_tiles);
    CHECK(my_t12_first == float(t12(0, 0)), "t12 首元素: 你的 %.0f 实际 %.0f", my_t12_first,
          float(t12(0, 0)));

    char b[64], b0[64];
    to_str(t12.layout(), b, sizeof(b));
    to_str(t00.layout(), b0, sizeof(b0));
    if (my_t12_layout[0]) CHECK(strcmp(my_t12_layout, b) == 0, "t12 layout: 预测 %s 实际 %s",
                                my_t12_layout, b);
    CHECK((same_as_t00 == 1) == (strcmp(b, b0) == 0), "是否与 t00 同 layout: 你答 %s (t00=%s t12=%s)",
          same_as_t00 ? "是" : "否", b0, b);

    printf("\n  zipped_divide(BT,(4,4)) = ");
    print(zipped_divide(BT, tile).layout());
    printf("\n  想一想: mode1 的两个 stride 各代表什么?\n");
}

// ============================================================
// 练习 4 — thr_layout 决定分配模式                       ★★☆
// ============================================================
static float g_v32[32];

void ex4() {
    banner("练习 4: thr_layout 决定分配模式");
    for (int i = 0; i < 32; ++i) g_v32[i] = i;

    auto VT = make_tensor(make_gmem_ptr(g_v32), make_layout(Int<32>{}, Int<1>{}));

    // ---- TODO: 填 tid=0 和 tid=1 拿到的第一个元素 ----
    // 三种 thr_layout: (1) 8:1   (2) 8:4   (3) 4:1
    const int t0_of[3] = {0, 0, 0};  // 各自 tid=0 的首元素
    const int t1_of[3] = {1, 4, 1};  // 各自 tid=1 的首元素
    // ---- TODO: 哪个 thr_layout 有问题(线程间重复)? 填序号 1/2/3 ----
    const int broken_one = 2;
    // -------------------------------------------------------------

    auto a = make_layout(Int<8>{}, Int<1>{});
    auto b = make_layout(Int<8>{}, Int<4>{});
    auto c = make_layout(Int<4>{}, Int<1>{});

    printf("  (1) thr = 8:1\n");
    for (int t = 0; t < 3; ++t) {
        auto p = local_partition(VT, a, t);
        printf("      tid%d: ", t);
        for (int i = 0; i < size(p); ++i) printf("%2.0f ", float(p(i)));
        printf("\n");
    }
    printf("  (2) thr = 8:4\n");
    for (int t = 0; t < 3; ++t) {
        auto p = local_partition(VT, b, t);
        printf("      tid%d: ", t);
        for (int i = 0; i < size(p); ++i) printf("%2.0f ", float(p(i)));
        printf("\n");
    }
    printf("  (3) thr = 4:1\n");
    for (int t = 0; t < 3; ++t) {
        auto p = local_partition(VT, c, t);
        printf("      tid%d: ", t);
        for (int i = 0; i < size(p); ++i) printf("%2.0f ", float(p(i)));
        printf("\n");
    }

    if (t0_of[0] == TODO_INT) {
        printf("    (未填写，跳过 —— 上面的输出就是答案，先自己预测再对照)\n");
        return;
    }

    auto p0a = local_partition(VT, a, 0);
    auto p1a = local_partition(VT, a, 1);
    auto p0b = local_partition(VT, b, 0);
    auto p1b = local_partition(VT, b, 1);
    auto p0c = local_partition(VT, c, 0);
    auto p1c = local_partition(VT, c, 1);
    const int real0[3] = {int(p0a(0)), int(p0b(0)), int(p0c(0))};
    const int real1[3] = {int(p1a(0)), int(p1b(0)), int(p1c(0))};

    for (int i = 0; i < 3; ++i) {
        CHECK(t0_of[i] == real0[i], "(%d) tid0 首元素: 你的 %d 实际 %d", i + 1, t0_of[i], real0[i]);
        CHECK(t1_of[i] == real1[i], "(%d) tid1 首元素: 你的 %d 实际 %d", i + 1, t1_of[i], real1[i]);
    }
    CHECK(broken_one == 2, "有问题的是 (%d) —— 8:4 让不同 tid 拿到相同元素", broken_one);
}

// ============================================================
// 练习 5 — 两级范式                                      ★★★
// ============================================================
static float g_b64[64];

void ex5() {
    banner("练习 5: tile -> partition 两级范式");
    for (int i = 0; i < 64; ++i) g_b64[i] = i;

    auto BT = make_tensor(make_gmem_ptr(g_b64),
                          make_layout(make_shape(Int<8>{}, Int<8>{}),
                                      make_stride(Int<8>{}, Int<1>{})));

    // ---- TODO ----
    const float my_blk_first = 4;  // local_tile(BT,(4,4),(0,1)) 的首元素
    // tid=3 在 2x2 线程布局下拿到的 4 个值 (按 (i,j) 遍历顺序: (0,0)(1,0)(0,1)(1,1))
    const int my_tid3[4] = {13, 29, 15, 31};
    // --------------

    auto blk = local_tile(BT, make_shape(Int<4>{}, Int<4>{}), make_coord(0, 1));
    auto thr = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<2>{}, Int<1>{}));
    auto mine = local_partition(blk, thr, 3);

    if (my_blk_first == TODO_F) {
        printf("    (未填写，跳过)\n");
        printf("    参考: blk 首元素=%.0f\n", float(blk(0, 0)));
        printf("    tid3 拿到: ");
        for (int j = 0; j < size<1>(mine); ++j)
            for (int i = 0; i < size<0>(mine); ++i) printf("%.0f ", float(mine(i, j)));
        printf("\n");
        return;
    }

    CHECK(my_blk_first == float(blk(0, 0)), "blk 首元素: 你的 %.0f 实际 %.0f", my_blk_first,
          float(blk(0, 0)));

    int k = 0;
    bool ok = true;
    for (int j = 0; j < size<1>(mine); ++j)
        for (int i = 0; i < size<0>(mine); ++i, ++k)
            if (my_tid3[k] != int(mine(i, j))) {
                ok = false;
                printf("    第 %d 个: 你的 %d 实际 %d\n", k, my_tid3[k], int(mine(i, j)));
            }
    CHECK(ok, "tid3 的 4 个元素全部正确");
}

// ============================================================
// 练习 6 — make_fragment_like                            ★★☆
// ============================================================
void ex6() {
    banner("练习 6: make_fragment_like");
    for (int i = 0; i < 64; ++i) g_b64[i] = i;

    auto BT = make_tensor(make_gmem_ptr(g_b64),
                          make_layout(make_shape(Int<8>{}, Int<8>{}),
                                      make_stride(Int<8>{}, Int<1>{})));
    auto blk = local_tile(BT, make_shape(Int<4>{}, Int<4>{}), make_coord(1, 0));
    auto thr = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<2>{}, Int<1>{}));
    auto mine = local_partition(blk, thr, 0);

    printf("  mine 的 layout = ");
    print(mine.layout());
    printf("\n");

    // ---- TODO ----
    const char* my_frag = "";          // make_fragment_like(mine) 的 layout
    const int my_frag_size = TODO_INT;   // size
    const int my_frag_cosize = TODO_INT; // cosize
    const int my_mine_cosize = TODO_INT; // mine 的 cosize
    // --------------

    auto frag = make_fragment_like(mine);
    char b[64];
    to_str(frag.layout(), b, sizeof(b));

    if (my_frag[0] == '\0' && my_frag_size == TODO_INT) {
        printf("    (未填写，跳过)\n");
        printf("    参考: make_fragment_like = %s\n", b);
        printf("          frag size=%d cosize=%d ; mine size=%d cosize=%d\n", int(size(frag)),
               int(cosize(frag.layout())), int(size(mine)), int(cosize(mine.layout())));
        return;
    }

    if (my_frag[0]) CHECK(strcmp(my_frag, b) == 0, "frag layout: 预测 %s 实际 %s", my_frag, b);
    if (my_frag_size != TODO_INT)
        CHECK(my_frag_size == int(size(frag)), "frag size: 你的 %d 实际 %d", my_frag_size,
              int(size(frag)));
    if (my_frag_cosize != TODO_INT)
        CHECK(my_frag_cosize == int(cosize(frag.layout())), "frag cosize: 你的 %d 实际 %d", my_frag_cosize,
              int(cosize(frag.layout())));
    if (my_mine_cosize != TODO_INT)
        CHECK(my_mine_cosize == int(cosize(mine.layout())), "mine cosize: 你的 %d 实际 %d", my_mine_cosize,
              int(cosize(mine.layout())));
    printf("    想一想: cosize 差距说明了什么? (提示: 谁指向大数组, 谁是独立小块)\n");
}

// ============================================================
// 练习 7 — 改 capstone                                   ★★★
// ============================================================
void ex7() {
    banner("练习 7: 改 capstone (在 capstone 文件里做)");
    printf("  1. 跑 ../cute_tensor_capstone 记录四个版本耗时。\n");
    printf("  2. 把 M/N 从 (4096,512) 改成 (512,4096) 重跑。\n");
    printf("     哪个版本受影响最大? 为什么?\n");
    printf("     线索: v2/v3 把 x 放进 smem, x 的长度就是 N。\n");
    printf("  3. N=4096 时 smem 需要 4096*4 = 16KB。\n");
    printf("     H100/H200 每 block 上限约 228KB —— 够用吗?\n");
    printf("     那 N = 65536 呢? 算一下, 并想想该怎么改 (提示: 分段加载)。\n");
}

int main(int argc, char** argv) {
    void (*all[])() = {ex1, ex2, ex3, ex4, ex5, ex6, ex7};
    const int n = sizeof(all) / sizeof(all[0]);

    if (argc > 1) {
        int w = atoi(argv[1]);
        if (w < 1 || w > n) {
            printf("用法: %s [1-%d]\n", argv[0], n);
            return 1;
        }
        all[w - 1]();
    } else {
        for (int i = 0; i < n; ++i) all[i]();
    }

    printf("\n===== 结果: %d PASS, %d FAIL =====\n", g_pass, g_fail);
    if (g_pass == 0 && g_fail == 0) printf("还没填答案 —— 打开 ex.cu 找 TODO。\n");
    return g_fail == 0 ? 0 : 1;
}
