// Section 01 练习框架
//
// 用法:
//   make ex && ./ex          跑全部
//   ./ex 3                   只跑第 3 题
//
// 每题只需填 TODO 处。框架会自动检查并打印 PASS/FAIL。
// 题目描述见 ../README.md 的"练习"一节。
//
// 全部做完后对照 solutions.md。

#include <unistd.h>  // dup / dup2 / close，用于把 print() 的输出抓成字符串

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cute/tensor.hpp>

using namespace cute;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, fmt, ...)                                       \
    do {                                                            \
        if (cond) {                                                 \
            ++g_pass;                                               \
            printf("    [PASS] " fmt "\n", ##__VA_ARGS__);          \
        } else {                                                    \
            ++g_fail;                                               \
            printf("    [FAIL] " fmt "\n", ##__VA_ARGS__);          \
        }                                                           \
    } while (0)

#define TODO_INT (-999999)  // 未填写的哨兵值

static void banner(const char* s) { printf("\n===== %s =====\n", s); }

// 把 cute::print(layout) 的输出抓成字符串，用于和你的文字预测逐字比较。
// 做法是临时把 stdout 重定向到一个临时文件，print 完再还原。
template <class Layout>
static void layout_to_str(Layout const& layout, char* buf, size_t n) {
    buf[0] = '\0';
    FILE* tmp = tmpfile();
    if (!tmp) return;
    int saved = dup(fileno(stdout));
    fflush(stdout);
    dup2(fileno(tmp), fileno(stdout));
    print(layout);
    fflush(stdout);
    dup2(saved, fileno(stdout));
    close(saved);
    rewind(tmp);
    size_t got = fread(buf, 1, n - 1, tmp);
    buf[got] = '\0';
    fclose(tmp);
}

// ============================================================
// 练习 1 — colex 手算                                    ★☆☆
// ============================================================
// L = (3,5):(5,1)
// 先在纸上算出三个答案，填进下面三个常量，再运行验证。
// 注意第 2 题：L(7) 要先把 7 按 colex 展开成坐标 —— 见 README §2。
void ex1() {
    banner("练习 1: colex 手算");

    auto L = make_layout(make_shape(Int<3>{}, Int<5>{}), make_stride(Int<5>{}, Int<1>{}));
    printf("  L = ");
    print(L);
    printf("\n");

    // ---- TODO: 填入你手算的答案 ----
    const int my_L_2_4  = 14;  // L(2, 4) = ?
    const int my_L_7    = 3;  // L(7)    = ?
    const int my_size   = 15;  // size(L)   = ?
    const int my_cosize = 15;  // cosize(L) = ?
    // --------------------------------

    if (my_L_2_4 == TODO_INT) {
        printf("    (未填写，跳过)\n");
        return;
    }

    CHECK(my_L_2_4 == int(L(2, 4)), "L(2,4): 你的答案 %d, 实际 %d", my_L_2_4, int(L(2, 4)));
    CHECK(my_L_7 == int(L(7)), "L(7):   你的答案 %d, 实际 %d", my_L_7, int(L(7)));
    CHECK(my_size == int(size(L)), "size:   你的答案 %d, 实际 %d", my_size, int(size(L)));
    CHECK(my_cosize == int(cosize(L)), "cosize: 你的答案 %d, 实际 %d", my_cosize, int(cosize(L)));

    // 辅助：把 colex 展开打出来，方便你核对思路
    printf("  参考 - colex 展开 (shape=(3,5) -> k%%3, k/3):\n");
    for (int k = 0; k < 9; ++k) {
        auto c = idx2crd(k, shape(L));
        printf("    k=%d -> (%d,%d) -> L(k)=%2d\n", k, int(get<0>(c)), int(get<1>(c)), int(L(k)));
    }
}

// ============================================================
// 练习 2 — 三步法设计 B                                  ★★☆
// ============================================================
// A = (6,8):(8,1)，取 row 2-3 / col 4-7 的 2x4 子块。
// 按 README §3.2 的三步法:
//   step1  写出 A 的逻辑基 L_0, L_1   (注意 A 有 6 行!)
//   step2  B.shape  = (取几个, 取几个)
//          B.stride = (row步进 * L_0, col步进 * L_1)
//   step3  base = A(起点坐标)
void ex2() {
    banner("练习 2: 三步法设计 B");

    auto A = make_layout(make_shape(Int<6>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
    printf("  A = ");
    print(A);
    printf("\n  目标: row 2-3 x col 4-7 (2x4 子块)\n");

    // ---- TODO step1: A 的逻辑基 ----
    const int L0 = 1;  // L_0 = ?
    const int L1 = 6;  // L_1 = ?  (提示: = shape_0)
    // ---- TODO step2: B 的 shape 与 stride ----
    const int b_shape_0  = 2;
    const int b_shape_1  = 4;
    const int b_stride_0 = 1;  // row 步进 1 -> 1 * L_0
    const int b_stride_1 = 6;  // col 步进 1 -> 1 * L_1
    // ---- TODO step3: 起点偏移 ----
    const int my_base = 20;     // = A(2, 4)
    // --------------------------------------------

    if (L0 == TODO_INT || b_shape_0 == TODO_INT || my_base == TODO_INT) {
        printf("    (未填写，跳过)\n");
        return;
    }

    CHECK(L0 == 1 && L1 == 6, "逻辑基: 你的 L=(%d,%d), 应为 (1,6)", L0, L1);
    CHECK(my_base == int(A(2, 4)), "base: 你的 %d, A(2,4)=%d", my_base, int(A(2, 4)));

    auto B = make_layout(make_shape(b_shape_0, b_shape_1), make_stride(b_stride_0, b_stride_1));
    auto AB = composition(A, B);
    printf("  B    = ");
    print(B);
    printf("\n  A∘B  = ");
    print(AB);
    printf("\n");

    bool ok = (b_shape_0 == 2 && b_shape_1 == 4);
    for (int j = 0; j < 4 && ok; ++j)
        for (int i = 0; i < 2 && ok; ++i) {
            int got = my_base + int(AB(i, j));
            int want = int(A(2 + i, 4 + j));
            if (got != want) {
                ok = false;
                printf("    (%d,%d): got %d want %d\n", i, j, got, want);
            }
        }
    CHECK(ok, "base + (A∘B)(i,j) == A(2+i, 4+j) 全部成立");
}

// ============================================================
// 练习 3 — 反向算 A∘B                                    ★★☆
// ============================================================
// A = (4,8):(8,1)，L = (1,4)。
// 用 README §3.3 的单 mode 规则手算，填入预测字符串（格式如 "_2:_1"）。
// 留空跳过该小题。注意其中有一个会跨界分裂！
void ex3() {
    banner("练习 3: 反向算 A∘B");

    auto A = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
    printf("  A = ");
    print(A);
    printf("   L = (1, 4)\n");
    printf("  规则: d = delta * L_m => stride = delta * A_stride[m]\n");
    printf("  前提: delta * s <= shape_m，否则跨界分裂\n\n");

    // ---- TODO: 填入预测 ----
    const char* expect[4] = {
        "20",  // A o (2:4)
        "34",  // A o (4:2)    <- 小心，检查 delta*s 是否越界
        "24",  // A o (2:8)
        "",  // A o ((2,4):(4,1))   多 mode，逐 mode 独立算
    };
    // ------------------------

    char got[4][64];
    layout_to_str(composition(A, make_layout(Int<2>{}, Int<4>{})), got[0], sizeof(got[0]));
    layout_to_str(composition(A, make_layout(Int<4>{}, Int<2>{})), got[1], sizeof(got[1]));
    layout_to_str(composition(A, make_layout(Int<2>{}, Int<8>{})), got[2], sizeof(got[2]));
    layout_to_str(composition(A, make_layout(make_shape(Int<2>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{}))),
                  got[3], sizeof(got[3]));

    const char* desc[4] = {"A o (2:4)", "A o (4:2)", "A o (2:8)", "A o ((2,4):(4,1))"};
    for (int i = 0; i < 4; ++i) {
        if (expect[i][0] == '\0') {
            printf("    %-20s -> %-22s (未预测)\n", desc[i], got[i]);
        } else {
            CHECK(strcmp(expect[i], got[i]) == 0, "%-20s 预测 %-14s 实际 %s", desc[i], expect[i],
                  got[i]);
        }
    }

    // 后半题: 反过来用三步法设计 B —— 取偶数行 (row 0,2) 的全部 8 列
    printf("\n  后半题: 取 A 的偶数行 (row 0,2) x 全部 8 列\n");

    // ---- TODO: 填 B 的 shape 与 stride ----
    const int s0 = TODO_INT, s1 = TODO_INT;
    const int d0 = TODO_INT;  // row 步进 2 -> 2 * L_0
    const int d1 = TODO_INT;  // col 步进 1 -> 1 * L_1
    // ---------------------------------------

    if (s0 == TODO_INT) {
        printf("    (未填写，跳过)\n");
        return;
    }

    auto B2 = make_layout(make_shape(s0, s1), make_stride(d0, d1));
    auto R = composition(A, B2);
    printf("  B = ");
    print(B2);
    printf("   A∘B = ");
    print(R);
    printf("\n");

    bool ok = (s0 == 2 && s1 == 8);
    for (int j = 0; j < 8 && ok; ++j)
        for (int i = 0; i < 2 && ok; ++i)
            if (int(R(i, j)) != int(A(2 * i, j))) {
                ok = false;
                printf("    (%d,%d): got %d want %d\n", i, j, int(R(i, j)), int(A(2 * i, j)));
            }
    CHECK(ok, "A∘B(i,j) == A(2i, j) 全部成立");
}

// ============================================================
// 练习 4 — 预测 coalesce                                 ★★☆
// ============================================================
// 先把预测填进 expect[] 字符串（照 CuTe 的打印格式，如 "_8:_1"
// 或 "(_2,_4):(_4,_1)"），运行后框架会跟实际输出逐字比较。
void ex4() {
    banner("练习 4: 预测 coalesce");

    // ---- TODO: 填入你的预测；留空字符串表示跳过该小题 ----
    const char* expect[5] = {
        "",  // (2,4):(4,1)
        "",  // (2,4):(1,2)
        "",  // (4,2):(1,4)
        "",  // (2,3,4):(1,2,6)
        "",  // (2,3,4):(1,2,8)
    };
    // ------------------------------------------------------

    auto l0 = make_layout(make_shape(Int<2>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
    auto l1 = make_layout(make_shape(Int<2>{}, Int<4>{}), make_stride(Int<1>{}, Int<2>{}));
    auto l2 = make_layout(make_shape(Int<4>{}, Int<2>{}), make_stride(Int<1>{}, Int<4>{}));
    auto l3 = make_layout(make_shape(Int<2>{}, Int<3>{}, Int<4>{}),
                          make_stride(Int<1>{}, Int<2>{}, Int<6>{}));
    auto l4 = make_layout(make_shape(Int<2>{}, Int<3>{}, Int<4>{}),
                          make_stride(Int<1>{}, Int<2>{}, Int<8>{}));

    const char* desc[5] = {"(2,4):(4,1)", "(2,4):(1,2)", "(4,2):(1,4)", "(2,3,4):(1,2,6)",
                           "(2,3,4):(1,2,8)"};

    char got[5][64];
    layout_to_str(coalesce(l0), got[0], sizeof(got[0]));
    layout_to_str(coalesce(l1), got[1], sizeof(got[1]));
    layout_to_str(coalesce(l2), got[2], sizeof(got[2]));
    layout_to_str(coalesce(l3), got[3], sizeof(got[3]));
    layout_to_str(coalesce(l4), got[4], sizeof(got[4]));

    for (int i = 0; i < 5; ++i) {
        if (expect[i][0] == '\0') {
            printf("    (%d) %-18s -> %-22s (未预测)\n", i + 1, desc[i], got[i]);
            continue;
        }
        CHECK(strcmp(expect[i], got[i]) == 0, "(%d) %-18s 预测 %-16s 实际 %s", i + 1, desc[i],
              expect[i], got[i]);
    }
    printf("  想一想: 第 4 与第 5 只差一个 stride，为什么结果不同?\n");
}

// ============================================================
// 练习 5 — complement 的含义                             ★★★
// ============================================================
void ex5() {
    banner("练习 5: complement");

    // ---- TODO: 预测三个 complement 的结果（格式如 "_4:_4"），留空跳过 ----
    const char* expect[3] = {
        "",  // complement(4:1, 16)
        "",  // complement(4:4, 16)
        "",  // complement(2:1, 16)
    };
    // ---------------------------------------------------------------------

    auto a = make_layout(Int<4>{}, Int<1>{});
    auto b = make_layout(Int<4>{}, Int<4>{});
    auto c = make_layout(Int<2>{}, Int<1>{});

    char got[3][64];
    layout_to_str(complement(a, Int<16>{}), got[0], sizeof(got[0]));
    layout_to_str(complement(b, Int<16>{}), got[1], sizeof(got[1]));
    layout_to_str(complement(c, Int<16>{}), got[2], sizeof(got[2]));

    const char* desc[3] = {"complement(4:1, 16)", "complement(4:4, 16)", "complement(2:1, 16)"};
    for (int i = 0; i < 3; ++i) {
        if (expect[i][0] == '\0') {
            printf("    %-22s -> %-12s (未预测)\n", desc[i], got[i]);
        } else {
            CHECK(strcmp(expect[i], got[i]) == 0, "%-22s 预测 %-10s 实际 %s", desc[i], expect[i],
                  got[i]);
        }
    }

    // 第 2 题的后半：4:4 与它的补集拼起来，把 0-15 怎么分给 4 个"线程"?
    printf("\n  4:4 与补集组合后的完整 layout:\n");
    auto full = make_layout(b, complement(b, Int<16>{}));
    printf("    ");
    print(full);
    printf("\n");
    for (int t = 0; t < 4; ++t) {
        printf("    thread %d 拿到: ", t);
        for (int v = 0; v < int(size<1>(full)); ++v) printf("%2d ", int(full(t, v)));
        printf("\n");
    }
    printf("  对照你的预期: 是否不重不漏地覆盖 0-15?\n");
}

// ============================================================
// 练习 6 — 嵌套 mode                                     ★★★
// ============================================================
void ex6() {
    banner("练习 6: 嵌套 mode");

    auto A = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
    auto AN = composition(A, make_layout(make_shape(Int<8>{}, Int<4>{}),
                                         make_stride(Int<1>{}, Int<8>{})));
    printf("  AN = ");
    print(AN);
    printf("\n");

    // ---- TODO (1): 层次查询 ----
    const int my_rank   = TODO_INT;
    const int my_size   = TODO_INT;
    const int my_size0  = TODO_INT;  // size<0>(AN)
    // ---- TODO (2): AN(6,1) 的嵌套坐标 ----
    // mode0 内部按 colex 展平: flat = i0 + 4*i1，反推 flat=6 时 i0,i1 = ?
    const int my_i0 = TODO_INT;
    const int my_i1 = TODO_INT;
    // ---- TODO (3): coalesce(AN) 的预测（格式如 "(_4,_8):(_8,_1)"）----
    const char* my_coalesce = "";
    // ---- TODO (4): composition(A, (2,16):(4,1)) 会产生嵌套吗？----
    // 填 1 表示会，0 表示不会，TODO_INT 跳过
    const int will_nest = TODO_INT;
    // -----------------------------------------------------------

    if (my_rank != TODO_INT) {
        CHECK(my_rank == int(rank(AN)), "rank: 你的 %d, 实际 %d", my_rank, int(rank(AN)));
        CHECK(my_size == int(size(AN)), "size: 你的 %d, 实际 %d", my_size, int(size(AN)));
        CHECK(my_size0 == int(size<0>(AN)), "size<0>: 你的 %d, 实际 %d", my_size0,
              int(size<0>(AN)));
    } else {
        printf("    (1) 未填写 —— 参考: rank=%d size=%d size<0>=%d\n", int(rank(AN)),
               int(size(AN)), int(size<0>(AN)));
    }

    if (my_i0 != TODO_INT) {
        int want = int(AN(6, 1));
        int got = int(AN(make_coord(my_i0, my_i1), 1));
        CHECK(got == want, "AN((%d,%d),1)=%d 应等于 AN(6,1)=%d", my_i0, my_i1, got, want);
    } else {
        printf("    (2) 未填写 —— AN(6,1) = %d\n", int(AN(6, 1)));
    }

    char buf[64];
    layout_to_str(coalesce(AN), buf, sizeof(buf));
    if (my_coalesce[0] != '\0') {
        CHECK(strcmp(my_coalesce, buf) == 0, "coalesce(AN): 预测 %s 实际 %s", my_coalesce, buf);
    } else {
        printf("    (3) 未预测 —— coalesce(AN) = %s\n", buf);
    }

    auto A2 = composition(A, make_layout(make_shape(Int<2>{}, Int<16>{}),
                                         make_stride(Int<4>{}, Int<1>{})));
    char buf2[64];
    layout_to_str(A2, buf2, sizeof(buf2));
    const int actually_nests = (rank<1>(A2) > 1) ? 1 : 0;
    if (will_nest != TODO_INT) {
        CHECK(will_nest == actually_nests, "(2,16):(4,1) 是否嵌套: 你答 %s, 实际 %s (%s)",
              will_nest ? "会" : "不会", actually_nests ? "会" : "不会", buf2);
    } else {
        printf("    (4) 未填写 —— composition(A,(2,16):(4,1)) = %s\n", buf2);
    }
}

// ============================================================
// 练习 7 — 改 capstone 的 PADDING                        ★★★
// ============================================================
// 这题在 ../cute_layout_capstone.cu 里做，本函数只给检查清单。
void ex7() {
    banner("练习 7: capstone PADDING (在 capstone 文件里做)");
    printf("  步骤:\n");
    printf("    1. 编辑 ../cute_layout_capstone.cu，把 transpose_tiled_padded 的\n");
    printf("       constexpr int PADDING = 1; 依次改成 2 和 4，各跑一次记录带宽。\n");
    printf("    2. 回答: 哪个最快? 为什么 PADDING=1 通常就够?\n");
    printf("       线索: smem 32 个 bank x 4 字节; int 是 4 字节;\n");
    printf("       转置读 smem 时 32 个线程的地址间隔 = stride 个元素。\n");
    printf("       stride=32 -> 全部落到同一个 bank (32-way conflict)\n");
    printf("       stride=33 -> 相邻线程错开 1 个 bank -> 无 conflict\n");
    printf("    3. 换成 double (8 字节) 后 PADDING=1 还够吗? 先推理再验证。\n");
}

int main(int argc, char** argv) {
    void (*all[])() = {ex1, ex2, ex3, ex4, ex5, ex6, ex7};
    const int n = sizeof(all) / sizeof(all[0]);

    if (argc > 1) {
        int which = atoi(argv[1]);
        if (which < 1 || which > n) {
            printf("用法: %s [1-%d]\n", argv[0], n);
            return 1;
        }
        all[which - 1]();
    } else {
        for (int i = 0; i < n; ++i) all[i]();
    }

    printf("\n===== 结果: %d PASS, %d FAIL =====\n", g_pass, g_fail);
    if (g_pass == 0 && g_fail == 0) printf("还没填答案 —— 打开 ex.cu 找 TODO。\n");
    return g_fail == 0 ? 0 : 1;
}
