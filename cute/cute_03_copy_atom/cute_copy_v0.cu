// v0: Copy_Atom 是什么
//
// 讲解见本目录 README.md 的 §1 ~ §3。本文件只是那些内容的可执行版本：
// 每个小节的编号（§1a, §1b, ...）都在 README 里有对应的图和推导。
//
// 学习目标：
//   1. copy(S, D) 最朴素的用法             (README §1)
//   2. Copy_Atom 的三件套 ThrID/ValLayout  (README §2)
//   3. 向量宽度从哪来: UniversalCopy<T>    (README §2.2)
//   4. max_common_vector: 能不能向量化     (README §3)
//
// 核心概念：
//   - copy(S, D)          : 逐元素搬，CuTe 自己挑指令
//   - Copy_Atom<Op, T>    : 一次不可分割的搬运操作
//   - ThrID               : 这个 atom 需要几个线程协作
//   - NumValSrc/Dst       : 每个线程搬几个 T
//   - max_common_vector   : src/dst 共同的最大连续长度

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include "common.h"

using namespace cute;

int main() {
    print_separator("CuTe Copy Atom: 最小的搬运单元");

    // ========== 1. 最朴素的 copy ==========
    print_separator("1. copy(S, D)");

    // Copy 的输入输出都是 Tensor（Section 02），不是裸指针。
    // copy(S,D) 不需要你指定任何指令，CuTe 会自己挑一个。
    float src[16], dst[16];
    for (int i = 0; i < 16; ++i) {
        src[i] = float(i);
        dst[i] = -1.f;
    }

    auto lay = make_layout(Int<16>{}, Int<1>{});
    auto S = make_tensor(make_gmem_ptr(src), lay);
    auto D = make_tensor(make_gmem_ptr(dst), lay);

    printf("S = ");
    print(S.layout());
    printf("   D = ");
    print(D.layout());
    printf("\n");

    printf("copy 前 dst = ");
    for (int i = 0; i < 16; ++i) printf("%g ", dst[i]);
    printf("\n");

    copy(S, D);  // <- 就这一行

    printf("copy 后 dst = ");
    for (int i = 0; i < 16; ++i) printf("%g ", dst[i]);
    printf("\n");
    printf("=> copy(S,D) 要求 size(S) == size(D)，形状可以不同\n");

    // ========== 2. Copy_Atom 的三件套 ==========
    print_separator("2. Copy_Atom 的内部结构");

    // 一个 Copy_Atom 回答三个问题：
    //   ThrID        : 这次操作要几个线程一起干？
    //   NumValSrc    : 每个线程从 src 读几个元素？
    //   NumValDst    : 每个线程往 dst 写几个元素？
    // 只有 ThrID/NumVal 都对上，指令才能发出去。

    printf("--- 标量搬运: 1 个线程搬 1 个 float ---\n");
    print_atom<Copy_Atom<UniversalCopy<float>, float>>("Copy_Atom<UniversalCopy<float>, float>");

    printf("\n--- 向量搬运: 1 个线程搬 128 bit = 4 个 float ---\n");
    print_atom<Copy_Atom<UniversalCopy<uint128_t>, float>>(
        "Copy_Atom<UniversalCopy<uint128_t>, float>");
    printf("  => NumVal 从 1 变成 4: 一条指令搬 4 个 float (LDG.E.128)\n");
    printf("  => 向量宽度写在 UniversalCopy<> 的模板参数里，不是 atom 的第二个参数\n");

    printf("\n--- 同一个 Op，换元素类型 ---\n");
    print_atom<Copy_Atom<UniversalCopy<uint128_t>, half_t>>(
        "Copy_Atom<UniversalCopy<uint128_t>, half_t>");
    printf("  => 128 bit / 16 bit = 8 个 half\n");

    printf("\n--- AutoVectorizingCopy: 宽度交给 CuTe 推 ---\n");
    print_atom<Copy_Atom<AutoVectorizingCopy, float>>("Copy_Atom<AutoVectorizingCopy, float>");
    printf("  => NumVal=1 是\"占位\"，真实宽度在 copy() 时按 layout 现场推导\n");
    printf("  => 这也是 copy(S,D) 不写 atom 时的默认行为\n");

    // ========== 3. 显式指定 atom 的 copy ==========
    print_separator("3. 用指定的 atom 搬运");

    // 传 atom 时，src/dst 必须是 rank-1、且 size 恰好等于 NumVal。
    // 这是 Copy_Atom::call 的静态检查：形状不匹配直接编译失败（不是运行时报错）。
    printf("atom = Copy_Atom<UniversalCopy<uint128_t>, float>   NumVal = %d\n",
           Copy_Atom<UniversalCopy<uint128_t>, float>::NumValSrc);
    printf("要求: src/dst 是 rank-1，且 size 恰好 == NumVal\n");
    printf("  make_layout(Int<4>{}, Int<1>{})  size=4  -> OK\n");
    printf("  make_layout(Int<8>{}, Int<1>{})  size=8  -> 编译失败\n");

    // 这个 copy 放到 device 上做（见 v1 的 kernel）。
    // 原因很重要: UniversalCopy<uint128_t> 会把 float* 重新解释成 uint128_t*，
    // 这在 host 代码里违反 C++ 的 strict aliasing 规则 —— -O2 以上编译器
    // 会认为"写 uint128_t 不可能影响 float 数组"，从而把这次写整个优化掉。
    printf("\n注意: 宽向量 atom 属于 device 代码。\n");
    printf("  UniversalCopy<uint128_t> 会把 float* punning 成 uint128_t*，\n");
    printf("  在 host 上违反 strict aliasing: -O2 以上这次写会被优化掉（静默出错）。\n");
    printf("  => 宽向量搬运一律在 kernel 里做，v1 会给出可运行的版本。\n");

    // ========== 4. copy_if: 带谓词的搬运 ==========
    print_separator("4. copy_if: 边界处理");

    // 真实 kernel 里 M/N 很少刚好被 tile 整除，越界的那部分不能搬。
    // copy_if(pred, S, D) 只在 pred 为真的位置搬运。
    float s5[8], d5[8];
    for (int i = 0; i < 8; ++i) {
        s5[i] = float(100 + i);
        d5[i] = -1.f;
    }
    auto S5 = make_tensor(make_gmem_ptr(s5), make_layout(Int<8>{}, Int<1>{}));
    auto D5 = make_tensor(make_gmem_ptr(d5), make_layout(Int<8>{}, Int<1>{}));

    auto pred = make_tensor<bool>(make_layout(Int<8>{}, Int<1>{}));
    for (int i = 0; i < 8; ++i) pred(i) = (i % 2 == 0);

    printf("pred = ");
    for (int i = 0; i < 8; ++i) printf("%d ", int(pred(i)));
    printf("  (只搬偶数位)\n");

    copy_if(pred, S5, D5);

    printf("copy_if 后 d5 = ");
    for (int i = 0; i < 8; ++i) printf("%g ", d5[i]);
    printf("\n");
    printf("=> -1 的位置没被碰过。这是处理\"最后一个不完整 tile\"的标准手段\n");

    // ========== 5. 能不能向量化: max_common_vector ==========
    print_separator("5. max_common_vector");

    // 向量化的前提: src 和 dst 在内存里都得有一段连续、且长度一致。
    // max_common_vector(S,D) 就是这个"共同最大连续长度"。
    float* p = (float*)0x1000;  // 只看 layout，不解引用

    auto C1 = make_tensor(make_gmem_ptr(p), make_layout(Int<32>{}, Int<1>{}));
    auto C2 = make_tensor(make_gmem_ptr(p + 64), make_layout(Int<32>{}, Int<1>{}));
    auto Z  = make_tensor(make_gmem_ptr(p + 64), make_layout(Int<32>{}, Int<2>{}));

    printf("两个都连续      : max_common_vector = %2d  -> 可以用 128bit\n",
           int(max_common_vector(C1, C2)));
    printf("dst 步长为 2    : max_common_vector = %2d  -> 只能标量搬\n",
           int(max_common_vector(C1, Z)));
    printf("\nmax_common_layout(连续,连续) = ");
    print(max_common_layout(C1, C2));
    printf("\n");
    printf("=> 这就是 AutoVectorizingCopy 内部用来选宽度的判据\n");
    printf("=> 也解释了 Section 01 为什么强调 coalesce: 能塌缩成 N:1 才谈得上向量化\n");

    // ========== 6. recast: 换元素类型看同一块内存 ==========
    print_separator("6. recast");

    // 向量化的另一种写法：把 Tensor 重新解释成更宽的类型。
    auto T = make_tensor(make_gmem_ptr(p),
                         make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})));
    printf("T                 = ");
    print(T.layout());
    printf("\n");
    printf("recast<float4>(T) = ");
    print(recast<float4>(T).layout());
    printf("   <- 8 列变成 2 列，每列 4 个 float\n");
    printf("=> 最后一维必须连续且能被整除，否则 recast 编译失败\n");

    // ========== 7. 总结 ==========
    print_separator("总结");
    printf("本程序演示的内容 (讲解见 README):\n");
    printf("  §1  copy(S,D) 最朴素的用法          -> README §1\n");
    printf("  §2  Copy_Atom 三件套                -> README §2\n");
    printf("  §3  显式 atom 的形状要求 + 别在 host 上做宽向量  -> README §2.3\n");
    printf("  §4  copy_if 边界处理                -> README §2.4\n");
    printf("  §5  max_common_vector               -> README §3\n");
    printf("  §6  recast                          -> README §3.2\n");
    printf("\n下一步: v1 会把 atom 铺到多个线程上 (make_tiled_copy)\n");

    return 0;
}
