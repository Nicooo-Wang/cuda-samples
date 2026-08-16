// v1: make_tiled_copy —— 把一个 atom 铺到一整个 thread block 上
//
// 讲解见本目录 README.md 的 §4 ~ §7。
//
// 学习目标：
//   1. host 描述 / kernel 索引 —— CuTe 的标准分工          (README §4)
//   2. make_tiled_copy(atom, thr_layout, val_layout)       (README §5)
//   3. partition_S / partition_D 的返回形状                (README §6)
//   4. thr_layout 写错 -> 访存不合并（静默变慢）           (README §7)
//   5. 向量化真正要求的是"连续那一维的 stride 静态"        (README §7.3)
//
// 核心概念：
//   - TiledCopy   : atom + "谁搬哪一份"的分工表
//   - Tiler_MN    : 这个 TiledCopy 一次覆盖多大一块
//   - get_slice(t): 取出线程 t 的那一份
//   - partition_S : 把 Tensor 按分工表切给当前线程

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 通用的 tiled copy kernel。
//
// 注意参数：进来的是 Tensor 和 TiledCopy —— 两个都在 host 上构造好了。
// kernel 里只做一件事：把"整块"按分工表切成"我这一份"，然后搬。
//
// 这是 CuTe 的标准分工（CUTLASS 官方 examples 全都如此，见 README §4）：
//   host   : 描述数据长什么样、谁搬哪一份  (layout / atom / TiledCopy / Tensor)
//   kernel : 用 blockIdx / threadIdx 去索引 (local_tile / get_slice / partition)
//
// 之所以敢这样按值传，是因为静态 layout 是**空类型**（sizeof == 1）：
// 传 TiledCopy 一个字节都没传，传 Tensor 只传了里面那个指针。§1 会打印出来。
// ---------------------------------------------------------------------------
template <class TensorS, class TensorD, class TiledCopy>
__global__ void tiled_copy_kernel(TensorS S, TensorD D, TiledCopy tc) {
    auto thr = tc.get_slice(threadIdx.x);  // 取出"我"这一份
    copy(tc, thr.partition_S(S), thr.partition_D(D));
}

// 报告每个线程实际碰到的地址，用来观察合并访存
template <class TensorS, class TiledCopy>
__global__ void report_kernel(TensorS S, TiledCopy tc, int* out) {
    auto p = tc.get_slice(threadIdx.x).partition_S(S);

    if (threadIdx.x < 8) {
        out[threadIdx.x * 8] = int(size(p));
        for (int i = 0; i < int(size(p)) && i < 7; ++i)
            out[threadIdx.x * 8 + 1 + i] = int(&p(i) - &S(0));
    }
}

template <class TensorS, class TiledCopy>
void show_addresses(const char* title, TensorS S, TiledCopy tc, int* d_out, int nthr) {
    int h[64] = {};
    report_kernel<<<1, nthr>>>(S, tc, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));

    printf("%s\n", title);
    for (int t = 0; t < 5 && t < nthr; ++t) {
        printf("  thr%-2d 拿到 %d 个元素, 偏移 =", t, h[t * 8]);
        for (int i = 0; i < h[t * 8] && i < 7; ++i) printf(" %3d", h[t * 8 + 1 + i]);
        printf("\n");
    }
}

// 打印一个类型占多少字节，以及它是不是空类型
template <class T>
void show_size(const char* name, T const&) {
    printf("  %-44s sizeof = %-3zu  %s\n", name, sizeof(T),
           is_empty<T>::value ? "<- 空类型，传参不占空间" : "");
}

int main() {
    print_separator("make_tiled_copy: 从一个 atom 到一个 block");

    constexpr int M = 8, N = 16, NE = M * N;
    float* h_src = new float[NE];
    float* h_dst = new float[NE];
    for (int i = 0; i < NE; ++i) h_src[i] = float(i);

    float *d_src, *d_dst;
    int* d_out;
    CUDA_CHECK(cudaMalloc(&d_src, NE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, NE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, 64 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, NE * sizeof(float), cudaMemcpyHostToDevice));

    auto verify = [&](const char* name) {
        CUDA_CHECK(cudaMemcpy(h_dst, d_dst, NE * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < NE; ++i)
            if (h_dst[i] != h_src[i]) ++bad;
        printf("  %s: %s\n", name, bad == 0 ? "8x16 全部搬到位" : "有元素没搬到");
    };

    // ========== 1. host 描述, kernel 索引 ==========
    print_separator("1. 谁在 host 上做, 谁在 kernel 里做");

    // layout / Tensor / TiledCopy 全都在 host 上构造。
    // 注意 make_gmem_ptr 包的是 device 指针 —— host 上只是"描述"它，从不解引用。
    auto lay = make_layout(make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));
    auto S = make_tensor(make_gmem_ptr(d_src), lay);
    auto D = make_tensor(make_gmem_ptr(d_dst), lay);

    printf("Tensor 的 layout = ");
    print(lay);
    printf("\n\n为什么敢把这些按值传进 kernel —— 看它们占多少字节:\n");
    show_size("Layout<Shape<_8,_16>,Stride<_16,_1>>", lay);
    show_size("Copy_Atom<UniversalCopy<float>, float>", Copy_Atom<UniversalCopy<float>, float>{});
    show_size("Tensor<gmem_ptr<float>, 静态 layout>", S);

    int M_rt = M, N_rt = N;  // 故意用运行时 int, 做对比
    auto lay_all_dyn = make_layout(make_shape(M_rt, N_rt), make_stride(N_rt, 1));
    auto S_all_dyn = make_tensor(make_gmem_ptr(d_src), lay_all_dyn);
    printf("\n换成运行时的 shape/stride 就要真的占空间了:\n");
    show_size("Layout<Shape<int,int>,Stride<int,int>>", lay_all_dyn);
    show_size("Tensor<gmem_ptr<float>, 动态 layout>", S_all_dyn);

    printf("\n=> 静态 layout 的形状信息全在**类型**里, 运行时零字节。\n");
    printf("=> 所以 CuTe 的惯例是: host 描述 (layout/atom/TiledCopy/Tensor),\n");
    printf("   kernel 只用 blockIdx/threadIdx 做索引。CUTLASS 官方 examples 全是这个写法。\n");

    // ========== 2. 标量版: 32 线程 x 1 值 ==========
    print_separator("2. 标量 TiledCopy (32 thr x 1 val)");

    // make_tiled_copy 要三样东西：
    //   atom       : 一次原子操作搬多少 (v0 讲过)
    //   thr_layout : 线程怎么排 —— 决定合并访存
    //   val_layout : 每个线程一次拿几个、怎么排
    auto atom_scalar = Copy_Atom<UniversalCopy<float>, float>{};
    auto thr_rm = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
    auto val_1 = make_layout(make_shape(Int<1>{}, Int<1>{}));
    auto tc_scalar = make_tiled_copy(atom_scalar, thr_rm, val_1);

    printf("thr_layout = ");
    print(thr_rm);
    printf("   <- 8x4 = 32 个线程，row-major\n");
    printf("val_layout = ");
    print(val_1);
    printf("   <- 每线程 1 个值\n");
    printf("Tiler_MN   = ");
    print(typename decltype(tc_scalar)::Tiler_MN{});
    printf("   <- 一次覆盖 8x4 一块\n");
    printf("线程数     = %d\n", int(size(tc_scalar)));

    CUDA_CHECK(cudaMemset(d_dst, 0xff, NE * sizeof(float)));
    tiled_copy_kernel<<<1, int(size(tc_scalar))>>>(S, D, tc_scalar);
    CUDA_CHECK(cudaDeviceSynchronize());
    verify("标量版");

    printf("\n注意 Tiler_MN 只有 8x4，而 Tensor 是 8x16。\n");
    printf("partition_S 返回 rank-3: ((atom内),rest_m,rest_n)，copy() 会自动遍历 rest。\n");
    show_addresses("每个线程实际碰到的偏移:", S, tc_scalar, d_out, int(size(tc_scalar)));
    printf("  => 相邻线程拿相邻地址(0,1,2,3)，一个 warp 合并成整齐的 32B 事务\n");

    // ========== 3. 向量版: 32 线程 x 4 值 ==========
    print_separator("3. 向量 TiledCopy (32 thr x 4 val)");

    // 换两处：atom 变成 128bit，val_layout 在列方向给 4 个。
    // 列方向是连续方向，所以这 4 个刚好能凑成一条 LDG.E.128。
    auto atom_vec = Copy_Atom<UniversalCopy<uint128_t>, float>{};
    auto val_4 = make_layout(make_shape(Int<1>{}, Int<4>{}));
    auto tc_vec = make_tiled_copy(atom_vec, thr_rm, val_4);

    printf("atom       = UniversalCopy<uint128_t>  (NumVal=4)\n");
    printf("val_layout = ");
    print(val_4);
    printf("   <- 每线程 4 个值，放在连续方向\n");
    printf("Tiler_MN   = ");
    print(typename decltype(tc_vec)::Tiler_MN{});
    printf("   <- 覆盖面积变成 8x16 (4 倍)\n");
    printf("线程数     = %d   <- 没变，但每人搬 4 个\n", int(size(tc_vec)));

    CUDA_CHECK(cudaMemset(d_dst, 0xff, NE * sizeof(float)));
    tiled_copy_kernel<<<1, int(size(tc_vec))>>>(S, D, tc_vec);
    CUDA_CHECK(cudaDeviceSynchronize());
    verify("向量版");

    show_addresses("\n每个线程实际碰到的偏移:", S, tc_vec, d_out, int(size(tc_vec)));
    printf("  => 每个线程 4 个连续地址 -> 一条 128bit 指令\n");
    printf("  => 线程间隔 4 -> warp 内仍然是一整段连续区间\n");

    // ========== 4. thr_layout 写错的后果 ==========
    print_separator("4. thr_layout 写错 -> 访存不合并");

    // 只把 thr_layout 从 row-major 改成 col-major，其他不动。
    // 结果依然正确，但访存模式塌了 —— 这类 bug 不会报错，只会变慢。
    auto thr_cm = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<1>{}, Int<8>{}));
    auto tc_bad = make_tiled_copy(atom_scalar, thr_cm, val_1);

    printf("thr_layout = ");
    print(thr_cm);
    printf("   <- col-major (故意写错)\n");

    CUDA_CHECK(cudaMemset(d_dst, 0xff, NE * sizeof(float)));
    tiled_copy_kernel<<<1, int(size(tc_bad))>>>(S, D, tc_bad);
    CUDA_CHECK(cudaDeviceSynchronize());
    verify("col-major thr_layout");

    show_addresses("\n每个线程实际碰到的偏移:", S, tc_bad, d_out, int(size(tc_bad)));
    printf("  => 相邻线程地址相差 %d (一整行)!\n", N);
    printf("  => 32 个线程散落在 32 个不同的 32B 段 -> 访存事务数暴涨\n");
    printf("  => 结果依然正确，所以这类 bug 只能靠看 layout 或 profiler 发现\n");

    // ========== 5. 向量化到底要求什么 ==========
    print_separator("5. 向量化要求的是 stride 静态, 不是 shape 静态");

    // 常见误解: "要向量化就得把整个 layout 写成编译期的"。
    // 实际只要求**被向量化那一维的 stride** 是静态的, shape 可以是运行时值。
    //
    // 下面这个 Tensor 的 shape 全是运行时 int, 只有末维 stride 是 Int<1>{},
    // 128bit atom 照样能用 —— 直接跑一遍证明。
    auto lay_dyn_shape = make_layout(make_shape(M_rt, N_rt), make_stride(N_rt, Int<1>{}));
    auto S_dyn = make_tensor(make_gmem_ptr(d_src), lay_dyn_shape);
    auto D_dyn = make_tensor(make_gmem_ptr(d_dst), lay_dyn_shape);

    printf("layout = ");
    print(lay_dyn_shape);
    printf("   <- shape 是运行时的, 只有末维 stride 是 _1\n");

    CUDA_CHECK(cudaMemset(d_dst, 0xff, NE * sizeof(float)));
    tiled_copy_kernel<<<1, int(size(tc_vec))>>>(S_dyn, D_dyn, tc_vec);
    CUDA_CHECK(cudaDeviceSynchronize());
    verify("动态 shape + 128bit atom");

    printf("\n三种写法, 用同一个 128bit 的 TiledCopy 沿列方向向量化:\n");
    printf("  %-46s %s\n", "layout 写法", "结果");
    printf("  %-46s %s\n", "make_stride(N, 1)          全动态", "编译失败");
    printf("  %-46s %s\n", "make_stride(N, Int<1>{})   末维 stride 静态", "通过 (刚跑过)");
    printf("  %-46s %s\n", "make_stride(Int<N>{}, Int<1>{}) 全静态", "通过 (§3 跑过)");
    printf("\n编译失败时的报错是:\n");
    printf("  \"Copy_Traits: src failed to vectorize into registers.\"\n");
    printf("原因: CuTe 要在编译期证明这 4 个元素相邻, 靠的是那一维的 stride == _1。\n");
    printf("shape 是多少它不关心 —— 官方 tiled_copy.cu 就用运行时 shape 做 128bit 搬运,\n");
    printf("因为默认 LayoutLeft 的第 0 维 stride 恰好是静态的 _1。\n");
    printf("=> 实践建议: 尺寸该动态就动态, 但**连续那一维的 stride 一定写成 Int<1>{}**。\n");

    // ========== 6. 总结 ==========
    print_separator("总结");
    printf("本程序演示的内容 (讲解见 README):\n");
    printf("  §1  host 描述 / kernel 索引, 以及静态 layout 是空类型  -> README §4\n");
    printf("  §2  标量 TiledCopy + partition 的 rank-3 形状          -> README §5 §6\n");
    printf("  §3  向量 TiledCopy: val_layout 放在连续方向            -> README §5.2\n");
    printf("  §4  thr_layout 写错 -> 静默不合并                      -> README §7\n");
    printf("  §5  向量化要求连续维 stride 静态                       -> README §7.3\n");
    printf("\n下一步: capstone 会用这些做一个 gmem->smem->gmem 的 memcpy\n");

    delete[] h_src;
    delete[] h_dst;
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
