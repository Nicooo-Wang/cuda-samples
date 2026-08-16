// v1: make_tiled_copy —— 把一个 atom 铺到一整个 thread block 上
//
// 讲解见本目录 README.md 的 §4 ~ §6。
//
// 学习目标：
//   1. make_tiled_copy(atom, thr_layout, val_layout)   (README §4)
//   2. partition_S / partition_D 的返回形状             (README §5)
//   3. thr_layout 写错 -> 访存不合并（静默变慢）        (README §6)
//   4. 向量化要求编译期 stride（一个真实的坑）          (README §6.3)
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
// 通用的 tiled copy kernel。M/N 用模板参数传，是为了让 stride 成为编译期常量
// —— 向量化 atom 需要这一点（见 §6.3 和 README §6.3）。
// ---------------------------------------------------------------------------
template <int M, int N, class TiledCopy>
__global__ void tiled_copy_kernel(const float* src, float* dst, TiledCopy tc) {
    auto lay = make_layout(make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));
    auto S = make_tensor(make_gmem_ptr(src), lay);
    auto D = make_tensor(make_gmem_ptr(dst), lay);

    auto thr = tc.get_slice(threadIdx.x);  // 取出"我"这一份
    copy(tc, thr.partition_S(S), thr.partition_D(D));
}

// 报告每个线程实际碰到的地址，用来观察合并访存
template <int M, int N, class TiledCopy>
__global__ void report_kernel(const float* base, TiledCopy tc, int* out) {
    auto lay = make_layout(make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));
    auto S = make_tensor(make_gmem_ptr(base), lay);
    auto p = tc.get_slice(threadIdx.x).partition_S(S);

    if (threadIdx.x < 8) {
        out[threadIdx.x * 8] = int(size(p));
        for (int i = 0; i < int(size(p)) && i < 7; ++i)
            out[threadIdx.x * 8 + 1 + i] = int(&p(i) - base);
    }
}

template <int M, int N, class TiledCopy>
void show_addresses(const char* title, const float* d_src, TiledCopy tc, int* d_out, int nthr) {
    int h[64] = {};
    report_kernel<M, N><<<1, nthr>>>(d_src, tc, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));

    printf("%s\n", title);
    for (int t = 0; t < 5 && t < nthr; ++t) {
        printf("  thr%-2d 拿到 %d 个元素, 偏移 =", t, h[t * 8]);
        for (int i = 0; i < h[t * 8] && i < 7; ++i) printf(" %3d", h[t * 8 + 1 + i]);
        printf("\n");
    }
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

    // ========== 1. 标量版: 32 线程 x 1 值 ==========
    print_separator("1. 标量 TiledCopy (32 thr x 1 val)");

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
    tiled_copy_kernel<M, N><<<1, int(size(tc_scalar))>>>(d_src, d_dst, tc_scalar);
    CUDA_CHECK(cudaDeviceSynchronize());
    verify("标量版");

    printf("\n注意 Tiler_MN 只有 8x4，而 Tensor 是 8x16。\n");
    printf("partition_S 返回 rank-3: ((atom内),rest_m,rest_n)，copy() 会自动遍历 rest。\n");
    show_addresses<M, N>("每个线程实际碰到的偏移:", d_src, tc_scalar, d_out,
                         int(size(tc_scalar)));
    printf("  => 相邻线程拿相邻地址(0,1,2,3)，一个 warp 合并成整齐的 32B 事务\n");

    // ========== 2. 向量版: 32 线程 x 4 值 ==========
    print_separator("2. 向量 TiledCopy (32 thr x 4 val)");

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
    tiled_copy_kernel<M, N><<<1, int(size(tc_vec))>>>(d_src, d_dst, tc_vec);
    CUDA_CHECK(cudaDeviceSynchronize());
    verify("向量版");

    show_addresses<M, N>("\n每个线程实际碰到的偏移:", d_src, tc_vec, d_out, int(size(tc_vec)));
    printf("  => 每个线程 4 个连续地址 -> 一条 128bit 指令\n");
    printf("  => 线程间隔 4 -> warp 内仍然是一整段连续区间\n");

    // ========== 3. thr_layout 写错的后果 ==========
    print_separator("3. thr_layout 写错 -> 访存不合并");

    // 只把 thr_layout 从 row-major 改成 col-major，其他不动。
    // 结果依然正确，但访存模式塌了 —— 这类 bug 不会报错，只会变慢。
    auto thr_cm = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<1>{}, Int<8>{}));
    auto tc_bad = make_tiled_copy(atom_scalar, thr_cm, val_1);

    printf("thr_layout = ");
    print(thr_cm);
    printf("   <- col-major (故意写错)\n");

    CUDA_CHECK(cudaMemset(d_dst, 0xff, NE * sizeof(float)));
    tiled_copy_kernel<M, N><<<1, int(size(tc_bad))>>>(d_src, d_dst, tc_bad);
    CUDA_CHECK(cudaDeviceSynchronize());
    verify("col-major thr_layout");

    show_addresses<M, N>("\n每个线程实际碰到的偏移:", d_src, tc_bad, d_out, int(size(tc_bad)));
    printf("  => 相邻线程地址相差 %d (一整行)!\n", N);
    printf("  => 32 个线程散落在 32 个不同的 32B 段 -> 访存事务数暴涨\n");
    printf("  => 结果依然正确，所以这类 bug 只能靠看 layout 或 profiler 发现\n");

    // ========== 4. 向量化要求编译期 stride ==========
    print_separator("4. 一个真实的坑: 向量化需要编译期 stride");

    printf("本文件所有 kernel 都把 M/N 作为模板参数传入，于是\n");
    printf("  make_stride(Int<N>{}, Int<1>{})   <- 编译期常量\n");
    printf("如果换成运行时的 make_stride(N, 1)，向量 atom 会编译失败:\n");
    printf("  \"Copy_Traits: src failed to vectorize into registers\"\n");
    printf("原因: CuTe 必须在编译期证明这 4 个元素连续，才能发 128bit 指令。\n");
    printf("=> 这就是 Section 01 强调\"能编译期确定就用 Int<N>\"的实际后果\n");

    // ========== 5. 总结 ==========
    print_separator("总结");
    printf("本程序演示的内容 (讲解见 README):\n");
    printf("  §1  标量 TiledCopy + partition 的 rank-3 形状  -> README §4 §5\n");
    printf("  §2  向量 TiledCopy: val_layout 放在连续方向    -> README §5.2\n");
    printf("  §3  thr_layout 写错 -> 静默不合并              -> README §6\n");
    printf("  §4  向量化需要编译期 stride                    -> README §6.3\n");
    printf("\n下一步: capstone 会用这些做一个 gmem->smem->gmem 的 memcpy\n");

    delete[] h_src;
    delete[] h_dst;
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
