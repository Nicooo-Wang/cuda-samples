// cute_04 v1 —— 协作式 Copy Atom: ThrID > 1
//
// 对应 README §5 §6。
//
// cute_03 里所有 atom 的 ThrID 都是 1（一个线程独立发一条指令）。
// 本文件引入第一个"多线程协作才能发出一条指令"的 atom: ldmatrix。
// 它的存在理由不是更快，而是 Tensor Core 要求的线程↔数据映射手写不出来。

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// §5  ThrID = 1 vs ThrID = 32
// ---------------------------------------------------------------------------
template <class Atom>
void show_atom(const char* name) {
    printf("\n  %s\n", name);
    printf("    ThrID size = %-3d  NumValSrc = %-3d  NumValDst = %d\n",
           int(size(typename Atom::ThrID{})), Atom::NumValSrc, Atom::NumValDst);
    printf("    ValLayoutSrc = ");
    print(typename Atom::ValLayoutSrc{});
    printf("\n    ValLayoutDst = ");
    print(typename Atom::ValLayoutDst{});
    printf("\n");
}

void section5_thrid() {
    print_separator("§5  ThrID: 一条指令要几个线程一起发");

    printf("cute_03 见过的 atom 全都是 ThrID = 1:");
    show_atom<Copy_Atom<UniversalCopy<float>, float>>("Copy_Atom<UniversalCopy<float>, float>");
    show_atom<Copy_Atom<UniversalCopy<uint128_t>, float>>(
        "Copy_Atom<UniversalCopy<uint128_t>, float>");

    printf("\nldmatrix 完全不同 —— 32 个线程协作发一条指令:");
    show_atom<Copy_Atom<SM75_U32x4_LDSM_N, half_t>>("Copy_Atom<SM75_U32x4_LDSM_N, half_t>");

    using A4 = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    int thr = int(size(typename A4::ThrID{}));
    printf("\n  一次操作搬的总量 = ThrID * NumVal = %d * %d = %d 个 half\n", thr, A4::NumValSrc,
           thr * A4::NumValSrc);
    printf("  = 4 个 8x8 的 half 矩阵 (这就是名字里 U32x4 的 4)\n");

    printf("\n其他两个变体:\n");
    show_atom<Copy_Atom<SM75_U32x2_LDSM_N, half_t>>("SM75_U32x2_LDSM_N (2 个矩阵)");
    show_atom<Copy_Atom<SM75_U16x8_LDSM_T, half_t>>("SM75_U16x8_LDSM_T (T = 转置加载)");

    printf("\n_N 和 _T 的区别: _T 在加载时顺手把 8x8 转置了 —— 这解决了\n");
    printf("\"B 矩阵在内存里是 row-major 但 MMA 要 col-major\"的问题, 不用额外一次转置。\n");
}

// ---------------------------------------------------------------------------
// §6  ldmatrix 的线程↔数据映射
// ---------------------------------------------------------------------------
template <class MMA, class SLay>
__global__ void ldsm_kernel(const half_t* g, int* out_first, int* out_all, MMA mma, SLay slay) {
    __shared__ __align__(128) half_t s[16 * 16];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s[i] = g[i];
    __syncthreads();

    auto sA = make_tensor(make_smem_ptr(s), slay);

    // 1) 先让 TiledMMA 说出它要的 fragment 形状
    auto thr_mma = mma.get_thread_slice(threadIdx.x);
    auto tCsA = thr_mma.partition_A(sA);
    auto tCrA = thr_mma.make_fragment_A(tCsA);

    // 2) 用 make_tiled_copy_A 让 ldmatrix 去匹配这个形状
    auto tiled_ldsm = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
    auto thr_ldsm = tiled_ldsm.get_slice(threadIdx.x);
    auto tXsA = thr_ldsm.partition_S(sA);
    auto tXrA = thr_ldsm.retile_D(tCrA);  // 同一块寄存器, 换成 ldmatrix 的视角

    copy(tiled_ldsm, tXsA, tXrA);

    if (thread0()) {
        printf("    tCsA (MMA 要的 smem 视图) = ");
        print(tCsA);
        printf("\n    tCrA (MMA 要的寄存器)     = ");
        print(tCrA);
        printf("\n    tXsA (ldmatrix 的 smem)   = ");
        print(tXsA);
        printf("\n    tXrA (retile 后的寄存器)  = ");
        print(tXrA);
        printf("\n");
    }

    out_first[threadIdx.x] = int(float(tCrA(0)));
    for (int i = 0; i < int(size(tCrA)); ++i) out_all[threadIdx.x * 8 + i] = int(float(tCrA(i)));
}

void section6_mapping() {
    print_separator("§6  ldmatrix 到底把哪个元素给了哪个线程");

    auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{});
    auto slay = make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}));

    printf("用 SM80_16x8x16_F32F16F16F32_TN 做参照 (warp 级 MMA, %d 线程)\n", int(size(mma)));
    printf("smem 是 16x16 的 half, row-major, 值 = 线性下标 0..255\n");

    half_t* h = (half_t*)malloc(256 * sizeof(half_t));
    for (int i = 0; i < 256; ++i) h[i] = half_t(float(i));

    half_t* d;
    int *d_first, *d_all;
    CUDA_CHECK(cudaMalloc(&d, 256 * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_first, 32 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all, 32 * 8 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, h, 256 * sizeof(half_t), cudaMemcpyHostToDevice));

    printf("\n  kernel 内打印的四个形状:\n");
    ldsm_kernel<<<1, int(size(mma))>>>(d, d_first, d_all, mma, slay);
    CUDA_CHECK(cudaDeviceSynchronize());

    int first[32], all[32 * 8];
    CUDA_CHECK(cudaMemcpy(first, d_first, sizeof(first), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(all, d_all, sizeof(all), cudaMemcpyDeviceToHost));

    printf("\n  每个线程拿到的第一个元素:\n    ");
    for (int t = 0; t < 32; ++t) {
        printf("%3d ", first[t]);
        if (t % 16 == 15) printf("\n    ");
    }
    printf("\n  注意不是 0 1 2 3 ... —— thr0..3 拿的是 0 2 4 6, thr4 跳到 16。\n");

    printf("\n  thr0 拿到的全部 8 个: ");
    for (int i = 0; i < 8; ++i) printf("%d ", all[i]);
    printf("\n  thr1 拿到的全部 8 个: ");
    for (int i = 0; i < 8; ++i) printf("%d ", all[8 + i]);
    printf("\n");

    printf("\n这个顺序是 Tensor Core 硬件规定的, 没有任何直观规律可循。\n");
    printf("**这就是协作式 atom 的价值**: 你说\"我要给 MMA 喂 A\", CuTe 生成正确的映射。\n");
    printf("手写这个映射几乎必错, 而且错了不报错, 只是算出错的结果。\n");

    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFree(d_first));
    CUDA_CHECK(cudaFree(d_all));
    free(h);
}

// ---------------------------------------------------------------------------
// §7  make_tiled_copy_A / retile_D 的分工
// ---------------------------------------------------------------------------
void section7_retile() {
    print_separator("§7  为什么需要 make_tiled_copy_A 和 retile_D");

    printf("同一块寄存器有两个视角:\n\n");
    printf("  MMA 的视角      : 我要 (MMA,MMA_M,MMA_K) 这样的 fragment\n");
    printf("  ldmatrix 的视角 : 我一次搬 (8,1) 这么一片\n\n");
    printf("两者形状不同但指向同一块寄存器。retile_D 就是做这个视角转换:\n\n");
    printf("  auto tCrA = thr_mma.make_fragment_A(tCsA);   // MMA 视角, 分配寄存器\n");
    printf("  auto tXrA = thr_ldsm.retile_D(tCrA);         // 换成 ldmatrix 视角\n");
    printf("  copy(tiled_ldsm, tXsA, tXrA);                // 按 ldmatrix 视角搬\n");
    printf("  gemm(mma, tCrA, tCrB, tCrC);                 // 按 MMA 视角算\n\n");
    printf("retile_D 不搬数据、不分配内存, 只是换一个 layout 去看同一块寄存器。\n");

    printf("\n三个函数的分工:\n");
    printf("  %-24s %s\n", "make_tiled_copy_A(atom,mma)", "生成\"给 MMA 的 A 喂数据\"的 TiledCopy");
    printf("  %-24s %s\n", "partition_S(sA)", "我这个线程要从 smem 读哪些");
    printf("  %-24s %s\n", "retile_D(tCrA)", "把 MMA 的 fragment 换成 copy 的视角");

    printf("\n对 B 矩阵有对应的 make_tiled_copy_B。\n");
    printf("Section 05 会看到 Hopper 的 WGMMA 直接吃 smem, 连这一步都省了。\n");
}

int main() {
    printf("cute_04 v1 —— 协作式 Copy Atom (ThrID > 1)\n");
    printf("对应 README §5 §6 §7\n");

    section5_thrid();
    section6_mapping();
    section7_retile();

    printf("\n");
    print_separator("小结");
    printf("  §5  ldmatrix 的 ThrID = 32: 32 线程协作搬 4 个 8x8 矩阵\n");
    printf("  §6  它给出的线程->数据映射是硬件规定的诡异顺序, 手写必错\n");
    printf("  §7  make_tiled_copy_A + retile_D: 让 MMA 自己说出它要的排布\n");
    printf("\nv1 OK\n");
    return 0;
}
