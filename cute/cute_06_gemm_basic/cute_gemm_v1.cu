// cute_06 v1 —— smem 中转: 每个 CTA 只从 gmem 读一次
//
// 对应 README §2。
//
// v0 的致命伤: 每个 K 循环都从 gmem 读 A/B, 而 A/B 在 gmem 里被大量 CTA 重复读。
// 经典解法: **先把 A/B 的一块搬进 smem, 之后 K 循环只读 smem**。
//
// 这一版用 Ampere 的思路:
//   gmem -> smem:  cp.async (SM80_CP_ASYNC_CACHEALWAYS, 每线程 128b 向量)
//   smem -> 寄存器: ldmatrix (SM75_U32x4_LDSM_N 原子)
//   mma            : SM80 warp 级 MMA (cute_05 v0 学过的)
//
// 同步还是 __syncthreads —— 因为所有线程既搬又算, 这是 Ampere 单缓冲。
//
//   §2.1  数据流: gmem -> smem -> 寄存器 -> MMA   (README §2.1)
//   §2.2  关键: ldmatrix 从 smem 取数, 天然配 swizzle 布局 (README §2.2)
//   §2.3  实测: 和 v0 比, 快在哪             (README §2.3)
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_gemm_v1

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 尺寸配置
//
// gmem (TN 摆法):
//   A: M x K half, row-major, stride = (K,1)
//   B: N x K half, row-major, stride = (K,1)   <- 存 B^T
//   C: M x N float, row-major, stride = (N,1)
//
// CTA tile: BM=64, BN=64, BK=64。
//   4 个 warp 排成 2x2, 每个 warp 覆盖 32x32x16 (见 make_mma 的 Tile)。
//   BK=64: cp.async 每轮搬 64x64 half, K 方向 4 步 (每步 BK=64)。
//
// smem layout: 用 Swizzle<3,3,3> 原子 (cute_04 §3 的 swizzle),
//   这里是为了让 ldmatrix 取数不撞 bank —— 不是给 WGMMA 用的。
//
// 每个 CTA: 128 线程 = 4 个 warp。
// ---------------------------------------------------------------------------
constexpr int BM = 64, BN = 64, BK = 64;
constexpr int NTHR = 128;  // 4 warp

// 注意第三个参数 Tile<_32,_32,_16>: 这是**排列** (permutation), 不是形状。
// 它把 4 个 16x8x16 原子重排成 32x32x16 的"大原子" ——
// ldmatrix 一次取 4x8 half, 只有按 32x32 排布时它的线程映射才能对上。
// 没有这个 Tile, make_tiled_copy_A(s2r_atom, mma) 会报
// "TiledCopy uses too few vals" (见 §2.2 的说明)。
CUTE_HOST_DEVICE static auto make_mma() {
    return make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_2, _2>>{},
                          Tile<_32, _32, _16>{});
}

// gmem -> smem: cp.async, 每线程拷 8 个 half (128b), k-major
CUTE_HOST_DEVICE static auto make_stage_copy() {
    return make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
                           Layout<Shape<_16, _8>, Stride<_8, _1>>{},  // Thr 16x8 k-major
                           Layout<Shape<_1, _8>>{});                  // Val 1x8 k-major
}

// smem -> 寄存器: ldmatrix (cute_04 §4 讲过)。一条 ldmatrix 从 smem 取 4x8 half
// 装进 4 个寄存器, 天然配 swizzle 布局 —— 它自己就是为这个设计的。
CUTE_HOST_DEVICE static auto make_ldmatrix() {
    return Copy_Atom<SM75_U32x4_LDSM_N, half_t>{};
}

// ===========================================================================
// §2.1  kernel
//
// 骨架和 v0 一样 (grid + k 循环), 只把"从 gmem 读"换成"从 smem 读":
//   k 循环每轮:  gmem -> smem (cp.async)
//               ldmatrix 取数 + mma
//
// 注意 __syncthreads 两次: 搬完等所有人, 算完等所有人 (单缓冲的代价)。
// ===========================================================================
template <class SLayA, class SLayB>
__global__ void gemm_smem(const half_t* A, const half_t* B, float* C, int M, int N, int K,
                          SLayA slayA, SLayB slayB) {
    __shared__ __align__(128) half_t rawA[cosize_v<SLayA>];
    __shared__ __align__(128) half_t rawB[cosize_v<SLayB>];

    auto sA = make_tensor(make_smem_ptr(rawA), slayA);  // (BM,BK)
    auto sB = make_tensor(make_smem_ptr(rawB), slayB);  // (BN,BK)

    auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, Int<1>{}));
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

    auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(blockIdx.y, _));
    auto gB = local_tile(mB, Shape<Int<BN>, Int<BK>>{}, make_coord(blockIdx.x, _));
    auto gC = local_tile(mC, Shape<Int<BM>, Int<BN>>{}, make_coord(blockIdx.y, blockIdx.x));

    // gmem -> smem 的 TiledCopy
    auto tcopy = make_stage_copy();
    ThrCopy tc = tcopy.get_slice(threadIdx.x);

    auto mma = make_mma();
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto tCrC = thr.partition_fragment_C(gC);
    clear(tCrC);

    // smem -> 寄存器的 TiledCopy: 必须**用 mma 定制** (见 make_mma 的注释)
    auto s2r_a = make_tiled_copy_A(make_ldmatrix(), mma);
    auto s2r_b = make_tiled_copy_B(make_ldmatrix(), mma);
    ThrCopy tlA = s2r_a.get_slice(threadIdx.x);
    ThrCopy tlB = s2r_b.get_slice(threadIdx.x);

    // A/B 的 smem 视图 (ldmatrix 用) —— 每个线程自己那一份
    auto tCsA = thr.partition_A(sA);  // (MMA, MMA_M, MMA_K)
    auto tCsB = thr.partition_B(sB);

    int nk = K / BK;
    for (int k = 0; k < nk; ++k) {
        // gmem -> smem (cp.async)
        cute::copy(tc.partition_S(gA(_, _, k)), tc.partition_D(sA));
        cute::copy(tc.partition_S(gB(_, _, k)), tc.partition_D(sB));
        __syncthreads();  // 搬完, 所有人才能开始读 smem

        // smem -> 寄存器 (ldmatrix) + mma
        auto tCrA = thr.make_fragment_A(tCsA);
        auto tCrB = thr.make_fragment_B(tCsB);
        cute::copy(s2r_a, tlA.partition_S(sA), tlA.retile_D(tCrA));
        cute::copy(s2r_b, tlB.partition_S(sB), tlB.retile_D(tCrB));

        gemm(mma, tCrA, tCrB, tCrC);

        __syncthreads();  // 算完, 才能覆盖 smem
    }

    cute::copy(tCrC, thr.partition_C(gC));
}

// ===========================================================================
// §2.3  验证 + 计时
// ===========================================================================
static void run(int M, int N, int K, bool verify) {
    size_t nA = size_t(M) * K, nB = size_t(N) * K, nC = size_t(M) * N;

    half_t *h_A = new half_t[nA], *h_B = new half_t[nB];
    float *h_C = new float[nC], *h_ref = new float[nC];
    fill_pm1(h_A, nA, 52);
    fill_pm1(h_B, nB, 62);
    gemm_cpu(h_A, h_B, h_ref, M, N, K);

    half_t *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, nA * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_B, nB * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_C, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nA * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, nB * sizeof(half_t), cudaMemcpyHostToDevice));

    // smem layout: 手写 swizzle 原子 (cute_04 §3), 不是 GMMA —— 这里没有 WGMMA
    auto swizzle_atom = composition(Swizzle<3, 3, 3>{},
                                    Layout<Shape<_8, Shape<_8, _8>>,
                                           Stride<_8, Stride<_1, _64>>>{});
    auto slayA = tile_to_shape(swizzle_atom, make_shape(Int<BM>{}, Int<BK>{}));
    auto slayB = tile_to_shape(swizzle_atom, make_shape(Int<BN>{}, Int<BK>{}));

    dim3 grid(N / BN, M / BM);
    auto launch = [&] { gemm_smem<<<grid, NTHR>>>(d_A, d_B, d_C, M, N, K, slayA, slayB); };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    if (verify) {
        CUDA_CHECK(cudaMemcpy(h_C, d_C, nC * sizeof(float), cudaMemcpyDeviceToHost));
        auto r = check_close(h_C, h_ref, nC);
        printf("  验证 %dx%dx%d: %s (bad=%d, maxerr=%g)\n", M, N, K,
               r.ok() ? "完全一致" : "不一致", r.bad, r.maxerr);
    }

    float ms = time_kernel(launch, 5, 20);
    printf("  %dx%dx%d: %.3f ms   %.1f TFLOP/s\n", M, N, K, ms,
           gemm_tflops(M, N, K, ms));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_ref;
}

// ===========================================================================
int main() {
    printf("cute_06 v1 —— smem 中转: 每个 CTA 只从 gmem 读一次\n");
    printf("对应 README §2\n");

    print_separator("§2.3  正确性 + 实测");
    run(128, 128, 128, true);
    run(512, 256, 512, true);
    run(2048, 2048, 2048, false);

    print_separator("小结");
    printf("  v0 (naive)     2048^3: ~11.7 TFLOP/s\n");
    printf("  v1 (smem 中转) 2048^3: 看上面打印\n");
    printf("  快在哪: A/B 每块只从 gmem 读一次, K 循环全在 smem 里转。\n");
    printf("  但还是慢: 搬和算串行 (__syncthreads 两次/轮), 且只有单缓冲。\n");
    printf("\n  下一步 (v2): 双缓冲 + cp.async 流水线 —— 让搬 k+1 和算 k 重叠。\n");

    printf("\nv1 OK\n");
    return 0;
}
