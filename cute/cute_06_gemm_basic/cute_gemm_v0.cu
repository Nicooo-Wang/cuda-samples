// cute_06 v0 —— naive GEMM: 不用 smem, 先搭骨架
//
// 对应 README §1。
//
// cute_05 的 capstone 已经是 TMA + WGMMA 的完整 GEMM 了 —— 那为什么还要
// 从 naive 讲起? 因为 cute_06 的目标不是"能跑", 而是"为什么这么跑":
//
//   naive 版把 GEMM 的**骨架**摆出来 (grid 怎么铺、K 循环怎么写),
//   后面的每一版只换其中一块, 骨架不动。
//
// 这一版没有任何 smem、没有任何流水线: 每个 CTA 直接
//   1. 从 gmem 读 A/B 的 tile 进寄存器 (fragment)
//   2. 用 TiledMMA 算 C 的一块
//   3. 写回 gmem
//
// 它慢是必然的 (访存没合并、没有复用), 但它是"正确性基线"——
// 后面每一版都要和它逐点一致。
//
//   §1.1  骨架: grid 铺法 + local_tile 的 k 循环   (README §1.1)
//   §1.2  逐点验证                                  (README §1.2)
//   §1.3  它为什么慢                                (README §1.3)
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_gemm_v0

#include <cute/tensor.hpp>
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
// 这一版用 Ampere 的 warp 级 MMA (SM80_16x8x16, 32 线程一条指令):
//   4 个 warp 排成 4x1 -> 覆盖 C 的 (16*4) x (8*1) = 64 x 8
//   一个 CTA = 128 线程, 算 64x8 一小块。
//
// grid = (N/8, M/64): blockIdx.x 走 N, blockIdx.y 走 M。
// ---------------------------------------------------------------------------
constexpr int BM = 64, BN = 8, BK = 16;  // CTA tile: 64x8, K 一步 16
constexpr int NTHR = 128;                 // 4 个 warp

CUTE_HOST_DEVICE static auto make_mma() {
    // 4 个 warp 沿 M 排成一列: M 重复 4 次 (16*4=64), N 重复 1 次 (8*1=8)
    return make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_4, _1, _1>>{});
}

// ===========================================================================
// §1.1  naive kernel
//
// 每个 CTA 算 C 的 64x8 一小块, 沿 K 循环。这是整个 cute_06 的骨架:
//   1. local_tile 从 gmem 里切出本 CTA 负责的 A/B 块 (沿 K 是序列)
//   2. partition_fragment 把 A/B 搬进寄存器
//   3. gemm 累加
//   4. 写回
// 后面每一版都是在这个骨架里换 1/2/3 的实现。
// ===========================================================================
__global__ void gemm_naive(const half_t* A, const half_t* B, float* C, int M, int N, int K) {
    // 整块视图
    auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, Int<1>{}));
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

    // 本 CTA 负责的那一块: blockIdx.y 是 M 方向, blockIdx.x 是 N 方向
    auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(blockIdx.y, _));
    auto gB = local_tile(mB, Shape<Int<BN>, Int<BK>>{}, make_coord(blockIdx.x, _));
    auto gC = local_tile(mC, Shape<Int<BM>, Int<BN>>{}, make_coord(blockIdx.y, blockIdx.x));

    auto mma = make_mma();
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);

    // 累加器 (寄存器), 在 K 循环外清零
    auto tCrC = thr.partition_fragment_C(gC);
    clear(tCrC);

    int nk = K / BK;
    for (int k = 0; k < nk; ++k) {
        // 我这份 A/B 在 gmem 里的位置
        auto tAgA = thr.partition_A(gA(_, _, k));
        auto tBgB = thr.partition_B(gB(_, _, k));
        // 我这份 A/B 在寄存器里的容器
        auto tArA = thr.partition_fragment_A(gA(_, _, k));
        auto tBrB = thr.partition_fragment_B(gB(_, _, k));
        // 搬进来 (每个线程各自从 gmem 读)
        copy(tAgA, tArA);
        copy(tBgB, tBrB);
        // 算
        gemm(mma, tArA, tBrB, tCrC);
    }

    // 写回
    copy(tCrC, thr.partition_C(gC));
}

// ===========================================================================
// §1.2  验证 + 计时
// ===========================================================================
static void run(int M, int N, int K, bool verify) {
    size_t nA = size_t(M) * K, nB = size_t(N) * K, nC = size_t(M) * N;

    half_t *h_A = new half_t[nA], *h_B = new half_t[nB];
    float *h_C = new float[nC], *h_ref = new float[nC];
    fill_pm1(h_A, nA, 51);
    fill_pm1(h_B, nB, 61);
    gemm_cpu(h_A, h_B, h_ref, M, N, K);

    half_t *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, nA * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_B, nB * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_C, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nA * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, nB * sizeof(half_t), cudaMemcpyHostToDevice));

    dim3 grid(N / BN, M / BM);  // 每 CTA 64x8
    auto launch = [&] { gemm_naive<<<grid, NTHR>>>(d_A, d_B, d_C, M, N, K); };

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
    printf("cute_06 v0 —— naive GEMM: 不用 smem, 先搭骨架\n");
    printf("对应 README §1\n");

    print_separator("§1.2  正确性基线");
    run(128, 128, 128, true);
    run(512, 256, 512, true);

    print_separator("§1.3  它有多慢");
    printf("\n  2048x2048x2048 的访存: 每个 CTA 都直接读 gmem, 没有任何复用。\n");
    printf("  (CUDA 会缓存一部分, 但同一份数据被相邻 CTA 反复读, 缓存也救不回来。)\n");
    run(2048, 2048, 2048, false);

    print_separator("小结");
    printf("  这一版的价值是**骨架**: grid 铺法 + k 循环 + fragment 累加。\n");
    printf("  它的三个致命伤 (全部来自访存):\n");
    printf("    1. 每个 CTA 只算 64x8, gmem 读几乎没合并 (B 的 8 行每行只有 8 列)\n");
    printf("    2. A/B 每个元素被多少个 CTA 重复读? 看 N 方向:\n");
    printf("       一个 A 元素被 N/8 个 CTA 读 (每个 CTA 对应一列 C)\n");
    printf("    3. 没有 smem = 没有跨 K 循环的复用\n");
    printf("\n  下一步 (v1): 把 A/B 先搬进 smem —— 每个 CTA 只从 gmem 读一次,\n");
    printf("  之后 K 循环全在 smem 里转。这就是经典 GEMM 的 Ampere 形态。\n");

    printf("\nv0 OK\n");
    return 0;
}
