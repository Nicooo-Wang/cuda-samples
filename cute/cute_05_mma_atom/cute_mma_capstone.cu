// cute_05 capstone —— 把两个引擎拼起来: TMA + WGMMA 的完整 GEMM
//
// 对应 README §5。
//
// v0/v1/v2 各自解决了一件事:
//   v0  MMA atom 是什么形状, fragment 怎么分
//   v1  WGMMA 怎么发, 它对 smem layout 的要求
//   v2  TMA 怎么把数据喂进 smem
//
// 这一版把它们拼成一个**能跑真实尺寸**的 GEMM:
//   grid 铺开 -> 每个 CTA 负责一个 C tile -> 沿 K 循环 (TMA 搬 + WGMMA 算)
//
// 和 v2 的区别只有一处: v2 是一个 CTA 算整个 C, 这里是 grid 里每个 CTA
// 算 C 的一块。也就是加了 blockIdx 那一层坐标。
//
//   §5.1  单 CTA -> 整个 grid: local_tile 多了一层     (README §5.1)
//   §5.2  完整 kernel, 逐段对照三个 v 学到的东西        (README §5.2)
//   §5.3  实测: 不同尺寸的吞吐, 以及和 cuBLAS 的差距    (README §5.3)
//
// **这一版没有流水线** —— 搬和算仍然串行。这是故意的: 把"两个引擎各自正确"
// 讲干净, 重叠留给 cute_06。所以下面的 TFLOP/s 数字会明显低于 cuBLAS,
// README §5.3 会解释差在哪, 那正是 cute_06 的开场问题。
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_mma_capstone

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/device_kernel.h>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 尺寸配置
//
// gmem (TN 摆法, 和前面三版一致):
//   A: M x K half, row-major, stride = (K,1)
//   B: N x K half, row-major, stride = (K,1)    <- 存 B^T
//   C: M x N float, row-major, stride = (N,1)
//
// CTA tile: 每个 CTA 算 C 的 BM x BN 一块, 沿 K 方向走 K/BK 轮。
//   BM=128: 用 2x1 的 WGMMA 排布覆盖 (原子 M=64)
//   BN= 64: 正好一个 WGMMA 原子的 N
//   BK= 64: GMMA::Layout_K_SW128_Atom<half> 要求 K % 64 == 0
//
// grid = (N/BN, M/BM), 每个 CTA 128 线程 (1 个 warpgroup)。
// ---------------------------------------------------------------------------
constexpr int BM = 128, BN = 64, BK = 64;
constexpr int NTHR = 128;

// TiledMMA: 就用**裸原子**, 不加 thr_layout。
//
// 这里有一个和 v0 不一样的地方, 值得单独说 (README §5.1 有完整推导):
//
//   v0 里 make_tiled_mma(atom, Layout<Shape<_2,_2,_1>>) 的第二个参数是
//   "用几个 warp 去拼", 因为 SM80 的原子只要 32 个线程, 拼 4 个才凑够 128。
//
//   WGMMA 的原子**本身就要 128 个线程**(一个 warpgroup)。所以那个参数在这里
//   变成了"用几个 **warpgroup**"—— 写 Layout<Shape<_2,_1,_1>> 会得到
//   size(mma) == 256, 也就是要求两个 warpgroup。如果 blockDim 只有 128,
//   一半的 C 就没有线程去算, 结果静默出错 (我第一版就是这么错的)。
//
//   那 BM=128 比原子的 M=64 大一倍, 谁去覆盖? **CuTe 自动重复原子**:
//     partition_fragment_C(128x64 的 gC) -> ((_2,_2,_8),_2,_1), 每线程 64 个 float
//                                                        ^^ MMA_M = 2, 发两次
//   一个 warpgroup 沿 M 方向把同一条指令发两次就够了, 不需要多要线程。
CUTE_HOST_DEVICE static auto make_wgmma() {
    return make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
}

// ===========================================================================
// §5.2  完整 kernel
//
// 结构上就是 v2 的 kernel 加了 blockIdx 那一层。逐段标了它来自哪一版。
// ===========================================================================
template <class TmaA, class TmaB, class SLayA, class SLayB>
__global__ __launch_bounds__(NTHR) void gemm_tma_wgmma(
    CUTLASS_GRID_CONSTANT TmaA const tma_a, CUTLASS_GRID_CONSTANT TmaB const tma_b, float* C,
    int M, int N, int K, SLayA slayA, SLayB slayB) {
    // ---- smem (v1: 必须 128B 对齐; v2: 必须带 PIPE mode) ----
    __shared__ __align__(128) half_t rawA[cosize_v<SLayA>];
    __shared__ __align__(128) half_t rawB[cosize_v<SLayB>];
    __shared__ __align__(8) uint64_t bar[1];

    auto sA = make_tensor(make_smem_ptr(rawA), slayA);  // (BM,BK,1)
    auto sB = make_tensor(make_smem_ptr(rawB), slayB);  // (BN,BK,1)

    // ---- gmem 坐标 tensor (v2 条件 1) ----
    auto mA = tma_a.get_tma_tensor(make_shape(M, K));
    auto mB = tma_b.get_tma_tensor(make_shape(N, K));

    // ---- §5.1 这一层是新的: 用 blockIdx 选出本 CTA 负责的那一块 ----
    // gA: (BM,BK,k) —— 固定 M 方向的第 blockIdx.y 块, K 方向留成序列
    // gB: (BN,BK,k) —— 固定 N 方向的第 blockIdx.x 块
    auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(blockIdx.y, _));
    auto gB = local_tile(mB, Shape<Int<BN>, Int<BK>>{}, make_coord(blockIdx.x, _));

    // ---- TMA partition (v2 条件 5) ----
    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                      group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                                      group_modes<0, 2>(gB));
    constexpr int txb = sizeof(make_tensor_like(tensor<0>(tAsA)))
                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

    // ---- 累加器 (v0: fragment; v1: C 仍是真寄存器) ----
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));
    auto gC = local_tile(mC, Shape<Int<BM>, Int<BN>>{}, make_coord(blockIdx.y, blockIdx.x));

    auto mma = make_wgmma();
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto tCrC = thr.partition_fragment_C(gC);
    clear(tCrC);

    // ---- mbarrier 初始化 (v2) ----
    int warp = cutlass::canonical_warp_idx_sync();
    int one = cute::elect_one_sync();
    using Bar = cutlass::arch::ClusterTransactionBarrier;
    if (warp == 0 && one) Bar::init(&bar[0], 1);
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    auto sA2 = sA(_, _, Int<0>{});
    auto sB2 = sB(_, _, Int<0>{});
    int nk = K / BK;

    // ---- mainloop: 搬一块, 算一块 (串行, 没有重叠 —— 见文件头说明) ----
    for (int k = 0; k < nk; ++k) {
        if (warp == 0 && one) {
            Bar::arrive_and_expect_tx(&bar[0], txb);
            copy(tma_a.with(bar[0]), tAgA(_, k), tAsA(_, Int<0>{}));
            copy(tma_b.with(bar[0]), tBgB(_, k), tBsB(_, Int<0>{}));
        }
        Bar::wait(&bar[0], k & 1);  // phase 每轮翻转

        auto tCrA = thr.make_fragment_A(thr.partition_A(sA2));
        auto tCrB = thr.make_fragment_B(thr.partition_B(sB2));
        warpgroup_arrive();
        gemm(mma, tCrA, tCrB, tCrC);  // v1 的四句套路
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        __syncthreads();  // 算完才能覆盖 smem
    }

    // ---- 写回 ----
    copy(tCrC, thr.partition_C(gC));
}

// ---------------------------------------------------------------------------
// host 侧: 分配、建 descriptor、launch、验证、计时
// ---------------------------------------------------------------------------
struct Result {
    int M, N, K;
    float ms;
    double tflops;
    bool ok;
};

static Result run_gemm(int M, int N, int K, bool verify) {
    size_t nA = size_t(M) * K, nB = size_t(N) * K, nC = size_t(M) * N;

    half_t *h_A = new half_t[nA], *h_B = new half_t[nB];
    float* h_C = new float[nC];
    fill_pm1(h_A, nA, 31);
    fill_pm1(h_B, nB, 41);

    half_t *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, nA * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_B, nB * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_C, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nA * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, nB * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, nC * sizeof(float)));

    // descriptor 必须用真实设备指针在 host 建 (v2 条件 2)
    auto mA = make_tensor(make_gmem_ptr(d_A), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(d_B), make_shape(N, K), make_stride(K, Int<1>{}));

    auto slayA = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                               make_shape(Int<BM>{}, Int<BK>{}, Int<1>{}));
    auto slayB = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                               make_shape(Int<BN>{}, Int<BK>{}, Int<1>{}));
    auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, mA, slayA(_, _, Int<0>{}),
                               make_shape(Int<BM>{}, Int<BK>{}));
    auto tma_b = make_tma_atom(SM90_TMA_LOAD{}, mB, slayB(_, _, Int<0>{}),
                               make_shape(Int<BN>{}, Int<BK>{}));

    dim3 grid(N / BN, M / BM), block(NTHR);
    auto launch = [&] {
        gemm_tma_wgmma<<<grid, block>>>(tma_a, tma_b, d_C, M, N, K, slayA, slayB);
    };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    bool ok = true;
    if (verify) {
        float* h_ref = new float[nC];
        gemm_cpu(h_A, h_B, h_ref, M, N, K);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, nC * sizeof(float), cudaMemcpyDeviceToHost));
        auto r = check_close(h_C, h_ref, nC);
        ok = r.ok();
        printf("    验证 %dx%dx%d: %s (bad=%d, maxerr=%g)\n", M, N, K,
               ok ? "完全一致" : "不一致", r.bad, r.maxerr);
        delete[] h_ref;
    }

    float ms = time_kernel(launch, 5, 20);
    double tf = gemm_tflops(M, N, K, ms);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return {M, N, K, ms, tf, ok};
}

// ===========================================================================
int main() {
    printf("cute_05 capstone —— TMA + WGMMA 的完整 GEMM\n");
    printf("对应 README §5\n");
    printf("\nCTA tile = %dx%dx%d, 每 CTA %d 线程 (1 warpgroup)\n", BM, BN, BK, NTHR);
    printf("TiledMMA = SM90_64x64x16 原子, 沿 M 自动发 %d 次 -> 覆盖 %dx%d\n", BM / 64, BM, BN);

    print_separator("§5.2  正确性 —— 小尺寸和 CPU 逐点比");
    run_gemm(256, 256, 256, true);
    run_gemm(512, 256, 512, true);

    print_separator("§5.3  吞吐 —— 这一版能跑多快");
    printf("\n  %-18s %10s %12s\n", "shape", "time(ms)", "TFLOP/s");
    Result rs[] = {
        run_gemm(1024, 1024, 1024, false),
        run_gemm(2048, 2048, 2048, false),
        run_gemm(4096, 4096, 4096, false),
    };
    for (auto& r : rs) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%dx%dx%d", r.M, r.N, r.K);
        printf("  %-18s %10.3f %12.1f\n", buf, r.ms, r.tflops);
    }

    print_separator("小结 —— 以及为什么还不够快");
    printf("  这一版把三件事拼齐了:\n");
    printf("    v0 的 fragment 概念 + v1 的 WGMMA 四句 + v2 的 TMA 通路\n");
    printf("    再加上 grid 那一层 blockIdx -> 每个 CTA 一块 C\n");

    printf("\n  但它离 cuBLAS (本机 4096^3 实测约 805 TFLOP/s) 还差得远, 原因有三个,\n");
    printf("  每一个都是 cute_06 的一节:\n");
    printf("    1. **没有流水线**: 搬完才算, 算完才搬, 两个引擎各闲一半。\n");
    printf("       -> cute_06 v3: 多 stage + PipelineState (cute_04 §5 已跑通骨架)\n");
    printf("    2. **没有 warp 分工**: 同一批线程既发 TMA 又等 WGMMA。\n");
    printf("       -> cute_06 v4: Warp Specialization\n");
    printf("    3. **tile 太小、没有 cluster**: BM=%d 只有一个 warpgroup 在算。\n", BM);
    printf("       -> cute_06 capstone: 更大的 tile + Block Cluster + multicast\n");

    printf("\ncapstone OK\n");
    return 0;
}
