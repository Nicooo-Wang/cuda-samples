// cute_06 capstone —— Block Cluster: 把 CTA 组成小队
//
// 对应 README §6。
//
// v0-v4 一路把 GEMM 从 naive 推到 WS。这一版加最后一块拼图: **Block Cluster**。
//
// 什么是 cluster? 把几个 CTA 绑定成一组, 它们:
//   - 保证同时驻留在同一个 SM 上 (更小的调度粒度)
//   - 可以互相同步 (cluster_sync, 比 __syncthreads 高一层)
//   - 可以用 TMA multicast 共享同一块 gmem 数据 (见 §6.2 的讨论)
//
// 这一版是**非 multicast** 的 cluster: 每个 CTA 用自己的 TMA 搬自己的 A/B 块,
// 但 cluster 机制完整保留 —— block_rank_in_cluster / cluster_sync /
// 按 cluster 排布的 grid。
//
// 为什么不直接上 multicast? 因为 hand-rolled multicast 需要 TMA TiledCopy 的
// 全套 machinery (make_tma_copy + get_slice + partition_S/D), 那是 CUTLASS
// sm90_mma_tma_gmma_ss_warpspecialized.hpp 的几百行。这一版先把 cluster 机制
// 讲干净, multicast 的原理和它在 CUTLASS 里的用法见 README §6.2。
//
//   §6.1  cluster 机制: 怎么启动、怎么同步       (README §6.1)
//   §6.2  multicast: 原理 + 为什么留给 CUTLASS   (README §6.2)
//   §6.3  实测: cluster 和 v3 比                  (README §6.3)
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_gemm_capstone

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/device_kernel.h>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 尺寸配置
//
// CTA tile: BM=128, BN=128, BK=64 (和 v3 相同)。
// Cluster: 2x2 = 4 个 CTA 一组。
//
// grid 的含义和 v3 相同: (N/BN, M/BM)。区别只在 launch 时
// 把 grid 按 cluster 分组 —— cluster 是调度和同步的单位。
//
// 这一版每个 CTA 还是算自己那一块 C, 数据各搬各的 (非 multicast)。
// cluster 的价值在这里:
//   1. 保证 4 个 CTA 同时驻留 -> 后续要加 multicast 时它们才能共享 smem 流量
//   2. cluster_sync 给 4 个 CTA 一个比 grid 更细的同步面
// ---------------------------------------------------------------------------
constexpr int BM = 128, BN = 128, BK = 64;
constexpr int NTHR = 128;
constexpr int STAGES = 3;
constexpr int CLUSTER_M = 2, CLUSTER_N = 2;  // cluster 形状 (m, n)

CUTE_HOST_DEVICE static auto make_wgmma() {
    return make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
}

template <class SLA, class SLB>
struct SharedStorage {
    alignas(128) cute::ArrayEngine<half_t, cosize_v<SLA>> A;
    alignas(128) cute::ArrayEngine<half_t, cosize_v<SLB>> B;
    uint64_t tma_barrier[STAGES];
    uint64_t mma_barrier[STAGES];
};

// ===========================================================================
// §6.1  kernel —— v3 + cluster
//
// 和 v3 的差别只有三处 (都标了 ★):
//   ★1  launch 用 cluster_dims + launch_kernel_on_cluster (host 侧)
//   ★2  用 block_rank_in_cluster() 拿到本 CTA 在 cluster 里的身份
//   ★3  barrier 初始化后加 cluster_sync() (全 cluster 的栅栏)
// ===========================================================================
template <class TmaA, class TmaB, class SLA, class SLB>
__global__ __launch_bounds__(NTHR) void gemm_cluster(
    CUTLASS_GRID_CONSTANT TmaA const tma_a, CUTLASS_GRID_CONSTANT TmaB const tma_b, float* C,
    int M, int N, int K, SLA sla, SLB slb) {
    extern __shared__ char shared_memory[];
    SharedStorage<SLA, SLB>& smem = *reinterpret_cast<SharedStorage<SLA, SLB>*>(shared_memory);

    auto sA = make_tensor(make_smem_ptr(smem.A.begin()), sla);  // (BM,BK,STAGES)
    auto sB = make_tensor(make_smem_ptr(smem.B.begin()), slb);  // (BN,BK,STAGES)

    // ★2  cluster 里的身份 (0..CLUSTER_M*CLUSTER_N-1)
    // 这一版非 multicast 用不到它, 但它是 cluster 编程的基础 API。
    int cluster_local = cute::block_rank_in_cluster();

    // gmem 坐标 tensor + local_tile (和 v3 一样)
    auto mA = tma_a.get_tma_tensor(make_shape(M, K));
    auto mB = tma_b.get_tma_tensor(make_shape(N, K));
    auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(blockIdx.y, _));
    auto gB = local_tile(mB, Shape<Int<BN>, Int<BK>>{}, make_coord(blockIdx.x, _));

    // TMA partition: 非 multicast, 每个 CTA 搬自己的 (和 v3 一样)
    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                      group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                                      group_modes<0, 2>(gB));
    constexpr int txb = sizeof(make_tensor_like(tensor<0>(tAsA)))
                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

    // 累加器
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));
    auto gC = local_tile(mC, Shape<Int<BM>, Int<BN>>{}, make_coord(blockIdx.y, blockIdx.x));
    auto mma = make_wgmma();
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto tCrC = thr.partition_fragment_C(gC);
    clear(tCrC);

    // barrier 初始化
    int warp = cutlass::canonical_warp_idx_sync();
    int one = cute::elect_one_sync();
    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBar = cutlass::arch::ClusterBarrier;
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;
    if (warp == 0 && one) {
        for (int s = 0; s < STAGES; ++s) {
            ProducerBar::init(&producer_mbar[s], 1);
            ConsumerBar::init(&consumer_mbar[s], NTHR);
        }
    }
    cutlass::arch::fence_barrier_init();
    // ★3  cluster 栅栏: 保证所有 CTA 的 barrier 都初始化完了再开始
    cute::cluster_sync();

    int nk = K / BK;
    int k_tile = 0;
    if (warp == 0 && one) {
        for (int s = 0; s < STAGES; ++s) {
            ProducerBar::arrive_and_expect_tx(&producer_mbar[s], txb);
            copy(tma_a.with(producer_mbar[s]), tAgA(_, k_tile), tAsA(_, s));
            copy(tma_b.with(producer_mbar[s]), tBgB(_, k_tile), tBsB(_, s));
            ++k_tile;
        }
    }

    auto write_state = cutlass::PipelineState<STAGES>();
    auto read_state = cutlass::PipelineState<STAGES>();
    int k_tile_count = nk;
    while (k_tile_count > -STAGES) {
        int read_pipe = read_state.index();
        ProducerBar::wait(&producer_mbar[read_pipe], read_state.phase());

        auto tCrA = thr.make_fragment_A(thr.partition_A(sA(_, _, read_pipe)));
        auto tCrB = thr.make_fragment_B(thr.partition_B(sB(_, _, read_pipe)));
        warpgroup_arrive();
        gemm(mma, tCrA, tCrB, tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        ConsumerBar::arrive(&consumer_mbar[read_pipe]);
        ++read_state;

        if (warp == 0 && one && k_tile_count > 0) {
            int pipe = write_state.index();
            ConsumerBar::wait(&consumer_mbar[pipe], write_state.phase());
            ProducerBar::arrive_and_expect_tx(&producer_mbar[pipe], txb);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
            ++k_tile;
            ++write_state;
        }
        --k_tile_count;
    }

    copy(tCrC, thr.partition_C(gC));
}

// ===========================================================================
// §6.3  验证 + 计时
// ===========================================================================
static void run(int M, int N, int K, bool verify) {
    size_t nA = size_t(M) * K, nB = size_t(N) * K, nC = size_t(M) * N;

    half_t *h_A = new half_t[nA], *h_B = new half_t[nB];
    float *h_C = new float[nC], *h_ref = new float[nC];
    fill_pm1(h_A, nA, 56);
    fill_pm1(h_B, nB, 66);
    gemm_cpu(h_A, h_B, h_ref, M, N, K);

    half_t *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, nA * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_B, nB * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_C, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nA * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, nB * sizeof(half_t), cudaMemcpyHostToDevice));

    auto sla = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                             make_shape(Int<BM>{}, Int<BK>{}, Int<STAGES>{}));
    auto slb = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                             make_shape(Int<BN>{}, Int<BK>{}, Int<STAGES>{}));

    // TMA descriptor (host 侧, 真实设备指针)
    auto mA = make_tensor(make_gmem_ptr(d_A), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(d_B), make_shape(N, K), make_stride(K, Int<1>{}));
    auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, mA, sla(_, _, Int<0>{}),
                               make_shape(Int<BM>{}, Int<BK>{}));
    auto tma_b = make_tma_atom(SM90_TMA_LOAD{}, mB, slb(_, _, Int<0>{}),
                               make_shape(Int<BN>{}, Int<BK>{}));

    using Kernel = decltype(&gemm_cluster<decltype(tma_a), decltype(tma_b), decltype(sla),
                                          decltype(slb)>);
    Kernel kptr = &gemm_cluster<decltype(tma_a), decltype(tma_b), decltype(sla), decltype(slb)>;
    size_t smem_bytes = sizeof(SharedStorage<decltype(sla), decltype(slb)>);
    CUDA_CHECK(cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_bytes));

    // ★1  launch: 带 cluster_dims, 走 cutlass::launch_kernel_on_cluster
    dim3 cluster_dims(CLUSTER_N, CLUSTER_M);   // (x, y)
    dim3 grid(N / BN, M / BM);                 // 总 CTA 数 (cluster 的整数倍)
    dim3 block(NTHR);
    cutlass::ClusterLaunchParams params{grid, block, cluster_dims, int(smem_bytes)};
    void const* kptr_v = reinterpret_cast<void const*>(kptr);

    auto launch = [&] {
        cutlass::launch_kernel_on_cluster(params, kptr_v, tma_a, tma_b, d_C, M, N, K, sla, slb);
    };

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
    printf("cute_06 capstone —— Block Cluster (非 multicast 版)\n");
    printf("对应 README §6\n");
    printf("\nCluster = %dx%d, CTA tile = %dx%dx%d, STAGES = %d\n", CLUSTER_M, CLUSTER_N, BM, BN,
           BK, STAGES);

    print_separator("§6.3  正确性 + 实测");
    run(256, 256, 256, true);
    run(512, 256, 512, true);
    run(2048, 2048, 2048, false);

    print_separator("汇总 —— 六版 + cuBLAS");
    printf("\n  %-30s %12s\n", "版本", "TFLOP/s (2048^3)");
    printf("  %-30s %12s\n", "v0 naive", "~11.7");
    printf("  %-30s %12s\n", "v1 smem 单缓冲", "~66.9");
    printf("  %-30s %12s\n", "v2 cp.async 3-stage", "~70.6");
    printf("  %-30s %12s\n", "v3 TMA+WGMMA 3-stage", "~428");
    printf("  %-30s %12s\n", "v4 Warp Spec (手写最小)", "~393");
    printf("  %-30s %12s\n", "capstone cluster", "看上面打印");
    printf("  %-30s %12s\n", "cuBLAS fp16 (参考)", "~878");

    print_separator("诚实总结 —— 差在哪, 还差多少");
    printf("  手写 GEMM 追不上 cuBLAS 是**正常的**, 不是没学会。差在:\n");
    printf("    1. tile 调度 (cuBLAS 用 swizzled rasterization 让 L2 复用最大化)\n");
    printf("    2. epilogue 用 TMA store + 融合算子, 这里还是普通 copy\n");
    printf("    3. setmaxnreg + 完整的 WS 编排\n");
    printf("    4. 多精度 / split-K / 各种 kernel 选择 (cuBLAS 是几十个 kernel 的库)\n");
    printf("\n  但这条路径 (TMA+WGMMA+pipeline+cluster) 就是 CUTLASS 的路径。\n");
    printf("  下一章 (cute_07): Persistent Block —— 把 grid 固定成 SM 个数,\n");
    printf("  每个 CTA 循环吃多个 tile, 省掉反复 launch 的开销, 还能做 tile 调度。\n");

    printf("\ncapstone OK\n");
    return 0;
}
