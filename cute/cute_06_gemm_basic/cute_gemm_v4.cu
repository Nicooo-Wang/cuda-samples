// cute_06 v4 —— Warp Specialization: 谁搬, 谁算
//
// 对应 README §5。
//
// 回顾 v3 的 mainloop: 128 个线程**既发 TMA 又做 WGMMA**。
// 发 TMA 的只有 1 个 lane, 其余 127 个在等; 然后 128 个一起做 WGMMA。
// 这就是"线程分工"的问题:
//
//    同一批线程, 既当搬运工又当计算工, 两件事串在一条指令流上。
//
// Warp Specialization 的答案: **把线程分成两组, 各干各的**。
//   producer 组: 只发 TMA (一路抢跑, 把流水线灌满)
//   consumer 组: 只做 WGMMA (一路算, 不关心数据从哪来)
//   两组用 mbarrier 通信 (就是 v3 那两组 barrier)。
//
// 为什么有用? 两点:
//   1. consumer 不再有"发 TMA"的指令混在指令流里 —— 指令流更干净, 更利于
//      编译器安排 WGMMA (寄存器压力也降了: producer 不用持有累加器)。
//   2. producer 可以**提前发**下一批 (它没有别的活), 流水线更容易灌满。
//
// 这一版是**手写的最小 WS**: 256 线程 = 2 个 warpgroup,
//   warps 0-3  (wg0): consumer —— WGMMA
//   warps 4-7  (wg1): producer —— TMA
//
// 注意两件事 (都是 cute_04 §5 踩过的坑):
//   - consumer 必须落在 wg0 (warps 0-3), producer 在 wg1。
//     反过来 warpgroup_arrive/commit 会出问题。
//   - producer 的 TMA 只能由 1 个 lane 发, 而且必须限定在 producer 组:
//     elect_one_sync 是**每个 warp 选一个** -> 要 (one && warp == 4)。
//
//   §5.1  分工图 + 谁在等谁                  (README §5.1)
//   §5.2  kernel: producer/consumer mainloop  (README §5.2)
//   §5.3  实测: 和 v3 比                      (README §5.3)
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_gemm_v4

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
// 和 v3 相同 (128x128x64, STAGES=3), 只是线程数翻倍:
//   256 线程 = 2 个 warpgroup。
//   wg0 (warps 0-3)  = consumer: 做 WGMMA
//   wg1 (warps 4-7)  = producer: 发 TMA
// ---------------------------------------------------------------------------
constexpr int BM = 128, BN = 128, BK = 64;
constexpr int NTHR = 256;              // 2 个 warpgroup
constexpr int STAGES = 4;              // producer 抢跑, 可以多灌几批
constexpr int PRODUCER_WARP = 4;       // producer 的 leader warp (wg1 的第一个)

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
// §5.2  kernel
//
// producer (wg1) 的代码路径和 consumer (wg0) 完全不同, 用一个 if (warp_group)
// 分开。两组各自转自己的 PipelineState, 通过 barrier 通信。
// ===========================================================================
template <class TmaA, class TmaB, class SLA, class SLB>
__global__ __launch_bounds__(NTHR) void gemm_ws(
    CUTLASS_GRID_CONSTANT TmaA const tma_a, CUTLASS_GRID_CONSTANT TmaB const tma_b, float* C,
    int M, int N, int K, SLA sla, SLB slb) {
    extern __shared__ char shared_memory[];
    SharedStorage<SLA, SLB>& smem = *reinterpret_cast<SharedStorage<SLA, SLB>*>(shared_memory);

    auto sA = make_tensor(make_smem_ptr(smem.A.begin()), sla);  // (BM,BK,STAGES)
    auto sB = make_tensor(make_smem_ptr(smem.B.begin()), slb);  // (BN,BK,STAGES)

    auto mA = tma_a.get_tma_tensor(make_shape(M, K));
    auto mB = tma_b.get_tma_tensor(make_shape(N, K));
    auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(blockIdx.y, _));
    auto gB = local_tile(mB, Shape<Int<BN>, Int<BK>>{}, make_coord(blockIdx.x, _));

    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                      group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                                      group_modes<0, 2>(gB));
    constexpr int txb = sizeof(make_tensor_like(tensor<0>(tAsA)))
                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

    int warp = cutlass::canonical_warp_idx_sync();   // 0..7
    int one = cute::elect_one_sync();                // 每个 warp 一个 lane
    bool is_producer = (warp >= PRODUCER_WARP);      // wg1 = producer
    bool issue_tma = (one && warp == PRODUCER_WARP); // 全 block 只有 1 个 lane 发 TMA

    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBar = cutlass::arch::ClusterBarrier;
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    // barrier 初始化: 任意一个 warp 做都行, 但要全 block 同步过
    if (warp == 0 && one) {
        for (int s = 0; s < STAGES; ++s) {
            ProducerBar::init(&producer_mbar[s], 1);
            ConsumerBar::init(&consumer_mbar[s], NTHR / 2);  // consumer 只有 128 线程
        }
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    int nk = K / BK;
    auto write_state = cutlass::PipelineState<STAGES>();  // producer 的
    auto read_state = cutlass::PipelineState<STAGES>();   // consumer 的
    int k_tile = 0;

    if (is_producer) {
        // ============ producer: 只发 TMA, 一路抢跑 ============
        // prologue: 一口气灌满 (不等人算)
        if (issue_tma) {
            for (int s = 0; s < STAGES; ++s) {
                ProducerBar::arrive_and_expect_tx(&producer_mbar[s], txb);
                copy(tma_a.with(producer_mbar[s]), tAgA(_, k_tile), tAsA(_, s));
                copy(tma_b.with(producer_mbar[s]), tBgB(_, k_tile), tBsB(_, s));
                ++k_tile;
            }
        }
        // mainloop: 等 consumer 释放 stage, 再补发
        while (k_tile < nk) {
            int pipe = write_state.index();
            ConsumerBar::wait(&consumer_mbar[pipe], write_state.phase());
            if (issue_tma) {
                ProducerBar::arrive_and_expect_tx(&producer_mbar[pipe], txb);
                copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
                copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
            }
            ++k_tile;
            ++write_state;
        }
    } else {
        // ============ consumer: 只做 WGMMA ============
        auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));
        auto gC = local_tile(mC, Shape<Int<BM>, Int<BN>>{}, make_coord(blockIdx.y, blockIdx.x));
        auto mma = make_wgmma();
        ThrMMA thr = mma.get_thread_slice(threadIdx.x);
        auto tCrC = thr.partition_fragment_C(gC);
        clear(tCrC);

        // 算够 nk 批, 每批一个 stage
        for (int k = 0; k < nk; ++k) {
            int pipe = read_state.index();
            ProducerBar::wait(&producer_mbar[pipe], read_state.phase());

            auto tCrA = thr.make_fragment_A(thr.partition_A(sA(_, _, pipe)));
            auto tCrB = thr.make_fragment_B(thr.partition_B(sB(_, _, pipe)));
            warpgroup_arrive();
            gemm(mma, tCrA, tCrB, tCrC);
            warpgroup_commit_batch();
            warpgroup_wait<0>();

            ConsumerBar::arrive(&consumer_mbar[pipe]);  // 释放给 producer
            ++read_state;
        }

        copy(tCrC, thr.partition_C(gC));
    }
}

// ===========================================================================
// §5.3  验证 + 计时
// ===========================================================================
static void run(int M, int N, int K, bool verify) {
    size_t nA = size_t(M) * K, nB = size_t(N) * K, nC = size_t(M) * N;

    half_t *h_A = new half_t[nA], *h_B = new half_t[nB];
    float *h_C = new float[nC], *h_ref = new float[nC];
    fill_pm1(h_A, nA, 55);
    fill_pm1(h_B, nB, 65);
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

    auto mA = make_tensor(make_gmem_ptr(d_A), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(d_B), make_shape(N, K), make_stride(K, Int<1>{}));
    auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, mA, sla(_, _, Int<0>{}),
                               make_shape(Int<BM>{}, Int<BK>{}));
    auto tma_b = make_tma_atom(SM90_TMA_LOAD{}, mB, slb(_, _, Int<0>{}),
                               make_shape(Int<BN>{}, Int<BK>{}));

    using Kernel = decltype(&gemm_ws<decltype(tma_a), decltype(tma_b), decltype(sla),
                                     decltype(slb)>);
    Kernel kptr = &gemm_ws<decltype(tma_a), decltype(tma_b), decltype(sla), decltype(slb)>;
    size_t smem_bytes = sizeof(SharedStorage<decltype(sla), decltype(slb)>);
    CUDA_CHECK(cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_bytes));

    dim3 grid(N / BN, M / BM), block(NTHR);
    auto launch = [&] { gemm_ws<<<grid, block, smem_bytes>>>(tma_a, tma_b, d_C, M, N, K, sla,
                                                             slb); };

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
    printf("cute_06 v4 —— Warp Specialization: 谁搬, 谁算\n");
    printf("对应 README §5\n");
    printf("\n%d 线程 = 2 warpgroup: wg0 consumer (WGMMA) + wg1 producer (TMA)\n", NTHR);
    printf("CTA tile = %dx%dx%d, STAGES = %d\n", BM, BN, BK, STAGES);

    print_separator("§5.3  正确性 + 实测");
    run(256, 256, 256, true);
    run(512, 256, 512, true);
    run(2048, 2048, 2048, false);

    print_separator("小结");
    printf("  v3 (128 线程, 又搬又算) 2048^3: ~421 TFLOP/s\n");
    printf("  v4 (WS, 256 线程分工)   2048^3: 看上面打印 (诚实说: 未必更快)\n");
    printf("\n  为什么手写最小 WS 不一定会赢?\n");
    printf("    1. 256 线程 = 2 个 warpgroup 挤一个 SM, 占用率降了\n");
    printf("    2. consumer 只有 128 线程算, producer 在 128x128 tile 上\n");
    printf("       抢跑的空间不大 (TMA 本来就够快)\n");
    printf("    3. 真正的 WS 收益靠 setmaxnreg (producer 少给寄存器) +\n");
    printf("       更大的 tile + 更细的 epilogue 分工, 那需要 CUTLASS 全套\n");
    printf("  所以: 概念上 WS 是对的 (谁搬谁算分开), 工程上要全套才兑现。\n");
    printf("  这也是 CUTLASS 官方 WS kernel 结构复杂的原因。\n");
    printf("\n  下一步 (capstone): Block Cluster + TMA multicast ——\n");
    printf("  让 cluster 里多个 CTA 共享同一块 A, 省一半 gmem 流量。\n");

    printf("\nv4 OK\n");
    return 0;
}
