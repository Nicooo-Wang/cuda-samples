// cute_06 v3 —— TMA + WGMMA + mbarrier 多 stage: Hopper 的正解
//
// 对应 README §4。
//
// v2 用 cp.async 把搬和算重叠了, 但 cp.async 仍然:
//   - 占指令流 (每线程要发 load)
//   - 占寄存器 (in-flight 数据 + 地址)
//   - 同步靠 cp_async_wait + __syncthreads, 粒度粗
//
// Hopper 把这两件事都换成**硬件异步引擎**:
//   TMA    : 一个 lane 发一条指令描述整块, 硬件后台搬
//   WGMMA  : 一个 warpgroup 发一条指令, 硬件直接读 smem 累加
//   同步   : mbarrier (producer/consumer), 按字节数等
//
// 这一版 = cute_04 §5 跑通的单 CTA 多 stage, 铺满整个 grid。
// cute_04 用的是手写 barrier + k&1 phase; 这一版用 cutlass::PipelineState
// 的正式写法 —— 它是 CUTLASS 官方 wgmma_tma_sm90.cu 的结构。
//
//   §4.1  数据流: TMA -> smem(多份) -> WGMMA -> 累加器   (README §4.1)
//   §4.2  同步: producer/consumer 双 barrier + PipelineState (README §4.2)
//   §4.3  实测: 和 v2 比, 快多少                        (README §4.3)
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_gemm_v3

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
// gmem (TN 摆法):
//   A: M x K half, row-major, stride = (K,1)
//   B: N x K half, row-major, stride = (K,1)   <- 存 B^T
//   C: M x N float, row-major, stride = (N,1)
//
// CTA tile: BM=128, BN=128, BK=64。
//   BM=128: WGMMA 原子 M=64, CuTe 自动沿 M 发 2 次 (cute_05 §5.1)
//   BN=128: 同上沿 N 发 2 次
//   BK=64 : GMMA::Layout_K_SW128_Atom 要求 K % 64 == 0
//
// 每个 CTA 128 线程 (1 个 warpgroup)。STAGES=3。
//
// 注意: 128x128x64 的 A 是 16KB, B 也是 16KB, 3 stage 就是 96KB > 48KB
// 静态 smem 上限。所以这一版用 **extern __shared__ + cudaFuncSetAttribute**
// (cute_04 §5.3 的台阶)。SharedStorage 结构体和官方 wgmma_tma_sm90.cu 一致。
// ---------------------------------------------------------------------------
constexpr int BM = 128, BN = 128, BK = 64;
constexpr int NTHR = 128;
constexpr int STAGES = 3;

CUTE_HOST_DEVICE static auto make_wgmma() {
    return make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
}

// smem 结构体: A/B 各 STAGES 份 + 两组 barrier (producer 用 TMA 的, consumer 用 MMA 的)
template <class SLA, class SLB>
struct SharedStorage {
    alignas(128) cute::ArrayEngine<half_t, cosize_v<SLA>> A;
    alignas(128) cute::ArrayEngine<half_t, cosize_v<SLB>> B;
    uint64_t tma_barrier[STAGES];
    uint64_t mma_barrier[STAGES];
};

// ===========================================================================
// §4.1+4.2  kernel —— 官方 wgmma_tma_sm90.cu 的结构
//
// mainloop 的核心是**两个循环 index 独立转**:
//   read_state  (consumer): 等 producer 的 barrier, 做 WGMMA, 通知 consumer barrier
//   write_state (producer): 等 consumer 的 barrier, 发 TMA, 通知 producer barrier
//
// 注意两个 PipelineState **都从 0 开始**, 不要在 prologue 里预推进 ——
// 预推进会死锁 (cute_04 §5.2 的坑)。
// ===========================================================================
template <class TmaA, class TmaB, class SLA, class SLB>
__global__ __launch_bounds__(NTHR) void gemm_tma_wgmma(
    CUTLASS_GRID_CONSTANT TmaA const tma_a, CUTLASS_GRID_CONSTANT TmaB const tma_b, float* C,
    int M, int N, int K, SLA sla, SLB slb) {
    // ---- 动态 smem (48KB 装不下 3 stage 的 128x128x64, 见文件头) ----
    extern __shared__ char shared_memory[];
    SharedStorage<SLA, SLB>& smem = *reinterpret_cast<SharedStorage<SLA, SLB>*>(shared_memory);

    auto sA = make_tensor(make_smem_ptr(smem.A.begin()), sla);  // (BM,BK,STAGES)
    auto sB = make_tensor(make_smem_ptr(smem.B.begin()), slb);  // (BN,BK,STAGES)

    // ---- gmem 坐标 tensor + local_tile (cute_05 capstone 的 grid 那层) ----
    auto mA = tma_a.get_tma_tensor(make_shape(M, K));
    auto mB = tma_b.get_tma_tensor(make_shape(N, K));
    auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(blockIdx.y, _));
    auto gB = local_tile(mB, Shape<Int<BN>, Int<BK>>{}, make_coord(blockIdx.x, _));

    // ---- TMA partition ----
    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                      group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                                      group_modes<0, 2>(gB));
    constexpr int txb = sizeof(make_tensor_like(tensor<0>(tAsA)))
                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

    // ---- 累加器 ----
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));
    auto gC = local_tile(mC, Shape<Int<BM>, Int<BN>>{}, make_coord(blockIdx.y, blockIdx.x));
    auto mma = make_wgmma();
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto tCrC = thr.partition_fragment_C(gC);
    clear(tCrC);

    // ---- barrier 初始化 (每个 stage 一组) ----
    int warp = cutlass::canonical_warp_idx_sync();
    int one = cute::elect_one_sync();
    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;  // TMA 用
    using ConsumerBar = cutlass::arch::ClusterBarrier;             // MMA 用
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;
    if (warp == 0 && one) {
        for (int s = 0; s < STAGES; ++s) {
            ProducerBar::init(&producer_mbar[s], 1);
            ConsumerBar::init(&consumer_mbar[s], NTHR);
        }
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    // ---- prologue: 把 STAGES 批全发出去 (灌满流水线) ----
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

    // ---- mainloop: 双 state 独立转 ----
    auto write_state = cutlass::PipelineState<STAGES>();  // 都从 0, 不要 ++
    auto read_state = cutlass::PipelineState<STAGES>();

    int k_tile_count = nk;
    while (k_tile_count > -STAGES) {
        // ===== consumer: 等这批数据到齐, 算, 释放 =====
        int read_pipe = read_state.index();
        ProducerBar::wait(&producer_mbar[read_pipe], read_state.phase());

        auto tCrA = thr.make_fragment_A(thr.partition_A(sA(_, _, read_pipe)));
        auto tCrB = thr.make_fragment_B(thr.partition_B(sB(_, _, read_pipe)));
        warpgroup_arrive();
        gemm(mma, tCrA, tCrB, tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        ConsumerBar::arrive(&consumer_mbar[read_pipe]);  // 这批算完了, 释放给 producer
        ++read_state;

        // ===== producer: 等对应 stage 空了, 发下一批 =====
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

    // ---- 写回 ----
    copy(tCrC, thr.partition_C(gC));
}

// ===========================================================================
// §4.3  验证 + 计时
// ===========================================================================
static void run(int M, int N, int K, bool verify) {
    size_t nA = size_t(M) * K, nB = size_t(N) * K, nC = size_t(M) * N;

    half_t *h_A = new half_t[nA], *h_B = new half_t[nB];
    float *h_C = new float[nC], *h_ref = new float[nC];
    fill_pm1(h_A, nA, 54);
    fill_pm1(h_B, nB, 64);
    gemm_cpu(h_A, h_B, h_ref, M, N, K);

    half_t *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, nA * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_B, nB * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_C, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nA * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, nB * sizeof(half_t), cudaMemcpyHostToDevice));

    // smem layout: GMMA SW128 atom, 带 PIPE mode (TMA 条件 4)
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

    // 动态 smem + cudaFuncSetAttribute (48KB 台阶)
    using Kernel = decltype(&gemm_tma_wgmma<decltype(tma_a), decltype(tma_b), decltype(sla),
                                            decltype(slb)>);
    Kernel kptr = &gemm_tma_wgmma<decltype(tma_a), decltype(tma_b), decltype(sla), decltype(slb)>;
    size_t smem_bytes = sizeof(SharedStorage<decltype(sla), decltype(slb)>);
    CUDA_CHECK(cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_bytes));

    dim3 grid(N / BN, M / BM), block(NTHR);
    auto launch = [&] { gemm_tma_wgmma<<<grid, block, smem_bytes>>>(tma_a, tma_b, d_C, M, N, K,
                                                                    sla, slb); };

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
    printf("cute_06 v3 —— TMA + WGMMA + mbarrier 多 stage (Hopper 正解)\n");
    printf("对应 README §4\n");
    printf("\nCTA tile = %dx%dx%d, STAGES = %d, 动态 smem\n", BM, BN, BK, STAGES);

    print_separator("§4.3  正确性 + 实测");
    run(256, 256, 256, true);
    run(512, 256, 512, true);
    run(2048, 2048, 2048, false);

    print_separator("小结");
    printf("  v2 (cp.async 3-stage) 2048^3: ~70 TFLOP/s\n");
    printf("  v3 (TMA+WGMMA)        2048^3: 看上面打印\n");
    printf("  快在哪: TMA 不占指令流/寄存器, WGMMA 一次算 64x64,\n");
    printf("          mbarrier 让同步粒度从全 block 变成按 stage。\n");
    printf("  还没用上: warp 分工 (谁发 TMA 谁算) 和 cluster。\n");
    printf("\n  下一步 (v4): Warp Specialization —— 把 128 线程拆成\n");
    printf("  producer (只发 TMA) + consumer (只算 WGMMA)。\n");

    printf("\nv3 OK\n");
    return 0;
}
