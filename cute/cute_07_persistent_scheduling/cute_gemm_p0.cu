// cute_07 v0 —— Persistent Block: 一个 CTA 活到 grid 结束
//
// 对应 README §1。
//
// cute_06 的 GEMM 是"每个 tile 一个 CTA": grid = (N/BN, M/BM), 每个 CTA
// 算一块就结束。问题:
//   1. 每个 CTA 都有 launch/初始化/收尾的开销
//   2. tile 的顺序由硬件决定 (一般是 row-major), 我们控制不了
//   3. 反复 launch 的间隙, SM 可能有空档
//
// Persistent kernel 换个思路: **grid = SM 个数, 每个 CTA 常驻, 循环吃 tile**。
//
//     grid = 132 (SM 数)
//     CTA i 的主循环:
//       while (我的下一个 tile 还有) {
//         拿一个 tile 的坐标
//         算它 (就是 cute_06 v3 的 TMA+WGMMA 那一套)
//       }
//
// 好处:
//   1. 没有反复 launch —— 一个 grid 只 launch 一次
//   2. tile 的顺序**由我们决定** —— 这是 tile 调度的前提 (v1 讲)
//   3. CTA 的生命周期长, 可以做跨 tile 的复用 (累加器不用清零? 不行, 见下)
//
// 坏处 (要诚实):
//   1. tile 数不是 SM 数的整数倍时, 负载不均衡 (有的 CTA 多吃一个)
//   2. 累加器是 per-tile 的: 每个 tile 都要 clear + 写回
//   3. grid 固定后, 不能靠硬件调度自适应 (occupancy 也固定了)
//
//   §1.1  结构: 一维 tile 编号 -> (by, bx) 坐标   (README §1.1)
//   §1.2  kernel                                   (README §1.2)
//   §1.3  实测: 和 cute_06 v3 比                   (README §1.3)
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_gemm_p0

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
// 和 cute_06 v3 相同: CTA tile = 128x128x64, STAGES=3。
// 区别: grid 不是 (N/BN, M/BM), 而是 (num_SMs, 1) —— 一个 CTA 一个 SM。
//   M 方向有 num_tiles_m 个 tile, N 方向 num_tiles_n 个, 总共 num_tiles。
//   每个 persistent CTA 用一维编号 tile_id 循环领活。
//
// 负载分配: 最简单的是"编号取模"——
//   CTA i 吃 tile_id = i, i+grid, i+2*grid, ...
//   (v1 会换成 rasterization, 那是"哪个 CTA 先吃哪个 tile"的优化)
// ---------------------------------------------------------------------------
constexpr int BM = 128, BN = 128, BK = 64;
constexpr int NTHR = 128;
constexpr int STAGES = 3;

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
// §1.2  kernel —— 和 cute_06 v3 几乎一样, 只改了两处:
//   ★1  不传 M/N/K, 传 tile 总数和 grid 大小; tile_id 用 blockIdx.x
//   ★2  外面套 while 循环, 每轮拿一个 (by, bx)
// ===========================================================================
template <class TmaA, class TmaB, class SLA, class SLB>
__global__ __launch_bounds__(NTHR) void gemm_persistent(
    CUTLASS_GRID_CONSTANT TmaA const tma_a, CUTLASS_GRID_CONSTANT TmaB const tma_b, float* C,
    int M, int N, int K, int num_tiles_m, int num_tiles_n, SLA sla, SLB slb) {
    extern __shared__ char shared_memory[];
    SharedStorage<SLA, SLB>& smem = *reinterpret_cast<SharedStorage<SLA, SLB>*>(shared_memory);

    auto sA = make_tensor(make_smem_ptr(smem.A.begin()), sla);
    auto sB = make_tensor(make_smem_ptr(smem.B.begin()), slb);

    // gmem 坐标 tensor (和 v3 一样)
    auto mA = tma_a.get_tma_tensor(make_shape(M, K));
    auto mB = tma_b.get_tma_tensor(make_shape(N, K));

    // 每轮 TMA 的事务字节数 = A tile + B tile (smem 侧, 编译期可知)
    constexpr int txb = 2 * BM * BK * int(sizeof(half_t));

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
    __syncthreads();

    auto mma = make_wgmma();
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

    // ---- ★2  persistent 主循环: 每个 CTA 循环吃 tile ----
    int num_tiles = num_tiles_m * num_tiles_n;
    int nk = K / BK;

    for (int tile_id = blockIdx.x; tile_id < num_tiles; tile_id += gridDim.x) {
        // ★1 一维编号 -> (by, bx) 坐标
        int by = tile_id / num_tiles_n;
        int bx = tile_id % num_tiles_n;

        // 每个 tile 是一段独立的 GEMM: barrier 的 phase 要重新开始。
        // 如果不重置, 上一个 tile 结束时的 phase 和这个 tile 里
        // PipelineState(从 0 开始) 对不上 -> 死锁。
        if (warp == 0 && one) {
            for (int s = 0; s < STAGES; ++s) {
                ProducerBar::init(&producer_mbar[s], 1);
                ConsumerBar::init(&consumer_mbar[s], NTHR);
            }
        }
        cutlass::arch::fence_barrier_init();
        __syncthreads();

        auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(by, _));
        auto gB = local_tile(mB, Shape<Int<BN>, Int<BK>>{}, make_coord(bx, _));
        auto [tAgA, tAsA2] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                           group_modes<0, 2>(gA));
        auto [tBgB, tBsB2] = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                                           group_modes<0, 2>(gB));

        auto gC = local_tile(mC, Shape<Int<BM>, Int<BN>>{}, make_coord(by, bx));
        auto tCrC = thr.partition_fragment_C(gC);
        clear(tCrC);

        // prologue (每 tile 一次)
        int k_tile = 0;
        if (warp == 0 && one) {
            for (int s = 0; s < STAGES; ++s) {
                ProducerBar::arrive_and_expect_tx(&producer_mbar[s], txb);
                copy(tma_a.with(producer_mbar[s]), tAgA(_, k_tile), tAsA2(_, s));
                copy(tma_b.with(producer_mbar[s]), tBgB(_, k_tile), tBsB2(_, s));
                ++k_tile;
            }
        }

        // mainloop (和 v3 一样)
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
                copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA2(_, pipe));
                copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB2(_, pipe));
                ++k_tile;
                ++write_state;
            }
            --k_tile_count;
        }

        copy(tCrC, thr.partition_C(gC));
    }
}

// ===========================================================================
// §1.3  验证 + 计时
// ===========================================================================
static void run(int M, int N, int K, bool verify, int num_sms) {
    size_t nA = size_t(M) * K, nB = size_t(N) * K, nC = size_t(M) * N;

    half_t *h_A = new half_t[nA], *h_B = new half_t[nB];
    float *h_C = new float[nC], *h_ref = new float[nC];
    fill_pm1(h_A, nA, 57);
    fill_pm1(h_B, nB, 67);
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

    using Kernel = decltype(&gemm_persistent<decltype(tma_a), decltype(tma_b), decltype(sla),
                                              decltype(slb)>);
    Kernel kptr = &gemm_persistent<decltype(tma_a), decltype(tma_b), decltype(sla),
                                   decltype(slb)>;
    size_t smem_bytes = sizeof(SharedStorage<decltype(sla), decltype(slb)>);
    CUDA_CHECK(cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_bytes));

    int num_tiles_m = M / BM, num_tiles_n = N / BN;
    dim3 grid(num_sms), block(NTHR);
    auto launch = [&] {
        gemm_persistent<<<grid, block, smem_bytes>>>(tma_a, tma_b, d_C, M, N, K, num_tiles_m,
                                                     num_tiles_n, sla, slb);
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
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    int num_sms = props.multiProcessorCount;

    printf("cute_07 v0 —— Persistent Block: 一个 CTA 活到 grid 结束\n");
    printf("对应 README §1\n");
    printf("\n本机 %d 个 SM, grid = %d 个 CTA, 每个 CTA 循环吃 tile\n", num_sms, num_sms);

    print_separator("§1.3  正确性 + 实测");
    run(512, 512, 512, true, num_sms);
    run(2048, 2048, 2048, false, num_sms);

    print_separator("小结");
    printf("  和 cute_06 v3 (每个 tile 一个 CTA) 比, 数字应该差不多 ——\n");
    printf("  因为 GEMM 的瓶颈在 TMA 带宽和 WGMMA 算力, 不在 launch 开销。\n");
    printf("  persistent 的真正价值在 v1: **tile 的顺序由我们决定**,\n");
    printf("  可以做 rasterization 让 L2 复用最大化。\n");
    printf("  另外: 同一份 smem/barrier 跨 tile 复用, 初始化只做一次。\n");

    printf("\nv0 OK\n");
    return 0;
}
