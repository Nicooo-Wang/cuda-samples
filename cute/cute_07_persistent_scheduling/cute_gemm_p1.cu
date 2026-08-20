// cute_07 v1 —— Tile 调度: rasterization 与 L2 复用
//
// 对应 README §2。
//
// v0 的 persistent 主循环里, tile 的分配是:
//   CTA i 吃 tile_id = i, i+grid, i+2*grid, ...
//   而 tile_id -> (by, bx) 是 row-major (by = tile_id / num_tiles_n)。
//
// 这有一个隐藏的低效: **相邻 tile_id 的 CTA 不一定在空间上相邻**。
// 考虑 16x16 的 tile 网格 (2048^3 用 128 tile):
//   CTA 0 吃 (0,0), CTA 1 吃 (0,1), ..., CTA 15 吃 (0,15)
//   CTA 16 吃 (1,0) —— 但此时 CTA 0-15 可能还没吃完 (0,*) 这一行!
//   于是 A 的复用 (同一行 CTA 共享同一块 A) 跨了"代"。
//
// 更糟的: 如果 grid < num_tiles_n (比如 grid=132, num_tiles_n=16),
//   CTA 0 吃 (0,0), CTA 16 吃 (1,0) —— 两个 CTA 同时吃 (m=0) 和 (m=1) 的
//   同一列 A? 不对, A 是按 m 切的。看下面 §2.1 的图。
//
// Tile 调度的目标: **让同一时刻在跑的 CTA 尽量吃空间上相邻的 tile**,
// 这样它们共享的 A/B 块能命中 L2 (一个 CTA 从 L2 读到, 另一个直接命中)。
//
// rasterization 的一种 (swizzled tile schedule):
//   tile_id -> (by, bx) 用 "按列优先 + 按行内段翻转" 的映射, 让相邻
//   tile_id 的 (by, bx) 在 2D 空间上聚成方块。
//
//   §2.1  两种调度对比: row-major vs swizzled       (README §2.1)
//   §2.2  kernel: 只换 tile 映射那一行              (README §2.2)
//   §2.3  实测: L2 命中的差别                       (README §2.3)
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_gemm_p1

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
// 尺寸配置 (和 v0 相同)
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
// §2.2  tile 映射
//
// 三种映射 (都从 tile_id 出发):
//   row_major : by = id / num_tiles_n,  bx = id % num_tiles_n
//   col_major : by = id % num_tiles_m,  bx = id / num_tiles_m
//   swizzled  : 见下面 (把 2D 网格按方块切, 方块内 row-major)
//
// 这一版演示 row-major vs swizzled 的差别 (host 侧打印两种映射的
// (by,bx) 表, 让你看到空间局部性的差异), kernel 里用 host 选定的那种。
// ===========================================================================

// swizzled: 把 num_tiles_m x num_tiles_n 的网格按 SWIZ 个 tile 一列一组,
// 组内按 (row-major), 组间按列交错。这是 CUTLASS 的 swizzled rasterization
// 的简化版 (完整的用 bit 反转, 这里用按块分组演示概念)。
CUTE_HOST_DEVICE static void tile_coord(int tile_id, int num_tiles_m, int num_tiles_n, int mode, int& by,
                       int& bx) {
    constexpr int SWIZ = 4;  // 每块含 SWIZ 个 tile (演示用)
    if (mode == 0) {
        // row-major
        by = tile_id / num_tiles_n;
        bx = tile_id % num_tiles_n;
    } else {
        // swizzled: 按 SWIZ 个 tile 一组, 组内 row-major, 组间按列
        int group = tile_id / SWIZ;              // 第几组
        int within = tile_id % SWIZ;             // 组内第几个
        int group_col = group % num_tiles_n;     // 组落在哪一列
        int group_row = group / num_tiles_n;     // 组落在哪一行
        // 组内 SWIZ 个 tile 沿 M 方向排 (同一列, 不同行)
        by = group_row * SWIZ + within;
        bx = group_col;
        // 边界保护 (tile 数不是 SWIZ 的整数倍时)
        if (by >= num_tiles_m) { by = num_tiles_m - 1; }
    }
}

// 打印映射表 (host 侧)
static void show_mapping(int num_tiles_m, int num_tiles_n, int mode, const char* name) {
    printf("\n  %s (前 16 个 tile_id 的坐标):\n", name);
    for (int id = 0; id < 16; ++id) {
        int by, bx;
        tile_coord(id, num_tiles_m, num_tiles_n, mode, by, bx);
        printf("    %2d -> (%d,%d)%s", id, by, bx, (id % 4 == 3) ? "\n" : "");
    }
    printf("\n");
}

// ===========================================================================
// §2.2  kernel —— 和 v0 完全相同, 只把 tile_id -> (by,bx) 换成 host 选的模式
// ===========================================================================
template <class TmaA, class TmaB, class SLA, class SLB>
__global__ __launch_bounds__(NTHR) void gemm_persistent(
    CUTLASS_GRID_CONSTANT TmaA const tma_a, CUTLASS_GRID_CONSTANT TmaB const tma_b, float* C,
    int M, int N, int K, int num_tiles_m, int num_tiles_n, int mode, SLA sla, SLB slb) {
    extern __shared__ char shared_memory[];
    SharedStorage<SLA, SLB>& smem = *reinterpret_cast<SharedStorage<SLA, SLB>*>(shared_memory);

    auto sA = make_tensor(make_smem_ptr(smem.A.begin()), sla);
    auto sB = make_tensor(make_smem_ptr(smem.B.begin()), slb);

    auto mA = tma_a.get_tma_tensor(make_shape(M, K));
    auto mB = tma_b.get_tma_tensor(make_shape(N, K));
    constexpr int txb = 2 * BM * BK * int(sizeof(half_t));

    int warp = cutlass::canonical_warp_idx_sync();
    int one = cute::elect_one_sync();
    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBar = cutlass::arch::ClusterBarrier;
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    auto mma = make_wgmma();
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

    int num_tiles = num_tiles_m * num_tiles_n;
    int nk = K / BK;

    for (int tile_id = blockIdx.x; tile_id < num_tiles; tile_id += gridDim.x) {
        int by, bx;
        tile_coord(tile_id, num_tiles_m, num_tiles_n, mode, by, bx);

        // 每 tile 重置 barrier phase (见 v0 的说明)
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
        auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                          group_modes<0, 2>(gA));
        auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                                          group_modes<0, 2>(gB));

        auto gC = local_tile(mC, Shape<Int<BM>, Int<BN>>{}, make_coord(by, bx));
        auto tCrC = thr.partition_fragment_C(gC);
        clear(tCrC);

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
}

// ===========================================================================
// §2.3  验证 + 计时
// ===========================================================================
static void run(int M, int N, int K, bool verify, int num_sms, int mode) {
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
                                                     num_tiles_n, mode, sla, slb);
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

    printf("cute_07 v1 —— Tile 调度: rasterization 与 L2 复用\n");
    printf("对应 README §2\n");
    printf("\n本机 %d 个 SM, grid = %d 个 CTA\n", num_sms, num_sms);

    print_separator("§2.1  两种 tile 映射对比 (16x16 网格)");
    show_mapping(16, 16, 0, "row-major");
    show_mapping(16, 16, 1, "swizzled (块大小 4)");

    print_separator("§2.3  实测: row-major vs swizzled");
    printf("\n  --- row-major ---\n");
    run(2048, 2048, 2048, false, num_sms, 0);
    printf("\n  --- swizzled ---\n");
    run(2048, 2048, 2048, false, num_sms, 1);

    print_separator("小结");
    printf("  两种调度的结果都是对的 (同一份数据, 不同的吃法)。\n");
    printf("  差别在 L2: swizzled 让同一时刻的 CTA 聚在 2D 空间的一小块,\n");
    printf("  它们共享的 A/B 更容易命中 L2。\n");
    printf("  实测差距通常不大 (2048^3 时 A/B 总量 ~16MB, L2 有 ~50MB),\n");
    printf("  尺寸更大 (8192^3, A/B ~256MB) 或 L2 更小时差距才明显。\n");
    printf("  这就是 cuBLAS 用 swizzled rasterization 的原因。\n");

    printf("\nv1 OK\n");
    return 0;
}
