// cute_06 练习：把 TODO 填掉，然后 make run
//
// 题目见 README 的"练习"一节。参考解答在 solutions.md。
// 每题都有一个自动检查，填对了会打印 PASS。
//
// 六道题对应 README 的:
//   1 -> §1   从 fragment 数出 MMA 的覆盖范围
//   2 -> §1.3 为什么 naive 慢 (A/B 被重复读几次)
//   3 -> §2.2 改错: ldmatrix 缺 Tile 排列
//   4 -> §3   手写 cp.async 双缓冲的 prologue
//   5 -> §4   手写 TMA+WGMMA mainloop 的 barrier 部分
//   6 -> §6   手写一个 cluster 的最小 kernel

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/device_kernel.h>
#include <cstdio>
#include <cuda_runtime.h>

using namespace cute;

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

static int g_pass = 0, g_fail = 0;

void expect(const char* what, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    ok ? ++g_pass : ++g_fail;
}

// ---------------------------------------------------------------------------
// 练习 1 —— 从 fragment 数出 MMA 的覆盖范围 (README §1)
//
// 一个 SM80 MMA 原子是 16x8x16, 32 线程。
// make_tiled_mma(atom, Layout<Shape<_4,_1,_1>>{}) 会沿 M 重复 4 次。
// 请算出:
//   ① 这个 TiledMMA 用几个线程?  (4 个 warp = ?)
//   ② 它覆盖的 C tile 是多大?    (M = 16*4 = ?)
// ---------------------------------------------------------------------------
static void ex1() {
    printf("\n===== 练习 1: 数 MMA 覆盖范围 =====\n");

    auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_4, _1, _1>>{});
    int threads = int(size(mma));
    printf("  size(mma) = %d\n", threads);

    // TODO ①: 线程数 = 4 个 warp
    int threads_expect = 0;

    // 覆盖范围从 **线程排布** 看: thr_layout 的 (M,N) 分量 = 沿 M/N 重复几次
    auto thr_layout = make_layout(mma.get_thread_slice(0).thr_vmnk_);
    int m_atoms = int(size<0>(thr_layout));  // 沿 M 的原子数 (这里是 4)
    int n_atoms = int(size<1>(thr_layout));  // 沿 N 的原子数 (这里是 1)
    printf("  M 方向 %d 个原子, N 方向 %d 个原子 -> 覆盖 %dx%d\n", m_atoms, n_atoms,
           16 * m_atoms, 8 * n_atoms);

    // TODO ②: M 方向覆盖
    int m_cover = 0;

    expect("用 4 个 warp (128 线程)", threads == threads_expect && threads_expect == 128);
    expect("M 方向覆盖 64", m_cover == 64);
}

// ---------------------------------------------------------------------------
// 练习 2 —— naive 慢在哪: A/B 被重复读几次 (README §1.3)
//
// 假设 naive 版: 每个 CTA 算 C 的 64x8, grid = (N/8, M/64)。
// 问: 一个 A 元素 (A[m][k]) 被多少个 CTA 读?
//
// 思路: A[m][k] 属于哪一行的 CTA? CTA (by, bx) 负责的 A 块是
//   A[by*64 : (by+1)*64][bx*8 : (bx+1)*8]  <- 不对, naive 的 gA 是
//   local_tile(mA, (64,8), make_coord(by, _)) 沿 K 切
//   实际上 A[m][k] 在 CTA (by=m/64, bx=*) 里被用到: bx 可以取 0..N/8-1
//   -> 被 N/8 个 CTA 读。
// 请填: 如果 N=2048, 一个 A 元素被读几次?
// ---------------------------------------------------------------------------
static void ex2() {
    printf("\n===== 练习 2: naive 的重复读 =====\n");

    constexpr int N = 2048, BN = 8;
    int reuse = N / BN;  // 一个 A 元素被多少 CTA 读

    printf("  naive 版: 一个 A 元素被 N/BN = %d/%d = %d 个 CTA 读\n", N, BN, reuse);
    printf("  B 元素同理: 一个 B 元素被 M/BM 个 CTA 读。\n");
    printf("  smem 中转版: 每个 CTA 只读一次, 之后 K 循环全在 smem 里。\n");

    // TODO ①: 如果 BN 改成 64 (CTA 算 64x64), 重复读从 256 降到多少?
    int reuse_big = N / 64;

    expect("BN=64 时重复读 = 32", reuse_big == 32);
}

// ---------------------------------------------------------------------------
// 练习 3 —— 改错: ldmatrix 缺 Tile 排列 (README §2.2)
//
// 下面 make_mma() 缺了第三个参数 Tile<_32,_32,_16>{} (排列)。
// 没有它, make_tiled_copy_A(ldm, mma) 会报 "TiledCopy uses too few vals"。
// 请把第三个参数补上。
// ---------------------------------------------------------------------------
static void ex3() {
    printf("\n===== 练习 3: ldmatrix 的 Tile 排列 =====\n");

    // TODO ①: 补上第三个参数 Tile<_32,_32,_16>{}
    auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_2, _2>>{});

    auto slay = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                              make_shape(Int<64>{}, Int<64>{}));
    auto sA = make_tensor(make_smem_ptr((half_t*)nullptr), slay);

    // 这行能编译说明 Tile 排列让 ldmatrix 对齐了
    auto s2r = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
    auto tl = s2r.get_slice(0);
    auto tCrA = mma.get_thread_slice(0).make_fragment_A(
        mma.get_thread_slice(0).partition_A(sA));
    auto dst = tl.retile_D(tCrA);
    printf("  ldmatrix TiledCopy 对齐成功, retile_D shape = ");
    print(shape(dst)); printf("\n");

    expect("ldmatrix 对齐成功 (能编译说明 Tile 对了)", true);
}

// ---------------------------------------------------------------------------
// 练习 4 —— 手写 cp.async 双缓冲的 prologue (README §3)
//
// 下面 kernel 的双缓冲 mainloop 缺 prologue —— 请补上:
//   STAGES=2, 所以 prologue 要发 STAGES-1 = 1 批 (填进 stage 0)。
//
// 提示: 用 tc.partition_S(gA(_,_,0)) 和 tc.partition_D(sA(_,_,0))。
//       (gA/gB/sA/sB 的定义都在 kernel 里)
// ---------------------------------------------------------------------------
template <class SLayA, class SLayB>
__global__ void gemm_pipe_ex(const half_t* A, const half_t* B, float* C, int M, int N, int K,
                             SLayA slayA, SLayB slayB) {
    constexpr int BM = 64, BN = 64, BK = 64;
    constexpr int NTHR = 128, STAGES = 2;

    __shared__ __align__(128) half_t rawA[cosize_v<SLayA>];
    __shared__ __align__(128) half_t rawB[cosize_v<SLayB>];
    auto sA = make_tensor(make_smem_ptr(rawA), slayA);  // (BM,BK,STAGES)
    auto sB = make_tensor(make_smem_ptr(rawB), slayB);

    auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, Int<1>{}));
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));
    auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(blockIdx.y, _));
    auto gB = local_tile(mB, Shape<Int<BN>, Int<BK>>{}, make_coord(blockIdx.x, _));
    auto gC = local_tile(mC, Shape<Int<BM>, Int<BN>>{}, make_coord(blockIdx.y, blockIdx.x));

    auto tcopy = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
                                 Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
    ThrCopy tc = tcopy.get_slice(threadIdx.x);

    auto mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_2, _2>>{},
                              Tile<_32, _32, _16>{});
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto tCrC = thr.partition_fragment_C(gC);
    clear(tCrC);

    int nk = K / BK;
    int k_tile = 0;

    // ---- TODO ①: prologue —— 发 STAGES-1 批 (这里 = 1 批) ----
    // 提示: 填进 stage 0, 然后 k_tile++ 和 cp_async_fence()
    for (int s = 0; s < STAGES - 1; ++s) {
        cute::copy(tc.partition_S(gA(_, _, k_tile)), tc.partition_D(sA(_, _, s)));
        cute::copy(tc.partition_S(gB(_, _, k_tile)), tc.partition_D(sB(_, _, s)));
        cp_async_fence();
        ++k_tile;
    }

    int k_pipe_write = STAGES - 1;
    int k_pipe_read = 0;
    for (int k = 0; k < nk; ++k) {
        int k_next = k + (STAGES - 1);
        k_next = (k_next >= nk) ? nk - 1 : k_next;
        if (k_next > k) {
            cute::copy(tc.partition_S(gA(_, _, k_next)), tc.partition_D(sA(_, _, k_pipe_write)));
            cute::copy(tc.partition_S(gB(_, _, k_next)), tc.partition_D(sB(_, _, k_pipe_write)));
            cp_async_fence();
        }
        k_pipe_write = (k_pipe_write + 1) % STAGES;

        cp_async_wait<0>();
        __syncthreads();

        auto s2r_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
        auto s2r_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
        ThrCopy tlA = s2r_a.get_slice(threadIdx.x);
        ThrCopy tlB = s2r_b.get_slice(threadIdx.x);
        auto tCrA = thr.make_fragment_A(thr.partition_A(sA(_, _, k_pipe_read)));
        auto tCrB = thr.make_fragment_B(thr.partition_B(sB(_, _, k_pipe_read)));
        cute::copy(s2r_a, tlA.partition_S(sA(_, _, k_pipe_read)), tlA.retile_D(tCrA));
        cute::copy(s2r_b, tlB.partition_S(sB(_, _, k_pipe_read)), tlB.retile_D(tCrB));
        gemm(mma, tCrA, tCrB, tCrC);

        k_pipe_read = (k_pipe_read + 1) % STAGES;
    }

    cute::copy(tCrC, thr.partition_C(gC));
}

static void ex4() {
    printf("\n===== 练习 4: cp.async 双缓冲 prologue =====\n");

    constexpr int BM = 64, BN = 64, BK = 64, NTHR = 128;
    int M = 128, N = 128, K = 128;
    size_t nA = size_t(M) * K, nB = size_t(N) * K, nC = size_t(M) * N;

    half_t *hA = new half_t[nA], *hB = new half_t[nB];
    float *hC = new float[nC], *hR = new float[nC];
    for (int i = 0; i < int(nA); ++i) hA[i] = half_t(float((i % 5) - 2));
    for (int i = 0; i < int(nB); ++i) hB[i] = half_t(float(((i * 3) % 5) - 2));
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0;
            for (int k = 0; k < K; ++k) acc += float(hA[m * K + k]) * float(hB[n * K + k]);
            hR[m * N + n] = acc;
        }

    half_t *dA, *dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, nA * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, nB * sizeof(half_t), cudaMemcpyHostToDevice));

    auto swizzle_atom = composition(Swizzle<3, 3, 3>{},
                                    Layout<Shape<_8, Shape<_8, _8>>,
                                           Stride<_8, Stride<_1, _64>>>{});
    auto slayA = tile_to_shape(swizzle_atom, make_shape(Int<BM>{}, Int<BK>{}, Int<2>{}));
    auto slayB = tile_to_shape(swizzle_atom, make_shape(Int<BN>{}, Int<BK>{}, Int<2>{}));

    dim3 grid(N / BN, M / BM);
    gemm_pipe_ex<<<grid, NTHR>>>(dA, dB, dC, M, N, K, slayA, slayB);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC, dC, nC * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < int(nC); ++i)
        if (fabsf(hC[i] - hR[i]) > 1e-2f) ++bad;
    printf("  结果与参考不一致的元素个数 = %d\n", bad);
    expect("双缓冲 prologue 正确 -> 结果一致", bad == 0);
}

// ---------------------------------------------------------------------------
// 练习 5 —— 手写 TMA+WGMMA mainloop 的 barrier 部分 (README §4)
//
// 下面 kernel 是 v3 的简化版 (单 CTA, 单 stage), mainloop 的
// producer (发 TMA) 和 consumer (WGMMA) 之间缺同步。
//
// 请补: consumer 在 WGMMA 前要等 producer 的 barrier, 在 WGMMA 后要
//       释放 (arrive) 到 consumer barrier。producer 发 TMA 前要等
//       consumer barrier, 发完要 arrive_and_expect_tx 到 producer barrier。
//
// 提示: 单 stage 时 phase 恒为 0。
// ---------------------------------------------------------------------------
template <class TmaA, class TmaB, class SLA3, class SLB3>
__global__ void gemm_tma_ex(CUTLASS_GRID_CONSTANT TmaA const ta,
                            CUTLASS_GRID_CONSTANT TmaB const tb, float* C, SLA3 sla3,
                            SLB3 slb3) {
    // 单 k tile (K = BK): 一次 TMA 搬整块, 一次 WGMMA 算完
    constexpr int BM = 64, BN = 64, BK = 64;
    constexpr int NTHR = 128;
    __shared__ __align__(128) half_t rawA[cosize_v<SLA3>];
    __shared__ __align__(128) half_t rawB[cosize_v<SLB3>];
    __shared__ __align__(8) uint64_t bar[1];
    auto sA = make_tensor(make_smem_ptr(rawA), sla3);
    auto sB = make_tensor(make_smem_ptr(rawB), slb3);

    auto mA = ta.get_tma_tensor(make_shape(Int<BM>{}, Int<BK>{}));
    auto mB = tb.get_tma_tensor(make_shape(Int<BN>{}, Int<BK>{}));
    auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(0, _));
    auto gB = local_tile(mB, Shape<Int<BN>, Int<BK>>{}, make_coord(0, _));

    auto [tAgA, tAsA] = tma_partition(ta, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                      group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tb, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                                      group_modes<0, 2>(gB));
    constexpr int txb = sizeof(make_tensor_like(tensor<0>(tAsA)))
                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

    auto mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(Int<BM>{}, Int<BN>{}),
                          make_stride(Int<BN>{}, Int<1>{}));
    auto tCrC = thr.partition_fragment_C(mC);
    clear(tCrC);

    int warp = cutlass::canonical_warp_idx_sync();
    int one = cute::elect_one_sync();
    using Bar = cutlass::arch::ClusterTransactionBarrier;
    if (warp == 0 && one) Bar::init(&bar[0], 1);
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    auto sA2 = sA(_, _, Int<0>{});
    auto sB2 = sB(_, _, Int<0>{});

    // TODO ①: producer —— 1 个 lane 发 TMA (arrive_and_expect_tx + copy)
    if (warp == 0 && one) {
        // TODO: 填 arrive_and_expect_tx(&bar[0], txb)
        Bar::arrive_and_expect_tx(&bar[0], txb);
        copy(ta.with(bar[0]), tAgA(_, 0), tAsA(_, Int<0>{}));
        copy(tb.with(bar[0]), tBgB(_, 0), tBsB(_, Int<0>{}));
    }

    // TODO ②: consumer —— 等这批到齐 (单次 TMA, phase = 0)
    Bar::wait(&bar[0], 0);

    auto tCrA = thr.make_fragment_A(thr.partition_A(sA2));
    auto tCrB = thr.make_fragment_B(thr.partition_B(sB2));
    warpgroup_arrive();
    gemm(mma, tCrA, tCrB, tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    copy(tCrC, thr.partition_C(mC));
}

static void ex5() {
    printf("\n===== 练习 5: TMA+WGMMA 的 barrier 同步 =====\n");

    constexpr int BM = 64, BN = 64, BK = 64;
    constexpr int GK = 64;   // 单 k tile (K == BK)
    half_t *hA = new half_t[BM * GK], *hB = new half_t[BN * GK];
    float *hC = new float[BM * BN], *hR = new float[BM * BN];
    for (int i = 0; i < BM * GK; ++i) hA[i] = half_t(float((i % 7) - 3));
    for (int i = 0; i < BN * GK; ++i) hB[i] = half_t(float(((i * 5) % 7) - 3));
    for (int m = 0; m < BM; ++m)
        for (int n = 0; n < BN; ++n) {
            float acc = 0;
            for (int k = 0; k < GK; ++k) acc += float(hA[m * GK + k]) * float(hB[n * GK + k]);
            hR[m * BN + n] = acc;
        }

    half_t *dA, *dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, BM * GK * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&dB, BN * GK * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&dC, BM * BN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, BM * GK * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, BN * GK * sizeof(half_t), cudaMemcpyHostToDevice));

    auto mA = make_tensor(make_gmem_ptr(dA), make_shape(Int<BM>{}, Int<GK>{}),
                          make_stride(Int<GK>{}, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(dB), make_shape(Int<BN>{}, Int<GK>{}),
                          make_stride(Int<GK>{}, Int<1>{}));
    auto sla3 = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                              make_shape(Int<BM>{}, Int<BK>{}, Int<1>{}));
    auto slb3 = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                              make_shape(Int<BN>{}, Int<BK>{}, Int<1>{}));
    auto ta = make_tma_atom(SM90_TMA_LOAD{}, mA, sla3(_, _, Int<0>{}),
                            make_shape(Int<BM>{}, Int<BK>{}));
    auto tb = make_tma_atom(SM90_TMA_LOAD{}, mB, slb3(_, _, Int<0>{}),
                            make_shape(Int<BN>{}, Int<BK>{}));

    gemm_tma_ex<<<1, 128>>>(ta, tb, dC, sla3, slb3);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC, dC, BM * BN * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < BM * BN; ++i)
        if (fabsf(hC[i] - hR[i]) > 1e-2f) ++bad;
    printf("  结果与参考不一致的元素个数 = %d\n", bad);
    expect("barrier 同步正确 -> 结果一致", bad == 0);
}

// ---------------------------------------------------------------------------
// 练习 6 —— 手写一个 cluster 的最小 kernel (README §6)
//
// 下面 kernel 用 cluster 启动 (2x2), 每个 CTA 打印自己的 rank。
// 请补: 打印 block_rank_in_cluster() 的值, 并用 cluster_sync() 同步。
// ---------------------------------------------------------------------------
__global__ void cluster_ex() {
    // TODO ①: 用 cute::block_rank_in_cluster() 拿到 cluster 里的 rank
    int rank = cute::block_rank_in_cluster();

    if (threadIdx.x == 0) printf("  CTA (%d,%d) rank=%d\n", blockIdx.x, blockIdx.y, rank);

    // TODO ②: 全 cluster 同步 (cute::cluster_sync())
    cute::cluster_sync();

    if (threadIdx.x == 0) printf("  CTA (%d,%d) after sync\n", blockIdx.x, blockIdx.y);
}

static void ex6() {
    printf("\n===== 练习 6: cluster 最小 kernel =====\n");

    dim3 cluster(2, 2), grid(2, 2), block(32);
    cutlass::ClusterLaunchParams params{grid, block, cluster, 0};
    void const* kptr = reinterpret_cast<void const*>(&cluster_ex);
    cutlass::launch_kernel_on_cluster(params, kptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 上面打印里应该看到 4 个 rank (0,1,2,3), 且每个 CTA 都过了 sync
    printf("  (看上面的输出: 应有 4 个 rank, 4 行 after sync)\n");
    expect("cluster 启动成功且同步无死锁", true);
}

// ===========================================================================
int main() {
    printf("cute_06 练习 —— 填 TODO 后 make run\n");

    ex1();
    ex2();
    ex3();
    ex4();
    ex5();
    ex6();

    printf("\n===== 结果: %d 通过, %d 失败 =====\n", g_pass, g_fail);
    if (g_fail == 0) {
        printf("全部通过!\n");
        return 0;
    }
    printf("还有 %d 题没填或填错。\n", g_fail);
    return 1;
}
