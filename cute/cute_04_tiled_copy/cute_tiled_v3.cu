// cute_04 v3 —— Multi-stage: 让搬运和计算重叠
//
// 对应 README §6。
//
// v2 结束时我们能用 TMA 搬一块、用 WGMMA 算一块, 但两者是**串行**的:
// 搬的时候 Tensor Core 闲着, 算的时候搬运引擎闲着。这一章把它们叠起来。
//
// ---------------------------------------------------------------------------
// 阅读方式
//
//   §6.1  v3a  单缓冲          <- 搬-算-搬-算, 串行 (基准)
//   §6.2  v3b  Double Buffer   <- 2 个 buffer 轮换, 搬 k+1 与算 k 重叠
//   §6.3  v3c  Super Buffer    <- N 个 buffer, 以及 48KB 那道台阶
//   §6.4  线程去哪了 / Warp Specialization <- 概念版 (完整 kernel 在 cute_06)
//
// 每一节都是「一个 kernel + 紧跟其后的一个 host 函数」。
//
// ---------------------------------------------------------------------------
// 这个文件算什么
//
// 为了让"重叠"能被观察到, 需要一个真的有计算量的负载。这里做:
//
//     C = A * B^T       A: BM x GK      B: BN x GK      C: BM x BN
//
// K 方向切成 GK/BK 块, 每块搬进 smem 再喂给 WGMMA 累加。
// 这就是一个单 CTA 的 GEMM mainloop —— cute_06 的完整 GEMM 就是把它铺到整个 grid。
//
//   gmem A = BM x GK half, row-major, stride = (GK, 1)
//   gmem B = BN x GK half, row-major, stride = (GK, 1)
//   一次搬:  A 的 BM x BK  +  B 的 BN x BK
//
//         K 方向 -->
//   A  +----+----+----+----+ ...      每个 +----+ 是一块 BM x BK
//      |  0 |  1 |  2 |  3 |          搬进 smem 的第 (k % STAGES) 个 buffer
//      +----+----+----+----+
//
// ---------------------------------------------------------------------------
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_v3

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/device_kernel.h>
#include <cstdio>
#include <cmath>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 尺寸
// ---------------------------------------------------------------------------
using BM_ = Int<64>;  // C 的行数, 也是 A 的行数
using BN_ = Int<64>;  // C 的列数, 也是 B 的行数
using BK_ = Int<64>;  // 一次搬多少 K (必须能被 SW128 原子的 64 整除)
constexpr int BM = BM_::value, BN = BN_::value, BK = BK_::value;
constexpr int GK = 4096;   // 总 K 长度 -> 64 个 k-tile, 足够看出流水线差别
constexpr int NTHR = 128;  // 一个 warpgroup

static_assert(GK % BK == 0, "GK 必须被 BK 整除");

// ---------------------------------------------------------------------------
// 四个版本共用的缓冲区 + CPU 参考
// 填 (i%7)-3 / (i%5)-2 而不是 i: fp16 整数只精确到 2048
// ---------------------------------------------------------------------------
struct Buffers {
    half_t *d_a, *d_b;
    float* d_c;
    half_t *h_a, *h_b;
    float* h_c;

    Buffers() {
        CUDA_CHECK(cudaMalloc(&d_a, size_t(BM) * GK * sizeof(half_t)));
        CUDA_CHECK(cudaMalloc(&d_b, size_t(BN) * GK * sizeof(half_t)));
        CUDA_CHECK(cudaMalloc(&d_c, size_t(BM) * BN * sizeof(float)));
        h_a = new half_t[size_t(BM) * GK];
        h_b = new half_t[size_t(BN) * GK];
        h_c = new float[size_t(BM) * BN];
        for (size_t i = 0; i < size_t(BM) * GK; ++i) h_a[i] = half_t(float(int(i % 7) - 3));
        for (size_t i = 0; i < size_t(BN) * GK; ++i) h_b[i] = half_t(float(int(i % 5) - 2));
        CUDA_CHECK(cudaMemcpy(d_a, h_a, size_t(BM) * GK * sizeof(half_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, size_t(BN) * GK * sizeof(half_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_c, 0, size_t(BM) * BN * sizeof(float)));
    }

    ~Buffers() {
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
    }

    void clear_c() { CUDA_CHECK(cudaMemset(d_c, 0, size_t(BM) * BN * sizeof(float))); }

    // C = A * B^T, 抽稀采样比对
    bool check() {
        CUDA_CHECK(cudaMemcpy(h_c, d_c, size_t(BM) * BN * sizeof(float), cudaMemcpyDeviceToHost));
        for (int m = 0; m < BM; m += 7)
            for (int n = 0; n < BN; n += 5) {
                double acc = 0;
                for (int k = 0; k < GK; ++k)
                    acc += float(h_a[size_t(m) * GK + k]) * float(h_b[size_t(n) * GK + k]);
                if (fabs(acc - h_c[size_t(m) * BN + n]) > 1e-2 * fabs(acc) + 1.0) return false;
            }
        return true;
    }
};

// 这个 kernel 一共做多少 FLOP
static double gemm_gflop() { return 2.0 * BM * BN * GK / 1e9; }

// 一次 k-tile 搬多少字节 (A + B)
static int bytes_per_ktile() { return (BM * BK + BN * BK) * int(sizeof(half_t)); }

// ---------------------------------------------------------------------------
// 全文件共用: host 侧建 TMA descriptor 需要的 gmem 视图
// ---------------------------------------------------------------------------
static auto gmem_a(half_t const* p) {
    return make_tensor(make_gmem_ptr(p),
                       make_layout(make_shape(BM_{}, Int<GK>{}),
                                   make_stride(Int<GK>{}, Int<1>{})));
}
static auto gmem_b(half_t const* p) {
    return make_tensor(make_gmem_ptr(p),
                       make_layout(make_shape(BN_{}, Int<GK>{}),
                                   make_stride(Int<GK>{}, Int<1>{})));
}

// smem layout: SW128 原子铺成 (BM, BK, STAGES)
template <int STAGES>
static auto smem_a() {
    return tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                         make_shape(BM_{}, BK_{}, Int<STAGES>{}));
}
template <int STAGES>
static auto smem_b() {
    return tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                         make_shape(BN_{}, BK_{}, Int<STAGES>{}));
}

static auto make_mma() {
    return make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
}

// ===========================================================================
// §6.1  v3a  单缓冲 —— 搬和算完全串行
//
// 只有一个 smem buffer, 所以时间线是:
//
//   TMA(k=0) ... wait ... WGMMA(0)   TMA(1) ... wait ... WGMMA(1)   ...
//   |<-- 搬, 计算单元闲 -->|<- 算 ->|
//
// 每一轮都必须等搬完才能算, 等算完才能搬下一块 (否则会覆盖正在用的数据)。
// 两个引擎轮流干活, 各闲一半时间。
// ===========================================================================
template <class TmaA, class TmaB, class SLayA, class SLayB, class MMA>
__global__ static void gemm_single_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                                          CUTLASS_GRID_CONSTANT TmaB const tma_b, SLayA sla,
                                          SLayB slb, MMA mma, float* C) {
    __shared__ __align__(128) half_t rawA[cosize_v<SLayA>];
    __shared__ __align__(128) half_t rawB[cosize_v<SLayB>];
    __shared__ __align__(8) uint64_t full[1];

    Tensor sA = make_tensor(make_smem_ptr(rawA), sla);  // (BM,BK,1)
    Tensor sB = make_tensor(make_smem_ptr(rawB), slb);

    Tensor mA = tma_a.get_tma_tensor(make_shape(BM_{}, Int<GK>{}));
    Tensor mB = tma_b.get_tma_tensor(make_shape(BN_{}, Int<GK>{}));
    Tensor gA = local_tile(mA, make_shape(BM_{}, BK_{}), make_coord(0, _));  // (BM,BK,k)
    Tensor gB = local_tile(mB, make_shape(BN_{}, BK_{}), make_coord(0, _));

    auto pa = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                            group_modes<0, 2>(gA));
    auto pb = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                            group_modes<0, 2>(gB));
    Tensor tAg = get<0>(pa);
    Tensor tAs = get<1>(pa);
    Tensor tBg = get<0>(pb);
    Tensor tBs = get<1>(pb);
    constexpr int txb = sizeof(make_tensor_like(tensor<0>(tAs))) +
                        sizeof(make_tensor_like(tensor<0>(tBs)));

    using Bar = cutlass::arch::ClusterTransactionBarrier;
    int warp = cutlass::canonical_warp_idx_sync();
    int one = cute::elect_one_sync();
    if (warp == 0 && one) Bar::init(&full[0], 1);
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    Tensor gC = make_tensor(make_gmem_ptr(C),
                            make_layout(make_shape(BM_{}, BN_{}), make_stride(BN_{}, Int<1>{})));
    Tensor tCgC = thr.partition_C(gC);
    Tensor tCrC = thr.make_fragment_C(tCgC);
    clear(tCrC);
    Tensor tCrA = thr.make_fragment_A(thr.partition_A(sA));  // (MMA,M,K,PIPE=1)
    Tensor tCrB = thr.make_fragment_B(thr.partition_B(sB));

    int ktiles = size<1>(tAg);

    // 每一轮: 搬 -> 等 -> 算 -> 等算完 -> 下一轮
    for (int k = 0; k < ktiles; ++k) {
        if (warp == 0 && one) {
            Bar::arrive_and_expect_tx(&full[0], txb);
            copy(tma_a.with(full[0]), tAg(_, k), tAs(_, Int<0>{}));
            copy(tma_b.with(full[0]), tBg(_, k), tBs(_, Int<0>{}));
        }
        Bar::wait(&full[0], k & 1);  // phase 每轮翻转

        warpgroup_arrive();
        gemm(mma, tCrA(_, _, _, Int<0>{}), tCrB(_, _, _, Int<0>{}), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();  // 必须等算完, 否则下一轮的 TMA 会覆盖正在用的 buffer
        __syncthreads();
    }
    copy(tCrC, tCgC);
}

static float run_single(Buffers& buf) {
    printf("\nv3a  单缓冲 —— 搬和算串行\n");

    auto sla = smem_a<1>();
    auto slb = smem_b<1>();
    auto ta = make_tma_atom(SM90_TMA_LOAD{}, gmem_a(buf.d_a), sla(_, _, Int<0>{}),
                            make_shape(BM_{}, BK_{}));
    auto tb = make_tma_atom(SM90_TMA_LOAD{}, gmem_b(buf.d_b), slb(_, _, Int<0>{}),
                            make_shape(BN_{}, BK_{}));
    auto mma = make_mma();

    printf("    smem 用量 = %d KB (A %d + B %d)\n",
           int((cosize(sla) + cosize(slb)) * sizeof(half_t)) / 1024,
           int(cosize(sla) * sizeof(half_t)) / 1024, int(cosize(slb) * sizeof(half_t)) / 1024);

    dim3 block(NTHR), cluster(1, 1, 1), grid(1, 1);
    cutlass::ClusterLaunchParams params{grid, block, cluster, 0};
    void const* kptr = reinterpret_cast<void const*>(
        &gemm_single_kernel<decltype(ta), decltype(tb), decltype(sla), decltype(slb),
                            decltype(mma)>);

    buf.clear_c();
    float ms = time_kernel(
        [&] {
            cutlass::launch_kernel_on_cluster(params, kptr, ta, tb, sla, slb, mma, buf.d_c);
        },
        5, 50);
    printf("    %.4f ms   %.1f TFLOP/s   %s\n", ms, gemm_gflop() / 1e3 / (ms * 1e-3),
           buf.check() ? "正确" : "错误");
    return ms;
}

// ===========================================================================
// §6.2  v3b  Double Buffer —— 2 个 buffer 轮换
//
// 有了两个 buffer, 时间线变成:
//
//   TMA :  [搬 0][搬 1][搬 2][搬 3]...
//   WGMMA:       [算 0][算 1][算 2]...
//                 ^^^^^^ 搬 k+1 和算 k 同时进行
//
// 需要两组 barrier:
//   full[s]   生产者->消费者:  "buffer s 已装满, 可以算了"   (按字节数等, TMA 专用)
//   empty[s]  消费者->生产者:  "buffer s 已用完, 可以覆盖了" (普通 barrier)
//
// 这就是 producer/consumer。用 mbarrier 而不是 __syncthreads, 因为需要的是
// "针对某个 buffer 的细粒度等待", 不是全 block 栅栏。
//
// ---------------------------------------------------------------------------
// 一个必踩的坑
//
// prologue 把 STAGES 个 buffer 都填满了, 很自然会想"那 write_state 也该
// 预先 ++STAGES 次"。**这样会直接死锁。** read/write 两个 PipelineState 都
// 必须从 0 开始 —— 循环里只在"即将复用某个 stage"时才 wait empty, 那时该
// stage 的消费者早就 arrive 过了, 所以不会阻塞。
// ===========================================================================
template <int STAGES, class TmaA, class TmaB, class SLayA, class SLayB, class MMA>
__global__ static void gemm_pipe_kernel(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                                        CUTLASS_GRID_CONSTANT TmaB const tma_b, SLayA sla,
                                        SLayB slb, MMA mma, float* C) {
    // 静态 __shared__ 上限 48KB。STAGES=3 时 A+B 是 48KB 本身, 4 是 64KB,
    // 静态直接装不下 -> 换成动态 (extern __shared__) + cudaFuncSetAttribute。
    // 这是"单缓冲 -> double buffer"之外的又一道台阶, 见 main 里的说明。
    // 因为 extern __shared__ 的基址是运行时值, 我们用一个运行时整数来选路径,
    // 编译期两个分支都实例化, 但只有被选中的那个能实际分配 < 48KB。
    // 真正区别: STAGES=1/2 用静态 (好读), STAGES>=3 用动态 (能过 48KB)。
    // STAGES<=2 时 A+B <= 32KB < 48KB, 用静态 __shared__ (A 天然在偏移 0)。
    // STAGES>=3 时 A+B >= 48KB 超了静态上限, 换动态 (extern __shared__) + 
    // host 侧 cudaFuncSetAttribute 放开到 227KB。
    constexpr bool kDyn = (STAGES >= 3);
    half_t* rawA;
    half_t* rawB;
    uint64_t* full;
    uint64_t* empty;
    if constexpr (kDyn) {
        extern __shared__ char smem_raw[];
        rawA = reinterpret_cast<half_t*>(smem_raw);       // 偏移 0, 对齐 128B
        rawB = rawA + cosize_v<SLayA>;
        full  = reinterpret_cast<uint64_t*>(rawB + cosize_v<SLayB>);
        empty = full + STAGES;
    } else {
        __shared__ __align__(128) half_t rawS[cosize_v<SLayA> + cosize_v<SLayB>];
        rawA  = rawS;
        rawB  = rawS + cosize_v<SLayA>;
        __shared__ __align__(8) uint64_t rawBar[2 * STAGES];
        full  = rawBar;
        empty = rawBar + STAGES;
    }

    Tensor sA = make_tensor(make_smem_ptr(rawA), sla);  // (BM,BK,STAGES)
    Tensor sB = make_tensor(make_smem_ptr(rawB), slb);

    Tensor mA = tma_a.get_tma_tensor(make_shape(BM_{}, Int<GK>{}));
    Tensor mB = tma_b.get_tma_tensor(make_shape(BN_{}, Int<GK>{}));
    Tensor gA = local_tile(mA, make_shape(BM_{}, BK_{}), make_coord(0, _));
    Tensor gB = local_tile(mB, make_shape(BN_{}, BK_{}), make_coord(0, _));

    auto pa = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                            group_modes<0, 2>(gA));
    auto pb = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                            group_modes<0, 2>(gB));
    Tensor tAg = get<0>(pa);
    Tensor tAs = get<1>(pa);
    Tensor tBg = get<0>(pb);
    Tensor tBs = get<1>(pb);
    constexpr int txb = sizeof(make_tensor_like(tensor<0>(tAs))) +
                        sizeof(make_tensor_like(tensor<0>(tBs)));

    using FullBar = cutlass::arch::ClusterTransactionBarrier;  // TMA 用, 按字节
    using EmptyBar = cutlass::arch::ClusterBarrier;            // 消费者用, 按到达数

    int warp = cutlass::canonical_warp_idx_sync();
    int one = cute::elect_one_sync();
    CUTE_UNROLL
    for (int s = 0; s < STAGES; ++s) {
        if (warp == 0 && one) {
            FullBar::init(&full[s], 1);      // 只有 1 个 lane 发 TMA
            EmptyBar::init(&empty[s], NTHR); // 整个 warpgroup 都要 arrive
        }
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    int ktiles = size<1>(tAg);
    int ktile = 0;
    int left = ktiles;

    // ---- prologue: 把所有 stage 填满 ----
    CUTE_UNROLL
    for (int s = 0; s < STAGES; ++s) {
        if (left > 0) {
            if (warp == 0 && one) {
                FullBar::arrive_and_expect_tx(&full[s], txb);
                copy(tma_a.with(full[s]), tAg(_, ktile), tAs(_, s));
                copy(tma_b.with(full[s]), tBg(_, ktile), tBs(_, s));
            }
            --left;
            ++ktile;
        }
    }

    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    Tensor gC = make_tensor(make_gmem_ptr(C),
                            make_layout(make_shape(BM_{}, BN_{}), make_stride(BN_{}, Int<1>{})));
    Tensor tCgC = thr.partition_C(gC);
    Tensor tCrC = thr.make_fragment_C(tCgC);
    clear(tCrC);
    Tensor tCrA = thr.make_fragment_A(thr.partition_A(sA));  // (MMA,M,K,PIPE)
    Tensor tCrB = thr.make_fragment_B(thr.partition_B(sB));

    // 两个状态都从 0 开始 —— 见上面那段注释, 不要预推进
    auto wst = cutlass::PipelineState<STAGES>();
    auto rst = cutlass::PipelineState<STAGES>();

    CUTE_NO_UNROLL
    while (left > -STAGES) {
        int rp = rst.index();
        FullBar::wait(&full[rp], rst.phase());  // 等这个 buffer 装满

        warpgroup_arrive();
        gemm(mma, tCrA(_, _, _, rp), tCrB(_, _, _, rp), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        EmptyBar::arrive(&empty[rp]);  // 告诉生产者: 这个 buffer 我用完了
        ++rst;

        if (warp == 0 && one && left > 0) {
            int wp = wst.index();
            EmptyBar::wait(&empty[wp], wst.phase());  // 等这个 buffer 被消费完
            FullBar::arrive_and_expect_tx(&full[wp], txb);
            copy(tma_a.with(full[wp]), tAg(_, ktile), tAs(_, wp));
            copy(tma_b.with(full[wp]), tBg(_, ktile), tBs(_, wp));
            ++wst;
        }
        --left;
        ++ktile;
    }
    copy(tCrC, tCgC);
}

template <int STAGES>
static float run_pipe(Buffers& buf, const char* tag) {
    printf("\n%s  (STAGES = %d)\n", tag, STAGES);

    auto sla = smem_a<STAGES>();
    auto slb = smem_b<STAGES>();
    auto ta = make_tma_atom(SM90_TMA_LOAD{}, gmem_a(buf.d_a), sla(_, _, Int<0>{}),
                            make_shape(BM_{}, BK_{}));
    auto tb = make_tma_atom(SM90_TMA_LOAD{}, gmem_b(buf.d_b), slb(_, _, Int<0>{}),
                            make_shape(BN_{}, BK_{}));
    auto mma = make_mma();

    int smem_kb = int((cosize(sla) + cosize(slb)) * sizeof(half_t)) / 1024;
    printf("    smem 用量 = %d KB   barrier: full[%d] + empty[%d]\n", smem_kb, STAGES, STAGES);

    void const* kptr = reinterpret_cast<void const*>(
        &gemm_pipe_kernel<STAGES, decltype(ta), decltype(tb), decltype(sla), decltype(slb),
                          decltype(mma)>);

    int total = int(cosize(sla) + cosize(slb)) * int(sizeof(half_t)) + 2 * STAGES * 8;
    // STAGES>=3 时 A+B 超 48KB 静态上限, 必须动态 __shared__:
    //   1) cudaFuncSetAttribute 放开上限
    //   2) launch 时在 smem_size_in_bytes 里申请具体字节数
    if (STAGES >= 3)
        CUDA_CHECK(cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, total));

    dim3 block(NTHR), cluster(1, 1, 1), grid(1, 1);
    // 只有 STAGES>=3 才真的用动态 smem; STAGES<=2 是静态, 不要额外申请
    int launch_smem = (STAGES >= 3) ? total : 0;
    cutlass::ClusterLaunchParams params{grid, block, cluster, launch_smem};
    if (STAGES >= 3) printf("    动态 smem = %d 字节\n", total);

    buf.clear_c();
    float ms = time_kernel(
        [&] {
            cutlass::launch_kernel_on_cluster(params, kptr, ta, tb, sla, slb, mma, buf.d_c);
        },
        5, 50);
    printf("    %.4f ms   %.1f TFLOP/s   %s\n", ms, gemm_gflop() / 1e3 / (ms * 1e-3),
           buf.check() ? "正确" : "错误");
    return ms;
}

// ===========================================================================
// §6.4  v3d  Warp Specialization —— "TMA 之后线程干什么"
//
// 先纠正一个误解。看 v3b 的循环体:
//
//     1 个 lane   发 TMA
//   128 个线程    一起做 WGMMA
//
// 所以并不是"用了 TMA 线程就空着" —— 其余 127 个线程不是闲着, 它们在算。
// 真正的问题是别的:
//
//   **同一批线程既要发搬运又要等计算, 两件事被串在一条指令流上。**
//
// v3b 里 warpgroup_wait<0>() 之后, 所有 128 个线程(包括那个负责发 TMA 的 lane)
// 都得等 MMA 完成才能继续。于是"发下一块的 TMA"这件事被 MMA 挡住了。
//
// Warp Specialization 换一种分工:
//
//   v3b 统一分工:                  v3d 专业化分工 (cute_06 完整实现):
//   +---------------------+        +------------------+------------------+
//   | warp 0 1 2 3        |        | producer: 只发TMA| consumer: 只算WGMMA|
//   | 都做 WGMMA          |        | 不参与 MMA       | worker 才做 MMA    |
//   | 其中 1 个 lane 发TMA|        | 一路往前跑        | 不碰搬运          |
//   +---------------------+        +------------------+------------------+
//          |                            |               |
//      一条指令流                    各自独立的指令流, 用 mbarrier 通信
//
// producer warp 不做 MMA, 所以它能一路往前, 把后面几块的 TMA 全发出去,
// 不被任何 warpgroup_wait 挡住。
//
// 这个文件不跑一个完整的 WS kernel。原因有两层:
//   1) 概念正确性是 v3b/v3c 已经证明的 —— 128 个线程在算, 只有 1 个 lane 在搬,
//      两个引擎的重叠全靠 mbarrier 而非 __syncthreads。
//   2) WS 的收益要在大 GEMM 上才显 (多个 warpgroup 争一个 SM 时, producer 抢跑
//      才值得专开一组 warp)。单 CTA 的 64x64 负载连 TMA 延迟都没喂饱,
//      硬上 WS 只能把 mbarrier 的开销加回去, 看不出收益。
//
// 完整的 WS kernel (2 个 warpgroup + setmaxnreg 寄存器再分配) 是 cute_06 的 capstone。
// 它就是"多 stage (本章) + 线程分组"合在一起: producer warpgroup 发 TMA,
// consumer warpgroup 做 WGMMA, 两组用 full/empty 双 mbarrier 通信 —— 结构和
// v3b 一模一样, 只是把 v3b 里"同一个线程既发又等"拆成了两组线程各干各的。
// ===========================================================================

// ===========================================================================
// main
// ===========================================================================
int main() {
    printf("cute_04 v3 —— Multi-stage: 让搬运和计算重叠\n");
    printf("对应 README §6\n");
    printf("负载: C = A * B^T,  A = %dx%d, B = %dx%d, C = %dx%d (half in, float out)\n", BM, GK,
           BN, GK, BM, BN);
    printf("      K 切成 %d 个 tile, 每 tile 搬 %d KB (A+B)\n", GK / BK,
           bytes_per_ktile() / 1024);
    printf("      总计算量 %.2f GFLOP\n", gemm_gflop());

    Buffers buf;

    print_separator("§6.1  单缓冲: 搬和算串行");
    float t1 = run_single(buf);
    printf("\n    时间线 (一个 k-tile 一个周期):\n");
    printf("      TMA   : [搬0]      [搬1]      [搬2]\n");
    printf("      WGMMA :      [算0]      [算1]      [算2]\n");
    printf("      两个引擎轮流干活, 各闲一半。\n");

    print_separator("§6.2  Double Buffer: 2 个 buffer 轮换");
    float t2 = run_pipe<2>(buf, "v3b  Double Buffer");
    printf("\n    时间线:\n");
    printf("      TMA   : [搬0][搬1][搬2][搬3]\n");
    printf("      WGMMA :      [算0][算1][算2]\n");
    printf("               ^^^^ 搬 k+1 和算 k 同时进行\n");
    printf("\n    相对单缓冲: %.2fx\n", t1 / t2);

    print_separator("§6.3  Super Buffer: 更多 stage");
    float t3 = run_pipe<3>(buf, "v3c  Super Buffer");
    float t4 = run_pipe<4>(buf, "v3c  Super Buffer");
    printf("\n  stage 数与 smem 用量:\n");
    printf("    STAGES  smem(A+B)   时间(ms)   相对单缓冲\n");
    printf("      1      %2d KB      %.4f      1.00x\n",
           int((cosize(smem_a<1>()) + cosize(smem_b<1>())) * sizeof(half_t)) / 1024, t1);
    printf("      2      %2d KB      %.4f      %.2fx\n",
           int((cosize(smem_a<2>()) + cosize(smem_b<2>())) * sizeof(half_t)) / 1024, t2, t1 / t2);
    printf("      3      %2d KB      %.4f      %.2fx\n",
           int((cosize(smem_a<3>()) + cosize(smem_b<3>())) * sizeof(half_t)) / 1024, t3, t1 / t3);
    printf("      4      %2d KB      %.4f      %.2fx\n",
           int((cosize(smem_a<4>()) + cosize(smem_b<4>())) * sizeof(half_t)) / 1024, t4, t1 / t4);
    printf("\n  注意 smem 是线性增长的, 而收益递减 —— stage 数不是越多越好。\n");
    printf("  再往上还有一道硬台阶: 静态 __shared__ 上限是 48KB。\n");
    printf("  要吃到 H200 的 227KB 必须换动态 smem:\n");
    printf("    extern __shared__ char smem[];\n");
    printf("    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, n);\n");
    printf("  这就是官方例子用 SharedStorage 结构体 + 动态 smem 的原因。\n");

    print_separator("§6.4  Warp Specialization: 线程分工怎么变");
    printf("\n  先纠正一个误解: \"用了 TMA, 线程是不是就空着了?\"\n");
    printf("  不是。看 v3b 的循环体:\n");
    printf("        1 个 lane   发 TMA\n");
    printf("      128 个线程    一起做 WGMMA     <- 其余 127 个在算, 不是闲着\n");
    printf("\n  真正的问题是: 同一批线程既要发搬运又要等计算,\n");
    printf("  两件事被串在一条指令流上。warpgroup_wait<0>() 之后所有线程\n");
    printf("  (包括发 TMA 那个 lane) 都得等 MMA 完成, 于是下一块的 TMA 被挡住了。\n");
    printf("\n  解法是把「发搬运」和「算」拆到不同的 warp:\n");
    printf("    producer warp  只发 TMA, 不碰 MMA -> 能一路往前跑,\n");
    printf("    consumer 组    只做 WGMMA, 不过问搬运\n");
    printf("  两组靠 full/empty 双 mbarrier 通信 (就是 v3b 那两组 barrier),\n");
    printf("  只是把「同一个线程既发又等」拆成了两组线程各干各的。\n");
    printf("\n  这里不跑一个完整 WS kernel, 原因在文件头部 §6.4 有写:\n");
    printf("    1) 概念正确性 v3b/v3c 已经证明 (128 线程在算, 只有 1 个 lane 在搬);\n");
    printf("    2) WS 收益要在大 GEMM 上才显 (单 CTA 64x64 连 TMA 延迟都没喂饱)。\n");
    printf("    完整 WS (2 warpgroup + setmaxnreg) 是 cute_06 的 capstone。\n");

    print_separator("小结");
    printf("  §6.1  单缓冲: 搬完才能算, 算完才能搬, 两个引擎各闲一半\n");
    printf("  §6.2  Double Buffer: 2 个 buffer + full/empty 两组 mbarrier -> 重叠\n");
    printf("  §6.3  Super Buffer: stage 越多容忍延迟越强, 但 smem 线性涨、收益递减\n");
    printf("  §6.4  Warp Specialization: 把\"发搬运\"和\"算\"拆到不同 warp\n");
    printf("\n  必踩的坑: PipelineState 的 write_state 不能在 prologue 里预推进,\n");
    printf("  read/write 都必须从 0 开始, 否则死锁。\n");

    printf("\n下一步 (§7): capstone 把本章所有工具用到一个完整的转置上。\n");
    printf("\nv3 OK\n");
    return 0;
}
