// cute_05 v2 —— TMA: 给 MMA 喂数据的那条通路
//
// 对应 README §4。
//
// v1 结束时 WGMMA 已经能算了, 但数据是这样进 smem 的:
//
//     for (int i = threadIdx.x; i < BM*BK; i += NTHR)
//         sA(i/BK, i%BK) = gA(i/BK, i%BK);      // 128 个线程各搬各的
//     __syncthreads();
//
// 这段代码有三个隐性成本, 在 v1 那种"只算一块"的场景下看不出来, 一旦进入
// 真实 GEMM 的 K 循环就全暴露:
//
//   1. **占指令流**: 每个线程都要算地址、发 load、等 load。这些指令和后面的
//      WGMMA 抢同一个发射端口。
//   2. **占寄存器**: 地址、循环变量、in-flight 的数据都要寄存器, 而 WGMMA 的
//      累加器已经吃掉每线程 32 个 float 了。
//   3. **同步粒度粗**: __syncthreads() 是整个 block 的栅栏, 没法做到
//      "A 搬完了就先算 A 那部分"。
//
// TMA (Tensor Memory Accelerator) 把这三条一次解决:
//     一个线程发一条指令描述整块 -> 硬件后台搬 -> mbarrier 按字节数等
//
// 这一版就是把 v1 的搬运那几行换成 TMA, 其它一律不动, 然后看差别。
//
//   §4.1  没有 TMA 怎么搬 vs 有 TMA 怎么搬  (README §4.1)
//   §4.2  TMA 的五个硬性条件, 逐条对着代码看 (README §4.2)
//   §4.3  多 K tile: 真实 GEMM 的 K 循环长什么样 (README §4.3)
//
// cute_04 §5 已经讲过 TMA 的机制本身 (descriptor 怎么建、tma_partition 干什么)。
// 这一版不重复那些, 只讲**它怎么接到 MMA 上**。
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_mma_v2

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
// 全局配置
//
// gmem 上的整个矩阵 (TN 摆法):
//   A: GM x GK half, row-major, stride = (GK,1)
//   B: GN x GK half, row-major, stride = (GK,1)   <- 存的是 B^T
//   C: GM x GN float, row-major, stride = (GN,1)
//
// 一个 CTA 负责整个 C (GM x GN), 沿 K 方向循环 GK/BK 次:
//   每次 TMA 搬进 A 的 BM x BK 和 B 的 BN x BK, 喂给 WGMMA 累加。
//
// 尺寸选择的理由:
//   BM=BN=64  : 正好是 WGMMA 原子的 M/N
//   BK=64     : GMMA::Layout_K_SW128_Atom<half> 要求 K % 64 == 0 (v1 §3.4)
//   GK=256    : 4 个 K tile, 足以看出循环结构, 又小到能和 CPU 逐点比
// ---------------------------------------------------------------------------
constexpr int GM = 64, GN = 64, GK = 256;
constexpr int BM = 64, BN = 64, BK = 64;
constexpr int NTHR = 128;  // 一个 warpgroup
constexpr int NK = GK / BK;

CUTE_HOST_DEVICE static auto make_wgmma() {
    return make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
}

// 四个版本共用的缓冲区 —— 构造时分配并填好, 析构时释放
struct Buffers {
    half_t *d_A, *d_B, *h_A, *h_B;
    float *d_C, *h_C, *h_ref;

    Buffers() {
        h_A = new half_t[size_t(GM) * GK];
        h_B = new half_t[size_t(GN) * GK];
        h_C = new float[size_t(GM) * GN];
        h_ref = new float[size_t(GM) * GN];
        fill_pm1(h_A, size_t(GM) * GK, 7);
        fill_pm1(h_B, size_t(GN) * GK, 8);
        gemm_cpu(h_A, h_B, h_ref, GM, GN, GK);

        CUDA_CHECK(cudaMalloc(&d_A, size_t(GM) * GK * sizeof(half_t)));
        CUDA_CHECK(cudaMalloc(&d_B, size_t(GN) * GK * sizeof(half_t)));
        CUDA_CHECK(cudaMalloc(&d_C, size_t(GM) * GN * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_t(GM) * GK * sizeof(half_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_t(GN) * GK * sizeof(half_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, size_t(GM) * GN * sizeof(float)));
    }

    ~Buffers() {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_ref;
    }

    CheckResult check() {
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size_t(GM) * GN * sizeof(float), cudaMemcpyDeviceToHost));
        return check_close(h_C, h_ref, size_t(GM) * GN);
    }
};

// ===========================================================================
// §4.1a  没有 TMA: 每个线程算地址, 自己搬
//
// 这就是 v1 §3.3 那段搬运, 只是外面套了一层 K 循环。
// 留意 mainloop 里"搬"和"算"各占多少行 —— 搬运的代码量和计算相当。
// ===========================================================================
template <class SLay>
__global__ void gemm_manual_load(const half_t* A, const half_t* B, float* C, SLay slay) {
    __shared__ __align__(128) half_t rawA[cosize_v<SLay>];
    __shared__ __align__(128) half_t rawB[cosize_v<SLay>];
    auto sA = make_tensor(make_smem_ptr(rawA), slay);
    auto sB = make_tensor(make_smem_ptr(rawB), slay);

    auto mma = make_wgmma();
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(Int<GM>{}, Int<GN>{}),
                          make_stride(Int<GN>{}, Int<1>{}));
    auto tCrC = thr.partition_fragment_C(gC);
    clear(tCrC);  // 累加器在 K 循环外清零, 循环里一路累加

    auto mA = make_tensor(make_gmem_ptr(A), make_shape(Int<GM>{}, Int<GK>{}),
                          make_stride(Int<GK>{}, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(Int<GN>{}, Int<GK>{}),
                          make_stride(Int<GK>{}, Int<1>{}));

    for (int k = 0; k < NK; ++k) {
        // ---- 搬: 128 个线程各搬各的, 每人 BM*BK/NTHR = 32 个元素 ----
        for (int i = threadIdx.x; i < BM * BK; i += NTHR)
            sA(i / BK, i % BK) = mA(i / BK, k * BK + i % BK);
        for (int i = threadIdx.x; i < BN * BK; i += NTHR)
            sB(i / BK, i % BK) = mB(i / BK, k * BK + i % BK);
        __syncthreads();  // 粗粒度: 整个 block 等所有人搬完

        // ---- 算 ----
        auto tCrA = thr.make_fragment_A(thr.partition_A(sA));
        auto tCrB = thr.make_fragment_B(thr.partition_B(sB));
        warpgroup_arrive();
        gemm(mma, tCrA, tCrB, tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        __syncthreads();  // 算完才能覆盖 smem, 又一次全 block 栅栏
    }

    copy(tCrC, thr.partition_C(gC));
}

static void run_manual_load() {
    print_separator("§4.1a  没有 TMA: 每个线程算地址自己搬");

    Buffers buf;
    auto slay = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                              make_shape(Int<BM>{}, Int<BK>{}));

    printf("\n  C[%dx%d] = A[%dx%d] * B[%dx%d]^T, K 切成 %d 个 tile\n", GM, GN, GM, GK, GN, GK, NK);
    printf("  搬运: %d 个线程, 每人每轮搬 %d 个 half, 用 __syncthreads 同步\n", NTHR,
           BM * BK / NTHR);

    gemm_manual_load<<<1, NTHR>>>(buf.d_A, buf.d_B, buf.d_C, slay);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto r = buf.check();
    printf("\n  与 CPU 参考比对: %s   (bad=%d, maxerr=%g)\n", r.ok() ? "完全一致" : "不一致", r.bad,
           r.maxerr);
}

// ===========================================================================
// §4.1b / §4.2  有 TMA: 一个线程描述整块, 硬件搬
//
// 和上面那个 kernel 逐行对比, 变的只有搬运那几行。五个硬性条件都标在代码里
// (完整解释见 cute_04 §5.2, 这里只标位置):
//
//   条件 1: src 必须是 tma.get_tma_tensor(shape) 给出的**坐标** tensor
//   条件 2: descriptor 必须在 host 用**真实设备指针**构造 (见 run_tma_load)
//   条件 3: smem 必须 __align__(128)
//   条件 4: smem layout 必须带 PIPE mode -> (BM,BK,PIPE)
//   条件 5: partition 必须用 tma_partition, 不是 partition_S/D
// ===========================================================================
template <class TmaA, class TmaB, class SLay3>
__global__ void gemm_tma_load(CUTLASS_GRID_CONSTANT TmaA const tma_a,
                              CUTLASS_GRID_CONSTANT TmaB const tma_b, float* C, SLay3 slay3) {
    // 条件 3: 128B 对齐
    __shared__ __align__(128) half_t rawA[cosize_v<SLay3>];
    __shared__ __align__(128) half_t rawB[cosize_v<SLay3>];
    __shared__ __align__(8) uint64_t bar[1];

    // 条件 4: layout 带 PIPE mode。这里 PIPE=1 (单缓冲), 多 stage 是 cute_04 §6 的事
    auto sA = make_tensor(make_smem_ptr(rawA), slay3);  // (BM,BK,1)
    auto sB = make_tensor(make_smem_ptr(rawB), slay3);

    // 条件 1: 坐标 tensor —— 不是数据 tensor
    auto mA = tma_a.get_tma_tensor(make_shape(Int<GM>{}, Int<GK>{}));
    auto mB = tma_b.get_tma_tensor(make_shape(Int<GN>{}, Int<GK>{}));
    // 沿 K 切块: (BM,BK,k) —— 最后一维就是 K tile 的序号
    auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(0, _));
    auto gB = local_tile(mB, Shape<Int<BN>, Int<BK>>{}, make_coord(0, _));

    // 条件 5: tma_partition
    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                      group_modes<0, 2>(gA));
    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                                      group_modes<0, 2>(gB));
    // 这一轮 TMA 要搬多少字节 —— mbarrier 就按这个数等
    constexpr int txb = sizeof(make_tensor_like(tensor<0>(tAsA)))
                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

    auto mma = make_wgmma();
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(Int<GM>{}, Int<GN>{}),
                          make_stride(Int<GN>{}, Int<1>{}));
    auto tCrC = thr.partition_fragment_C(gC);
    clear(tCrC);

    // mbarrier: 只有一个 lane 负责初始化
    int warp = cutlass::canonical_warp_idx_sync();
    int one = cute::elect_one_sync();
    using Bar = cutlass::arch::ClusterTransactionBarrier;
    if (warp == 0 && one) Bar::init(&bar[0], 1);
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    auto sA2 = sA(_, _, Int<0>{});
    auto sB2 = sB(_, _, Int<0>{});

    for (int k = 0; k < NK; ++k) {
        // ---- 搬: 一个 lane 发两条 TMA, 整块就位 ----
        if (warp == 0 && one) {
            Bar::arrive_and_expect_tx(&bar[0], txb);       // 声明这轮要收多少字节
            copy(tma_a.with(bar[0]), tAgA(_, k), tAsA(_, Int<0>{}));
            copy(tma_b.with(bar[0]), tBgB(_, k), tBsB(_, Int<0>{}));
        }
        // 所有线程等这批字节到齐。phase 每轮翻转 -> k&1
        Bar::wait(&bar[0], k & 1);

        // ---- 算: 和上面那个 kernel 一模一样 ----
        auto tCrA = thr.make_fragment_A(thr.partition_A(sA2));
        auto tCrB = thr.make_fragment_B(thr.partition_B(sB2));
        warpgroup_arrive();
        gemm(mma, tCrA, tCrB, tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        __syncthreads();  // 算完才能让下一轮 TMA 覆盖 smem
    }

    copy(tCrC, thr.partition_C(gC));
}

static void run_tma_load() {
    print_separator("§4.1b  有 TMA: 一个线程描述整块, 硬件搬");

    Buffers buf;

    // 条件 2: descriptor 必须用**真实设备指针**在 host 构造
    auto mA = make_tensor(make_gmem_ptr(buf.d_A), make_shape(Int<GM>{}, Int<GK>{}),
                          make_stride(Int<GK>{}, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(buf.d_B), make_shape(Int<GN>{}, Int<GK>{}),
                          make_stride(Int<GK>{}, Int<1>{}));

    // 条件 4: 带 PIPE mode 的 smem layout; 建 atom 时传它的切片
    auto slay3 = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                               make_shape(Int<BM>{}, Int<BK>{}, Int<1>{}));
    auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, mA, slay3(_, _, Int<0>{}),
                               make_shape(Int<BM>{}, Int<BK>{}));
    auto tma_b = make_tma_atom(SM90_TMA_LOAD{}, mB, slay3(_, _, Int<0>{}),
                               make_shape(Int<BN>{}, Int<BK>{}));

    printf("\n  同一个 GEMM, 只把搬运那几行换成 TMA:\n");
    printf("    发起者   : 1 个 lane (elect_one_sync)  <- 不是 128 个线程\n");
    printf("    同步     : mbarrier 按字节数等         <- 不是 __syncthreads\n");
    printf("    地址计算 : 硬件做                      <- 不是每线程算\n");

    gemm_tma_load<<<1, NTHR>>>(tma_a, tma_b, buf.d_C, slay3);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto r = buf.check();
    printf("\n  与 CPU 参考比对: %s   (bad=%d, maxerr=%g)\n", r.ok() ? "完全一致" : "不一致", r.bad,
           r.maxerr);
}

// ===========================================================================
// §4.3  两条通路并排, 以及它们各自的成本
// ===========================================================================
static void compare_paths() {
    print_separator("§4.3  两条通路的结构对比");

    printf("\n  手写搬运 (§4.1a):\n");
    printf("    ┌─────────────────────────────────────────────┐\n");
    printf("    │ for i = tid; i < BM*BK; i += 128            │  128 个线程\n");
    printf("    │     sA(...) = mA(...)   <- 每线程算地址      │  都在算地址\n");
    printf("    │ __syncthreads()         <- 全 block 栅栏    │  发 load\n");
    printf("    └─────────────────────────────────────────────┘  等 load\n");

    printf("\n  TMA (§4.1b):\n");
    printf("    ┌─────────────────────────────────────────────┐\n");
    printf("    │ if (one lane) {                             │  1 个 lane\n");
    printf("    │     expect_tx(bar, %5d) <- 声明字节数       │  发 2 条指令\n",
           2 * BM * BK * int(sizeof(half_t)));
    printf("    │     copy(tma.with(bar), gA, sA)             │\n");
    printf("    │ }                                           │  其余 127 个\n");
    printf("    │ Bar::wait(bar, phase)   <- 按字节等          │  线程什么都不做\n");
    printf("    └─────────────────────────────────────────────┘\n");

    printf("\n  三个差别:\n");
    printf("    1. 指令流: 128 线程 x N 条 load  ->  1 lane x 1 条 TMA\n");
    printf("       省下的发射槽让给 WGMMA。\n");
    printf("    2. 寄存器: 不用存地址和 in-flight 数据。\n");
    printf("       WGMMA 的累加器已经占了每线程 32 个 float, 这很关键。\n");
    printf("    3. 同步  : __syncthreads (全 block) -> mbarrier (按字节)。\n");
    printf("       mbarrier 才能做 producer/consumer, 这是多 stage 流水线的前提。\n");

    printf("\n  但注意: 这一版**没有变快**, 因为搬和算仍然是串行的 ——\n");
    printf("    TMA 搬完 -> 等 -> WGMMA 算完 -> 等 -> 下一轮 TMA\n");
    printf("  TMA 真正的价值要等它和多 stage 组合才兑现 (cute_04 §6 已跑通,\n");
    printf("  cute_06 会把它铺满整个 grid)。这一节只证明「通路能换、换了对」。\n");
}

// ===========================================================================
int main() {
    printf("cute_05 v2 —— TMA: 给 MMA 喂数据的那条通路\n");
    printf("对应 README §4\n");
    printf("\ngmem: A[%dx%d] B[%dx%d] half (TN 摆法), C[%dx%d] float\n", GM, GK, GN, GK, GM, GN);
    printf("tile: 每轮搬 A[%dx%d] + B[%dx%d] 进 smem, K 方向 %d 轮\n", BM, BK, BN, BK, NK);

    run_manual_load();
    run_tma_load();
    compare_paths();

    print_separator("小结");
    printf("  §4.1  同一个 GEMM 的两种数据通路, 结果完全一致\n");
    printf("  §4.2  TMA 的五个硬性条件在代码里逐条标了位置\n");
    printf("  §4.3  TMA 省的是**指令流、寄存器、同步粒度**, 不是单纯的带宽\n");
    printf("\n  现在两个引擎都会用了: TMA 搬, WGMMA 算。\n");
    printf("  capstone: 把它们拼成一个完整的、单 CTA 的 GEMM。\n");

    printf("\nv2 OK\n");
    return 0;
}
