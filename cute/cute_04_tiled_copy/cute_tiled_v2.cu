// cute_04 v2 —— TMA: 把搬运交给硬件
//
// 对应 README §5。
//
// v1 结束时, gmem->smem 已经是 CuTe 的 copy() 了, 但底下仍然是
// "256 个线程各自算地址、各自发一条 load"。SM90 提供了另一条路: TMA。
//
// 这个文件按"先看旧写法, 再看新写法"的顺序排:
//
//   §5.1  cp.async 手写搬运      <- 没有 TMA 时怎么搬 (对照基准)
//   §5.2  TMA 搬同一块 tile      <- 有了 TMA 怎么搬
//   §5.3  TMA 的四种 swizzle 模式 <- SW128 / SW64 / SW32 / INTER 怎么选
//   §5.4  WGMMA 才挑 layout      <- TMA 不挑, 消费者挑
//
// 每一节都是「一个 kernel + 紧跟其后的一个 host 函数」, 自带缓冲区和验证。
//
// ---------------------------------------------------------------------------
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_v2

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/device_kernel.h>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 全文件统一的尺寸
//
//   gmem 里的 A:  GM x GK 个 half, row-major, stride = (GK, 1)
//                 256 x 128 half = 64KB
//
//   一个 CTA 搬走其中一块 tile:  TM x TK = 128 x 64 half = 16KB
//                 tile 在 gmem 里不是连续的 —— 每行取 TK 个, 跨 GK 到下一行
//
//   +---------------- GK = 128 ----------------+
//   | <--- TK = 64 --->                        |  ^
//   | +---------------+                        |  |
//   | |   本 CTA 要   |                        |  | TM = 128
//   | |   搬的 tile   |                        |  |
//   | +---------------+                        |  v
//   |                                          |     ^
//   |                                          |     | GM = 256
//   +------------------------------------------+     v
// ---------------------------------------------------------------------------
constexpr int GM = 256, GK = 128;  // 整个 A 矩阵
constexpr int TM = 128, TK = 64;   // 一个 CTA 搬的 tile
constexpr int NTHR = 128;          // 每 CTA 线程数 (一个 warpgroup)

// ---------------------------------------------------------------------------
// §5.1 / §5.2 共用的缓冲区
//
// h_a 填 i % 1024: fp16 整数只精确到 2048, 填 i 会在大下标失精度导致误报
// ---------------------------------------------------------------------------
struct Buffers {
    static constexpr size_t a_elems = size_t(GM) * GK;
    static constexpr size_t t_elems = size_t(TM) * TK;

    half_t* d_a;    // gmem 里的 A
    half_t* d_out;  // kernel 把 smem 原样倒出来, 供 host 比对
    half_t* h_a;
    half_t* h_out;

    Buffers() {
        CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(half_t)));
        CUDA_CHECK(cudaMalloc(&d_out, t_elems * sizeof(half_t)));
        h_a = new half_t[a_elems];
        h_out = new half_t[t_elems];
        for (size_t i = 0; i < a_elems; ++i) h_a[i] = half_t(float(i % 1024));
        CUDA_CHECK(cudaMemcpy(d_a, h_a, a_elems * sizeof(half_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, t_elems * sizeof(half_t)));
    }

    ~Buffers() {
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_out));
        delete[] h_a;
        delete[] h_out;
    }

    // tile (0,0) 落数是否正确: smem 的 (r,c) 应该等于 A 的 (r,c)
    bool check() {
        CUDA_CHECK(cudaMemcpy(h_out, d_out, t_elems * sizeof(half_t), cudaMemcpyDeviceToHost));
        for (int r = 0; r < TM; ++r)
            for (int c = 0; c < TK; ++c)
                if (float(h_out[r * TK + c]) != float(h_a[r * GK + c])) return false;
        return true;
    }
};

// half 的 bank: 一个 bank 4 字节 = 2 个 half, 所以偏移 x2 变字节
template <class Lay>
static int col_conflict_half(Lay lay, int lanes = 32) {
    return max_bank_requests(lanes, [&](int l) { return int(lay(l, 0)) * 2; });
}

// ===========================================================================
// §5.1  没有 TMA 时怎么搬: cp.async, 每个线程算自己的地址
//
// 这就是 v1 的写法搬到 half + 更大 tile 上。要点:
//   - 128 个线程每人负责 tile 的一部分, 各自算地址、各自发一条 cp.async
//   - 同步用 cp_async_wait + __syncthreads (全 block 栅栏)
//   - tile 边界要靠 predicate 兜 (这里 tile 整除, 所以省了)
// ===========================================================================
template <class SLay, class TC>
__global__ static void copy_cpasync_kernel(const half_t* __restrict__ a, half_t* __restrict__ out,
                                           SLay slay, TC tc) {
    __shared__ __align__(128) half_t raw[cosize_v<SLay>];
    auto sA = make_tensor(make_smem_ptr(raw), slay);

    // gmem 视图 + 取出本 CTA 的 tile (0,0)
    auto mA = make_tensor(make_gmem_ptr(a),
                          make_layout(make_shape(Int<GM>{}, Int<GK>{}),
                                      make_stride(Int<GK>{}, Int<1>{})));
    auto gA = local_tile(mA, Shape<Int<TM>, Int<TK>>{}, make_coord(0, 0));

    // 128 个线程分工: 每人搬 TM*TK/128 = 64 个 half
    auto thr = tc.get_slice(threadIdx.x);
    copy(tc, thr.partition_S(gA), thr.partition_D(sA));

    // cp.async 是异步的, 要等它落地; 然后还要 __syncthreads 让全 block 都看到
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // 倒出来给 host 比对
    for (int i = threadIdx.x; i < TM * TK; i += blockDim.x) out[i] = sA(i / TK, i % TK);
}

static void section51_cpasync() {
    print_separator("§5.1  没有 TMA: cp.async, 每线程算自己的地址");

    printf("  gmem A = %dx%d half, row-major, stride = (%d,1)\n", GM, GK, GK);
    printf("  本 CTA 搬 tile (0,0) = %dx%d half = %d KB, 用 %d 个线程\n\n", TM, TK,
           TM * TK * 2 / 1024, NTHR);

    // smem 用 GMMA 的 SW128 原子铺开 —— 和 §5.2 完全一样, 便于对照
    auto slay = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                              make_shape(Int<TM>{}, Int<TK>{}));

    // 128-bit atom: 每线程一次 8 个 half; 线程排成 (16,8), 每人 (1,8)
    auto tc = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
                              make_layout(make_shape(Int<16>{}, Int<8>{}),
                                          make_stride(Int<8>{}, Int<1>{})),
                              make_layout(make_shape(Int<1>{}, Int<8>{})));

    Buffers buf;
    copy_cpasync_kernel<<<1, NTHR>>>(buf.d_a, buf.d_out, slay, tc);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("    smem layout = ");
    print(slay);
    printf("\n    落数 = %s\n", buf.check() ? "正确" : "错误");

    printf("\n  这一版的成本清单:\n");
    printf("    发指令的线程数   128 个 (每人一条 cp.async)\n");
    printf("    地址计算         每线程各算一次\n");
    printf("    边界处理         要自己写 predicate (本例整除, 省了)\n");
    printf("    同步             cp_async_wait + __syncthreads, 全 block 栅栏\n");
    printf("    寄存器           每线程都要占几个存地址\n");
}

// ===========================================================================
// §5.2  有了 TMA: 一个线程描述整块, 硬件自己搬
//
// 同一块 tile, 换成 TMA。五个硬性条件都标在代码里:
//   条件 1  src 必须是 tma.get_tma_tensor(shape) —— 坐标 tensor, 不是数据 tensor
//   条件 2  descriptor 必须在 host 用真实设备指针构造 (见下面的 host 函数)
//   条件 3  smem 必须 __align__(128)
//   条件 4  smem layout 必须带 PIPE 维, 建 atom 时传切片 slay(_,_,Int<0>{})
//   条件 5  partition 用 tma_partition, 不是 partition_S/D
// ===========================================================================
template <class TmaAtom, class SLay>
__global__ static void copy_tma_kernel(CUTLASS_GRID_CONSTANT TmaAtom const tma, SLay slay,
                                       half_t* __restrict__ out, bool announce) {
    __shared__ __align__(128) half_t raw[cosize_v<SLay>];  // 条件 3
    __shared__ __align__(8) uint64_t bar[1];               // mbarrier, 不是 __syncthreads

    auto sA = make_tensor(make_smem_ptr(raw), slay);  // (TM,TK,PIPE)   条件 4

    // 条件 1: 坐标 tensor。它不存数据, 只提供"我要 gmem 的哪一块"的坐标
    auto mA = tma.get_tma_tensor(make_shape(Int<GM>{}, Int<GK>{}));
    auto gA = local_tile(mA, Shape<Int<TM>, Int<TK>>{}, make_coord(0, 0));

    // 条件 5: TMA 专用 partition。group_modes<0,2> 把 (TM,TK,*) 压成 ((TM,TK),*),
    //         因为 mode-0 整个交给 TMA 负责
    auto p = tma_partition(tma, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                           group_modes<0, 2>(gA));
    auto tAg = get<0>(p);  // (TMA,)      gmem 侧
    auto tAs = get<1>(p);  // (TMA,PIPE)  smem 侧

    // 这一次搬多少字节 —— mbarrier 要按字节数等
    constexpr int tx_bytes = sizeof(make_tensor_like(tensor<0>(tAs)));

    int warp = cutlass::canonical_warp_idx_sync();
    int one = cute::elect_one_sync();  // 从一个 warp 里选出唯一一个 lane
    using Bar = cutlass::arch::ClusterTransactionBarrier;

    if (warp == 0 && one) Bar::init(&bar[0], 1);
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    // ---- 整个搬运就这三行, 而且只有一个 lane 在跑 ----
    if (warp == 0 && one) {
        Bar::arrive_and_expect_tx(&bar[0], tx_bytes);       // 告诉 barrier 要等多少字节
        copy(tma.with(bar[0]), tAg, tAs(_, Int<0>{}));      // 一条指令描述整块
    }
    Bar::wait(&bar[0], 0);  // 所有线程在这里等硬件搬完
    // 注意: 其余 127 个线程从头到尾没参与搬运

    if (announce && thread0())
        printf("    TMA 一次搬 %d 字节 (= %d KB)\n", tx_bytes, tx_bytes / 1024);

    auto s2 = sA(_, _, Int<0>{});
    for (int i = threadIdx.x; i < TM * TK; i += blockDim.x) out[i] = s2(i / TK, i % TK);
}

// TMA 版的 host 侧: descriptor 在这里构造 (条件 2)
template <class SLay3>
static bool launch_tma_copy(SLay3 slay3, Buffers& buf, bool verbose) {
    // gmem 视图必须用**真实设备指针**建 —— 用 nullptr 会在运行时报
    // "Failed to initialize the TMA descriptor 201"
    auto mA = make_tensor(make_gmem_ptr(buf.d_a),
                          make_layout(make_shape(Int<GM>{}, Int<GK>{}),
                                      make_stride(Int<GK>{}, Int<1>{})));

    // 条件 4: 建 atom 时传 PIPE 的一个切片, 不是整个三维 layout
    auto tma = make_tma_atom(SM90_TMA_LOAD{}, mA, slay3(_, _, Int<0>{}),
                             make_shape(Int<TM>{}, Int<TK>{}));

    if (verbose) {
        printf("    TMA atom  = ");
        print(tma);
        printf("\n");
    }

    // TMA kernel 要用 cluster launch 启动
    dim3 block(NTHR), cluster(1, 1, 1), grid(1, 1);
    cutlass::ClusterLaunchParams params{grid, block, cluster, 0};
    void const* kptr =
        reinterpret_cast<void const*>(&copy_tma_kernel<decltype(tma), SLay3>);
    cutlass::launch_kernel_on_cluster(params, kptr, tma, slay3, buf.d_out, verbose);
    CUDA_CHECK(cudaDeviceSynchronize());
    return buf.check();
}

static void section52_tma() {
    print_separator("§5.2  有了 TMA: 一个线程描述整块");

    printf("  搬的是和 §5.1 完全相同的 tile。\n\n");

    // 条件 4: 必须带 PIPE 维。这里 PIPE=1, 到 v3 做多 stage 时它会变成 2/3/4
    auto slay3 = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                               make_shape(Int<TM>{}, Int<TK>{}, Int<1>{}));

    Buffers buf;
    bool ok = launch_tma_copy(slay3, buf, true);
    printf("    smem layout = ");
    print(slay3);
    printf("\n    落数 = %s\n", ok ? "正确" : "错误");

    printf("\n  和 §5.1 并排看:\n\n");
    printf("                  cp.async (§5.1)              TMA (§5.2)\n");
    printf("    ------------  ---------------------------  --------------------------\n");
    printf("    发指令线程    128 个, 每人一条             1 个 lane, 一共一条\n");
    printf("    地址计算      每线程各算一次               host 侧 descriptor 算好\n");
    printf("    边界处理      自己写 predicate             硬件自动填 0\n");
    printf("    同步          cp_async_wait+syncthreads    mbarrier 按字节数等\n");
    printf("    寄存器占用    每线程存地址                 几乎为 0\n");
    printf("    swizzle       由 smem layout 决定          写进 descriptor\n");

    printf("\n  descriptor 里存了什么 (host 侧那 128 字节):\n");
    printf("      gmem 基地址\n");
    printf("      gmem 形状 (%d, %d), stride (%d, 1)\n", GM, GK, GK);
    printf("      tile 形状 (%d, %d)\n", TM, TK);
    printf("      smem 的 swizzle 模式      <- 就是 §5.3 要讲的那四种\n");
    printf("      元素类型 / 越界填充策略\n");
    printf("    kernel 里只说\"搬第 (0,0) 块\", 其余全在 descriptor 里。\n");
}

// ===========================================================================
// §5.3  TMA 的四种 swizzle 模式
//
// descriptor 里的 swizzle 字段只有几个取值。CuTe 把它们封成四个 layout 原子,
// 名字里的数字是"一行占多少字节":
//
//   SW128  一行 128 字节 = 64 个 half
//   SW64   一行  64 字节 = 32 个 half
//   SW32   一行  32 字节 = 16 个 half
//   INTER  一行  16 字节 =  8 个 half   (Sw<0>, 即不 swizzle)
//
// 这一节用同一个 §5.2 的 kernel 跑四种模式, 看它们的差别。
// ===========================================================================
template <class Atom>
static void try_swizzle_mode(const char* name, Atom atom, int row_bytes) {
    auto slay3 = tile_to_shape(atom, make_shape(Int<TM>{}, Int<TK>{}, Int<1>{}));
    Buffers buf;
    bool ok = launch_tma_copy(slay3, buf, false);
    auto s2 = slay3(_, _, Int<0>{});
    printf("    %-7s 一行 %3d 字节   落数 %-4s   consumer 列读 = %2d-way   cosize = %d\n", name,
           row_bytes, ok ? "正确" : "错误", col_conflict_half(s2), int(cosize(s2)));
}

static void section53_swizzle_modes() {
    print_separator("§5.3  TMA 的四种 swizzle 模式");

    printf("  四个原子的实际参数 (Sw<B,M,S> o (行,列):(stride)):\n");
    printf("    SW128 = ");
    print(GMMA::Layout_K_SW128_Atom<half_t>{});
    printf("\n    SW64  = ");
    print(GMMA::Layout_K_SW64_Atom<half_t>{});
    printf("\n    SW32  = ");
    print(GMMA::Layout_K_SW32_Atom<half_t>{});
    printf("\n    INTER = ");
    print(GMMA::Layout_K_INTER_Atom<half_t>{});
    printf("\n\n");

    printf("  同一个 TMA kernel, 只换 smem layout 原子 (tile %dx%d half):\n", TM, TK);
    try_swizzle_mode("SW128", GMMA::Layout_K_SW128_Atom<half_t>{}, 128);
    try_swizzle_mode("SW64", GMMA::Layout_K_SW64_Atom<half_t>{}, 64);
    try_swizzle_mode("SW32", GMMA::Layout_K_SW32_Atom<half_t>{}, 32);
    try_swizzle_mode("INTER", GMMA::Layout_K_INTER_Atom<half_t>{}, 16);

    printf("\n  两点结论:\n");
    printf("    1) 四种模式 TMA 都搬得对 —— 选哪个不影响正确性, 只影响\n");
    printf("       consumer 读 smem 时撞几路 bank。\n");
    printf("    2) 四个原子的 M 全 = 4 (2^4 = 16 个 half = 32 字节)。这正是\n");
    printf("       §3.4 那条规则的体现: 先保住向量宽度, 再消冲突。\n");

    printf("\n  怎么选: 唯一的硬约束是 TK 必须被原子的 K 长度整除。\n");
    printf("    TK (half) | SW128(64) | SW64(32) | SW32(16) | INTER(8)\n");
    printf("       64     |    可     |    可    |    可    |   可\n");
    printf("       32     |   不可    |    可    |    可    |   可\n");
    printf("       16     |   不可    |   不可   |    可    |   可\n");
    printf("  违反了会编译期报:\n");
    printf("    \"tile_to_shape: block shape does not divide the target shape\"\n");
    printf("\n  实用规则: 选能用的里面一行字节数最大的 (对齐最大 = 访存最宽)。\n");
    printf("  本例 TK=%d -> 选 SW128。\n", TK);
}

// ===========================================================================
// §5.4  谁挑 layout: TMA 不挑, WGMMA 挑
//
// §5.3 已经看到 TMA 对四种官方模式都能搬。那 plain row-major 行不行?
// 行 —— TMA 照样搬得对。真正拒绝它的是下游的 WGMMA, 而且是编译期拒绝。
// ===========================================================================
constexpr int BM = 64, BN = 64, BK = 64;

template <class MMA, class SLayA, class SLayB>
__global__ static void wgmma_kernel(const half_t* A, const half_t* B, float* C, MMA mma,
                                    SLayA sla, SLayB slb) {
    __shared__ __align__(128) half_t rawA[cosize_v<SLayA>];
    __shared__ __align__(128) half_t rawB[cosize_v<SLayB>];

    auto sA = make_tensor(make_smem_ptr(rawA), sla);
    auto sB = make_tensor(make_smem_ptr(rawB), slb);

    // 朴素装载 (不是本节重点, 只为把数据摆进 smem)
    for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) sA(i / BK, i % BK) = A[i];
    for (int i = threadIdx.x; i < BN * BK; i += blockDim.x) sB(i / BK, i % BK) = B[i];
    __syncthreads();

    auto gC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(Int<BM>{}, Int<BN>{}),
                                                        make_stride(Int<BN>{}, Int<1>{})));
    auto thr = mma.get_thread_slice(threadIdx.x);
    auto tCsA = thr.partition_A(sA);
    auto tCsB = thr.partition_B(sB);
    auto tCgC = thr.partition_C(gC);

    auto tCrA = thr.make_fragment_A(tCsA);  // SM90: DescriptorIterator, 不是寄存器!
    auto tCrB = thr.make_fragment_B(tCsB);
    auto tCrC = thr.make_fragment_C(tCgC);
    clear(tCrC);

    if (thread0()) {
        printf("      make_fragment_A = ");
        print(tCrA);
        printf("\n      每线程占 %d 字节 (只是个描述符, 不是 A 的数据)\n", int(sizeof(tCrA)));
    }

    // WGMMA 固定四步。注意没有 copy(tCsA, tCrA) —— 硬件直接读 smem
    warpgroup_arrive();
    gemm(mma, tCrA, tCrB, tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    copy(tCrC, tCgC);
}

template <class SLayA, class SLayB>
static void run_wgmma(const char* tag, SLayA sla, SLayB slb) {
    auto mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

    half_t *dA, *dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, BM * BK * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&dB, BN * BK * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&dC, BM * BN * sizeof(float)));
    half_t* hA = new half_t[BM * BK];
    half_t* hB = new half_t[BN * BK];
    for (int i = 0; i < BM * BK; ++i) hA[i] = half_t(float((i % 7) - 3));
    for (int i = 0; i < BN * BK; ++i) hB[i] = half_t(float((i % 5) - 2));
    CUDA_CHECK(cudaMemcpy(dA, hA, BM * BK * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, BN * BK * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, BM * BN * sizeof(float)));

    printf("    %s\n", tag);
    wgmma_kernel<<<1, int(size(mma))>>>(dA, dB, dC, mma, sla, slb);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* hC = new float[BM * BN];
    CUDA_CHECK(cudaMemcpy(hC, dC, BM * BN * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int m = 0; m < BM; ++m)
        for (int n = 0; n < BN; ++n) {
            double acc = 0;
            for (int k = 0; k < BK; ++k) acc += float(hA[m * BK + k]) * float(hB[n * BK + k]);
            if (fabs(acc - hC[m * BN + n]) > 1e-3) ++bad;
        }
    printf("      C = A*B^T  %s\n", bad == 0 ? "正确" : "错误");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    delete[] hA;
    delete[] hB;
    delete[] hC;
}

static void section54_who_picks() {
    print_separator("§5.4  谁挑 layout: TMA 不挑, WGMMA 挑");

    // 先证明 TMA 连 plain row-major 都搬得对
    printf("  先看 TMA 对 plain row-major (完全没 swizzle):\n");
    {
        auto plain3 = make_layout(make_shape(Int<TM>{}, Int<TK>{}, Int<1>{}),
                                  make_stride(Int<TK>{}, Int<1>{}, Int<TM * TK>{}));
        Buffers buf;
        bool ok = launch_tma_copy(plain3, buf, false);
        printf("    plain row-major   落数 %-4s   consumer 列读 = %2d-way\n",
               ok ? "正确" : "错误", col_conflict_half(plain3(_, _, Int<0>{})));
    }
    printf("  -> TMA 搬得对。加上 §5.3 的四种, 五种 layout 全都搬得对。\n");
    printf("     很容易误解成\"TMA 要求 swizzle\" —— 它不要求。\n");

    printf("\n  再看 WGMMA (SM90_64x64x16_F32F16F16_SS, %dx%dx%d half):\n", BM, BN, BK);
    auto mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
    printf("    size(mma) = %d  <- 一个 warpgroup = 4 个 warp, 不是 32\n\n", int(size(mma)));

    auto shA = make_shape(Int<BM>{}, Int<BK>{});
    auto shB = make_shape(Int<BN>{}, Int<BK>{});
    run_wgmma("SW128 atom", tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{}, shA),
              tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{}, shB));
    run_wgmma("INTER atom (Sw<0>, 无 swizzle 但是规范形式)",
              tile_to_shape(GMMA::Layout_K_INTER_Atom<half_t>{}, shA),
              tile_to_shape(GMMA::Layout_K_INTER_Atom<half_t>{}, shB));

    printf("\n    plain row-major  ->  编译期失败, 放不进这个程序里跑\n");
    printf("    padded stride+8  ->  同样编译期失败\n");
    printf("    报错都是:\n");
    printf("      static assertion failed:\n");
    printf("      \"Not a canonical GMMA_K Layout: Expected stride failure.\"\n");
    printf("    (想亲眼看到: 把 exercises/ex.cu 里 EX5_TRY_PLAIN_WGMMA 改成 1)\n");

    printf("\n  为什么 WGMMA 这么挑: 它不经过寄存器, 而是把 smem 地址和摆法编码成\n");
    printf("  一个 descriptor, 硬件照它直读 smem。descriptor 里只有几个比特存\n");
    printf("  swizzle 模式, 能表达的摆法就 §5.3 那四种。\n");

    printf("\n  所以这一章的 layout 是一份合同:\n");
    printf("    TMA (生产者) ---- smem layout ----> WGMMA (消费者)\n");
    printf("     不挑, 都能搬          由消费者定       只认 4 种规范形式\n");
}

int main() {
    printf("cute_04 v2 —— TMA: 把搬运交给硬件\n");
    printf("对应 README §5    需要 -arch=sm_90a\n");

    section51_cpasync();
    section52_tma();
    section53_swizzle_modes();
    section54_who_picks();

    print_separator("小结");
    printf("  §5.1  cp.async: 128 个线程各算地址各发指令\n");
    printf("  §5.2  TMA: 1 个 lane 描述整块, 硬件搬, mbarrier 按字节等\n");
    printf("  §5.3  swizzle 模式写在 descriptor 里, 四种可选, 按 TK 整除规则挑最宽的\n");
    printf("  §5.4  TMA 不挑 layout, WGMMA 编译期挑 —— layout 是两者之间的合同\n");

    printf("\n下一步 (§6): 现在 TMA 搬一块、WGMMA 算一块, 但它们是**串行**的 ——\n");
    printf("搬的时候计算单元闲着, 算的时候搬运引擎闲着。v3 讲怎么让两者重叠。\n");
    printf("\nv2 OK\n");
    return 0;
}
