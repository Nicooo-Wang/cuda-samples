// cute_04 v2 —— SM90: TMA 与 WGMMA 对 smem layout 的真实要求
//
// 对应 README §5。
//
// v0/v1 讲的是「swizzle 是什么、怎么用」。这个文件回答: SM90 的硬件为什么强制要它。
//
// 三个实测结论 (都能在下面的输出里看到):
//   1) TMA 不挑 layout —— 连 plain row-major 都能搬对。
//   2) WGMMA 才是硬墙 —— plain 和 padded 在编译期就被拒绝。
//   3) SM90 不需要 ldmatrix —— WGMMA 用 descriptor 直读 smem, 没有寄存器 fragment。
//
// ---------------------------------------------------------------------------
// 阅读方式
//
// §5.1  TMA 是什么硬件, 和 cp.async 的区别
// §5.2  跑一次真实的 TMA load, 看它接受哪些 layout
// §5.3  跑一次真实的 WGMMA, 看 fragment 是什么 + 哪些 layout 编译不过
// §5.4  GMMA 四个官方原子的参数与 K 整除规则
//
// 每节都是「一个 kernel + 紧跟其后的 host 函数」, 自带缓冲区和验证。
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

// half 下的 bank: 一个 bank 4 字节 = 2 个 half, 所以传字节偏移进 max_bank_requests
template <class Lay>
static int col_conflict_half(Lay lay, int lanes = 32) {
    return max_bank_requests(lanes, [&](int l) { return int(lay(l, 0)) * 2; });
}

// ===========================================================================
// §5.1  TMA 是什么
// ===========================================================================
static void section51_what_is_tma() {
    print_separator("§5.1  TMA (Tensor Memory Accelerator) 是什么");

    printf("Hopper 新增的独立硬件单元, 专做 gmem <-> smem 的整块搬运。\n\n");
    printf("和 SM80 cp.async 的根本区别:\n\n");
    printf("  SM80 cp.async —— 每个线程算自己的地址, 发自己那一份\n");
    printf("    +--------------------------------------------+\n");
    printf("    | t0 算地址->发   t1 算地址->发  ...  t255   |  256 个线程都在算地址\n");
    printf("    +--------------------------------------------+\n\n");
    printf("  SM90 TMA —— 一个线程描述整块, 硬件自己搬\n");
    printf("    +--------------------------------------------+\n");
    printf("    | 1 个线程: \"把 gmem(128,64) 搬到 smem 这里\" |  其余 255 个可以去干别的\n");
    printf("    |            -> 硬件接手                     |\n");
    printf("    +--------------------------------------------+\n");

    printf("\nTMA 的硬件特性:\n");
    printf("  descriptor 驱动   host 侧建一个 128B 描述符, 记下 gmem 形状/步长/smem 摆法\n");
    printf("  一个线程发起      elect_one_sync() 选一个 lane 发指令\n");
    printf("  自带边界处理      越界自动填 0, 不需要写 predicate\n");
    printf("  异步 + mbarrier   不用 __syncthreads, 用 ClusterTransactionBarrier 按字节等\n");
    printf("  硬件做 swizzle    descriptor 里带 swizzle 模式, 搬的过程中就摆好\n");
    printf("  multicast         一次搬运可以灌进 cluster 里多个 CTA 的 smem\n");

    printf("\n最后两条是关键: swizzle 是写进 TMA descriptor 的,\n");
    printf("所以 smem layout 不能随便填 —— 它是 TMA 和 WGMMA 之间的一份合同。\n");
}

// ===========================================================================
// §5.2  真实的 TMA load: 它接受哪些 layout?
//
// 五个硬性条件 (踩过坑的都在这里):
//   1) src 必须是 tma.get_tma_tensor(shape), 不能是普通 gmem tensor
//   2) descriptor 必须在 host 用真实设备指针构造
//   3) smem 要 __align__(128)
//   4) smem layout 必须带 PIPE 维, 构造 atom 时传切片 slay(_,_,Int<0>{})
//   5) partition 用 tma_partition, 不是 partition_S/D
// ===========================================================================
constexpr int TM = 128, TN = 64;  // 一块 tile: 128x64 half, 和官方例子同量级
constexpr int GM = 256, GN = 128;

template <class TmaAtom, class SLay>
__global__ static void tma_load_kernel(CUTLASS_GRID_CONSTANT TmaAtom const tma, SLay slay,
                                       half_t* out, int gm, int gn) {
    __shared__ __align__(128) half_t raw[cosize_v<SLay>];  // 条件 3
    __shared__ __align__(8) uint64_t producer_mbar[1];

    auto mA = tma.get_tma_tensor(make_shape(gm, gn));  // 条件 1
    auto gA = local_tile(mA, Shape<Int<TM>, Int<TN>>{}, make_coord(0, 0));
    auto sA = make_tensor(make_smem_ptr(raw), slay);  // (TM,TN,PIPE)  条件 4

    // 条件 5: TMA 专用的 partition
    auto [tAg, tAs] = tma_partition(tma, Int<0>{}, Layout<_1>{},
                                    group_modes<0, 2>(sA), group_modes<0, 2>(gA));

    constexpr int tx_bytes = sizeof(make_tensor_like(tensor<0>(tAs)));

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_pred = cute::elect_one_sync();  // 只选一个 lane 发指令
    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;

    if (warp_idx == 0 && lane_pred) ProducerBar::init(&producer_mbar[0], 1);
    cluster_sync();

    if (warp_idx == 0 && lane_pred) {
        ProducerBar::arrive_and_expect_tx(&producer_mbar[0], tx_bytes);
        copy(tma.with(producer_mbar[0]), tAg, tAs(_, Int<0>{}));  // 整块交给硬件
    }
    ProducerBar::wait(&producer_mbar[0], 0);  // 按字节数等, 不是 __syncthreads
    __syncthreads();

    if (thread0()) printf("      TMA 一次搬 %d 字节\n", tx_bytes);

    // 按逻辑坐标读回, 验证落数正确
    auto s2 = sA(_, _, Int<0>{});
    for (int i = threadIdx.x; i < TM * TN; i += blockDim.x) out[i] = s2(i / TN, i % TN);
}

template <class SLay3>
static void tma_try(const char* tag, SLay3 slay3) {
    half_t *d, *dout;
    CUDA_CHECK(cudaMalloc(&d, size_t(GM) * GN * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&dout, size_t(TM) * TN * sizeof(half_t)));
    half_t* h = new half_t[size_t(GM) * GN];
    // 填 i % 1024: fp16 整数只精确到 2048, 填 i 会在大下标失精度导致误报
    for (int i = 0; i < GM * GN; ++i) h[i] = half_t(float(i % 1024));
    CUDA_CHECK(cudaMemcpy(d, h, size_t(GM) * GN * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout, 0, size_t(TM) * TN * sizeof(half_t)));

    auto g = make_tensor(make_gmem_ptr(d),
                         make_layout(make_shape(GM, GN), make_stride(GN, Int<1>{})));
    // 条件 2: host 侧用真实指针建 descriptor; 条件 4: 传 PIPE 的一个切片
    auto tma = make_tma_atom(SM90_TMA_LOAD{}, g, slay3(_, _, Int<0>{}),
                             make_shape(Int<TM>{}, Int<TN>{}));

    printf("    %s\n", tag);
    dim3 dimBlock(128), dimCluster(1, 1, 1), dimGrid(1, 1);
    cutlass::ClusterLaunchParams params{dimGrid, dimBlock, dimCluster, 0};
    void const* kptr = reinterpret_cast<void const*>(&tma_load_kernel<decltype(tma), SLay3>);
    cutlass::Status st =
        cutlass::launch_kernel_on_cluster(params, kptr, tma, slay3, dout, GM, GN);
    cudaError_t e = cudaDeviceSynchronize();

    half_t* ho = new half_t[size_t(TM) * TN];
    CUDA_CHECK(cudaMemcpy(ho, dout, size_t(TM) * TN * sizeof(half_t), cudaMemcpyDeviceToHost));
    bool ok = (e == cudaSuccess && st == cutlass::Status::kSuccess);
    if (ok)
        for (int r = 0; r < TM && ok; ++r)
            for (int c = 0; c < TN; ++c)
                if (float(ho[r * TN + c]) != float(h[r * GN + c])) {
                    ok = false;
                    break;
                }

    auto s2 = slay3(_, _, Int<0>{});
    printf("      TMA = %-4s   落数 = %-4s   consumer 侧列读 = %2d-way   cosize = %d\n",
           e == cudaSuccess ? "OK" : "FAIL", ok ? "正确" : "错误", col_conflict_half(s2),
           int(cosize(s2)));

    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFree(dout));
    delete[] h;
    delete[] ho;
}

static void section52_tma_layouts() {
    print_separator("§5.2  TMA 接受哪些 smem layout? (实测 128x64 half)");

    auto sh = make_shape(Int<TM>{}, Int<TN>{}, Int<1>{});
    tma_try("SW128 atom", tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{}, sh));
    tma_try("SW64 atom", tile_to_shape(GMMA::Layout_K_SW64_Atom<half_t>{}, sh));
    tma_try("SW32 atom", tile_to_shape(GMMA::Layout_K_SW32_Atom<half_t>{}, sh));
    tma_try("INTER atom (Sw<0>, 即不 swizzle)",
            tile_to_shape(GMMA::Layout_K_INTER_Atom<half_t>{}, sh));
    tma_try("plain row-major (完全没 swizzle)",
            make_layout(make_shape(Int<TM>{}, Int<TN>{}, Int<1>{}),
                        make_stride(Int<TN>{}, Int<1>{}, Int<TM * TN>{})));

    printf("\n  结论: TMA 自己不挑 layout —— 五种全都搬对了, 连 plain row-major 也一样。\n");
    printf("  swizzle 的价值不在 TMA 这一侧, 而在 consumer 那一侧 (最后一列数字):\n");
    printf("  数据摆进 smem 之后, 谁去读它时会撞几路 bank。\n");
    printf("\n  这一点值得强调, 因为很容易误解成\"TMA 要求 swizzle\"。\n");
    printf("  TMA 不要求。下一节的 WGMMA 才要求。\n");
}

// ===========================================================================
// §5.3  WGMMA: 真正的硬墙
//
// 一个 warpgroup (128 线程) 做 C = A * B^T, A/B 都在 smem。
// 注意 mainloop 里没有 copy(tCsA, tCrA) —— fragment 是 descriptor, 不是寄存器。
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
static void wgmma_try(const char* tag, SLayA sla, SLayB slb) {
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
    cudaError_t e = cudaDeviceSynchronize();

    float* hC = new float[BM * BN];
    CUDA_CHECK(cudaMemcpy(hC, dC, BM * BN * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int m = 0; m < BM; ++m)
        for (int n = 0; n < BN; ++n) {
            double acc = 0;
            for (int k = 0; k < BK; ++k) acc += float(hA[m * BK + k]) * float(hB[n * BK + k]);
            if (fabs(acc - hC[m * BN + n]) > 1e-3) ++bad;
        }
    printf("      launch = %-9s  C = A*B^T 结果 %s\n", cudaGetErrorString(e),
           (e == cudaSuccess && bad == 0) ? "正确" : "错误");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    delete[] hA;
    delete[] hB;
    delete[] hC;
}

static void section53_wgmma() {
    print_separator("§5.3  WGMMA 才是硬墙 (实测 64x64x64 half)");

    auto mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
    printf("size(mma) = %d   <- 一个 warpgroup = 4 个 warp, 不是 32\n\n", int(size(mma)));

    auto shA = make_shape(Int<BM>{}, Int<BK>{});
    auto shB = make_shape(Int<BN>{}, Int<BK>{});

    wgmma_try("SW128 atom (官方推荐)",
              tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{}, shA),
              tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{}, shB));
    wgmma_try("INTER atom (Sw<0>, 无 swizzle 但是规范形式)",
              tile_to_shape(GMMA::Layout_K_INTER_Atom<half_t>{}, shA),
              tile_to_shape(GMMA::Layout_K_INTER_Atom<half_t>{}, shB));

    printf("\n    plain row-major  ->  编译期就失败, 无法放进这个程序里跑。\n");
    printf("    padded stride+8  ->  同样编译期失败。\n");
    printf("    两者的报错都是:\n");
    printf("      static assertion failed:\n");
    printf("      \"Not a canonical GMMA_K Layout: Expected stride failure.\"\n");
    printf("    (想亲眼看到? 把 exercises/ 里练习 5 的注释打开编译一次。)\n");

    printf("\n  为什么 WGMMA 这么挑: 它不用寄存器读数据, 而是把 smem 地址和摆法\n");
    printf("  编码成一个 descriptor, 硬件按 descriptor 直读 smem。descriptor 里只有\n");
    printf("  几个比特存 swizzle 模式, 能表达的摆法就那么几种。\n");

    printf("\n  SM80 和 SM90 的链路差异:\n");
    printf("    SM80:  smem --ldmatrix--> 寄存器 --mma--> 结果\n");
    printf("           (显式搬一次, 要 fragment 寄存器, 要 retile_D)\n");
    printf("    SM90:  smem -------descriptor-------> wgmma --> 结果\n");
    printf("           (不搬! 没有 fragment 寄存器, mainloop 里没有 copy(tCsA,tCrA))\n");
    printf("\n  所以本章不讲 ldmatrix: 在 SM90 上它是多余的一跳。\n");
}

// ===========================================================================
// §5.4  GMMA 四个官方原子
// ===========================================================================
static void section54_gmma_atoms() {
    print_separator("§5.4  GMMA 官方 swizzle 原子 (WGMMA 认的规范 layout)");

    printf("四个原子的实际参数:\n");
    printf("  SW128 = ");
    print(GMMA::Layout_K_SW128_Atom<half_t>{});
    printf("\n  SW64  = ");
    print(GMMA::Layout_K_SW64_Atom<half_t>{});
    printf("\n  SW32  = ");
    print(GMMA::Layout_K_SW32_Atom<half_t>{});
    printf("\n  INTER = ");
    print(GMMA::Layout_K_INTER_Atom<half_t>{});
    printf("\n");

    printf("\n三点值得注意:\n");
    printf("  1) 名字里的数字是字节数: SW128 = 一行 128 字节 = 64 个 half\n");
    printf("  2) M 全都是 4: 2^4 = 16 个 half = 32 字节。对上 v0 §3.4 的规则 ——\n");
    printf("     先保证向量化 (16B/32B 访问), 再消冲突。这就是官方参数为什么长这样。\n");
    printf("  3) INTER 是 Sw<0>, 即\"不 swizzle\", 但它也是合法的规范 layout。\n");

    printf("\n各原子铺到 (128,64) 之后的 consumer 侧列读冲突:\n");
    auto sh = make_shape(Int<128>{}, Int<64>{});
    auto show = [&](const char* tag, auto lay) {
        printf("  %-8s 列读 = %2d-way   cosize = %d\n", tag, col_conflict_half(lay),
               int(cosize(lay)));
    };
    show("SW128", tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{}, sh));
    show("SW64", tile_to_shape(GMMA::Layout_K_SW64_Atom<half_t>{}, sh));
    show("SW32", tile_to_shape(GMMA::Layout_K_SW32_Atom<half_t>{}, sh));
    show("INTER", tile_to_shape(GMMA::Layout_K_INTER_Atom<half_t>{}, sh));
    show("plain", make_layout(sh, make_stride(Int<64>{}, Int<1>{})));

    printf("\n  (这一列数字取决于 tile 铺开后行怎么分布, 不是\"SW128 一定最好\"。\n");
    printf("   选原子的依据是下面的 K 整除规则, 不是这一列。)\n");

    printf("\ntile_to_shape 把原子铺到需要的大小:\n");
    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{}, sh);
    printf("  tile_to_shape(SW128, (128,64))        = ");
    print(sA);
    printf("\n");
    auto sA3 = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                             make_shape(Int<128>{}, Int<64>{}, Int<3>{}));
    printf("  加一个 PIPE=3 的 stage 维             = ");
    print(sA3);
    printf("\n  (cute_06 的多 stage 流水线就靠这第三维)\n");

    printf("\n唯一的规则: BK 必须能被原子的 K 长度整除。\n");
    printf("  BK (half) | SW128(64) | SW64(32) | SW32(16) | INTER(8)\n");
    printf("     64     |     可    |    可    |    可    |   可\n");
    printf("     32     |   不可    |    可    |    可    |   可\n");
    printf("     16     |   不可    |  不可    |    可    |   可\n");
    printf("\n违反了会编译期报: \"tile_to_shape: block shape does not divide the target shape\"\n");
    printf("实用规则: 选能用的里面 K 最长的 (对齐最大 = 访存最宽)。BK=64 选 SW128。\n");
}

int main() {
    printf("cute_04 v2 —— SM90: TMA 与 WGMMA 对 smem layout 的要求\n");
    printf("对应 README §5    需要 -arch=sm_90a\n");

    section51_what_is_tma();
    section52_tma_layouts();
    section53_wgmma();
    section54_gmma_atoms();

    print_separator("小结");
    printf("  §5.1  TMA = 一个线程描述整块, 硬件搬; descriptor 里带 swizzle 模式\n");
    printf("  §5.2  TMA 不挑 layout: 五种全搬对, 连 plain row-major 也一样\n");
    printf("  §5.3  WGMMA 挑: plain / padded 编译期被拒 (Not a canonical GMMA_K Layout)\n");
    printf("  §5.3  SM90 没有 fragment 寄存器, descriptor 直读 smem -> 不需要 ldmatrix\n");
    printf("  §5.4  四个官方原子 M 全 = 4; 选原子只看 BK 能否被 K 长度整除\n");
    printf("\n所以: SM90 上必须用 swizzle, 而且只能用官方那几种。\n");
    printf("\nv2 OK\n");
    return 0;
}
