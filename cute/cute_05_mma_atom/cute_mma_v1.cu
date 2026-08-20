// cute_05 v1 —— WGMMA: Hopper 把 MMA 改成了什么样
//
// 对应 README §3。
//
// v0 里的 Ampere MMA 是这样工作的:
//     smem --(ldmatrix, 每线程搬自己那份)--> 寄存器 --> mma 指令
// 三个操作数 A/B/C 全在寄存器里, 一个 warp (32 线程) 发一条指令。
//
// Hopper 的 WGMMA 把这条链路改了两处, 这一版就是为了把这两处讲清楚:
//
//   1. **发指令的单位从 warp 变成 warpgroup** (32 -> 128 线程)
//   2. **A/B 不再进寄存器**: 硬件直接按 descriptor 去读 smem
//      -> mainloop 里那句 copy(tCsA, tCrA) 消失了, ldmatrix 这一步被硬件吃掉
//
// 代价是 WGMMA 对 smem 的摆法有硬性要求 (cute_04 §4.2/§4.5 已经撞过这堵墙):
// smem layout 必须来自 GMMA::Layout_*_Atom 系列, 普通 row-major 编译期就被拒。
//
//   §3.1  WGMMA vs MMA: 把两个 atom 并排打出来   (README §3.1)
//   §3.2  fragment 变成了 descriptor              (README §3.2)
//   §3.3  跑一条真的 WGMMA                        (README §3.3)
//   §3.4  smem layout: 谁能用, 谁不能用           (README §3.4)
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_mma_v1

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 全局配置
//
// WGMMA 的原子形状是 64x64x16 (M=64, N=64, K=16), 一个 warpgroup = 128 线程。
// 但 K=16 只是**一条指令**的 K; smem layout 的 K 必须能被 GMMA atom 的 K 整除,
// SW128 atom 对 half 是 (8,64), 所以 K 至少取 64。(见 README §3.4 的表)
//
// 数据摆法 (TN):
//   A: BM x BK = 64 x 64 half, row-major, stride = (BK,1) = (64,1)
//   B: BN x BK = 64 x 64 half, row-major, stride = (BK,1) = (64,1)   <- 存 B^T
//   C: BM x BN = 64 x 64 float, row-major, stride = (BN,1) = (64,1)
//
// 一个 CTA (128 线程 = 1 warpgroup) 一次算完整个 64x64x64。
// ---------------------------------------------------------------------------
constexpr int BM = 64, BN = 64, BK = 64;
constexpr int NTHR = 128;  // = size(TiledMMA) = 一个 warpgroup

// WGMMA 的 TiledMMA。SS = 两个操作数都从 smem 读 (Smem-Smem)。
// GMMA::Major::K 表示 A 和 B 在 smem 里都是 K 方向连续 —— 和 TN 摆法一致。
CUTE_HOST_DEVICE static auto make_wgmma() {
    return make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
}

// ===========================================================================
// §3.1  把两个 atom 并排打出来
//
// 这一节不跑 kernel, 只是把 v0 的 SM80 atom 和这一版的 SM90 atom 打在一起,
// 让差别自己显出来。README §3.1 有对这张输出的逐行解读。
// ===========================================================================
static void compare_atoms() {
    print_separator("§3.1  MMA vs WGMMA —— 两个 atom 并排看");

    printf("\n  --- Ampere: SM80_16x8x16_F32F16F16F32_TN (v0 用的那个) ---\n");
    print(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}));
    printf("\n");

    printf("\n  --- Hopper: SM90_64x64x16_F32F16F16_SS ---\n");
    print(make_wgmma());
    printf("\n");

    printf("\n  三处差别:\n");
    printf("    1. ThrID:  _32 -> _128\n");
    printf("       一条 WGMMA 由**一个 warpgroup (4 个 warp)** 共同发出。\n");
    printf("       所以 blockDim 至少 128, 且必须是 128 的整数倍。\n");
    printf("\n    2. Shape_MNK: (16,8,16) -> (64,64,16)\n");
    printf("       一条指令算的块大了 16 倍 (128 vs 2048 个 C 元素)。\n");
    printf("\n    3. LayoutA_TV: 注意 SM90 的是 (_128,(_64,_16)):(_0,(_1,_64))\n");
    printf("       **线程那一维的 stride 是 _0** —— 意思是 A 不按线程切分!\n");
    printf("       每个线程「看到」的是整块 A。因为根本不是线程去读, 是硬件去读。\n");
    printf("       对比 SM80 的 ((_4,_8),(_2,_2,_2)):((_32,_1),...) —— 那个 _32\n");
    printf("       是实打实的「线程 i 拿第 32*i 个元素」。\n");
}

// ===========================================================================
// §3.2  fragment 变成了 descriptor
//
// v0 里 partition_fragment_A 给出的是寄存器 tensor。WGMMA 下, A/B 的
// "fragment" 是 GMMA::DescriptorIterator —— 一个描述 smem 布局的句柄,
// 不是数据本身。
//
// 这解释了为什么 mainloop 里没有 copy(tCsA, tCrA):
// 没有东西要搬, 硬件拿着 descriptor 自己去 smem 取。
// ===========================================================================
// 注意这里必须在 **kernel 里** 打印。
// descriptor 是从 smem 的实际地址算出来的 (cast_smem_ptr_to_uint),
// 在 host 上拿一个 nullptr 的 smem tensor 去 make_fragment_A, CuTe 会打印
// "cast_smem_ptr_to_uint not supported but used"。
// 这本身就是一条信息: **descriptor 是运行时的东西, 不像 layout 那样纯编译期。**
template <class SLayA>
__global__ void show_descriptor_kernel(SLayA slayA) {
    __shared__ __align__(128) half_t rawA[cosize_v<SLayA>];
    auto sA = make_tensor(make_smem_ptr(rawA), slayA);

    auto mma = make_wgmma();
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);

    if (threadIdx.x == 0) {
        auto tCsA = thr.partition_A(sA);
        printf("\n  partition_A(sA)     -> ");
        print(tCsA);
        printf("\n    ^ 还是 smem tensor (看 smem_ptr 前缀)\n");

        auto tCrA = thr.make_fragment_A(tCsA);
        printf("\n  make_fragment_A(..) -> ");
        print(tCrA);
        printf("\n    ^ **GMMA::DescriptorIterator** —— 不是寄存器, 是「去哪读」的描述\n");

        auto gC = make_tensor(make_gmem_ptr((float*)nullptr), make_shape(Int<BM>{}, Int<BN>{}),
                              make_stride(Int<BN>{}, Int<1>{}));
        auto tCrC = thr.partition_fragment_C(gC);
        printf("\n  partition_fragment_C -> ");
        print(tCrC);
        printf("\n    ^ 累加器仍是**真寄存器**, 每线程 %d 个 float\n", int(size(tCrC)));
        printf("      64x64 = 4096 个 C 元素 / 128 线程 = 32 个/线程。对上了。\n");
    }
}

static void show_descriptor() {
    print_separator("§3.2  fragment 变成了 descriptor");

    // smem layout 必须来自 GMMA atom 系列 (§3.4 会验证「为什么必须」)
    auto slayA = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                               make_shape(Int<BM>{}, Int<BK>{}));

    printf("\n  smem layout (tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>, (64,64))):\n    ");
    print(slayA);
    printf("\n    ^ 前面的 Sw<3,4,3> 就是 cute_04 讲的 swizzle, GMMA atom 自带\n");

    show_descriptor_kernel<<<1, NTHR>>>(slayA);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n  所以 Hopper 的 mainloop 比 Ampere 少一步:\n");
    printf("    Ampere: smem --ldmatrix--> 寄存器 --mma--> 累加器\n");
    printf("    Hopper: smem ----------------wgmma-------> 累加器\n");
}

// ===========================================================================
// §3.3  跑一条真的 WGMMA
//
// 完整链路 (和 v0 比, 中间少了一步):
//   1. gmem -> smem  : 普通 copy (这一章不关心搬运效率, 那是 cute_04 的事)
//   2. 清累加器      : clear()
//   3. 算            : warpgroup_arrive(); gemm(); commit; wait  <- 四句一组
//   4. 寄存器 -> gmem: copy()
//
// 第 3 步的四句是 WGMMA 的固定套路, 不能省:
//   warpgroup_arrive()        告诉硬件"我要开始发 WGMMA 了, smem 已就绪"
//   gemm(mma, ...)            发指令 (**异步**, 发完就返回)
//   warpgroup_commit_batch()  把刚发的这批打包
//   warpgroup_wait<0>()       等这批做完 (<0> = 允许 0 批still在飞)
//
// 少了 wait 就会在累加器还没写完时去读它 —— 数据错且难查。
// ===========================================================================
template <class SLayA, class SLayB>
__global__ void wgmma_one(const half_t* A, const half_t* B, float* C, SLayA slayA, SLayB slayB) {
    // smem: WGMMA 要求 128B 对齐 (和 TMA 一样的要求, cute_04 §2.1)
    __shared__ __align__(128) half_t rawA[cosize_v<SLayA>];
    __shared__ __align__(128) half_t rawB[cosize_v<SLayB>];

    auto sA = make_tensor(make_smem_ptr(rawA), slayA);
    auto sB = make_tensor(make_smem_ptr(rawB), slayB);

    // 1. gmem -> smem。这里用最直白的按行搬, 每个线程搬若干个元素。
    //    (怎么搬得快是 cute_04 的主题, 这一章只关心搬到之后怎么算)
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(Int<BM>{}, Int<BK>{}),
                          make_stride(Int<BK>{}, Int<1>{}));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(Int<BN>{}, Int<BK>{}),
                          make_stride(Int<BK>{}, Int<1>{}));

    for (int i = threadIdx.x; i < BM * BK; i += NTHR) sA(i / BK, i % BK) = gA(i / BK, i % BK);
    for (int i = threadIdx.x; i < BN * BK; i += NTHR) sB(i / BK, i % BK) = gB(i / BK, i % BK);

    __syncthreads();  // smem 写完才能让 WGMMA 去读

    // 2. 累加器
    auto mma = make_wgmma();
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(Int<BM>{}, Int<BN>{}),
                          make_stride(Int<BN>{}, Int<1>{}));
    auto tCrC = thr.partition_fragment_C(gC);
    clear(tCrC);

    // A/B 的 "fragment" —— 实际是 descriptor, 不占寄存器
    auto tCsA = thr.partition_A(sA);
    auto tCsB = thr.partition_B(sB);
    auto tCrA = thr.make_fragment_A(tCsA);
    auto tCrB = thr.make_fragment_B(tCsB);

    // 3. WGMMA 四句套路
    warpgroup_arrive();
    gemm(mma, tCrA, tCrB, tCrC);  // 注意: 没有 copy(tCsA,tCrA)!
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    // 4. 写回
    copy(tCrC, thr.partition_C(gC));
}

static void run_wgmma_one() {
    print_separator("§3.3  跑一条真的 WGMMA");

    half_t *h_A = new half_t[BM * BK], *h_B = new half_t[BN * BK];
    float *h_C = new float[BM * BN], *h_ref = new float[BM * BN];
    fill_pm1(h_A, BM * BK, 11);
    fill_pm1(h_B, BN * BK, 22);
    gemm_cpu(h_A, h_B, h_ref, BM, BN, BK);

    half_t *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, BM * BK * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_B, BN * BK * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_C, BM * BN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, BM * BK * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, BN * BK * sizeof(half_t), cudaMemcpyHostToDevice));

    // host 侧描述 smem 摆法 —— GMMA 官方 atom, 自带 swizzle
    auto slayA = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                               make_shape(Int<BM>{}, Int<BK>{}));
    auto slayB = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                               make_shape(Int<BN>{}, Int<BK>{}));

    printf("\n  C[%dx%d] = A[%dx%d] * B[%dx%d]^T\n", BM, BN, BM, BK, BN, BK);
    printf("  由 1 个 warpgroup (%d 线程) 发 WGMMA 完成\n", NTHR);
    printf("  K=%d / 一条指令的 K=16 -> 内部发了 %d 条 WGMMA (CuTe 自动展开)\n", BK, BK / 16);

    wgmma_one<<<1, NTHR>>>(d_A, d_B, d_C, slayA, slayB);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, BM * BN * sizeof(float), cudaMemcpyDeviceToHost));

    auto r = check_close(h_C, h_ref, BM * BN);
    printf("\n  与 CPU 参考比对: %s   (bad=%d, maxerr=%g)\n", r.ok() ? "完全一致" : "不一致", r.bad,
           r.maxerr);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_ref;
}

// ===========================================================================
// §3.4  smem layout: 谁能用, 谁不能用
//
// WGMMA 直接读 smem, 所以它对 smem 的摆法有**编译期**的硬性要求。
// 这一节把四种 GMMA 官方 atom 的几何打出来, 并说明那条 K 整除规则。
//
// 关于"普通 row-major 会怎样": 它在编译期就被拒:
//   static assertion failed: 「Not a canonical GMMA_K Layout: Expected stride failure.」
// 这一条无法在同一个程序里演示 —— 一旦写进来整个文件就编译不过。
// exercises 里有一道题专门让你去撞这堵墙。
// ===========================================================================
template <class Atom>
static void show_gmma_atom(const char* name, int need_k) {
    auto lay = tile_to_shape(Atom{}, make_shape(Int<BM>{}, Int<BK>{}));
    printf("    %-22s K 需被 %2d 整除   ->  ", name, need_k);
    print(lay);
    printf("\n");
}

static void show_gmma_layouts() {
    print_separator("§3.4  WGMMA 认哪些 smem layout");

    printf("\n  GMMA 官方 smem atom (half_t, K-major 系列):\n\n");
    show_gmma_atom<GMMA::Layout_K_SW128_Atom<half_t>>("Layout_K_SW128_Atom", 64);
    show_gmma_atom<GMMA::Layout_K_SW64_Atom<half_t>>("Layout_K_SW64_Atom", 32);
    show_gmma_atom<GMMA::Layout_K_SW32_Atom<half_t>>("Layout_K_SW32_Atom", 16);
    show_gmma_atom<GMMA::Layout_K_INTER_Atom<half_t>>("Layout_K_INTER_Atom", 8);

    printf("\n  两条规则:\n");
    printf("    1. K 必须被 atom 自带的 K 长度整除, 否则编译期报\n");
    printf("       「block shape does not divide the target shape」。\n");
    printf("       本文件 BK = %d, 所以四种都能用。\n", BK);
    printf("    2. 普通 row-major / padding 过的 layout 直接**编译期拒绝**:\n");
    printf("       「Not a canonical GMMA_K Layout: Expected stride failure.」\n");
    printf("       这就是 cute_04 §4.2 说「padding 在 SM90 上是死路」的真正原因 ——\n");
    printf("       不是 TMA 拒绝 (TMA 全都能搬), 是 WGMMA 拒绝。\n");

    printf("\n  为什么 WGMMA 这么挑? 因为它是硬件直接寻址 smem:\n");
    printf("    descriptor 里只有 (起始地址, 几个固定的 stride 编码),\n");
    printf("    放不下任意 layout。所以只能从硬件认识的那几种里选。\n");
}

// ===========================================================================
int main() {
    printf("cute_05 v1 —— WGMMA: Hopper 把 MMA 改成了什么样\n");
    printf("对应 README §3\n");

    compare_atoms();
    show_descriptor();
    run_wgmma_one();
    show_gmma_layouts();

    print_separator("小结");
    printf("  §3.1  warp (32) -> warpgroup (128); 16x8x16 -> 64x64x16\n");
    printf("  §3.2  A/B 的 fragment 变成 DescriptorIterator, 不占寄存器\n");
    printf("        -> mainloop 里没有 copy(tCsA, tCrA), ldmatrix 被硬件吃掉\n");
    printf("        -> 但累加器 C 仍是真寄存器 (每线程 32 个 float)\n");
    printf("  §3.3  固定四句: arrive / gemm / commit_batch / wait<0>\n");
    printf("  §3.4  代价: smem layout 必须是 GMMA 认识的那几种, 编译期强制\n");
    printf("\n  现在「怎么算」解决了。但 WGMMA 读的是 smem —— 数据怎么进 smem?\n");
    printf("  v2: 用 TMA 把数据喂进来, 这是 Hopper 的另一半。\n");

    printf("\nv1 OK\n");
    return 0;
}
