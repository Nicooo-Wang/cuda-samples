// cute_05 练习：把 TODO 填掉，然后 make run
//
// 题目见 README 的"练习"一节。参考解答在 solutions.md。
// 每题都有一个自动检查，填对了会打印 PASS。
//
// 六道题对应 README 的:
//   1 -> §1.3  从 TV 布局手算一个线程拿几个元素
//   2 -> §3.2  fragment 是寄存器还是描述符
//   3 -> §3.3  手写 WGMMA 的四句
//   4 -> §5.1  修一个把 WGMMA 当成 SM80 拼的 bug
//   5 -> §4.2  手写一段 TMA 搬运
//   6 -> §5    手写一个单 CTA 的 TMA + WGMMA GEMM

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/arch/barrier.h>
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
// 练习 1 —— 从 TV 布局手算 (README §1.3)
//
// LayoutA_TV = ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
//             └─ 32 线程 4x8 ─┘└─ 每股 8 元素 ─┘└线程 stride┘└─ 值 stride ─┘
//
// 一个线程拿到的 8 个 A 元素的**逻辑坐标**是((m,n), k)级联。这一题只要求:
// 数出 B 和 C 每个线程各拿几个元素 (不看坐标, 只数个数), 并和 total/32 对上。
// 把 ①② 填成整数即可。
// ---------------------------------------------------------------------------
static void ex1() {
    printf("\n===== 练习 1: 从 TV 布局数每个线程拿几个 =====\n");

    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{});
    ThrMMA thr = mma.get_thread_slice(0);

    auto gA = make_tensor(make_gmem_ptr((half_t*)nullptr),
                          make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}));
    auto gB = make_tensor(make_gmem_ptr((half_t*)nullptr),
                          make_shape(Int<8>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}));
    auto gC = make_tensor(make_gmem_ptr((float*)nullptr),
                          make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));

    int a = size(thr.partition_fragment_A(gA));
    int b = size(thr.partition_fragment_B(gB));
    int c = size(thr.partition_fragment_C(gC));

    printf("  fragment A: %d 个 half/线程 (期望 256/32 = 8)\n", a);
    printf("  fragment B: 请数出来 (B 是 8x16 = 128/32 = ?)\n");
    printf("  fragment C: 请数出来 (C 是 16x8 = 128/32 = ?)\n");

    // TODO ①: 把 0 换成 B 每个线程该拿的个数
    int b_expect = 0;
    // TODO ②: 把 0 换成 C 每个线程该拿的个数
    int c_expect = 0;

    expect("A 每人 8 个", a == 8);
    expect("B 每人 4 个", b == b_expect && b_expect == 4);
    expect("C 每人 4 个", c == c_expect && c_expect == 4);
}

// ---------------------------------------------------------------------------
// 练习 2 —— fragment 是寄存器还是描述符 (README §3.2)
//
// v1 的关键结论分三段:
//   (1) SM80 的 partition_fragment_A 返回**寄存器** (真持有数据的数组)
//   (2) WGMMA 的 make_fragment_A 返回 GMMA::DescriptorIterator (描述符)
//   (3) 累加器 C 在两种架构下都是**真寄存器**
//
// 下面已经写好了 (1) 和 (3) 的代码, 请你补 (2): 同样拿 smem 的 A 做 partition_A,
// 再 make_fragment_A。填完应该能编译, 并且会让你数清楚每个返回值几个元素。
// ---------------------------------------------------------------------------
template <class SLA>
__global__ void ex2_kernel(SLA slayA, int* c2) {
    __shared__ __align__(128) half_t rawA[cosize_v<SLA>];
    auto sA = make_tensor(make_smem_ptr(rawA), slayA);

    auto mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);

    if (threadIdx.x == 0) {
        // TODO ①: 对 sA 做 partition_A
        auto tAsA = thr.partition_A(sA);
        // TODO ②: 对 tAsA 做 make_fragment_A
        auto tArA = thr.make_fragment_A(tAsA);

        printf("  partition_A 的 shape: "); print(shape(tAsA)); printf("\n");
        printf("  make_fragment_A 的 shape: "); print(shape(tArA)); printf("\n");
        printf("  注意看第二个的输出开头: GMMA::DescriptorIterator, 不是寄存器数组\n");
        printf("  但 tArA 元素的个数 (size) = %d\n", int(size(tArA)));

        // WGMMA 的累加器 C 仍然是真寄存器
        auto gC = make_tensor(make_gmem_ptr((float*)nullptr),
                              make_shape(Int<64>{}, Int<64>{}), make_stride(Int<64>{}, Int<1>{}));
        auto tCrC = thr.partition_fragment_C(gC);
        c2[0] = int(size(tCrC));  // 32
    }
}

static void ex2(int* counts) {
    printf("\n===== 练习 2: fragment 是寄存器还是描述符 =====\n");

    // --- SM80: partition_fragment_A 是真寄存器 ---
    {
        TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{});
        ThrMMA thr = mma.get_thread_slice(0);
        auto gA = make_tensor(make_gmem_ptr((half_t*)nullptr),
                              make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}));
        auto aA = thr.partition_fragment_A(gA);
        printf("  SM80 partition_fragment_A: %d 个元素 -> 真寄存器 ✓\n", int(size(aA)));
        counts[0] = int(size(aA));  // 8
    }

    // --- WGMMA: make_fragment_A 返回描述符 ---
    // 注意: descriptor 是从 smem 的**实际地址**算出来的, 必须在 kernel 里看。
    // 在 host 上用一块普通数组当 smem, make_fragment_A 会报
    // "cast_smem_ptr_to_uint not supported" —— 这本身就是一条信息:
    // **descriptor 是运行时的东西, 不是 layout 那样纯编译期。**
    {
        auto slayA = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                                   make_shape(Int<64>{}, Int<64>{}));
        printf("  smem layout: "); print(slayA); printf("\n");

        // counts 要能被 kernel 写, 得放设备上。只让 kernel 写第 2 格,
        // host 侧再合回 counts[2] —— 不要把整个数组拷来拷去覆盖 counts[0]。
        int* d_c2;
        CUDA_CHECK(cudaMalloc(&d_c2, sizeof(int)));
        ex2_kernel<<<1, 128>>>(slayA, d_c2);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&counts[2], d_c2, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_c2));
        printf("  WGMMA partition_fragment_C: %d 个 float -> 仍是真寄存器 ✓\n", counts[2]);
    }

    expect("SM80 A fragment 每人 8 个元素", counts[0] == 8);
    expect("WGMMA 累加器每人 32 个 float", counts[2] == 32);
}

// ---------------------------------------------------------------------------
// 练习 3 —— 手写 WGMMA 四句 (README §3.3)
//
// 下面 kernel 的 WGMMA 只差一句。把它填对即可。
// 提示: 四句固定套路里, 你现在缺的是"等它们做完"那句。
// ---------------------------------------------------------------------------
template <class SLA, class SLB>
__global__ void wgmma_ex(const half_t* A, const half_t* B, float* C, SLA sla, SLB slb) {
    constexpr int BM = 64, BN = 64, BK = 64;
    __shared__ __align__(128) half_t rawA[cosize_v<SLA>];
    __shared__ __align__(128) half_t rawB[cosize_v<SLB>];
    auto sA = make_tensor(make_smem_ptr(rawA), sla);
    auto sB = make_tensor(make_smem_ptr(rawB), slb);

    auto gA = make_tensor(make_gmem_ptr(A), make_shape(Int<BM>{}, Int<BK>{}),
                          make_stride(Int<BK>{}, Int<1>{}));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(Int<BN>{}, Int<BK>{}),
                          make_stride(Int<BK>{}, Int<1>{}));
    for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) sA(i / BK, i % BK) = gA(i / BK, i % BK);
    for (int i = threadIdx.x; i < BN * BK; i += blockDim.x) sB(i / BK, i % BK) = gB(i / BK, i % BK);
    __syncthreads();

    auto mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(Int<BM>{}, Int<BN>{}),
                          make_stride(Int<BN>{}, Int<1>{}));
    auto tCrC = thr.partition_fragment_C(gC);
    clear(tCrC);

    auto tCrA = thr.make_fragment_A(thr.partition_A(sA));
    auto tCrB = thr.make_fragment_B(thr.partition_B(sB));

    warpgroup_arrive();
    gemm(mma, tCrA, tCrB, tCrC);
    warpgroup_commit_batch();
    // TODO ①: 填上缺的那句 (等这批 WGMMA 做完)
    warpgroup_wait<0>();

    copy(tCrC, thr.partition_C(gC));
}

static void ex3() {
    printf("\n===== 练习 3: 完整 WGMMA 四句 =====\n");

    constexpr int BM = 64, BN = 64, BK = 64;
    half_t *hA = new half_t[BM * BK], *hB = new half_t[BN * BK];
    float *hC = new float[BM * BN], *hR = new float[BM * BN];
    for (int i = 0; i < BM * BK; ++i) hA[i] = half_t(float((i % 3) - 1));
    for (int i = 0; i < BN * BK; ++i) hB[i] = half_t(float(((i * 7) % 3) - 1));
    // CPU 参考 (在 host 用 float 累加)
    for (int m = 0; m < BM; ++m)
        for (int n = 0; n < BN; ++n) {
            float acc = 0;
            for (int k = 0; k < BK; ++k) acc += float(hA[m * BK + k]) * float(hB[n * BK + k]);
            hR[m * BN + n] = acc;
        }

    half_t *dA, *dB;
    float *dC;
    CUDA_CHECK(cudaMalloc(&dA, BM * BK * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&dB, BN * BK * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&dC, BM * BN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, BM * BK * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, BN * BK * sizeof(half_t), cudaMemcpyHostToDevice));

    auto sla = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{}, make_shape(Int<BM>{}, Int<BK>{}));
    auto slb = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{}, make_shape(Int<BN>{}, Int<BK>{}));
    wgmma_ex<<<1, 128>>>(dA, dB, dC, sla, slb);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC, dC, BM * BN * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < BM * BN; ++i)
        if (fabsf(hC[i] - hR[i]) > 1e-2f) ++bad;
    printf("  结果与参考不一致的元素个数 = %d\n", bad);
    expect("WGMMA 四句齐全 -> 结果正确", bad == 0);
}

// ---------------------------------------------------------------------------
// 练习 4 —— 修一个把 WGMMA 当成 SM80 拼的 bug (README §5.1)
//
// 有人想在 128x64 的 C tile 上跑 WGMMA, 想起 v0 §2.3 的"多个 warp 拼",
// 于是写了 make_tiled_mma(atom, Layout<Shape<_2,_1,_1>>{})。
//
// 这在 SM90 上是错的 (WGMMA 原子已经要求 128 线程)。
// 请修: 既不引入第二个 warpgroup, 又让 128x64 全覆盖。
// 提示: 裸原子 + CuTe 自动重复 (partition_fragment_C 会给出 MMA_M=_2)。
// ---------------------------------------------------------------------------
static void ex4() {
    printf("\n===== 练习 4: 修 WGMMA 的 TiledMMA 陷阱 =====\n");

    constexpr int BM = 128, BN = 64;

    // TODO ①: 把 make_tiled_mma 修对 —— 不要让它 size==256 (两个 warpgroup)
    auto mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{},
                              Layout<Shape<_2, _1, _1>>{});  // <-- 改成裸原子

    auto gC = make_tensor(make_gmem_ptr((float*)nullptr),
                          make_shape(Int<BM>{}, Int<BN>{}), make_stride(Int<BN>{}, Int<1>{}));
    ThrMMA thr = mma.get_thread_slice(0);
    auto tCrC = thr.partition_fragment_C(gC);

    int threads = int(size(mma));
    int c_per_thread = int(size(tCrC));
    printf("  size(mma)      = %d   (期望 128 = 一个 warpgroup)\n", threads);
    printf("  fragmentC/线程 = %d   (期望 64 = 128x64/128)\n", c_per_thread);

    expect("只用一个 warpgroup (128 线程)", threads == 128);
    expect("128x64 全覆盖 -> 每线程 64 个 C", c_per_thread == 64);
}

// ---------------------------------------------------------------------------
// 练习 5 —— 手写一段 TMA 搬运 (README §4.2)
//
// 场景: 一个 CTA (128 线程) 要搬 A[128x64] (128x64 half row-major) 进 smem,
// 然后用 WGMMA 算。这里的注意点不是算, 而是**把 TMA 搬对**。
//
// 已经填好大半, 剩三处 TODO:
//   ① 构造坐标 tensor (条件 1)
//   ② 事务字节数 txb (条件 5 配套)
//   ③ 真正发 TMA 的那一行 (条件 2/3 在 host 侧, 见 run_ex5)
// ---------------------------------------------------------------------------
template <class TmaA, class SLA3>
__global__ void tma_ex(CUTLASS_GRID_CONSTANT TmaA const tma_a, float* out, SLA3 sla3) {
    __shared__ __align__(128) half_t rawS[cosize_v<SLA3>];
    __shared__ __align__(8) uint64_t bar[1];

    auto sA = make_tensor(make_smem_ptr(rawS), sla3);  // (128,64,1)
    constexpr int BM = 128, BK = 64;

    // TODO ①: 用 tma_a.get_tma_tensor 得到坐标 tensor (形状 M=128, K=64)
    auto mA = tma_a.get_tma_tensor(make_shape(Int<128>{}, Int<64>{}));
    auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(0, _));

    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                      group_modes<0, 2>(gA));
    // TODO ②: 这一轮 TMA 要搬多少字节 (A 是 128x64 half)
    constexpr int ntrans = int(sizeof(half_t)) * BM * BK;

    int warp = cutlass::canonical_warp_idx_sync();
    int one = cute::elect_one_sync();
    using Bar = cutlass::arch::ClusterTransactionBarrier;
    if (warp == 0 && one) Bar::init(&bar[0], 1);
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    if (warp == 0 && one) {
        Bar::arrive_and_expect_tx(&bar[0], ntrans);
        // TODO ③: 发 TMA (copy(tma_a.with(bar[0]), tAgA(_,0), tAsA(_, Int<0>{})))
        copy(tma_a.with(bar[0]), tAgA(_, 0), tAsA(_, Int<0>{}));
    }
    Bar::wait(&bar[0], 0);
    __syncthreads();

    // 验证: 每个线程核对 smem 里的一小段 (A[i][j] = float(i*64+j) % ... )
    // 我们会在 host 把 A 填成 A[i*64+j] = i + 256*j 这种可逆的, 便于核对。
    bool ok = true;
    int nthreads = 128;
    for (int i = threadIdx.x; i < BM * BK; i += nthreads) {
        int r = i / BK, c = i % BK;
        // 从 smem 读出来
        float got = float(sA(r, c, Int<0>{}));
        float want = float(r) + 256.f * float(c);
        if (got != want) ok = false;
    }
    if (threadIdx.x == 0)
        out[0] = ok ? 1.0f : 0.0f;  // host 端查这个
}

static void ex5() {
    printf("\n===== 练习 5: 手写 TMA 搬运 =====\n");

    constexpr int BM = 128, BK = 64;
    half_t* hA = new half_t[BM * BK];
    for (int i = 0; i < BM * BK; ++i) hA[i] = half_t(float((i / BK)) + 256.f * float((i % BK)));

    half_t* dA;
    CUDA_CHECK(cudaMalloc(&dA, BM * BK * sizeof(half_t)));
    CUDA_CHECK(cudaMemcpy(dA, hA, BM * BK * sizeof(half_t), cudaMemcpyHostToDevice));

    auto mA = make_tensor(make_gmem_ptr(dA), make_shape(Int<BM>{}, Int<BK>{}),
                          make_stride(Int<BK>{}, Int<1>{}));
    auto sla3 = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                              make_shape(Int<BM>{}, Int<BK>{}, Int<1>{}));
    auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, mA, sla3(_, _, Int<0>{}),
                               make_shape(Int<BM>{}, Int<BK>{}));

    float* dOut;
    CUDA_CHECK(cudaMalloc(&dOut, sizeof(float)));
    tma_ex<<<1, 128>>>(tma_a, dOut, sla3);
    CUDA_CHECK(cudaDeviceSynchronize());
    float hOut;
    CUDA_CHECK(cudaMemcpy(&hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost));

    expect("smem 里的每个元素都和 gmem 对上", hOut == 1.0f);

    cudaFree(dA);
    cudaFree(dOut);
    delete[] hA;
}

// ---------------------------------------------------------------------------
// 练习 6 —— 手写一个单 CTA 的 TMA + WGMMA GEMM (README §5)
//
// 把练习 5 的 TMA 和练习 3 的 WGMMA 拼起来, 加一层 k 循环。
// 这是 capstone 去掉 grid 那层之后的最小版。
//
// 尺寸: BM=64, BN=64, BK=64, K=128 (2 个 k tile)。128 线程。
// 注意 mbarrier 的 phase 每轮要翻转 (k & 1)。
// ---------------------------------------------------------------------------
template <class TmaA, class TmaB, class SLA3, class SLB3>
__global__ void gemm_ex(CUTLASS_GRID_CONSTANT TmaA const ta, CUTLASS_GRID_CONSTANT TmaB const tb,
                        const half_t* A, const half_t* B, float* C, SLA3 sla3, SLB3 slb3) {
    constexpr int BM = 64, BN = 64, BK = 64;
    constexpr int NK = 2;
    __shared__ __align__(128) half_t rawA[cosize_v<SLA3>];
    __shared__ __align__(128) half_t rawB[cosize_v<SLB3>];
    __shared__ __align__(8) uint64_t bar[1];

    auto sA = make_tensor(make_smem_ptr(rawA), sla3);
    auto sB = make_tensor(make_smem_ptr(rawB), slb3);

    auto mA = ta.get_tma_tensor(make_shape(Int<BM>{}, Int<BK * NK>{}));
    auto mB = tb.get_tma_tensor(make_shape(Int<BN>{}, Int<BK * NK>{}));
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

    for (int k = 0; k < NK; ++k) {
        if (warp == 0 && one) {
            Bar::arrive_and_expect_tx(&bar[0], txb);
            copy(ta.with(bar[0]), tAgA(_, k), tAsA(_, Int<0>{}));
            copy(tb.with(bar[0]), tBgB(_, k), tBsB(_, Int<0>{}));
        }
        Bar::wait(&bar[0], k & 1);

        auto tCrA = thr.make_fragment_A(thr.partition_A(sA2));
        auto tCrB = thr.make_fragment_B(thr.partition_B(sB2));
        warpgroup_arrive();
        gemm(mma, tCrA, tCrB, tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        __syncthreads();  // 算完才能覆盖 smem
    }

    copy(tCrC, thr.partition_C(mC));
}

static void ex6() {
    printf("\n===== 练习 6: 单 CTA 的 TMA + WGMMA GEMM =====\n");

    constexpr int BM = 64, BN = 64, BK = 64, NK = 2;
    constexpr int GK = BK * NK;
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

    gemm_ex<<<1, 128>>>(ta, tb, dA, dB, dC, sla3, slb3);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC, dC, BM * BN * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int m = 0; m < BM; ++m)
        for (int n = 0; n < BN; ++n)
            if (fabsf(hC[m * BN + n] - hR[m * BN + n]) > 1e-2f) ++bad;
    printf("  结果与参考不一致的元素个数 = %d\n", bad);
    expect("TMA 搬 + WGMMA 算 -> 结果正确", bad == 0);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;
    delete[] hR;
}

// ===========================================================================
int main() {
    printf("cute_05 练习 —— 填 TODO 后 make run\n");

    int counts[3] = {0, 0, 0};
    ex1();
    ex2(counts);
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
