// cute_07 练习：把 TODO 填掉，然后 make run
//
// 题目见 README 的"练习"一节。参考解答在 solutions.md。
// 每题都有一个自动检查，填对了会打印 PASS。
//
// 三道题对应 README 的:
//   1 -> §2.1  tile 编号 -> 坐标的两种映射
//   2 -> §1.1  persistent 主循环的边界条件
//   3 -> §1.2  barrier phase 重置 (改错)

#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <cstdio>

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
// 练习 1 —— tile 编号 -> 坐标的两种映射 (README §2.1)
//
// 一个 8x8 的 tile 网格 (num_tiles_m = 8, num_tiles_n = 8)。
// 请实现两种映射:
//   ① row-major : by = id / num_tiles_n, bx = id % num_tiles_n
//   ② swizzled  : 按 SWIZ=2 个 tile 一组沿 M 排 (见代码里的公式)
//
// 检查点: 第 5 个 tile (id=5) 在两种映射下分别是什么坐标?
// ---------------------------------------------------------------------------
static void ex1() {
    printf("\n===== 练习 1: tile 映射 =====\n");

    constexpr int NUM_M = 8, NUM_N = 8;
    constexpr int SWIZ = 2;

    // TODO ①: row-major
    int by_rm = 0, bx_rm = 0;
    {
        int id = 5;
        by_rm = id / NUM_N;
        bx_rm = id % NUM_N;
    }

    // TODO ②: swizzled —— 按 SWIZ 个 tile 一组沿 M 排
    int by_sw = 0, bx_sw = 0;
    {
        int id = 5;
        int group = id / SWIZ;   // 第几组
        int within = id % SWIZ;  // 组内第几个
        int group_row = group / NUM_N;   // 组在哪一行
        int group_col = group % NUM_N;   // 组在哪一列
        by_sw = group_row * SWIZ + within;
        bx_sw = group_col;
    }

    printf("  id=5: row-major -> (%d,%d), swizzled -> (%d,%d)\n", by_rm, bx_rm, by_sw, bx_sw);

    expect("row-major: (0,5)", by_rm == 0 && bx_rm == 5);
    expect("swizzled: (1,2)", by_sw == 1 && bx_sw == 2);
}

// ---------------------------------------------------------------------------
// 练习 2 —— persistent 主循环的边界条件 (README §1.1)
//
// persistent kernel: grid = 4 个 CTA, 共 10 个 tile。
// CTA i 吃 tile_id = i, i+4, i+8, ...
// 请列出每个 CTA 吃到哪些 tile, 并算出有没有负载不均衡。
// ---------------------------------------------------------------------------
static void ex2() {
    printf("\n===== 练习 2: persistent 边界条件 =====\n");

    constexpr int NUM_CTAS = 4, NUM_TILES = 10;
    printf("  %d 个 CTA 吃 %d 个 tile:\n", NUM_CTAS, NUM_TILES);
    for (int i = 0; i < NUM_CTAS; ++i) {
        printf("    CTA %d: ", i);
        for (int tid = i; tid < NUM_TILES; tid += NUM_CTAS) printf("%d ", tid);
        printf("\n");
    }

    // TODO ①: 每个 CTA 平均吃几个 tile? (整数除法)
    int avg = NUM_TILES / NUM_CTAS;

    // TODO ②: 吃最多的 CTA 吃几个? (NUM_TILES 对 NUM_CTAS 取模, 前几个 CTA 多吃一个)
    int max_tiles = NUM_TILES / NUM_CTAS + (NUM_TILES % NUM_CTAS != 0 ? 1 : 0);

    printf("  平均 %.1f 个, 最多 %d 个\n", float(NUM_TILES) / NUM_CTAS, max_tiles);
    expect("平均 2.5 个", avg == 2);
    expect("最多 3 个 (负载不均衡 3/2.5 = 1.2x)", max_tiles == 3);
}

// ---------------------------------------------------------------------------
// 练习 3 —— barrier phase 重置 (改错, README §1.2)
//
// 下面 kernel 是 persistent 的简化版: 每个 CTA 吃 2 个 tile, 每个 tile
// 只做一次 TMA (单 stage)。它缺了"每个 tile 重置 barrier"这一步 ——
// 请补上, 否则第二个 tile 会死锁 (phase 对不上)。
//
// 提示: 在 for 循环体开头, 用 ProducerBar::init 重置 bar, 然后 fence + sync。
//       和 v0 的写法一样。
// ---------------------------------------------------------------------------
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/device_kernel.h>

template <class TmaA, class SLA3>
__global__ void persistent_ex(CUTLASS_GRID_CONSTANT TmaA const ta, float* out, int num_tiles,
                              SLA3 sla3) {
    __shared__ __align__(128) half_t rawS[cosize_v<SLA3>];
    __shared__ __align__(8) uint64_t bar[1];
    auto sA = make_tensor(make_smem_ptr(rawS), sla3);  // (BM,BK,1)

    constexpr int BM = 128, BK = 64;
    auto mA = ta.get_tma_tensor(make_shape(Int<BM * 4>{}, Int<BK>{}));  // (BM*4, BK): 4 个 M-tile  // 4 个 tile 的 K 方向
    auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(_, 0));  // (BM,BK,4) M-tiles

    auto [tAgA, tAsA] = tma_partition(ta, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                      group_modes<0, 2>(gA));
    constexpr int txb = sizeof(make_tensor_like(tensor<0>(tAsA)));

    int warp = cutlass::canonical_warp_idx_sync();
    int one = cute::elect_one_sync();
    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;
    if (warp == 0 && one) ProducerBar::init(&bar[0], 1);
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    for (int t = blockIdx.x; t < num_tiles; t += gridDim.x) {
        // ---- TODO ①: 每 tile 重置 barrier phase ----
        // 不重置的话, 第二个 tile 会死锁 (barrier 的 phase 还停在上一个 tile)。
        // 提示: 三行 —— ProducerBar::init(&bar[0], 1);
        //                 cutlass::arch::fence_barrier_init();
        //                 __syncthreads();
        // 现在这里是空的, 跑一下看会怎样 (死锁 / 数据错), 再填。

        if (warp == 0 && one) {
            ProducerBar::arrive_and_expect_tx(&bar[0], txb);
            copy(ta.with(bar[0]), tAgA(_, t), tAsA(_, Int<0>{}));
        }
        ProducerBar::wait(&bar[0], 0);
        __syncthreads();

        // 验证: 每个线程核对一段数据 (数据 = tile 序号, 便于区分)
        if (threadIdx.x == 0) {
            bool ok = true;
            for (int i = 0; i < BM; i += 128) {
                float got = float(sA(i, 0, Int<0>{}));
                if (got != float(t)) ok = false;
            }
            out[blockIdx.x * 4 + t] = ok ? 1.0f : 0.0f;
        }
    }
}

static void ex3() {
    printf("\n===== 练习 3: barrier phase 重置 =====\n");

    constexpr int BM = 128, BK = 64;
    constexpr int NUM_TILES = 4, NUM_CTAS = 2;

    half_t* hA = new half_t[BM * BK * NUM_TILES];
    // M-tile t 占据行 [t*BM, (t+1)*BM): 填 t, 便于核对
    for (int t = 0; t < NUM_TILES; ++t)
        for (int r = 0; r < BM; ++r)
            for (int c = 0; c < BK; ++c) hA[t * BM * BK + r * BK + c] = half_t(float(t));

    half_t* dA;
    CUDA_CHECK(cudaMalloc(&dA, BM * BK * NUM_TILES * sizeof(half_t)));
    CUDA_CHECK(cudaMemcpy(dA, hA, BM * BK * NUM_TILES * sizeof(half_t), cudaMemcpyHostToDevice));

    // 4 个 tile 沿 M 方向排: (BM*4, BK), 和 kernel 里的 get_tma_tensor 一致
    auto mA = make_tensor(make_gmem_ptr(dA), make_shape(Int<BM * NUM_TILES>{}, Int<BK>{}),
                          make_stride(Int<BK>{}, Int<1>{}));
    auto sla3 = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                              make_shape(Int<BM>{}, Int<BK>{}, Int<1>{}));
    auto ta = make_tma_atom(SM90_TMA_LOAD{}, mA, sla3(_, _, Int<0>{}),
                            make_shape(Int<BM>{}, Int<BK>{}));

    float* dOut;
    CUDA_CHECK(cudaMalloc(&dOut, NUM_CTAS * NUM_TILES * sizeof(float)));
    CUDA_CHECK(cudaMemset(dOut, 0, NUM_CTAS * NUM_TILES * sizeof(float)));

    persistent_ex<<<NUM_CTAS, 128>>>(ta, dOut, NUM_TILES, sla3);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* hOut = new float[NUM_CTAS * NUM_TILES];
    CUDA_CHECK(cudaMemcpy(hOut, dOut, NUM_CTAS * NUM_TILES * sizeof(float), cudaMemcpyDeviceToHost));

    // 每个 (CTA, tile) 都应该验证过 (out=1), 没被吃到的 tile 是 0 (正常)
    int bad = 0;
    for (int c = 0; c < NUM_CTAS; ++c)
        for (int t = c; t < NUM_TILES; t += NUM_CTAS)
            if (hOut[c * NUM_TILES + t] != 1.0f) ++bad;
    printf("  验证失败的元素个数 = %d (应为 0)\n", bad);
    expect("persistent 双 tile 无死锁且数据正确", bad == 0);
}

// ===========================================================================
int main() {
    printf("cute_07 练习 —— 填 TODO 后 make run\n");

    ex1();
    ex2();
    ex3();

    printf("\n===== 结果: %d 通过, %d 失败 =====\n", g_pass, g_fail);
    if (g_fail == 0) {
        printf("全部通过!\n");
        return 0;
    }
    printf("还有 %d 题没填或填错。\n", g_fail);
    return 1;
}
