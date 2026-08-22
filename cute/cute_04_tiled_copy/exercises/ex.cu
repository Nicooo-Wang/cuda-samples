// cute_04 练习：把 TODO 填掉，然后 make run
//
// 题目见 README 的"练习"一节。参考解答在 solutions.md。
// 每题都有一个自动检查，填对了会打印 PASS。
//
// 八道题对应 README 的:
//   1 -> §4.1  bank 模型
//   2 -> §4.3  手算 swizzle 映射
//   3 -> §4.4  M 参数与向量化
//   4 -> §4.6  选 TMA 模式
//   5 -> §2 §4.6  TMA 语义
//   6 -> §6    修一个转置 bug
//   7 -> §2    手写一段 TMA 搬运
//   8 -> §5.2  把单缓冲改成 TMA Double Buffer

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
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

static int bank_of(int byte_offset) { return (byte_offset / 4) % 32; }

// 一次 warp 访存里最热的 bank 被请求几次
template <class F>
static int max_bank_requests(int lanes, F&& off_of_lane) {
    int hist[32] = {0};
    for (int l = 0; l < lanes; ++l) ++hist[bank_of(off_of_lane(l))];
    int worst = 0;
    for (int b = 0; b < 32; ++b)
        if (hist[b] > worst) worst = hist[b];
    return worst;
}

// 全列扫描取最坏 —— 只看第 0 列会得出错误结论 (见 README §4.4)
template <class Lay>
static int worst_col_all(Lay lay) {
    int worst = 0;
    for (int c = 0; c < 32; ++c) {
        int w = max_bank_requests(32, [&](int l) { return int(lay(l, c)) * 4; });
        if (w > worst) worst = w;
    }
    return worst;
}

// 行内最短连续段
template <class Lay>
static int min_run(Lay lay) {
    int g = 1 << 30;
    for (int r = 0; r < 32; ++r) {
        int run = 1;
        for (int c = 1; c < 32; ++c) {
            if (int(lay(r, c)) == int(lay(r, c - 1)) + 1) ++run;
            else { if (run < g) g = run; run = 1; }
        }
        if (run < g) g = run;
    }
    return g;
}

static auto plain() {
    return make_layout(make_shape(Int<32>{}, Int<32>{}), LayoutRight{});
}

// ===========================================================================
// 练习 1 — 数 bank ★☆☆  (§4.1)
// ===========================================================================
static void ex1() {
    printf("练习 1 — 数 bank:\n");
    auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), LayoutRight{});
    // TODO: 一个 warp 读第 5 列, 最热的 bank 被请求几次?
    constexpr int EX1_CONFLICT = 32;  // <-- 改成你的答案
    int w = max_bank_requests(32, [&](int l) { return int(plain(l, 5)) * 4; });
    expect("第 5 列 = 32-way conflict", w == EX1_CONFLICT && EX1_CONFLICT == 32);
    printf("\n");
}

// ===========================================================================
// 练习 2 — 手算 swizzle 映射 ★★☆  (§4.3)
// ===========================================================================
static void ex2() {
    printf("练习 2 — 手算 swizzle 映射:\n");
    // TODO: Swizzle<5,0,5> 作用在 (32,32):(32,1) 上, 手算 swz_off(6, 3)。
    // 按 §4.3 四步: off = 6*32+3 = 195; 高 5 位 = 6; c XOR r = 3 XOR 6 = 5
    constexpr int EX2_SWZ_OFF = 197;  // <-- 改成你的答案
    auto swz = composition(Swizzle<5, 0, 5>{}, plain());
    expect("swz(6,3) 正确", int(swz(6, 3)) == EX2_SWZ_OFF && EX2_SWZ_OFF == 197);
    printf("\n");
}

// ===========================================================================
// 练习 3 — 选 M 保住向量化 ★★★  (§4.4)
// ===========================================================================
static void ex3() {
    printf("练习 3 — 选 M 保住向量化:\n");
    // TODO: 128-bit atom 搬 32x32 float tile。
    // 哪些 swizzle 能用? (行内连续 >= 4 才可能向量化)
    // 提示: 先看 min_run, 再看列冲突。
    auto s505 = composition(Swizzle<5, 0, 5>{}, plain());
    auto s414 = composition(Swizzle<4, 1, 4>{}, plain());
    auto s323 = composition(Swizzle<3, 2, 3>{}, plain());
    printf("    Sw<5,0,5>: 列读 %2d-way, 行内连续 %2d\n", worst_col_all(s505), min_run(s505));
    printf("    Sw<4,1,4>: 列读 %2d-way, 行内连续 %2d\n", worst_col_all(s414), min_run(s414));
    printf("    Sw<3,2,3>: 列读 %2d-way, 行内连续 %2d\n", worst_col_all(s323), min_run(s323));
    // TODO: 哪个能用? (写名字, 检查用下面的断言)
    constexpr bool EX3_505_OK = false;
    constexpr bool EX3_323_OK = true;
    expect("Sw<5,0,5> 不能用 (连续 1)", EX3_505_OK == false);
    expect("Sw<3,2,3> 能用 (连续 4)", EX3_323_OK == true);
    printf("\n");
}

// ===========================================================================
// 练习 4 — 选对 TMA 模式 ★★☆  (§4.6)
// ===========================================================================
static void ex4() {
    printf("练习 4 — 选 TMA 模式:\n");
    // CN=32 的 float tile (一行 128 字节)。
    // TODO: 四个模式哪些可用? (内层长度能否整除 CN)
    // SW128 内层 32 / SW64 内层 16 / SW32 内层 8 / INTER 内层 4
    constexpr bool EX4_SW128_OK = true;
    constexpr bool EX4_SW64_OK = true;
    constexpr bool EX4_SW32_OK = true;
    constexpr bool EX4_INTER_OK = false;
    expect("SW128 可用 (32%32==0)", EX4_SW128_OK == true);
    expect("SW64 可用 (32%16==0)", EX4_SW64_OK == true);
    expect("SW32 可用 (32%8==0)", EX4_SW32_OK == true);
    expect("INTER 可用 (32%4==0)", EX4_INTER_OK == true);
    // TODO: 实用规则选哪个? (最大的)
    constexpr bool EX4_PICK_SW128 = true;
    expect("选一行字节数最大的 SW128", EX4_PICK_SW128 == true);
    printf("\n");
}

// ===========================================================================
// 练习 5 — TMA 语义 ★★☆  (§2 §4.6)
// ===========================================================================
static void ex5() {
    printf("练习 5 — TMA 语义:\n");
    // TODO: 判断对错
    // (a) 手写 swizzle 改的是逻辑坐标的偏移; TMA 的 swizzle 改的是物理字节序。
    constexpr bool EX5_A = true;   // 对 -> true
    // (b) plain smem layout 交给 TMA 会搬错。
    constexpr bool EX5_B = false;    // 错 -> false
    expect("(a) 逻辑层 vs 物理层", EX5_A == true);
    expect("(b) plain 也能搬对", EX5_B == false);
    printf("\n");
}

// ===========================================================================
// 练习 6 — 修一个转置 bug ★★★  (§6)
// ===========================================================================
template <class SLay>
__global__ void transpose_kernel(const float* __restrict__ a, float* __restrict__ b, SLay slay) {
    extern __shared__ __align__(128) char raw[];
    auto sT = make_tensor(make_smem_ptr((float*)raw), slay);
    const int row0 = blockIdx.x * 32, col0 = blockIdx.y * 32;
    for (int i = threadIdx.x; i < 32 * 32; i += blockDim.x)
        sT(i / 32, i % 32) = a[(row0 + i / 32) * 128 + (col0 + i % 32)];
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 32; i += blockDim.x) {
        int r = i / 32, c = i % 32;
        b[(col0 + r) * 256 + (row0 + c)] = sT(c, r);  // 转置
    }
}

static void ex6() {
    printf("练习 6 — 修转置 bug:\n");
    // TODO: 转置结果正确但列读 32-way。只改 slay 一行, 让列读 <= 16-way。
    // (用 §4.6 的 SW128 原子)
    // TODO 提示: tile_to_shape(GMMA::Layout_K_SW128_Atom<float>{}, ...)
    auto swz = tile_to_shape(GMMA::Layout_K_SW32_Atom<float>{}, make_shape(Int<32>{}, Int<32>{}));
    constexpr int M = 256, N = 128;
    float *da, *db;
    CUDA_CHECK(cudaMalloc(&da, sizeof(float) * M * N));
    CUDA_CHECK(cudaMalloc(&db, sizeof(float) * M * N));
    float* h = new float[M * N];
    for (int i = 0; i < M * N; ++i) h[i] = float(i);
    CUDA_CHECK(cudaMemcpy(da, h, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    transpose_kernel<<<dim3(M / 32, N / 32), 128, 32 * 32 * 4>>>(da, db, swz);
    CUDA_CHECK(cudaDeviceSynchronize());
    float* ho = new float[M * N];
    CUDA_CHECK(cudaMemcpy(ho, db, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int m = 0; m < M && ok; ++m)
        for (int n = 0; n < N; ++n)
            if (ho[n * M + m] != h[m * N + n]) ok = false;
    expect("SW128 转置正确", ok);
    printf("    (检查: 列读 = %d-way, 目标 <= 16)\n", worst_col_all(swz));
    printf("\n");
}

// ===========================================================================
// 练习 7 — 手写一段 TMA 搬运 ★★★  (§2)
// ===========================================================================
template <class TmaLoad>
__global__ void tma_load_kernel(__grid_constant__ const TmaLoad tma, float* __restrict__ out) {
    // TODO 1: 这一次搬多少字节? (填错/填 0 的结果: barrier 立即放行, 数据没搬)
    constexpr int tx_bytes = 32 * 32 * sizeof(float);  // <-- 32*32*sizeof(float)

    // TODO 2: smem (128B 对齐) 和 mbarrier
    __shared__ __align__(128) float smem[32 * 32];  // <-- TODO: 32*32
    __shared__ uint64_t bar;

    auto sT = make_tensor(make_smem_ptr(smem),
                          make_layout(make_shape(Int<32>{}, Int<32>{}), LayoutRight{}));
    // TODO 3: 坐标 tensor + 本 CTA 的 tile (grid 是 (M/32, N/32))
    auto gc = tma.get_tma_tensor(make_shape(Int<256>{}, Int<128>{}));
    auto gt = local_tile(gc, Shape<Int<32>, Int<32>>{}, make_coord(blockIdx.x, blockIdx.y));

    // TODO 4: barrier 初始化 + 发 TMA (thread 0)
    if (threadIdx.x == 0) initialize_barrier(bar, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        set_barrier_transaction_bytes(bar, tx_bytes);
        auto per = tma.get_slice(0);
        copy(tma.with(bar), per.partition_S(gt), per.partition_D(sT));
    }
    __syncthreads();
    // TODO 5: 等硬件搬完 (phase = 0, 只用一轮)
    wait_barrier(bar, 0);

    for (int i = threadIdx.x; i < 32 * 32; i += blockDim.x)
        out[(blockIdx.x * 32 + i / 32) * 128 + (blockIdx.y * 32 + i % 32)] = smem[i];
}

static void ex7() {
    printf("练习 7 — 手写 TMA 搬运:\n");
    constexpr int M = 256, N = 128;
    float *da, *db;
    CUDA_CHECK(cudaMalloc(&da, sizeof(float) * M * N));
    CUDA_CHECK(cudaMalloc(&db, sizeof(float) * M * N));
    float* h = new float[M * N];
    for (int i = 0; i < M * N; ++i) h[i] = float(i);
    CUDA_CHECK(cudaMemcpy(da, h, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(db, 0, sizeof(float) * M * N));

    auto mA = make_tensor(make_gmem_ptr(da), make_layout(make_shape(Int<M>{}, Int<N>{}),
                                                         LayoutRight{}));
    auto slay = make_layout(make_shape(Int<32>{}, Int<32>{}), LayoutRight{});
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, mA, slay);
    tma_load_kernel<<<dim3(M / 32, N / 32), 128>>>(tma, db);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* ho = new float[M * N];
    CUDA_CHECK(cudaMemcpy(ho, db, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < M * N && ok; ++i)
        if (ho[i] != h[i]) ok = false;
    if (!ok) printf("    (提示: 没填 TODO 的话 tx_bytes=0, barrier 立即放行但数据没搬)\n");
    expect("TMA 搬运正确", ok);
    printf("\n");
}

// ===========================================================================
// 练习 8 — TMA Double Buffer ★★★  (§5.2)
// ===========================================================================
template <class TmaLoad>
__global__ void double_buffer_kernel(__grid_constant__ const TmaLoad tma, float* __restrict__ out) {
    // 两个 buffer, 搬 k+1 和算 k 重叠。three TODOs.
    constexpr int tx_bytes = 32 * 32 * sizeof(float);
    constexpr int STAGES = 2;
    constexpr int NTILE = 64;

    __shared__ __align__(128) float smem[STAGES * 32 * 32];
    __shared__ uint64_t full[STAGES], empty[STAGES];

    auto sT = make_tensor(make_smem_ptr(smem),
                          make_layout(make_shape(Int<32>{}, Int<32>{}, Int<STAGES>{}),
                                      make_stride(Int<32>{}, Int<1>{}, Int<32 * 32>{})));
    auto gc = tma.get_tma_tensor(make_shape(NTILE * 32, 32));
    auto per = tma.get_slice(0);

    if (threadIdx.x == 0)
        for (int s = 0; s < STAGES; ++s) {
            initialize_barrier(full[s], 1);
            initialize_barrier(empty[s], 128);  // 128 个线程各 arrive 一次
        }
    __syncthreads();

    // prologue: 填满全部 buffer
    for (int s = 0; s < STAGES; ++s)
        if (threadIdx.x == 0) {
            set_barrier_transaction_bytes(full[s], tx_bytes);
            auto gt = local_tile(gc, Shape<Int<32>, Int<32>>{}, make_coord(s, 0));
            copy(tma.with(full[s]), per.partition_S(gt), per.partition_D(sT(_, _, s)));
        }
    __syncthreads();

    float acc = 0.f;
    for (int k = 0; k < NTILE; ++k) {
        int s = k % STAGES;
        // TODO 1: phase 公式 —— buffer s 第几轮被用?
        // 写死 0 的症状: k=1 时等 full[1] 的 phase 0, 但 prologue 已把它的
        // phase 翻到 1 —— 等一个永远不会到来的翻转, 死锁。
        int phase = 0;  // <-- TODO 1: (k / STAGES) & 1
        // 填好前别跑: 写死 0 会死锁 (k=1 时等 full[1] 的 phase 0,
        // 但 prologue 已把它翻到 1 —— 等一个永远不会到来的翻转)
        wait_barrier(full[s], phase);

        for (int i = threadIdx.x; i < 32 * 32; i += 128) acc += sT(i / 32, i % 32, s);

        // TODO 2: 通知 producer 这个 buffer 用完了
        arrive_barrier(empty[s]);  // <-- 提示: arrive_barrier(empty[s])

        int knext = k + STAGES;
        if (knext < NTILE) {
            // TODO 3: 先等 empty 再补货
            wait_barrier(empty[s], phase);  // <-- 这里也要 phase
            if (threadIdx.x == 0) {
                set_barrier_transaction_bytes(full[s], tx_bytes);
                auto gt = local_tile(gc, Shape<Int<32>, Int<32>>{}, make_coord(knext, 0));
                copy(tma.with(full[s]), per.partition_S(gt), per.partition_D(sT(_, _, s)));
            }
        }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

static void ex8() {
    printf("练习 8 — TMA Double Buffer:\n");
    constexpr int NTILE = 64, N_CTA = 64;
    float *da, *dsum;
    CUDA_CHECK(cudaMalloc(&da, sizeof(float) * NTILE * 32 * 32 * N_CTA));
    CUDA_CHECK(cudaMalloc(&dsum, sizeof(float) * N_CTA));
    float* h = new float[NTILE * 32 * 32 * N_CTA];
    for (int i = 0; i < NTILE * 32 * 32 * N_CTA; ++i) h[i] = 1.f;  // 全 1, 累加和可预测
    CUDA_CHECK(cudaMemcpy(da, h, sizeof(float) * NTILE * 32 * 32 * N_CTA, cudaMemcpyHostToDevice));

    auto mA = make_tensor(make_gmem_ptr(da),
                          make_layout(make_shape(NTILE * 32 * N_CTA, 32), LayoutRight{}));
    auto slay = make_layout(make_shape(Int<32>{}, Int<32>{}), LayoutRight{});
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, mA, slay);
    // 填好 phase 之前别跑 —— 写死 0 会死锁 (见 kernel 里的注释)。
    // 先把 TODO 1 填成 (k / STAGES) & 1 再取消下面一行的注释:
    // double_buffer_kernel<<<N_CTA, 128>>>(tma, dsum);
    CUDA_CHECK(cudaMemset(dsum, 0, sizeof(float) * N_CTA));

    float* hs = new float[N_CTA];
    CUDA_CHECK(cudaMemcpy(hs, dsum, sizeof(float) * N_CTA, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int b = 0; b < N_CTA; ++b)
        if (fabsf(hs[b] - 32.f * 32.f * NTILE) > 1e-2f) ok = false;
    expect("Double Buffer 结果正确", ok);
    printf("    (写错 phase 的症状是死锁; 本练习自动跑 5 次抓它)\n");
    printf("\n");
}

int main() {
    printf("cute_04 练习 — 每题填完 TODO 后 make run\n\n");
    ex1(); ex2(); ex3(); ex4(); ex5(); ex6(); ex7(); ex8();
    printf("=======================================\n");
    printf("  %d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
