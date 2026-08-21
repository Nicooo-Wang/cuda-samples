// cute_04 v4 —— Multi-stage: 让搬运和计算重叠
//
// 对应 README §5。
//
// v1-v3 把一块 tile 的搬运完全搞定了: TMA 搬进去、swizzle 摆好、搬回来。
// 但整个流程是**串行**的: 搬的时候计算单元闲着, 算的时候搬运引擎闲着。
// 这一版用**多个 smem buffer** 让两者重叠起来。
//
//   §5.1  单缓冲: 搬和算串行                    <- 基准
//   §5.2  Double Buffer: 2 个 buffer 轮换        <- full/empty 两组 barrier
//   §5.3  Super Buffer: N 个 buffer, 48KB 台阶   <- 以及为什么必须动态 smem
//   §5.4  tma_partition: 另一种 TMA 写法          <- cute_05/06 用的是这个
//
// ---------------------------------------------------------------------------
// 负载是什么
//
// 为了把"重叠"看清楚, 需要一段真实但简单的计算。这里做:
//
//     A 是 GM x GN 的 float 矩阵 (row-major), 按列切成 NTILE 块,
//     每块 CM x CN, 依次搬进 smem, 每个元素做一段 FMA 运算 (代表计算),
//     累加结果。
//
// 注意: 这一版**不需要 WGMMA** (那是 cute_05 的事)。计算就用普通 FMA,
// 只要把"读 smem + 算"这一步做成比搬运慢, 重叠的收益就显形了。
//
// ---------------------------------------------------------------------------
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_v4

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 尺寸
// ---------------------------------------------------------------------------
constexpr int CM = 32, CN = 64;  // 一个 tile: 32x64 float = 8KB
constexpr int NTHR = 128;        // 每 CTA 线程数
constexpr int NTILE = 128;       // 每个 CTA 扫多少个 tile
constexpr int ITER = 8;          // 每元素 FMA 次数 -> 制造计算量

// 全 grid 的规模 (HBM 驻留, 才看得出重叠):
//   528 CTA * 128 tile * 8KB = 553 MB
constexpr int N_CTA = 528;

// 一个 tile 有多少元素 (供 kernel 内的循环用)
constexpr int TILE_ELEMS = CM * CN;

// ---------------------------------------------------------------------------
// 缓冲区: 一个很大的 A + 每 CTA 一个累加结果
// ---------------------------------------------------------------------------
struct Buffers {
    static constexpr size_t a_elems = size_t(N_CTA) * NTILE * TILE_ELEMS;
    static constexpr size_t a_bytes = a_elems * sizeof(float);

    float* d_a;
    float* d_sum;
    float* h_a;
    float* h_sum;

    Buffers() {
        CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
        CUDA_CHECK(cudaMalloc(&d_sum, N_CTA * sizeof(float)));
        h_a = new float[a_elems];
        h_sum = new float[N_CTA];
        for (size_t i = 0; i < a_elems; ++i) h_a[i] = float(i % 1024);
        CUDA_CHECK(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_sum, 0, N_CTA * sizeof(float)));
    }

    ~Buffers() {
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_sum));
        delete[] h_a;
        delete[] h_sum;
    }
};

// "计算": 每个元素做 ITER 次 FMA。值是确定的, 便于 CPU 对照。
__device__ __forceinline__ float work(float x) {
#pragma unroll
    for (int i = 0; i < ITER; ++i) x = fmaf(x, 1.0000001f, 1e-6f);
    return x;
}

// ===========================================================================
// §5.1  单缓冲: 搬和算串行
//
// 只有一个 smem buffer, 所以每一轮必须:
//   1. 等 TMA 搬完 (wait_barrier full)
//   2. 全 CTA 读 smem 计算
//   3. 等大家读完 (__syncthreads), 才能覆盖这块 smem
// 时间线: [搬0][算0][搬1][算1]... —— 两个引擎轮流干活, 各闲一半。
// ===========================================================================
template <class TmaLoad>
__global__ static void single_kernel(__grid_constant__ const TmaLoad tma, float* __restrict__ sum) {
    constexpr int tx_bytes = TILE_ELEMS * sizeof(float);

    __shared__ __align__(128) float smem[TILE_ELEMS];
    __shared__ uint64_t full;  // 生产者 -> 消费者: "buffer 已装满"

    auto sT = make_tensor(make_smem_ptr(smem),
                          make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{}));
    auto gcoord = tma.get_tma_tensor(make_shape(N_CTA * NTILE * CM, CN));
    auto per = tma.get_slice(0);

    int base = blockIdx.x * NTILE;  // 本 CTA 负责的 tile 起点

    if (threadIdx.x == 0) initialize_barrier(full, 1);
    __syncthreads();

    float acc = 0.f;
    for (int k = 0; k < NTILE; ++k) {
        // ---- 生产: 一个线程发 TMA ----
        if (threadIdx.x == 0) {
            set_barrier_transaction_bytes(full, tx_bytes);
            auto gt = local_tile(gcoord, Shape<Int<CM>, Int<CN>>{}, make_coord(base + k, 0));
            copy(tma.with(full), per.partition_S(gt), per.partition_D(sT));
        }
        __syncthreads();          // 1) 等 barrier 初始化/上一轮读完 (见下)
        wait_barrier(full, k & 1);  // 2) 等硬件搬完

        // ---- 消费: 全 CTA 一起算 ----
        for (int i = threadIdx.x; i < TILE_ELEMS; i += NTHR) acc += work(sT(i / CN, i % CN));

        __syncthreads();  // 3) 等全 CTA 读完, 下一轮才能覆盖
    }
    if (threadIdx.x == 0) sum[blockIdx.x] = acc;
}

// ===========================================================================
// §5.2  Double Buffer: 2 个 buffer 轮换
//
// 有了两个 buffer, 搬 k+1 可以和算 k 同时进行:
//   TMA   : [搬0][搬1][搬2][搬3]
//   计算   :      [算0][算1][算2]
// 需要**两组 barrier**:
//   full[s]   生产者->消费者: "buffer s 已装满"   按字节数等 (TMA 专用)
//   empty[s]  消费者->生产者: "buffer s 已用完"   按到达数等 (NTHR 个线程)
//
// 用 mbarrier 而不是 __syncthreads: __syncthreads 是全 block 栅栏, 强制所有人
// 停在同一行; 这里要的是"针对某个 buffer 的细粒度等待", 两组工作各走各的。
// ===========================================================================
template <int STAGES, class TmaLoad>
__global__ static void pipe_kernel(__grid_constant__ const TmaLoad tma, float* __restrict__ sum) {
    constexpr int tx_bytes = TILE_ELEMS * sizeof(float);

    __shared__ __align__(128) float smem[STAGES * TILE_ELEMS];
    __shared__ uint64_t full[STAGES], empty[STAGES];

    // (CM, CN, STAGES) 的三维 smem: 第 3 维是 buffer 号
    auto sT = make_tensor(make_smem_ptr(smem),
                          make_layout(make_shape(Int<CM>{}, Int<CN>{}, Int<STAGES>{}),
                                      make_stride(Int<CN>{}, Int<1>{}, Int<TILE_ELEMS>{})));
    auto gcoord = tma.get_tma_tensor(make_shape(N_CTA * NTILE * CM, CN));
    auto per = tma.get_slice(0);
    int base = blockIdx.x * NTILE;

    if (threadIdx.x == 0)
        for (int s = 0; s < STAGES; ++s) {
            initialize_barrier(full[s], 1);
            initialize_barrier(empty[s], NTHR);  // 消费者 NTHR 个线程都要 arrive
        }
    __syncthreads();

    // ---- prologue: 先把 STAGES 个 buffer 都填满 ----
    // tile k 永远落在 buffer k % STAGES, 所以 prologue 填 tile 0..STAGES-1
    for (int s = 0; s < STAGES && s < NTILE; ++s) {
        if (threadIdx.x == 0) {
            set_barrier_transaction_bytes(full[s], tx_bytes);
            auto gt = local_tile(gcoord, Shape<Int<CM>, Int<CN>>{}, make_coord(base + s, 0));
            copy(tma.with(full[s]), per.partition_S(gt), per.partition_D(sT(_, _, s)));
        }
    }
    __syncthreads();

    float acc = 0.f;
    for (int k = 0; k < NTILE; ++k) {
        int s = k % STAGES;
        int phase = (k / STAGES) & 1;  // buffer s 第几轮被用

        // 等 producer: buffer s 装满
        wait_barrier(full[s], phase);

        // 消费: 全 CTA 一起算
        for (int i = threadIdx.x; i < TILE_ELEMS; i += NTHR) acc += work(sT(i / CN, i % CN, s));

        // 通知 producer: buffer s 用完了, 可以覆盖
        arrive_barrier(empty[s]);

        // 为下一轮补货: 把 tile k+STAGES 搬进刚空出来的 buffer s
        int knext = k + STAGES;
        if (knext < NTILE) {
            // 注意: 先等 empty —— 否则可能覆盖还没读完的数据
            wait_barrier(empty[s], phase);
            if (threadIdx.x == 0) {
                set_barrier_transaction_bytes(full[s], tx_bytes);
                auto gt = local_tile(gcoord, Shape<Int<CM>, Int<CN>>{}, make_coord(base + knext, 0));
                copy(tma.with(full[s]), per.partition_S(gt), per.partition_D(sT(_, _, s)));
            }
        }
    }
    if (threadIdx.x == 0) sum[blockIdx.x] = acc;
}

// ===========================================================================
// §5.3  Super Buffer: 更多 stage, 以及 48KB 那道台阶
//
// stage 越多, TMA 能提前搬得越远, 容忍的延迟越长。但:
//   - smem 线性增长, 收益递减 (看实测表)
//   - 静态 __shared__ 上限 48KB: 3 个 8KB 的 buffer + 两个 barrier 数组
//     也够用, 但 4 个就爆了。吃满 H200 的 227KB 必须动态 smem。
//
// 用模板参数 STAGES 统一跑 1/2/3/4, 数字直接比较。
// ===========================================================================
template <int STAGES, class TmaLoad>
static void run_pipe(const char* tag, TmaLoad tma, Buffers& buf, float single_ms) {
    float ms = time_kernel([&] { pipe_kernel<STAGES><<<N_CTA, NTHR>>>(tma, buf.d_sum); });

    printf("  %-22s smem %2d KB    %8.3f ms   %.2fx\n", tag,
           STAGES * TILE_ELEMS * 4 / 1024, ms, single_ms / ms);
}

static void section51_single() {
    print_separator("§5.1  单缓冲: 搬和算串行");

    Buffers buf;
    auto mA = make_tensor(make_gmem_ptr(buf.d_a),
                          make_layout(make_shape(N_CTA * NTILE * CM, CN), LayoutRight{}));
    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, mA, slay);

    float ms = time_kernel([&] { single_kernel<<<N_CTA, NTHR>>>(tma, buf.d_sum); });

    printf("  数据规模: %d CTA x %d tile x %dx%d float = %.0f MB (HBM 驻留, L2 装不下)\n",
           N_CTA, NTILE, CM, CN, double(Buffers::a_bytes) / 1e6);
    printf("  每 tile: TMA 搬 8KB, 全 CTA 做 %d 次 FMA/元素\n\n", ITER);
    printf("  时间线 (一个 k-tile 一个周期):\n");
    printf("    TMA    : [搬0]      [搬1]      [搬2]\n");
    printf("    计算   :      [算0]      [算1]      [算2]\n");
    printf("    两个引擎轮流干活, 各闲一半。\n\n");
    printf("  单缓冲   %.3f ms\n", ms);

    // 给后面的版本用
    extern float g_single_ms;
    g_single_ms = ms;
}
float g_single_ms = 0.f;

static void section52_double() {
    print_separator("§5.2  Double Buffer: 2 个 buffer 轮换");

    Buffers buf;
    auto mA = make_tensor(make_gmem_ptr(buf.d_a),
                          make_layout(make_shape(N_CTA * NTILE * CM, CN), LayoutRight{}));
    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, mA, slay);

    printf("  搬 k+1 和算 k 重叠:\n");
    printf("    TMA    : [搬0][搬1][搬2][搬3]\n");
    printf("    计算   :      [算0][算1][算2]\n\n");
    printf("  full[s]: 生产者->消费者, 按字节数等 (TMA 专用)\n");
    printf("  empty[s]: 消费者->生产者, 按到达数等 (NTHR 个线程各 arrive 一次)\n\n");

    run_pipe<1>("单缓冲", tma, buf, g_single_ms);
    run_pipe<2>("Double Buffer", tma, buf, g_single_ms);

    printf("\n  prologue 填满 STAGES 个 buffer 后, 注意 empty 的 phase:\n");
    printf("  write_state 不能预推进 (cute_06 用 PipelineState 时踩过的坑),\n");
    printf("  这里手写 barrier 同样成立: full/empty 都从 phase 0 开始。\n");
}

static void section53_super() {
    print_separator("§5.3  Super Buffer: 更多 stage");

    Buffers buf;
    auto mA = make_tensor(make_gmem_ptr(buf.d_a),
                          make_layout(make_shape(N_CTA * NTILE * CM, CN), LayoutRight{}));
    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, mA, slay);

    run_pipe<3>("Super Buffer (3)", tma, buf, g_single_ms);
    run_pipe<4>("Super Buffer (4)", tma, buf, g_single_ms);

    printf("\n  注意 smem 是线性增长的, 收益却递减 —— stage 不是越多越好。\n");
    printf("  再往上还有一道**硬台阶**: 静态 __shared__ 上限 48KB (0xc000)。\n");
    printf("  本例 3 个 8KB buffer 勉强在限内, 4 个 stage 配大 tile 直接 ptxas 报:\n\n");
    printf("    ptxas error: uses too much shared data (0xc000 max)\n\n");
    printf("  要吃到 H200 的 227KB 必须换**动态 smem**:\n\n");
    printf("    extern __shared__ char smem[];   // kernel 里\n");
    printf("    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);\n");
    printf("    k<<<grid, block, bytes>>>();     // launch 时申请\n\n");
    printf("  这就是官方例子 (wgmma_tma_sm90.cu) 用 SharedStorage + 动态 smem 的原因\n");
    printf("  —— SM90 GEMM 的 smem 动辄 100KB 以上, 绕不开这道台阶。\n");
}

// ===========================================================================
// §5.4  tma_partition: 另一种 TMA 写法 (cute_05/06 用的是这个)
//
// v1-v4 用的都是 make_tma_copy + get_slice(0) + partition_S/D —— 把 TMA 当成
// "只有 1 个线程的 TiledCopy"。这非常直观, 但 CUTLASS 的 GEMM 代码 (cute_05
// capstone, cute_06 v3+) 用的是另一套: make_tma_atom + tma_partition。
//
// 两者的关系:
//   make_tma_copy  =  TiledCopy 包装: 处理"1 个线程"的切片, 适合手写小 kernel
//   make_tma_atom  =  裸原子: 不包装, 你要自己告诉它"从哪个 slice 发",
//                     好处是能表达多 CTA 协作 (multicast, cute_06 用)
//
// 两个必须知道的差异:
//   1) smem layout 必须带 PIPE 维 (CM, CN, 1), 建 atom 时传切片 (_,_,0)
//   2) partition 用 tma_partition, 得到 gmem 侧 (TMA,) 和 smem 侧 (TMA, PIPE)
//
// 写法虽然不同, 搬的仍是同一件事: 一个线程发一条指令, 硬件搬, barrier 等。
// ===========================================================================
template <class TmaAtom, class SLay>
__global__ static void tma_partition_kernel(__grid_constant__ const TmaAtom tma, SLay slay,
                                            float* __restrict__ out) {
    __shared__ __align__(128) float smem[cosize_v<SLay>];  // (CM, CN, PIPE)
    __shared__ __align__(8) uint64_t bar[1];

    auto sT = make_tensor(make_smem_ptr(smem), slay);
    auto gcoord = tma.get_tma_tensor(make_shape(N_CTA * NTILE * CM, CN));
    auto gt = local_tile(gcoord, Shape<Int<CM>, Int<CN>>{}, make_coord(blockIdx.x, 0));

    // tma_partition 把 (CM,CN,PIPE) 的 smem 和坐标的 gmem 切成两个视图:
    //   tAg = (TMA,)       gmem 侧 —— mode-0 整块交给 TMA
    //   tAs = (TMA, PIPE)  smem 侧
    auto p = tma_partition(tma, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sT),
                           group_modes<0, 2>(gt));
    auto tAg = get<0>(p);
    auto tAs = get<1>(p);

    // 这一次搬多少字节 (mbarrier 要按字节数等)
    constexpr int tx_bytes = sizeof(make_tensor_like(tensor<0>(tAs)));

    // TMA atom 不带"线程映射", 要从 warp 里自己选一个 lane 发
    int one = cute::elect_one_sync();
    int warp = cutlass::canonical_warp_idx_sync();
    using Bar = cutlass::arch::ClusterTransactionBarrier;

    if (warp == 0 && one) Bar::init(&bar[0], 1);
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    if (warp == 0 && one) {
        Bar::arrive_and_expect_tx(&bar[0], tx_bytes);
        copy(tma.with(bar[0]), tAg, tAs(_, Int<0>{}));
    }
    Bar::wait(&bar[0], 0);

    // 倒回 gmem 验证
    auto s2 = sT(_, _, Int<0>{});
    auto mOut = make_tensor(make_gmem_ptr(out),
                            make_layout(make_shape(N_CTA * NTILE * CM, CN), LayoutRight{}));
    auto gOut = local_tile(mOut, Shape<Int<CM>, Int<CN>>{}, make_coord(blockIdx.x, 0));
    for (int i = threadIdx.x; i < TILE_ELEMS; i += NTHR) gOut(i / CN, i % CN) = s2(i / CN, i % CN);
}

static void section54_tma_partition() {
    print_separator("§5.4  tma_partition: cute_05/06 的写法");

    Buffers buf;
    auto mA = make_tensor(make_gmem_ptr(buf.d_a),
                          make_layout(make_shape(N_CTA * NTILE * CM, CN), LayoutRight{}));

    // atom 写法: 三维 smem layout, 建 atom 时传 PIPE=0 的切片
    auto slay3 = make_layout(make_shape(Int<CM>{}, Int<CN>{}, Int<1>{}),
                             make_stride(Int<CN>{}, Int<1>{}, Int<TILE_ELEMS>{}));
    auto tma = make_tma_atom(SM90_TMA_LOAD{}, mA, slay3(_, _, Int<0>{}),
                             make_shape(Int<CM>{}, Int<CN>{}));

    printf("  同一个搬运, 两种写法的并排:\n\n");
    printf("                    make_tma_copy (§2-§4)      make_tma_atom (本节)\n");
    printf("    --------------  ------------------------  ------------------------\n");
    printf("    host 构造       make_tma_copy(LOAD, g, sl)  make_tma_atom(LOAD, g, sl(_,_,0), tile)\n");
    printf("    smem layout     二维 (CM,CN)                三维 (CM,CN,PIPE)\n");
    printf("    kernel 切片     get_slice(0)               elect_one_sync 选 lane\n");
    printf("    partition       partition_S/D             tma_partition -> (TMA,) (TMA,PIPE)\n");
    printf("    barrier         initialize_barrier        ClusterTransactionBarrier\n");
    printf("    launch          普通 <<<>>>                普通 <<<>>> 即可\n");
    printf("    multicast       不支持                    支持 (cute_06)\n");

    // 验证搬得对: 每个 CTA 搬 tile blockIdx.x, 倒回 gmem 比对
    tma_partition_kernel<<<N_CTA, NTHR>>>(tma, slay3, buf.d_a);  // 就地搬回同一块
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(buf.h_a, buf.d_a, Buffers::a_bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < Buffers::a_elems; ++i)
        if (buf.h_a[i] != float(i % 1024)) ok = false;
    printf("\n  每个 CTA 用 tma_partition 搬自己的 tile 再原样写回: %s\n",
           ok ? "正确 (数据没被搬乱)" : "错误");

    printf("\n  为什么 CUTLASS 用这套不用那套:\n");
    printf("    make_tma_copy 把 TMA 当成\"1 线程的 TiledCopy\", 概念上最顺, 但\n");
    printf("    multicast (一个 CTA 搬给整簇) 需要把\"第几个 CTA\"也当成 slice 来\n");
    printf("    表达 —— TiledCopy 的线程映射不够用。tma_partition 把 slice 的选择\n");
    printf("    完全交给你, 于是 multicast 只是换一个 slice 号。\n");
    printf("    (cute_06 的 cluster capstone 会见到; 这里先把写法认熟。)\n");
}

int main() {
    printf("cute_04 v4 —— Multi-stage: 让搬运和计算重叠\n");
    printf("对应 README §5    需要 -arch=sm_90a\n");

    section51_single();
    section52_double();
    section53_super();
    section54_tma_partition();

    print_separator("小结");
    printf("  §5.1  单缓冲: 搬完才能算, 算完才能搬, 两个引擎各闲一半\n");
    printf("  §5.2  Double Buffer: 2 个 buffer + full/empty 两组 mbarrier -> 重叠\n");
    printf("  §5.3  Super Buffer: stage 越多容忍延迟越强, 但 smem 线性涨、收益递减\n");
    printf("        静态 smem 上限 48KB, 再大必须动态 smem + cudaFuncSetAttribute\n");
    printf("  §5.4  make_tma_atom + tma_partition 是 cute_05/06 的写法, 认熟它\n");

    printf("\n下一步 (§6): capstone 把本章所有工具 (TMA load/store + swizzle) 用到\n");
    printf("一个完整的转置上, 并且实测 swizzle 对 consumer 的意义。\n");
    printf("\nv4 OK\n");
    return 0;
}
