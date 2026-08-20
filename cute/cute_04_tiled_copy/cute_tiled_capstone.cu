// cute_04 capstone —— 矩阵转置: 本章所有工具的合练
//
// 对应 README §6。
//
// 转置是把本章讲过的所有东西串起来最干净的任务:
//
//   gmem 读  ->  TMA load  (一个线程描述整块, 硬件搬)
//   smem 摆  ->  swizzle   (按列读不撞 bank —— 转置必然要按列读)
//   gmem 写  ->  TMA store (不用 barrier, 用 fence)
//
// 六版, 每一版只换一件事:
//
//   t1  手写搬运 + plain  smem        <- v0 的写法, 32-way 冲突
//   t2  手写搬运 + SW128 smem         <- 手写搬运也能享受 swizzle
//   t3  TMA load  + plain  smem       <- v1 的写法
//   t4  TMA load  + SW128 smem        <- v3 的写法
//   t5  TMA load  + SW128 + TMA store <- 两端都是硬件 (本章终点)
//   t6  越界版: 不整除的矩阵          <- §3.3 的边界能力实战
//
// ---------------------------------------------------------------------------
// 任务定义
//
//   输入:  A 是 M x N 的 float, row-major (行 stride = N)
//   输出:  B 是 N x M 的 float, row-major, B(n,m) = A(m,n)
//
//   每个 CTA 负责 A 的一个 CM x CN 的 tile:
//     1) TMA load 搬进 smem
//     2) 转置着读出来 (读 smem 的"行" -> 写 gmem 的"列", 这就是按列读)
//     3) TMA store 搬回 gmem 的对应位置
//
//   A[m, n] = m*N + n        B 的正确值: B[n, m] = m*N + n
//
// ---------------------------------------------------------------------------
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_capstone

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 尺寸
// ---------------------------------------------------------------------------
constexpr int M = 1024, N = 1024;  // A 是 M x N float (4 MB)
constexpr int CM = 32, CN = 32;    // tile 32x32 (4 KB)
constexpr int NTHR = 128;

static dim3 grid() { return dim3(M / CM, N / CN); }

// ---------------------------------------------------------------------------
// 缓冲区: 一个 A, 一个 B
// ---------------------------------------------------------------------------
struct Buffers {
    static constexpr size_t elems = size_t(M) * N;
    static constexpr size_t bytes = elems * sizeof(float);

    float* d_a;
    float* d_b;
    float* h_a;
    float* h_b;

    Buffers() {
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        h_a = new float[elems];
        h_b = new float[elems];
        for (size_t i = 0; i < elems; ++i) h_a[i] = float(i);
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_b, 0, bytes));
    }

    ~Buffers() {
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        delete[] h_a;
        delete[] h_b;
    }

    // B[n,m] 应该等于 A[m,n]
    bool check() {
        CUDA_CHECK(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
                if (h_b[n * M + m] != h_a[m * N + n]) return false;
        return true;
    }
};

// 读 smem 时的列冲突 (每列扫描取最坏)
template <class Lay>
static int worst_col_conflict(Lay lay) {
    int worst = 0;
    for (int c = 0; c < CN; ++c) {
        int w = max_bank_requests(32, [&](int l) { return int(lay(l, c)) * 4; });
        if (w > worst) worst = w;
    }
    return worst;
}

// ===========================================================================
// 版本 A: 手写搬运 (v0 的写法), 转置在 smem 和 gmem 之间完成
//
// 读 smem 时 "sT(行 = lane, 列 = r)" —— 32 个 lane 读同一列 -> 撞 bank。
// ===========================================================================
template <class SLay>
__global__ static void transpose_hand_kernel(const float* __restrict__ a, float* __restrict__ b,
                                             SLay slay) {
    extern __shared__ __align__(128) char raw[];
    auto sT = make_tensor(make_smem_ptr(reinterpret_cast<float*>(raw)), slay);

    const int row0 = blockIdx.x * CM, col0 = blockIdx.y * CN;

    // 手写 load: 每个线程算自己的地址 (v0 §1.1)
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x)
        sT(i / CN, i % CN) = a[(row0 + i / CN) * N + (col0 + i % CN)];
    __syncthreads();

    // 转置着写 gmem: 读 sT 的第 r 列 (按列读!), 写 b 的连续位置 (合并写)
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) {
        int r = i / CN;  // 目标 b 的行 = 源 a 的列
        int c = i % CN;  // 目标 b 的列 = 源 a 的行
        b[(col0 + r) * M + (row0 + c)] = sT(c, r);
    }
}

// ===========================================================================
// 版本 B: TMA load + 手写 store (v1/v3 的写法)
// ===========================================================================
template <class SLay, class TmaLoad>
__global__ static void transpose_tma_kernel(__grid_constant__ const TmaLoad tma, SLay slay,
                                            float* __restrict__ b) {
    constexpr int tx_bytes = CM * CN * sizeof(float);
    extern __shared__ __align__(128) char raw[];
    __shared__ uint64_t bar;

    auto sT = make_tensor(make_smem_ptr(reinterpret_cast<float*>(raw)), slay);

    auto gc = tma.get_tma_tensor(make_shape(Int<M>{}, Int<N>{}));
    auto gt = local_tile(gc, Shape<Int<CM>, Int<CN>>{}, make_coord(blockIdx.x, blockIdx.y));

    if (threadIdx.x == 0) initialize_barrier(bar, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        set_barrier_transaction_bytes(bar, tx_bytes);
        auto per = tma.get_slice(0);
        copy(tma.with(bar), per.partition_S(gt), per.partition_D(sT));
    }
    __syncthreads();
    wait_barrier(bar, 0);

    const int row0 = blockIdx.x * CM, col0 = blockIdx.y * CN;
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) {
        int r = i / CN, c = i % CN;
        b[(col0 + r) * M + (row0 + c)] = sT(c, r);
    }
}

// ===========================================================================
// 版本 C: TMA load + TMA store (本章终点, §3.2 的完整版)
//
// 转置怎么做? 这里有两个选择, 本章用**手写转置 + TMA store**:
//
//   方案 A (本章): plain smem -> 全 CTA 手写转置 -> TMA store
//                 转置是"读写 smem"这个小循环, 其余都是硬件搬运
//   方案 B (不采用): 用转置视图 -> TMA store
//                 行不通: TMA store 按 smem **物理字节序**搬出, 视图只改
//                 逻辑坐标不改物理字节序 (§4.6 的教训) —— 实测搬出去是
//                 原样的行, 不是转置后的列
//
// 所以"交给硬件"的部分是搬运, 转置本身仍然是一段普通代码 —— 它本来
// 就是"读 A 的一列写 B 的一行"这个逻辑操作, 不是搬运。
// ===========================================================================
template <class SLay, class TmaLoad, class TmaStore>
__global__ static void transpose_both_kernel(__grid_constant__ const TmaLoad tma_load,
                                             __grid_constant__ const TmaStore tma_store,
                                             SLay slay) {
    constexpr int tx_bytes = CM * CN * sizeof(float);
    extern __shared__ __align__(128) char raw[];
    __shared__ uint64_t bar;

    auto sT = make_tensor(make_smem_ptr(reinterpret_cast<float*>(raw)), slay);
    auto blk = make_coord(blockIdx.x, blockIdx.y);

    if (threadIdx.x == 0) initialize_barrier(bar, 1);
    __syncthreads();

    // load: gmem -> smem (mbarrier 等字节)
    if (threadIdx.x == 0) {
        set_barrier_transaction_bytes(bar, tx_bytes);
        auto gc = tma_load.get_tma_tensor(make_shape(Int<M>{}, Int<N>{}));
        auto gt = local_tile(gc, Shape<Int<CM>, Int<CN>>{}, blk);
        auto per = tma_load.get_slice(0);
        copy(tma_load.with(bar), per.partition_S(gt), per.partition_D(sT));
    }
    __syncthreads();
    wait_barrier(bar, 0);

    // 转置: 先全 CTA 把源值读进临时数组, 再写目标位置。
    // 不能原地 sT(r,c)=sT(c,r): 目标 (r,c) 的源 (c,r) 可能刚被别的线程改过,
    // 所以读和写之间必须隔一道 __syncthreads。
    __shared__ float tmp[CM * CN];
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) {
        int r = i / CN, c = i % CN;
        tmp[i] = sT(c, r);  // 读
    }
    __syncthreads();
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) {
        int r = i / CN, c = i % CN;
        sT(r, c) = tmp[i];  // 写
    }
    __syncthreads();
    tma_store_fence();  // 保证上面的写对 TMA 可见 (事前挡)

    // store: smem -> gmem (fence, 不用 barrier)
    if (threadIdx.x == 0) {
        auto gc = tma_store.get_tma_tensor(make_shape(Int<N>{}, Int<M>{}));  // 注意形状反过来
        auto gt = local_tile(gc, Shape<Int<CN>, Int<CM>>{}, make_coord(blockIdx.y, blockIdx.x));
        //                                   ^^^^^^^^ tile 也跟着转置
        auto per = tma_store.get_slice(0);
        copy(tma_store, per.partition_S(sT), per.partition_D(gt));
        tma_store_arrive();
    }
    tma_store_wait<0>();
}

// ===========================================================================
// host 侧: 六个版本, 每版一个 host 函数
// ===========================================================================
template <class SLay>
static void run_hand(const char* tag, SLay slay, Buffers& buf) {
    CUDA_CHECK(cudaMemset(buf.d_b, 0, Buffers::bytes));
    transpose_hand_kernel<<<grid(), NTHR, cosize_v<SLay> * sizeof(float)>>>(buf.d_a, buf.d_b, slay);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  %-34s 列读 %2d-way   %s\n", tag, worst_col_conflict(slay),
           buf.check() ? "正确" : "错误");
}

template <class SLay>
static void run_tma(const char* tag, SLay slay, Buffers& buf) {
    auto mA = make_tensor(make_gmem_ptr(buf.d_a),
                          make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, mA, slay);
    CUDA_CHECK(cudaMemset(buf.d_b, 0, Buffers::bytes));
    transpose_tma_kernel<<<grid(), NTHR, cosize_v<SLay> * sizeof(float)>>>(tma, slay, buf.d_b);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  %-34s 列读 %2d-way   %s\n", tag, worst_col_conflict(slay),
           buf.check() ? "正确" : "错误");
}

static void run_both(const char* tag, Buffers& buf) {
    auto mA = make_tensor(make_gmem_ptr(buf.d_a),
                          make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
    auto mB = make_tensor(make_gmem_ptr(buf.d_b),
                          make_layout(make_shape(Int<N>{}, Int<M>{}), LayoutRight{}));
    auto slay = tile_to_shape(GMMA::Layout_K_SW128_Atom<float>{},
                              make_shape(Int<CM>{}, Int<CN>{}));
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, mA, slay);
    auto tma_store = make_tma_copy(SM90_TMA_STORE{}, mB, slay);
    CUDA_CHECK(cudaMemset(buf.d_b, 0, Buffers::bytes));
    transpose_both_kernel<<<grid(), NTHR, cosize_v<decltype(slay)> * sizeof(float)>>>(
        tma_load, tma_store, slay);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  %-34s 列读 %2d-way   %s\n", tag, worst_col_conflict(slay),
           buf.check() ? "正确" : "错误");
}

static void section6_rounds() {
    print_separator("§6  capstone: 转置六版");

    Buffers buf;
    auto plain = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto swz = tile_to_shape(GMMA::Layout_K_SW128_Atom<float>{}, make_shape(Int<CM>{}, Int<CN>{}));

    printf("  A: %dx%d float, row-major;  B = A^T (B[n,m] = A[m,n])\n", M, N);
    printf("  tile %dx%d, grid (%d,%d), 每 CTA %d 线程\n\n", CM, CN, grid().x, grid().y, NTHR);
    printf("  每一版只换一件事:\n\n");

    run_hand("t1  手写 + plain smem", plain, buf);
    run_hand("t2  手写 + SW128 smem", swz, buf);
    run_tma("t3  TMA load + plain smem", plain, buf);
    run_tma("t4  TMA load + SW128 smem", swz, buf);
    run_both("t5  TMA load + TMA store", buf);

    printf("\n  怎么读这张表:\n");
    printf("    t1 -> t2   手写搬运也吃 swizzle 的红利: 列读 32-way -> 16-way\n");
    printf("    t1 -> t3   TMA 替换手写 load, 结果一样 —— 搬运换硬件, 转置逻辑不动\n");
    printf("    t2 -> t4   TMA + swizzle 是本章主线: 搬运换硬件, 摆法由 descriptor 管\n");
    printf("    t4 -> t5   两端都交给硬件: 两条搬运指令, 其余线程只做转置本身\n");

    printf("\n  注意这里 swizzle 只把列读从 32-way 降到 16-way:\n");
    printf("  因为 tile 是 32x32, SW128 原子的 8 行模式 (32x32 的 4x4 分块) 只\n");
    printf("  用了一半。tile 更大时 (比如 128x128, 见 cute_05 之后) 会降到 8-way。\n");
    printf("  这不妨碍结论: plain 的 32-way 是实打实的最坏情况。\n");
}

// ===========================================================================
// §6.2  越界版: 不整除的矩阵
// ===========================================================================
constexpr int OM = 1000, ON = 640;  // 1000 % 32 = 8, 640 % 32 = 0 -> 一半 tile 越界

template <class TmaLoad, class TmaStore>
__global__ static void transpose_oob_kernel(__grid_constant__ const TmaLoad tma_load,
                                            __grid_constant__ const TmaStore tma_store) {
    constexpr int tx_bytes = CM * CN * sizeof(float);
    __shared__ __align__(128) float smem[CM * CN];
    __shared__ uint64_t bar;

    auto sT = make_tensor(make_smem_ptr(smem),
                          make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{}));
    auto blk = make_coord(blockIdx.x, blockIdx.y);

    if (threadIdx.x == 0) initialize_barrier(bar, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        set_barrier_transaction_bytes(bar, tx_bytes);
        auto gc = tma_load.get_tma_tensor(make_shape(Int<OM>{}, Int<ON>{}));
        auto gt = local_tile(gc, Shape<Int<CM>, Int<CN>>{}, blk);
        auto per = tma_load.get_slice(0);
        copy(tma_load.with(bar), per.partition_S(gt), per.partition_D(sT));
    }
    __syncthreads();
    wait_barrier(bar, 0);

    // 转置 (界外部分是无意义的 0, 转过去也是 0, 不影响界内)
    __shared__ float tmp[CM * CN];
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) {
        int r = i / CN, c = i % CN;
        tmp[i] = sT(c, r);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) {
        int r = i / CN, c = i % CN;
        sT(r, c) = tmp[i];
    }
    __syncthreads();
    tma_store_fence();

    if (threadIdx.x == 0) {
        auto gc = tma_store.get_tma_tensor(make_shape(Int<ON>{}, Int<OM>{}));
        auto gt = local_tile(gc, Shape<Int<CN>, Int<CM>>{}, make_coord(blockIdx.y, blockIdx.x));
        auto per = tma_store.get_slice(0);
        copy(tma_store, per.partition_S(sT), per.partition_D(gt));
        tma_store_arrive();
    }
    tma_store_wait<0>();
}

static void section6_oob() {
    print_separator("§6.2  越界版: 不整除的矩阵");

    const int gM = (OM + CM - 1) / CM, gN = (ON + CN - 1) / CN;  // (32, 20)
    float *da, *db;
    CUDA_CHECK(cudaMalloc(&da, sizeof(float) * OM * ON));
    CUDA_CHECK(cudaMalloc(&db, sizeof(float) * ON * OM));
    float* ha = new float[OM * ON];
    float* hb = new float[ON * OM];
    for (int m = 0; m < OM; ++m)
        for (int n = 0; n < ON; ++n) ha[m * ON + n] = float(m * ON + n);
    CUDA_CHECK(cudaMemcpy(da, ha, sizeof(float) * OM * ON, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(db, 0, sizeof(float) * ON * OM));

    auto mA = make_tensor(make_gmem_ptr(da), make_layout(make_shape(Int<OM>{}, Int<ON>{}),
                                                         LayoutRight{}));
    auto mB = make_tensor(make_gmem_ptr(db), make_layout(make_shape(Int<ON>{}, Int<OM>{}),
                                                         LayoutRight{}));
    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, mA, slay);
    auto tma_store = make_tma_copy(SM90_TMA_STORE{}, mB, slay);

    transpose_oob_kernel<<<dim3(gM, gN), NTHR>>>(tma_load, tma_store);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hb, db, sizeof(float) * ON * OM, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int m = 0; m < OM && ok; ++m)
        for (int n = 0; n < ON; ++n)
            if (hb[n * OM + m] != ha[m * ON + n]) ok = false;

    printf("  矩阵 %dx%d (1000 %% 32 = 8, 行方向不整除), grid = (%d,%d)\n", OM, ON, gM, gN);
    printf("  最右一列 CTA 的界外部分由硬件填 0, 转置后再由硬件跳过不写。\n");
    printf("  全程零 predicate。结果 = %s\n", ok ? "正确" : "错误");

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    delete[] ha;
    delete[] hb;
}

int main() {
    printf("cute_04 capstone —— 矩阵转置: 本章所有工具的合练\n");
    printf("对应 README §6    需要 -arch=sm_90a\n");

    section6_rounds();
    section6_oob();

    print_separator("小结");
    printf("  t1/t2   手写搬运: swizzle 是逻辑层重排, 只影响读 smem 的冲突\n");
    printf("  t3/t4   TMA load: 搬运换硬件, swizzle 由 descriptor 在物理层做\n");
    printf("  t5      两端 TMA: 两条搬运指令, 其余线程只做转置\n");
    printf("  t6      不整除: 硬件自动兜边界, 零 predicate\n");

    printf("\n这一章到此为止:\n");
    printf("  v0  手写搬运的基准\n");
    printf("  v1  TMA load + mbarrier + 坐标 tensor\n");
    printf("  v2  TMA store + fence + 边界自动处理\n");
    printf("  v3  smem 怎么摆: bank / padding / swizzle / descriptor 四种模式\n");
    printf("  v4  多 stage: 搬运和计算重叠 + tma_partition 写法\n");
    printf("  capstone  全部串成一个转置\n");

    printf("\n下一章 (cute_05 MMA): 数据摆好了, 谁来算? Tensor Core。\n");
    printf("WGMMA 会直接按 descriptor 读这块 smem —— 那时你会感谢 §4 的 swizzle。\n");
    printf("\ncapstone OK\n");
    return 0;
}
