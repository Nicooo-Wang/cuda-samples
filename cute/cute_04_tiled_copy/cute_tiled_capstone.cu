// cute_04 capstone —— 矩阵转置: naive / padding / swizzle 三版对比
//
// 对应 README §7 + §8。
//
// 转置是"smem layout 决定性能"最干净的例子:
//   写 smem 按行 (合并), 读 smem 按列 (冲突) —— 冲突全部集中在一处, 好观察。
//
// ---------------------------------------------------------------------------
// 阅读方式
//
// 每个版本都是「一个 kernel + 紧跟其后的一个 host 函数」, host 函数里自带这一版
// 需要的全部东西: 缓冲区、smem layout、launch、验证、计时。
// 从上往下顺读即可, 不需要跳到 main 里去找参数是怎么来的。
//
// v2/v3/v4 共用同一个 kernel, 唯一的差别是各自 host 函数里声明的那一个 layout ——
// 这就是 cute_03 §4「host 描述, kernel 索引」的直接回报。
// ---------------------------------------------------------------------------
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_capstone

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
// ---------------------------------------------------------------------------
constexpr int TILE = 32;    // smem 里中转的方块边长 (32x32 float)
constexpr int NTHR = 256;   // 每 block 线程数
constexpr int M = 8192, N = 8192;

// ---------------------------------------------------------------------------
// 四个版本共用的缓冲区 —— 构造时分配并填好, 析构时释放
//
// 每个 run_* 开头写一行 `Buffers buf;` 就得到一套干净的 in/out。
// ---------------------------------------------------------------------------
struct Buffers {
    static constexpr size_t bytes = size_t(M) * N * sizeof(float);

    float* d_in;
    float* d_out;
    float* h_in;
    float* h_out;

    Buffers() {
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        h_in = (float*)malloc(bytes);
        h_out = (float*)malloc(bytes);
        for (size_t i = 0; i < size_t(M) * N; ++i) h_in[i] = float(i % 1024);
        CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    }

    ~Buffers() {
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        free(h_in);
        free(h_out);
    }

    // 抽稀采样比对: 8192x8192 全量比对太慢, 按互质步长扫一遍足够抓出 layout 错误
    bool check() {
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < M; i += 97)
            for (int j = 0; j < N; j += 89)
                if (h_out[size_t(j) * M + i] != h_in[size_t(i) * N + j]) return false;
        return true;
    }
};

struct Result {
    const char* name;
    float ms;
    bool ok;
};

static dim3 grid() { return dim3(N / TILE, M / TILE); }

// 每版结尾都是这一句: 算带宽、打印、打包成 Result
static Result report(const char* name, float ms, bool ok) {
    printf("    %.3f ms   %.1f GB/s   %s\n", ms,
           transpose_bandwidth_gbs(size_t(M) * N, sizeof(float), ms), ok ? "正确" : "错误");
    return {name, ms, ok};
}

// 打印一个 smem layout 的关键属性: 摆法、占多少、列读撞得多狠
template <class SLay>
static void describe(SLay slay) {
    printf("    layout = ");
    print(slay);
    printf("   cosize = %d", int(cosize(slay)));
    printf("   列读 = %d-way\n", max_bank_requests(32, [&](int l) { return int(slay(l, 0)) * 4; }));
}

// ===========================================================================
// v1  完全不过 smem: 直接跨步写
//
// gmem 读是合并的 (一个 warp 读同一行连续 32 个), 但写是跨步的 —— 相邻 lane
// 写到相隔 M 个 float 的位置, 32 次写命中 32 个不同的 128B sector。
// 不用 smem, 就只能在读和写里牺牲一边。
// ===========================================================================
__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out) {
    int by = blockIdx.y * TILE;
    int col = blockIdx.x * TILE + threadIdx.x % TILE;
#pragma unroll
    for (int r = by + threadIdx.x / TILE; r < by + TILE; r += NTHR / TILE)
        if (r < M && col < N) out[size_t(col) * M + r] = in[size_t(r) * N + col];
}

static Result run_naive() {
    printf("\nv1  naive —— 不过 smem, 写 gmem 跨步\n");

    Buffers buf;
    float ms = time_kernel([&] { transpose_naive<<<grid(), NTHR>>>(buf.d_in, buf.d_out); });
    return report("naive (no smem)", ms, buf.check());
}

// ===========================================================================
// v2/v3/v4 共用的 kernel —— 过 smem, 唯一的变量是传进来的 slay
//
// 读写两边都合并了: gmem 按行读 -> smem, smem 按列读 -> gmem 按行写。
// 代价是所有冲突都被挤到「按列读 smem」这一步, 于是 layout 成了唯一的变量。
//
// smem 数组按 cosize_v 开 —— padding 版的 cosize 比 size 大, 用 size 会溢出。
// ===========================================================================
template <class SLay>
__global__ void transpose_smem(const float* __restrict__ in, float* __restrict__ out, SLay slay) {
    __shared__ __align__(128) float raw[cosize_v<SLay>];
    auto s = make_tensor(make_smem_ptr(raw), slay);

    int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
    int tx = threadIdx.x % TILE, ty = threadIdx.x / TILE;

    // gmem -> smem: 按行读, 合并
#pragma unroll
    for (int r = ty; r < TILE; r += NTHR / TILE)
        if (by + r < M && bx + tx < N) s(r, tx) = in[size_t(by + r) * N + bx + tx];

    __syncthreads();

    // smem -> gmem: 按列读 smem (冲突在这里), 按行写 gmem (合并)
#pragma unroll
    for (int r = ty; r < TILE; r += NTHR / TILE)
        if (bx + r < N && by + tx < M) out[size_t(bx + r) * M + by + tx] = s(tx, r);
}

// ---------------------------------------------------------------------------
// v2  plain —— 行 stride = 32 = bank 数, 整列撞同一个 bank
// ---------------------------------------------------------------------------
static Result run_smem_plain() {
    printf("\nv2  plain —— 过 smem, 但没管 bank\n");

    auto slay = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                            make_stride(Int<TILE>{}, Int<1>{}));
    describe(slay);

    Buffers buf;
    float ms = time_kernel([&] { transpose_smem<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay); });
    return report("smem plain (32-way conf)", ms, buf.check());
}

// ---------------------------------------------------------------------------
// v3  padded —— 行 stride 改成 33, 每行错开一个 bank
// ---------------------------------------------------------------------------
static Result run_smem_padded() {
    printf("\nv3  padded —— stride 33, 用多占 smem 换掉冲突\n");

    auto slay = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                            make_stride(Int<TILE + 1>{}, Int<1>{}));
    describe(slay);
    printf("    代价: 比 plain 多占 %d 个 float, 且行首不再 128B 对齐\n",
           int(cosize(slay)) - TILE * TILE);

    Buffers buf;
    float ms = time_kernel([&] { transpose_smem<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay); });
    return report("smem padded (stride 33)", ms, buf.check());
}

// ---------------------------------------------------------------------------
// v4  swizzle —— 不改 shape 也不多占一个字节, 只改坐标->偏移的映射
// ---------------------------------------------------------------------------
static Result run_smem_swizzle() {
    printf("\nv4  swizzle —— 同样消掉冲突, 但一个字节都不多占\n");

    auto plain = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                             make_stride(Int<TILE>{}, Int<1>{}));
    auto slay = composition(Swizzle<5, 0, 5>{}, plain);
    describe(slay);
    printf("    SM90 上只能走这条路: WGMMA 编译期拒绝 padding 过的 layout (见 v2 §5.3)\n");
    printf("    注意 M=0 -> 行内一个连续对都不剩, 宽向量 atom 用不了 (见 v0 §3.4)\n");

    Buffers buf;
    float ms = time_kernel([&] { transpose_smem<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay); });
    return report("smem Swizzle<5,0,5>", ms, buf.check());
}

// ---------------------------------------------------------------------------
// v5  swizzle M=2 —— 同样消冲突, 但保住 4 个 float 连续
//
// v4 的 Swizzle<5,0,5> 把冲突消到 1-way, 代价是 M=0 打乱了行内连续性。
// Swizzle<3,2,3> 换个平衡点: 4-way 冲突, 但 4 个 float 仍连续 -> 宽向量能用。
// 这是 v0 §3.4 那张表在真实 kernel 上的验证。
// ---------------------------------------------------------------------------
static Result run_smem_swizzle_m2() {
    printf("\nv5  Swizzle<3,2,3> —— M=2, 用 4-way 冲突换回行内连续性\n");

    auto plain = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                             make_stride(Int<TILE>{}, Int<1>{}));
    auto slay = composition(Swizzle<3, 2, 3>{}, plain);
    describe(slay);
    printf("    GMMA 官方原子走的就是这条路线 (它们 M 全 = 4), 见 v2 §5.4\n");

    Buffers buf;
    float ms = time_kernel([&] { transpose_smem<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay); });
    return report("smem Swizzle<3,2,3>", ms, buf.check());
}

// ===========================================================================
// v6  TMA 版转置 —— 把搬运交给硬件
//
// v1-v5 的 gmem->smem 都是"每线程算地址"。这一步换成 TMA:
//   一个 lane 描述整块, 硬件搬, mbarrier 按字节等 (五个硬性条件见 v2 §5.2)。
//
// 注意这里用的是 half, 不是 float, 矩阵也小 (2048^2)。原因是对 TMA 而言
// 128 字节一行的 swizzle 模式天生是给 16-bit 元素设计的 (v2 §5.3);
// 而且 TMA 的价值在本章已经由 v3 的流水线证明了 —— 它擅长"整块搬进 smem
// 喂给 WGMMA", 不是"在纯转置这种转换操作上抢带宽"。
// 所以 v6 不放进 v1-v5 的 float 对比表, 它只是证明 TMA 这一步能换, 且换对。
// ===========================================================================
constexpr int HALF_M = 2048, HALF_N = 2048;  // 256 MB half
constexpr int HTILE = 128;                    // 一个 CTA 搬 128x128 half = 32KB
constexpr int HNTHR = 128;

template <class Tma, class SLay, class TS, class TC>
__global__ void transpose_tma(CUTLASS_GRID_CONSTANT Tma const tma, SLay slay3, TS stlay, TC tc,
                              half_t* out) {
    __shared__ __align__(128) half_t rawS[cosize_v<SLay>];  // 条件 3
    __shared__ __align__(8) uint64_t bar[1];

    auto s = make_tensor(make_smem_ptr(rawS), slay3);  // (HTILE,HTILE,1)   条件 4
    auto s2 = s(_, _, Int<0>{});
    // 转置视图: 读 sT 的行 = 读 s 的列 (v1 §4 的招)
    auto sT = make_tensor(make_smem_ptr(rawS), stlay);

    // 条件 1: 坐标 tensor
    auto mA = tma.get_tma_tensor(make_shape(HALF_M, HALF_N));
    auto gA = local_tile(mA, Shape<Int<HTILE>, Int<HTILE>>{}, make_coord(blockIdx.y, blockIdx.x));
    // 条件 5: tma_partition
    auto p = tma_partition(tma, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(s),
                           group_modes<0, 2>(gA));
    auto tAg = get<0>(p);
    auto tAs = get<1>(p);
    constexpr int txb = sizeof(make_tensor_like(tensor<0>(tAs)));

    int warp = cutlass::canonical_warp_idx_sync();
    int one = cute::elect_one_sync();
    using Bar = cutlass::arch::ClusterTransactionBarrier;
    if (warp == 0 && one) Bar::init(&bar[0], 1);
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    if (warp == 0 && one) {
        Bar::arrive_and_expect_tx(&bar[0], txb);
        copy(tma.with(bar[0]), tAg, tAs(_, Int<0>{}));  // 一条指令搬整块
    }
    Bar::wait(&bar[0], 0);
    __syncthreads();

    // store: 转置视图 -> gmem, 标量读 (转置视图下 smem 不连续, 不能向量化)
    auto thr = tc.get_slice(threadIdx.x);
    auto mOut = make_tensor(make_gmem_ptr(out),
                            make_layout(make_shape(HALF_N, HALF_M), make_stride(HALF_M, Int<1>{})));
    auto gOut = local_tile(mOut, Shape<Int<HTILE>, Int<HTILE>>{}, make_coord(blockIdx.x, blockIdx.y));
    copy(tc, thr.partition_S(sT), thr.partition_D(gOut));
}

static Result run_tma() {
    printf("\nv6  TMA 版转置  (half %dx%d, tile %d, 各 CTA 一个 lane 发 TMA)\n", HALF_M, HALF_N,
           HTILE);

    size_t bytes = size_t(HALF_M) * HALF_N * sizeof(half_t);
    half_t *d_a, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    half_t* h_a = new half_t[size_t(HALF_M) * HALF_N];
    half_t* h_out = new half_t[size_t(HALF_M) * HALF_N];
    for (size_t i = 0; i < size_t(HALF_M) * HALF_N; ++i) h_a[i] = half_t(float(i % 1024));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    // host 侧: 条件 2 用真实指针建 descriptor; 条件 4 传 PIPE 切片
    auto gm = make_tensor(make_gmem_ptr(d_a),
                          make_layout(make_shape(HALF_M, HALF_N), make_stride(HALF_N, Int<1>{})));
    auto slay3 = tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                               make_shape(Int<HTILE>{}, Int<HTILE>{}, Int<1>{}));
    auto tma = make_tma_atom(SM90_TMA_LOAD{}, gm, slay3(_, _, Int<0>{}),
                             make_shape(Int<HTILE>{}, Int<HTILE>{}));
    auto stlay = composition(slay3(_, _, Int<0>{}),
                             make_layout(make_shape(Int<HTILE>{}, Int<HTILE>{}),
                                         make_stride(Int<HTILE>{}, Int<1>{})));
    auto tc = make_tiled_copy(Copy_Atom<UniversalCopy<half_t>, half_t>{},
                              make_layout(make_shape(Int<16>{}, Int<8>{}),
                                          make_stride(Int<8>{}, Int<1>{})),
                              make_layout(make_shape(Int<1>{}, Int<1>{})));  // 标量读

    dim3 block(HNTHR), cluster(1, 1, 1), grid(HALF_N / HTILE, HALF_M / HTILE);
    cutlass::ClusterLaunchParams params{grid, block, cluster, 0};
    void const* kptr = reinterpret_cast<void const*>(
        &transpose_tma<decltype(tma), decltype(slay3), decltype(stlay), decltype(tc)>);

    float ms = time_kernel(
        [&] { cutlass::launch_kernel_on_cluster(params, kptr, tma, slay3, stlay, tc, d_out); },
        5, 100);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < HALF_M; i += 13)
        for (int j = 0; j < HALF_N; j += 11)
            if (h_out[size_t(j) * HALF_M + i] != h_a[size_t(i) * HALF_N + j]) ok = false;

    printf("    转置结果 %s    %.4f ms   %.1f GB/s\n", ok ? "正确" : "错误", ms,
           transpose_bandwidth_gbs(size_t(HALF_M) * HALF_N, sizeof(half_t), ms));
    printf("    (half/%d^2, 和 v1-v5 的 float/8192^2 不可直接比 —— 见上面的说明)\n", HALF_M);

    cudaFree(d_a);
    cudaFree(d_out);
    delete[] h_a;
    delete[] h_out;
    return {"TMA (half)", ms, ok};
}

// ===========================================================================
// main —— 按顺序跑六版, 汇总
// ===========================================================================
int main() {
    printf("cute_04 capstone —— %dx%d float 转置\n", M, N);
    printf("TILE = %d, 每 block %d 线程, grid = (%d,%d)\n", TILE, NTHR, N / TILE, M / TILE);

    Result results[] = {
        run_naive(), run_smem_plain(), run_smem_padded(), run_smem_swizzle(),
        run_smem_swizzle_m2(),
    };
    constexpr int NRES = sizeof(results) / sizeof(results[0]);

    print_separator("汇总");
    printf("  %-26s %10s %12s %6s\n", "version", "time(ms)", "GB/s", "ok");
    for (auto& r : results)
        printf("  %-26s %10.3f %12.1f %6s\n", r.name, r.ms,
               transpose_bandwidth_gbs(size_t(M) * N, sizeof(float), r.ms), r.ok ? "yes" : "NO");

    printf("\n  相对 plain smem 的加速:\n");
    for (int i = 2; i < NRES; ++i) printf("    %-26s %.2fx\n", results[i].name,
                                          results[1].ms / results[i].ms);

    printf("\n  三种消冲突方案在 SM90 上的可用性:\n");
    printf("    padding stride 33   最快, 但 WGMMA 编译期拒绝  -> SM90 上不能用\n");
    printf("    Swizzle<5,0,5>      冲突最少, 但 M=0 挡住宽向量 atom\n");
    printf("    Swizzle<3,2,3>      M=2 保住 4-float 连续 -> 和 GMMA 官方原子同路线\n");

    printf("\n  读一遍 + 写一遍 = %.1f GB, 本机 HBM 理论带宽约 4.9 TB/s\n",
           2.0 * Buffers::bytes / 1e9);

    print_separator("v6: TMA 版转置 (单独展示, 证明搬运这步能交给硬件)");
    run_tma();
    printf("\n  为什么 v6 不在上面的 float 表里:\n");
    printf("    TMA 的 SW128 descriptor 天生是 16-bit 的 (v2 §5.3), 用 float 要另配几何;\n");
    printf("    而且 TMA 擅长的是\"整块搬进 smem 喂 WGMMA\"(v3 的流水线已经证明),\n");
    printf("    不是纯转置这种转换操作 —— 后者瓶颈在 bank 冲突, 不在搬运指令。\n");

    printf("\ncapstone OK\n");
    return 0;
}
