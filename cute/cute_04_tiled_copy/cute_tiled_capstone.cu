// cute_04 capstone —— 矩阵转置: naive / padding / swizzle 三版对比
//
// 对应 README §8。
//
// 转置是"smem layout 决定性能"最干净的例子:
//   写 smem 按行 (合并), 读 smem 按列 (冲突) —— 冲突全部集中在一处, 好观察。
//
// 阅读方式: 每一版都是「kernel + 紧跟其后的 launch 代码」一个整体, 从上往下顺读,
// 不需要在 main 和 kernel 之间来回跳。main 只负责建缓冲区、按顺序叫这四版。
// 三个 smem layout 也各自写在自己那一版里, 而不是攒在 main 顶部。
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_capstone

#include <cute/tensor.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 全局配置 + 四版共用的测量脚手架
//
// 先把"每版都要做的事"抽干净, 下面每一版就只剩它自己独有的那几行。
// ---------------------------------------------------------------------------
constexpr int TILE = 32;
constexpr int NTHR = 256;
constexpr int M = 8192, N = 8192;

// 四版共享的缓冲区。main 里建一次, 按引用传给每个 run_*。
struct Bufs {
    const float* h_in;
    float* h_out;
    float* d_in;
    float* d_out;
    size_t bytes;
};

struct Row {
    const char* name;
    float ms;
    bool ok;
};

static dim3 grid() { return dim3(N / TILE, M / TILE); }

// 抽稀采样比对: 8192x8192 全量比对太慢, 按互质步长扫一遍足够抓出 layout 错误
static bool verify(const Bufs& b) {
    for (int i = 0; i < M; i += 97)
        for (int j = 0; j < N; j += 89)
            if (b.h_out[size_t(j) * M + i] != b.h_in[size_t(i) * N + j]) return false;
    return true;
}

// 每版都是这三步: 清 dst -> 计时 -> 拷回来验证。抽出来免得四份重复。
template <class Launch>
static Row measure(const char* name, Bufs& b, Launch&& launch) {
    CUDA_CHECK(cudaMemset(b.d_out, 0, b.bytes));
    float ms = time_kernel(launch);
    CUDA_CHECK(cudaMemcpy(b.h_out, b.d_out, b.bytes, cudaMemcpyDeviceToHost));
    Row r{name, ms, verify(b)};
    printf("    %.3f ms   %.1f GB/s   %s\n", r.ms,
           transpose_bandwidth_gbs(size_t(M) * N, sizeof(float), r.ms), r.ok ? "正确" : "错误");
    return r;
}

// 打印一个 smem layout 的关键属性: 摆法、占多少、列读撞得多狠
template <class SLay>
static void describe(SLay slay) {
    printf("    layout = ");
    print(slay);
    printf("   cosize = %d", int(cosize(slay)));
    int worst = max_bank_requests(32, [&](int l) { return int(slay(l, 0)) * 4; });
    printf("   列读 = %d-way\n", worst);
}

// ---------------------------------------------------------------------------
// v1  完全不过 smem: 直接跨步写
//
// gmem 读是合并的 (一个 warp 读同一行的连续 32 个), 但写是跨步的 —— 相邻 lane
// 写到相隔 M 个 float 的位置, 32 次写命中 32 个不同的 128B sector。
// 这是 baseline: 不用 smem, 就只能在读和写里牺牲一边。
// ---------------------------------------------------------------------------
__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out) {
    int by = blockIdx.y * TILE;
    int col = blockIdx.x * TILE + threadIdx.x % TILE;
#pragma unroll
    for (int r = by + threadIdx.x / TILE; r < by + TILE; r += NTHR / TILE)
        if (r < M && col < N) out[size_t(col) * M + r] = in[size_t(r) * N + col];
}

static Row run_naive(Bufs& b) {
    printf("\nv1  naive —— 不过 smem, 写 gmem 跨步\n");
    return measure("naive (no smem)", b,
                   [&] { transpose_naive<<<grid(), NTHR>>>(b.d_in, b.d_out); });
}

// ---------------------------------------------------------------------------
// v2/v3/v4  过 smem —— 三版共用下面这一个 kernel, 唯一的差别是传进来的 slay
//
// 读写两边都合并了: gmem 按行读 -> smem, smem 按列读 -> gmem 按行写。
// 代价是所有冲突都被挤到"按列读 smem"这一步, 于是 layout 成了唯一的变量。
//
// 这正是 cute_03 §4 那套分工的回报: 「smem 怎么摆」是 host 传进来的参数,
// kernel 只管索引。下面三个 run_* 各自声明自己的 layout, kernel 一个字不改。
// smem 数组按 cosize_v 开 —— padding 版的 cosize 比 size 大, 用 size 会溢出。
// ---------------------------------------------------------------------------
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

// --- v2: plain。行 stride = 32 = bank 数, 列方向整列撞同一个 bank ---
static Row run_smem_plain(Bufs& b) {
    auto plain =
        make_layout(make_shape(Int<TILE>{}, Int<TILE>{}), make_stride(Int<TILE>{}, Int<1>{}));

    printf("\nv2  plain —— 过 smem, 但没管 bank\n");
    describe(plain);
    return measure("smem plain (32-way conf)", b,
                   [&] { transpose_smem<<<grid(), NTHR>>>(b.d_in, b.d_out, plain); });
}

// --- v3: padding。行 stride 改成 33, 每行错开一个 bank ---
static Row run_smem_padded(Bufs& b) {
    auto pad =
        make_layout(make_shape(Int<TILE>{}, Int<TILE>{}), make_stride(Int<TILE + 1>{}, Int<1>{}));

    printf("\nv3  padded —— stride 33, 用多占 smem 换掉冲突\n");
    describe(pad);
    printf("    代价: 比 plain 多占 %d 个 float, 且行首不再 128B 对齐\n",
           int(cosize(pad)) - TILE * TILE);
    return measure("smem padded (stride 33)", b,
                   [&] { transpose_smem<<<grid(), NTHR>>>(b.d_in, b.d_out, pad); });
}

// --- v4: swizzle。不改 shape 也不多占一个字节, 只改坐标->偏移的映射 ---
static Row run_smem_swizzle(Bufs& b) {
    auto plain =
        make_layout(make_shape(Int<TILE>{}, Int<TILE>{}), make_stride(Int<TILE>{}, Int<1>{}));
    auto swz = composition(Swizzle<5, 0, 5>{}, plain);

    printf("\nv4  swizzle —— 同样消掉冲突, 但一个字节都不多占\n");
    describe(swz);
    printf("    这是 Hopper 唯一能走的路: TMA / WGMMA 不接受 padding 过的 layout\n");
    return measure("smem Swizzle<5,0,5>", b,
                   [&] { transpose_smem<<<grid(), NTHR>>>(b.d_in, b.d_out, swz); });
}

// ---------------------------------------------------------------------------
// main —— 只做三件事: 建缓冲区, 按顺序叫四版, 汇总
// ---------------------------------------------------------------------------
int main() {
    printf("cute_04 capstone —— %dx%d float 转置\n", M, N);
    printf("TILE = %d, 每 block %d 线程, grid = (%d,%d)\n", TILE, NTHR, N / TILE, M / TILE);

    Bufs b{};
    b.bytes = size_t(M) * N * sizeof(float);

    float* h_in = (float*)malloc(b.bytes);
    for (size_t i = 0; i < size_t(M) * N; ++i) h_in[i] = float(i % 1024);
    b.h_in = h_in;
    b.h_out = (float*)malloc(b.bytes);

    CUDA_CHECK(cudaMalloc(&b.d_in, b.bytes));
    CUDA_CHECK(cudaMalloc(&b.d_out, b.bytes));
    CUDA_CHECK(cudaMemcpy(b.d_in, h_in, b.bytes, cudaMemcpyHostToDevice));

    Row rows[4] = {run_naive(b), run_smem_plain(b), run_smem_padded(b), run_smem_swizzle(b)};

    print_separator("汇总");
    printf("  %-26s %10s %12s %6s\n", "version", "time(ms)", "GB/s", "ok");
    for (auto& r : rows)
        printf("  %-26s %10.3f %12.1f %6s\n", r.name, r.ms,
               transpose_bandwidth_gbs(size_t(M) * N, sizeof(float), r.ms), r.ok ? "yes" : "NO");

    printf("\n  相对 plain smem 的加速:\n");
    for (int i = 2; i < 4; ++i) printf("    %-26s %.2fx\n", rows[i].name, rows[1].ms / rows[i].ms);

    printf("\n  读一遍 + 写一遍 = %.1f GB, 本机 HBM 理论带宽约 4.9 TB/s\n", 2.0 * b.bytes / 1e9);

    CUDA_CHECK(cudaFree(b.d_in));
    CUDA_CHECK(cudaFree(b.d_out));
    free(h_in);
    free(b.h_out);
    printf("\ncapstone OK\n");
    return 0;
}
