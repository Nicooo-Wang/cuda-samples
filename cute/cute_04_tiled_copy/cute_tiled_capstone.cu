// cute_04 capstone —— 矩阵转置: naive / padding / swizzle 三版对比
//
// 对应 README §8。
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
    printf("    这是 Hopper 唯一能走的路: TMA / WGMMA 不接受 padding 过的 layout\n");

    Buffers buf;
    float ms = time_kernel([&] { transpose_smem<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay); });
    return report("smem Swizzle<5,0,5>", ms, buf.check());
}

// ===========================================================================
// main —— 按顺序跑四版, 汇总
// ===========================================================================
int main() {
    printf("cute_04 capstone —— %dx%d float 转置\n", M, N);
    printf("TILE = %d, 每 block %d 线程, grid = (%d,%d)\n", TILE, NTHR, N / TILE, M / TILE);

    Result results[] = {
        run_naive(), run_smem_plain(), run_smem_padded(), run_smem_swizzle(),
    };

    print_separator("汇总");
    printf("  %-26s %10s %12s %6s\n", "version", "time(ms)", "GB/s", "ok");
    for (auto& r : results)
        printf("  %-26s %10.3f %12.1f %6s\n", r.name, r.ms,
               transpose_bandwidth_gbs(size_t(M) * N, sizeof(float), r.ms), r.ok ? "yes" : "NO");

    printf("\n  相对 plain smem 的加速:\n");
    for (int i = 2; i < 4; ++i) printf("    %-26s %.2fx\n", results[i].name,
                                       results[1].ms / results[i].ms);

    printf("\n  读一遍 + 写一遍 = %.1f GB, 本机 HBM 理论带宽约 4.9 TB/s\n",
           2.0 * Buffers::bytes / 1e9);

    printf("\ncapstone OK\n");
    return 0;
}
