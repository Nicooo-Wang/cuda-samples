// cute_04 v1 —— 怎么用 Swizzle 做 Copy: 加了 swizzle 之后搬运逻辑要改什么?
//
// 对应 README §4。
//
// 这个文件回答一个具体问题: 我本来有一段能跑的搬运代码, 现在要加 swizzle,
// 我得改哪几行?
//
// 答案分两半:
//   标量搬运   -> 一行都不用改。swizzle 藏在 layout 里, s(r,c) 自动走新映射。
//   向量搬运   -> 逻辑代码同样不用改, 但 swizzle 的 M 参数必须选对, 否则编译失败。
//
// ---------------------------------------------------------------------------
// 阅读方式
//
// 每一版都是「一个 kernel + 紧跟其后的一个 host 函数」, host 函数里自带这一版
// 需要的全部东西: 缓冲区、layout、launch、验证、计时。从上往下顺读即可。
//
// v1a/v1b/v1c 共用同一个 kernel (transpose_scalar), 差别只有 host 侧那一行 layout。
// v2a/v2b 共用另一个 kernel (transpose_vec), 同样只差一行 layout。
// ---------------------------------------------------------------------------
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_v1

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

constexpr int TILE = 32;   // smem 方块边长
constexpr int NTHR = 256;  // 每 block 线程数
constexpr int M = 4096, N = 4096;

// ---------------------------------------------------------------------------
// 共用缓冲区 —— 构造时分配并填好, 析构时释放
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

    // 抽稀采样比对: 互质步长扫一遍足够抓出 layout 错误
    bool check() {
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < M; i += 97)
            for (int j = 0; j < N; j += 89)
                if (h_out[size_t(j) * M + i] != h_in[size_t(i) * N + j]) return false;
        return true;
    }
};

static dim3 grid() { return dim3(N / TILE, M / TILE); }

// 全列扫描取最坏, 不是只看第 0 列
template <class Lay>
static int worst_col_conflict(Lay lay) {
    int worst = 0;
    for (int c = 0; c < TILE; ++c) {
        int w = max_bank_requests(32, [&](int l) { return int(lay(l, c)) * 4; });
        if (w > worst) worst = w;
    }
    return worst;
}

template <class Lay>
static int min_run(Lay lay) {
    int g = 1 << 30;
    for (int r = 0; r < TILE; ++r) {
        int run = 1;
        for (int c = 1; c < TILE; ++c) {
            if (int(lay(r, c)) == int(lay(r, c - 1)) + 1) {
                ++run;
            } else {
                if (run < g) g = run;
                run = 1;
            }
        }
        if (run < g) g = run;
    }
    return g;
}

// 每版结尾都是这一句: 打印指标 + 带宽
template <class Lay>
static void report(const char* name, Lay slay, float ms, bool ok) {
    printf("    layout      = ");
    print(slay);
    printf("\n");
    printf("    列读 = %2d-way   行内最短连续 = %2d 个 float   cosize = %d\n",
           worst_col_conflict(slay), min_run(slay), int(cosize(slay)));
    printf("    %.3f ms   %.1f GB/s   %s\n", ms,
           transpose_bandwidth_gbs(size_t(M) * N, sizeof(float), ms), ok ? "正确" : "错误");
    (void)name;
}

// ===========================================================================
// 第一部分: 标量搬运 —— 加 swizzle 不用改一行代码
//
// 这个 kernel 被 v1a/v1b/v1c 三版共用。注意它对 swizzle 完全无感:
// 它只是写 s(r, tx) 和读 s(tx, r), 至于这两个坐标映到哪个偏移, 是 slay 的事。
// ===========================================================================
template <class SLay>
__global__ void transpose_scalar(const float* __restrict__ in, float* __restrict__ out,
                                 SLay slay) {
    __shared__ __align__(128) float raw[cosize_v<SLay>];
    auto s = make_tensor(make_smem_ptr(raw), slay);

    int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
    int tx = threadIdx.x % TILE, ty = threadIdx.x / TILE;

    // 写 smem: 按行走。同一 warp 的 32 个 lane 有连续的 tx -> gmem 合并读
#pragma unroll
    for (int r = ty; r < TILE; r += NTHR / TILE)
        if (by + r < M && bx + tx < N) s(r, tx) = in[size_t(by + r) * N + bx + tx];

    __syncthreads();

    // 读 smem: 按列走。s(tx, r) 的第一个下标是 tx -> 一个 warp 扫过一整列
    // 冲突全在这一行。写 gmem 那边 tx 连续 -> 合并写。
#pragma unroll
    for (int r = ty; r < TILE; r += NTHR / TILE)
        if (bx + r < N && by + tx < M) out[size_t(bx + r) * M + by + tx] = s(tx, r);
}

// ---------------------------------------------------------------------------
// v1a  标量 + plain —— 基准, 32-way 冲突
// ---------------------------------------------------------------------------
static float run_scalar_plain() {
    printf("\nv1a  标量搬运 + plain layout (基准)\n");

    auto slay = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                            make_stride(Int<TILE>{}, Int<1>{}));

    Buffers buf;
    float ms = time_kernel([&] { transpose_scalar<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay); });
    report("scalar plain", slay, ms, buf.check());
    return ms;
}

// ---------------------------------------------------------------------------
// v1b  标量 + Swizzle<5,0,5> —— kernel 一行没动, 只是 host 多套了一层
// ---------------------------------------------------------------------------
static float run_scalar_swz505() {
    printf("\nv1b  标量搬运 + Swizzle<5,0,5>   <- kernel 代码和 v1a 完全相同\n");

    auto plain = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                             make_stride(Int<TILE>{}, Int<1>{}));
    auto slay = composition(Swizzle<5, 0, 5>{}, plain);  // <- 唯一新增的一行

    Buffers buf;
    float ms = time_kernel([&] { transpose_scalar<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay); });
    report("scalar Sw<5,0,5>", slay, ms, buf.check());
    return ms;
}

// ---------------------------------------------------------------------------
// v1c  标量 + Swizzle<3,2,3> —— M=2, 保住 4 个 float 连续
// ---------------------------------------------------------------------------
static float run_scalar_swz323() {
    printf("\nv1c  标量搬运 + Swizzle<3,2,3>   <- 同一个 kernel, 又只换了这一行\n");

    auto plain = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                             make_stride(Int<TILE>{}, Int<1>{}));
    auto slay = composition(Swizzle<3, 2, 3>{}, plain);

    Buffers buf;
    float ms = time_kernel([&] { transpose_scalar<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay); });
    report("scalar Sw<3,2,3>", slay, ms, buf.check());
    return ms;
}

// ===========================================================================
// 第二部分: 向量搬运 —— 这里 swizzle 的 M 参数开始有硬约束
//
// 上面三版是标量搬 (一次一个 float)。真实 kernel 会用 128-bit atom 一次搬 4 个,
// 这时 gmem->smem 那一步走 TiledCopy。
//
// 逻辑代码依然不用改, 但 M 必须 >= 2 (2^2 = 4 个 float 连续), 否则
// make_tiled_copy 那一步就编译失败。这是本文件的核心结论。
// ===========================================================================
template <class SLay, class TC>
__global__ void transpose_vec(const float* __restrict__ in, float* __restrict__ out, SLay slay,
                              TC tc) {
    __shared__ __align__(128) float raw[cosize_v<SLay>];
    auto s = make_tensor(make_smem_ptr(raw), slay);

    // gmem -> smem: 用 128-bit atom, 每线程 4 个连续 float
    auto mIn = make_tensor(make_gmem_ptr(in),
                           make_layout(make_shape(Int<M>{}, Int<N>{}),
                                       make_stride(Int<N>{}, Int<1>{})));
    auto gIn = local_tile(mIn, Shape<Int<TILE>, Int<TILE>>{},
                          make_coord(blockIdx.y, blockIdx.x));

    auto thr = tc.get_slice(threadIdx.x);
    copy(tc, thr.partition_S(gIn), thr.partition_D(s));  // <- 这一行要求 layout 可向量化

    __syncthreads();

    // smem -> gmem: 按列读 smem, 按行写 gmem (这一步仍是标量, 因为列方向本就不连续)
    int tx = threadIdx.x % TILE, ty = threadIdx.x / TILE;
#pragma unroll
    for (int r = ty; r < TILE; r += NTHR / TILE)
        out[size_t(blockIdx.x * TILE + r) * M + blockIdx.y * TILE + tx] = s(tx, r);
}

// 128-bit atom: 32 个线程横排 x 8 组, 每线程 4 个连续 float
static auto make_vec_copy() {
    return make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                           make_layout(make_shape(Int<32>{}, Int<8>{}),
                                       make_stride(Int<8>{}, Int<1>{})),
                           make_layout(make_shape(Int<1>{}, Int<4>{})));
}

// ---------------------------------------------------------------------------
// v2a  向量 + plain
// ---------------------------------------------------------------------------
static float run_vec_plain() {
    printf("\nv2a  128-bit atom + plain layout\n");

    auto slay = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                            make_stride(Int<TILE>{}, Int<1>{}));
    auto tc = make_vec_copy();

    Buffers buf;
    float ms =
        time_kernel([&] { transpose_vec<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay, tc); });
    report("vec plain", slay, ms, buf.check());
    return ms;
}

// ---------------------------------------------------------------------------
// v2b  向量 + Swizzle<3,2,3> —— M=2 刚好够 128-bit
// ---------------------------------------------------------------------------
static float run_vec_swz323() {
    printf("\nv2b  128-bit atom + Swizzle<3,2,3>  (M=2 -> 4 个 float 连续, 刚好够)\n");

    auto plain = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                             make_stride(Int<TILE>{}, Int<1>{}));
    auto slay = composition(Swizzle<3, 2, 3>{}, plain);
    auto tc = make_vec_copy();

    Buffers buf;
    float ms =
        time_kernel([&] { transpose_vec<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay, tc); });
    report("vec Sw<3,2,3>", slay, ms, buf.check());

    printf("\n    换成 Swizzle<5,0,5> 会怎样? 把下面这行的注释去掉试试:\n");
    printf("      // auto bad = composition(Swizzle<5,0,5>{}, plain);\n");
    printf("      // copy(tc, thr.partition_S(gIn), thr.partition_D(make_tensor(..., bad)));\n");
    printf("    编译期直接失败:\n");
    printf("      static assertion failed: \"Copy_Traits: dst failed to vectorize into\n");
    printf("      registers. Layout is incompatible with this CopyOp.\"\n");
    printf("    因为 M=0 时行内一个连续对都不剩, 4 个 float 凑不出一条 128-bit 指令。\n");
    return ms;
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    printf("cute_04 v1 —— 怎么用 Swizzle 做 Copy\n");
    printf("对应 README §4      %dx%d float 转置, TILE=%d, NTHR=%d\n", M, N, TILE, NTHR);

    print_separator("第一部分: 标量搬运 —— 加 swizzle 不改一行 kernel 代码");
    float a = run_scalar_plain();
    float b = run_scalar_swz505();
    float c = run_scalar_swz323();

    printf("\n  三版用的是同一个 transpose_scalar kernel。差别只有 host 侧那一行:\n");
    printf("    v1a:  auto slay = make_layout(...);\n");
    printf("    v1b:  auto slay = composition(Swizzle<5,0,5>{}, plain);\n");
    printf("    v1c:  auto slay = composition(Swizzle<3,2,3>{}, plain);\n");
    printf("\n  加速 (相对 plain):  Sw<5,0,5> %.2fx    Sw<3,2,3> %.2fx\n", a / b, a / c);
    printf("  标量搬运下两者基本打平: 冲突都消掉了, 而搬运本来就是一次一个 float,\n");
    printf("  行内连续性没被用上, 所以 M 的差别在这里看不出来。\n");
    printf("  M 的代价要到第二部分 (向量搬运) 才会显形。\n");

    print_separator("第二部分: 向量搬运 —— M 参数变成硬约束");
    float d = run_vec_plain();
    float e = run_vec_swz323();
    printf("\n  加速 (相对 vec plain): Sw<3,2,3> %.2fx\n", d / e);

    print_separator("小结: 加 swizzle 要改什么");
    printf("  逻辑代码            : 一行都不用改。s(r,c) 自动走 swizzle 后的映射。\n");
    printf("  host 侧 layout 声明 : 加一层 composition(Swizzle<B,M,S>{}, plain)。\n");
    printf("  smem 数组大小       : 写 cosize_v<SLay>, 不要写 size。\n");
    printf("  对齐                : __align__(128), TMA 硬性要求。\n");
    printf("  用了宽向量 atom     : M 必须 >= log2(向量元素数), 否则编译失败。\n");
    printf("\n  一句话: swizzle 是 layout 的属性, 不是搬运代码的属性。\n");
    printf("  这就是 cute_03 §4 「host 描述, kernel 索引」的直接回报。\n");

    printf("\n下一步: v2 看 SM90 的 TMA 和 WGMMA 对 layout 的真实要求。\n");
    printf("\nv1 OK\n");
    return 0;
}
