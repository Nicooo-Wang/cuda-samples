// cute_04 v1 —— 用 CuTe 语义做 Copy: 加了 swizzle, 搬运代码要改什么?
//
// 对应 README §4。
//
// 这个文件回答两个问题:
//   1) 我有一段能跑的搬运代码, 要加 swizzle, 得改哪几行?   -> 一行都不用改
//   2) gmem->smem 用了 copy(), smem->gmem 能不能也用 copy()? -> 能, 见 v1c
//
// ---------------------------------------------------------------------------
// 阅读方式
//
// 每一版都是「一个 kernel + 紧跟其后的一个 host 函数」, host 函数里自带这一版
// 需要的全部东西: 缓冲区、layout、launch、验证、计时。从上往下顺读即可。
//
//   v1a  两个方向都是裸下标         <- 起点
//   v1b  load 用 copy(), store 裸写 <- 半途而废: 语义不一致
//   v1c  两个方向都用 copy()        <- 目标写法 (转置视图)
//   v1d  反例: 把转置塞进 gmem      <- 正确但慢 6 倍, 说明"合并"优先于"消冲突"
//
// ---------------------------------------------------------------------------
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_v1

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 尺寸: gmem 是 M x N 的 float, row-major, stride = (N, 1)
//       一个 block 负责一个 TILE x TILE 的方块, 用 NTHR 个线程搬
// ---------------------------------------------------------------------------
constexpr int M = 4096, N = 4096;  // 整个矩阵 (4096x4096 float = 64MB)
constexpr int TILE = 32;           // smem 方块边长 (32x32 float = 4KB)
constexpr int NTHR = 256;          // 每 block 线程数 -> 每线程搬 1024/256 = 4 个

static dim3 grid() { return dim3(N / TILE, M / TILE); }  // (128, 128)

// ---------------------------------------------------------------------------
// 四个版本共用的缓冲区 —— 构造时分配并填好, 析构时释放
// 每个 run_* 开头写一行 `Buffers buf;` 就得到一套干净的 in/out
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

// ---------------------------------------------------------------------------
// 指标: 全列扫描取最坏 (只看第 0 列会把 Sw<3,2,3> 误判成 1-way)
// ---------------------------------------------------------------------------
template <class Lay>
static int worst_col_conflict(Lay lay) {
    int worst = 0;
    for (int c = 0; c < TILE; ++c) {
        int w = max_bank_requests(32, [&](int l) { return int(lay(l, c)) * 4; });
        if (w > worst) worst = w;
    }
    return worst;
}

// 行内最短连续段: 决定能不能用宽向量指令
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

template <class Lay>
static void report(Lay slay, float ms, bool ok) {
    printf("    layout = ");
    print(slay);
    printf("\n");
    printf("    列读 = %2d-way   行内最短连续 = %2d 个 float   cosize = %d\n",
           worst_col_conflict(slay), min_run(slay), int(cosize(slay)));
    printf("    %.3f ms   %.1f GB/s   %s\n", ms,
           transpose_bandwidth_gbs(size_t(M) * N, sizeof(float), ms), ok ? "正确" : "错误");
}

// 本文件统一用这两个 layout
static auto plain_layout() {
    return make_layout(make_shape(Int<TILE>{}, Int<TILE>{}), make_stride(Int<TILE>{}, Int<1>{}));
}

// ===========================================================================
// v1a  两个方向都是裸下标 —— 起点
//
// 转置的三步: gmem 按行读 -> smem -> 按列读 smem -> gmem 按行写。
// 两次 gmem 访问都是合并的, 代价是中间"按列读 smem"那一步撞 bank。
// ===========================================================================
template <class SLay>
__global__ void transpose_raw(const float* __restrict__ in, float* __restrict__ out, SLay slay) {
    __shared__ __align__(128) float raw[cosize_v<SLay>];
    auto s = make_tensor(make_smem_ptr(raw), slay);

    int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
    int tx = threadIdx.x % TILE, ty = threadIdx.x / TILE;

    // 写 smem: 按行走, 同一 warp 的 32 个 lane 有连续的 tx -> gmem 合并读
#pragma unroll
    for (int r = ty; r < TILE; r += NTHR / TILE) s(r, tx) = in[size_t(by + r) * N + bx + tx];

    __syncthreads();

    // 读 smem: s(tx, r) 第一个下标是 tx -> 一个 warp 扫过一整列, 冲突全在这里
#pragma unroll
    for (int r = ty; r < TILE; r += NTHR / TILE) out[size_t(bx + r) * M + by + tx] = s(tx, r);
}

static float run_raw() {
    printf("\nv1a  两个方向都是裸下标 (起点)\n");
    printf("    写: s(r, tx) = in[...]        读: out[...] = s(tx, r)\n");

    auto slay = composition(Swizzle<3, 2, 3>{}, plain_layout());

    Buffers buf;
    float ms = time_kernel([&] { transpose_raw<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay); });
    report(slay, ms, buf.check());
    return ms;
}

// ===========================================================================
// v1b  load 用 copy(), store 还是裸写 —— 语义不一致
//
// gmem->smem 交给 TiledCopy 之后, smem->gmem 却还是手算下标。
// 能跑, 但两个方向风格不统一: 一边是"声明分工表", 一边是"手算地址"。
// ===========================================================================
template <class SLay, class TC>
__global__ void transpose_half_cute(const float* __restrict__ in, float* __restrict__ out,
                                    SLay slay, TC tc) {
    __shared__ __align__(128) float raw[cosize_v<SLay>];
    auto s = make_tensor(make_smem_ptr(raw), slay);

    // ---- load: 用 CuTe 语义 ----
    auto mIn = make_tensor(make_gmem_ptr(in),
                           make_layout(make_shape(Int<M>{}, Int<N>{}),
                                       make_stride(Int<N>{}, Int<1>{})));
    auto gIn = local_tile(mIn, Shape<Int<TILE>, Int<TILE>>{}, make_coord(blockIdx.y, blockIdx.x));

    auto thr = tc.get_slice(threadIdx.x);
    copy(tc, thr.partition_S(gIn), thr.partition_D(s));

    __syncthreads();

    // ---- store: 退回裸下标 ----   <- 就是这里不一致
    int tx = threadIdx.x % TILE, ty = threadIdx.x / TILE;
#pragma unroll
    for (int r = ty; r < TILE; r += NTHR / TILE)
        out[size_t(blockIdx.x * TILE + r) * M + blockIdx.y * TILE + tx] = s(tx, r);
}

// 标量 TiledCopy: 8x32 个线程, 每线程 4 个连续 float
static auto make_scalar_copy() {
    return make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{},
                           make_layout(make_shape(Int<8>{}, Int<32>{}),
                                       make_stride(Int<32>{}, Int<1>{})),
                           make_layout(make_shape(Int<4>{}, Int<1>{})));
}

static float run_half_cute() {
    printf("\nv1b  load 用 copy(), store 仍是裸下标 (语义不一致)\n");
    printf("    写: copy(tc, partition_S(gIn), partition_D(s))\n");
    printf("    读: out[...] = s(tx, r)      <- 还是手算地址\n");

    auto slay = composition(Swizzle<3, 2, 3>{}, plain_layout());
    auto tc = make_scalar_copy();

    Buffers buf;
    float ms = time_kernel(
        [&] { transpose_half_cute<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay, tc); });
    report(slay, ms, buf.check());
    return ms;
}

// ===========================================================================
// v1c  两个方向都用 copy() —— 目标写法
//
// 关键是 store 侧需要一个"转置视图" sT, 满足 sT(m,n) == s(n,m)。
// swizzled layout 不能靠交换 mode 来转置, 但可以 composition 一个转置 layout:
//
//   sT_lay = composition(slay, (TILE,TILE):(TILE,1))
//
// 原理: 给 slay 喂线性下标 k = m*TILE + n, 它按自己的 shape 反解成
//       (row,col) = (k%TILE, k/TILE) = (n, m), 于是拿到 slay(n,m)。
//
//   slay   = Sw<3,2,3> o (_32,_32):(_32,_1)
//   sT_lay = Sw<3,2,3> o (_32,_32):(_1,_32)     <- swizzle 保留, 只是 stride 换了
//
// 于是 "读 sT 的行" == "读 s 的列", 冲突还在原来的地方, 但代码是 copy()。
// ===========================================================================
template <class SLay, class STLay, class TC>
__global__ void transpose_full_cute(const float* __restrict__ in, float* __restrict__ out,
                                    SLay slay, STLay stlay, TC tc) {
    __shared__ __align__(128) float raw[cosize_v<SLay>];
    auto s = make_tensor(make_smem_ptr(raw), slay);    // (TILE,TILE)  正常视图, 写用
    auto sT = make_tensor(make_smem_ptr(raw), stlay);  // (TILE,TILE)  转置视图, 读用
                                                      // 同一块 raw, 只是两套映射

    auto thr = tc.get_slice(threadIdx.x);

    // ---- load: gmem 第 (by,bx) 块 -> smem ----
    auto mIn = make_tensor(make_gmem_ptr(in),
                           make_layout(make_shape(Int<M>{}, Int<N>{}),
                                       make_stride(Int<N>{}, Int<1>{})));
    auto gIn = local_tile(mIn, Shape<Int<TILE>, Int<TILE>>{}, make_coord(blockIdx.y, blockIdx.x));
    copy(tc, thr.partition_S(gIn), thr.partition_D(s));

    __syncthreads();

    // ---- store: smem 转置视图 -> gmem 第 (bx,by) 块 ----
    // 输出矩阵是 N x M, row-major。写 gmem 仍然按行 -> 合并写。
    auto mOut = make_tensor(make_gmem_ptr(out),
                            make_layout(make_shape(Int<N>{}, Int<M>{}),
                                        make_stride(Int<M>{}, Int<1>{})));
    auto gOut = local_tile(mOut, Shape<Int<TILE>, Int<TILE>>{}, make_coord(blockIdx.x, blockIdx.y));
    copy(tc, thr.partition_S(sT), thr.partition_D(gOut));
}

template <class Swz>
static float run_full_cute(const char* tag, Swz swz, bool verbose) {
    printf("\nv1c  两个方向都用 copy()   [%s]\n", tag);
    if (verbose)
        printf("    写: copy(tc, partition_S(gIn), partition_D(s))\n"
               "    读: copy(tc, partition_S(sT),  partition_D(gOut))   <- sT 是转置视图\n");

    auto slay = composition(swz, plain_layout());
    // 转置视图: 把一个 (TILE,TILE):(TILE,1) 复合进去
    auto stlay = composition(slay, make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                                               make_stride(Int<TILE>{}, Int<1>{})));
    auto tc = make_scalar_copy();

    printf("    正常视图 = ");
    print(slay);
    printf("\n    转置视图 = ");
    print(stlay);
    printf("\n");

    Buffers buf;
    float ms = time_kernel(
        [&] { transpose_full_cute<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay, stlay, tc); });
    // 指标看转置视图的行读 —— 那才是真正的 smem 列访问
    printf("    读 sT 的行 (= 读 s 的列) = %d-way\n",
           max_bank_requests(32, [&](int l) { return int(stlay(0, l)) * 4; }));
    report(slay, ms, buf.check());
    return ms;
}

// ===========================================================================
// v1d  反例: 把转置塞进 gmem 的 stride
//
// 既然要转置, 一个更"省事"的想法: 不建转置视图, 直接让 gmem 目的 layout
// 用 (1, N) 而不是 (N, 1), 转置就自动完成了:
//
//   auto gT = make_tensor(ptr + offset, make_layout(shape, make_stride(Int<1>{}, Int<N>{})));
//
// 结果**正确, 但慢 6 倍**, 而且 swizzle 完全失效 —— 因为 gmem 写变成了非合并。
// 这一版存在的唯一目的就是量化这件事。
// ===========================================================================
template <class SLay, class TC>
__global__ void transpose_bad_gmem(const float* __restrict__ in, float* __restrict__ out,
                                   SLay slay, TC tc) {
    __shared__ __align__(128) float raw[cosize_v<SLay>];
    auto s = make_tensor(make_smem_ptr(raw), slay);
    auto thr = tc.get_slice(threadIdx.x);

    auto mIn = make_tensor(make_gmem_ptr(in),
                           make_layout(make_shape(Int<M>{}, Int<N>{}),
                                       make_stride(Int<N>{}, Int<1>{})));
    auto gIn = local_tile(mIn, Shape<Int<TILE>, Int<TILE>>{}, make_coord(blockIdx.y, blockIdx.x));
    copy(tc, thr.partition_S(gIn), thr.partition_D(s));

    __syncthreads();

    // 转置放在 gmem 侧: stride (1, N) —— 相邻 lane 写的地址相隔 N 个 float
    auto gT = make_tensor(make_gmem_ptr(out + size_t(blockIdx.x * TILE) * N + blockIdx.y * TILE),
                          make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                                      make_stride(Int<1>{}, Int<N>{})));
    copy(tc, thr.partition_S(s), thr.partition_D(gT));
}

template <class Swz>
static float run_bad_gmem(const char* tag, Swz swz) {
    printf("\nv1d  反例: 转置塞进 gmem stride (1,N)   [%s]\n", tag);

    auto slay = composition(swz, plain_layout());
    auto tc = make_scalar_copy();

    Buffers buf;
    float ms = time_kernel(
        [&] { transpose_bad_gmem<<<grid(), NTHR>>>(buf.d_in, buf.d_out, slay, tc); });
    report(slay, ms, buf.check());
    return ms;
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    printf("cute_04 v1 —— 用 CuTe 语义做 Copy\n");
    printf("对应 README §4    gmem = %dx%d float row-major stride=(%d,1)\n", M, N, N);
    printf("                  一个 block 搬 %dx%d, 每 block %d 线程 (每线程 %d 个 float)\n",
           TILE, TILE, NTHR, TILE * TILE / NTHR);

    print_separator("第一部分: 从裸下标走到全 CuTe 语义");
    float a = run_raw();
    float b = run_half_cute();
    float c = run_full_cute("Swizzle<3,2,3>", Swizzle<3, 2, 3>{}, true);

    printf("\n  三版做的是同一件事, 区别只在\"怎么表达搬运\":\n");
    printf("    v1a  裸 / 裸        %.3f ms\n", a);
    printf("    v1b  copy / 裸      %.3f ms\n", b);
    printf("    v1c  copy / copy    %.3f ms   <- 语义统一, 而且最快\n", c);
    printf("\n  store 侧能用 copy() 的关键是转置视图:\n");
    printf("    composition(slay, (TILE,TILE):(TILE,1))\n");
    printf("  swizzle 被保留下来, 只是 stride 从 (32,1) 变成 (1,32)。\n");

    print_separator("第二部分: swizzle 在全 CuTe 版里还起作用吗");
    float d = run_full_cute("无 swizzle (Sw<0,2,3>)", Swizzle<0, 2, 3>{}, false);
    printf("\n  加速 (相对无 swizzle): Sw<3,2,3> %.2fx\n", d / c);
    printf("  起作用 —— 转置视图把冲突留在了原来的位置, swizzle 照样消它。\n");

    print_separator("第三部分: 反例 —— 把转置塞进 gmem stride");
    float e = run_bad_gmem("无 swizzle (Sw<0,2,3>)", Swizzle<0, 2, 3>{});
    float f = run_bad_gmem("Swizzle<3,2,3>", Swizzle<3, 2, 3>{});

    printf("\n  两组数字放一起看:\n");
    printf("    %-34s %8s %8s %8s\n", "写法", "无swz", "Sw323", "swz 收益");
    printf("    %-34s %7.3f  %7.3f  %7.2fx\n", "v1c 转置在 smem 侧 (gmem 合并)", d, c, d / c);
    printf("    %-34s %7.3f  %7.3f  %7.2fx\n", "v1d 转置在 gmem 侧 (gmem 不合并)", e, f, e / f);
    printf("\n  v1d 里 swizzle 收益基本归零, 而且整体慢好几倍 —— 非合并的 gmem 写\n");
    printf("  成了唯一瓶颈, bank 冲突根本排不上号。\n");
    printf("\n  优化顺序: 先保证 gmem 合并, 再谈 smem 消冲突。\n");
    printf("  量级差太多, 顺序错了后面全白做。\n");

    print_separator("小结: 加 swizzle 要改什么");
    printf("  逻辑代码            : 一行都不用改。s(r,c) 自动走 swizzle 后的映射。\n");
    printf("  host 侧 layout 声明 : 加一层 composition(Swizzle<B,M,S>{}, plain)。\n");
    printf("  smem 数组大小       : 写 cosize_v<SLay>, 不要写 size。\n");
    printf("  对齐                : __align__(128), TMA 硬性要求。\n");
    printf("  想用 copy() 读 smem : 建一个转置视图 composition(slay, (T,T):(T,1))。\n");
    printf("  用了宽向量 atom     : M 必须 >= log2(向量元素数), 否则编译失败。\n");
    printf("\n  一句话: swizzle 是 layout 的属性, 不是搬运代码的属性。\n");

    printf("\n下一步 (§5): 到这里 gmem->smem 还是\"每个线程算自己的地址\"。\n");
    printf("SM90 有专门的搬运硬件 TMA —— 一个线程描述整块, 硬件自己搬。v2 讲它。\n");
    printf("\nv1 OK\n");
    return 0;
}
