// capstone: 用 Copy_Atom 写一个高带宽 memcpy
//
// 讲解见本目录 README.md 的 §8。
//
// 四个版本，逐步加东西，看带宽怎么变：
//   v1 naive      : 1 线程 1 float，标量搬                         (baseline)
//   v2 vectorized : TiledCopy + 128bit atom                        (访存宽度)
//   v3 smem       : gmem -> smem -> gmem，用 cp.async 异步         (多一跳)
//   v4 pipelined  : double buffer，搬下一块的同时写出这一块         (延迟隐藏)
//
// 对照组: cudaMemcpy(DeviceToDevice)
//
// 结论预告: memcpy 是纯带宽型任务，v2 就能贴近硬件上限；
//           v3 多绕一跳 smem 反而更慢 —— 这正是本章想让你看到的。
//
// ---------------------------------------------------------------------------
// 阅读方式
//
// 每个版本都是「一个 kernel + 紧跟其后的一个 host 函数」，host 函数里自带这一版
// 需要的全部东西：缓冲区、Tensor、layout、TiledCopy、launch、验证、计时。
// 从上往下顺读即可，不需要跳到 main 里去找参数是怎么来的。
//
// main 只做两件事：按顺序叫这五个 run_*，然后汇总。
// ---------------------------------------------------------------------------
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_copy_capstone

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 全局配置
//
// BLOCK_ELEMS —— 一个 block 一轮搬多少个 float。名字里是 BLOCK 而不是 TILE,
//   因为它描述的是「block 级的工作量」, 不是「每线程多少个」。
//   每线程的份额要除一下:  BLOCK_ELEMS / NTHR = 4 个 float = 128 bit 一条指令。
// ---------------------------------------------------------------------------
constexpr int NTHR = 256;
constexpr int BLOCK_ELEMS = NTHR * 4;          // 一个 block 一轮搬 1024 个 float
constexpr int N = 64 * 1024 * 1024;            // 64M float = 256 MB
static_assert(N % BLOCK_ELEMS == 0, "取整除的尺寸, 省掉尾块处理");
constexpr size_t BYTES = size_t(N) * sizeof(float);

// ---------------------------------------------------------------------------
// 五个版本共用的缓冲区 —— 构造时分配并填好, 析构时释放
//
// 每个 run_* 开头写一行 `Buffers buf;` 就得到一套干净的 src/dst,
// 不需要从 main 传进来, 也不用担心上一版留下的脏数据。
// ---------------------------------------------------------------------------
struct Buffers {
    float* d_src;
    float* d_dst;
    float* h_src;
    float* h_dst;

    Buffers() {
        CUDA_CHECK(cudaMalloc(&d_src, BYTES));
        CUDA_CHECK(cudaMalloc(&d_dst, BYTES));
        h_src = new float[N];
        h_dst = new float[N];
        for (int i = 0; i < N; ++i) h_src[i] = float(i % 1000);
        CUDA_CHECK(cudaMemcpy(d_src, h_src, BYTES, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_dst, 0, BYTES));  // 没搬到的地方会留 0, 便于发现漏搬
    }

    ~Buffers() {
        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));
        delete[] h_src;
        delete[] h_dst;
    }

    // 拷回来逐个比对
    bool check() {
        CUDA_CHECK(cudaMemcpy(h_dst, d_dst, BYTES, cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i)
            if (h_dst[i] != h_src[i]) return false;
        return true;
    }
};

struct Result {
    const char* name;
    float ms;
    double gbs;
    bool ok;
};

// 每版结尾都是这一句: 算带宽、打印、打包成 Result
static Result report(const char* name, int grid, float ms, bool ok) {
    double gbs = copy_bandwidth_gbs(BYTES, ms);
    printf("  grid=%-6d %.3f ms   %.0f GB/s   %s\n", grid, ms, gbs, ok ? "正确" : "错误");
    return {name, ms, gbs, ok};
}

// ===========================================================================
// 0. 对照组: cudaMemcpy D2D
//
// 厂商实现的下限参考。它是纯 DRAM 到 DRAM, 不经过我们能写的任何代码,
// 所以它的数字基本就是这台机器的 memcpy 上限。
// ===========================================================================
static Result run_cudamemcpy() {
    print_separator("0. 对照组: cudaMemcpy D2D");

    Buffers buf;
    float ms = time_kernel(
        [&] { CUDA_CHECK(cudaMemcpy(buf.d_dst, buf.d_src, BYTES, cudaMemcpyDeviceToDevice)); });
    return report("cudaMemcpy D2D", 0, ms, buf.check());
}

// ===========================================================================
// v1: naive —— 每个线程一个 float, grid-stride loop
//
// 这一版故意不用 CuTe, 用最朴素的写法作 baseline。
// 每线程一次只搬 1 个 float = 一条 32 bit 指令, 指令发射数是 v2 的 4 倍。
// ===========================================================================
__global__ void memcpy_naive(const float* __restrict__ src, float* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) dst[i] = src[i];
}

static Result run_naive() {
    print_separator("1. naive: 1 线程 1 float");

    Buffers buf;
    int grid = 1024;
    float ms = time_kernel([&] { memcpy_naive<<<grid, NTHR>>>(buf.d_src, buf.d_dst, N); });
    return report("naive (scalar)", grid, ms, buf.check());
}

// ===========================================================================
// v2: TiledCopy + 128 bit atom
//
// kernel 只剩两个动作 (README §4 的分工):
//   local_tile(mS, block_shape, blockIdx.x)  -> 我这个 block 负责哪一段
//   get_slice(threadIdx.x).partition_S/D     -> 我这个线程负责哪 4 个
//
// Tensor 和 TiledCopy 都是下面 run_vectorized() 里现构造、直接传进来的。
// ===========================================================================
template <class TensorS, class TensorD, class TiledCopy>
__global__ void memcpy_vectorized(TensorS mS, TensorD mD, TiledCopy tc) {
    auto block_shape = Shape<Int<BLOCK_ELEMS>>{};

    auto gS = local_tile(mS, block_shape, make_coord(blockIdx.x));  // (BLOCK_ELEMS)
    auto gD = local_tile(mD, block_shape, make_coord(blockIdx.x));

    auto thr = tc.get_slice(threadIdx.x);
    copy(tc, thr.partition_S(gS), thr.partition_D(gD));
}

static Result run_vectorized() {
    print_separator("2. TiledCopy + 128bit atom");

    Buffers buf;

    // 全局 Tensor: shape 是运行时的 N, 但 stride 钉成 Int<1>{} ——
    // 向量化只需要连续维的 stride 是静态的 (README §7.3)。
    auto mS = make_tensor(make_gmem_ptr(buf.d_src),
                          make_layout(make_shape(N), make_stride(Int<1>{})));
    auto mD = make_tensor(make_gmem_ptr(buf.d_dst),
                          make_layout(make_shape(N), make_stride(Int<1>{})));

    // 256 个线程, 每人 4 个连续 float -> 刚好一条 128 bit 指令
    auto tc = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                              make_layout(Int<NTHR>{}, Int<1>{}),
                              make_layout(Int<BLOCK_ELEMS / NTHR>{}, Int<1>{}));

    printf("  mS = ");
    print(mS.layout());
    printf("   Tiler_MN = ");
    print(typename decltype(tc)::Tiler_MN{});
    printf("   线程数 = %d\n", int(size(tc)));

    int grid = N / BLOCK_ELEMS;
    float ms = time_kernel([&] { memcpy_vectorized<<<grid, NTHR>>>(mS, mD, tc); });
    return report("TiledCopy 128bit", grid, ms, buf.check());
}

// ===========================================================================
// v3: gmem -> smem -> gmem, 用 cp.async
//
// cp.async 是「发出去就不管了」的异步搬运: 数据由硬件直接从 gmem 写进 smem,
// 不经过寄存器, 也不占线程的指令流。线程发完可以立刻去干别的。
//
// 三个控制原语:
//   cp_async_fence()    给「此前发出的所有 cp.async」打一道围栏
//   cp_async_wait<K>()  等到「还在飞的围栏数 <= K」为止
//   cp_async_wait<0>()  等全部落地
//
// v3 的流程是  发 load -> fence -> wait<0> -> 发 store,
// 也就是发完立刻就等 —— 等待期间无事可做, cp.async 的异步性完全没用上。
// **这一版的意义就是把这个浪费暴露出来**, v4 才会把「等」和「发下一块」重叠。
//
// smem layout 同样由 host 传进来 (对应 CUTLASS sgemm_sm80 的 sA_layout)。
// smem 数组按 cosize_v 开 —— 用 size 在带 padding 的 layout 上会溢出。
// ===========================================================================
template <class TensorS, class TensorD, class SLayout, class TCLoad, class TCStore>
__global__ void memcpy_smem(TensorS mS, TensorD mD, SLayout slay, TCLoad tc_load,
                            TCStore tc_store) {
    __shared__ float raw[cosize_v<SLayout>];
    auto sT = make_tensor(make_smem_ptr(raw), slay);

    auto block_shape = Shape<Int<BLOCK_ELEMS>>{};
    auto gS = local_tile(mS, block_shape, make_coord(blockIdx.x));
    auto gD = local_tile(mD, block_shape, make_coord(blockIdx.x));

    // gmem -> smem: 异步
    auto thr_load = tc_load.get_slice(threadIdx.x);
    copy(tc_load, thr_load.partition_S(gS), thr_load.partition_D(sT));

    cp_async_fence();    // 打围栏
    cp_async_wait<0>();  // 立刻等它落地 —— 这里就是被浪费的地方
    __syncthreads();     // smem 对全 block 可见

    // smem -> gmem: 普通向量 atom
    auto thr_store = tc_store.get_slice(threadIdx.x);
    copy(tc_store, thr_store.partition_S(sT), thr_store.partition_D(gD));
}

static Result run_smem() {
    print_separator("3. gmem -> smem -> gmem (cp.async)");

    Buffers buf;

    auto mS = make_tensor(make_gmem_ptr(buf.d_src),
                          make_layout(make_shape(N), make_stride(Int<1>{})));
    auto mD = make_tensor(make_gmem_ptr(buf.d_dst),
                          make_layout(make_shape(N), make_stride(Int<1>{})));

    auto thr_lay = make_layout(Int<NTHR>{}, Int<1>{});
    auto val_lay = make_layout(Int<BLOCK_ELEMS / NTHR>{}, Int<1>{});

    // load 用 cp.async atom, store 用普通向量 atom —— 同一套 thr/val 分工
    auto tc_load = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>{},
                                   thr_lay, val_lay);
    auto tc_store = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{}, thr_lay, val_lay);

    auto slay = make_layout(Int<BLOCK_ELEMS>{}, Int<1>{});  // 单缓冲

    printf("  smem layout = ");
    print(slay);
    printf("   cosize = %d float\n", int(cosize(slay)));

    int grid = N / BLOCK_ELEMS;
    float ms =
        time_kernel([&] { memcpy_smem<<<grid, NTHR>>>(mS, mD, slay, tc_load, tc_store); });
    return report("smem staging", grid, ms, buf.check());
}

// ===========================================================================
// v4: double buffer —— 把「等这一块」和「发下一块」重叠起来
//
// v3 的问题是发完立刻就等。要重叠, 需要两件东西:
//   1) 两块 smem, 一块在收数据的同时另一块在被写出   -> layout 加一个 PIPE 维
//   2) 一个「等旧的、但别等新的」的等待原语           -> cp_async_wait<1>
//
// 每个 block 负责 ROUNDS 轮, 每轮搬 BLOCK_ELEMS 个 float。时间轴:
//
//   预填:  发 round 0 的 load  (buf 0)
//   r=0 :  发 round 1 的 load (buf 1) | 等 round 0 | 写出 round 0 (buf 0)
//   r=1 :  发 round 2 的 load (buf 0) | 等 round 1 | 写出 round 1 (buf 1)
//   r=2 :  发 round 3 的 load (buf 1) | 等 round 2 | 写出 round 2 (buf 0)
//          ~~~~~~~~~~~~~~~~~~~~~~~~~    ~~~~~~~~~~
//          这两件事在时间上是重叠的: 等的时候, 下一块已经在飞了
//
// 「预填」不是 benchmark 的预热, 而是流水线的**装填**: 循环体的形状是
// 「先发下一块, 再等这一块」, 所以进循环之前必须already有一块在飞,
// 否则第一次的「等这一块」无从可等。Section 06 的多 stage 流水线是同一个结构,
// 那里要预填 PIPE-1 块。
//
// cp_async_wait<K> 的 K = 允许还有几道围栏在飞:
//   有下一块 -> wait<1>: 放过刚发的那道, 只等前一道
//   没有了   -> wait<0>: 全部等干净
// ===========================================================================
template <int ROUNDS, class TensorS, class TensorD, class SLayout, class TCLoad, class TCStore>
__global__ void memcpy_pipelined(TensorS mS, TensorD mD, SLayout slay, TCLoad tc_load,
                                 TCStore tc_store) {
    __shared__ float raw[cosize_v<SLayout>];
    auto sT = make_tensor(make_smem_ptr(raw), slay);  // (BLOCK_ELEMS, 2)

    auto thr_load = tc_load.get_slice(threadIdx.x);
    auto thr_store = tc_store.get_slice(threadIdx.x);

    auto block_shape = Shape<Int<BLOCK_ELEMS>>{};
    int base_round = blockIdx.x * ROUNDS;                    // 我负责的第一轮的编号
    int total_rounds = size<0>(shape(mS)) / BLOCK_ELEMS;     // 全局一共几轮

    // 发起第 r 轮的 gmem->smem, 落到第 buf 块 smem。
    // 捕获用 [&]: 这是 CUTLASS mainloop 里的常规写法, 捕的都是寄存器变量和
    // 空类型 (TiledCopy sizeof==1), 不产生任何实际的间接开销。
    auto issue_load = [&](int r, int buf) {
        if (base_round + r >= total_rounds) return;
        auto gS = local_tile(mS, block_shape, make_coord(base_round + r));
        copy(tc_load, thr_load.partition_S(gS), thr_load.partition_D(sT(_, buf)));
        cp_async_fence();
    };

    issue_load(0, 0);  // 装填流水线: 先让第 0 轮飞起来

    for (int r = 0; r < ROUNDS; ++r) {
        if (base_round + r >= total_rounds) break;
        int buf = r & 1;  // 0,1,0,1,... 两块 smem 交替

        // 先把下一轮发出去, 此时这一轮还在飞 —— 重叠就发生在这里
        bool has_next = (r + 1 < ROUNDS) && (base_round + r + 1 < total_rounds);
        if (has_next) issue_load(r + 1, buf ^ 1);

        if (has_next)
            cp_async_wait<1>();  // 放过刚发的那道围栏, 只等这一轮
        else
            cp_async_wait<0>();  // 最后一轮, 等干净
        __syncthreads();

        auto gD = local_tile(mD, block_shape, make_coord(base_round + r));
        copy(tc_store, thr_store.partition_S(sT(_, buf)), thr_store.partition_D(gD));

        __syncthreads();  // 写完才能让 r+2 轮复用这块 smem
    }
}

static Result run_pipelined() {
    print_separator("4. double buffer 流水线");

    constexpr int ROUNDS = 4;  // 每个 block 处理 4 轮

    Buffers buf;

    auto mS = make_tensor(make_gmem_ptr(buf.d_src),
                          make_layout(make_shape(N), make_stride(Int<1>{})));
    auto mD = make_tensor(make_gmem_ptr(buf.d_dst),
                          make_layout(make_shape(N), make_stride(Int<1>{})));

    auto thr_lay = make_layout(Int<NTHR>{}, Int<1>{});
    auto val_lay = make_layout(Int<BLOCK_ELEMS / NTHR>{}, Int<1>{});

    auto tc_load = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>{},
                                   thr_lay, val_lay);
    auto tc_store = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{}, thr_lay, val_lay);

    // 双缓冲: 第二维就是 PIPE。kernel 里用 sT(_, buf) 取其中一块。
    auto slay = make_layout(make_shape(Int<BLOCK_ELEMS>{}, Int<2>{}));

    printf("  smem layout = ");
    print(slay);
    printf("   cosize = %d float (两块)   每 block %d 轮\n", int(cosize(slay)), ROUNDS);

    int grid = (N / BLOCK_ELEMS + ROUNDS - 1) / ROUNDS;
    float ms = time_kernel(
        [&] { memcpy_pipelined<ROUNDS><<<grid, NTHR>>>(mS, mD, slay, tc_load, tc_store); });
    return report("double buffer", grid, ms, buf.check());
}

// ===========================================================================
// main —— 按顺序跑五版, 汇总
// ===========================================================================
int main() {
    print_separator("Capstone: 用 Copy_Atom 写高带宽 memcpy");
    printf("数据量: %d float = %.0f MB (读+写 = %.0f MB)\n", N, BYTES / 1e6, 2 * BYTES / 1e6);
    printf("配置:   NTHR=%d  BLOCK_ELEMS=%d  ->  每线程 %d 个 float = %d bit\n", NTHR, BLOCK_ELEMS,
           BLOCK_ELEMS / NTHR, BLOCK_ELEMS / NTHR * 32);

    Result results[] = {
        run_cudamemcpy(), run_naive(), run_vectorized(), run_smem(), run_pipelined(),
    };
    constexpr int NRES = sizeof(results) / sizeof(results[0]);

    print_separator("汇总");
    printf("%-20s %10s %12s %7s\n", "version", "time(ms)", "GB/s", "ok");
    for (auto& r : results)
        printf("%-20s %10.3f %12.0f %7s\n", r.name, r.ms, r.gbs, r.ok ? "yes" : "NO");

    printf("\n相对 cudaMemcpy 的比例:\n");
    for (int i = 1; i < NRES; ++i)
        printf("  %-20s %.2fx\n", results[i].name, results[i].gbs / results[0].gbs);

    return 0;
}
