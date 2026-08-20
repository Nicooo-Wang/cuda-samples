// cute_04 v2 —— TMA store 与边界: 把搬回去那一步也交给硬件
//
// 对应 README §3。
//
// v1 只换了 gmem->smem。搬回 gmem 那一步还是 v0 的手写循环。这一版把它也
// 换成 TMA, 于是整条通路两端都是硬件搬运。
//
//   §3.1  TMA store: smem -> gmem          <- 和 load 反过来, 同步机制也不同
//   §3.2  两端都用 TMA: 完整的一趟          <- load + store 串起来
//   §3.3  边界: tile 不整除时硬件自己兜      <- v0 要写 predicate, TMA 不用
//
// ---------------------------------------------------------------------------
// load 和 store 最重要的区别: 同步方向反了
//
//   TMA load    gmem -> smem    数据是**搬完之后**才能用   -> 事后等: mbarrier
//   TMA store   smem -> gmem    数据要**搬之前**就写好     -> 事前挡: fence
//
//   load :   发起 --------> [硬件搬] --------> wait_barrier --> 读 smem
//                                                ^^^^^^^^ 等它搬完
//
//   store:   写 smem --> fence --> 发起 --------> [硬件搬] --> (可选 wait)
//                        ^^^^^ 保证前面的写对硬件可见
//
// 这就是为什么 store 那段代码里看不到 mbarrier。
//
// ---------------------------------------------------------------------------
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_v2

#include <cute/tensor.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 和 v0/v1 相同的尺寸 (§3.3 会另用一组不整除的尺寸)
// ---------------------------------------------------------------------------
constexpr int M = 256, N = 128;
constexpr int CM = 32, CN = 32;
constexpr int NTHR = 128;

static dim3 grid() { return dim3(M / CM, N / CN); }

struct Buffers {
    static constexpr size_t elems = size_t(M) * N;
    static constexpr size_t bytes = elems * sizeof(float);

    float* d_in;
    float* d_out;
    float* h_in;
    float* h_out;

    Buffers() {
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        h_in = new float[elems];
        h_out = new float[elems];
        for (size_t i = 0; i < elems; ++i) h_in[i] = float(i);
        CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    }

    ~Buffers() {
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        delete[] h_in;
        delete[] h_out;
    }

    bool check() {
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < elems; ++i)
            if (h_out[i] != h_in[i]) return false;
        return true;
    }
};

// ===========================================================================
// §3.1  TMA store: smem -> gmem
//
// 为了把 store 单独看清楚, 这个 kernel 不做 load —— 直接在 smem 里造数据,
// 然后 TMA store 出去。三个新东西:
//
//   tma_store_fence()     保证前面对 smem 的普通写, 对 TMA 硬件可见
//   copy(tma, S, D)       注意没有 .with(bar) —— store 不用 barrier
//   tma_store_arrive/wait 提交 + 等待 (只在还要复用 smem 时才需要)
//
// 方向也反了: partition_S 作用在 smem 上, partition_D 作用在 gmem 坐标上。
// ===========================================================================
template <class TmaStore>
__global__ static void store_only_kernel(__grid_constant__ const TmaStore tma) {
    __shared__ __align__(128) float smem[CM * CN];

    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto sT = make_tensor(make_smem_ptr(smem), slay);

    // 在 smem 里造出"这个位置在全局矩阵里的线性下标"
    const int row0 = blockIdx.x * CM, col0 = blockIdx.y * CN;
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x)
        sT(i / CN, i % CN) = float((row0 + i / CN) * N + (col0 + i % CN));

    __syncthreads();     // 1) 等全 CTA 写完 smem (普通的线程间同步)
    tma_store_fence();   // 2) 再保证这些写对 TMA 硬件 (异步 proxy) 可见
                         //    少了这一行是**竞态**: 可能搬出去半新半旧的数据

    if (threadIdx.x == 0) {
        auto gcoord = tma.get_tma_tensor(make_shape(Int<M>{}, Int<N>{}));
        auto gtile =
            local_tile(gcoord, Shape<Int<CM>, Int<CN>>{}, make_coord(blockIdx.x, blockIdx.y));
        auto per_cta = tma.get_slice(0);

        // 注意两点: 没有 .with(bar); partition_S 是 smem, partition_D 是 gmem
        copy(tma, per_cta.partition_S(sT), per_cta.partition_D(gtile));
        tma_store_arrive();  // 提交这一批 store
    }
    tma_store_wait<0>();  // 等到 0 个未完成 —— 本 kernel 之后不再用 smem,
                          // 其实可以省; 但要复用 smem 就必须有 (v3 会用到)
}

static void section31_tma_store() {
    print_separator("§3.1  TMA store: smem -> gmem");

    Buffers buf;

    // host 侧: 和 load 一模一样的两步, 只是把 SM90_TMA_LOAD 换成 SM90_TMA_STORE
    auto mOut = make_tensor(make_gmem_ptr(buf.d_out),
                            make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto tma_store = make_tma_copy(SM90_TMA_STORE{}, mOut, slay);
    //                             ^^^^^^^^^^^^^^ 唯一的区别

    store_only_kernel<<<grid(), NTHR>>>(tma_store);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("    kernel 在 smem 里造 out[i] = i, 再 TMA store 出去\n");
    printf("    结果 = %s\n", buf.check() ? "正确" : "错误");

    printf("\n  load 和 store 的三点差别:\n\n");
    printf("                  TMA load                   TMA store\n");
    printf("    ------------  -------------------------  -------------------------\n");
    printf("    方向          gmem -> smem               smem -> gmem\n");
    printf("    同步机制      mbarrier (等字节数)        fence (挡在发起之前)\n");
    printf("    同步时机      搬完之后等                 发起之前挡\n");
    printf("    copy 写法     copy(tma.with(bar), S, D)  copy(tma, S, D)\n");
    printf("    partition_S   gmem 坐标                  smem\n");
    printf("    partition_D   smem                       gmem 坐标\n");
}

// ===========================================================================
// §3.2  两端都用 TMA: 完整的一趟
//
// 把 §2.1 的 load 和 §3.1 的 store 串起来。整个 kernel 里:
//   - 搬运指令一共两条, 都由 thread 0 发
//   - 其余 127 个线程只负责在中间"读一下 smem" (这里做一个平方, 代表计算)
// ===========================================================================
template <class TmaLoad, class TmaStore>
__global__ static void load_store_kernel(__grid_constant__ const TmaLoad tma_load,
                                         __grid_constant__ const TmaStore tma_store) {
    constexpr int tx_bytes = CM * CN * sizeof(float);
    __shared__ __align__(128) float smem[CM * CN];
    __shared__ uint64_t bar;

    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto sT = make_tensor(make_smem_ptr(smem), slay);
    auto blk = make_coord(blockIdx.x, blockIdx.y);

    if (threadIdx.x == 0) initialize_barrier(bar, 1);
    __syncthreads();

    // ---- load: gmem -> smem ----
    if (threadIdx.x == 0) {
        set_barrier_transaction_bytes(bar, tx_bytes);
        auto gc = tma_load.get_tma_tensor(make_shape(Int<M>{}, Int<N>{}));
        auto gt = local_tile(gc, Shape<Int<CM>, Int<CN>>{}, blk);
        auto per = tma_load.get_slice(0);
        copy(tma_load.with(bar), per.partition_S(gt), per.partition_D(sT));
    }
    __syncthreads();
    wait_barrier(bar, 0);

    // ---- 中间的"计算": 全 CTA 参与, 原地平方 ----
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) {
        float v = sT(i / CN, i % CN);
        sT(i / CN, i % CN) = v * v;
    }

    // ---- store: smem -> gmem ----
    __syncthreads();
    tma_store_fence();
    if (threadIdx.x == 0) {
        auto gc = tma_store.get_tma_tensor(make_shape(Int<M>{}, Int<N>{}));
        auto gt = local_tile(gc, Shape<Int<CM>, Int<CN>>{}, blk);
        auto per = tma_store.get_slice(0);
        copy(tma_store, per.partition_S(sT), per.partition_D(gt));
        tma_store_arrive();
    }
    tma_store_wait<0>();
}

static void section32_round_trip() {
    print_separator("§3.2  两端都用 TMA: load -> 计算 -> store");

    Buffers buf;
    auto mIn = make_tensor(make_gmem_ptr(buf.d_in),
                           make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
    auto mOut = make_tensor(make_gmem_ptr(buf.d_out),
                            make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});

    // 两个方向各要一个 descriptor —— 不能共用
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, mIn, slay);
    auto tma_store = make_tma_copy(SM90_TMA_STORE{}, mOut, slay);

    load_store_kernel<<<grid(), NTHR>>>(tma_load, tma_store);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(buf.h_out, buf.d_out, Buffers::bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < Buffers::elems; ++i)
        if (buf.h_out[i] != buf.h_in[i] * buf.h_in[i]) ok = false;

    printf("    out = in^2 的结果 = %s\n", ok ? "正确" : "错误");

    printf("\n  这个 kernel 里搬运指令一共两条, 都由 thread 0 发。\n");
    printf("  但**不是说其他 127 个线程闲着** —— 中间那段平方是全 CTA 一起做的。\n");
    printf("  TMA 省掉的是\"算地址、发搬运指令\"这件事, 不是省掉线程。\n");

    printf("\n  两个方向各要一个 descriptor:\n");
    printf("    make_tma_copy(SM90_TMA_LOAD{},  mIn,  slay)\n");
    printf("    make_tma_copy(SM90_TMA_STORE{}, mOut, slay)\n");
    printf("  一个 descriptor 绑死一个 gmem 张量 + 一个方向, 不能共用。\n");
}

// ===========================================================================
// §3.3  边界: tile 不整除时, 硬件自己兜
//
// 前面的尺寸都是整除的。真实矩阵不会这么配合。
// v0 那种手写搬运碰到不整除, 必须自己写 predicate:
//
//     if (row0 + r < M && col0 + c < N) smem[...] = in[...];
//     else                              smem[...] = 0;
//
// TMA 不用写。硬件读到越界坐标就跳过, smem 里对应位置填 0。
// 这里用 M=40, N=36, tile 32x32 —— 右下角那个 CTA 只有 8x4 是有效的。
// ===========================================================================
constexpr int OM = 40, ON = 36;  // 故意不被 32 整除

template <class TmaLoad>
__global__ static void oob_kernel(__grid_constant__ const TmaLoad tma, float* __restrict__ dump,
                                  int gridN) {
    constexpr int tx_bytes = CM * CN * sizeof(float);
    __shared__ __align__(128) float smem[CM * CN];
    __shared__ uint64_t bar;

    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto sT = make_tensor(make_smem_ptr(smem), slay);

    // 先把 smem 污染成 -1, 好看清楚哪些位置是硬件真写过的
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) smem[i] = -1.f;
    __syncthreads();

    if (threadIdx.x == 0) initialize_barrier(bar, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        set_barrier_transaction_bytes(bar, tx_bytes);
        auto gc = tma.get_tma_tensor(make_shape(Int<OM>{}, Int<ON>{}));
        auto gt = local_tile(gc, Shape<Int<CM>, Int<CN>>{}, make_coord(blockIdx.x, blockIdx.y));
        auto per = tma.get_slice(0);
        copy(tma.with(bar), per.partition_S(gt), per.partition_D(sT));
    }
    __syncthreads();
    wait_barrier(bar, 0);

    // 把整块 smem 原样倒出来 (按 tile 排, 不是按矩阵排), host 侧检查
    int tile_id = blockIdx.x * gridN + blockIdx.y;
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) dump[tile_id * CM * CN + i] = smem[i];
}

static void section33_out_of_bounds() {
    print_separator("§3.3  边界: tile 不整除时硬件自己兜");

    const int gM = (OM + CM - 1) / CM, gN = (ON + CN - 1) / CN;
    printf("  矩阵 %dx%d, tile %dx%d -> grid (%d,%d)\n", OM, ON, CM, CN, gM, gN);
    printf("  右下角 CTA (%d,%d) 覆盖行 %d..%d, 列 %d..%d, 只有 %dx%d = %d 个元素有效\n\n",
           gM - 1, gN - 1, (gM - 1) * CM, gM * CM - 1, (gN - 1) * CN, gN * CN - 1, OM - (gM - 1) * CM,
           ON - (gN - 1) * CN, (OM - (gM - 1) * CM) * (ON - (gN - 1) * CN));

    float *d_in, *d_dump;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * OM * ON));
    CUDA_CHECK(cudaMalloc(&d_dump, sizeof(float) * gM * gN * CM * CN));
    float* h_in = new float[OM * ON];
    for (int i = 0; i < OM * ON; ++i) h_in[i] = float(i + 1);  // 从 1 开始, 好和填充的 0 区分
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * OM * ON, cudaMemcpyHostToDevice));

    auto mIn = make_tensor(make_gmem_ptr(d_in),
                           make_layout(make_shape(Int<OM>{}, Int<ON>{}), LayoutRight{}));
    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, mIn, slay);

    oob_kernel<<<dim3(gM, gN), NTHR>>>(tma, d_dump, gN);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_dump = new float[gM * gN * CM * CN];
    CUDA_CHECK(cudaMemcpy(h_dump, d_dump, sizeof(float) * gM * gN * CM * CN,
                          cudaMemcpyDeviceToHost));

    // 逐 tile 检查: 界内 = 原值, 界外 = 0 (不是 -1, 说明硬件确实写过)
    bool ok = true;
    int zeros = 0, dirty = 0;
    for (int bx = 0; bx < gM; ++bx)
        for (int by = 0; by < gN; ++by)
            for (int r = 0; r < CM; ++r)
                for (int c = 0; c < CN; ++c) {
                    int gr = bx * CM + r, gc = by * CN + c;
                    float got = h_dump[(bx * gN + by) * CM * CN + r * CN + c];
                    float want = (gr < OM && gc < ON) ? h_in[gr * ON + gc] : 0.f;
                    if (got != want) ok = false;
                    if (got == 0.f) ++zeros;
                    if (got == -1.f) ++dirty;
                }

    printf("    界内元素 = 原值, 界外元素 = 0    -> %s\n", ok ? "正确" : "错误");
    printf("    被填 0 的位置 %d 个, 仍是 -1 (硬件没碰过) 的位置 %d 个\n", zeros, dirty);
    printf("\n  kernel 里一行 predicate 都没有 —— if (r < M) 那种判断完全不需要。\n");
    printf("  这是 §2.2 那个坐标 tensor 的直接好处: 坐标越界是合法的, 硬件看到\n");
    printf("  越界坐标就跳过; 而普通 tensor 切出来的越界指针一读就崩。\n");

    printf("\n  唯一的硬性要求在 gmem 一侧: 除最内层外, 每一维的 stride 必须是\n");
    printf("  16 字节的整数倍。本例 stride = (%d, 1) float = (%d, 4) 字节,\n", ON, ON * 4);
    printf("  %d %% 16 == %d, 满足。不满足时要先把矩阵 pad 到合适的宽度。\n", ON * 4, (ON * 4) % 16);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_dump));
    delete[] h_in;
    delete[] h_dump;
}

int main() {
    printf("cute_04 v2 —— TMA store 与边界\n");
    printf("对应 README §3    需要 -arch=sm_90a\n");

    section31_tma_store();
    section32_round_trip();
    section33_out_of_bounds();

    print_separator("小结");
    printf("  §3.1  store 用 fence 挡在发起之前, 不用 mbarrier; S/D 方向反过来\n");
    printf("  §3.2  两端都 TMA: 一趟 load -> 计算 -> store, 搬运指令共两条\n");
    printf("  §3.3  边界靠硬件 predicate, 越界填 0, kernel 里零 predicate\n");

    printf("\n下一步 (§4): 到这里搬运本身已经会了。但**搬进 smem 之后怎么摆**\n");
    printf("还没管过 —— 摆错了 consumer 读的时候会撞 bank。v3 讲 swizzle。\n");
    printf("\nv2 OK\n");
    return 0;
}
