// cute_04 v1 —— TMA load: 一个线程描述整块, 硬件自己搬
//
// 对应 README §2。
//
// v0 的搬运是 128 个线程各算各的地址、各发各的 load。这一版**只换 gmem->smem
// 这一步**, 换成 TMA —— 其余部分 (smem 布局、搬回 gmem、验证) 和 v0 一字不差。
//
//   §2.1  最小的 TMA load                <- 完整跑通一次, 认识五个新东西
//   §2.2  坐标 tensor 里到底装了什么      <- get_tma_tensor 打印出来看
//   §2.3  barrier 的三个动作和 phase      <- 同一个 barrier 连搬两块
//
// ---------------------------------------------------------------------------
// TMA 的两步走
//
// 和 v0 那种"全在 kernel 里"的搬运不同, TMA 天生分成 host / kernel 两步:
//
//   host 侧    make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout)
//                 -> 造出一个 descriptor (128 字节), 里面记着 gmem 的基地址、
//                    形状、stride, 以及 tile 形状和 smem 的摆法
//
//   kernel 侧  copy(tma.with(bar), 源, 目标)
//                 -> 一条指令: "按 descriptor 搬第 (bx,by) 块"
//
// 为什么要分两步: descriptor 里的东西 (基地址、stride、tile 形状) 对所有 CTA
// 都一样, 只在 host 上算一次即可; kernel 里每个 CTA 只需要说"我要第几块"。
//
// ---------------------------------------------------------------------------
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_v1

#include <cute/tensor.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 和 v0 完全相同的尺寸
// ---------------------------------------------------------------------------
constexpr int M = 256, N = 128;  // gmem 矩阵: 256x128 float
constexpr int CM = 32, CN = 32;  // 一个 CTA 负责的 tile: 32x32 float = 4KB
constexpr int NTHR = 128;

static dim3 grid() { return dim3(M / CM, N / CN); }

// 和 v0 完全相同的缓冲区
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
// §2.1  最小的 TMA load
//
// 和 v0 §1.2 逐行对照, 只有 gmem->smem 那一段变了。新出现五个东西:
//
//   1) __grid_constant__ const   descriptor 传参的硬性要求
//   2) tma.get_tma_tensor(...)   源不是数据 tensor, 是坐标 tensor
//   3) __shared__ uint64_t bar   mbarrier: TMA 完成的通知机制
//   4) set_barrier_transaction_bytes(bar, N)   "我要等 N 个字节"
//   5) tma.get_slice(0) + partition_S/D        谁搬哪一份 (TMA 只有一个"线程")
// ===========================================================================
template <class TmaLoad>
__global__ static void copy_tma_kernel(__grid_constant__ const TmaLoad tma,  // 1)
                                       float* __restrict__ out, bool announce) {
    // 这一次搬多少字节 —— barrier 要按字节数等, 所以必须先算出来
    constexpr int tma_transaction_bytes = CM * CN * sizeof(float);

    __shared__ __align__(128) float smem[CM * CN];  // TMA 硬性要求 128B 对齐
    __shared__ uint64_t bar;                        // 3) mbarrier, 64 位, 在 smem 里

    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto sT = make_tensor(make_smem_ptr(smem), slay);

    // 2) 源: 坐标 tensor。它不存数据, 只提供"我要 gmem 的哪一块"这个坐标。
    //    真正的 gmem 地址在 host 侧的 descriptor 里, kernel 摸不到也不需要。
    auto gcoord = tma.get_tma_tensor(make_shape(Int<M>{}, Int<N>{}));
    auto gtile = local_tile(gcoord, Shape<Int<CM>, Int<CN>>{},
                            make_coord(blockIdx.x, blockIdx.y));  // 和 v0 一模一样的切法

    if (announce && thread0() && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("    gcoord = ");
        print(gcoord);
        printf("\n    gtile  = ");
        print(gtile);
        printf("\n    一次搬 %d 字节\n", tma_transaction_bytes);
    }

    // ---- barrier 初始化: 只有一个线程会发 TMA, 所以 arrival count = 1 ----
    if (threadIdx.x == 0) initialize_barrier(bar, /* arrival count */ 1);
    __syncthreads();  // 等 barrier 初始化对全 CTA 可见

    // ---- 整个搬运只有一个线程在跑 ----
    if (threadIdx.x == 0) {
        // 4) 告诉 barrier: 这一轮我要等这么多字节到位
        set_barrier_transaction_bytes(bar, tma_transaction_bytes);

        // 5) TMA 的"线程数"是 1 (整块交给硬件), 所以 get_slice(0)
        auto per_cta = tma.get_slice(0);
        copy(tma.with(bar), per_cta.partition_S(gtile), per_cta.partition_D(sT));
        //   ^^^^^^^^^^^^^ 把 barrier 绑给这次搬运: 硬件搬完会去 barrier 上销账
    }

    // ---- 所有线程在这里等硬件搬完 ----
    // phase = 0: 这个 barrier 初始化后第一次用。§2.3 讲什么时候要翻成 1。
    __syncthreads();       // 先解决 thread 0 和其他线程的分歧
    wait_barrier(bar, 0);  // 这一行之后, smem 里的数据可见

    // ---- 搬回 gmem: 和 v0 一字不差 ----
    auto mOut = make_tensor(make_gmem_ptr(out),
                            make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
    auto gOut = local_tile(mOut, Shape<Int<CM>, Int<CN>>{}, make_coord(blockIdx.x, blockIdx.y));
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) gOut(i / CN, i % CN) = sT(i / CN, i % CN);
}

static void section21_minimal_tma() {
    print_separator("§2.1  最小的 TMA load");

    Buffers buf;

    // ---- host 侧的两步 ----
    // 第 1 步: gmem tensor —— 必须用**真实设备指针**建。
    //          用 nullptr 建 descriptor 会在运行时报 "Failed to initialize the
    //          TMA descriptor 201", 而且错误信息完全不提是指针的问题。
    auto mIn = make_tensor(make_gmem_ptr(buf.d_in),
                           make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));

    // 第 2 步: smem layout —— 描述"搬进来之后在 smem 里怎么摆"
    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});

    // 造 descriptor。tile 形状由 smem layout 的 shape 决定 (这里就是 32x32)
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, mIn, slay);

    printf("  host 侧造出来的 tma 对象:\n\n");
    print(tma);
    printf("\n");
    printf("  Tiler_MN = (%d,%d) 就是 tile 形状; Copy_Atom 的 ValLayout 说明\n", CM, CN);
    printf("  TMA 眼里这是「1 个线程搬 %d 个元素」—— 所以 kernel 里 get_slice(0)。\n\n",
           CM * CN);

    copy_tma_kernel<<<grid(), NTHR>>>(tma, buf.d_out, true);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n    搬运结果 = %s\n", buf.check() ? "正确" : "错误");

    printf("\n  和 v0 的成本清单并排:\n\n");
    printf("                    v0 手写 (§1)                 v1 TMA (§2)\n");
    printf("    --------------  ---------------------------  ---------------------------\n");
    printf("    发指令线程数    %-3d 个, 每人发自己那几条      1 个, 一共一条\n", NTHR);
    printf("    地址计算        每线程每趟各算一次           host 侧 descriptor 算好\n");
    printf("    边界处理        自己写 if (r<M && c<N)       硬件自动 predicate (§3.3)\n");
    printf("    同步            __syncthreads() 全员栅栏     mbarrier 按字节数等\n");
    printf("    寄存器          每线程存地址和中转值         几乎为 0\n");
}

// ===========================================================================
// §2.2  坐标 tensor 里到底装了什么
//
// §2.1 里最反直觉的一行是这个:
//
//     auto gcoord = tma.get_tma_tensor(make_shape(M, N));
//
// 为什么不能直接把 v0 那个 mIn 传进去? 因为 TMA 的 copy 取的不是"值", 是
// "坐标": 硬件拿着坐标去问 descriptor 要地址。所以这个 tensor 里装的是
// (i, j) 这样的坐标对, 不是 float。
//
// 这一节把它打印出来, 看清楚每个 CTA 拿到的是哪一块坐标。
// ===========================================================================
template <class TmaLoad>
__global__ static void print_coord_kernel(__grid_constant__ const TmaLoad tma, int show_bx,
                                          int show_by) {
    auto gcoord = tma.get_tma_tensor(make_shape(Int<M>{}, Int<N>{}));
    auto gtile = local_tile(gcoord, Shape<Int<CM>, Int<CN>>{}, make_coord(blockIdx.x, blockIdx.y));

    if (thread0() && int(blockIdx.x) == show_bx && int(blockIdx.y) == show_by) {
        printf("    CTA (%d,%d) 的 gtile:\n      ", show_bx, show_by);
        print(gtile);
        printf("\n      前 4x4 个坐标:\n");
        for (int r = 0; r < 4; ++r) {
            printf("        ");
            for (int c = 0; c < 4; ++c) {
                auto crd = gtile(r, c);
                printf("(%3d,%3d) ", int(get<0>(crd)), int(get<1>(crd)));
            }
            printf("\n");
        }
    }
}

static void section22_coord_tensor() {
    print_separator("§2.2  坐标 tensor 里装的是坐标, 不是数据");

    Buffers buf;
    auto mIn = make_tensor(make_gmem_ptr(buf.d_in),
                           make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, mIn, slay);

    printf("  普通 gmem tensor (v0 用的那个) 装的是数据:\n    ");
    print(mIn);
    printf("\n      ^ gmem_ptr[32b](地址) —— 一个真实指针\n\n");

    printf("  坐标 tensor (TMA 用的) 装的是坐标:\n");
    print_coord_kernel<<<grid(), 32>>>(tma, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    print_coord_kernel<<<grid(), 32>>>(tma, 2, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n      ^ ArithTuple(基坐标) —— 没有指针, 只有坐标\n");
    printf("        CTA (2,1) 从 (%d,%d) 开始, 正是 blockIdx * tile 形状。\n", 2 * CM, 1 * CN);

    printf("\n  为什么必须这样: 用普通 tensor 切片会切出**越界的裸指针**\n");
    printf("  (tile 超出矩阵时), 一读就崩; 而坐标可以越界 —— 硬件看到越界坐标\n");
    printf("  就跳过不读。这是 §3.3 那个\"边界自动处理\"的底层原因。\n");
}

// ===========================================================================
// §2.3  barrier 的三个动作, 和 phase 为什么要翻转
//
// §2.1 里 barrier 出现了三次, 各干一件事:
//
//   initialize_barrier(bar, cnt)              开张: 有 cnt 个参与者
//   set_barrier_transaction_bytes(bar, n)     记账: 这一轮要收 n 个字节
//   wait_barrier(bar, phase)                  等账收齐
//
// mbarrier 没有"计数器归零"的概念, 它用一个 **phase bit** 表示"第几轮":
//
//   初始化后    phase = 0
//   收齐一轮后  phase 自动翻成 1
//   再收齐一轮  翻回 0
//   ...
//
// 所以 wait 时要传"我在等的这一轮的 phase":第 k 次使用传 k & 1。
// 传错了症状是**死锁** —— 等一个永远不会到来的翻转。
// ===========================================================================
template <class TmaLoad>
__global__ static void two_tiles_kernel(__grid_constant__ const TmaLoad tma,
                                        float* __restrict__ out) {
    constexpr int tx_bytes = CM * CN * sizeof(float);
    __shared__ __align__(128) float smem[CM * CN];
    __shared__ uint64_t bar;

    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto sT = make_tensor(make_smem_ptr(smem), slay);
    auto gcoord = tma.get_tma_tensor(make_shape(Int<M>{}, Int<N>{}));
    auto per_cta = tma.get_slice(0);

    if (threadIdx.x == 0) initialize_barrier(bar, 1);
    __syncthreads();

    // 同一个 barrier 连用两轮, 中间不重新初始化
    for (int t = 0; t < 2; ++t) {
        if (threadIdx.x == 0) {
            set_barrier_transaction_bytes(bar, tx_bytes);
            auto gtile = local_tile(gcoord, Shape<Int<CM>, Int<CN>>{}, make_coord(t, 0));
            copy(tma.with(bar), per_cta.partition_S(gtile), per_cta.partition_D(sT));
        }
        __syncthreads();
        wait_barrier(bar, t & 1);  // <- 第 0 轮等 phase 0, 第 1 轮等 phase 1
                                   //    写死 0 会在第二轮死锁

        for (int i = threadIdx.x; i < CM * CN; i += blockDim.x)
            out[t * CM * CN + i] = sT(i / CN, i % CN);
        __syncthreads();  // 等大家读完再覆盖 smem
    }
}

static void section23_barrier_phase() {
    print_separator("§2.3  barrier 的三个动作和 phase 翻转");

    printf("  一个 barrier 连搬两块 (tile (0,0) 和 tile (1,0)), 中间不重新初始化。\n\n");

    Buffers buf;
    auto mIn = make_tensor(make_gmem_ptr(buf.d_in),
                           make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
    auto slay = make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{});
    auto tma = make_tma_copy(SM90_TMA_LOAD{}, mIn, slay);

    two_tiles_kernel<<<1, NTHR>>>(tma, buf.d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 前两块 tile 应该分别等于 in 的 (0..31, 0..31) 和 (32..63, 0..31)
    CUDA_CHECK(cudaMemcpy(buf.h_out, buf.d_out, 2 * CM * CN * sizeof(float),
                          cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int t = 0; t < 2; ++t)
        for (int r = 0; r < CM; ++r)
            for (int c = 0; c < CN; ++c)
                if (buf.h_out[t * CM * CN + r * CN + c] != buf.h_in[(t * CM + r) * N + c])
                    ok = false;

    printf("    两轮都搬对 = %s\n", ok ? "正确" : "错误");

    printf("\n  phase 的规律:\n");
    printf("    第 0 次用这个 barrier  ->  wait_barrier(bar, 0)\n");
    printf("    第 1 次用这个 barrier  ->  wait_barrier(bar, 1)\n");
    printf("    第 2 次用这个 barrier  ->  wait_barrier(bar, 0)\n");
    printf("    ...  第 k 次 -> k & 1\n");
    printf("  代码里就是那一行 wait_barrier(bar, t & 1)。\n");
    printf("  写成 wait_barrier(bar, 0) 的症状是**第二轮死锁**, 不是算错。\n");
    printf("  v3 做多 stage 时每个 buffer 各有一个 barrier, phase 公式会变成\n");
    printf("  (k / STAGES) & 1 —— 那是同一条规律。\n");
}

int main() {
    printf("cute_04 v1 —— TMA load: 一个线程描述整块, 硬件自己搬\n");
    printf("对应 README §2    需要 -arch=sm_90a\n");

    section21_minimal_tma();
    section22_coord_tensor();
    section23_barrier_phase();

    print_separator("小结");
    printf("  §2.1  host 造 descriptor (make_tma_copy), kernel 发一条 copy\n");
    printf("        新东西五个: __grid_constant__ / 坐标 tensor / mbarrier /\n");
    printf("        transaction bytes / get_slice(0)\n");
    printf("  §2.2  坐标 tensor 装的是 (i,j) 而不是数据 —— 越界也安全\n");
    printf("  §2.3  barrier 三动作 (init / expect_tx / wait), phase 第 k 次用 k&1\n");

    printf("\n下一步 (§3): 反方向的 TMA store 怎么写 (它不用 barrier, 用 fence),\n");
    printf("以及 tile 不整除时硬件怎么自动兜住边界。\n");
    printf("\nv1 OK\n");
    return 0;
}
