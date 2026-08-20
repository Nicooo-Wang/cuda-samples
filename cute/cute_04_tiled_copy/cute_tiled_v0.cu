// cute_04 v0 —— 用一个 warp 手写搬运: 本章的基准
//
// 对应 README §1。
//
// 这一章要学的是 TMA —— Hopper 上专门做 gmem<->smem 整块搬运的硬件。
// 但在见到 TMA 之前, 先得有一个"没有 TMA 时怎么搬"的版本作参照, 否则
// 看不出 TMA 到底省掉了什么。
//
// 这个文件就是那个参照: 一个最朴素的、每个线程自己算地址的搬运。
// 不用任何异步指令, 不用任何 CuTe 高级设施 —— 就是普通的 load/store。
//
//   §1.1  一个 warp 搬一个 tile: 每个 lane 自己算地址
//   §1.2  同一件事用 CuTe 的坐标写法表达 (为 v1 的 TMA 做准备)
//
// ---------------------------------------------------------------------------
// 这一章从头到尾的任务
//
//   gmem 里有一个 M x N 的 float 矩阵, row-major, stride = (N, 1)。
//   把它切成 CM x CN 的 tile, 每个 CTA 负责一块:
//     1) 把自己那块 tile 搬进 smem
//     2) (在 smem 里做点什么)
//     3) 搬回 gmem
//
//   +--------------- N = 128 ---------------+
//   | <-- CN=32 -->                         |  ^
//   | +-----------+-----------+---+---+     |  |
//   | | CTA (0,0) | CTA (0,1) |   |   |     |  | CM = 32
//   | +-----------+-----------+---+---+     |  v
//   | | CTA (1,0) | CTA (1,1) |   |   |     |
//   | +-----------+-----------+---+---+     |     ^
//   | |    ...                          |   |     | M = 256
//   +---------------------------------------+     v
//
//   grid = (M/CM, N/CN)。这个任务贯穿 v0 -> v1 -> v2 -> v3,
//   每一版只换"怎么搬"这一件事。
//
// ---------------------------------------------------------------------------
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_v0

#include <cute/tensor.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 全章统一的尺寸
// ---------------------------------------------------------------------------
constexpr int M = 256, N = 128;  // gmem 矩阵: 256x128 float = 128KB
constexpr int CM = 32, CN = 32;  // 一个 CTA 负责的 tile: 32x32 float = 4KB
constexpr int NTHR = 128;        // 每 CTA 线程数

static dim3 grid() { return dim3(M / CM, N / CN); }  // (8, 4)

// ---------------------------------------------------------------------------
// 全章共用的缓冲区 —— 构造时分配并填好, 析构时释放
// 每个 host 函数开头写一行 `Buffers buf;` 就得到一套干净的 in/out
// ---------------------------------------------------------------------------
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

    // out 应该和 in 逐元素相同 (这一章的 kernel 都是"搬过去再搬回来")
    bool check() {
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < elems; ++i)
            if (h_out[i] != h_in[i]) return false;
        return true;
    }
};

// ===========================================================================
// §1.1  最朴素的写法: 每个线程自己算地址
//
// 这就是"没有任何硬件辅助"时你会写的代码。三件事全靠线程自己干:
//
//   a) 算 gmem 地址   —— 从 blockIdx / threadIdx 推出自己该读哪个元素
//   b) 发 load/store  —— 每个线程一条一条地发
//   c) 同步           —— __syncthreads() 等所有人写完 smem
// ===========================================================================
__global__ static void copy_warp_kernel(const float* __restrict__ in, float* __restrict__ out) {
    __shared__ float smem[CM * CN];

    // 本 CTA 负责的 tile 左上角在 gmem 里的位置
    const int row0 = blockIdx.x * CM;
    const int col0 = blockIdx.y * CN;

    // ---- a) 每个线程算自己的地址, b) 每个线程发自己的 load ----
    //
    // 128 个线程搬 32x32 = 1024 个元素, 每人 8 个, 分 8 趟。
    // 第 i 个元素在 tile 内的坐标是 (i / CN, i % CN):
    //
    //     线程 t 第 0 趟搬 tile 的第 t 个元素
    //     线程 t 第 1 趟搬 tile 的第 t + 128 个元素
    //     ...
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) {
        int r = i / CN;  // <- 这两行就是"自己算地址"
        int c = i % CN;
        smem[r * CN + c] = in[(row0 + r) * N + (col0 + c)];
    }

    // ---- c) 同步: 等全 CTA 都写完 smem ----
    __syncthreads();

    // 搬回 gmem (这一章不做实际计算, 直接原样搬回)
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) {
        int r = i / CN;
        int c = i % CN;
        out[(row0 + r) * N + (col0 + c)] = smem[r * CN + c];
    }
}

static void section11_warp_copy() {
    print_separator("§1.1  最朴素的搬运: 每个线程自己算地址");

    printf("  gmem 矩阵 %dx%d float, row-major, stride = (%d, 1)\n", M, N, N);
    printf("  tile %dx%d float = %d KB, grid = (%d, %d), 每 CTA %d 线程\n\n", CM, CN,
           CM * CN * 4 / 1024, grid().x, grid().y, NTHR);

    Buffers buf;
    copy_warp_kernel<<<grid(), NTHR>>>(buf.d_in, buf.d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("    搬运结果 = %s\n", buf.check() ? "正确" : "错误");

    printf("\n  这一版的成本清单 —— 记住这五行, v1 会逐条对照:\n\n");
    printf("    发指令的线程数    %d 个 (每人发自己那几条 load/store)\n", NTHR);
    printf("    地址计算          每线程每趟各算一次 (i/CN, i%%CN, 再乘 stride)\n");
    printf("    边界处理          tile 不整除时要自己写 if (r < M && c < N)\n");
    printf("    同步              __syncthreads(), 全 CTA 栅栏\n");
    printf("    寄存器            每线程要几个寄存器存地址和中转值\n");
}

// ===========================================================================
// §1.2  同一件事, 用 CuTe 的坐标写法
//
// 上面的 `in[(row0+r)*N + (col0+c)]` 是手写的地址算术。CuTe 把"矩阵的形状和
// 摆法"打包成 Layout, 把"指针 + Layout"打包成 Tensor, 于是同样的访问写成
// 二维坐标 gIn(r, c), 地址算术交给 Layout。
//
// 这一节的产出不是性能, 而是 v1 要用到的两个东西:
//
//   make_tensor(make_gmem_ptr(p), layout)   把裸指针变成能按坐标访问的 tensor
//   local_tile(mIn, tile_shape, coord)      从大 tensor 里切出本 CTA 那一块
//
// v1 的 TMA 用的正是同一套坐标写法 —— 只不过 tensor 里装的不是数据, 是坐标。
// ===========================================================================
__global__ static void copy_cute_kernel(const float* __restrict__ in, float* __restrict__ out,
                                        bool announce) {
    __shared__ float smem[CM * CN];

    // gmem 的两个视图: 形状 (M,N), 行 stride = N, 列 stride = 1
    auto mIn = make_tensor(make_gmem_ptr(in),
                           make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
    auto mOut = make_tensor(make_gmem_ptr(out),
                            make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));

    // 切出本 CTA 的 tile: 把 (M,N) 按 (CM,CN) 分块, 取第 (blockIdx.x, blockIdx.y) 块
    auto blk = make_coord(blockIdx.x, blockIdx.y);
    auto gIn = local_tile(mIn, Shape<Int<CM>, Int<CN>>{}, blk);    // (CM, CN)
    auto gOut = local_tile(mOut, Shape<Int<CM>, Int<CN>>{}, blk);  // (CM, CN)

    // smem 也建成 tensor, 于是三边都能用 (r, c) 访问
    auto sT = make_tensor(make_smem_ptr(smem),
                          make_layout(make_shape(Int<CM>{}, Int<CN>{}), LayoutRight{}));

    if (announce && thread0() && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("    mIn  = ");
        print(mIn);
        printf("\n    gIn  = ");
        print(gIn);
        printf("      <- 本 CTA 的一块, 坐标已经偏移到 tile 左上角\n    sT   = ");
        print(sT);
        printf("\n");
    }

    // 搬运本身: 地址算术没了, 只剩坐标
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) sT(i / CN, i % CN) = gIn(i / CN, i % CN);
    __syncthreads();
    for (int i = threadIdx.x; i < CM * CN; i += blockDim.x) gOut(i / CN, i % CN) = sT(i / CN, i % CN);
}

static void section12_cute_coords() {
    print_separator("§1.2  同一件事, 用 CuTe 的坐标写法");

    Buffers buf;
    copy_cute_kernel<<<grid(), NTHR>>>(buf.d_in, buf.d_out, true);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n    搬运结果 = %s\n", buf.check() ? "正确" : "错误");

    printf("\n  和 §1.1 比, 变的只有写法:\n");
    printf("    §1.1   in[(row0 + r) * N + (col0 + c)]     <- 手写地址算术\n");
    printf("    §1.2   gIn(r, c)                          <- Layout 负责算地址\n");
    printf("  成本清单一条没变 —— 还是 %d 个线程各发各的 load。\n", NTHR);
    printf("  但 local_tile / make_tensor 这套坐标写法, v1 的 TMA 会原样用上。\n");
}

int main() {
    printf("cute_04 v0 —— 用一个 warp 手写搬运: 本章的基准\n");
    printf("对应 README §1    需要 -arch=sm_90a\n");

    section11_warp_copy();
    section12_cute_coords();

    print_separator("小结");
    printf("  §1.1  搬运 = 每线程算地址 + 每线程发指令 + __syncthreads\n");
    printf("  §1.2  CuTe 的 Tensor/local_tile 把地址算术收进 Layout, 但发指令的\n");
    printf("        还是那 %d 个线程 —— 成本清单一条没少。\n", NTHR);

    printf("\n下一步 (§2): Hopper 有一个专门的硬件单元 TMA, 能让**一个线程**\n");
    printf("描述整块搬运、硬件自己去搬。v1 只换 gmem->smem 这一步, 其余不动。\n");
    printf("\nv0 OK\n");
    return 0;
}
