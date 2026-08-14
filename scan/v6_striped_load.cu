// v6: striped 载入 + smem 转置 —— 修掉 v5 唯一剩下的访存缺陷。
//
// v5 的载入是这样的（VPT=4，每线程连续拿 4 个 float4）：
//
//   lane:      0              1              2         ...
//   读的地址:  [0 1 2 3]      [4 5 6 7]      [8 9 10 11]
//              ^^^^^^^^^ 单位是 float4
//
// 一个 lane 连续读 4 个 float4 = 64 字节，于是相邻 lane 的起始地址相隔 64 字节。
// 一个 warp 32 条 lane 铺开 32 × 64 = 2048 字节，而硬件一次访存事务只有 32 字节宽——
// 这 32 条 lane 没有任何两条能合并进同一个事务，一条指令拆成 32 次。
// 这叫 blocked 排布：数据在【线程内部】连续，但在【warp 内部】是跳着的。
//
// v5 为什么要这么排：第 ① 层是"线程串行扫自己那 VPT*4 个元素"，
// 必须拿到一段连续的元素，前缀才对得上。v5 的注释里写了"striped 排布访存更漂亮，
// 但线程手里的元素不连续，结果是错的"——这一版就是来解决那个"但是"的。
//
// 做法：载入和计算用两套排布，中间过一次 shared memory 转置。
//
//   ① striped 载入：线程 t 读 in[i * BLOCK + t]，i = 0..VPT-1
//        lane:      0    1    2   ...  31
//        i=0 读:    0    1    2   ...  31    <- 连续 32 个 float4 = 512 字节，完全合并
//        i=1 读:  256  257  258   ... 287    <- 再来一整段，同样合并
//      每一趟一个 warp 都在读连续地址，事务数从 32 降到 4（512B / 128B）。
//
//   ② 写进 smem 时仍按 striped 的下标，读出来时按 blocked 的下标，
//      这一读一写就完成了转置。之后第 ①~③ 层的 scan 和 v5 一字不差。
//
//   ③ 写出时反着来一次：算完的结果按 blocked 写进 smem，按 striped 读出来写 global。
//
// 代价是多一块 SEG/4 个 float4 的 smem（SEG=4096 时 16 KB）和 4 次 __syncthreads()。
// 收益是 global 的读和写都从 32 路拆分变成完全合并。
//
// 实测 0.1313 → 0.1244 ms（2044 → 2158 GB/s），5%。
//
// ⚠️ 为什么只有 5%——这一版顺带把 v5 的瓶颈定位清楚了
//   把 lookback 整个摘掉（结果当然是错的，只为了看时间）实测 0.0932 ms。
//   也就是说"载入 + 块内 scan + 写出"这部分只花 0.093ms，
//   而完整版是 0.122ms —— lookback 自己吃掉 0.029ms，占 24%。
//   访存优化只能作用在那 76% 上，所以 5% 已经接近这条路能拿的上限。
//
//   参照值（这台机器，N=32M）：
//     纯拷贝 2N          0.073 ms / 3694 GB/s   <- 硬件天花板
//     本版摘掉 lookback  0.093 ms               <- 已经比 CUB 快
//     CUB DeviceScan     0.099 ms / 2726 GB/s
//     本版（完整）       0.124 ms / 2158 GB/s
//     v5                 0.131 ms / 2044 GB/s
//
//   结论有点反直觉：我们的块内实现并不比 CUB 慢，差距全在 lookback 的等待上。
//   CUB 在这一步做了更多工程：状态数组按 cache line 对齐、用 __ldg 走只读路径、
//   窗口大小随 tile 数自适应，等等。想继续追就得往那个方向走，
//   而不是继续优化块内的 scan —— 那部分已经到头了。
//
// 还能往下走的方向（都不在这一版）：
//   - 用 atomicAdd 领 tile 号替代 blockIdx，去掉"GPU 按 blockIdx 顺序调度"这个假设。
//     正确性上更严谨，性能不变（v5 的文件头里讲了这件事）。
//   - inclusive / 任意 binary op / 任意长度（非 SEG 整数倍）的泛化，工程活，无新知识点。
//   - 更大的 SEG 配合 cp.async 流水线载入，把载入延迟藏到块内 scan 后面。
//     实测 SEG 再往上走收益转负（见 v5 文件头的取舍），要配合流水线才划算。
#include "common.h"

constexpr int SEG   = 4096;       // 和 v5 一样，理由见 v5 的文件头
constexpr int TILES = N / SEG;     // 8192 个 block
constexpr int BLOCK = 256;
constexpr int WARPS = BLOCK / 32;  // 8 个 warp
constexpr int VPT   = SEG / 4 / BLOCK;  // 每线程 4 个 float4（= 16 个元素）
static_assert(N % SEG == 0, "为简化代码这里要求整除；补 0 到 SEG 整数倍即可，scan 不依赖越界元素");
static_assert(WARPS <= 32, "第 ③ 层要用一个 warp 扫完所有 warp 的小计，所以 warp 数不能超过 32");

// tile 状态，和 v5 完全一样。INVALID 必须是 0，这样 cudaMemset 清零就是初始状态。
#define FLAG_INVALID   0  // 什么都还没发布
#define FLAG_AGGREGATE 1  // 只知道本段小计
#define FLAG_PREFIX    2  // 已经知道完整前缀了

// flag 和 value 打包在一起，__align__(8) 保证一条 8 字节指令就能整体读写。
// 为什么必须打包见 v5 的文件头（避免"旗子立起来了、值还没落地"的撕裂）。
struct __align__(8) TileState {
    float value;
    int   flag;
};

__device__ __forceinline__ TileState load_state(const TileState* p) {
    unsigned long long w = *(volatile unsigned long long*)p;
    return *(TileState*)&w;
}
__device__ __forceinline__ void store_state(TileState* p, float value, int flag) {
    TileState s;
    s.value = value;
    s.flag  = flag;
    *(volatile unsigned long long*)p = *(unsigned long long*)&s;
}

// ---------------- block 内 scan：和 v5 一字不差 ----------------

__device__ __forceinline__ float warp_inclusive_scan(float v) {
    for (int d = 1; d < 32; d *= 2) {
        float t = __shfl_up_sync(0xFFFFFFFF, v, d);
        if ((threadIdx.x & 31) >= d) v += t;
    }
    return v;
}

__device__ float block_scan_float4(float4* vv, float* s_warp) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // ---- ① 线程内：串行 exclusive scan 自己那 VPT*4 个元素，全程在寄存器里 ----
    // 转置之后这 VPT 个 float4 已经是连续的一段，和 v5 的语义完全一致。
    float thread_total = 0.0f;
    for (int i = 0; i < VPT; ++i) {
        float4& v = vv[i];
        const float x = v.x, y = v.y, z = v.z, w = v.w;
        v.x = thread_total;
        v.y = thread_total + x;
        v.z = thread_total + x + y;
        v.w = thread_total + x + y + z;
        thread_total += x + y + z + w;
    }

    // ---- ② warp 内：对 thread_total 做 scan，减掉自己那份就成了 exclusive ----
    float warp_scan = warp_inclusive_scan(thread_total);
    float thread_prefix = warp_scan - thread_total;
    if (lane == 31) s_warp[warp] = warp_scan;
    __syncthreads();

    // ---- ③ block 内：warp 0 扫一遍 WARPS 个小计 ----
    if (warp == 0) {
        float t = (lane < WARPS) ? s_warp[lane] : 0.0f;
        float s = warp_inclusive_scan(t);
        if (lane < WARPS) s_warp[lane] = s - t;
        if (lane == WARPS - 1) s_warp[WARPS] = s;  // 顺手存下 block 总和
    }
    __syncthreads();

    // ---- 合成：三层前缀相加，摊到自己那 VPT*4 个元素上 ----
    const float base = s_warp[warp] + thread_prefix;
    for (int i = 0; i < VPT; ++i) {
        vv[i].x += base; vv[i].y += base; vv[i].z += base; vv[i].w += base;
    }
    return s_warp[WARPS];
}

// ---------------- lookback：和 v5 一字不差，注释见 v5 ----------------

__device__ float lookback_warp(const int tile, const TileState* state) {
    const int lane = threadIdx.x & 31;
    float exclusive = 0.0f;

    for (int window = tile - 32; ; window -= 32) {
        const int p = window + lane;

        TileState s;
        do {
            if (p < 0) {  // 数组头之前，当成"前缀 = 0"的 PREFIX
                s.value = 0.0f;
                s.flag  = FLAG_PREFIX;
            } else {
                s = load_state(&state[p]);
            }
        } while (__any_sync(0xFFFFFFFF, s.flag == FLAG_INVALID));

        // 找出窗口里最靠后的那个 PREFIX，它之前的都不用管
        const unsigned prefix_mask = __ballot_sync(0xFFFFFFFF, s.flag == FLAG_PREFIX);
        const int cut = (prefix_mask == 0) ? 0 : (31 - __clz(prefix_mask));

        float sum = (lane >= cut) ? s.value : 0.0f;
        for (int d = 16; d > 0; d >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, d);
        exclusive += sum;

        if (prefix_mask != 0) return exclusive;
    }
}

// ---------------- kernel：本版的改动全在这里 ----------------

__global__ void scan_striped_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                                    TileState* state) {
    __shared__ float s_warp[WARPS + 1];
    __shared__ float s_exclusive;         // warp 0 问回来的结果，广播给全 block
    __shared__ float4 s_xpose[SEG / 4];   // 本版新增：striped <-> blocked 的中转站

    const int tid = threadIdx.x;
    const int tile = blockIdx.x;
    const size_t base4 = (size_t)tile * (SEG / 4);

    // ---- 1a. striped 载入：一个 warp 每趟读连续的 32 个 float4，完全合并 ----
    for (int i = 0; i < VPT; ++i) s_xpose[i * BLOCK + tid] = in[base4 + i * BLOCK + tid];
    __syncthreads();

    // ---- 1b. 按 blocked 下标读出来，转置完成。现在每个线程手里是连续的一段 ----
    float4 v[VPT];
    for (int i = 0; i < VPT; ++i) v[i] = s_xpose[tid * VPT + i];
    __syncthreads();  // 等所有线程读完，下面写出阶段要复用 s_xpose

    // ---- 2. 块内 scan，和 v5 一样 ----
    float total = block_scan_float4(v, s_warp);

    // ---- 3. 由 warp 0 负责跟别的 block 打交道（lookback 需要一整个 warp 协作）----
    if (tid < 32) {
        // 先发小计：后面的 block 就算问到我，也能拿走它继续往前推，不必干等
        if (tid == 0 && tile > 0) store_state(&state[tile], total, FLAG_AGGREGATE);
        __syncwarp();

        float exclusive = (tile == 0) ? 0.0f : lookback_warp(tile, state);

        // 升级成 PREFIX：后面的 block 问到我就能一步收工
        if (tid == 0) {
            store_state(&state[tile], exclusive + total, FLAG_PREFIX);
            s_exclusive = exclusive;
        }
    }
    __syncthreads();  // 等 warp 0 把 s_exclusive 填好

    // ---- 4a. 加上 tile 前缀，按 blocked 下标写回 smem ----
    const float e = s_exclusive;
    for (int i = 0; i < VPT; ++i) {
        float4 t = v[i];
        t.x += e; t.y += e; t.z += e; t.w += e;
        s_xpose[tid * VPT + i] = t;
    }
    __syncthreads();

    // ---- 4b. 按 striped 下标读出来写 global，写入同样完全合并 ----
    for (int i = 0; i < VPT; ++i) out[base4 + i * BLOCK + tid] = s_xpose[i * BLOCK + tid];
}

// ---------------- main：只做编排 ----------------

int main() {
    const size_t bytes = (size_t)N * sizeof(float);
    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);

    fill_input(h_in, N);
    cpu_exclusive_scan(h_in, h_ref, N, N);  // seg = N：这一版是全局 scan

    float *d_in, *d_out;
    TileState* d_state;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_state, (size_t)TILES * sizeof(TileState)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    // d_state 不用在这里清零：每次 launch 前都要清，所以清零放在 benchmark 的 lambda 里

    printf("scan v6 striped load  [全局 scan]  (N = %d, SEG = %d, block = %d, tiles = %d, "
           "smem %zu B)\n", N, SEG, BLOCK, TILES, sizeof(float4) * (SEG / 4));

    float ms = benchmark([&] {
        // flag 必须每次都清回 INVALID(0)，否则第二次 launch 会读到上一次的残留状态
        CUDA_CHECK(cudaMemset(d_state, 0, (size_t)TILES * sizeof(TileState)));
        scan_striped_kernel<<<TILES, BLOCK>>>((const float4*)d_in, (float4*)d_out, d_state);
    });
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    report_perf("time (1 kernel)", ms, 2);  // 读 in 一次 + 写 out 一次
    bool pass = verify(h_out, h_ref, N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_state));
    free(h_in); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
