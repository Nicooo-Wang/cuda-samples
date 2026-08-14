// v5: single-pass scan（decoupled lookback）—— 一个 kernel 干完，全局访存回到 2N 的理论下限。
//     这是 CUB / 现代库里真正在用的算法（Merrill & Garland, 2016）。
//
// v3/v4 的结构性浪费：三个 kernel 要把整个数组过两遍（① 读写 2N，③ 又读写 2N = 4N）。
//   为什么必须分成三个 kernel？因为 block 之间要通信（第 i 段得知道前面所有段的总和），
//   而 CUDA 里 block 之间唯一的官方同步点就是"kernel 结束"。
//
// decoupled lookback 的破局思路：别等，自己去问。
//   每个 block 算完自己那段的局部和之后，把结果【发布】到全局内存，
//   然后回头去问前面的 block："你的前缀算出来了吗？"
//   - 前驱已经知道自己的【完整前缀】(PREFIX) → 太好了，直接拿走，收工
//   - 前驱只算出了【本段小计】(AGGREGATE)     → 那就把它的小计收下，继续往更前面问
//   - 前驱还什么都没发布 (INVALID)           → 原地等一下，再问一次
//
//   关键在于第二种情况：不需要等前驱把自己的事办完，拿它的"半成品"接着往前推就行。
//   这就是 decoupled（解耦）的意思——也是它能一趟搞定的原因。
//
// 每个 tile 一个状态槽，flag 和 value 打包在同一个 8 字节结构里（见 TileState）：
//   flag  : INVALID(0) / AGGREGATE(1) / PREFIX(2)
//   value : flag=AGGREGATE 时是本段小计；flag=PREFIX 时是本段的完整 inclusive 前缀
//
// ⚠️ 为什么要把 flag 和 value 打包成 8 字节
//   拆成两个数组的写法（flag[] + agg[]/prefix[]）需要手写内存序：
//       发布：先写 value → __threadfence() → 再写 flag
//       读取：先读 flag  → __threadfence() → 再读 value
//   顺序反了就可能看到"旗子立起来了、值还没落地"的中间态。
//   打包成一个 8 字节、8 字节对齐的结构之后，一条 ld.volatile.global.b64 就同时拿到两者，
//   天然不可能撕裂，两处 fence 都省了。
//   实测这一下从 0.38ms 提到 0.19ms（省掉的是每个窗口两次访存往返，见 lookback 的注释）。
//   （另一种正规写法是 cuda::atomic_ref 的 release/acquire，语义更精确，
//     但在这条热路径上实测反而更慢——8 字节打包已经把竞态从根上消掉了。）
//
// ⚠️ 为什么不会死锁
//   本 block 在自旋等前驱，如果前驱压根还没被调度上 GPU，那就死等了。
//   靠的是这条实践事实：GPU 大体按 blockIdx 递增的顺序调度 block，
//   所以 t 在跑的时候，t-1 必定已经跑完或正在跑。CUB 依赖的也是这一条。
//   更保险的做法是不用 blockIdx，而是每个 block 用 atomicAdd(&counter, 1) 自己领一个号——
//   领到号就说明自己已经在 GPU 上了，号小的一定先领到，顺序就有了保证。
//
// ⚠️ 为什么 SEG 从 1024 加大到 4096（每线程 4 个 float4）
//   lookback 是一条【串行的依赖链】：tile 越多，链越长。
//   SEG=1024 时有 32768 个 tile，实测每个 tile 平均要看 4 个窗口（= 4 次访存延迟首尾相接）
//   才能凑齐前缀，单这一项就要 0.12ms——比整个 scan 本身（0.07ms）还贵。
//   SEG=4096 把 tile 数降到 8192，链短了 4 倍，实测 0.19ms → 0.13ms。
//   这是 v5 特有的取舍：v0~v4 的 block 之间互不相干，tile 多少无所谓；
//   v5 一旦让 block 互相等待，"tile 数"就成了性能参数。
//
// 收益：全局访存 4N → 2N（读一次 in、写一次 out，就这样）。
//   状态数组的流量是 TILES × 8B = 64KB，相比 256MB 可以忽略。
//   实测 0.135ms / 2044 GB/s（v4 是 0.168ms）。
//   作为参照：这台机器纯拷贝 2N 是 0.073ms / 3694 GB/s，
//   CUB 的 DeviceScan 是 0.099ms / 2726 GB/s，我们还差 1.36 倍。
//
// 差距在哪：不在块内实现，而在 lookback 的等待上。
//   v6 顺手把这件事量化了——把 lookback 摘掉只需 0.093ms，比 CUB 还快；
//   lookback 自己吃掉 0.029ms，占 24%。详见 v6 的文件头。
//   v6 还会修掉这一版载入时的非合并访存（每线程连续拿 VPT 个 float4，
//   导致相邻 lane 的地址相隔 64 字节，一个 warp 拆成 32 次事务）。
#include "common.h"

constexpr int SEG   = 4096;       // 一个 block 负责的分段长度（比 v0~v4 的 1024 大 4 倍，理由见文件头）
constexpr int TILES = N / SEG;     // 8192 个 block
constexpr int BLOCK = 256;         // 线程数保持 256
constexpr int WARPS = BLOCK / 32;  // 8 个 warp
constexpr int VPT   = SEG / 4 / BLOCK;  // 每线程 4 个 float4（= 16 个元素）
static_assert(N % SEG == 0, "为简化代码这里要求整除；补 0 到 SEG 整数倍即可，scan 不依赖越界元素");
static_assert(WARPS <= 32, "第 ③ 层要用一个 warp 扫完所有 warp 的小计，所以 warp 数不能超过 32");

// tile 状态。INVALID 必须是 0，这样 cudaMemset 清零就是初始状态。
// 用宏而不是 constexpr int，是因为 device 代码里也要用（constexpr 变量默认只在 host 端有定义）。
#define FLAG_INVALID   0  // 什么都还没发布
#define FLAG_AGGREGATE 1  // 只知道本段小计
#define FLAG_PREFIX    2  // 已经知道完整前缀了

// flag 和 value 打包在一起，__align__(8) 保证一条 8 字节指令就能整体读写（见文件头说明）。
struct __align__(8) TileState {
    float value;
    int   flag;
};

// 8 字节整体读/写。走 volatile 是为了每次都真的访存（绕过 L1，看得见别的 SM 写的值），
// 也防止编译器把"读一次"提到循环外面变成死循环。
// 借 unsigned long long 搬运：volatile 结构体不能直接赋值（没有 volatile 版的拷贝构造）。
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

// ---------------- block 内 scan：在 v4 的基础上，每线程多管几个 float4 ----------------

__device__ __forceinline__ float warp_inclusive_scan(float v) {
    for (int d = 1; d < 32; d *= 2) {
        float t = __shfl_up_sync(0xFFFFFFFF, v, d);
        if ((threadIdx.x & 31) >= d) v += t;
    }
    return v;
}

// 把 vv 这 VPT 个 float4（= SEG/BLOCK 个元素）就地换成"本 block 内的 exclusive scan"，
// 返回整个 block（SEG 个元素）的总和。三层结构和 v4 完全一样，只是第 ① 层从
// "扫 1 个 float4"变成"串行扫过自己的 VPT 个 float4"。
__device__ float block_scan_float4(float4* vv, float* s_warp) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // ---- ① 线程内：串行 exclusive scan 自己那 VPT*4 个元素，全程在寄存器里 ----
    float thread_total = 0.0f;  // 边扫边累，扫完就是这个线程的总和
    for (int i = 0; i < VPT; ++i) {
        float4& v = vv[i];
        // 先存下原值，再原地改成 exclusive（改完就读不到原值了）
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
        if (lane == WARPS - 1) s_warp[WARPS] = s;          // 顺手存下 block 总和
    }
    __syncthreads();

    // ---- 合成：三层前缀相加，摊到自己那 VPT*4 个元素上 ----
    const float base = s_warp[warp] + thread_prefix;
    for (int i = 0; i < VPT; ++i) {
        vv[i].x += base; vv[i].y += base; vv[i].z += base; vv[i].w += base;
    }
    return s_warp[WARPS];
}

// ---------------- lookback：本版的主角 ----------------

// 回头问前面的 tile，凑出"我前面所有元素的和"。
//
// 由 warp 0 的 32 个 lane 一起执行：一轮看 32 个前驱（一个"窗口"），
// 不够就把窗口整体往前挪 32 个再看一轮。
//
// 为什么必须是一整个 warp、而不是 thread 0 一个人往前数：
//   直觉上"前面的 block 早跑完了，问一两个就能撞上 PREFIX"——实测不是这样。
//   SEG=1024（32768 个 tile）时，thread 0 平均要往前走 43 步才碰到一个 PREFIX：
//   同时驻留在 GPU 上的上千个 block 几乎同时发布 AGGREGATE，
//   而 PREFIX 只能从 tile 0 开始像多米诺一样往后传，后面的 block 问到的基本都是 AGGREGATE。
//   43 步串行 = 43 次访存延迟首尾相接，实测 1.43ms，比 v4 还慢 5 倍。
//   改成一个 warp 一轮看 32 个，43 步压成 2 轮 → 0.38ms。CUB 就是这么做的。
//   （再叠加"加大 SEG 减少 tile 数"，最终到 0.13ms，见文件头。）
__device__ float lookback_warp(const int tile, const TileState* state) {
    const int lane = threadIdx.x & 31;
    float exclusive = 0.0f;

    for (int window = tile - 32; ; window -= 32) {
        const int p = window + lane;  // 本 lane 负责盯的那个前驱

        // 自旋等，直到窗口里【所有】前驱都发布了东西。
        // __any_sync 让 32 个 lane 步调一致地退出，后面的 ballot / reduce 才有意义。
        TileState s;
        do {
            if (p < 0) {  // 数组头之前，当成"前缀 = 0"的 PREFIX
                s.value = 0.0f;
                s.flag  = FLAG_PREFIX;
            } else {
                s = load_state(&state[p]);  // 一条 8 字节读，flag 和 value 必然是同一时刻的
            }
        } while (__any_sync(0xFFFFFFFF, s.flag == FLAG_INVALID));

        // 找出窗口里【最靠后】的那个 PREFIX：它已经把它前面的全部算进去了，
        // 所以只需要"它的 value + 它后面那些 tile 的 value"，它之前的都不用管。
        const unsigned prefix_mask = __ballot_sync(0xFFFFFFFF, s.flag == FLAG_PREFIX);
        const int cut = (prefix_mask == 0) ? 0 : (31 - __clz(prefix_mask));

        // 把 lane >= cut 的值加起来（蝶形 reduce，5 轮，结果广播到所有 lane）
        float sum = (lane >= cut) ? s.value : 0.0f;
        for (int d = 16; d > 0; d >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, d);
        exclusive += sum;

        // 窗口里有 PREFIX 就收工。p < 0 的 lane 永远报 PREFIX，
        // 所以窗口挪到数组头之前时必然退出，不会无限循环。
        if (prefix_mask != 0) return exclusive;
    }
}

__global__ void scan_single_pass_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                                        TileState* state) {
    __shared__ float s_warp[WARPS + 1];
    __shared__ float s_exclusive;  // thread 0 问回来的结果，广播给全 block

    const int tile = blockIdx.x;
    const size_t base4 = (size_t)tile * (SEG / 4);  // 本 tile 在 float4 视角下的起点

    // ---- 1. 先把自己那段扫完 ----
    // 每个线程拿【连续】的 VPT 个 float4（blocked 排布），这样第 ① 层串行扫的
    // 就是自己那 16 个连续元素，语义直接对得上。
    //
    // 代价是访存不合并：一个 lane 连读 4 个 float4 = 64 字节，相邻 lane 的起始地址
    // 就相隔 64 字节，一个 warp 的 32 条 lane 没有两条能并进同一个事务。
    // 直接换成 striped 排布（每个线程取 i*BLOCK+tid）访存漂亮，但线程手里的元素
    // 不再连续，串行扫出来的前缀就不是这一段的前缀了，结果是错的。
    // v6 用 shared memory 转置一下解决这个矛盾：载入按 striped、计算按 blocked。
    float4 v[VPT];
    for (int i = 0; i < VPT; ++i) v[i] = in[base4 + threadIdx.x * VPT + i];
    float total = block_scan_float4(v, s_warp);

    // ---- 2~4. 由 warp 0 负责跟别的 block 打交道（lookback 需要一整个 warp 协作）----
    if (threadIdx.x < 32) {
        // 先发小计：后面的 block 就算问到我，也能拿走它继续往前推，不必干等。
        if (threadIdx.x == 0 && tile > 0) store_state(&state[tile], total, FLAG_AGGREGATE);
        __syncwarp();

        // tile 0 前面没人，不用问，前缀就是 0
        float exclusive = (tile == 0) ? 0.0f : lookback_warp(tile, state);

        // 升级成 PREFIX：后面的 block 问到我就能一步收工，不用再往前走
        if (threadIdx.x == 0) {
            store_state(&state[tile], exclusive + total, FLAG_PREFIX);
            s_exclusive = exclusive;  // 广播给全 block
        }
    }
    __syncthreads();  // 等 warp 0 把 s_exclusive 填好

    // ---- 5. 摊到自己那 VPT*4 个元素上，写出 ----
    const float e = s_exclusive;
    for (int i = 0; i < VPT; ++i) {
        float4 t = v[i];
        t.x += e; t.y += e; t.z += e; t.w += e;
        out[base4 + threadIdx.x * VPT + i] = t;
    }
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
    // d_state 不用在这里清零：每次 launch 前都要清，所以清零放在 benchmark 的 lambda 里。

    printf("scan v5 single-pass lookback  [全局 scan]  (N = %d, SEG = %d, block = %d, tiles = %d)\n",
           N, SEG, BLOCK, TILES);

    float ms = benchmark([&] {
        // flag 必须每次都清回 INVALID(0)，否则第二次 launch 会读到上一次的残留状态
        CUDA_CHECK(cudaMemset(d_state, 0, (size_t)TILES * sizeof(TileState)));
        scan_single_pass_kernel<<<TILES, BLOCK>>>((const float4*)d_in, (float4*)d_out, d_state);
    });
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    report_perf("time (1 kernel)", ms, 2);  // 读 in 一次 + 写 out 一次，回到理论下限
    bool pass = verify(h_out, h_ref, N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_state));
    free(h_in); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
