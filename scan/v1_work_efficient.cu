// v1: work-efficient scan（Blelloch）——上扫 + 下扫，加法次数从 O(n log n) 降到 O(n)。
//     对应 GPU Gems 3 第 39 章 Algorithm 3/4 / Listing 39-2。
//
// 比 v0 改了什么：只换了 block 内部的算法，别的都没动（还是分段 scan，还是 2N 全局访存）。
//   v0：10 轮 × 1024 线程 ≈ 10n 次加法
//   v1：2(n-1) 次加法 + (n-1) 次交换 ≈ 2n     ← 和串行 CPU 同一个数量级，这才叫 work-efficient
//
// 核心思路：把数组看成一棵平衡二叉树，走两趟。
//   【上扫 / reduce】自底向上，把每个子树的和攒到子树最右边那个位置。
//       跑完之后 temp[n-1] 就是整段的总和（这一趟其实就是一次树形 reduction）。
//   【下扫 / down-sweep】自顶向下，把根清零，然后每一层做一次"交换 + 加"：
//       左孩子拿走父亲的值，右孩子拿"父亲 + 左孩子原值"。
//       这样每个节点最终拿到的正好是"它左边所有叶子的和"，也就是 exclusive scan。
//
// 一段 n=1024、BLOCK=512（2 元素/线程）的实际形状：
//   上扫 10 轮，活跃线程数 512 → 256 → ... → 1；
//   下扫 10 轮反着来，1 → 2 → ... → 512。首轮 offset=512 → ai=511, bi=1023，正是根节点那一刀。
//
// 为什么这一版可以原地做（不像 v0 要双缓冲）：
//   每一轮里 ai / bi 的取值互不重叠——一个位置在同一轮里要么被读、要么被写，不会又读又写。
//   所以 v0 的读写竞争在这里天然不存在，shared memory 也就省了一半（4KB vs 8KB）。
//
// 【重要：本版实测比 v0 慢】少做了一大半加法，跑出来反而是 0.40ms vs v0 的 0.34ms。
//   为什么？树形 stride 的访问模式会狠狠地撞 shared memory bank conflict，越到树的中间层越严重。
//   比如上扫第 5 轮 offset=16，bi = 32*tid+31，相邻线程的地址差 32 个 float——
//   32 个 bank 一个循环，于是一整个 warp 全落在同一个 bank 上，32 路冲突，硬件串行访问 32 次。
//   ncu 实测（N=32M）：
//       指令数        v0 = 208M   →  v1 = 155M    少了 25%，work-efficient 确实起作用了
//       bank conflict v0 = 0.9M   →  v1 = 41M     涨了 45 倍！  ← 就是它把省下来的算力吃回去了
//   顺带解释了 v0 为什么冲突这么少：v0 的访问是 buf[tid] 和 buf[tid-offset]，
//   连续下标、天然无冲突；它慢是慢在纯粹的加法次数上。
//
//   这一版的价值在于"算法上正确的方向"——v2 只需在下标上加几个字，就能把冲突消掉，
//   拿到 0.27ms（比 v0 快 28%）。先换对算法，再修访存，两步都做完才见收益。
//   如果只看 v0→v1 就下结论"work-efficient 是骗人的"，那就学错了。
#include "common.h"

constexpr int SEG   = 1024;      // 一个 block 负责的分段长度
constexpr int TILES = N / SEG;    // 32768 个 block
constexpr int BLOCK = SEG / 2;    // 512：本版 2 元素/线程（树的每一层最多需要 n/2 个线程）
static_assert(N % SEG == 0, "为简化代码这里要求整除；补 0 到 SEG 整数倍即可，scan 不依赖越界元素");

// ---------------- kernel：本版唯一的主角 ----------------
//
// 代码块 A~E 的分块标注沿用文档 Listing 39-2 的叫法，
// v2 消除 bank conflict 时正好是逐块给这五处打补丁，对照着看能一眼看出改了哪几行。

__global__ void scan_work_efficient_kernel(const float* __restrict__ in, float* __restrict__ out,
                                           int n) {
    __shared__ float temp[SEG];  // 原地做，只要 1 份（v0 双缓冲要 2 份）

    const int tid = threadIdx.x;
    const float* src = in + (size_t)blockIdx.x * n;
    float* dst = out + (size_t)blockIdx.x * n;

    // ---- Block A：载入，每个线程管相邻的两个元素 ----
    temp[2 * tid]     = src[2 * tid];
    temp[2 * tid + 1] = src[2 * tid + 1];

    // ---- 上扫 / reduce：自底向上求和，跑完 temp[n-1] = 整段总和 ----
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {  // d = 这一层的活跃线程数：512, 256, ..., 1
        __syncthreads();
        if (tid < d) {
            // ---- Block B：算出这一层要合并的两个位置 ----
            int ai = offset * (2 * tid + 1) - 1;  // 左孩子（子树最右端）
            int bi = offset * (2 * tid + 2) - 1;  // 父节点（更大子树的最右端）
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // ---- Block C：把根清零，下扫才能得到 exclusive 语义 ----
    // 原来 temp[n-1] 存的是整段总和，这里丢掉它——因为 exclusive scan 里
    // 最后一个位置要的是"前 n-1 个的和"，而它会在下扫过程中被重新填上。
    if (tid == 0) temp[n - 1] = 0.0f;

    // ---- 下扫 / down-sweep：自顶向下分发前缀 ----
    for (int d = 1; d < n; d *= 2) {  // d = 这一层的活跃线程数：1, 2, ..., 512
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            // ---- Block D：和 Block B 完全一样的下标公式 ----
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            // 交换 + 加：左孩子拿走父亲的值，父亲位置变成"父亲 + 左孩子原值"
            float t   = temp[ai];
            temp[ai]  = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // ---- Block E：写回 ----
    dst[2 * tid]     = temp[2 * tid];
    dst[2 * tid + 1] = temp[2 * tid + 1];
}

// ---------------- main：只做编排 ----------------

int main() {
    const size_t bytes = (size_t)N * sizeof(float);
    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);

    fill_input(h_in, N);
    cpu_exclusive_scan(h_in, h_ref, N, SEG);  // 本版是分段 scan，参考也按 SEG 分段

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("scan v1 work-efficient  [分段 scan]  (N = %d, SEG = %d, block = %d, tiles = %d)\n",
           N, SEG, BLOCK, TILES);

    float ms = benchmark([&] {
        scan_work_efficient_kernel<<<TILES, BLOCK>>>(d_in, d_out, SEG);
    });
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    report_perf("time", ms, 2);  // 读 in 一次 + 写 out 一次
    bool pass = verify(h_out, h_ref, N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
