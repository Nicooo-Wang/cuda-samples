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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

constexpr int N     = 1 << 25;   // 32M 个 float = 128MB。这台机器 L2 有 60MB，
                                 // 输入+输出 256MB 才够把 L2 甩开，否则测的是 L2 带宽不是 HBM
constexpr int SEG   = 1024;      // 一个 block 负责的分段长度
constexpr int TILES = N / SEG;    // 32768 个 block
constexpr int BLOCK = SEG / 2;    // 512：本版 2 元素/线程（树的每一层最多需要 n/2 个线程）
static_assert(N % SEG == 0, "为简化代码这里要求整除；补 0 到 SEG 整数倍即可，scan 不依赖越界元素");

// ---------------- 宿主端样板（fill / cpu 参考 / 校验 / 打印，四个独立函数）----------------

// 填 {-2,-1,0,1,2} 的小整数，固定种子保证各版本输入一致、结果可横向对比。
//
// 为什么刻意用小整数而不是 [-1,1) 的随机浮点：
//   前缀和是一次随机游走，量级大约 √N·σ ≈ 8e3，远小于 2^24（float 能精确表示的整数上限）。
//   于是不管求和顺序怎么变，每一个中间结果都是一个精确的 fp32 整数——
//   GPU 结果必须和 double 参考【逐位相等】，容差可以直接取 0。
//   这样任何 FAIL 都一定是真 bug（下标算错、同步漏了），而不是浮点噪声，省掉了调容差的功夫。
void fill_input(float* in, int n) {
    srand(0);
    for (int i = 0; i < n; ++i) in[i] = (float)(rand() % 5 - 2);
}

// CPU 参考：分段 exclusive scan。每 seg 个元素归零重新开始。
// 传 seg = n 就退化成全局 scan（v3 之后就是这么用的）。
void cpu_exclusive_scan(const float* in, float* out, int n, int seg) {
    double run = 0.0;  // 参考值用 double 累加，比被测的 float 更准
    for (int i = 0; i < n; ++i) {
        if (i % seg == 0) run = 0.0;  // 新的一段，前缀清零
        out[i] = (float)run;          // exclusive：先写"我之前的和"
        run += in[i];                 // 再把自己算进去
    }
}

// 逐位精确比较（见 fill_input 的说明，小整数输入下容差为 0）。
bool verify(const float* gpu, const float* ref, int n) {
    double max_diff = 0.0;
    int first_bad = -1;
    for (int i = 0; i < n; ++i) {
        double d = fabs((double)gpu[i] - (double)ref[i]);
        if (d > max_diff) max_diff = d;
        if (d != 0.0 && first_bad < 0) first_bad = i;
    }
    bool pass = (max_diff == 0.0);
    printf("  max abs diff    : %.1f", max_diff);
    if (first_bad >= 0) printf("  (first mismatch at i=%d: gpu=%.1f ref=%.1f)", first_bad,
                               gpu[first_bad], ref[first_bad]);
    printf("\n  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// passes = 这一版实际把整个数组读写了几遍。
// effective bw 恒按 2N（读一次+写一次，任何 scan 的理论下限）计算——
// 这是六个版本之间唯一可比的尺子；passes > 2 说明这一版有额外的来回搬运。
void report_perf(const char* tag, float ms, int passes) {
    double lower_bound_bytes = 2.0 * N * sizeof(float);
    printf("  %-16s: %.4f ms\n", tag, ms);
    printf("  effective bw    : %.1f GB/s   (按下限 2N 算)\n",
           lower_bound_bytes / (ms * 1e-3) / 1e9);
    if (passes != 2)
        printf("  actual traffic  : %.1f GB/s   (实际搬了 %dN)\n",
               (double)passes * N * sizeof(float) / (ms * 1e-3) / 1e9, passes);
}

// 计时：先热身一次，再取 5 次里最快的。
// 为什么要热身：第一次 launch 要付内核加载、页表建立等一次性开销，
// 实测能比稳定态慢一倍以上——只测一次冷启动，六个版本之间就没法比了。
// 取最快值而不是平均值：把偶发的调度抖动（别的进程抢 SM）排除掉。
template <typename Launch>
float benchmark(Launch launch) {
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    launch();  // 热身，不计时
    CUDA_CHECK(cudaDeviceSynchronize());

    float best = 1e30f;
    for (int r = 0; r < 5; ++r) {
        CUDA_CHECK(cudaEventRecord(ev0));
        launch();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return best;
}

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
