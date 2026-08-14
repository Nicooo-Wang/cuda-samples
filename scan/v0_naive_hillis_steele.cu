// v0: 朴素并行 scan（Hillis-Steele），双缓冲避免原地读写竞争。
//     对应 GPU Gems 3 第 39 章 Algorithm 2 / Listing 39-1。
//
// 什么是 exclusive scan（前缀和）：
//   out[0] = 0,  out[k] = in[0] + in[1] + ... + in[k-1]
//   和 reduction 的区别：reduction 只输出一个数，scan 每个位置都要输出自己"之前所有元素的和"。
//   串行写法一行就够（out[k] = out[k-1] + in[k-1]），但它有 100% 的串行依赖——
//   怎么把这个依赖拆开并行掉，就是这一整套教程要讲的事。
//
// 本版算的是【分段 scan】：每 SEG=1024 个元素独立算一段，段与段之间不通信。
//   这正是文档前三节的"单 block 算法"，v0~v2 都停在这里；
//   跨段拼接（也就是真正的全局 scan）从 v3 开始讲。
//   用 N/SEG = 16384 个 block 跑，是为了把 GPU 填满，让带宽数字有意义。
//
// 算法（每个 block 内部，1 线程 1 元素）：
//   载入时整体右移一位（thread 0 填 0）→ 得到 exclusive 语义
//   for offset = 1, 2, 4, ..., 512:
//       tid >= offset 的线程：out[tid] = in[tid] + in[tid-offset]
//       其余线程：          out[tid] = in[tid]              （原样拷过去）
//   10 轮之后每个位置就攒够了自己前面所有元素。
//
// 为什么要双缓冲（两块 shared memory 轮流当输入/输出）：
//   如果原地做（temp[tid] += temp[tid-offset]），同一个 block 里 1024 个线程分 32 个 warp
//   并不是同时执行的。warp 1 可能已经写完了 temp[32..63]，warp 0 才开始读 temp[0..31]——
//   于是有的线程读到的是"这一轮更新后"的值，有的是更新前的值，结果就错了。
//   文档的 Algorithm 1 就是原地版本，它在纸上成立、在真实硬件上是错的。
//   双缓冲让"读"和"写"落在两块不同的内存上，天然没有这个竞争。
//
// 本版的毛病（后面几版逐一解决）：
//   1) 【最要命】加法次数是 O(n log n)：10 轮 × 1024 线程 ≈ 10n 次加法，
//      而串行 CPU 只需要 n 次。多出来的这个 log 因子是纯浪费的算力。
//      → v1 换成 Blelloch 算法，降到 2n，这叫 work-efficient。
//   2) 每轮都有 offset 个线程只是在做无意义的拷贝，越到后面闲的越多。
//   3) 双缓冲要 2 倍 shared memory（8KB），而 v1 原地做只要 4KB。
//   注意：本版全局访存只有 2N（读一次写一次），已经是理论下限了，
//   但实测带宽只有 ~790 GB/s，而这台机器纯拷贝能跑 3694 GB/s——差了近 5 倍。
//   说明 v0~v3 全都不是带宽瓶颈，卡在片上（加法次数、smem 访问、barrier）。
//   所以这几版的优化全部发生在片上；带宽要到 v4 之后才成为真正的天花板。
#include "common.h"

constexpr int SEG   = 1024;      // 一个 block 负责的分段长度
constexpr int TILES = N / SEG;    // 32768 个 block
constexpr int BLOCK = SEG;        // 本版 1 线程 1 元素
static_assert(N % SEG == 0, "为简化代码这里要求整除；补 0 到 SEG 整数倍即可，scan 不依赖越界元素");

// ---------------- kernel：本版唯一的主角 ----------------

__global__ void scan_naive_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float buf[2][SEG];  // 双缓冲，两块轮流当输入/输出

    const int tid = threadIdx.x;
    const float* src = in + (size_t)blockIdx.x * n;
    float* dst = out + (size_t)blockIdx.x * n;

    // 载入时整体右移一位：buf[tid] = src[tid-1]，thread 0 填 0。
    // 这一步就把 inclusive 变成了 exclusive——不需要在最后再移一次。
    int pin = 0, pout = 1;
    buf[pout][tid] = (tid > 0) ? src[tid - 1] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pin = pout;       // 上一轮的输出变成这一轮的输入
        pout = 1 - pout;  // 写到另一块去，读写不撞车

        if (tid >= offset)
            buf[pout][tid] = buf[pin][tid] + buf[pin][tid - offset];
        else
            buf[pout][tid] = buf[pin][tid];  // 前 offset 个线程无事可做，只能原样拷过去

        __syncthreads();  // 全 block 都写完这一轮，才能进入下一轮
    }

    dst[tid] = buf[pout][tid];
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

    printf("scan v0 naive hillis-steele  [分段 scan]  (N = %d, SEG = %d, block = %d, tiles = %d)\n",
           N, SEG, BLOCK, TILES);

    float ms = benchmark([&] {
        scan_naive_kernel<<<TILES, BLOCK>>>(d_in, d_out, SEG);
    });
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    report_perf("time", ms, 2);  // 读 in 一次 + 写 out 一次
    bool pass = verify(h_out, h_ref, N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
