// v4: 用 warp shuffle + float4 重写 block 内 scan —— 三个 kernel 的结构和 v3 完全一样。
//
// 到这里我们跟文档（2007 年，G80）告别了：__shfl_up_sync 是 Kepler(2012) 才有的指令，
// 现代 GPU 上的 scan 都长这个样子（CUB 也是），不再走 shared memory 树。
//
// 比 v3 改了什么：只换掉了 block 内那个 scan 原语。
//   v3（Blelloch 树）  ：20 次 __syncthreads()，1024 个中间值全程住在 shared memory
//   v4（三层 shuffle） ： 2 次 __syncthreads()，shared memory 只用来存 8 个 warp 的小计
//
// 三层结构（BLOCK=256，每线程一个 float4 = 4 个元素，正好 SEG=1024）：
//
//   ① 线程内：串行 scan 自己那 4 个（3 次加法，纯寄存器）
//        v = (a,b,c,d)  →  (0, a, a+b, a+b+c)，thread_total = a+b+c+d
//   ② warp 内：对 32 个 thread_total 做 shuffle scan（5 轮），得到"我前面 31 个线程的和"
//        __shfl_up_sync 直接在寄存器之间倒腾数据，不碰内存、不用同步
//   ③ block 内：8 个 warp 的总和进 shared memory，warp 0 再 shuffle scan 一次
//
//   最后每个线程的基准值 = ③给的 warp 前缀 + ②给的 warp 内前缀，加到 ① 的四个值上，
//   一次 float4 写出。
//
// 为什么 shuffle 比 shared memory 快：
//   - 不经过内存，warp 内寄存器直接交换，延迟低得多，也不存在 bank conflict
//   - warp 内天然同步（32 个线程锁步执行），5 轮里一次 __syncthreads() 都不用
//   - float4 让每个线程一条指令搬 16 字节，访存请求数少 4 倍
//
// 没变的：还是①②③三个 kernel、还是 4N 全局访存、还是全局 scan。
//   剩下的 4N→2N 是 v5 的事。
//
// 实测 0.17ms，比 v3 的 0.46ms 快 2.7 倍——只换了块内实现，kernel 结构一个字没动。
//   注意此时"实际搬运"已经到 3225 GB/s，很接近这台机器纯拷贝的 3820 GB/s 上限了：
//   也就是说这一版已经把带宽吃满，唯一还能省的就是那多余的 2N。
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

#define CEIL(a, b) (((a) + (b) - 1) / (b))

constexpr int N     = 1 << 25;    // 32M 个 float = 128MB。这台机器 L2 有 60MB，
                                  // 输入+输出 256MB 才够把 L2 甩开，否则测的是 L2 带宽不是 HBM
constexpr int SEG   = 1024;       // 一个 block 负责的分段长度
constexpr int TILES = N / SEG;     // 32768 个 block
constexpr int BLOCK = SEG / 4;     // 256：本版每线程一个 float4
constexpr int WARPS = BLOCK / 32;  // 8 个 warp
static_assert(N % SEG == 0, "为简化代码这里要求整除；补 0 到 SEG 整数倍即可，scan 不依赖越界元素");
static_assert(WARPS <= 32, "第 ③ 层要用一个 warp 扫完所有 warp 的小计，所以 warp 数不能超过 32");

// sums 数组补齐到 SEG 的整数倍，多出来的位置填 0（同 v3）
constexpr int SUMS_PADDED = CEIL(TILES, SEG) * SEG;

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
// 本版传 seg = n，也就是退化成一整条全局 scan。
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

// ---------------- block 内 scan：本版的主角 ----------------

// warp 内 inclusive scan：5 轮 shuffle。
// __shfl_up_sync(mask, val, d) = "把 lane-d 那个线程的 val 拿过来"（lane < d 的拿不到，保持原值）。
// 第 1 轮每个 lane 攒到 2 个值，第 2 轮 4 个，... 第 5 轮 32 个，正好扫完一个 warp。
__device__ __forceinline__ float warp_inclusive_scan(float v) {
    for (int d = 1; d < 32; d *= 2) {
        float t = __shfl_up_sync(0xFFFFFFFF, v, d);
        if ((threadIdx.x & 31) >= d) v += t;  // lane < d 的线程没有来源，不能加
    }
    return v;
}

// 把 v 这 4 个元素就地换成"本 block 内的 exclusive scan + carry"，返回整个 block（SEG 个元素）的总和。
// s_warp 需要 WARPS+1 个 float：前 WARPS 个存各 warp 的小计，最后一个借来存 block 总和。
// 全 block 必须一起进来（内部有 __syncthreads()），返回值对所有线程都有效。
__device__ float block_scan_float4(float4& v, float* s_warp, float carry) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // ---- ① 线程内：串行 exclusive scan 自己那 4 个 ----
    float thread_total = v.x + v.y + v.z + v.w;
    // 从后往前改，就不用临时变量：w 要的是 x+y+z，z 要的是 x+y，y 要的是 x，x 要的是 0
    v.w = v.x + v.y + v.z;
    v.z = v.x + v.y;
    v.y = v.x;
    v.x = 0.0f;

    // ---- ② warp 内：对 thread_total 做 scan，减掉自己那份就成了 exclusive ----
    float warp_scan = warp_inclusive_scan(thread_total);
    float thread_prefix = warp_scan - thread_total;  // 我前面那些线程的和

    // 每个 warp 的最后一个 lane 手里就是这个 warp 的小计
    if (lane == 31) s_warp[warp] = warp_scan;
    __syncthreads();

    // ---- ③ block 内：warp 0 扫一遍 WARPS 个小计，换成 exclusive 写回 ----
    if (warp == 0) {
        float t = (lane < WARPS) ? s_warp[lane] : 0.0f;
        float s = warp_inclusive_scan(t);
        if (lane < WARPS) s_warp[lane] = s - t;  // exclusive：第 i 个 warp 前面所有 warp 的和
        // lane == WARPS-1 的 s 是所有小计的 inclusive 和，也就是整个 block 的总和。
        // 顺手存下来，省得调用方自己再拼一遍。
        if (lane == WARPS - 1) s_warp[WARPS] = s;
    }
    __syncthreads();

    // ---- 合成：三层前缀相加，摊到自己那 4 个元素上 ----
    float base = carry + s_warp[warp] + thread_prefix;
    v.x += base; v.y += base; v.z += base; v.w += base;

    return s_warp[WARPS];  // 全 block 读同一个地址 → 广播，不花钱
}

// ---------------- 三个 kernel（结构和 v3 一样）----------------

// ① 每个 block 独立 scan 自己那一段，并把段总和写进 sums
__global__ void scan_tiles(const float4* __restrict__ in, float4* __restrict__ out,
                           float* __restrict__ sums) {
    __shared__ float s_warp[WARPS + 1];

    const size_t idx = (size_t)blockIdx.x * BLOCK + threadIdx.x;
    float4 v = in[idx];
    float total = block_scan_float4(v, s_warp, 0.0f);

    out[idx] = v;
    if (threadIdx.x == 0) sums[blockIdx.x] = total;
}

// ② 对 sums 自己做 exclusive scan（原地）。单个 block，按段分块循环，carry 跨轮累进。
//    和 v3 同一个思路，只是换成了 shuffle 版的 block scan。
__global__ void scan_sums_one_block(float4* sums, int count4) {
    __shared__ float s_warp[WARPS + 1];
    float carry = 0.0f;  // 前面所有段的总和；全 block 每个线程都持有同一份

    for (int base = 0; base < count4; base += BLOCK) {
        float4 v = sums[base + threadIdx.x];
        carry += block_scan_float4(v, s_warp, carry);
        sums[base + threadIdx.x] = v;
        __syncthreads();  // s_warp 下一轮要复用，等所有线程读完再进下一轮
    }
}

// ③ 第 i 段的每个元素加上 incr[i]
__global__ void uniform_add(float4* __restrict__ out, const float* __restrict__ incr) {
    const float add = incr[blockIdx.x];  // 全 block 同一个值
    const size_t idx = (size_t)blockIdx.x * BLOCK + threadIdx.x;
    float4 v = out[idx];
    v.x += add; v.y += add; v.z += add; v.w += add;
    out[idx] = v;
}

// ---------------- main：只做编排 ----------------

int main() {
    const size_t bytes = (size_t)N * sizeof(float);
    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);

    fill_input(h_in, N);
    cpu_exclusive_scan(h_in, h_ref, N, N);  // seg = N：这一版是全局 scan

    float *d_in, *d_out, *d_sums;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_sums, (size_t)SUMS_PADDED * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sums, 0, (size_t)SUMS_PADDED * sizeof(float)));  // 补位填 0

    printf("scan v4 warp shuffle  [全局 scan]  (N = %d, SEG = %d, block = %d, tiles = %d)\n",
           N, SEG, BLOCK, TILES);

    float ms = benchmark([&] {
        scan_tiles<<<TILES, BLOCK>>>((const float4*)d_in, (float4*)d_out, d_sums);
        scan_sums_one_block<<<1, BLOCK>>>((float4*)d_sums, SUMS_PADDED / 4);
        uniform_add<<<TILES, BLOCK>>>((float4*)d_out, d_sums);
    });
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    report_perf("time (3 kernels)", ms, 4);  // ① 读+写 2N，③ 读+写 2N
    bool pass = verify(h_out, h_ref, N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sums));
    free(h_in); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
