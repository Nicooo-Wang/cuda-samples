// v2: 在 v1 基础上消除 shared memory bank conflict —— 只改下标，算法一个字没动。
//     对应 GPU Gems 3 第 39 章 Listing 39-3 / 39-4。
//
// 先说 v1 到底堵在哪：
//   shared memory 分成 32 个 bank（bank = addr/4 % 32，每个 bank 一次只能服务一个地址）。
//   v1 的树形访问 bi = offset*(2*tid+2)-1 里，相邻线程的地址差 2*offset 个 float：
//       offset=1  → 差 2  → 2 路冲突
//       offset=16 → 差 32 → 一整个 warp 全落在同一个 bank，32 路冲突，硬件串行访问 32 次
//   越到树的中间层越严重。这就是 v1 明明比 v0 少做 25% 的指令、跑出来却更慢的原因。
//   ncu 实测（N=32M，bank conflict 总数）：v0 = 0.9M，v1 = 41M，本版 = 0.37M。
//   时间：v0 = 0.34ms → v1 = 0.40ms → 本版 = 0.27ms。
//   也就是说 work-efficient 省下的算力，一直到这一版修好访存才真正拿到手。
//
// 怎么修：给下标掺一点 padding，把地址挤到不同的 bank 上。
//   把每 32 个 float 之后插一个空槽（下标每过 32 就 +1），
//   原来差 32 的两个地址就变成差 33，33 % 32 = 1 → 落在相邻 bank，冲突消失。
//   代价是 shared memory 多用 SEG/32 = 32 个 float，可以忽略。
//
// 【对文档的更正】文档写的是 NUM_BANKS = 16 / LOG_NUM_BANKS = 4，
//   那是 2007 年 G80 的参数（当时 shared memory 按半个 warp 16 个 bank 组织）。
//   从 Fermi 起一直到现在的 Hopper(sm_90) 都是 32 个 bank，所以这里取 LOG_NUM_BANKS = 5。
//   照抄文档的 4 只能消掉一半冲突。
//
// 【这一版其实改了两件事】文档把它们捆在一起给了，这里如实说明：
//   1) shared memory 加 padding（上面讲的，Block B/C/D 各加一次 CONFLICT_FREE_OFFSET）
//   2) Block A/E 的载入/写回从"每线程两个相邻元素"改成"两个半区"（tid 和 tid + n/2）：
//      v1 里 512 个线程访问 src[0..1023]，每个线程读相邻两个 float，一个 warp 覆盖 64 个 float
//      但是以 2 为跨步交错的；改成半区之后，一个 warp 的 32 个线程读连续的 32 个 float，
//      两次访问各自完全合并。这一条改的是全局访存效率，和 bank conflict 无关。
//
// 没变的：还是分段 scan，还是 2n 次加法，还是 2N 全局访存（读一次写一次）。
//
// 本版的毛病（v3 解决）：
//   每段只有 1024 个元素，段与段之间还是各算各的——这还不是真正的全局前缀和。
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
constexpr int BLOCK = SEG / 2;    // 512：2 元素/线程
static_assert(N % SEG == 0, "为简化代码这里要求整除；补 0 到 SEG 整数倍即可，scan 不依赖越界元素");

// bank 数量的 log2。sm_90 是 32 个 bank → 5（文档写的 4 是 G80 时代的值，见文件头说明）。
constexpr int LOG_NUM_BANKS = 5;
// 下标每过 32 就多插一个空槽。
// 文档的宏还有一个 >> (2*LOG_NUM_BANKS) 项，用来防"padding 本身又撞 bank"，
// 那是给远大于 1024 的数组准备的；SEG=1024 时那一项恒为 0，这里省掉以免干扰阅读。
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)
// 加了 padding 之后 shared memory 要开大一点
constexpr int SMEM_SIZE = SEG + CONFLICT_FREE_OFFSET(SEG - 1) + 1;

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
// 和 v1 逐块对照着看：Block A~E 五处，每处都只是在下标上加了一点东西。

__global__ void scan_conflict_free_kernel(const float* __restrict__ in, float* __restrict__ out,
                                          int n) {
    __shared__ float temp[SMEM_SIZE];  // 比 n 多 32 个槽，就是那些 padding

    const int tid = threadIdx.x;
    const float* src = in + (size_t)blockIdx.x * n;
    float* dst = out + (size_t)blockIdx.x * n;

    // ---- Block A：改成"两个半区"，一个 warp 读连续 32 个 float，完全合并 ----
    int ai = tid;          // 前半区
    int bi = tid + n / 2;  // 后半区
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);  // 存下来，Block E 写回时还要用
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = src[ai];
    temp[bi + bankOffsetB] = src[bi];

    // ---- 上扫 / reduce：和 v1 一模一样，只是 ai/bi 各加了一次 padding 偏移 ----
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            // ---- Block B ----
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            a += CONFLICT_FREE_OFFSET(a);  // 就这两行是新加的
            b += CONFLICT_FREE_OFFSET(b);
            temp[b] += temp[a];
        }
        offset *= 2;
    }

    // ---- Block C：清根。位置也要跟着 padding 走 ----
    if (tid == 0) temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0.0f;

    // ---- 下扫 / down-sweep：同样只是加了 padding 偏移 ----
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            // ---- Block D：和 Block B 相同的下标公式 ----
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            a += CONFLICT_FREE_OFFSET(a);
            b += CONFLICT_FREE_OFFSET(b);
            float t   = temp[a];
            temp[a]   = temp[b];
            temp[b]  += t;
        }
    }
    __syncthreads();

    // ---- Block E：和 Block A 对称，两个半区写回，两次都完全合并 ----
    dst[ai] = temp[ai + bankOffsetA];
    dst[bi] = temp[bi + bankOffsetB];
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

    printf("scan v2 bank-conflict-free  [分段 scan]  (N = %d, SEG = %d, block = %d, tiles = %d)\n",
           N, SEG, BLOCK, TILES);

    float ms = benchmark([&] {
        scan_conflict_free_kernel<<<TILES, BLOCK>>>(d_in, d_out, SEG);
    });
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    report_perf("time", ms, 2);  // 读 in 一次 + 写 out 一次
    bool pass = verify(h_out, h_ref, N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
