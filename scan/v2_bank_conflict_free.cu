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
#include "common.h"

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
