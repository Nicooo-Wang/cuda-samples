// v3: 任意长度的【全局 scan】—— 三个 kernel 把 32768 个独立分段拼成一条完整的前缀和。
//     对应 GPU Gems 3 第 39 章 39.2.4 节。
//
// v0~v2 都只是分段 scan：每 1024 个元素各算各的，out[1024] 又从 0 开始。
// 这一版才第一次算出真正的前缀和：out[k] = in[0] + ... + in[k-1]，k 一路数到 32M。
//
// 关键观察：每一段少加的，正好是"它前面所有段的总和"这一个常数。
//   所以把段总和自己再 scan 一遍，就知道每段该补多少了。三步：
//
//   in     [ 3 1 2 | 5 0 1 | 4 2 2 ]        （为了看得清，这里假装 SEG=3）
//     ↓ ① scan_tiles：每段独立做 exclusive scan，顺手把段总和存到 sums
//   out    [ 0 3 4 | 0 5 5 | 0 4 6 ]
//   sums   [   6   |   6   |   8   ]
//     ↓ ② scan_sums：对 sums 自己做 exclusive scan → 每段要补的偏移量
//   incr   [   0   |   6   |  12   ]
//     ↓ ③ uniform_add：第 i 段的每个元素都加上 incr[i]
//   out    [ 0 3 4 | 6 11 11 | 12 16 18 ]   ← 这才是全局 exclusive scan
//
// 代码上的复用：①③ 用的 block 内 scan 和 v2 一字不差，
//   所以把它抽成 __device__ blelloch_block_scan()，① 和 ② 都调它，正文只剩"三步怎么串"。
//   多带一个 carry 参数，② 里跨轮累进前缀要用（见那个 kernel 的注释）。
//
// 【代价：全局访存从 2N 翻到 4N】
//   ① 读 in 写 out（2N）+ ③ 读 out 改 out（又 2N）= 4N。
//   effective bw（按 2N 下限算）于是只有实际带宽的一半，
//   这正是 v5 要解决的问题——它用一个 kernel 干完全部，回到 2N。
//
// ② 只有一个 block 在干活，是不是浪费？
//   它要处理 N/SEG = 32768 个数，占总数据量的 0.1%，先不用管。
//   但它确实是个串行瓶颈（一个 block 只能用 132 个 SM 里的 1 个），
//   而 v5 的 decoupled lookback 干掉的就是它。
//
// 实测 0.46ms，是六版里最慢的——这一版把 block 内算法（还是 v2 的 Blelloch 树）
//   和"多 kernel 拼接"两件事叠在一起了，慢的主要是前者：v4 只换掉块内实现，
//   同样是三个 kernel、同样 4N 访存，就能跑到 0.17ms。
#include "common.h"

constexpr int SEG   = 1024;      // 一个 block 负责的分段长度
constexpr int TILES = N / SEG;    // 32768 个 block
constexpr int BLOCK = SEG / 2;    // 512：2 元素/线程
static_assert(N % SEG == 0, "为简化代码这里要求整除；补 0 到 SEG 整数倍即可，scan 不依赖越界元素");

constexpr int LOG_NUM_BANKS = 5;  // sm_90 是 32 个 bank（文档写的 4 是 G80 时代的值，见 v2）
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)
constexpr int SMEM_SIZE = SEG + CONFLICT_FREE_OFFSET(SEG - 1) + 1;

// sums 数组补齐到 SEG 的整数倍，多出来的位置填 0：
// 这样第 ② 步可以无脑按 SEG 分块，不用写任何越界判断（补的 0 对前缀和没影响）。
constexpr int SUMS_PADDED = CEIL(TILES, SEG) * SEG;

// ---------------- block 内 scan：从 v2 原样搬过来，只多了 carry 和"返回总和" ----------------

// 对 src[0..n) 做 exclusive scan，每个结果加上 carry 后写进 dst，返回这一段的总和。
// 下标算法和 v2 完全一致（两半区载入 + CONFLICT_FREE_OFFSET padding），不再重复注释。
// 要求：blockDim.x == n/2，temp 至少 SMEM_SIZE 个 float。
__device__ float blelloch_block_scan(float* temp, const float* src, float* dst, int n,
                                     float carry) {
    const int tid = threadIdx.x;
    const int ai = tid, bi = tid + n / 2;
    const int offA = CONFLICT_FREE_OFFSET(ai), offB = CONFLICT_FREE_OFFSET(bi);
    const int last = n - 1 + CONFLICT_FREE_OFFSET(n - 1);

    temp[ai + offA] = src[ai];
    temp[bi + offB] = src[bi];

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {  // 上扫
        __syncthreads();
        if (tid < d) {
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            a += CONFLICT_FREE_OFFSET(a);
            b += CONFLICT_FREE_OFFSET(b);
            temp[b] += temp[a];
        }
        offset *= 2;
    }
    __syncthreads();

    // 上扫跑完 temp[last] 就是整段总和。下扫会把它冲掉，所以先抢下来当返回值。
    // 全 block 读同一个地址 → 硬件广播，没有 bank conflict。
    const float total = temp[last];
    __syncthreads();               // 等所有线程都读完，thread 0 才能覆盖它
    if (tid == 0) temp[last] = 0.0f;

    for (int d = 1; d < n; d *= 2) {  // 下扫
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            a += CONFLICT_FREE_OFFSET(a);
            b += CONFLICT_FREE_OFFSET(b);
            float t  = temp[a];
            temp[a]  = temp[b];
            temp[b] += t;
        }
    }
    __syncthreads();

    dst[ai] = temp[ai + offA] + carry;
    dst[bi] = temp[bi + offB] + carry;
    return total;
}

// ---------------- 三个 kernel ----------------

// ① 每个 block 独立 scan 自己那一段，并把段总和写进 sums
__global__ void scan_tiles(const float* __restrict__ in, float* __restrict__ out,
                           float* __restrict__ sums, int n) {
    __shared__ float temp[SMEM_SIZE];
    const size_t base = (size_t)blockIdx.x * n;
    float total = blelloch_block_scan(temp, in + base, out + base, n, 0.0f);
    if (threadIdx.x == 0) sums[blockIdx.x] = total;
}

// ② 对 sums 自己做 exclusive scan（原地）。单个 block，按 SEG 分块循环。
//
// 为什么不递归调用整套三 kernel（文档的做法）？
//   count = 32768，分成 32 块，一个 block 串着跑 32 轮就完了：
//   carry 是个寄存器变量，跨轮累进，比"再开一层 sums/incr 数组 + 再来一次 uniform_add"简单得多。
//   而且这个"局部 scan + carry 往后传"的形状，正是 v5 里 decoupled lookback 的雏形——
//   区别只在于：这里是一个 block 串行地传 carry，v5 是 32768 个 block 并行地互相要 carry。
__global__ void scan_sums_one_block(float* sums, int count, int n) {
    __shared__ float temp[SMEM_SIZE];
    float carry = 0.0f;  // 前面所有块的总和；全 block 每个线程都持有同一份
    for (int base = 0; base < count; base += n) {
        carry += blelloch_block_scan(temp, sums + base, sums + base, n, carry);
        __syncthreads();  // 下一轮要复用 temp，先等所有线程把这一轮的结果写完
    }
}

// ③ 第 i 段的每个元素加上 incr[i]
__global__ void uniform_add(float* __restrict__ out, const float* __restrict__ incr, int n) {
    const float add = incr[blockIdx.x];  // 全 block 同一个值
    float* p = out + (size_t)blockIdx.x * n;
    p[threadIdx.x]         += add;
    p[threadIdx.x + n / 2] += add;
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

    printf("scan v3 arbitrary length  [全局 scan]  (N = %d, SEG = %d, block = %d, tiles = %d)\n",
           N, SEG, BLOCK, TILES);

    float ms = benchmark([&] {
        scan_tiles<<<TILES, BLOCK>>>(d_in, d_out, d_sums, SEG);
        scan_sums_one_block<<<1, BLOCK>>>(d_sums, SUMS_PADDED, SEG);
        uniform_add<<<TILES, BLOCK>>>(d_out, d_sums, SEG);
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
