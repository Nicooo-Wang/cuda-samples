// v6: float4 向量化载入 —— 每个线程一条指令读 16 字节而不是 4 字节。
//
// v5 已经把结构做对了（固定 grid、寄存器累加、两趟结束），但它只跑到 1756 GB/s，
// 而这台机器纯读 64MB 的上限是 3135 GB/s——才用了 56%。
//
// 瓶颈在载入指令的粒度：v5 的 grid-stride loop 里是 `sum += in[i]`，
// 每个线程每次只取一个 float。一个 warp 32 条 lane 取 32 × 4 = 128 字节，
// 正好一个 cache line，访存是合并的没错，但【指令数】太多：
// 16M 个元素要发 16M / 32 = 52 万条 warp 级载入指令。
//
// 改成 float4 之后每条指令搬 16 字节，一个 warp 一次拿 512 字节：
//   - 载入指令数降到 1/4（52 万 -> 13 万）
//   - 每条指令的访存事务数从 4 个 32B sector 变成 16 个，但事务总数不变——
//     省的是指令发射和地址计算的开销，不是带宽本身
//   - 循环体里的加法从 1 次变成 4 次，而这 4 次在寄存器里，几乎免费
//
// 这就是为什么向量化对访存密集型 kernel 特别有效：带宽没变，但发指令的开销摊薄了。
// 同样的道理在 scan 的 v4 里也用过（那里是 block 内 scan 换成 float4）。
//
// 实测 0.0382 -> 0.0256 ms（1756 -> 2618 GB/s），1.5 倍，达到纯读上限的 84%。
//
// 前提条件：N 必须是 4 的整数倍，且首地址 16 字节对齐。
//   cudaMalloc 返回的指针至少 256 字节对齐，所以后一条自动满足；
//   前一条这里靠 N = 2^24 保证。真实场景要处理尾巴：
//   把 n/4*4 之前的部分用 float4 扫，剩下的 n%4 个元素单独加。
#include "common.h"

constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
static_assert(N % 4 == 0, "float4 载入要求 N 是 4 的整数倍");

__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 第一趟：float4 版的 grid-stride loop。n4 是 float4 的个数（= 元素数 / 4）。
__global__ void reduce_v6(const float4* __restrict__ in, float* __restrict__ out, int n4) {
    __shared__ float s[WARPS_PER_BLOCK];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;

    // 一条指令取 4 个 float，然后在寄存器里加起来。
    // 同一时刻 warp 内 32 条 lane 访问连续的 32 个 float4 = 512 字节，完全合并。
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n4; i += gridDim.x * blockDim.x) {
        const float4 v = in[i];
        sum += v.x + v.y + v.z + v.w;
    }

    sum = warp_reduce(sum);
    if (lane == 0) s[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < WARPS_PER_BLOCK) ? s[lane] : 0.0f;
        sum = warp_reduce(sum);
        if (lane == 0) out[blockIdx.x] = sum;
    }
}

// 第二趟：把 grid 个部分和归约成 1 个。数量太少（1056 个），
// 用标量版就够了，没必要为它再写一个 float4 版本。
__global__ void reduce_tail(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float s[WARPS_PER_BLOCK];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += in[i];

    sum = warp_reduce(sum);
    if (lane == 0) s[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < WARPS_PER_BLOCK) ? s[lane] : 0.0f;
        sum = warp_reduce(sum);
        if (lane == 0) out[0] = sum;
    }
}

int main() {
    const size_t bytes = (size_t)N * sizeof(float);
    float* h_in = (float*)malloc(bytes);
    fill_inputs(h_in, N);
    double cpu_sum = cpu_reference(h_in, N);

    int dev = 0, num_sm = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev));
    const int grid = num_sm * 8;

    float *d_in, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, (size_t)grid * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("v6 float4 vectorized  N = %d, grid = %d (%d SMs x 8)\n", N, grid, num_sm);

    float ms = benchmark([&] {
        reduce_v6<<<grid, BLOCK_SIZE>>>((const float4*)d_in, d_partial, N / 4);
        reduce_tail<<<1, BLOCK_SIZE>>>(d_partial, d_partial, grid);
    });

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_partial, sizeof(float), cudaMemcpyDeviceToHost));

    report_perf("time (2 kernels)", ms);
    bool pass = check_result(gpu_sum, cpu_sum, h_in, N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    free(h_in);
    return pass ? 0 : 1;
}
