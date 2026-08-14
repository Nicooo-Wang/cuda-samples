// v5: grid-stride loop —— 固定 grid 大小，每个线程串行吃掉多个元素
//
// 前面几版的 grid 随 N 变化（N=16M 时第一趟就有 32768 个 block），
// 而且需要反复启动 kernel 直到剩一个数。
//
// 这一版把 grid 固定成刚好填满 GPU 的大小，每个线程用 grid-stride loop
// 在寄存器里累加自己那一份，然后整个 grid 只剩 gridDim.x 个部分和。
// 两趟就能结束：第一趟 N -> grid，第二趟 grid -> 1。
//
// 好处：
//   - kernel launch 次数固定（2 次），不随 N 增长
//   - 线程在寄存器里累加，中间结果不落 shared/global
//   - grid-stride 的访存模式天然合并（相邻线程访问相邻地址）
//
// 实测 0.044 -> 0.038 ms（1540 -> 1771 GB/s），1.15 倍。
// 结构上这一版已经对了，但离纯读上限 3135 GB/s 还差 1.8 倍——
// 差距在载入指令的粒度上（每次只取一个 float），v6 换成 float4 解决。
#include "common.h"

constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduce_v5(const float* in, float* out, int n) {
    __shared__ float s[WARPS_PER_BLOCK];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;

    // grid-stride loop：每个线程跨 gridDim.x * blockDim.x 步长扫描整个数组。
    // 同一时刻 warp 内 32 个线程访问连续的 32 个 float，合并访存。
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        sum += in[i];
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

int main() {
    const size_t bytes = (size_t)N * sizeof(float);
    float* h_in = (float*)malloc(bytes);
    fill_inputs(h_in, N);
    double cpu_sum = cpu_reference(h_in, N);

    // grid 取"刚好填满 GPU"：每个 SM 跑若干个 block
    int dev = 0, num_sm = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev));
    const int grid = num_sm * 8;

    float *d_in, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_partial, (size_t)grid * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("v5 grid-stride loop  N = %d, grid = %d (%d SMs x 8)\n", N, grid, num_sm);

    float ms = benchmark([&] {
        // 第一趟：N 个元素 -> grid 个部分和
        reduce_v5<<<grid, BLOCK_SIZE>>>(d_in, d_partial, N);
        // 第二趟：grid 个部分和 -> 1 个（单 block，grid-stride loop 自己会兜住剩余元素）
        reduce_v5<<<1, BLOCK_SIZE>>>(d_partial, d_partial, grid);
    });

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_partial, sizeof(float), cudaMemcpyDeviceToHost));

    report_perf("time", ms);
    bool pass = check_result(gpu_sum, cpu_sum, h_in, N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    free(h_in);
    return pass ? 0 : 1;
}
