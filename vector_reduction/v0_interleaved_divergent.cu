// v0: 最朴素的树形归约（interleaved addressing）
// 每个 block 处理 256 个元素，stride 从 1 开始翻倍，只有 tid 是 2*stride 倍数的线程干活。
//
// 问题：if (tid % (2 * stride) == 0) 让 warp 内大部分线程闲置。
// 第一轮 32 个线程里只有 16 个干活，第二轮只剩 8 个……而 warp 是整体调度的，
// 闲置线程也要陪着走完指令，这就是 warp divergence。
//
// 实测 0.255 ms / 263 GB/s，是八版里最慢的。作为参照，这台机器纯读 64MB 的上限是
// 3135 GB/s——也就是说这一版只用到了硬件的 8%，后面七版就是来补这 92% 的。
#include "common.h"

constexpr int BLOCK_SIZE = 256;

__global__ void reduce_v0(const float* in, float* out, int n) {
    __shared__ float s[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    s[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = s[0];
}

int main() {
    const size_t bytes = (size_t)N * sizeof(float);
    float* h_in = (float*)malloc(bytes);
    fill_inputs(h_in, N);
    double cpu_sum = cpu_reference(h_in, N);

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, (size_t)CEIL(N, BLOCK_SIZE) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_in, bytes, cudaMemcpyHostToDevice));

    printf("v0 interleaved (divergent)  N = %d\n", N);

    float ms = benchmark([&] {
        int cur = N;
        float *src = d_a, *dst = d_b;
        while (cur > 1) {
            const int grid = CEIL(cur, BLOCK_SIZE);
            reduce_v0<<<grid, BLOCK_SIZE>>>(src, dst, cur);
            cur = grid;
            float* t = src; src = dst; dst = t;
        }
    });

    // benchmark 把 d_a/d_b 里的数据弄乱了，重新跑一次完整的归约（不计时）获取正确结果
    CUDA_CHECK(cudaMemcpy(d_a, h_in, bytes, cudaMemcpyHostToDevice));
    int cur = N;
    float *src = d_a, *dst = d_b;
    while (cur > 1) {
        const int grid = CEIL(cur, BLOCK_SIZE);
        reduce_v0<<<grid, BLOCK_SIZE>>>(src, dst, cur);
        cur = grid;
        float* t = src; src = dst; dst = t;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, src, sizeof(float), cudaMemcpyDeviceToHost));

    report_perf("time", ms);
    bool pass = check_result(gpu_sum, cpu_sum, h_in, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_in);
    return pass ? 0 : 1;
}
