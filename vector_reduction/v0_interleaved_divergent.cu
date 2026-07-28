// v0: 最朴素的树形归约（interleaved addressing）
// 每个 block 处理 256 个元素，stride 从 1 开始翻倍，只有 tid 是 2*stride 倍数的线程干活。
//
// 问题：if (tid % (2 * stride) == 0) 让 warp 内大部分线程闲置。
// 第一轮 32 个线程里只有 16 个干活，第二轮只剩 8 个……而 warp 是整体调度的，
// 闲置线程也要陪着走完指令，这就是 warp divergence。
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

constexpr int N = 1 << 24;       // 16M 个元素
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

    // 输入全填 1.0：和恰好是 N = 2^24，float 能精确表示，
    // 所以可以要求 GPU 结果和 CPU 完全相等 —— 漏加一个元素都会被抓出来。
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

    // ---- 1. 先算 CPU 参考 ----
    double cpu_sum = 0.0;
    for (int i = 0; i < N; ++i) cpu_sum += h_in[i];

    // ---- 2. 再跑 GPU ----
    // 每个 block 出一个部分和，反复归约直到只剩一个数。
    // 用两块 buffer 乒乓：如果原地写回 in[blockIdx.x]，block i 写的位置
    // 正好落在 block 0 的输入区间里，block 之间又没有全局同步，会 race。
    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, (size_t)CEIL(N, BLOCK_SIZE) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_in, bytes, cudaMemcpyHostToDevice));

    int cur = N;
    float *src = d_a, *dst = d_b;
    while (cur > 1) {
        const int grid = CEIL(cur, BLOCK_SIZE);
        reduce_v0<<<grid, BLOCK_SIZE>>>(src, dst, cur);
        cur = grid;
        float* t = src; src = dst; dst = t;   // 交换，下一趟读刚写出来的
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, src, sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 3. 验证 ----
    printf("v0 interleaved (divergent)  N = %d\n", N);
    printf("  cpu = %.1f, gpu = %.1f  ->  %s\n", cpu_sum, (double)gpu_sum,
           (double)gpu_sum == cpu_sum ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_in);
    return (double)gpu_sum == cpu_sum ? 0 : 1;
}
