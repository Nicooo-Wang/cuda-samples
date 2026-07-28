// v0: 最朴素的树形归约（interleaved addressing）
// 每个 block 处理 256 个元素，stride 从 1 开始翻倍，只有 tid 是 2*stride 倍数的线程干活。
//
// 问题：if (tid % (2 * stride) == 0) 让 warp 内大部分线程闲置。
// 第一轮 32 个线程里只有 16 个干活，第二轮只剩 8 个……而 warp 是整体调度的，
// 闲置线程也要陪着走完指令，这就是 warp divergence。
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

    // 用随机输入而不是全 1.0：全 1 时和恰好是 2^24，float 能精确表示，
    // 任何加法顺序都得到同一个精确值，归约顺序写错了也照样"通过" ——
    // 那种判据只能抓漏加元素，抓不到精度问题。
    // 随机值下树形归约和串行累加的舍入路径不同，才真正在检查数值精度。
    srand(0);
    for (int i = 0; i < N; ++i) h_in[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    // ---- 1. 先算 CPU 参考 ----
    // 用 double 累加：串行加 16M 个 float，float 累加器自身的误差会大到
    // 淹没被测对象，参考值必须比被测值更准。
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
    // 输入零均值，cpu_sum 本身会相消到接近 0，拿它当相对误差的分母会炸。
    // 用 sum|x_i| 作尺度：它是这次归约里被舍入的总量级，
    // 树形归约的误差上界正比于它（约 eps * log2(N) 倍）。
    double scale = 0.0;
    for (int i = 0; i < N; ++i) scale += fabs((double)h_in[i]);
    const double tol = 1e-9 * scale;  // 实测误差 ~4e-4，scale ~8.4e6，留约一个量级余量
    const double diff = fabs((double)gpu_sum - cpu_sum);
    const bool pass = diff <= tol;

    printf("v0 interleaved (divergent)  N = %d\n", N);
    printf("  cpu = %.6f, gpu = %.6f\n", cpu_sum, (double)gpu_sum);
    printf("  abs diff = %.3e (tol %.3e)  ->  %s\n", diff, tol, pass ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_in);
    return pass ? 0 : 1;
}
