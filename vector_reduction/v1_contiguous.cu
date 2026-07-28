// v1: 消除 warp divergence（contiguous addressing）
//
// v0 的活跃线程是 tid % (2*stride) == 0，散布在整个 block 里；
// 这里改成 tid < stride，活跃线程永远是前面连续的一段。
//
// 收益：同样是每轮减半的工作量，但现在闲置的 warp 是整个闲置，
// 可以直接退出调度，而不是每个 warp 里都有一半线程陪跑。
// 前 128 个线程活跃时正好是前 4 个 warp 满负荷，后 4 个 warp 完全不参与。
//
// 副作用：访问 s[tid] 和 s[tid+stride] 变成连续的，shared memory bank 冲突也更少。
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

constexpr int N = 1 << 24;
constexpr int BLOCK_SIZE = 256;

__global__ void reduce_v1(const float* in, float* out, int n) {
    __shared__ float s[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    s[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    // stride 从大到小，活跃线程始终是 [0, stride) 这一段连续区间
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = s[0];
}

int main() {
    const size_t bytes = (size_t)N * sizeof(float);
    float* h_in = (float*)malloc(bytes);
    // 随机输入而不是全 1.0：全 1 时和恰好是 2^24，float 能精确表示，
    // 任何加法顺序都得到同一个精确值，归约顺序写错了也照样"通过"。
    srand(0);
    for (int i = 0; i < N; ++i) h_in[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    // ---- 1. 先算 CPU 参考（double 累加，参考值要比被测值更准）----
    double cpu_sum = 0.0;
    for (int i = 0; i < N; ++i) cpu_sum += h_in[i];

    // ---- 2. 再跑 GPU ----
    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, (size_t)CEIL(N, BLOCK_SIZE) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_in, bytes, cudaMemcpyHostToDevice));

    int cur = N;
    float *src = d_a, *dst = d_b;
    while (cur > 1) {
        const int grid = CEIL(cur, BLOCK_SIZE);
        reduce_v1<<<grid, BLOCK_SIZE>>>(src, dst, cur);
        cur = grid;
        float* t = src; src = dst; dst = t;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, src, sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 3. 验证 ----
    // 零均值输入下 cpu_sum 会相消到接近 0，不能拿它做相对误差的分母。
    // 用 sum|x_i| 作尺度：树形归约的误差上界正比于它。
    double scale = 0.0;
    for (int i = 0; i < N; ++i) scale += fabs((double)h_in[i]);
    const double tol = 1e-9 * scale;
    const double diff = fabs((double)gpu_sum - cpu_sum);
    const bool pass = diff <= tol;

    printf("v1 contiguous (no divergence)  N = %d\n", N);
    printf("  cpu = %.6f, gpu = %.6f\n", cpu_sum, (double)gpu_sum);
    printf("  abs diff = %.3e (tol %.3e)  ->  %s\n", diff, tol, pass ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_in);
    return pass ? 0 : 1;
}
