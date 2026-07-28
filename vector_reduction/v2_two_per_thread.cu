// v2: 每个线程先加两个元素（more work per thread）
//
// v1 里每个 block 只处理 BLOCK_SIZE 个元素，而 kernel 一启动就有一半线程
// （stride = 128 那轮之后）逐渐闲下来。既然如此，不如在载入阶段就让每个线程
// 多干一份活：一个 block 覆盖 2 * BLOCK_SIZE 个元素，读的时候顺手加一次。
//
// 收益：block 数量减半 -> kernel launch 次数和总的 __syncthreads 次数都减半，
// 而这一次额外的加法几乎免费（本来就要从 global 读数据，带宽是瓶颈）。
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
constexpr int ELEMS_PER_BLOCK = 2 * BLOCK_SIZE;  // 每个 block 覆盖的元素数

__global__ void reduce_v2(const float* in, float* out, int n) {
    __shared__ float s[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * ELEMS_PER_BLOCK + tid;   // 注意步长是 2*blockDim.x

    // 载入时就把相隔 blockDim.x 的两个元素加起来。
    // 相隔 blockDim.x 而不是相邻，是为了让同一个 warp 的 32 个线程
    // 访问连续的 32 个 float，保持合并访存。
    float sum = (idx < n) ? in[idx] : 0.0f;
    if (idx + blockDim.x < n) sum += in[idx + blockDim.x];
    s[tid] = sum;
    __syncthreads();

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
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

    // ---- 1. 先算 CPU 参考 ----
    double cpu_sum = 0.0;
    for (int i = 0; i < N; ++i) cpu_sum += h_in[i];

    // ---- 2. 再跑 GPU ----
    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, (size_t)CEIL(N, ELEMS_PER_BLOCK) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_in, bytes, cudaMemcpyHostToDevice));

    int cur = N;
    float *src = d_a, *dst = d_b;
    while (cur > 1) {
        const int grid = CEIL(cur, ELEMS_PER_BLOCK);
        reduce_v2<<<grid, BLOCK_SIZE>>>(src, dst, cur);
        cur = grid;
        float* t = src; src = dst; dst = t;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, src, sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 3. 验证 ----
    printf("v2 two elements per thread  N = %d\n", N);
    printf("  cpu = %.1f, gpu = %.1f  ->  %s\n", cpu_sum, (double)gpu_sum,
           (double)gpu_sum == cpu_sum ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_in);
    return (double)gpu_sum == cpu_sum ? 0 : 1;
}
