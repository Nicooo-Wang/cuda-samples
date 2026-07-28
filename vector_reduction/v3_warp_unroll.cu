// v3: 最后 32 个元素用 warp 内展开，去掉多余的 __syncthreads
//
// 当 stride <= 32 时，活跃线程只剩一个 warp。warp 内的线程本来就是锁步执行的
// （同一条指令一起走），不需要 __syncthreads 来同步 —— 那是 block 级的栅栏，
// 对单个 warp 来说纯属浪费。
//
// 关键：必须用 __syncwarp()，不能什么都不加。
// 老教程里常见的写法是把 shared 指针声明成 volatile 然后裸展开，
// 那在 Volta 之前成立；Volta 引入独立线程调度（independent thread scheduling）后，
// warp 内的线程可以真的走散，必须显式 __syncwarp() 才能保证读到别人写的值。
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
constexpr int ELEMS_PER_BLOCK = 2 * BLOCK_SIZE;

__global__ void reduce_v3(const float* in, float* out, int n) {
    __shared__ float s[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * ELEMS_PER_BLOCK + tid;

    float sum = (idx < n) ? in[idx] : 0.0f;
    if (idx + blockDim.x < n) sum += in[idx + blockDim.x];
    s[tid] = sum;
    __syncthreads();

    // 先归约到只剩 32 个元素，这一段跨 warp，需要 block 级同步
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    // 剩下的 32 个由第 0 个 warp 独自完成，展开成 6 步，只需 warp 级同步
    if (tid < 32) {
        float v = s[tid] + s[tid + 32];
        __syncwarp();
        s[tid] = v;
        __syncwarp();
        v += s[tid + 16]; __syncwarp(); s[tid] = v; __syncwarp();
        v += s[tid + 8];  __syncwarp(); s[tid] = v; __syncwarp();
        v += s[tid + 4];  __syncwarp(); s[tid] = v; __syncwarp();
        v += s[tid + 2];  __syncwarp(); s[tid] = v; __syncwarp();
        v += s[tid + 1];
        if (tid == 0) out[blockIdx.x] = v;
    }
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
        reduce_v3<<<grid, BLOCK_SIZE>>>(src, dst, cur);
        cur = grid;
        float* t = src; src = dst; dst = t;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, src, sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 3. 验证 ----
    printf("v3 warp unrolled tail  N = %d\n", N);
    printf("  cpu = %.1f, gpu = %.1f  ->  %s\n", cpu_sum, (double)gpu_sum,
           (double)gpu_sum == cpu_sum ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_in);
    return (double)gpu_sum == cpu_sum ? 0 : 1;
}
