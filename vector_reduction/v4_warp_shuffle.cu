// v4: warp shuffle，彻底不用 shared memory 做 warp 内归约
//
// __shfl_down_sync 让线程直接读同一个 warp 里另一个线程的寄存器，
// 不经过 shared memory，也不需要同步指令（shuffle 本身自带同步语义）。
//
// 结构变成两级：
//   1. 每个 warp 用 5 次 shuffle 把 32 个值归约成 1 个（32 = 2^5）
//   2. 每个 warp 的结果写进 shared（只需 BLOCK_SIZE/32 = 8 个 float），
//      再由第 0 个 warp 做一次同样的 shuffle 归约
//
// 相比 v3：shared memory 用量从 256 个 float 降到 8 个，
// __syncthreads 从 3 次降到 1 次。
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
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;  // 8

// warp 内归约：5 次 shuffle，结束后 lane 0 持有整个 warp 的和
__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduce_v4(const float* in, float* out, int n) {
    __shared__ float s[WARPS_PER_BLOCK];  // 每个 warp 一个部分和

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * ELEMS_PER_BLOCK + tid;
    const int lane = tid % 32;
    const int warp = tid / 32;

    float sum = (idx < n) ? in[idx] : 0.0f;
    if (idx + blockDim.x < n) sum += in[idx + blockDim.x];

    // 第一级：warp 内归约
    sum = warp_reduce(sum);
    if (lane == 0) s[warp] = sum;
    __syncthreads();

    // 第二级：第 0 个 warp 把 8 个部分和再归约一次
    if (warp == 0) {
        sum = (lane < WARPS_PER_BLOCK) ? s[lane] : 0.0f;
        sum = warp_reduce(sum);
        if (lane == 0) out[blockIdx.x] = sum;
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
        reduce_v4<<<grid, BLOCK_SIZE>>>(src, dst, cur);
        cur = grid;
        float* t = src; src = dst; dst = t;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, src, sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 3. 验证 ----
    printf("v4 warp shuffle  N = %d\n", N);
    printf("  cpu = %.1f, gpu = %.1f  ->  %s\n", cpu_sum, (double)gpu_sum,
           (double)gpu_sum == cpu_sum ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_in);
    return (double)gpu_sum == cpu_sum ? 0 : 1;
}
