// V1 基线：2D softmax（按行归一化），一个 block 处理一行，三遍 pass
//   pass1 max  →  pass2 sum(exp(x-max))  →  pass3 normalize
// 每一遍各自从全局内存读一遍输入，所以共 3 次读（后续版本会逐步减少到 2 次、1 次）。
//
// 归约结构与 1D 版本一致：warp shuffle → s_mem[32] → warp0 收口。
// 2D 相比 1D 的关键简化：一行一 block，整行归约在 block 内就能完成，
//   不再需要任何跨 block 的原子操作（1D 里为了全局 max/sum 用到了 atomicMax/atomicAdd）。
//
// 约束：行宽 N <= 1024（一个 thread 处理一个元素，block_size = nextPow2(N)）。
//       短行场景（N <= 512）完全够用；更大的行留待后续分块版本。
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

// ---- pass 1：求每行最大值。一个 block 一行，结果直接写到 rowmax[row]（无需原子）。----
__global__ void row_max_kernel(const float* __restrict__ x, float* __restrict__ rowmax, int N) {
    __shared__ float s_mem[32];  // 最多 32 个 warp
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * N;

    // 一个 thread 一个元素，越界用 -FLT_MAX 参与 max 归约（不影响结果）
    float val = (tid < N) ? xr[tid] : -FLT_MAX;
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, off));

    int warpId = tid / warpSize, lane = tid % warpSize;
    if (lane == 0) s_mem[warpId] = val;
    __syncthreads();

    if (warpId == 0) {  // warp0 把各 warp 的部分结果再归约一次
        int warpNum = blockDim.x / warpSize;
        val = (lane < warpNum) ? s_mem[lane] : -FLT_MAX;
        for (int off = warpSize >> 1; off > 0; off >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, off));
        if (lane == 0) rowmax[row] = val;
    }
}

// ---- pass 2：求每行 sum(exp(x - max))。归约结构同 pass1，把 fmax 换成 +。----
__global__ void row_sum_kernel(const float* __restrict__ x, float* __restrict__ rowsum,
                               const float* __restrict__ rowmax, int N) {
    __shared__ float s_mem[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * N;
    float mx = rowmax[row];

    float val = (tid < N) ? expf(xr[tid] - mx) : 0.0f;
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, off);

    int warpId = tid / warpSize, lane = tid % warpSize;
    if (lane == 0) s_mem[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (lane < warpNum) ? s_mem[lane] : 0.0f;
        for (int off = warpSize >> 1; off > 0; off >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, off);
        if (lane == 0) rowsum[row] = val;
    }
}

// ---- pass 3：逐元素归一化  out = exp(x - max) / sum。----
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ out,
                               const float* __restrict__ rowsum,
                               const float* __restrict__ rowmax, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= N) return;
    const float* xr = x + (size_t)row * N;
    out[(size_t)row * N + tid] = expf(xr[tid] - rowmax[row]) / rowsum[row];
}

// CPU 参考：逐行 safe softmax，sum 用 double 累加，减少参考值自身误差
static void cpu_softmax(const float* x, float* out, int M, int N) {
    for (int r = 0; r < M; ++r) {
        const float* xr = x + (size_t)r * N;
        float* outr = out + (size_t)r * N;
        float mx = -FLT_MAX;
        for (int c = 0; c < N; ++c) mx = fmaxf(mx, xr[c]);
        double s = 0.0;
        for (int c = 0; c < N; ++c) { outr[c] = expf(xr[c] - mx); s += outr[c]; }
        for (int c = 0; c < N; ++c) outr[c] = (float)(outr[c] / s);
    }
}

// 精度验证：逐元素相对误差（对照 double 累加的 CPU 参考）+ 每行和 ≈ 1。
// 打印指标并返回是否通过。各版本共用同一套验证。
static bool validate_softmax(const float* out, const float* ref, int M, int N) {
    double max_rel = 0.0, max_rowsum_err = 0.0;
    size_t bad_idx = 0;
    for (int r = 0; r < M; ++r) {
        double s = 0.0;
        for (int c = 0; c < N; ++c) {
            size_t i = (size_t)r * N + c;
            s += out[i];
            double denom = fabs(ref[i]) > 1e-30 ? fabs(ref[i]) : 1e-30;
            double rel = fabs(out[i] - ref[i]) / denom;
            if (rel > max_rel) { max_rel = rel; bad_idx = i; }
        }
        max_rowsum_err = fmax(max_rowsum_err, fabs(s - 1.0));
    }
    bool pass = (max_rel < 1e-5 && max_rowsum_err < 1e-5);
    printf("  max rel error  : %.3e (at i=%zu)\n", max_rel, bad_idx);
    printf("  max |rowsum-1| : %.3e\n", max_rowsum_err);
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

static int next_pow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p < 32 ? 32 : p;  // 至少一个 warp
}

int main() {
    const int M = 1 << 16;  // 行数（写死）
    const int N = 1024;     // 行宽（写死，中宽行：一行需 8 warp 协作归约）

    size_t bytes = (size_t)M * N * sizeof(float);
    float *h_in = (float*)malloc(bytes), *h_out = (float*)malloc(bytes), *h_ref = (float*)malloc(bytes);

    srand(0);
    for (size_t i = 0; i < (size_t)M * N; ++i)
        h_in[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;  // [-10, 10)

    cpu_softmax(h_in, h_ref, M, N);

    float *d_in, *d_out, *d_max, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_max, (size_t)M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, (size_t)M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int block = next_pow2(N);
    int grid = M;  // 一个 block 一行
    printf("softmax V1 baseline (M = %d, N = %d, block = %d, grid = %d)\n", M, N, block, grid);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    row_max_kernel<<<grid, block>>>(d_in, d_max, N);
    row_sum_kernel<<<grid, block>>>(d_in, d_sum, d_max, N);
    softmax_kernel<<<grid, block>>>(d_in, d_out, d_sum, d_max, N);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    double gbps = 2.0 * bytes / (ms * 1e-3) / 1e9;  // useful: 1 读 + 1 写
    printf("  time            : %.4f ms\n", ms);
    printf("  useful bandwidth: %.2f GB/s  (2*M*N*4 = read in + write out)\n", gbps);
    printf("  rows/s          : %.3f M\n", M / (ms * 1e-3) / 1e6);
    bool pass = validate_softmax(h_out, h_ref, M, N);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_max); cudaFree(d_sum);
    free(h_in); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
