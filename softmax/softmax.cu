// Softmax: 三个 kernel 分步实现 (max -> sum -> normalize)
// 固定 shape，重点是练习 warp shuffle 归约 + 自定义 float atomicMax
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

// float 没有原生 atomicMax，用 int 位表示上的 CAS 循环模拟。
// 比较仍在 float 域上做 (fmaxf)，所以正负数都正确。
__device__ static float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// 第一步：求全局最大值。block 内两级 shuffle 归约，block 间用 atomicMax。
__global__ void max_kernel(const float* input, float* max_val, int N) {
    __shared__ float s_mem[32];  // 最多 32 个 warp
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    float val = (idx < N) ? input[idx] : (-FLT_MAX);
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    if (laneId == 0) s_mem[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : (-FLT_MAX);
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if (laneId == 0) atomicMaxFloat(max_val, val);
    }
}

// 第二步：求 sum(exp(x - max))，同样的归约结构，block 间用 atomicAdd。
__global__ void sum_kernel(const float* input, float* sum, const float* max_val, int N) {
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    float val = (idx < N) ? expf(input[idx] - *max_val) : 0.0f;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (laneId == 0) s_mem[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (laneId == 0) atomicAdd(sum, val);
    }
}

// 第三步：逐元素归一化
__global__ void softmax_kernel(const float* input, float* output, const float* sum,
                               const float* max_val, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) output[idx] = expf(input[idx] - *max_val) / (*sum);
}

// CPU 参考实现：safe softmax，累加用 double 减少参考值本身的误差
static void cpu_softmax(const float* input, float* output, int N) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; ++i) max_val = fmaxf(max_val, input[i]);

    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < N; ++i) output[i] = static_cast<float>(output[i] / sum);
}

int main() {
    const int N = 1 << 20;  // 固定 shape
    const size_t bytes = N * sizeof(float);

    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);

    srand(0);
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;  // [-10, 10)
    }

    const int block_size = 256;
    const int grid_size = CEIL(N, block_size);
    printf("softmax  (N = %d, block = %d, grid = %d)\n", N, block_size, grid_size);

    // ---- 1. 先算 CPU 参考 ----
    cpu_softmax(h_input, h_ref, N);

    // ---- 2. 再跑 GPU，单轮 ----
    float *d_input, *d_output, *d_sum, *d_max;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // max_val 初值必须是 -FLT_MAX，sum 初值 0：两者都是 atomic 累加出来的
    const float init_max = -FLT_MAX, init_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sum, &init_sum, sizeof(float), cudaMemcpyHostToDevice));

    max_kernel<<<grid_size, block_size>>>(d_input, d_max, N);
    sum_kernel<<<grid_size, block_size>>>(d_input, d_sum, d_max, N);
    softmax_kernel<<<grid_size, block_size>>>(d_input, d_output, d_sum, d_max, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // ---- 3. 精度验证：softmax 输出量级约 1/N，用相对误差判断 ----
    double max_rel_err = 0.0;
    int bad_idx = -1;
    double gpu_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        gpu_sum += h_output[i];
        const double denom = fabs(h_ref[i]) > 1e-30 ? fabs(h_ref[i]) : 1e-30;
        const double rel = fabs(h_output[i] - h_ref[i]) / denom;
        if (rel > max_rel_err) {
            max_rel_err = rel;
            bad_idx = i;
        }
    }

    printf("  sum of GPU output: %.6f (should be ~1.0)\n", gpu_sum);
    printf("  max rel error    : %.3e (at i = %d, gpu = %.9e, cpu = %.9e)\n", max_rel_err, bad_idx,
           h_output[bad_idx], h_ref[bad_idx]);
    printf("  %s\n",
           (max_rel_err < 1e-4 && fabs(gpu_sum - 1.0) < 1e-3) ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_max));
    free(h_input);
    free(h_output);
    free(h_ref);
    return 0;
}
