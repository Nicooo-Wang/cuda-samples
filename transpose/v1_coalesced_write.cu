// v1: 合并写入。索引反过来看，把 warp 沿 output 的行方向铺开：
// 写入 output[row * M + col] 连续（合并），读取 input[col * N + row] 跨度为 N（不合并）。
// 相比 v0，是把「不合并」从写侧挪到了读侧 —— 读不合并可以被 L1/L2 和只读缓存缓解，
// 所以通常比 v0 快。
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

constexpr int M = 4096;        // 输入是 M 行 N 列，转置成 N 行 M 列
constexpr int N = 4096;

__global__ void device_transpose_v1(const float* input, float* output, int M_, int N_) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N_ && col < M_) {
        output[row * M_ + col] = input[col * N_ + row];
    }
}

int main() {
    const size_t in_bytes = (size_t)M * N * sizeof(float);
    const size_t out_bytes = (size_t)N * M * sizeof(float);

    float* h_input = (float*)malloc(in_bytes);
    float* h_output = (float*)malloc(out_bytes);
    float* h_ref = (float*)malloc(out_bytes);

    srand(0);
    for (size_t i = 0; i < (size_t)M * N; ++i) h_input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    printf("v1 coalesced write  (%dx%d)\n", M, N);

    // ---- 1. 先算 CPU 参考 ----
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < N; ++col) h_ref[(size_t)col * M + row] = h_input[(size_t)row * N + col];

    // ---- 2. 再跑 GPU，单轮 ----
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, out_bytes));

    dim3 block(32, 32);
    // 注意：这里 grid 是按 output 的形状 (N 行 M 列) 划分的
    dim3 grid(CEIL(M, block.x), CEIL(N, block.y));
    device_transpose_v1<<<grid, block>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_bytes, cudaMemcpyDeviceToHost));

    // ---- 3. 验证：转置是纯搬运，要求逐元素 bit 级相等 ----
    size_t errors = 0, first_bad = 0;
    for (size_t i = 0; i < (size_t)N * M; ++i) {
        if (h_output[i] != h_ref[i]) {
            if (errors == 0) first_bad = i;
            ++errors;
        }
    }

    printf("  %s\n", errors == 0 ? "PASS" : "FAIL");
    if (errors != 0)
        printf("  %zu mismatches, first at %zu: gpu = %f, cpu = %f\n", errors, first_bad,
               h_output[first_bad], h_ref[first_bad]);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input); free(h_output); free(h_ref);
    return errors == 0 ? 0 : 1;
}
