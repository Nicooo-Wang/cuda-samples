// v0: 朴素实现。读取合并（沿 input 的行），写入不合并（沿 output 的列，跨度 M）
#include "common.h"

__global__ void device_transpose_v0(const float* input, float* output, int M_, int N_) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M_ && col < N_) {
        output[col * M_ + row] = input[row * N_ + col];
    }
}

int main() {
    const size_t bytes = (size_t)M * N * sizeof(float);
    float* h_input  = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    float* h_ref    = (float*)malloc(bytes);

    fill_random(h_input, (size_t)M * N);

    printf("v0 naive  (%dx%d)\n", M, N);

    // ---- 1. 先算 CPU 参考 ----
    cpu_transpose(h_input, h_ref, M, N);

    // ---- 2. 再跑 GPU，单轮 ----
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));

    dim3 block(32, 32);
    dim3 grid(CEIL(N, block.x), CEIL(M, block.y));
    device_transpose_v0<<<grid, block>>>(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // ---- 3. 验证：转置是纯搬运，要求逐元素 bit 级相等 ----
    const bool pass = verify_transpose(h_output, h_ref, M, N);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input); free(h_output); free(h_ref);
    return pass ? 0 : 1;
}
