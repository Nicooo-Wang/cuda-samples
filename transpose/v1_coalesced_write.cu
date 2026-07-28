// v1: 合并写入。索引反过来看，把 warp 沿 output 的行方向铺开：
// 写入 output[row * M + col] 连续（合并），读取 input[col * N + row] 跨度为 N（不合并）。
// 相比 v0，是把「不合并」从写侧挪到了读侧 —— 读不合并可以被 L1/L2 和只读缓存缓解，
// 所以通常比 v0 快。
#include "common.h"

__global__ void device_transpose_v1(const float* input, float* output, int M_, int N_) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N_ && col < M_) {
        output[row * M_ + col] = input[col * N_ + row];
    }
}

int main() {
    const size_t bytes = (size_t)M * N * sizeof(float);
    float* h_input  = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    float* h_ref    = (float*)malloc(bytes);

    fill_random(h_input, (size_t)M * N);

    printf("v1 coalesced write  (%dx%d)\n", M, N);

    // ---- 1. 先算 CPU 参考 ----
    cpu_transpose(h_input, h_ref, M, N);

    // ---- 2. 再跑 GPU，单轮 ----
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));

    dim3 block(32, 32);
    // 注意：这里 grid 是按 output 的形状 (N 行 M 列) 划分的
    dim3 grid(CEIL(M, block.x), CEIL(N, block.y));
    device_transpose_v1<<<grid, block>>>(d_input, d_output, M, N);
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
