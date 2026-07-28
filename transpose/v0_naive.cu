// v0: 朴素实现。读取合并（沿 input 的行），写入不合并（沿 output 的列，跨度 M）
#include "transpose_common.cuh"

__global__ void device_transpose_v0(const float* input, float* output, int M_, int N_) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M_ && col < N_) {
        output[col * M_ + row] = input[row * N_ + col];
    }
}

int main() {
    return run_transpose("v0 naive", [](const float* in, float* out, int m, int n) {
        dim3 block(32, 32);
        dim3 grid(CEIL(n, block.x), CEIL(m, block.y));
        device_transpose_v0<<<grid, block>>>(in, out, m, n);
    });
}
