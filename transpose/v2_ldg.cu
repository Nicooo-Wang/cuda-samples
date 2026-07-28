// v2: 在 v1 基础上显式用 __ldg 走只读数据缓存，减少不合并读取的影响。
// 现代编译器对 const __restrict__ 指针往往已经自动这么做，所以提速可能不明显，
// 这里显式写出来是为了看清意图。
#include "transpose_common.cuh"

__global__ void device_transpose_v2(const float* input, float* output, int M_, int N_) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N_ && col < M_) {
        output[row * M_ + col] = __ldg(&input[col * N_ + row]);
    }
}

int main() {
    return run_transpose("v2 __ldg", [](const float* in, float* out, int m, int n) {
        dim3 block(32, 32);
        dim3 grid(CEIL(m, block.x), CEIL(n, block.y));
        device_transpose_v2<<<grid, block>>>(in, out, m, n);
    });
}
