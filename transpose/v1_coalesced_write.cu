// v1: 合并写入。索引反过来看，把 warp 沿 output 的行方向铺开：
// 写入 output[row * M + col] 连续（合并），读取 input[col * N + row] 跨度为 N（不合并）。
// 相比 v0，是把「不合并」从写侧挪到了读侧 —— 读不合并可以被 L1/L2 和只读缓存缓解，
// 所以通常比 v0 快。
#include "transpose_common.cuh"

__global__ void device_transpose_v1(const float* input, float* output, int M_, int N_) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N_ && col < M_) {
        output[row * M_ + col] = input[col * N_ + row];
    }
}

int main() {
    return run_transpose("v1 coalesced write", [](const float* in, float* out, int m, int n) {
        dim3 block(32, 32);
        // 注意：这里 grid 是按 output 的形状 (N 行 M 列) 划分的
        dim3 grid(CEIL(m, block.x), CEIL(n, block.y));
        device_transpose_v1<<<grid, block>>>(in, out, m, n);
    });
}
