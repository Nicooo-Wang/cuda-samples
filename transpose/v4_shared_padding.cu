// v4: 共享内存中转 + padding 消除 bank conflict
#include "transpose_common.cuh"

template <const int TILE>
__global__ void device_transpose_v4(const float* input, float* output, int M_, int N_) {
    __shared__ float S[TILE][TILE + 1];  // 对共享内存做 padding，解决 bank conflict
    const int bx = blockIdx.x * TILE;
    const int by = blockIdx.y * TILE;
    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;

    if (y1 < M_ && x1 < N_) {
        S[threadIdx.y][threadIdx.x] = input[y1 * N_ + x1];  // 合并读取
    }
    __syncthreads();

    const int x2 = by + threadIdx.x;
    const int y2 = bx + threadIdx.y;
    if (y2 < N_ && x2 < M_) {
        // 做 padding 后，同一个 warp 中的 32 个线程（连续的 32 个 threadIdx.x 值）
        // 对应共享内存中跨度为 33 的数据。
        // 第一个线程访问第一个 bank 的第一层，第二个线程访问第二个 bank 的第二层，
        // 以此类推，32 个线程访问 32 个不同 bank，不存在 bank 冲突。
        output[y2 * M_ + x2] = S[threadIdx.x][threadIdx.y];  // 合并写入
    }
}

int main() {
    return run_transpose("v4 shared (padding)", [](const float* in, float* out, int m, int n) {
        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid(CEIL(n, TILE_DIM), CEIL(m, TILE_DIM));
        device_transpose_v4<TILE_DIM><<<grid, block>>>(in, out, m, n);
    });
}
