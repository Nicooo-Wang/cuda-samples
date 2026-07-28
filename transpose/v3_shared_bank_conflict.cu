// v3: 用共享内存中转，读写都合并，但共享内存读存在 32 路 bank conflict
#include "transpose_common.cuh"

template <const int TILE>
__global__ void device_transpose_v3(const float* input, float* output, int M_, int N_) {
    __shared__ float S[TILE][TILE];
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
        // 合并写入，但是存在 bank 冲突：
        // 同一个 warp 中的 32 个线程（连续的 32 个 threadIdx.x 值）
        // 对应共享内存中跨度为 32 的数据，也就是说这 32 个线程恰好访问
        // 同一个 bank 中的 32 个数据，导致 32 路 bank 冲突。
        output[y2 * M_ + x2] = S[threadIdx.x][threadIdx.y];
    }
}

int main() {
    return run_transpose("v3 shared (conflict)", [](const float* in, float* out, int m, int n) {
        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid(CEIL(n, TILE_DIM), CEIL(m, TILE_DIM));
        device_transpose_v3<TILE_DIM><<<grid, block>>>(in, out, m, n);
    });
}
