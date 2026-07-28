// v5: 共享内存中转 + swizzling 消除 bank conflict（不浪费 padding 的那一列）
#include "transpose_common.cuh"

template <const int TILE>
__global__ void device_transpose_v5(const float* input, float* output, int M_, int N_) {
    // 不做 padding，使用 swizzling 解决 bank conflict。
    // 注意：这里的 xor swizzle 要求 TILE 是 2 的幂（且 TILE <= 32 时列索引才不越界）。
    static_assert(TILE > 0 && (TILE & (TILE - 1)) == 0, "TILE must be a power of two");
    __shared__ float S[TILE][TILE];
    const int bx = blockIdx.x * TILE;
    const int by = blockIdx.y * TILE;
    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;

    if (y1 < M_ && x1 < N_) {
        S[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[y1 * N_ + x1];  // 合并读取
    }
    __syncthreads();

    const int x2 = by + threadIdx.x;
    const int y2 = bx + threadIdx.y;
    if (y2 < N_ && x2 < M_) {
        // swizzling 主要利用异或运算的两个性质来规避 bank conflict：
        // 1. 运算的封闭性  2. x1^y != x2^y 当且仅当 x1 != x2
        // 举例（4x4）：
        // 第一行的访存位置由 0,0,0,0... 变为 0,1,2,3...
        // 第二行的访存位置由 1,1,1,1... 变为 1,0,3,2...
        // 第三行的访存位置由 2,2,2,2... 变为 2,3,0,1...
        // 第四行的访存位置由 3,3,3,3... 变为 3,2,1,0...
        // 既能保证充分利用 shared memory 的空间（性质 1），
        // 又能保证 warp 中各线程不会访问同一 bank（性质 2）。
        // 读的逻辑位置是 [threadIdx.x][threadIdx.y]，
        // 由于 xor 可交换，写时的 (x^y) 和这里的 (x^y) 指向同一个槽位。
        output[y2 * M_ + x2] = S[threadIdx.x][threadIdx.x ^ threadIdx.y];  // 合并写入
    }
}

int main() {
    return run_transpose("v5 shared (swizzle)", [](const float* in, float* out, int m, int n) {
        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid(CEIL(n, TILE_DIM), CEIL(m, TILE_DIM));
        device_transpose_v5<TILE_DIM><<<grid, block>>>(in, out, m, n);
    });
}
