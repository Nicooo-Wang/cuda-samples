// v3: 用共享内存中转，读写都合并，但共享内存读存在 32 路 bank conflict
#include "common.h"

constexpr int TILE_DIM = 32;   // 共享内存版本的 tile 边长

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
    const size_t bytes = (size_t)M * N * sizeof(float);
    float* h_input  = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    float* h_ref    = (float*)malloc(bytes);

    fill_random(h_input, (size_t)M * N);

    printf("v3 shared (conflict)  (%dx%d)\n", M, N);

    // ---- 1. 先算 CPU 参考 ----
    cpu_transpose(h_input, h_ref, M, N);

    // ---- 2. 再跑 GPU，单轮 ----
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(CEIL(N, TILE_DIM), CEIL(M, TILE_DIM));
    device_transpose_v3<TILE_DIM><<<grid, block>>>(d_input, d_output, M, N);
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
