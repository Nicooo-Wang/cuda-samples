// v4: 共享内存中转 + padding 消除 bank conflict
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
constexpr int TILE_DIM = 32;   // 共享内存版本的 tile 边长

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
    const size_t in_bytes = (size_t)M * N * sizeof(float);
    const size_t out_bytes = (size_t)N * M * sizeof(float);

    float* h_input = (float*)malloc(in_bytes);
    float* h_output = (float*)malloc(out_bytes);
    float* h_ref = (float*)malloc(out_bytes);

    srand(0);
    for (size_t i = 0; i < (size_t)M * N; ++i) h_input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    printf("v4 shared (padding)  (%dx%d)\n", M, N);

    // ---- 1. 先算 CPU 参考 ----
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < N; ++col) h_ref[(size_t)col * M + row] = h_input[(size_t)row * N + col];

    // ---- 2. 再跑 GPU，单轮 ----
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, out_bytes));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(CEIL(N, TILE_DIM), CEIL(M, TILE_DIM));
    device_transpose_v4<TILE_DIM><<<grid, block>>>(d_input, d_output, M, N);
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
