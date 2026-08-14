// v3: smem padding —— 给 shared memory 的每行末尾加一点空隙，消除 bank conflict。
//
// 相对 v2 只改了一件事：smem 的 leading dimension 从 BK / BN 变成 BK+8 / BN+8。
// 实测 78 -> 130 TFLOP/s，一行改动带来 1.67 倍。
//
// 为什么会冲突：shared memory 分成 32 个 bank，每个 bank 4 字节宽，地址按 4 字节
// 轮转分配 bank，也就是 bank(addr) = (addr / 4) % 32。一个 warp 内如果有多个线程
// 落在同一个 bank 的不同地址上，硬件只能拆成多次访问串行完成。
//
// v2 里 As 的行距是 BK = 32 个 half = 64 字节 = 16 个 bank。于是相邻两行的同一列
// 恰好错开 16 个 bank，第 0 行和第 2 行、第 1 行和第 3 行……落在完全相同的 bank 上。
// wmma 装 a_frag 时一个 warp 要读 16 行，这 16 行只覆盖 16 个 bank，两两撞成 2 路冲突。
// Bs 的行距 BN = 128 个 half = 256 字节 = 64 个 bank，是 32 的整数倍，情况更糟：
// 所有行的同一列全都落在同一个 bank 上。
//
// 加 8 个 half（16 字节 = 4 个 bank）的 padding 之后：
//   As 行距 40 个 half = 80 字节 = 20 个 bank，20 和 32 的最小公倍数是 160，
//   要隔 8 行才会重合，16 行里只有 2 组重合，冲突大幅下降。
//   Bs 行距 136 个 half = 272 字节 = 68 个 bank，68 % 32 = 4，同理错开。
//
// 为什么 padding 取 8 而不是 1：float4 搬运要求地址 16 字节对齐，8 个 half 正好 16 字节，
// 所以只能按 8 的倍数加。加 1 个 half 会让每行起点错位 2 字节，float4 直接非法。
//
// 实测（改这个文件的 LDA / LDB 就能复现，M=N=K=2048 单次冷跑）：
//   都不加            78 TFLOP/s
//   只加 As           100
//   只加 Bs           100
//   两边都加 8        130   <- 本版
//   As 加 16, Bs 8    123   —— 更大的 padding 并不更好，还白占 smem
//   两边都加 16       124
// 单独补一边只能拿到一半收益，两边都补才有 1.67 倍：A 和 B 的读取都要过 smem，
// 任何一侧留着冲突都会成为新的瓶颈。
//
// 另一种解法是 swizzle：不加空隙，而是按位异或打乱行内的存放顺序，用同样的 smem
// 拿到同样的效果。transpose/v5_shared_swizzle.cu 里演示过，这里先用简单的 padding。
#include <cuda_fp16.h>
#include <mma.h>

#include "common.h"

using namespace nvcuda;

constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WARP_M = 64, WARP_N = 32;
constexpr int WTILE_M = WARP_M / WMMA_M;  // 4
constexpr int WTILE_N = WARP_N / WMMA_N;  // 2
constexpr int WARPS_M = BM / WARP_M;      // 2
constexpr int WARPS_N = BN / WARP_N;      // 4
constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;  // 256

// 这一版的全部改动：smem 的行距不再等于 tile 的宽度
constexpr int PAD = 8;         // 8 个 half = 16 字节，保持 float4 对齐
constexpr int LDA = BK + PAD;  // 40
constexpr int LDB = BN + PAD;  // 136

__global__ void matmul_v3(const half* __restrict__ A, const half* __restrict__ B,
                          float* __restrict__ C, int m, int n, int k) {
    __shared__ half As[BM * LDA];  // 10 KB（v2 是 8 KB，padding 的代价）
    __shared__ half Bs[BK * LDB];  // 8.5 KB

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WTILE_M][WTILE_N];
#pragma unroll
    for (int i = 0; i < WTILE_M; ++i)
#pragma unroll
        for (int j = 0; j < WTILE_N; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < k; k0 += BK) {
        // ---- global -> smem ----
        // 搬运的切分方式不变，只是写 smem 时行距用 LDA / LDB。
        // global 侧的行距仍然是 k / n，那边没有 padding。
        for (int i = tid; i < BM * (BK / 8); i += NUM_THREADS) {
            const int r = i / (BK / 8), c = (i % (BK / 8)) * 8;
            *(float4*)&As[r * LDA + c] = *(const float4*)&A[(size_t)(block_row + r) * k + k0 + c];
        }
        for (int i = tid; i < BK * (BN / 8); i += NUM_THREADS) {
            const int r = i / (BN / 8), c = (i % (BN / 8)) * 8;
            *(float4*)&Bs[r * LDB + c] = *(const float4*)&B[(size_t)(k0 + r) * n + block_col + c];
        }
        __syncthreads();

        // ---- smem -> 寄存器 -> tensor core ----
        // 同样只是把 load_matrix_sync 的 ld 参数从 BK / BN 换成 LDA / LDB
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WTILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WTILE_N];
#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
#pragma unroll
            for (int i = 0; i < WTILE_M; ++i)
                wmma::load_matrix_sync(a_frag[i], &As[(warp_m * WARP_M + i * WMMA_M) * LDA + kk],
                                       LDA);
#pragma unroll
            for (int j = 0; j < WTILE_N; ++j)
                wmma::load_matrix_sync(b_frag[j], &Bs[kk * LDB + warp_n * WARP_N + j * WMMA_N], LDB);
#pragma unroll
            for (int i = 0; i < WTILE_M; ++i)
#pragma unroll
                for (int j = 0; j < WTILE_N; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < WTILE_M; ++i)
#pragma unroll
        for (int j = 0; j < WTILE_N; ++j) {
            const size_t row = block_row + warp_m * WARP_M + i * WMMA_M;
            const size_t col = block_col + warp_n * WARP_N + j * WMMA_N;
            wmma::store_matrix_sync(&C[row * n + col], acc[i][j], n, wmma::mem_row_major);
        }
}

int main() {
    const size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;

    half* h_A = (half*)malloc(a_elems * sizeof(half));
    half* h_B = (half*)malloc(b_elems * sizeof(half));
    float* h_C = (float*)malloc(c_elems * sizeof(float));
    float* h_ref = (float*)malloc(c_elems * sizeof(float));
    fill_inputs(h_A, a_elems, h_B, b_elems);

    printf("v3 smem padding  (M=%d, N=%d, K=%d)\n", M, N, K);
    printf("  block tile %dx%dx%d, %d warp, LDA %d (BK+%d), LDB %d (BN+%d), smem %zu B\n", BM, BN,
           BK, NUM_THREADS / 32, LDA, PAD, LDB, PAD, (BM * LDA + BK * LDB) * sizeof(half));

    half *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, a_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, b_elems * sizeof(half), cudaMemcpyHostToDevice));

    cublas_reference(d_A, d_B, d_C, h_ref);
    CUDA_CHECK(cudaMemset(d_C, 0x7F, c_elems * sizeof(float)));

    dim3 block(NUM_THREADS);
    dim3 grid(CEIL(N, BN), CEIL(M, BM));

    // CUDA 12 默认 lazy module loading：首次 launch 才把 kernel 模块加载进来，
    // 这笔一次性开销会整个落在单次计时里（实测能把 v3 从 130 TFLOP/s 压到 87）。
    // 先查一次 kernel 属性触发加载，把它挤出计时区间。这不是预热——kernel 本身没执行过，
    // cache 和 TLB 仍然是冷的，和 ncu 看到的那一次一致。
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, matmul_v3));

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    matmul_v3<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    report_perf(ms);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
    const bool pass = check_result(h_C, h_ref);

    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return pass ? 0 : 1;
}
