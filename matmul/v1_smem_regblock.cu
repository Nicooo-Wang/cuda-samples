// v1: shared memory tiling + register blocking
//
// v0 的问题：A 的每一行 tile 会被 N/16 个 warp 各自从 global 读一遍，
// 实测 L1 流量约 4.29 GB（理论只需读 A+B = 16.7 MB），放大 ~256 倍。
// ncu 上表现为 l1tex__throughput ~100%，而 tensor pipe 只有 ~4%：
// tensor core 在等数据，瓶颈是 L1/LSU 数据通路。
//
// 两级复用把流量降下来：
//   1. shared memory tiling：一个 block 协作把 A 的 BM×BK、B 的 BK×BN 搬进 smem，
//      block 内所有 warp 共享，global 读取次数除以 tile 边长。
//   2. register blocking：每个 warp 算 WM×WN 的输出（多个 wmma tile），
//      载入的 a_frag / b_frag 在寄存器里被复用 WN/16 / WM/16 次，
//      把 smem→寄存器的流量也摊薄。这是 mma 指令占比能提上去的关键。
#include <mma.h>

#include "matmul_common.cuh"

using namespace nvcuda;

// block 级 tile：每个 block 算 BM×BN 的输出，K 方向每次推进 BK
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

// warp 级 tile：每个 warp 算 WM×WN 的输出
constexpr int WM = 64;
constexpr int WN = 64;
constexpr int WARPS_M = BM / WM;  // 2
constexpr int WARPS_N = BN / WN;  // 2
constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 4
constexpr int NUM_THREADS = NUM_WARPS * 32;   // 128

// 每个 warp 在 M / N 方向各持有几个 wmma tile 的 accumulator
constexpr int TM = WM / WMMA_M;  // 4
constexpr int TN = WN / WMMA_N;  // 4

// smem 的 leading dimension 加 padding：
// 不加的话 As 的 ld = BK = 32 halfs = 64B，wmma 从 smem 按列取数会撞 bank。
constexpr int PAD = 8;
constexpr int LDA_S = BK + PAD;  // As: BM x LDA_S
constexpr int LDB_S = BN + PAD;  // Bs: BK x LDB_S

// 每次 float4 搬 8 个 half（16B），是 global load 的最大宽度
constexpr int ELEMS_PER_LOAD = 8;

__global__ __launch_bounds__(NUM_THREADS) void matmul_v1(const half* __restrict__ A,
                                                         const half* __restrict__ B,
                                                         float* __restrict__ C, int m, int n,
                                                         int k) {
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + BM * LDA_S;

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // 每个 warp 的 TM×TN 个 accumulator 全程留在寄存器里，K 循环结束才写回 global
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < k; k0 += BK) {
        // ---- global -> smem，128 个线程 grid-stride 协作搬运，float4 向量化 ----
        // A tile: BM x BK
        for (int idx = tid; idx < BM * BK / ELEMS_PER_LOAD; idx += NUM_THREADS) {
            const int r = idx / (BK / ELEMS_PER_LOAD);
            const int c = (idx % (BK / ELEMS_PER_LOAD)) * ELEMS_PER_LOAD;
            *(float4*)&As[r * LDA_S + c] = *(const float4*)&A[(size_t)(block_row + r) * k + k0 + c];
        }
        // B tile: BK x BN
        for (int idx = tid; idx < BK * BN / ELEMS_PER_LOAD; idx += NUM_THREADS) {
            const int r = idx / (BN / ELEMS_PER_LOAD);
            const int c = (idx % (BN / ELEMS_PER_LOAD)) * ELEMS_PER_LOAD;
            *(float4*)&Bs[r * LDB_S + c] = *(const float4*)&B[(size_t)(k0 + r) * n + block_col + c];
        }
        __syncthreads();

        // ---- smem -> 寄存器 -> tensor core ----
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            // 先把这一小步 K 需要的 fragment 全部载入寄存器，
            // 然后 TM×TN 次 mma 复用它们：TM+TN 次 load 换 TM*TN 次 mma。
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[TM];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[TN];

            for (int i = 0; i < TM; ++i)
                wmma::load_matrix_sync(a_frag[i], &As[(warp_m * WM + i * WMMA_M) * LDA_S + kk],
                                       LDA_S);
            for (int j = 0; j < TN; ++j)
                wmma::load_matrix_sync(b_frag[j], &Bs[kk * LDB_S + warp_n * WN + j * WMMA_N], LDB_S);

            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        // 下一轮要覆盖 smem，得等所有 warp 都读完
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            const size_t row = block_row + warp_m * WM + i * WMMA_M;
            const size_t col = block_col + warp_n * WN + j * WMMA_N;
            wmma::store_matrix_sync(&C[row * n + col], acc[i][j], n, wmma::mem_row_major);
        }
    }
}

static size_t smem_bytes() {
    return (size_t)(BM * LDA_S + BK * LDB_S) * sizeof(half);
}

int main() {
    // 默认 48KB 的静态 smem 上限装不下，需要显式抬高动态 smem 配额
    CUDA_CHECK(cudaFuncSetAttribute(matmul_v1, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem_bytes()));

    return run_matmul("v1 smem tiling + regblock",
                      [](const half* A, const half* B, float* C, int m, int n, int k) {
                          dim3 block(NUM_THREADS);
                          dim3 grid(CEIL(n, BN), CEIL(m, BM));
                          matmul_v1<<<grid, block, smem_bytes()>>>(A, B, C, m, n, k);
                      });
}
