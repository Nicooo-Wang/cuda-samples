// v0: 最朴素的 wmma 版本 —— 每个 warp 负责一个 16x16 的输出 tile，
// A / B 的 fragment 每次都直接从 global memory 载入，没有任何数据复用。
//
// 来源：Section011/tensor.cu，这里修掉了原版的两个问题：
//   1. warpN 的映射写成了 blockIdx.y * blockDim.y / warpSize + threadIdx.y，
//      按运算优先级是 (blockIdx.y * 32) / 32 + threadIdx.y = blockIdx.y + threadIdx.y，
//      blockIdx.y 的步长被整除掉了，导致只覆盖了一小块 C，其余 72% 从未被写入。
//   2. accumulator 用 half：K 大了会失精甚至溢出（fp16 最大有限值 65504）。
//      在这个 kernel 里换成 float 是免费的，因为瓶颈根本不在 tensor core。
#include <mma.h>

#include "matmul_common.cuh"

using namespace nvcuda;

__global__ void matmul_v0(const half* A, const half* B, float* C, int m, int n, int k) {
    const int warp_m = blockIdx.x;                            // M 方向 tile 下标
    const int warp_n = blockIdx.y * blockDim.y + threadIdx.y;  // N 方向 tile 下标

    if (warp_m * WMMA_M >= m || warp_n * WMMA_N >= n) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // 沿 K 方向串行归约。注意用 size_t 做偏移：
    // int 在 M*K 超过 2^31 时会静默溢出，算出负的地址。
    for (int k0 = 0; k0 < k; k0 += WMMA_K) {
        const half* a_tile = A + (size_t)warp_m * WMMA_M * k + k0;
        const half* b_tile = B + (size_t)k0 * n + warp_n * WMMA_N;
        wmma::load_matrix_sync(a_frag, a_tile, k);
        wmma::load_matrix_sync(b_frag, b_tile, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_tile = C + (size_t)warp_m * WMMA_M * n + warp_n * WMMA_N;
    wmma::store_matrix_sync(c_tile, c_frag, n, wmma::mem_row_major);
}

int main() {
    return run_matmul("v0 wmma naive", [](const half* A, const half* B, float* C, int m, int n,
                                          int k) {
        // block 内 8 个 warp，全部排在 N 方向
        dim3 block(32, 8);
        dim3 grid(CEIL(m, WMMA_M), CEIL(CEIL(n, WMMA_N), block.y));
        matmul_v0<<<grid, block>>>(A, B, C, m, n, k);
    });
}
