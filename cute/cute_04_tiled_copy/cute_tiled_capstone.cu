// cute_04 capstone —— 矩阵转置: naive / padding / swizzle 三版对比
//
// 对应 README §8。
//
// 转置是"smem layout 决定性能"最干净的例子:
//   写 smem 按行 (合并), 读 smem 按列 (冲突) —— 冲突全部集中在一处, 好观察。
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_tiled_capstone

#include <cute/tensor.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

constexpr int TILE = 32;
constexpr int NTHR = 256;

// ---------------------------------------------------------------------------
// v1  完全不过 smem: 直接跨步写
// ---------------------------------------------------------------------------
template <int M, int N>
__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out) {
    int col = blockIdx.x * TILE + threadIdx.x % TILE;
    int row0 = blockIdx.y * TILE + threadIdx.x / TILE;
#pragma unroll
    for (int r = row0; r < TILE + blockIdx.y * TILE; r += NTHR / TILE) {
        if (r < M && col < N) out[col * M + r] = in[r * N + col];
    }
}

// ---------------------------------------------------------------------------
// v2/v3/v4  过 smem, 唯一的差别是 smem layout
// ---------------------------------------------------------------------------
template <int M, int N, class SLay>
__global__ void transpose_smem(const float* __restrict__ in, float* __restrict__ out, SLay slay) {
    // 按 cosize 开空间 —— padding 的 cosize 比 size 大
    __shared__ __align__(128) float raw[TILE * (TILE + 1)];
    auto s = make_tensor(make_smem_ptr(raw), slay);

    int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
    int tx = threadIdx.x % TILE, ty = threadIdx.x / TILE;

    // gmem -> smem: 按行读, 合并
#pragma unroll
    for (int r = ty; r < TILE; r += NTHR / TILE)
        if (by + r < M && bx + tx < N) s(r, tx) = in[(by + r) * N + bx + tx];

    __syncthreads();

    // smem -> gmem: 按列读 smem (冲突在这里), 按行写 gmem (合并)
#pragma unroll
    for (int r = ty; r < TILE; r += NTHR / TILE)
        if (bx + r < N && by + tx < M) out[(bx + r) * M + by + tx] = s(tx, r);
}

// ---------------------------------------------------------------------------
// 正确性检查
// ---------------------------------------------------------------------------
bool verify(const float* h_in, const float* h_out, int M, int N) {
    for (int i = 0; i < M; i += 97)
        for (int j = 0; j < N; j += 89)
            if (h_out[j * M + i] != h_in[i * N + j]) return false;
    return true;
}

int main() {
    constexpr int M = 8192, N = 8192;
    const size_t bytes = size_t(M) * N * sizeof(float);

    printf("cute_04 capstone —— 8192x8192 float 转置\n");
    printf("TILE = %d, 每 block %d 线程\n", TILE, NTHR);

    float *h_in = (float*)malloc(bytes), *h_out = (float*)malloc(bytes);
    for (size_t i = 0; i < size_t(M) * N; ++i) h_in[i] = float(i % 1024);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 grid(N / TILE, M / TILE), block(NTHR);

    auto plain = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                             make_stride(Int<TILE>{}, Int<1>{}));
    auto pad = make_layout(make_shape(Int<TILE>{}, Int<TILE>{}),
                           make_stride(Int<TILE + 1>{}, Int<1>{}));
    auto swz = composition(Swizzle<5, 0, 5>{}, plain);

    printf("\n  smem layout:\n");
    printf("    plain  = ");
    print(plain);
    printf("   cosize = %d\n", int(cosize(plain)));
    printf("    padded = ");
    print(pad);
    printf("   cosize = %d\n", int(cosize(pad)));
    printf("    swz    = ");
    print(swz);
    printf("   cosize = %d\n", int(cosize(swz)));

    printf("\n  %-26s %10s %12s %6s\n", "version", "time(ms)", "GB/s", "ok");

    struct Row {
        const char* name;
        float ms;
        bool ok;
    } rows[4];

    // v1 naive
    {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        float ms = time_kernel([&] { transpose_naive<M, N><<<grid, block>>>(d_in, d_out); });
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        rows[0] = {"naive (no smem)", ms, verify(h_in, h_out, M, N)};
    }
    // v2 plain smem
    {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        float ms =
            time_kernel([&] { transpose_smem<M, N><<<grid, block>>>(d_in, d_out, plain); });
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        rows[1] = {"smem plain (32-way conf)", ms, verify(h_in, h_out, M, N)};
    }
    // v3 padded
    {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        float ms = time_kernel([&] { transpose_smem<M, N><<<grid, block>>>(d_in, d_out, pad); });
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        rows[2] = {"smem padded (stride 33)", ms, verify(h_in, h_out, M, N)};
    }
    // v4 swizzle
    {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        float ms = time_kernel([&] { transpose_smem<M, N><<<grid, block>>>(d_in, d_out, swz); });
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        rows[3] = {"smem Swizzle<5,0,5>", ms, verify(h_in, h_out, M, N)};
    }

    for (auto& r : rows)
        printf("  %-26s %10.3f %12.1f %6s\n", r.name, r.ms,
               transpose_bandwidth_gbs(size_t(M) * N, sizeof(float), r.ms), r.ok ? "yes" : "NO");

    printf("\n  相对 plain smem 的加速:\n");
    for (int i = 2; i < 4; ++i) printf("    %-26s %.2fx\n", rows[i].name, rows[1].ms / rows[i].ms);

    printf("\n  读一遍 + 写一遍 = %.1f GB, 本机 HBM 理论带宽约 4.9 TB/s\n",
           2.0 * bytes / 1e9);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    printf("\ncapstone OK\n");
    return 0;
}
