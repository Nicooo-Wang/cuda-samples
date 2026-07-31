// v3: 向量化 capstone。每个线程用 float4 一次搬 4 个 float（256 线程 × 4 = 1024 = N）。
//
// 关键点：
//   - 全局 load/store 都走 128-bit（float4）事务，单次访存搬 4 个 float，带宽利用最高；
//   - x 直接放寄存器（每个线程持有自己的 4 个值），不再经过 shared memory，连 smem 带宽都省了；
//   - 归约仍用 v1/v2 那套两级 warp shuffle（每线程先把自己 4 个值的平方和累加成一个标量，再 shuffle）。
//
// 搬运量和 v2 一样（1*x + 1*w + 1*out = 3 倍 M*N*4），但因为向量化事务，实测带宽会明显更高。
// 要求 N 是 4 的倍数（这里 N_HIDDEN=1024 满足）。
#include "common.h"

constexpr int VEC   = 4;
constexpr int BLOCK = N_HIDDEN / VEC;  // 256：每个线程管 VEC 个元素
constexpr int WARPS = BLOCK / 32;      // 8

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void rmsnorm_vec_kernel(const float4* __restrict__ x4, const float4* __restrict__ w4,
                                   float4* __restrict__ out4, int N, float eps) {
    __shared__ float s[WARPS];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32, warp = tid / 32;

    float4 xv = x4[(size_t)row * (N / VEC) + tid];  // 一次读 4 个 float（向量化）
    float v = xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;  // 本线程 4 值的平方和

    v = warp_reduce_sum(v);            // 第一级：warp 内归约
    if (lane == 0) s[warp] = v;
    __syncthreads();
    if (warp == 0) {                   // 第二级：warp0 把 8 个部分和归约
        v = (lane < WARPS) ? s[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s[0] = v;
    }
    __syncthreads();

    float inv_rms = rsqrtf(s[0] / N + eps);
    float4 wv = w4[tid];               // 权重也向量化读
    float4 ov;
    ov.x = xv.x * inv_rms * wv.x;
    ov.y = xv.y * inv_rms * wv.y;
    ov.z = xv.z * inv_rms * wv.z;
    ov.w = xv.w * inv_rms * wv.w;
    out4[(size_t)row * (N / VEC) + tid] = ov;  // 一次写 4 个 float（向量化）
}

int main() {
    const int M = M_ROWS, N = N_HIDDEN;
    static_assert(N_HIDDEN % VEC == 0, "N 必须是 VEC 的倍数");
    const size_t bytes   = (size_t)M * N * sizeof(float);
    const size_t w_bytes = (size_t)N * sizeof(float);

    float *h_x = (float*)malloc(bytes), *h_w = (float*)malloc(w_bytes);
    float *h_out = (float*)malloc(bytes), *h_ref = (float*)malloc(bytes);

    fill_random(h_x, (size_t)M * N);
    fill_weights(h_w, N);
    cpu_rmsnorm(h_x, h_w, h_ref, M, N, EPS_RMS);

    float *d_x, *d_w, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice));

    printf("rmsnorm v3 vectorized  (M = %d, N = %d, block = %d, vec = %d)\n", M, N, BLOCK, VEC);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    rmsnorm_vec_kernel<<<M, BLOCK>>>(
        (const float4*)d_x, (const float4*)d_w, (float4*)d_out, N, EPS_RMS);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;  // 同 v2：1*x + 1*w + 1*out
    printf("  time            : %.4f ms\n", ms);
    printf("  useful bandwidth: %.2f GB/s  (1*x + 1*w + 1*out, 向量化)\n", gbps);
    bool pass = validate_rmsnorm(h_out, h_ref, M, N);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
    free(h_x); free(h_w); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
