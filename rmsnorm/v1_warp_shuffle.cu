// v1: 把 pass1 的归约从"shared memory 树形"换成"warp shuffle"（对标 vector_reduction/v4）。
//
// __shfl_down_sync 让线程直接读同 warp 内另一个线程的寄存器，不经过 shared memory，
// 也不需要同步指令（shuffle 本身带同步语义）。两遍 pass 结构和 v0 完全一样，还是读 x 2 次。
//
// 相比 v0：
//   - shared memory 用量从 1024 个 float 降到 32 个（只存各 warp 的部分和）；
//   - __syncthreads 从 ~10 次（log2(1024)）降到 1 次；
//   - 最后那个 warp 内的归约不再需要同步（warp 内指令天然按 lockstep 执行）。
#include "common.h"

constexpr int BLOCK = N_HIDDEN;          // 1024
constexpr int WARPS = BLOCK / 32;        // 32

// warp 内归约：5 次 shuffle 把 32 个值合成 1 个，落在 lane0。
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ---- pass 1：每行用 warp shuffle 归约出 sum(x^2) ----
__global__ void row_sumsq_kernel(const float* __restrict__ x, float* __restrict__ sumsq, int N) {
    __shared__ float s[WARPS];  // 每个 warp 一个部分和（只要 32 个 float）
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32, warp = tid / 32;
    const float* xr = x + (size_t)row * N;

    float v = (tid < N) ? xr[tid] * xr[tid] : 0.0f;

    v = warp_reduce_sum(v);            // 第一级：warp 内归约
    if (lane == 0) s[warp] = v;
    __syncthreads();                   // 全程只需这一次

    if (warp == 0) {                   // 第二级：warp0 把 32 个部分和再归约一次
        v = (lane < WARPS) ? s[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) sumsq[row] = v;
    }
}

// ---- pass 2：和 v0 相同，重读 x 做归一化 ----
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               const float* __restrict__ sumsq, float* __restrict__ out,
                               int N, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= N) return;
    const float* xr = x + (size_t)row * N;

    float inv_rms = rsqrtf(sumsq[row] / N + eps);
    out[(size_t)row * N + tid] = xr[tid] * inv_rms * w[tid];
}

int main() {
    const int M = M_ROWS, N = N_HIDDEN;
    const size_t bytes   = (size_t)M * N * sizeof(float);
    const size_t w_bytes = (size_t)N * sizeof(float);

    float *h_x = (float*)malloc(bytes), *h_w = (float*)malloc(w_bytes);
    float *h_out = (float*)malloc(bytes), *h_ref = (float*)malloc(bytes);

    fill_random(h_x, (size_t)M * N);
    fill_weights(h_w, N);
    cpu_rmsnorm(h_x, h_w, h_ref, M, N, EPS_RMS);

    float *d_x, *d_w, *d_sumsq, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_sumsq, (size_t)M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice));

    printf("rmsnorm v1 warp shuffle  (M = %d, N = %d, block = %d)\n", M, N, BLOCK);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    row_sumsq_kernel<<<M, BLOCK>>>(d_x, d_sumsq, N);
    rmsnorm_kernel<<<M, BLOCK>>>(d_x, d_w, d_sumsq, d_out, N, EPS_RMS);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    double gbps = 4.0 * bytes / (ms * 1e-3) / 1e9;  // 同 v0：2*x + 1*w + 1*out
    printf("  time            : %.4f ms\n", ms);
    printf("  useful bandwidth: %.2f GB/s  (2*x + 1*w + 1*out)\n", gbps);
    bool pass = validate_rmsnorm(h_out, h_ref, M, N);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_sumsq); cudaFree(d_out);
    free(h_x); free(h_w); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
