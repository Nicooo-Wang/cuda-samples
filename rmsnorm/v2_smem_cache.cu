// v2: 把 x 一次性 load 进 shared memory，归约和归一化都从 smem 里走 —— 全局只读 x 1 次。
//
// v0/v1 的问题在于：算完 sumsq 后还要再读一遍 x 才能归一化。这里把整行 x 先搬进 shared memory，
// 之后无论归约还是归一化都读 smem（片上，几乎不花钱），于是全局读 x 从 2 次降到 1 次。
// 这是"用 shared memory 换全局带宽"的经典套路，和 softmax 把整行搬进 smem 复用是同一个思路。
//
// 一个 kernel 搞定（不需要中间的 sumsq 数组）：
//   读 x→smem → warp shuffle 归约 sum(x^2) → 算 inv_rms → 从 smem 归一化写出。
// 实际搬运：读 x 1 次 + 读 w 1 次 + 写 out 1 次 = 3 倍 M*N*4。
#include "common.h"

constexpr int BLOCK = N_HIDDEN;  // 1024
constexpr int WARPS = BLOCK / 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void rmsnorm_smem_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    float* __restrict__ out, int N, float eps) {
    __shared__ float sx[BLOCK];   // 缓存整行 x，之后归约/归一化都从这里读
    __shared__ float s[WARPS];    // 各 warp 的部分和
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32, warp = tid / 32;
    const float* xr = x + (size_t)row * N;

    sx[tid] = (tid < N) ? xr[tid] : 0.0f;  // 唯一一次全局读 x
    __syncthreads();

    // 归约 sum(x^2)，来源是 smem 而不是全局
    float v = sx[tid] * sx[tid];
    v = warp_reduce_sum(v);
    if (lane == 0) s[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < WARPS) ? s[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s[0] = v;  // 借 s[0] 广播最终 sumsq
    }
    __syncthreads();

    float inv_rms = rsqrtf(s[0] / N + eps);
    out[(size_t)row * N + tid] = sx[tid] * inv_rms * w[tid];  // 归一化也读 smem，不再碰全局 x
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

    float *d_x, *d_w, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice));

    printf("rmsnorm v2 smem cache  (M = %d, N = %d, block = %d)\n", M, N, BLOCK);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    rmsnorm_smem_kernel<<<M, BLOCK>>>(d_x, d_w, d_out, N, EPS_RMS);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;  // 1*x + 1*w + 1*out
    printf("  time            : %.4f ms\n", ms);
    printf("  useful bandwidth: %.2f GB/s  (1*x + 1*w + 1*out)\n", gbps);
    bool pass = validate_rmsnorm(h_out, h_ref, M, N);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
    free(h_x); free(h_w); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
