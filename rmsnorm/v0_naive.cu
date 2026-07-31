// v0: 朴素实现。一个 block 处理一行，1 个线程管 1 个元素。
//
// 两遍 pass（对标 softmax/2d/v1_baseline 的结构）：
//   pass1（归约）: 用 shared memory 树形归约求每行的 sum(x^2)，写到 sumsq[row]  → 读 x 1 次
//   pass2（归一化）: out = (x / sqrt(sumsq/N + eps)) * w                       → 再读 x 1 次
// 共读 x 2 次（外加读 w 1 次、写 out 1 次）。
//
// 本版的"毛病"（后面几版逐一解决）：
//   1) x 被读了两遍——归约算完 sumsq 后，归一化又得重新读一遍 x。
//   2) 树形归约每轮只有一半线程干活，另一半空闲（warp divergence / 占用率低）；
//      而且最后 s<=32 的那个 warp 本可以用 shuffle 免同步，这里却还在 __syncthreads。
#include "common.h"

constexpr int BLOCK = N_HIDDEN;  // 1024：一个 block 正好覆盖一行

// ---- pass 1：每行树形归约出 sum(x^2) ----
__global__ void row_sumsq_kernel(const float* __restrict__ x, float* __restrict__ sumsq, int N) {
    __shared__ float sdata[BLOCK];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * N;

    sdata[tid] = (tid < N) ? xr[tid] * xr[tid] : 0.0f;  // 越界线程贡献 0
    __syncthreads();

    // 树形归约：每轮活跃线程数减半。朴素写法，毛病见文件头注释。
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) sumsq[row] = sdata[0];
}

// ---- pass 2：重读 x，按 sumsq 做归一化 ----
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               const float* __restrict__ sumsq, float* __restrict__ out,
                               int N, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= N) return;
    const float* xr = x + (size_t)row * N;

    float inv_rms = rsqrtf(sumsq[row] / N + eps);  // rsqrtf = 1/sqrt，一条硬件指令
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

    printf("rmsnorm v0 naive  (M = %d, N = %d, block = %d)\n", M, N, BLOCK);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    row_sumsq_kernel<<<M, BLOCK>>>(d_x, d_sumsq, N);        // pass1：归约
    rmsnorm_kernel<<<M, BLOCK>>>(d_x, d_w, d_sumsq, d_out, N, EPS_RMS);  // pass2：归一化
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // 实际搬运量：读 x 2 次 + 读 w 1 次 + 写 out 1 次 = 4 倍 M*N*4（w 摊到每行也是 M*N）。
    double gbps = 4.0 * bytes / (ms * 1e-3) / 1e9;
    printf("  time            : %.4f ms\n", ms);
    printf("  useful bandwidth: %.2f GB/s  (2*x + 1*w + 1*out)\n", gbps);
    bool pass = validate_rmsnorm(h_out, h_ref, M, N);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_sumsq); cudaFree(d_out);
    free(h_x); free(h_w); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
