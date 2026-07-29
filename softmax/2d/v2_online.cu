// V2 online：把 V1 的 max、sum 两遍 pass 融合成一遍，靠 online softmax 的 (m, s) 在线合并。
//   pass1（融合）: 一遍同时算出每行的 (m, s) = (最大值, sum(exp(x-m)))  → 读 1 次
//   pass2（归一化）: out = exp(x - m) / s                                → 读 1 次
// 共 2 次读（V1 是 3 次）。
//
// 核心：online softmax 把"先求 max 再求 sum"串行依赖，改成可以同时累加的 (m, s) 状态，
//   遇到新数据时实时 rescale。这正是 FlashAttention 能一遍算出 attention 的数值基础。
//
// online 合并的关键性质是【结合律】：merge(a, merge(b,c)) == merge(merge(a,b), c)。
// 有了结合律，(m, s) 就能像求和一样做树形 / warp shuffle 并行归约（否则只能串行）。
// 一个 block 仍处理一行，block 内归约结构与 V1 一致，只是每步搬运的是 (m, s) 两路。
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

// online 状态合并：把 (mb, sb) 并入 (m, s)。满足结合律，可做并行归约。
__device__ __forceinline__ void online_merge(float& m, float& s, float mb, float sb) {
    float mn = fmaxf(m, mb);                       // 新的最大值
    s = s * __expf(m - mn) + sb * __expf(mb - mn); // 旧和、新和都 rescale 到 mn 后相加
    m = mn;
}

// warp 内归约 (m, s)：用 shuffle 把高半 warp 的 (m, s) 取下来 online_merge，结果落在 lane0。
__device__ __forceinline__ void warp_reduce_ms(float& m, float& s) {
    for (int off = warpSize >> 1; off > 0; off >>= 1) {
        float mb = __shfl_down_sync(0xFFFFFFFF, m, off);
        float sb = __shfl_down_sync(0xFFFFFFFF, s, off);
        online_merge(m, s, mb, sb);
    }
}

// ---- pass 1（融合）：一遍算出每行 (m, s)，写到 rowmax[row] / rowsum[row]。----
__global__ void row_ms_kernel(const float* __restrict__ x, float* __restrict__ rowmax,
                              float* __restrict__ rowsum, int N) {
    __shared__ float s_m[32], s_s[32];  // 各 warp 的部分 (m, s)
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * N;

    // 一个 thread 一个元素：局部状态就是 (x, exp(x-x)) = (x, 1)。
    // 越界 thread 用 (-FLT_MAX, 0)，合并时不影响结果。
    float m = (tid < N) ? xr[tid] : -FLT_MAX;
    float s = (tid < N) ? 1.0f : 0.0f;

    warp_reduce_ms(m, s);

    int warpId = tid / warpSize, lane = tid % warpSize;
    if (lane == 0) { s_m[warpId] = m; s_s[warpId] = s; }
    __syncthreads();

    if (warpId == 0) {  // warp0 把各 warp 的 (m, s) 再归约一次
        int warpNum = blockDim.x / warpSize;
        m = (lane < warpNum) ? s_m[lane] : -FLT_MAX;
        s = (lane < warpNum) ? s_s[lane] : 0.0f;
        warp_reduce_ms(m, s);
        if (lane == 0) { rowmax[row] = m; rowsum[row] = s; }
    }
}

// ---- pass 2（归一化）：out = exp(x - m) / s。重读一遍输入。----
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ out,
                               const float* __restrict__ rowsum,
                               const float* __restrict__ rowmax, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= N) return;
    const float* xr = x + (size_t)row * N;
    out[(size_t)row * N + tid] = expf(xr[tid] - rowmax[row]) / rowsum[row];
}

// CPU 参考：逐行 safe softmax，sum 用 double 累加
static void cpu_softmax(const float* x, float* out, int M, int N) {
    for (int r = 0; r < M; ++r) {
        const float* xr = x + (size_t)r * N;
        float* outr = out + (size_t)r * N;
        float mx = -FLT_MAX;
        for (int c = 0; c < N; ++c) mx = fmaxf(mx, xr[c]);
        double s = 0.0;
        for (int c = 0; c < N; ++c) { outr[c] = expf(xr[c] - mx); s += outr[c]; }
        for (int c = 0; c < N; ++c) outr[c] = (float)(outr[c] / s);
    }
}

// 精度验证：逐元素相对误差 + 每行和 ≈ 1。各版本共用。
static bool validate_softmax(const float* out, const float* ref, int M, int N) {
    double max_rel = 0.0, max_rowsum_err = 0.0;
    size_t bad_idx = 0;
    for (int r = 0; r < M; ++r) {
        double s = 0.0;
        for (int c = 0; c < N; ++c) {
            size_t i = (size_t)r * N + c;
            s += out[i];
            double denom = fabs(ref[i]) > 1e-30 ? fabs(ref[i]) : 1e-30;
            double rel = fabs(out[i] - ref[i]) / denom;
            if (rel > max_rel) { max_rel = rel; bad_idx = i; }
        }
        max_rowsum_err = fmax(max_rowsum_err, fabs(s - 1.0));
    }
    bool pass = (max_rel < 1e-5 && max_rowsum_err < 1e-5);
    printf("  max rel error  : %.3e (at i=%zu)\n", max_rel, bad_idx);
    printf("  max |rowsum-1| : %.3e\n", max_rowsum_err);
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

static int next_pow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p < 32 ? 32 : p;
}

int main() {
    const int M = 1 << 16;  // 行数（写死）
    const int N = 1024;     // 行宽（写死，中宽行：一行需 8 warp 协作归约）

    size_t bytes = (size_t)M * N * sizeof(float);
    float *h_in = (float*)malloc(bytes), *h_out = (float*)malloc(bytes), *h_ref = (float*)malloc(bytes);

    srand(0);
    for (size_t i = 0; i < (size_t)M * N; ++i)
        h_in[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;

    cpu_softmax(h_in, h_ref, M, N);

    float *d_in, *d_out, *d_max, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_max, (size_t)M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, (size_t)M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int block = next_pow2(N);
    int grid = M;
    printf("softmax V2 online (M = %d, N = %d, block = %d, grid = %d)\n", M, N, block, grid);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    row_ms_kernel<<<grid, block>>>(d_in, d_max, d_sum, N);   // 融合 pass1
    softmax_kernel<<<grid, block>>>(d_in, d_out, d_sum, d_max, N);  // pass2
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    double gbps = 2.0 * bytes / (ms * 1e-3) / 1e9;  // useful: 1 读 + 1 写
    printf("  time            : %.4f ms\n", ms);
    printf("  useful bandwidth: %.2f GB/s  (2*M*N*4 = read in + write out)\n", gbps);
    printf("  rows/s          : %.3f M\n", M / (ms * 1e-3) / 1e6);
    bool pass = validate_softmax(h_out, h_ref, M, N);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_max); cudaFree(d_sum);
    free(h_in); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
