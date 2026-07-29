// V4 warp-per-row：核心思路仍是"一个 block 处理多行"，只改【行→资源】的映射。
//   N=1024 时一个 warp 已经装不下一行（32 thread × 4 elem = 128 < 1024），
//   所以一行改由 8 个 warp（256 thread）协同处理：
//   block = 4 行 × 8 warp/行 = 32 warp（1024 thread），grid = ceil(M / 4)。
//
// 一行跨了多个 warp，warp 内 shuffle 只能归约出 8 个部分 (m, s)，
// 因此必须补一级【跨 warp 归约】：8 个 lane0 把 (m, s) 写进共享内存，__syncthreads()，
// 再由该行的第 0 个 warp 用 online_merge 把这 8 对 (m, s) 合成整行结果。
// 单行算法和 V3 一致（float4 加载 + thread-local online + 归约），只是多了共享内存这一级。
// 教学点：行宽变大后，"一个 warp 一行"的漂亮性质失效；但决定性能的仍是【粒度/占用率】——
//   把多行塞进一个 block，既能填满 SM，又让同 block 的行在内存里连续、L2 局部性更好。
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

constexpr int VEC = 4;
constexpr int N = 1024;
constexpr int M = 1 << 16;
constexpr int THREADS_PER_ROW = N / VEC;                   // 256：一行需要 256 个 thread
constexpr int WPR = THREADS_PER_ROW / 32;                  // 8：一行 8 个 warp
constexpr int ROWS_PER_BLOCK = 4;                          // 一个 block 处理 4 行
constexpr int THREADS = ROWS_PER_BLOCK * THREADS_PER_ROW;  // 1024

__device__ __forceinline__ void online_merge(float& m, float& s, float mb, float sb) {
    float mn = fmaxf(m, mb);
    s = s * __expf(m - mn) + sb * __expf(mb - mn);
    m = mn;
}
__device__ __forceinline__ void warp_reduce_ms(float& m, float& s) {
    for (int off = warpSize >> 1; off > 0; off >>= 1) {
        float mb = __shfl_down_sync(0xFFFFFFFF, m, off);
        float sb = __shfl_down_sync(0xFFFFFFFF, s, off);
        online_merge(m, s, mb, sb);
    }
}

// 每个 block 处理 ROWS_PER_BLOCK 行，每行由 WPR=8 个 warp（256 thread）协同处理。
__global__ void row_ms_kernel(const float* __restrict__ x, float* __restrict__ rowmax,
                              float* __restrict__ rowsum) {
    int rowInBlock = threadIdx.x / THREADS_PER_ROW;  // 0..3
    int tid_in_row = threadIdx.x % THREADS_PER_ROW;  // 0..255
    int lane = tid_in_row % 32;
    int warpInRow = tid_in_row / 32;                 // 0..7
    int row = blockIdx.x * ROWS_PER_BLOCK + rowInBlock;
    if (row >= M) return;  // M % ROWS_PER_BLOCK == 0，整块统一，不会卡住 __syncthreads

    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * N);
    float4 v = xr[tid_in_row];
    float vals[VEC] = {v.x, v.y, v.z, v.w};

    float m = -FLT_MAX, s = 0.0f;
    #pragma unroll
    for (int e = 0; e < VEC; ++e) online_merge(m, s, vals[e], 1.0f);
    warp_reduce_ms(m, s);  // warp 内归约：每个 warp 得到 1/8 行的 (m, s)

    // 跨 warp 归约：每行 8 个 lane0 把部分结果写进共享内存
    __shared__ float sm_m[ROWS_PER_BLOCK * WPR];  // 32 floats
    __shared__ float sm_s[ROWS_PER_BLOCK * WPR];
    if (lane == 0) {
        sm_m[rowInBlock * WPR + warpInRow] = m;
        sm_s[rowInBlock * WPR + warpInRow] = s;
    }
    __syncthreads();

    if (warpInRow == 0) {
        // 只有 lane < WPR 有数据；其余 lane 填 online_merge 的单位元，
        // 这样整个 warp 都参与 warp_reduce_ms 的 full-mask shuffle（0xFFFFFFFF）。
        m = (lane < WPR) ? sm_m[rowInBlock * WPR + lane] : -FLT_MAX;
        s = (lane < WPR) ? sm_s[rowInBlock * WPR + lane] : 0.0f;
        warp_reduce_ms(m, s);
        if (lane == 0) { rowmax[row] = m; rowsum[row] = s; }
    }
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ out,
                               const float* __restrict__ rowsum,
                               const float* __restrict__ rowmax) {
    int rowInBlock = threadIdx.x / THREADS_PER_ROW;
    int tid_in_row = threadIdx.x % THREADS_PER_ROW;
    int row = blockIdx.x * ROWS_PER_BLOCK + rowInBlock;
    if (row >= M) return;

    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * N);
    float4* outr = reinterpret_cast<float4*>(out + (size_t)row * N);
    float mx = rowmax[row];
    float inv_s = 1.0f / rowsum[row];
    float4 v = xr[tid_in_row];
    float4 r;
    r.x = expf(v.x - mx) * inv_s;
    r.y = expf(v.y - mx) * inv_s;
    r.z = expf(v.z - mx) * inv_s;
    r.w = expf(v.w - mx) * inv_s;
    outr[tid_in_row] = r;
}

static void cpu_softmax(const float* x, float* out, int Mr, int Nr) {
    for (int r = 0; r < Mr; ++r) {
        const float* xr = x + (size_t)r * Nr;
        float* outr = out + (size_t)r * Nr;
        float mx = -FLT_MAX;
        for (int c = 0; c < Nr; ++c) mx = fmaxf(mx, xr[c]);
        double s = 0.0;
        for (int c = 0; c < Nr; ++c) { outr[c] = expf(xr[c] - mx); s += outr[c]; }
        for (int c = 0; c < Nr; ++c) outr[c] = (float)(outr[c] / s);
    }
}

static bool validate_softmax(const float* out, const float* ref, int Mr, int Nr) {
    double max_rel = 0.0, max_rowsum_err = 0.0;
    size_t bad_idx = 0;
    for (int r = 0; r < Mr; ++r) {
        double s = 0.0;
        for (int c = 0; c < Nr; ++c) {
            size_t i = (size_t)r * Nr + c;
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

int main() {
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

    int grid = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    printf("softmax V4 warp-per-row (M = %d, N = %d, rows/block = %d, warps/row = %d, threads = %d, grid = %d)\n",
           M, N, ROWS_PER_BLOCK, WPR, THREADS, grid);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    row_ms_kernel<<<grid, THREADS>>>(d_in, d_max, d_sum);
    softmax_kernel<<<grid, THREADS>>>(d_in, d_out, d_sum, d_max);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    double gbps = 2.0 * bytes / (ms * 1e-3) / 1e9;
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
