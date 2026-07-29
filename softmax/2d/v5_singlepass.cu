// V5 single-pass：一遍搞定 max/sum/normalize，整行只读一次、只写一次（V4 是读两遍）。
// 这是把 FlashAttention 的 per-thread 技巧用到独立 softmax 上：把 exp(x - m_local) 缓存在寄存器里，
// 等 (m_final, s_final) 出来后一次性 rescale 写出，不再需要第二遍读输入。
//
// N=1024 时一行由 8 个 warp（256 thread）协作：每 thread 只缓存自己的 4 个 exp（低寄存器压力），
// warp 内归约后，再走一次【跨 warp (m,s) 归约 + 广播】拿到整行 (m_final, s_final)，然后每 thread
// 用 factor = exp(m_local - m_final)/s_final 把自己的 exp_vals 一次性 rescale 写出。
//   关键：out[e] = exp(vals[e]-m_final)/s_final = exp(vals[e]-m_local)·exp(m_local-m_final)/s_final。
//
// 收益：① 输入只读一次（流量减半）② exp 每元素只算一次 ③ 不需要 rowmax/rowsum 全局数组。
// 教学点：寄存器里缓存中间结果 + 末尾 rescale，是"一遍流式"算法能用起来的数值关键。
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

// 单遍 kernel：归约 + 归一化一次完成，不需要 rowmax/rowsum。
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ out) {
    __shared__ float sm_m[ROWS_PER_BLOCK * WPR];   // 每行 8 个 warp 的部分 (m, s)
    __shared__ float sm_s[ROWS_PER_BLOCK * WPR];
    __shared__ float sm_fm[ROWS_PER_BLOCK];        // 整行最终 (m, s)，广播用
    __shared__ float sm_fs[ROWS_PER_BLOCK];

    int tid = threadIdx.x;
    int rowInBlock = tid / THREADS_PER_ROW;        // 0..3
    int tid_in_row = tid % THREADS_PER_ROW;        // 0..255
    int lane = tid_in_row % 32;
    int warpInRow = tid_in_row / 32;               // 0..7
    int row = blockIdx.x * ROWS_PER_BLOCK + rowInBlock;
    bool active = (row < M);

    // 1) 一次加载、thread-local max + exp 缓存 + 局部 sum
    float m_local = -FLT_MAX, s_local = 0.0f;
    float exp_vals[VEC] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (active) {
        const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * N);
        float4 v = xr[tid_in_row];
        float vals[VEC] = {v.x, v.y, v.z, v.w};
        m_local = vals[0];
        #pragma unroll
        for (int e = 1; e < VEC; ++e) m_local = fmaxf(m_local, vals[e]);
        #pragma unroll
        for (int e = 0; e < VEC; ++e) { exp_vals[e] = __expf(vals[e] - m_local); s_local += exp_vals[e]; }
    }

    // 2) warp 内归约
    float m = m_local, s = s_local;
    warp_reduce_ms(m, s);

    // 3) 跨 warp 归约：每行 8 个 lane0 写共享内存，再由该行 warp0 归约
    if (lane == 0) { sm_m[rowInBlock * WPR + warpInRow] = m; sm_s[rowInBlock * WPR + warpInRow] = s; }
    __syncthreads();
    if (warpInRow == 0) {
        m = (lane < WPR) ? sm_m[rowInBlock * WPR + lane] : -FLT_MAX;  // 多余 lane 填单位元
        s = (lane < WPR) ? sm_s[rowInBlock * WPR + lane] : 0.0f;
        warp_reduce_ms(m, s);
        if (lane == 0) { sm_fm[rowInBlock] = m; sm_fs[rowInBlock] = s; }
    }
    __syncthreads();

    // 4) 广播 (m_final, s_final)，每 thread 一次性 rescale 写出
    if (active) {
        float factor = __expf(m_local - sm_fm[rowInBlock]) / sm_fs[rowInBlock];
        float4 r;
        r.x = exp_vals[0] * factor;
        r.y = exp_vals[1] * factor;
        r.z = exp_vals[2] * factor;
        r.w = exp_vals[3] * factor;
        float4* outr = reinterpret_cast<float4*>(out + (size_t)row * N);
        outr[tid_in_row] = r;
    }
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

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int grid = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    printf("softmax V5 single-pass (M = %d, N = %d, rows/block = %d, warps/row = %d, grid = %d)\n",
           M, N, ROWS_PER_BLOCK, WPR, grid);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    softmax_kernel<<<grid, THREADS>>>(d_in, d_out);
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
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
