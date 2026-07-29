// V3 vectorized：float4（128-bit）向量化加载 + 每个 thread 处理多个元素。
//   N=1024 = 256 thread × 4 elem，一个 warp（32×4=128）盖不住一整行，
//   所以【一行由 8 个 warp 协作】→ warp 内先 shuffle 归约，warp 之间再走一次共享内存归约。
//   每 thread 先在自己 4 个元素上做 thread-local online (m, s)，再 warp 归约，最后跨 warp 归约。
//   仍是【一个 block 一行】（grid = M），V4 才会改成"一个 block 多行"。
//
// 教学点：
//  - float4 向量化加载：一条指令取 128 bit（4 个 float），比逐 float 更省指令、更利于合并访存。
//  - 解耦"thread 数"和"元素数"：元素多不一定要更多 thread，可以让每个 thread 多干点（ILP）。
//  - 向量化把 thread 数降到 N/4，但只要 N/4 > 32，跨 warp 归约（共享内存 + __syncthreads）就免不掉；
//    对比 N=128 的情形：那时 N/4 == 32，一个 warp 就够，共享内存才能彻底去掉。
// 读次数仍是 2（ms 融合 pass + normalize pass，和 V2 一样），这一版主打访存效率。
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

constexpr int VEC = 4;                     // 每 thread 处理 4 个 float（一条 float4 加载）
constexpr int N = 1024;                    // 行宽（写死，中宽行）
constexpr int M = 1 << 16;                 // 行数（写死）
constexpr int THREADS_PER_ROW = N / VEC;   // 256：一行需要 256 个 thread
constexpr int WPR = THREADS_PER_ROW / 32;  // 8：一行由 8 个 warp 协作（warps per row）

// online 状态合并（结合律，可并行归约）
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

// ---- pass 1（融合）：一个 block（8 warp）一行，float4 加载，
//      thread-local online → warp 内归约 → 跨 warp 归约（共享内存）。----
__global__ void row_ms_kernel(const float* __restrict__ x, float* __restrict__ rowmax,
                              float* __restrict__ rowsum) {
    __shared__ float sm_m[WPR], sm_s[WPR];  // 每个 warp 的部分 (m, s)
    int row = blockIdx.x;
    if (row >= M) return;
    int tid_in_row = threadIdx.x;       // 0 .. 255
    int lane = tid_in_row % 32;
    int warpInRow = tid_in_row / 32;    // 0 .. 7

    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * N);
    float4 v = xr[tid_in_row];                 // 连续 4 个 float，coalesced 128-bit load
    float vals[VEC] = {v.x, v.y, v.z, v.w};

    float m = -FLT_MAX, s = 0.0f;
    #pragma unroll
    for (int e = 0; e < VEC; ++e) online_merge(m, s, vals[e], 1.0f);  // (vals[e], exp0=1)

    warp_reduce_ms(m, s);                      // warp 内归约，结果落在 lane0

    // 跨 warp：8 个 warp 的部分和先写共享内存，再由 warp0 归约一次。
    if (lane == 0) { sm_m[warpInRow] = m; sm_s[warpInRow] = s; }
    __syncthreads();

    if (warpInRow == 0) {
        // 注意：整个 warp 都要进来，warp_reduce_ms 用的是全掩码 shuffle。
        // 多出来的 lane 用单位元 (-FLT_MAX, 0) 填充，合并时不影响结果。
        m = (lane < WPR) ? sm_m[lane] : -FLT_MAX;
        s = (lane < WPR) ? sm_s[lane] : 0.0f;
        warp_reduce_ms(m, s);
        if (lane == 0) { rowmax[row] = m; rowsum[row] = s; }
    }
}

// ---- pass 2（归一化）：同样一个 block 一行，float4 加载 / 存储，无需归约。----
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ out,
                               const float* __restrict__ rowsum,
                               const float* __restrict__ rowmax) {
    int row = blockIdx.x;
    if (row >= M) return;
    int tid_in_row = threadIdx.x;
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

// CPU 参考：逐行 safe softmax，sum 用 double 累加
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

// 精度验证：逐元素相对误差 + 每行和 ≈ 1。各版本共用。
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

    printf("softmax V3 vectorized (M = %d, N = %d, threads/row = %d, warps/row = %d, vec = %d)\n",
           M, N, THREADS_PER_ROW, WPR, VEC);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    row_ms_kernel<<<M, THREADS_PER_ROW>>>(d_in, d_max, d_sum);
    softmax_kernel<<<M, THREADS_PER_ROW>>>(d_in, d_out, d_sum, d_max);
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
