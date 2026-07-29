// V6 capstone：TMA（cp.async.bulk）异步搬数 + warp specialization（producer/consumer）。
//
// 思路（N=1024，一个 block 处理 ROWS=2 行）：
//   - producer warp（warp 0）：用 TMA 一条 cp.async.bulk 把整块 2×N 的 tile 从 global
//     异步搬到 shared，并用 mbarrier 登记"期望收到 BYTES 字节"。这一半和 N=128 版本完全一样，
//     只是 tile 字节数 BYTES 和 global 起始偏移随 N 变大。
//   - consumer warp（warp 1..16）：在 mbarrier 上等这批数据就绪，然后做单遍 online softmax。
//
// N=128 -> N=1024 的结构性变化（本版唯一的新东西）：
//   VEC=4 时一个 warp 只覆盖 32×4 = 128 个元素，一行 1024 已经装不下了 → 一行改由
//   WPR=8 个 warp（256 thread）协作。于是 warp 内 warp_reduce_ms 归约完还不够，必须再做一次
//   【跨 warp 的 (m, s) 归约 + 广播】：每个 warp 的 lane0 把自己的 (m,s) 写进 shared 暂存区，
//   __syncthreads() 后由每行的 warp0 用 lane<WPR 读回来再归约一次，写进 sm_fm/sm_fs 广播给全行。
//   注意 online_merge 的单位元是 (m=-FLT_MAX, s=0)，所以第二次归约里 lane>=WPR 必须填这个单位元
//   （warp_reduce_ms 是整 32 lane 的 shuffle，不能留未初始化值）。
//
// 另一个坑：__syncthreads() 是【整个 block】的栅栏，producer warp 也必须走到。所以 producer 发完
//   TMA 之后不能 return，要和 consumer 一起穿过这两个 __syncthreads()；越界行也只 guard 最后的写出，
//   不 guard 归约本身，否则 block 直接死锁。
//
// TMA / mbarrier 的同步模型（关键）：
//   1) mbarrier 初始化到达计数 = 1（只有 producer 会"到达"）。
//   2) producer 发 cp.async.bulk，紧接 mbarrier.arrive.expect_tx BYTES：这一步既"到达"（凑满 1），
//      又告诉 mbarrier"请等 BYTES 字节传完"。
//   3) 硬件传完后 mbarrier 的相位翻转；consumer 用 mbarrier.try_wait.parity 轮询到翻转即知数据就绪。
//
// 诚实说明：对【纯按行 softmax】这种小 tile，TMA+warp-spec 相对 V5 收益有限——它的价值在于
//   学会 Hopper 的 TMA 异步搬运 + mbarrier 握手 + producer/consumer 这套机制，这正是 FlashAttention
//   隐藏访存延迟的底子。编译用 -arch=sm_90a（cp.async.bulk 需要）。
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
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
constexpr int THREADS_PER_ROW = N / VEC;         // 256：一行需要 256 个 thread
constexpr int WPR = THREADS_PER_ROW / 32;        // 8：一行需要 8 个 consumer warp
constexpr int ROWS = 2;                          // 一个 block 处理 2 行
constexpr int THREADS = (1 + ROWS * WPR) * 32;   // 1 producer warp + 16 consumer warp = 544
constexpr uint32_t BYTES = ROWS * N * sizeof(float);  // 一个 tile 的字节数 = 8192

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

// ---- TMA / mbarrier 的 PTX 封装（sm_90a）----
// TMA 一维批量异步拷贝：global -> shared，传完后给 mbarrier 的 tx 计数减去 bytes。
__device__ __forceinline__ void tma_load(uint32_t smem_dst, const void* gmem_src,
                                         uint32_t bytes, uint32_t mbar) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(smem_dst), "l"(gmem_src), "r"(bytes), "r"(mbar) : "memory");
}
// 到达 mbarrier 并登记期望字节数（这一步凑满 init 计数 1）。
__device__ __forceinline__ void arrive_expect_tx(uint32_t mbar, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(mbar), "r"(bytes) : "memory");
}
// 轮询等待相位从 phase 翻转（数据就绪）。
__device__ __forceinline__ void mbarrier_wait(uint32_t mbar, int phase) {
    int done = 0;
    do {
        asm volatile(
            "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
            "  selp.u32 %0, 1, 0, p; }\n"
            : "=r"(done) : "r"(mbar), "r"(phase) : "memory");
    } while (!done);
}

__global__ void softmax_tma_kernel(const float* __restrict__ x, float* __restrict__ out) {
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int lane = tid % 32;
    int row_base = blockIdx.x * ROWS;

    __shared__ __align__(16) float smem[ROWS * N];  // tile 缓冲（8 KB）
    __shared__ __align__(8)  uint64_t mbar;          // mbarrier
    // 跨 warp 归约用的暂存区：和 tile 缓冲分开，别去 tile 里抠地方（tile 还要被 TMA 覆写）。
    __shared__ float sm_m[ROWS * WPR], sm_s[ROWS * WPR];  // 每行 WPR 个 warp 的部分 (m, s)
    __shared__ float sm_fm[ROWS], sm_fs[ROWS];            // 每行归约完的最终 (m, s)，用于广播

    if (tid == 0)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&mbar)), "r"(1) : "memory");
    __syncthreads();

    uint32_t mbar_addr = __cvta_generic_to_shared(&mbar);

    // ---- producer：warp0 的 lane0 发一次 TMA 把整块 tile 搬进 shared ----
    if (tid == 0) {
        tma_load(__cvta_generic_to_shared(smem), x + (size_t)row_base * N, BYTES, mbar_addr);
        arrive_expect_tx(mbar_addr, BYTES);
    }

    // ---- consumer：warp 1..(ROWS*WPR)，一行由 WPR=8 个 warp 协作 ----
    // 注意：producer warp 不能在这里 return，下面两个 __syncthreads() 是整个 block 的栅栏。
    float m_local = 0.0f, s_local = 0.0f;
    float exp_vals[VEC];
    int rowInTile = 0, warpInRow = 0, tid_in_row = 0, row = 0;
    bool is_consumer = (warpId >= 1);

    if (is_consumer) {
        int cw = warpId - 1;            // consumer warp 序号 0..15
        rowInTile = cw / WPR;           // 该 warp 负责 tile 内的哪一行：0..1
        warpInRow = cw % WPR;           // 在这一行里是第几个 warp：0..7
        tid_in_row = warpInRow * 32 + lane;   // 在这一行里的线程序号：0..255
        row = row_base + rowInTile;

        // 等数据就绪（相位 0 -> 1）
        mbarrier_wait(mbar_addr, 0);

        // 单遍 online softmax，数据来源是 shared 而不是 global
        const float4* sr = reinterpret_cast<const float4*>(smem + rowInTile * N);
        float4 v = sr[tid_in_row];
        float vals[VEC] = {v.x, v.y, v.z, v.w};

        m_local = vals[0];
        #pragma unroll
        for (int e = 1; e < VEC; ++e) m_local = fmaxf(m_local, vals[e]);

        s_local = 0.0f;
        #pragma unroll
        for (int e = 0; e < VEC; ++e) { exp_vals[e] = __expf(vals[e] - m_local); s_local += exp_vals[e]; }

        // warp 内归约：得到本 warp 覆盖的 128 个元素的 (m, s)，结果在 lane0
        float m = m_local, s = s_local;
        warp_reduce_ms(m, s);
        if (lane == 0) {
            sm_m[rowInTile * WPR + warpInRow] = m;
            sm_s[rowInTile * WPR + warpInRow] = s;
        }
    }

    __syncthreads();  // 整个 block（含 producer）都要到

    // ---- 跨 warp 归约：每行的 warp0 把该行 WPR 个部分结果再归约一次 ----
    if (is_consumer && warpInRow == 0) {
        // warp_reduce_ms 是整 32 lane 的 shuffle，lane >= WPR 必须填 online_merge 的单位元
        // (m = -FLT_MAX, s = 0)，否则会把未初始化的垃圾值混进归约。
        float m = -FLT_MAX, s = 0.0f;
        if (lane < WPR) {
            m = sm_m[rowInTile * WPR + lane];
            s = sm_s[rowInTile * WPR + lane];
        }
        warp_reduce_ms(m, s);
        if (lane == 0) { sm_fm[rowInTile] = m; sm_fs[rowInTile] = s; }
    }

    __syncthreads();  // 整个 block（含 producer）都要到

    // ---- 广播 + 一次性 rescale 写出 ----
    if (is_consumer) {
        float m = sm_fm[rowInTile];
        float s = sm_fs[rowInTile];

        float factor = __expf(m_local - m) / s;
        float4 r;
        r.x = exp_vals[0] * factor;
        r.y = exp_vals[1] * factor;
        r.z = exp_vals[2] * factor;
        r.w = exp_vals[3] * factor;
        if (row < M) {  // 只 guard 写出，不 guard 上面的归约（否则栅栏会分叉死锁）
            float4* outr = reinterpret_cast<float4*>(out + (size_t)row * N);
            outr[tid_in_row] = r;
        }
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

    int grid = (M + ROWS - 1) / ROWS;
    printf("softmax V6 TMA + warp-spec (M = %d, N = %d, rows/block = %d, warps/row = %d, "
           "threads = %d, grid = %d)\n",
           M, N, ROWS, WPR, THREADS, grid);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    softmax_tma_kernel<<<grid, THREADS>>>(d_in, d_out);
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
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
