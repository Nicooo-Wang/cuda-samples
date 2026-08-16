// capstone: 用 Copy_Atom 写一个高带宽 memcpy
//
// 讲解见本目录 README.md 的 §7。
//
// 四个版本，逐步加东西，看带宽怎么变：
//   v1 naive     : 1 线程 1 float，标量搬                        (baseline)
//   v2 vectorized: TiledCopy + 128bit atom                       (访存宽度)
//   v3 smem      : gmem -> smem -> gmem，用 cp.async 异步        (多一跳)
//   v4 pipelined : double buffer，搬下一块的同时写出这一块        (延迟隐藏)
//
// 对照组: cudaMemcpy(DeviceToDevice)
//
// 结论预告: memcpy 是纯带宽型任务，v2 就能贴近硬件上限；
//           v3 多绕一跳 smem 反而更慢 —— 这正是本章想让你看到的。

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// v1: naive —— 每个线程一个 float，grid-stride
// ---------------------------------------------------------------------------
__global__ void memcpy_naive(const float* __restrict__ src, float* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// v2: TiledCopy + 128bit atom
//
// 每个 block 处理 TILE 个 float。TILE 是编译期常量，所以 layout 的 stride
// 也是编译期的 —— 向量化 atom 需要这一点（见 v1 的 §4）。
// ---------------------------------------------------------------------------
template <int TILE, int NTHR>
__global__ void memcpy_vectorized(const float* __restrict__ src, float* __restrict__ dst, int n) {
    // 每线程搬 VEC 个 float，用 128bit 指令
    constexpr int VEC = TILE / NTHR;

    auto atom = Copy_Atom<UniversalCopy<uint128_t>, float>{};
    auto tc = make_tiled_copy(atom, make_layout(Int<NTHR>{}, Int<1>{}),
                              make_layout(Int<VEC>{}, Int<1>{}));

    int base = blockIdx.x * TILE;
    if (base + TILE > n) {
        // 尾块：退回标量，避免越界（README §7.3 讲为什么这样处理）
        for (int i = base + threadIdx.x; i < n; i += NTHR) dst[i] = src[i];
        return;
    }

    auto lay = make_layout(Int<TILE>{}, Int<1>{});
    auto S = make_tensor(make_gmem_ptr(src + base), lay);
    auto D = make_tensor(make_gmem_ptr(dst + base), lay);

    auto thr = tc.get_slice(threadIdx.x);
    copy(tc, thr.partition_S(S), thr.partition_D(D));
}

// ---------------------------------------------------------------------------
// v3: gmem -> smem -> gmem，用 cp.async
//
// cp.async 的意义: 搬运不占用线程的寄存器和指令流，发出去就能干别的。
// 这里只是把它跑通；真正的价值要到 v4 / Section 06 的 GEMM 才体现。
// ---------------------------------------------------------------------------
template <int TILE, int NTHR>
__global__ void memcpy_smem(const float* __restrict__ src, float* __restrict__ dst, int n) {
    constexpr int VEC = TILE / NTHR;
    __shared__ float smem[TILE];

    int base = blockIdx.x * TILE;
    if (base + TILE > n) {
        for (int i = base + threadIdx.x; i < n; i += NTHR) dst[i] = src[i];
        return;
    }

    auto lay = make_layout(Int<TILE>{}, Int<1>{});
    auto gS = make_tensor(make_gmem_ptr(src + base), lay);
    auto sT = make_tensor(make_smem_ptr(smem), lay);
    auto gD = make_tensor(make_gmem_ptr(dst + base), lay);

    // gmem -> smem 用 cp.async（异步，不经过寄存器）
    auto atom_async = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>{};
    auto tc_load = make_tiled_copy(atom_async, make_layout(Int<NTHR>{}, Int<1>{}),
                                   make_layout(Int<VEC>{}, Int<1>{}));
    auto thr_load = tc_load.get_slice(threadIdx.x);
    copy(tc_load, thr_load.partition_S(gS), thr_load.partition_D(sT));

    cp_async_fence();      // 标记"这一批发完了"
    cp_async_wait<0>();    // 等它落地
    __syncthreads();       // smem 对全 block 可见

    // smem -> gmem 用普通向量 atom
    auto tc_store = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                    make_layout(Int<NTHR>{}, Int<1>{}),
                                    make_layout(Int<VEC>{}, Int<1>{}));
    auto thr_store = tc_store.get_slice(threadIdx.x);
    copy(tc_store, thr_store.partition_S(sT), thr_store.partition_D(gD));
}

// ---------------------------------------------------------------------------
// v4: double buffer —— 搬下一块的同时写出这一块
//
// 每个 block 负责若干个连续 tile。用两块 smem 交替:
//   发起 tile k+1 的 load  ->  等 tile k  ->  写出 tile k
// 这样 load 的延迟被 store 的时间盖住一部分。
// ---------------------------------------------------------------------------
template <int TILE, int NTHR, int TILES_PER_BLOCK>
__global__ void memcpy_pipelined(const float* __restrict__ src, float* __restrict__ dst, int n) {
    constexpr int VEC = TILE / NTHR;
    __shared__ float smem[2][TILE];

    auto lay = make_layout(Int<TILE>{}, Int<1>{});
    auto atom_async = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>{};
    auto tc_load = make_tiled_copy(atom_async, make_layout(Int<NTHR>{}, Int<1>{}),
                                   make_layout(Int<VEC>{}, Int<1>{}));
    auto tc_store = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                    make_layout(Int<NTHR>{}, Int<1>{}),
                                    make_layout(Int<VEC>{}, Int<1>{}));
    auto thr_load = tc_load.get_slice(threadIdx.x);
    auto thr_store = tc_store.get_slice(threadIdx.x);

    int tile0 = blockIdx.x * TILES_PER_BLOCK;

    auto issue_load = [&](int t, int buf) {
        int base = (tile0 + t) * TILE;
        if (base + TILE > n) return;
        auto gS = make_tensor(make_gmem_ptr(src + base), lay);
        auto sT = make_tensor(make_smem_ptr(&smem[buf][0]), lay);
        copy(tc_load, thr_load.partition_S(gS), thr_load.partition_D(sT));
        cp_async_fence();
    };

    // 预热：先发起第 0 块
    issue_load(0, 0);

    for (int t = 0; t < TILES_PER_BLOCK; ++t) {
        int buf = t & 1;
        int base = (tile0 + t) * TILE;
        if (base >= n) break;

        // 先把下一块的 load 发出去（此时上一块还在飞）
        if (t + 1 < TILES_PER_BLOCK) issue_load(t + 1, buf ^ 1);

        // 等最早那一批落地。还在飞的批数: 发了下一块就是 1，否则 0
        if (t + 1 < TILES_PER_BLOCK) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        if (base + TILE > n) {
            for (int i = base + threadIdx.x; i < n; i += NTHR) dst[i] = src[i];
        } else {
            auto sT = make_tensor(make_smem_ptr(&smem[buf][0]), lay);
            auto gD = make_tensor(make_gmem_ptr(dst + base), lay);
            copy(tc_store, thr_store.partition_S(sT), thr_store.partition_D(gD));
        }
        __syncthreads();  // 下一轮要复用这块 smem
    }
}

int main() {
    print_separator("Capstone: 用 Copy_Atom 写高带宽 memcpy");

    constexpr int N = 64 * 1024 * 1024;  // 64M float = 256 MB
    constexpr size_t BYTES = size_t(N) * sizeof(float);
    constexpr int NTHR = 256;
    constexpr int TILE = NTHR * 4;  // 每线程 4 个 float -> 128bit

    printf("数据量: %d float = %.0f MB (读+写 = %.0f MB)\n", N, BYTES / 1e6, 2 * BYTES / 1e6);
    printf("配置:   NTHR=%d  TILE=%d (每线程 %d 个 float)\n", NTHR, TILE, TILE / NTHR);

    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, BYTES));
    CUDA_CHECK(cudaMalloc(&d_dst, BYTES));

    float* h_src = new float[N];
    for (int i = 0; i < N; ++i) h_src[i] = float(i % 1000);
    CUDA_CHECK(cudaMemcpy(d_src, h_src, BYTES, cudaMemcpyHostToDevice));

    float* h_dst = new float[N];
    auto check = [&](const char* name) {
        CUDA_CHECK(cudaMemcpy(h_dst, d_dst, BYTES, cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) {
            if (h_dst[i] != h_src[i]) {
                printf("  [%s] 第 %d 个元素不对: %g != %g\n", name, i, h_dst[i], h_src[i]);
                return false;
            }
        }
        return true;
    };

    struct Result {
        const char* name;
        float ms;
        double gbs;
        bool ok;
    };
    Result results[5];
    int nres = 0;

    // ---------- 对照组: cudaMemcpy ----------
    print_separator("0. 对照组: cudaMemcpy D2D");
    CUDA_CHECK(cudaMemset(d_dst, 0, BYTES));
    {
        float ms = time_kernel(
            [&] { CUDA_CHECK(cudaMemcpy(d_dst, d_src, BYTES, cudaMemcpyDeviceToDevice)); });
        bool ok = check("cudaMemcpy");
        double gbs = copy_bandwidth_gbs(BYTES, ms);
        printf("  %.3f ms   %.0f GB/s   %s\n", ms, gbs, ok ? "正确" : "错误");
        results[nres++] = {"cudaMemcpy D2D", ms, gbs, ok};
    }

    // ---------- v1: naive ----------
    print_separator("1. naive: 1 线程 1 float");
    CUDA_CHECK(cudaMemset(d_dst, 0, BYTES));
    {
        int grid = 1024;
        float ms = time_kernel([&] { memcpy_naive<<<grid, NTHR>>>(d_src, d_dst, N); });
        bool ok = check("naive");
        double gbs = copy_bandwidth_gbs(BYTES, ms);
        printf("  grid=%d  %.3f ms   %.0f GB/s   %s\n", grid, ms, gbs, ok ? "正确" : "错误");
        results[nres++] = {"naive (scalar)", ms, gbs, ok};
    }

    // ---------- v2: vectorized ----------
    print_separator("2. TiledCopy + 128bit atom");
    CUDA_CHECK(cudaMemset(d_dst, 0, BYTES));
    {
        int grid = (N + TILE - 1) / TILE;
        float ms =
            time_kernel([&] { memcpy_vectorized<TILE, NTHR><<<grid, NTHR>>>(d_src, d_dst, N); });
        bool ok = check("vectorized");
        double gbs = copy_bandwidth_gbs(BYTES, ms);
        printf("  grid=%d  %.3f ms   %.0f GB/s   %s\n", grid, ms, gbs, ok ? "正确" : "错误");
        results[nres++] = {"TiledCopy 128bit", ms, gbs, ok};
    }

    // ---------- v3: smem 中转 ----------
    print_separator("3. gmem -> smem -> gmem (cp.async)");
    CUDA_CHECK(cudaMemset(d_dst, 0, BYTES));
    {
        int grid = (N + TILE - 1) / TILE;
        float ms = time_kernel([&] { memcpy_smem<TILE, NTHR><<<grid, NTHR>>>(d_src, d_dst, N); });
        bool ok = check("smem");
        double gbs = copy_bandwidth_gbs(BYTES, ms);
        printf("  grid=%d  %.3f ms   %.0f GB/s   %s\n", grid, ms, gbs, ok ? "正确" : "错误");
        results[nres++] = {"smem staging", ms, gbs, ok};
    }

    // ---------- v4: double buffer ----------
    print_separator("4. double buffer 流水线");
    CUDA_CHECK(cudaMemset(d_dst, 0, BYTES));
    {
        constexpr int TPB = 4;  // 每 block 处理 4 个 tile
        int grid = (N + TILE * TPB - 1) / (TILE * TPB);
        float ms = time_kernel(
            [&] { memcpy_pipelined<TILE, NTHR, TPB><<<grid, NTHR>>>(d_src, d_dst, N); });
        bool ok = check("pipelined");
        double gbs = copy_bandwidth_gbs(BYTES, ms);
        printf("  grid=%d  %.3f ms   %.0f GB/s   %s\n", grid, ms, gbs, ok ? "正确" : "错误");
        results[nres++] = {"double buffer", ms, gbs, ok};
    }

    // ---------- 汇总 ----------
    print_separator("汇总");
    printf("%-20s %10s %12s %7s\n", "version", "time(ms)", "GB/s", "ok");
    for (int i = 0; i < nres; ++i)
        printf("%-20s %10.3f %12.0f %7s\n", results[i].name, results[i].ms, results[i].gbs,
               results[i].ok ? "yes" : "NO");

    printf("\n相对 cudaMemcpy 的比例:\n");
    for (int i = 1; i < nres; ++i)
        printf("  %-20s %.2fx\n", results[i].name, results[i].gbs / results[0].gbs);

    delete[] h_src;
    delete[] h_dst;
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    return 0;
}
