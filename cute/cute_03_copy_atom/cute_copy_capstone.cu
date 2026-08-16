// capstone: 用 Copy_Atom 写一个高带宽 memcpy
//
// 讲解见本目录 README.md 的 §8。
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
//
// 写法上注意 (README §4): 所有 layout / TiledCopy / Tensor 都在 host 上构造,
// kernel 只用 local_tile(blockIdx) + get_slice(threadIdx) 做索引。
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_copy_capstone

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// v1: naive —— 每个线程一个 float，grid-stride
//
// 这一版故意不用 CuTe，作为 baseline。
// ---------------------------------------------------------------------------
__global__ void memcpy_naive(const float* __restrict__ src, float* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// v2: TiledCopy + 128bit atom
//
// kernel 做两件事:
//   local_tile(mS, tiler, blockIdx.x)  -> 我这个 block 负责哪一块
//   get_slice(threadIdx.x).partition_S -> 我这个线程负责哪几个
// TiledCopy 和 Tensor 都是 host 传进来的。
// ---------------------------------------------------------------------------
template <int TILE, class TensorS, class TensorD, class TiledCopy>
__global__ void memcpy_vectorized(TensorS mS, TensorD mD, TiledCopy tc) {
    auto tiler = Shape<Int<TILE>>{};
    auto coord = make_coord(blockIdx.x);

    auto gS = local_tile(mS, tiler, coord);  // (TILE)
    auto gD = local_tile(mD, tiler, coord);  // (TILE)

    auto thr = tc.get_slice(threadIdx.x);
    copy(tc, thr.partition_S(gS), thr.partition_D(gD));
}

// ---------------------------------------------------------------------------
// v3: gmem -> smem -> gmem，用 cp.async
//
// smem 的 layout 也是 host 传进来的（和 CUTLASS sgemm_sm80 的 sA_layout 一样）。
// smem 数组按 cosize 开 —— 这里 layout 无 padding，cosize == size。
//
// cp.async 的意义: 搬运不占用线程的寄存器和指令流，发出去就能干别的。
// 这里只是把它跑通；真正的价值要到 v4 / Section 06 的 GEMM 才体现。
// ---------------------------------------------------------------------------
template <int TILE, class TensorS, class TensorD, class SLayout, class TCLoad, class TCStore>
__global__ void memcpy_smem(TensorS mS, TensorD mD, SLayout slay, TCLoad tc_load,
                            TCStore tc_store) {
    __shared__ float raw[cosize_v<SLayout>];
    auto sT = make_tensor(make_smem_ptr(raw), slay);

    auto tiler = Shape<Int<TILE>>{};
    auto coord = make_coord(blockIdx.x);
    auto gS = local_tile(mS, tiler, coord);
    auto gD = local_tile(mD, tiler, coord);

    // gmem -> smem 用 cp.async（异步，不经过寄存器）
    auto thr_load = tc_load.get_slice(threadIdx.x);
    copy(tc_load, thr_load.partition_S(gS), thr_load.partition_D(sT));

    cp_async_fence();    // 标记"这一批发完了"
    cp_async_wait<0>();  // 等它落地
    __syncthreads();     // smem 对全 block 可见

    // smem -> gmem 用普通向量 atom
    auto thr_store = tc_store.get_slice(threadIdx.x);
    copy(tc_store, thr_store.partition_S(sT), thr_store.partition_D(gD));
}

// ---------------------------------------------------------------------------
// v4: double buffer —— 搬下一块的同时写出这一块
//
// 每个 block 负责 TILES_PER_BLOCK 个连续 tile。用两块 smem 交替:
//   发起 tile k+1 的 load  ->  等 tile k  ->  写出 tile k
// 这样 load 的延迟被 store 的时间盖住一部分。
//
// smem layout 带一个 PIPE 维: (TILE, 2) —— 这就是 Section 06 多 stage 流水线
// 的最小形态，那里 PIPE 会变成 3~5。
// ---------------------------------------------------------------------------
template <int TILE, int TILES_PER_BLOCK, class TensorS, class TensorD, class SLayout,
          class TCLoad, class TCStore>
__global__ void memcpy_pipelined(TensorS mS, TensorD mD, SLayout slay, TCLoad tc_load,
                                 TCStore tc_store) {
    __shared__ float raw[cosize_v<SLayout>];
    auto sT = make_tensor(make_smem_ptr(raw), slay);  // (TILE, PIPE)

    auto thr_load = tc_load.get_slice(threadIdx.x);
    auto thr_store = tc_store.get_slice(threadIdx.x);

    auto tiler = Shape<Int<TILE>>{};
    int tile0 = blockIdx.x * TILES_PER_BLOCK;
    int ntile = size<0>(shape(mS)) / TILE;

    auto issue_load = [&](int t, int buf) {
        if (tile0 + t >= ntile) return;
        auto gS = local_tile(mS, tiler, make_coord(tile0 + t));
        copy(tc_load, thr_load.partition_S(gS), thr_load.partition_D(sT(_, buf)));
        cp_async_fence();
    };

    // 预热：先发起第 0 块
    issue_load(0, 0);

    for (int t = 0; t < TILES_PER_BLOCK; ++t) {
        if (tile0 + t >= ntile) break;
        int buf = t & 1;

        // 先把下一块的 load 发出去（此时上一块还在飞）
        bool has_next = (t + 1 < TILES_PER_BLOCK) && (tile0 + t + 1 < ntile);
        if (has_next) issue_load(t + 1, buf ^ 1);

        // 等最早那一批落地。cp_async_wait<N> 的 N = 允许还有几批在飞
        if (has_next) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        auto gD = local_tile(mD, tiler, make_coord(tile0 + t));
        copy(tc_store, thr_store.partition_S(sT(_, buf)), thr_store.partition_D(gD));

        __syncthreads();  // 下一轮要复用这块 smem
    }
}

int main() {
    print_separator("Capstone: 用 Copy_Atom 写高带宽 memcpy");

    constexpr int NTHR = 256;
    constexpr int TILE = NTHR * 4;       // 每线程 4 个 float -> 128bit
    constexpr int N = 64 * 1024 * 1024;  // 64M float = 256 MB
    static_assert(N % TILE == 0, "为了让 local_tile 不越界, 这里取整除的尺寸");
    constexpr size_t BYTES = size_t(N) * sizeof(float);

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

    // -----------------------------------------------------------------------
    // host 侧: 把"数据长什么样"和"谁搬哪一份"全部描述清楚 (README §4)
    // -----------------------------------------------------------------------
    // 全局 Tensor: shape 是运行时的 N, 但 stride 写成 Int<1>{} —— 向量化只需要
    // 这一点 (README §7.3)。
    auto mS = make_tensor(make_gmem_ptr(d_src), make_layout(make_shape(N), make_stride(Int<1>{})));
    auto mD = make_tensor(make_gmem_ptr(d_dst), make_layout(make_shape(N), make_stride(Int<1>{})));

    auto thr_lay = make_layout(Int<NTHR>{}, Int<1>{});
    auto val_lay = make_layout(Int<TILE / NTHR>{}, Int<1>{});

    auto tc_vec = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{}, thr_lay, val_lay);
    auto tc_async =
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>{}, thr_lay, val_lay);

    auto slay_1buf = make_layout(Int<TILE>{}, Int<1>{});                     // v3
    auto slay_2buf = make_layout(make_shape(Int<TILE>{}, Int<2>{}));         // v4: 带 PIPE 维

    printf("\nhost 侧构造好的东西:\n");
    printf("  mS       = ");
    print(mS.layout());
    printf("\n  tc_vec   Tiler_MN = ");
    print(typename decltype(tc_vec)::Tiler_MN{});
    printf("   线程数 = %d\n", int(size(tc_vec)));
    printf("  slay_1buf = ");
    print(slay_1buf);
    printf("   slay_2buf = ");
    print(slay_2buf);
    printf("\n");

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
        int grid = N / TILE;
        float ms =
            time_kernel([&] { memcpy_vectorized<TILE><<<grid, NTHR>>>(mS, mD, tc_vec); });
        bool ok = check("vectorized");
        double gbs = copy_bandwidth_gbs(BYTES, ms);
        printf("  grid=%d  %.3f ms   %.0f GB/s   %s\n", grid, ms, gbs, ok ? "正确" : "错误");
        results[nres++] = {"TiledCopy 128bit", ms, gbs, ok};
    }

    // ---------- v3: smem 中转 ----------
    print_separator("3. gmem -> smem -> gmem (cp.async)");
    CUDA_CHECK(cudaMemset(d_dst, 0, BYTES));
    {
        int grid = N / TILE;
        float ms = time_kernel([&] {
            memcpy_smem<TILE><<<grid, NTHR>>>(mS, mD, slay_1buf, tc_async, tc_vec);
        });
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
        int grid = (N / TILE + TPB - 1) / TPB;
        float ms = time_kernel([&] {
            memcpy_pipelined<TILE, TPB><<<grid, NTHR>>>(mS, mD, slay_2buf, tc_async, tc_vec);
        });
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
