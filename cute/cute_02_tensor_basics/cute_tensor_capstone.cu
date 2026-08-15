// capstone: 用 Tensor + partition 实现 GEMV
//
// 问题: y = A * x    A: MxN row-major, x: N, y: M
//
// 四个版本对比:
//   v0 baseline   传统 CUDA, 手写下标
//   v1 cute naive 用 Tensor 索引, 验证零开销
//   v2 cute smem  把 x 缓存到 shared memory
//   v3 cute part  用 local_tile + local_partition (README §5 的两级范式)

#include <cute/tensor.hpp>

#include "common.h"

using namespace cute;

constexpr int M = 4096;
constexpr int N = 512;
constexpr int BLOCK = 256;

// ---------- v0: 传统 CUDA ----------
__global__ void gemv_baseline(const float* A, const float* x, float* y, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) sum += A[i * n + j] * x[j];
    y[i] = sum;
}

// ---------- v1: CuTe Tensor 索引 ----------
__global__ void gemv_cute_naive(const float* A, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    auto tA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(Int<M>{}, Int<N>{}),
                                                        make_stride(Int<N>{}, Int<1>{})));
    auto tx = make_tensor(make_gmem_ptr(x), make_layout(Int<N>{}));
    auto ty = make_tensor(make_gmem_ptr(y), make_layout(Int<M>{}));

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) sum += tA(i, j) * tx(j);
    ty(i) = sum;
}

// ---------- v2: x 缓存到 shared memory ----------
__global__ void gemv_cute_smem(const float* A, const float* x, float* y) {
    __shared__ float smem_x[N];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    auto tA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(Int<M>{}, Int<N>{}),
                                                        make_stride(Int<N>{}, Int<1>{})));
    auto gx = make_tensor(make_gmem_ptr(x), make_layout(Int<N>{}));
    auto sx = make_tensor(make_smem_ptr(smem_x), make_layout(Int<N>{}));

    // 用 partition 协作把 x 搬进 smem (thr_layout stride 取 1 -> 合并访存)
    auto thr = make_layout(Int<BLOCK>{}, Int<1>{});
    auto my_dst = local_partition(sx, thr, tid);
    auto my_src = local_partition(gx, thr, tid);
    for (int k = 0; k < size(my_dst); ++k) my_dst(k) = my_src(k);

    __syncthreads();

    if (i < M) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) sum += tA(i, j) * sx(j);
        y[i] = sum;
    }
}

// ---------- v3: local_tile + local_partition 两级范式 ----------
__global__ void gemv_cute_partition(const float* A, const float* x, float* y) {
    __shared__ float smem_x[N];
    int tid = threadIdx.x;

    auto tA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(Int<M>{}, Int<N>{}),
                                                        make_stride(Int<N>{}, Int<1>{})));
    auto gx = make_tensor(make_gmem_ptr(x), make_layout(Int<N>{}));
    auto ty = make_tensor(make_gmem_ptr(y), make_layout(Int<M>{}));
    auto sx = make_tensor(make_smem_ptr(smem_x), make_layout(Int<N>{}));

    auto thr = make_layout(Int<BLOCK>{}, Int<1>{});

    // 阶段 1: 协作加载 x -> smem
    auto my_dst = local_partition(sx, thr, tid);
    auto my_src = local_partition(gx, thr, tid);
    for (int k = 0; k < size(my_dst); ++k) my_dst(k) = my_src(k);
    __syncthreads();

    // 阶段 2: local_tile 取出本 block 负责的 BLOCK 行 (A 的行条带 + y 的分段)
    auto blkA = local_tile(tA, make_shape(Int<BLOCK>{}, Int<N>{}), make_coord(blockIdx.x, 0));
    auto blky = local_tile(ty, make_shape(Int<BLOCK>{}), make_coord(blockIdx.x));

    // 阶段 3: 每个线程负责本块中的一行
    auto myA = local_partition(blkA, make_layout(make_shape(Int<BLOCK>{}, Int<1>{}),
                                                 make_stride(Int<1>{}, Int<1>{})),
                               tid);
    auto myy = local_partition(blky, thr, tid);

    for (int r = 0; r < size(myy); ++r) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) sum += myA(r, j) * sx(j);
        myy(r) = sum;
    }
}

// ---------- 计时 ----------
template <class Fn>
float bench(Fn launch, int warmup = 5, int iters = 100) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return ms / iters;
}

static bool verify(const float* got, const float* ref) {
    double worst = 0.0;
    int bad = -1;
    for (int i = 0; i < M; ++i) {
        double d = fabs((double)got[i] - (double)ref[i]);
        if (d > worst) {
            worst = d;
            if (d > 1e-2) bad = i;
        }
    }
    printf("  max abs diff = %.3e -> %s\n", worst, bad < 0 ? "PASS" : "FAIL");
    if (bad >= 0) printf("  first bad at %d: got %.4f ref %.4f\n", bad, got[bad], ref[bad]);
    return bad < 0;
}

int main() {
    print_separator("Section 02 Capstone: GEMV");
    printf("y = A * x   A: %d x %d,  block = %d\n", M, N, BLOCK);

    float* hA = (float*)malloc((size_t)M * N * sizeof(float));
    float* hx = (float*)malloc(N * sizeof(float));
    float* hy = (float*)malloc(M * sizeof(float));
    float* href = (float*)malloc(M * sizeof(float));

    srand(0);
    for (size_t i = 0; i < (size_t)M * N; ++i) hA[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < N; ++i) hx[i] = (float)rand() / RAND_MAX - 0.5f;

    printf("\nCPU 参考 ...\n");
    for (int i = 0; i < M; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) s += (double)hA[(size_t)i * N + j] * hx[j];
        href[i] = (float)s;
    }

    float *dA, *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(BLOCK), grid((M + BLOCK - 1) / BLOCK);

    auto run = [&](const char* name, auto launch) {
        print_separator(name);
        CUDA_CHECK(cudaMemset(dy, 0, M * sizeof(float)));
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hy, dy, M * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = verify(hy, href);
        float t = bench(launch);
        printf("  时间: %.4f ms\n", t);
        return ok ? t : -1.0f;
    };

    float t0 = run("v0 baseline", [&] { gemv_baseline<<<grid, block>>>(dA, dx, dy, M, N); });
    float t1 = run("v1 cute naive", [&] { gemv_cute_naive<<<grid, block>>>(dA, dx, dy); });
    float t2 = run("v2 cute + smem", [&] { gemv_cute_smem<<<grid, block>>>(dA, dx, dy); });
    float t3 = run("v3 cute + tile/partition",
                   [&] { gemv_cute_partition<<<grid, block>>>(dA, dx, dy); });

    print_separator("性能汇总");
    const double bytes = (double)M * N * sizeof(float);
    auto row = [&](const char* n, float t) {
        if (t < 0) {
            printf("  %-26s  FAILED\n", n);
        } else {
            printf("  %-26s  %.4f ms   %.1f GB/s   %.2fx\n", n, t, bytes / (t * 1e-3) / 1e9,
                   t0 / t);
        }
    };
    row("v0 baseline", t0);
    row("v1 cute naive", t1);
    row("v2 cute + smem", t2);
    row("v3 cute + tile/partition", t3);

    printf("\nGEMV 是访存瓶颈 (每个 A 元素只用一次), 所以几版差距不大 —— 这本身\n");
    printf("就是结论: CuTe 的抽象是零开销的, 写得更清楚不等于跑得更慢。\n");

    free(hA);
    free(hx);
    free(hy);
    free(href);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    return 0;
}
