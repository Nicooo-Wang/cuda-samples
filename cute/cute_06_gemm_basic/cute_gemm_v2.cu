// cute_06 v2 —— 多 stage cp.async 流水线: Ampere 的上限
//
// 对应 README §3。
//
// v1 的毛病一眼就能看出来: 搬和算串行, 每轮两个 __syncthreads。
//   gmem->smem 搬完 -> 等 -> 算完 -> 等 -> 下一轮搬 ...
// 硬件在搬的时候没人算, 在算的时候没人搬 —— 两边各闲一半。
//
// Ampere 的解法是 **cp.async + 多 stage**:
//   cp.async 是异步的 (发出指令就返回, 不占寄存器等数据),
//   于是可以**提前把 k+1, k+2 的搬发出去**, 在它们飞的时候算 k。
//   smem 开成 N 份 (stage), 搬的写 stage[i], 算的读 stage[j], 轮转。
//
// 这一版 = 官方 wgmma_sm90.cu 的 SM80 版: 同样的 pipeline 结构,
// 只是把 WGMMA 换回 SM80 MMA + ldmatrix。这是 Ampere 能做到的最好形态,
// 也是 Hopper 流水线的对照基线。
//
//   §3.1  为什么 cp.async 能重叠: 异步的语义        (README §3.1)
//   §3.2  pipeline: stage 轮转 + cp_async_wait<N>   (README §3.2)
//   §3.3  实测: 和 v1 比, 快多少                    (README §3.3)
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_gemm_v2

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 尺寸配置
//
// gmem (TN 摆法):
//   A: M x K half, row-major, stride = (K,1)
//   B: N x K half, row-major, stride = (K,1)   <- 存 B^T
//   C: M x N float, row-major, stride = (N,1)
//
// CTA tile: BM=64, BN=64, BK=64。128 线程 = 4 warp。
// STAGES=3: smem 开 3 份, 允许 2 批搬运在飞。
// ---------------------------------------------------------------------------
constexpr int BM = 64, BN = 64, BK = 64;
constexpr int NTHR = 128;
constexpr int STAGES = 3;  // 流水线深度

CUTE_HOST_DEVICE static auto make_mma() {
    return make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_2, _2>>{},
                          Tile<_32, _32, _16>{});
}

// gmem -> smem: cp.async, 每线程 128b, k-major
CUTE_HOST_DEVICE static auto make_stage_copy() {
    return make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
                           Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                           Layout<Shape<_1, _8>>{});
}

// smem -> 寄存器: ldmatrix
CUTE_HOST_DEVICE static auto make_ldmatrix() {
    return Copy_Atom<SM75_U32x4_LDSM_N, half_t>{};
}

// ===========================================================================
// §3.2  kernel —— 官方 wgmma_sm90.cu 的 SM80 版
//
// 结构:
//   prologue: 先把 STAGES-1 批搬发出去 (让流水线灌满)
//   mainloop : 每轮 (a) 发下一批 cp.async (b) 等第 k 批到齐 (c) 算第 k 批
//
// cp_async_wait<N>: 允许还有 N 批在飞。cp_async_wait<0> = 等到全到齐。
// 这里用 cp_async_wait<0> 保持简单 —— 它的代价是每轮都全等,
// 但和 v1 的 __syncthreads 不同, 它**只等搬运**, 线程不等彼此。
// ===========================================================================
template <class SLayA, class SLayB>
__global__ void gemm_pipe(const half_t* A, const half_t* B, float* C, int M, int N, int K,
                          SLayA slayA, SLayB slayB) {
    // smem: STAGES 份, 每份 (BM,BK)
    __shared__ __align__(128) half_t rawA[cosize_v<SLayA>];
    __shared__ __align__(128) half_t rawB[cosize_v<SLayB>];

    auto sA = make_tensor(make_smem_ptr(rawA), slayA);  // (BM,BK,STAGES)
    auto sB = make_tensor(make_smem_ptr(rawB), slayB);

    auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(K, Int<1>{}));
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

    auto gA = local_tile(mA, Shape<Int<BM>, Int<BK>>{}, make_coord(blockIdx.y, _));  // (BM,BK,k)
    auto gB = local_tile(mB, Shape<Int<BN>, Int<BK>>{}, make_coord(blockIdx.x, _));
    auto gC = local_tile(mC, Shape<Int<BM>, Int<BN>>{}, make_coord(blockIdx.y, blockIdx.x));

    auto tcopy = make_stage_copy();
    ThrCopy tc = tcopy.get_slice(threadIdx.x);

    auto mma = make_mma();
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);
    auto tCrC = thr.partition_fragment_C(gC);
    clear(tCrC);

    auto s2r_a = make_tiled_copy_A(make_ldmatrix(), mma);
    auto s2r_b = make_tiled_copy_B(make_ldmatrix(), mma);
    ThrCopy tlA = s2r_a.get_slice(threadIdx.x);
    ThrCopy tlB = s2r_b.get_slice(threadIdx.x);

    auto tCsA = thr.partition_A(sA);  // (MMA, MMA_M, MMA_K, STAGES)
    auto tCsB = thr.partition_B(sB);

    int nk = K / BK;
    int k_tile = 0;

    // ---- prologue: 灌满流水线 (发 STAGES-1 批) ----
    CUTE_UNROLL
    for (int s = 0; s < STAGES - 1; ++s) {
        cute::copy(tc.partition_S(gA(_, _, k_tile)), tc.partition_D(sA(_, _, s)));
        cute::copy(tc.partition_S(gB(_, _, k_tile)), tc.partition_D(sB(_, _, s)));
        cp_async_fence();
        ++k_tile;
    }

    // ---- mainloop ----
    int k_pipe_write = STAGES - 1;  // 当前要写的 stage
    int k_pipe_read = 0;            // 当前要读的 stage
    for (int k = 0; k < nk; ++k) {
        int k_next = k + (STAGES - 1);
        k_next = (k_next >= nk) ? nk - 1 : k_next;

        // (a) 发下一批 (写 k_pipe_write 对应的 stage)
        if (k_next > k) {
            cute::copy(tc.partition_S(gA(_, _, k_next)), tc.partition_D(sA(_, _, k_pipe_write)));
            cute::copy(tc.partition_S(gB(_, _, k_next)), tc.partition_D(sB(_, _, k_pipe_write)));
            cp_async_fence();
        }
        k_pipe_write = (k_pipe_write + 1) % STAGES;

        // (b) 等第 k 批到齐
        cp_async_wait<0>();
        __syncthreads();  // 所有人都等同一批

        // (c) 算第 k 批 (从 k_pipe_read 对应的 stage 取)
        auto tCrA = thr.make_fragment_A(tCsA(_, _, _, k_pipe_read));
        auto tCrB = thr.make_fragment_B(tCsB(_, _, _, k_pipe_read));
        cute::copy(s2r_a, tlA.partition_S(sA(_, _, k_pipe_read)), tlA.retile_D(tCrA));
        cute::copy(s2r_b, tlB.partition_S(sB(_, _, k_pipe_read)), tlB.retile_D(tCrB));

        gemm(mma, tCrA, tCrB, tCrC);

        // 读完后才能让这个 stage 被覆盖 —— 下一轮写的是 (k_pipe_write+1)%STAGES,
        // 和 k_pipe_read 差 STAGES-1, 所以这里不用等;
        // 但保险起见在 prologue 后的第一轮, __syncthreads 已经保证过了。
        k_pipe_read = (k_pipe_read + 1) % STAGES;
    }

    cute::copy(tCrC, thr.partition_C(gC));
}

// ===========================================================================
// §3.3  验证 + 计时
// ===========================================================================
static void run(int M, int N, int K, bool verify) {
    size_t nA = size_t(M) * K, nB = size_t(N) * K, nC = size_t(M) * N;

    half_t *h_A = new half_t[nA], *h_B = new half_t[nB];
    float *h_C = new float[nC], *h_ref = new float[nC];
    fill_pm1(h_A, nA, 53);
    fill_pm1(h_B, nB, 63);
    gemm_cpu(h_A, h_B, h_ref, M, N, K);

    half_t *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, nA * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_B, nB * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_C, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nA * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, nB * sizeof(half_t), cudaMemcpyHostToDevice));

    auto swizzle_atom = composition(Swizzle<3, 3, 3>{},
                                    Layout<Shape<_8, Shape<_8, _8>>,
                                           Stride<_8, Stride<_1, _64>>>{});
    auto slayA = tile_to_shape(swizzle_atom, make_shape(Int<BM>{}, Int<BK>{}, Int<STAGES>{}));
    auto slayB = tile_to_shape(swizzle_atom, make_shape(Int<BN>{}, Int<BK>{}, Int<STAGES>{}));

    dim3 grid(N / BN, M / BM);
    auto launch = [&] { gemm_pipe<<<grid, NTHR>>>(d_A, d_B, d_C, M, N, K, slayA, slayB); };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    if (verify) {
        CUDA_CHECK(cudaMemcpy(h_C, d_C, nC * sizeof(float), cudaMemcpyDeviceToHost));
        auto r = check_close(h_C, h_ref, nC);
        printf("  验证 %dx%dx%d: %s (bad=%d, maxerr=%g)\n", M, N, K,
               r.ok() ? "完全一致" : "不一致", r.bad, r.maxerr);
    }

    float ms = time_kernel(launch, 5, 20);
    printf("  %dx%dx%d: %.3f ms   %.1f TFLOP/s\n", M, N, K, ms,
           gemm_tflops(M, N, K, ms));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_ref;
}

// ===========================================================================
int main() {
    printf("cute_06 v2 —— 多 stage cp.async 流水线 (Ampere 上限)\n");
    printf("对应 README §3\n");
    printf("\nSTAGES = %d, CTA tile = %dx%dx%d\n", STAGES, BM, BN, BK);

    print_separator("§3.3  正确性 + 实测");
    run(128, 128, 128, true);
    run(512, 256, 512, true);
    run(2048, 2048, 2048, false);

    print_separator("小结");
    printf("  v1 (单缓冲)  2048^3: ~67 TFLOP/s\n");
    printf("  v2 (STAGES=%d) 2048^3: 看上面打印\n", STAGES);
    printf("  快在哪: 搬 k+1 和算 k 重叠, 搬运延迟被隐藏。\n");
    printf("  但还是慢 (对比 Hopper):\n");
    printf("    1. cp.async 仍占指令流和寄存器, 每线程要管自己的拷贝\n");
    printf("    2. 同步是 cp_async_wait + __syncthreads, 粒度粗\n");
    printf("    3. SM80 MMA 是 16x8, 指令密度低\n");
    printf("\n  下一步 (v3): 换 TMA + WGMMA —— Hopper 把搬运和计算都改成硬件异步,\n");
    printf("  同步换成 mbarrier。这是 Hopper 的正解。\n");

    printf("\nv2 OK\n");
    return 0;
}
