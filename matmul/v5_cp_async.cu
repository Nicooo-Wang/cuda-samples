// v5: cp.async 双缓冲 —— 让搬运和计算真正重叠起来。
//
// 前面四版有一个共同的结构性问题：K 循环里搬运和计算被两个 __syncthreads() 完全串起来。
//
//   v4:  [搬 tile] sync [算 tile] sync [搬 tile] sync [算 tile] sync ...
//         ^^^^^^^^ 搬的时候 tensor core 全空转，算的时候访存单元全空闲
//
// 这一版开两块 smem 轮换用，算第 i 块的同时预取第 i+1 块：
//
//   v5:  [发起搬 0]
//        [发起搬 1] [算 0]     <- 这两件事同时在跑
//        [发起搬 2] [算 1]
//        ...
//
// 靠两个东西实现：
//
// 1) cp.async：异步拷贝指令。普通的 *(float4*)dst = *(float4*)src 会先把数据读到寄存器
//    再写进 smem，线程必须停下来等数据到；cp.async 直接让 global -> smem，发起后线程
//    立刻继续往下走，不占寄存器也不阻塞。什么时候数据到位由 commit_group / wait_group 控制。
//
// 2) 两块 smem 缓冲（STAGES = 2）。因为预取的目标不能是正在被读的那一块。
//
// cp.async 的组语义：每次 commit_group 把之前发起的所有 cp.async 封成一组，
// wait_group N 表示"允许最新的 N 组还在飞，其余必须已完成"。这里 wait_group 0
// 是等 STAGES-2 = 0 组在飞，也就是等除了刚发起的那组以外全部到位。
//
// 实测 223 -> 249 TFLOP/s。同时 warp tile 在这一版可以放大到 64x64 了：
//   warp tile 64x32 (256 线程)  230 TFLOP/s
//   warp tile 64x64 (128 线程)  249   <- 本版
//   warp tile 32x32 (512 线程)  193
// 和 v4 正好相反（那里 64x32 是 224、64x64 是 184）。v4 里 64x64 更慢是因为 128 线程的
// occupancy 盖不住访存延迟；现在延迟被 cp.async 藏起来了，寄存器里 fragment 复用更充分
// 的那一侧就赢了。每个 warp 持 4x4 = 16 个 accumulator fragment，128 个寄存器，
// 是这版寄存器压力的来源。
//
// 另外这版 smem 用到 70 KB，超过静态 __shared__ 的 48 KB 上限，所以改成动态申请，
// 见下面 extern __shared__ 和 main() 里的 cudaFuncSetAttribute。
//
// 还能往下走的方向（都不在这一版）：
//   - 多级流水线。把 STAGES 改成 3 / 4 就能试，实测 2 级 249、3 级 254、4 级 199：
//     3 级只多 2%，4 级因为 140 KB smem 把每个 SM 能驻留的 block 压到 1 个，反而掉下去。
//     2048 规模下 global 延迟已经被两级基本盖住，多级流水线要在更大的矩阵、
//     或者搬运更慢（比如 fp8 反量化）的场景才划算。
//   - 寄存器双缓冲：smem -> 寄存器这一跳也做预取，把 load_matrix_sync 的延迟藏掉。
//   - swizzle 替代 padding，省下 padding 占的那部分 smem。
//   - ldmatrix + mma.sync 手写 PTX，绕开 wmma 对 fragment 布局的封装。
//   - 剩下到 cuBLAS 的差距主要要靠 Hopper 的 wgmma + TMA，那是 cute 教程的题材。
#include <cuda_fp16.h>
#include <mma.h>

#include "common.h"

using namespace nvcuda;

constexpr int BM = 128, BN = 128, BK = 64;
constexpr int WARP_M = 64, WARP_N = 64;   // 这一版放大到 64x64
constexpr int WTILE_M = WARP_M / WMMA_M;  // 4
constexpr int WTILE_N = WARP_N / WMMA_N;  // 4 —— 每个 warp 16 个 wmma tile
constexpr int WARPS_M = BM / WARP_M;      // 2
constexpr int WARPS_N = BN / WARP_N;      // 2
constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;  // 4 warp = 128 线程

constexpr int PAD = 8;
constexpr int LDA = BK + PAD;  // 72
constexpr int LDB = BN + PAD;  // 136

constexpr int STAGES = 2;  // 双缓冲。改成 3 / 4 就是多级流水线
constexpr size_t SMEM_BYTES = (size_t)STAGES * (BM * LDA + BK * LDB) * sizeof(half);  // 70 KB

// 发起一次 16 字节的 global -> smem 异步拷贝。
// cp.async 要的是 shared 窗口内的地址，不是通用地址，所以先用 __cvta_generic_to_shared 转换。
// .cg 是 cache global：数据只过 L2，不污染 L1，因为这些 tile 搬完就直接进 smem 了。
__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* global_src) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(global_src));
}

__global__ void matmul_v5(const half* __restrict__ A, const half* __restrict__ B,
                          float* __restrict__ C, int m, int n, int k) {
    // 动态 smem：大小由 launch 时的第三个参数给出，kernel 里只拿到一个起始指针，
    // 所以两块缓冲要自己算偏移切出来。
    extern __shared__ half smem[];
    half* As = smem;                               // STAGES 块，每块 BM * LDA
    half* Bs = smem + STAGES * BM * LDA;           // STAGES 块，每块 BK * LDB

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WTILE_M][WTILE_N];
#pragma unroll
    for (int i = 0; i < WTILE_M; ++i)
#pragma unroll
        for (int j = 0; j < WTILE_N; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    // 发起第 stage 块缓冲的搬运，对应 K 方向偏移 k0。切分方式和前面几版一样，
    // 只是把 float4 赋值换成 cp_async_16B，最后用 commit_group 把这批封成一组。
    auto issue_copy = [&](int stage, int k0) {
        half* as = As + stage * BM * LDA;
        half* bs = Bs + stage * BK * LDB;
        for (int i = tid; i < BM * (BK / 8); i += NUM_THREADS) {
            const int r = i / (BK / 8), c = (i % (BK / 8)) * 8;
            cp_async_16B(&as[r * LDA + c], &A[(size_t)(block_row + r) * k + k0 + c]);
        }
        for (int i = tid; i < BK * (BN / 8); i += NUM_THREADS) {
            const int r = i / (BN / 8), c = (i % (BN / 8)) * 8;
            cp_async_16B(&bs[r * LDB + c], &B[(size_t)(k0 + r) * n + block_col + c]);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    // 算第 stage 块缓冲，和 v4 的计算部分完全一样
    auto compute = [&](int stage) {
        const half* as = As + stage * BM * LDA;
        const half* bs = Bs + stage * BK * LDB;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WTILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WTILE_N];
#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
#pragma unroll
            for (int i = 0; i < WTILE_M; ++i)
                wmma::load_matrix_sync(a_frag[i], &as[(warp_m * WARP_M + i * WMMA_M) * LDA + kk],
                                       LDA);
#pragma unroll
            for (int j = 0; j < WTILE_N; ++j)
                wmma::load_matrix_sync(b_frag[j], &bs[kk * LDB + warp_n * WARP_N + j * WMMA_N], LDB);
#pragma unroll
            for (int i = 0; i < WTILE_M; ++i)
#pragma unroll
                for (int j = 0; j < WTILE_N; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
    };

    const int num_k_tiles = k / BK;

    // 序幕：先把前 STAGES-1 块灌进流水线
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) issue_copy(s, s * BK);

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // 等到只剩最新 STAGES-2 组还在飞，也就是本轮要算的那块已经到位
        asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));
        __syncthreads();

        // 先发起预取，再算。顺序很关键：发起是非阻塞的，紧接着的计算就成了拷贝的掩体。
        // 目标槽位 (kt + STAGES-1) % STAGES 上一轮已经算完，上面那个 barrier 保证了
        // 所有 warp 都读完了，可以安全覆盖。
        const int next_tile = kt + STAGES - 1;
        if (next_tile < num_k_tiles) {
            issue_copy(next_tile % STAGES, next_tile * BK);
        } else {
            // 尾部没有东西要搬了，但仍要 commit 一个空组，
            // 否则组计数会错位，wait_group 等的就不是本轮那一块了。
            asm volatile("cp.async.commit_group;\n" ::);
        }

        compute(kt % STAGES);
    }

#pragma unroll
    for (int i = 0; i < WTILE_M; ++i)
#pragma unroll
        for (int j = 0; j < WTILE_N; ++j) {
            const size_t row = block_row + warp_m * WARP_M + i * WMMA_M;
            const size_t col = block_col + warp_n * WARP_N + j * WMMA_N;
            wmma::store_matrix_sync(&C[row * n + col], acc[i][j], n, wmma::mem_row_major);
        }
}

int main() {
    const size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;

    half* h_A = (half*)malloc(a_elems * sizeof(half));
    half* h_B = (half*)malloc(b_elems * sizeof(half));
    float* h_C = (float*)malloc(c_elems * sizeof(float));
    float* h_ref = (float*)malloc(c_elems * sizeof(float));
    fill_inputs(h_A, a_elems, h_B, b_elems);

    printf("v5 cp.async double buffer  (M=%d, N=%d, K=%d)\n", M, N, K);
    printf("  block tile %dx%dx%d, warp tile %dx%d, %d warp, %d stage, smem %zu B\n", BM, BN, BK,
           WARP_M, WARP_N, NUM_THREADS / 32, STAGES, SMEM_BYTES);

    half *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, a_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, b_elems * sizeof(half), cudaMemcpyHostToDevice));

    cublas_reference(d_A, d_B, d_C, h_ref);
    CUDA_CHECK(cudaMemset(d_C, 0x7F, c_elems * sizeof(float)));

    // 动态 smem 超过 48 KB 必须显式 opt-in，否则 launch 直接返回 invalid argument。
    // 这一步顺带也把 kernel 模块加载了，作用和前几版的 cudaFuncGetAttributes 一样：
    // 把一次性开销挤出计时区间。
    CUDA_CHECK(cudaFuncSetAttribute(matmul_v5, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)SMEM_BYTES));

    dim3 block(NUM_THREADS);
    dim3 grid(CEIL(N, BN), CEIL(M, BM));

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    // 第三个参数是动态 smem 的字节数
    matmul_v5<<<grid, block, SMEM_BYTES>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    report_perf(ms);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
    const bool pass = check_result(h_C, h_ref);

    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return pass ? 0 : 1;
}
