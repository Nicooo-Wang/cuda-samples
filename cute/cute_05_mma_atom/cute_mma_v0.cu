// cute_05 v0 —— MMA Atom 的解剖: 一条 Tensor Core 指令长什么样
//
// 对应 README §1 ~ §2。
//
// cute_03/04 讲完了"数据怎么搬", 这一章换到另一半: "数据怎么算"。
// 起点是最朴素的问题 —— 一条 Tensor Core 指令到底是什么形状、
// 谁提供数据、结果落在谁的寄存器里。
//
// 这个文件全部用 **Ampere 的 warp 级 MMA** (SM80_16x8x16_F32F16F16F32_TN)。
// 不是因为它新, 恰恰因为它**看得见**: A/B/C 全在寄存器里, 每个线程持有几个元素
// 可以打印出来数。等 v1 换成 Hopper 的 WGMMA, 这些寄存器会消失一半 ——
// 到那时你才能看出 Hopper 改了什么。
//
//   §1.1  一条 MMA 指令的三个形状       (README §1.1)
//   §1.2  ThrID: 谁参与这条指令          (README §1.2)
//   §1.3  LayoutA/B/C_TV: 数据怎么分给线程 (README §1.3)
//   §2.1  partition_fragment_*: 拿到我这份 (README §2.1)
//   §2.2  跑一条真的 MMA, 逐点验证        (README §2.2)
//   §2.3  TiledMMA: 用多个 warp 拼大一点  (README §2.3)
//
// 多卡机器上请指定一张空闲卡:  CUDA_VISIBLE_DEVICES=<idle> ./cute_mma_v0

#include <cute/tensor.hpp>
#include <cstdio>

#include "common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// 全局配置
//
// 这一版只跑**一条** MMA 指令能覆盖的形状, 不多不少:
//   SM80_16x8x16 的意思就是 M=16, N=8, K=16。
// 所以矩阵就取这么大, 一个 warp 一次算完, 没有任何循环 —— 便于逐点核对。
//
// 数据摆法 (TN, 见 common.h 的说明):
//   A: M x K = 16 x 16 half, row-major, stride = (K,1) = (16,1)
//   B: N x K =  8 x 16 half, row-major, stride = (K,1) = (16,1)   <- 存的是 B^T
//   C: M x N = 16 x  8 float, row-major, stride = (N,1) = (8,1)
// ---------------------------------------------------------------------------
constexpr int MM = 16, NN = 8, KK = 16;

// ===========================================================================
// §1  MMA Atom 的静态解剖 —— 一行代码都不用跑, 全是编译期信息
//
// MMA_Atom 是一条硬件指令的 CuTe 封装。它带着四样东西:
//   ThrID       : 这条指令需要几个线程一起发 (Ampere: 32 = 一个 warp)
//   Shape_MNK   : 这条指令算多大的一块 (16x8x16)
//   LayoutA/B/C_TV : (线程, 值) -> 逻辑坐标 的映射, 即"数据怎么分给线程"
//
// 这一节只是把它们打印出来看。README §1 有对应的图。
// ===========================================================================
static void show_atom_anatomy() {
    print_separator("§1  MMA Atom 的解剖 —— Ampere warp 级 MMA");

    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{});

    printf("\n  make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}) 得到:\n\n");
    print(mma);
    printf("\n");

    printf("\n  §1.1  三个形状\n");
    printf("    Shape_MNK = (16,8,16)  <- 一条指令算 C[16x8] += A[16x16] * B[8x16]^T\n");
    printf("    名字里的 F32F16F16F32 = 累加器 f32, A f16, B f16, C f32\n");
    printf("    名字里的 TN = A 行主(K 连续) + B 行主(K 连续), 两边 K 都连续\n");

    printf("\n  §1.2  ThrID = _32  ->  这条指令由一个 warp 的 32 个线程共同发出\n");
    printf("    不是 1 个线程发, 也不是整个 block 发。size(mma) = %d\n", int(size(mma)));

    printf("\n  §1.3  LayoutA/B/C_TV —— (线程,值) -> 坐标\n");
    printf("    读法: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))\n");
    printf("      左边 (_4,_8)     = 32 个线程被看成 4x8 的排布\n");
    printf("      右边 (_2,_2,_2)  = 每个线程持有 8 个 A 元素\n");
    printf("    16x16 的 A 共 256 个 half, 分给 32 个线程 -> 每人 8 个。对上了。\n");
}

// ===========================================================================
// §2.1  partition_fragment_* —— 从"整块"到"我这一份"
//
// 上面 LayoutA_TV 是"32 个线程整体"的映射。真正写 kernel 时, 每个线程只关心
// 自己那几个寄存器。partition_fragment_A/B/C 就是把整块按 TV 布局切给当前线程。
//
// 这个函数只做一件事: 打印每个线程分到几个元素, 和上面手算的数对上。
// ===========================================================================
static void show_fragment_sizes() {
    print_separator("§2.1  partition_fragment_* —— 每个线程分到多少");

    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{});
    ThrMMA thr = mma.get_thread_slice(0);  // 线程 0 的视角

    // 注意: 这里的 tensor 只是拿来"描述形状"的, 指针是空的也无所谓 ——
    // partition_fragment_* 只看 layout, 不解引用。这是 cute_03 §4
    // 「host 描述, kernel 索引」的又一次体现。
    auto gA = make_tensor(make_gmem_ptr((half_t*)nullptr),
                          make_shape(Int<MM>{}, Int<KK>{}), make_stride(Int<KK>{}, Int<1>{}));
    auto gB = make_tensor(make_gmem_ptr((half_t*)nullptr),
                          make_shape(Int<NN>{}, Int<KK>{}), make_stride(Int<KK>{}, Int<1>{}));
    auto gC = make_tensor(make_gmem_ptr((float*)nullptr),
                          make_shape(Int<MM>{}, Int<NN>{}), make_stride(Int<NN>{}, Int<1>{}));

    auto fA = thr.partition_fragment_A(gA);
    auto fB = thr.partition_fragment_B(gB);
    auto fC = thr.partition_fragment_C(gC);

    printf("\n  A 是 %dx%d half = %d 个元素, 32 线程平分 -> 每人 %d\n", MM, KK, MM * KK,
           MM * KK / 32);
    printf("    partition_fragment_A -> "); print(fA); printf("\n");
    printf("    size = %d  ✓\n", int(size(fA)));

    printf("\n  B 是 %dx%d half = %d 个元素, 32 线程平分 -> 每人 %d\n", NN, KK, NN * KK,
           NN * KK / 32);
    printf("    partition_fragment_B -> "); print(fB); printf("\n");
    printf("    size = %d  ✓\n", int(size(fB)));

    printf("\n  C 是 %dx%d float = %d 个元素, 32 线程平分 -> 每人 %d\n", MM, NN, MM * NN,
           MM * NN / 32);
    printf("    partition_fragment_C -> "); print(fC); printf("\n");
    printf("    size = %d  ✓\n", int(size(fC)));

    printf("\n  这三个 fragment 都是**寄存器**里的 tensor (没有 smem_ptr/gmem_ptr 前缀)。\n");
    printf("  记住这一点 —— v1 换成 WGMMA 之后, A 和 B 的 fragment 会变成别的东西。\n");
}

// ===========================================================================
// §2.2  跑一条真的 MMA
//
// kernel 只有一个 warp, 只发一条 MMA 指令, 算 C[16x8] = A[16x16] * B[8x16]^T。
//
// 完整链路是四步, 每一步都对应 CuTe 的一个动作:
//   1. gmem -> 寄存器 : copy(),  按 TV 布局取到"我这一份"
//   2. 清累加器       : clear()
//   3. 算             : gemm(mma, fA, fB, fC)
//   4. 寄存器 -> gmem : copy(),  按同一个 TV 布局写回去
//
// 关键: 第 1 步和第 4 步用的是**同一个** partition, 所以每个线程读哪几个、
// 写哪几个是自洽的 —— 不需要你手算 lane id 到坐标的公式。
// ===========================================================================
__global__ void mma_one_instruction(const half_t* A, const half_t* B, float* C) {
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{});
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);

    // 整块的视图 (所有线程看到的是同一个)
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(Int<MM>{}, Int<KK>{}),
                          make_stride(Int<KK>{}, Int<1>{}));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(Int<NN>{}, Int<KK>{}),
                          make_stride(Int<KK>{}, Int<1>{}));
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(Int<MM>{}, Int<NN>{}),
                          make_stride(Int<NN>{}, Int<1>{}));

    // "我这一份"在 gmem 里的位置 (还是 gmem tensor, 只是被切碎了)
    auto tAgA = thr.partition_A(gA);  // (MMA, MMA_M, MMA_K)
    auto tBgB = thr.partition_B(gB);
    auto tCgC = thr.partition_C(gC);

    // "我这一份"在寄存器里的容器
    auto tArA = thr.partition_fragment_A(gA);
    auto tBrB = thr.partition_fragment_B(gB);
    auto tCrC = thr.partition_fragment_C(gC);

    // 1. 搬进寄存器 —— partition 已经把地址算好了, copy 只是照着搬
    copy(tAgA, tArA);
    copy(tBgB, tBrB);

    // 2. 累加器清零 (MMA 是 C += A*B, 不清零就是在旧值上累加)
    clear(tCrC);

    // 3. 算 —— 一条指令
    gemm(mma, tArA, tBrB, tCrC);

    // 4. 写回
    copy(tCrC, tCgC);
}

static void run_one_instruction() {
    print_separator("§2.2  跑一条真的 MMA 指令");

    size_t bytesA = size_t(MM) * KK * sizeof(half_t);
    size_t bytesB = size_t(NN) * KK * sizeof(half_t);
    size_t bytesC = size_t(MM) * NN * sizeof(float);

    half_t *h_A = new half_t[MM * KK], *h_B = new half_t[NN * KK];
    float *h_C = new float[MM * NN], *h_ref = new float[MM * NN];
    fill_pm1(h_A, MM * KK, 1);
    fill_pm1(h_B, NN * KK, 2);
    gemm_cpu(h_A, h_B, h_ref, MM, NN, KK);

    half_t *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    // 一个 warp, 一条指令
    mma_one_instruction<<<1, 32>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    auto r = check_close(h_C, h_ref, MM * NN);
    printf("\n  C[%dx%d] = A[%dx%d] * B[%dx%d]^T, 由 1 个 warp 的 1 条 MMA 指令完成\n", MM, NN, MM,
           KK, NN, KK);
    printf("  与 CPU 参考比对: %s   (bad=%d, maxerr=%g)\n", r.ok() ? "完全一致" : "不一致", r.bad,
           r.maxerr);

    printf("\n  C 的左上角 4x4:\n");
    for (int i = 0; i < 4; ++i) {
        printf("    ");
        for (int j = 0; j < 4; ++j) printf("%7.1f", h_C[i * NN + j]);
        printf("      (参考: ");
        for (int j = 0; j < 4; ++j) printf("%7.1f", h_ref[i * NN + j]);
        printf(")\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_ref;
}

// ===========================================================================
// §2.3  TiledMMA —— 一条指令不够大, 用多个 warp 拼
//
// 一条 SM80 MMA 只算 16x8。真实的 GEMM tile 是 128x128 起步, 所以要拼:
//   make_tiled_mma(atom, thr_layout) 里的 thr_layout 说的是
//   **用几个 warp、按什么排布** 去覆盖一块更大的区域。
//
//   Layout<Shape<_2,_2,_1>>  = 2x2 共 4 个 warp -> 128 个线程
//                              覆盖 M 方向 2 份、N 方向 2 份
//                           -> tile 变成 (16*2) x (8*2) = 32 x 16
//
// 注意这里放大的是**指令的重复次数**, 不是指令本身。硬件指令永远是 16x8x16。
// ===========================================================================
// TM/TN/TK 必须是**编译期常量**: fragment 是寄存器, 寄存器个数不能到运行时才知道。
// 传运行时的 int 进来会撞上 "Dynamic owning tensors not supported"。
// 这不是限制, 是提醒: tile 尺寸本来就该是编译期决定的。
constexpr int TM = 32, TN = 16, TK = 16;

__global__ void mma_tiled(const half_t* A, const half_t* B, float* C) {
    // 2x2 个 warp = 128 线程, 覆盖 32x16 的 C
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_2, _2, _1>>{});
    ThrMMA thr = mma.get_thread_slice(threadIdx.x);

    auto gA = make_tensor(make_gmem_ptr(A), make_shape(Int<TM>{}, Int<TK>{}),
                          make_stride(Int<TK>{}, Int<1>{}));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(Int<TN>{}, Int<TK>{}),
                          make_stride(Int<TK>{}, Int<1>{}));
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(Int<TM>{}, Int<TN>{}),
                          make_stride(Int<TN>{}, Int<1>{}));

    auto tArA = thr.partition_fragment_A(gA);
    auto tBrB = thr.partition_fragment_B(gB);
    auto tCrC = thr.partition_fragment_C(gC);

    copy(thr.partition_A(gA), tArA);
    copy(thr.partition_B(gB), tBrB);
    clear(tCrC);
    gemm(mma, tArA, tBrB, tCrC);
    copy(tCrC, thr.partition_C(gC));
}

static void run_tiled() {
    print_separator("§2.3  TiledMMA —— 4 个 warp 拼出 32x16");

    // 2x2 warp 覆盖的区域: M = 16*2 = 32, N = 8*2 = 16, K 仍是一条指令的 16
    // (TM/TN/TK 在 kernel 上方定义为编译期常量, 原因见那里的注释)

    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_2, _2, _1>>{});
    printf("\n  make_tiled_mma(atom, Layout<Shape<_2,_2,_1>>{}) 得到:\n");
    printf("    线程数 size(mma) = %d   (4 个 warp)\n", int(size(mma)));
    printf("    覆盖的 tile      = %dx%dx%d\n", TM, TN, TK);
    printf("    硬件指令仍然是   = 16x8x16 (发了 4 次)\n");

    half_t *h_A = new half_t[TM * TK], *h_B = new half_t[TN * TK];
    float *h_C = new float[TM * TN], *h_ref = new float[TM * TN];
    fill_pm1(h_A, TM * TK, 3);
    fill_pm1(h_B, TN * TK, 4);
    gemm_cpu(h_A, h_B, h_ref, TM, TN, TK);

    half_t *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, TM * TK * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_B, TN * TK * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_C, TM * TN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, TM * TK * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, TN * TK * sizeof(half_t), cudaMemcpyHostToDevice));

    mma_tiled<<<1, int(size(mma))>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, TM * TN * sizeof(float), cudaMemcpyDeviceToHost));

    auto r = check_close(h_C, h_ref, TM * TN);
    printf("\n  与 CPU 参考比对: %s   (bad=%d, maxerr=%g)\n", r.ok() ? "完全一致" : "不一致", r.bad,
           r.maxerr);

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
    printf("cute_05 v0 —— MMA Atom 的解剖 (Ampere warp 级 MMA)\n");
    printf("对应 README §1 ~ §2\n");

    show_atom_anatomy();
    show_fragment_sizes();
    run_one_instruction();
    run_tiled();

    print_separator("小结");
    printf("  §1  一个 MMA_Atom = ThrID(谁发) + Shape_MNK(算多大) + LayoutA/B/C_TV(怎么分)\n");
    printf("  §2  partition_fragment_* 按 TV 布局给每个线程切出寄存器份额\n");
    printf("      Ampere: A/B/C 三份都是**真寄存器**, 每线程 8/4/4 个元素\n");
    printf("      TiledMMA 用多个 warp 重复同一条指令去覆盖更大的 tile\n");
    printf("\n  下一步 (v1): Hopper 的 WGMMA 会把这张图改掉一半 ——\n");
    printf("      A 和 B 不再进寄存器, 硬件直接读 smem。为什么? 怎么写? 见 v1。\n");

    printf("\nv0 OK\n");
    return 0;
}
