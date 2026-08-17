// cute_03 练习：把 TODO 填掉，然后 make run
//
// 题目见 README 的"练习"一节。参考解答在 solutions.md。
// 每题都有一个自动检查，填对了会打印 PASS。

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cstdio>
#include <cuda_runtime.h>

using namespace cute;

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

static int g_pass = 0, g_fail = 0;

void expect(const char* what, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    ok ? ++g_pass : ++g_fail;
}

template <class L1, class L2>
void expect_layout(const char* what, L1 const& got, L2 const& want) {
    bool ok = (size(got) == size(want));
    for (int i = 0; ok && i < int(size(want)); ++i)
        if (int(got(i)) != int(want(i))) ok = false;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        printf("        got  = "); print(got);  printf("\n");
        printf("        want = "); print(want); printf("\n");
    }
    ok ? ++g_pass : ++g_fail;
}

// ===========================================================================
// 练习 1 — Copy_Atom 三件套 ★☆☆
//
// 不看代码先答：Copy_Atom<UniversalCopy<uint64_t>, half_t> 的 NumValSrc 是多少？
// 把你的答案写进 EX1_NUMVAL。
// ===========================================================================
constexpr int EX1_NUMVAL = 4;  // TODO: 改成你的答案

void ex1() {
    printf("\n--- 练习 1: Copy_Atom 三件套 ---\n");
    using Atom = Copy_Atom<UniversalCopy<uint64_t>, half_t>;
    printf("  实际 NumValSrc = %d  (ThrID size = %d)\n", Atom::NumValSrc,
           int(size(typename Atom::ThrID{})));
    expect("EX1_NUMVAL 等于实际值", EX1_NUMVAL == Atom::NumValSrc);
}

// ===========================================================================
// 练习 2 — 设计一个 TiledCopy ★★☆
//
// 目标：128 个线程搬一块 32x32 的 float，用满 128bit 向量指令。
// 要求 Tiler_MN 恰好是 (32,32)，线程数恰好 128。
//
// 先算总量: 32*32 = 1024 个 float / 128 线程 = 每线程 8 个。
// 而一条 128bit 指令只搬 4 个 float —— 所以每线程要发 2 条。
// 这 8 个怎么摆？val_layout 的两个维度相乘必须等于 8，
// 且**连续方向（列）必须是 4**，才能凑出 128bit。
//   => val_layout = (2,4)   行方向 2 个、列方向 4 个
// 再由 Tiler_MN = thr_shape * val_shape 反推 thr_layout。
// ===========================================================================
void ex2() {
    printf("\n--- 练习 2: 设计 TiledCopy ---\n");

    auto atom = Copy_Atom<UniversalCopy<uint128_t>, float>{};

    // TODO: 填 thr_layout。128 个线程，要让相邻线程访问相邻地址。
    //       提示: 32x32 一块、每线程 4 个 -> 列方向需要 32/4 = 8 个线程
    //       下面是能编译的占位值（1x1 个线程），改成正确的形状。
    auto thr_layout = make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));

    // TODO: 填 val_layout。每线程 8 个 float = (行方向 ?, 列方向 4)。
    //       注意: 列方向必须 >= 4，否则编译失败
    //       "TiledCopy uses too few vals for selected CopyAtom"。
    auto val_layout = make_layout(make_shape(Int<2>{}, Int<4>{}));

    auto tc = make_tiled_copy(atom, thr_layout, val_layout);

    printf("  Tiler_MN = "); print(typename decltype(tc)::Tiler_MN{}); printf("\n");
    printf("  线程数   = %d\n", int(size(tc)));

    expect("线程数 == 128", int(size(tc)) == 128);
    expect("Tiler_MN == (32,32)",
           int(size<0>(typename decltype(tc)::Tiler_MN{})) == 32 &&
               int(size<1>(typename decltype(tc)::Tiler_MN{})) == 32);
}

// ===========================================================================
// 练习 3 — 预测 partition_S 的形状 ★★☆
//
// A = 16x16 float (row-major)，tc = 32 线程 x 每人 4 个（128bit，列方向）。
// 先在纸上答：thr0 拿到的 4 个元素，偏移分别是多少？
// 把答案填进 EX3_OFFSETS，然后运行验证。
// ===========================================================================
constexpr int EX3_OFFSETS[4] = {0, 1, 2, 3};  // TODO

__global__ void ex3_kernel(const float* base, int* out) {
    auto lay = make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}));
    auto A = make_tensor(make_gmem_ptr(base), lay);

    auto tc = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                              make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                              make_layout(make_shape(Int<1>{}, Int<4>{})));
    auto p = tc.get_slice(0).partition_S(A);
    if (threadIdx.x == 0)
        for (int i = 0; i < 4; ++i) out[i] = int(&p(i) - base);
}

void ex3() {
    printf("\n--- 练习 3: 预测 partition_S ---\n");
    float* d_a;
    int* d_out;
    CUDA_CHECK(cudaMalloc(&d_a, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, 4 * sizeof(int)));
    ex3_kernel<<<1, 32>>>(d_a, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h[4];
    CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
    printf("  thr0 实际偏移 = %d %d %d %d\n", h[0], h[1], h[2], h[3]);

    bool ok = true;
    for (int i = 0; i < 4; ++i)
        if (h[i] != EX3_OFFSETS[i]) ok = false;
    expect("EX3_OFFSETS 猜对了", ok);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}

// ===========================================================================
// 练习 4 — 用 copy_if 处理尾块 ★★☆
//
// 一个长度 100 的数组，用 32 线程 x 每人 4 个（=128 一块）搬。
// 100 不是 128 的倍数，最后一块越界了。
// 用 copy_if + 谓词，把越界的位置挡住。
// ===========================================================================
__global__ void ex4_kernel(const float* src, float* dst, int n) {
    constexpr int TILE = 128, NTHR = 32, VEC = TILE / NTHR;

    auto tc = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{},
                              make_layout(Int<NTHR>{}, Int<1>{}),
                              make_layout(Int<VEC>{}, Int<1>{}));
    auto thr = tc.get_slice(threadIdx.x);

    int base = blockIdx.x * TILE;
    auto lay = make_layout(Int<TILE>{}, Int<1>{});
    auto S = make_tensor(make_gmem_ptr(src + base), lay);
    auto D = make_tensor(make_gmem_ptr(dst + base), lay);

    auto tS = thr.partition_S(S);
    auto tD = thr.partition_D(D);

    // 谓词 tensor：形状要和 tS 对得上
    auto pred = make_tensor<bool>(shape(tS));

    // TODO: 填谓词。第 i 个元素的全局下标是 base + (&tS(i) - (src + base))
    //       只有全局下标 < n 才允许搬。
    for (int i = 0; i < int(size(tS)); ++i) {
        if (base + (&tS(i) - (src + base)) >= n)
        {
            pred(i) = false; // TODO: 改成正确的条件
        }
        else
        {
            pred(i) = true; // TODO: 改成正确的条件
        }
    }

    copy_if(pred, tS, tD);
}

void ex4() {
    printf("\n--- 练习 4: copy_if 处理尾块 ---\n");
    constexpr int N = 100, TILE = 128;
    float *d_s, *d_d;
    CUDA_CHECK(cudaMalloc(&d_s, N * sizeof(float)));
    // dst 多分配一点，用来检测是否写越界
    CUDA_CHECK(cudaMalloc(&d_d, TILE * sizeof(float)));

    float hs[N], hd[TILE];
    for (int i = 0; i < N; ++i) hs[i] = float(i + 1);
    for (int i = 0; i < TILE; ++i) hd[i] = -1.f;
    CUDA_CHECK(cudaMemcpy(d_s, hs, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, hd, TILE * sizeof(float), cudaMemcpyHostToDevice));

    ex4_kernel<<<1, 32>>>(d_s, d_d, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hd, d_d, TILE * sizeof(float), cudaMemcpyDeviceToHost));

    bool copied = true, guarded = true;
    for (int i = 0; i < N; ++i)
        if (hd[i] != hs[i]) copied = false;
    for (int i = N; i < TILE; ++i)
        if (hd[i] != -1.f) guarded = false;

    printf("  前 100 个: %s   100..127: %s\n", copied ? "搬对了" : "没搬对",
           guarded ? "没被碰过" : "被写坏了");
    expect("有效范围全部搬到", copied);
    expect("越界部分没被写", guarded);

    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_d));
}

// ===========================================================================
// 练习 5 — max_common_vector ★★★
//
// 判断下面三对 src/dst 能不能用 128bit (=4 个 float) 搬。
// 先答再运行。
// ===========================================================================
constexpr int EX5_A = 32;  // TODO: max_common_vector(连续32, 连续32)
constexpr int EX5_B = 1;  // TODO: max_common_vector(连续32, stride-4 的 32)
constexpr int EX5_C = 1;  // TODO: max_common_vector((4,8):(8,1), (4,8):(1,4))

void ex5() {
    printf("\n--- 练习 5: max_common_vector ---\n");
    float* p = (float*)0x1000;

    auto c1 = make_tensor(make_gmem_ptr(p), make_layout(Int<32>{}, Int<1>{}));
    auto c2 = make_tensor(make_gmem_ptr(p + 128), make_layout(Int<32>{}, Int<1>{}));
    auto s4 = make_tensor(make_gmem_ptr(p + 128), make_layout(Int<32>{}, Int<4>{}));
    auto rm =
        make_tensor(make_gmem_ptr(p), make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})));
    auto cm = make_tensor(make_gmem_ptr(p + 128),
                          make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<1>{}, Int<4>{})));

    int a = int(max_common_vector(c1, c2));
    int b = int(max_common_vector(c1, s4));
    int c = int(max_common_vector(rm, cm));
    printf("  实际: 连续/连续=%d  连续/stride4=%d  row-major/col-major=%d\n", a, b, c);

    expect("EX5_A", EX5_A == a);
    expect("EX5_B", EX5_B == b);
    expect("EX5_C", EX5_C == c);
}

// ===========================================================================
// 练习 6 — 修一个 bug ★★★
//
// 下面这个 TiledCopy 结果是对的，但访存完全不合并。
// 找出问题并修好，让相邻线程访问相邻地址。
// ===========================================================================
__global__ void ex6_kernel(const float* base, int* out) {
    auto lay = make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}));
    auto A = make_tensor(make_gmem_ptr(base), lay);

    // TODO: 这个 thr_layout 是 col-major，导致相邻线程差了一整行。
    //       改成 row-major。
    auto thr_layout = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<8>{}, Int<1>{}));

    auto tc = make_tiled_copy(Copy_Atom<UniversalCopy<float>, float>{}, thr_layout,
                              make_layout(make_shape(Int<4>{}, Int<1>{})));

    auto p = tc.get_slice(threadIdx.x).partition_S(A);
    if (threadIdx.x < 4) out[threadIdx.x] = int(&p(0) - base);
}

void ex6() {
    printf("\n--- 练习 6: 修好不合并的访存 ---\n");
    float* d_a;
    int* d_out;
    CUDA_CHECK(cudaMalloc(&d_a, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, 4 * sizeof(int)));
    ex6_kernel<<<1, 32>>>(d_a, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h[4];
    CUDA_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
    printf("  thr0..3 的首地址偏移 = %d %d %d %d\n", h[0], h[1], h[2], h[3]);
    expect("相邻线程访问相邻地址 (0,1,2,3)",
           h[0] == 0 && h[1] == 1 && h[2] == 2 && h[3] == 3);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
}

// ===========================================================================
// 练习 7 — 向量化到底要求什么 ★★☆
//
// 三个候选 layout，哪些能用 128bit atom 沿连续方向向量化？
// 填一个 3 位的 bitmask:
//   bit0 = make_stride(N, 1)              全动态
//   bit1 = make_stride(N, Int<1>{})       只有末维 stride 静态
//   bit2 = make_stride(Int<16>{}, Int<1>{}) 全静态
//
// 先想: CuTe 要在编译期证明"这 4 个元素相邻"。这件事由 shape 决定还是 stride 决定？
// ===========================================================================
constexpr int EX7_MASK = 0b110;  // TODO

// 只有能向量化的 layout 才能实例化这个 kernel。
// 全动态那个如果传进来会编译失败 —— 所以下面只实例化你认为可以的。
template <class TensorS, class TensorD, class TiledCopy>
__global__ void ex7_kernel(TensorS S, TensorD D, TiledCopy tc) {
    auto thr = tc.get_slice(threadIdx.x);
    copy(tc, thr.partition_S(S), thr.partition_D(D));
}

void ex7() {
    printf("\n--- 练习 7: 向量化要求什么 ---\n");
    constexpr int M = 8, N = 16, NE = M * N;

    float *d_s, *d_d;
    CUDA_CHECK(cudaMalloc(&d_s, NE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d, NE * sizeof(float)));
    float hs[NE], hd[NE];
    for (int i = 0; i < NE; ++i) hs[i] = float(i);
    CUDA_CHECK(cudaMemcpy(d_s, hs, sizeof(hs), cudaMemcpyHostToDevice));

    auto tc = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                              make_layout(make_shape(Int<8>{}, Int<4>{}),
                                          make_stride(Int<4>{}, Int<1>{})),
                              make_layout(make_shape(Int<1>{}, Int<4>{})));

    int m_rt = M, n_rt = N;

    // 候选 1: 末维 stride 静态, shape 动态
    auto lay1 = make_layout(make_shape(m_rt, n_rt), make_stride(n_rt, Int<1>{}));
    CUDA_CHECK(cudaMemset(d_d, 0xff, NE * sizeof(float)));
    ex7_kernel<<<1, 32>>>(make_tensor(make_gmem_ptr(d_s), lay1),
                          make_tensor(make_gmem_ptr(d_d), lay1), tc);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hd, d_d, sizeof(hd), cudaMemcpyDeviceToHost));
    bool ok1 = true;
    for (int i = 0; i < NE; ++i)
        if (hd[i] != hs[i]) ok1 = false;

    // 候选 2: 全静态
    auto lay2 = make_layout(make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));
    CUDA_CHECK(cudaMemset(d_d, 0xff, NE * sizeof(float)));
    ex7_kernel<<<1, 32>>>(make_tensor(make_gmem_ptr(d_s), lay2),
                          make_tensor(make_gmem_ptr(d_d), lay2), tc);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hd, d_d, sizeof(hd), cudaMemcpyDeviceToHost));
    bool ok2 = true;
    for (int i = 0; i < NE; ++i)
        if (hd[i] != hs[i]) ok2 = false;

    // 候选 0 (全动态) 编译不过 —— 想验证的话把下面两行取消注释, 看报错:
    // auto lay0 = make_layout(make_shape(m_rt, n_rt), make_stride(n_rt, 1));
    // ex7_kernel<<<1,32>>>(make_tensor(make_gmem_ptr(d_s), lay0), ..., tc);

    printf("  stride(N, Int<1>{})       -> 编译通过, 搬运%s\n", ok1 ? "正确" : "错误");
    printf("  stride(Int<N>{}, Int<1>{}) -> 编译通过, 搬运%s\n", ok2 ? "正确" : "错误");
    printf("  stride(N, 1) 全动态        -> 编译失败 (见代码注释)\n");

    expect("两个 stride 静态的都搬对了", ok1 && ok2);
    expect("EX7_MASK == 0b110", EX7_MASK == 0b110);

    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_d));
}

// ===========================================================================
// 练习 8 — 把 layout 从 kernel 里搬到 host ★★☆
//
// 下面是"kernel 内构造"的老写法, 它把 layout 写死成了 stride (16,1)。
//
// 但 ex8() 里真实的数据是 8x16 嵌在一个**行 stride 为 20** 的缓冲区里
// (每行 16 个有效元素 + 4 个填充)。所以这个 kernel 现在读的是错的位置。
//
// 改成 README §4 的分工:
//   1. kernel 签名改成收 TensorS / TensorD / TiledCopy, 去掉模板参数 <M,N,LD>
//   2. 在 ex8() 里构造 layout / Tensor / TiledCopy, 传进去
//   3. kernel 里只留 get_slice + partition + copy
//
// 这题想让你体会的是: layout 属于**数据的性质**, 应该由知道数据长什么样的那一方
// (host) 来描述。写死在 kernel 里, 换个 stride 就得改 kernel。
// ===========================================================================

// TODO: 把这个 kernel 改成收 Tensor 和 TiledCopy。
//       改完之后它应该和 ex7_kernel 长得几乎一样。
template <class TensorS, class TensorD, class TiledCopy>
__global__ void ex8_kernel(TensorS S, TensorD D, TiledCopy tc, int* d_sz) {
    if(threadIdx.x == 0){
        *d_sz = sizeof(tc);
    }
    auto thr = tc.get_slice(threadIdx.x);
    copy(tc, thr.partition_S(S), thr.partition_D(D));
}

void ex8() {
    printf("\n--- 练习 8: layout 搬到 host ---\n");
    constexpr int M = 8, N = 16, LD = 20;  // 行 stride = 20, 每行 16 个有效 + 4 个填充
    constexpr int NBUF = M * LD;

    float *d_s, *d_d;
    int* d_sz;
    CUDA_CHECK(cudaMalloc(&d_s, NBUF * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d, NBUF * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sz, sizeof(int)));

    // 有效区填 1..128, 填充区填 -7 (搬错了就会把 -7 带过去)
    float hs[NBUF], hd[NBUF];
    for (int i = 0; i < NBUF; ++i) hs[i] = -7.f;
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) hs[r * LD + c] = float(r * N + c + 1);
    CUDA_CHECK(cudaMemcpy(d_s, hs, sizeof(hs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_d, 0, NBUF * sizeof(float)));

    // TODO: 改完 kernel 之后, 在这里构造 layout / Tensor / TiledCopy 再传进去。
    //       正确的 layout 是 make_stride(Int<LD>{}, Int<1>{})。
    auto layS = make_layout(make_shape(Int<M>{}, Int<N>{}), make_stride(Int<LD>{}, Int<1>{}));
    auto layD = layS;
    auto tS = make_tensor(make_gmem_ptr(d_s), layS);
    auto tD = make_tensor(make_gmem_ptr(d_d), layD);
    auto tc = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                              make_layout(make_shape(Int<M>{}, Int<N / 4>{}), make_stride(Int<N / 4>{}, Int<1>{})),
                              make_layout(make_shape(Int<1>{}, Int<4>{})));
    ex8_kernel<<<1, 32>>>(tS, tD, tc, d_sz);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hd, d_d, sizeof(hd), cudaMemcpyDeviceToHost));
    int sz = 0;
    CUDA_CHECK(cudaMemcpy(&sz, d_sz, sizeof(int), cudaMemcpyDeviceToHost));

    // 只检查有效区: 每行前 16 个要搬对
    bool ok = true;
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            if (hd[r * LD + c] != hs[r * LD + c]) ok = false;

    printf("  有效区搬运 = %s, sizeof(TiledCopy) = %d\n", ok ? "正确" : "错误", sz);
    if (!ok) printf("        (layout 的 stride 和真实数据不符 —— 见题目说明)\n");
    expect("8x16 有效区全部搬对 (stride 20 的缓冲区)", ok);
    expect("TiledCopy 是空类型 (sizeof == 1)", sz == 1);

    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_sz));
}

int main() {
    printf("========== cute_03 练习 ==========\n");
    ex1();
    ex2();
    ex3();
    ex4();
    ex5();
    ex6();
    ex7();
    ex8();
    printf("\n===== 结果: %d PASS, %d FAIL =====\n", g_pass, g_fail);
    if (g_fail > 0) printf("还有 TODO 没填 —— 打开 ex.cu 搜 TODO。\n");
    return g_fail == 0 ? 0 : 1;
}
