// cute_04 练习：把 TODO 填掉，然后 make run
//
// 题目见 README 的"练习"一节。参考解答在 solutions.md。
// 每题都有一个自动检查，填对了会打印 PASS。

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
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

static int bank_of(int byte_offset) { return (byte_offset / 4) % 32; }

// 一次 warp 访存里最热的 bank 被请求几次
template <class F>
static int max_bank_requests(int lanes, F&& off_of_lane) {
    int hist[32] = {0};
    for (int l = 0; l < lanes; ++l) ++hist[bank_of(off_of_lane(l))];
    int worst = 0;
    for (int b = 0; b < 32; ++b)
        if (hist[b] > worst) worst = hist[b];
    return worst;
}

// ===========================================================================
// 练习 1 — 数 bank ★☆☆
//
// 一个 (32,32):(32,1) 的 float tile。不运行代码先答：
//   一个 warp 读 s(0..31, 5)（第 5 列）时，最热的 bank 被请求几次？
// ===========================================================================
constexpr int EX1_CONFLICT = 0;  // TODO: 改成你的答案

void ex1() {
    printf("\n--- 练习 1: 数 bank ---\n");
    auto lay = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    int actual = max_bank_requests(32, [&](int l) { return int(lay(l, 5)) * 4; });
    printf("  实际 = %d-way\n", actual);
    expect("EX1_CONFLICT 等于实际值", EX1_CONFLICT == actual);
}

// ===========================================================================
// 练习 2 — 为 half 选 padding ★★☆
//
// 上面的 32-way 冲突是 float 的情形。现在换成 half（2 字节）：
// tile 是 (32,64) 的 half，row-major。一个 bank 装 4 字节 = 2 个 half。
//
// 要用 padding 消掉列方向的冲突，行 stride 至少要加多少个 half？
// 提示：先算 plain 情形的冲突有多严重，再想"错开一个 bank"需要几个 half。
// ===========================================================================
constexpr int EX2_PAD_HALVES = 0;  // TODO: 行 stride = 64 + EX2_PAD_HALVES

void ex2() {
    printf("\n--- 练习 2: 为 half 选 padding ---\n");
    auto plain = make_layout(make_shape(Int<32>{}, Int<64>{}), make_stride(Int<64>{}, Int<1>{}));
    int wp = max_bank_requests(32, [&](int l) { return int(plain(l, 0)) * 2; });
    printf("  plain (32,64):(64,1) 列冲突 = %d-way\n", wp);

    int stride = 64 + EX2_PAD_HALVES;
    int worst = max_bank_requests(32, [&](int l) { return (l * stride) * 2; });
    printf("  stride = %d 时列冲突 = %d-way\n", stride, worst);

    expect("padding 后无冲突", worst == 1);
    expect("padding 量最小（再小一点就有冲突）", EX2_PAD_HALVES > 0 && ({
               int less = 64 + EX2_PAD_HALVES - 1;
               max_bank_requests(32, [&](int l) { return (l * less) * 2; }) > 1;
           }));
}

// ===========================================================================
// 练习 3 — Swizzle 的三个不变量 ★★☆
//
// 给 (32,32):(32,1) 套上 Swizzle<5,0,5>。回答三个判断题（true/false）：
//   A: swizzle 之后 cosize 变大了
//   B: swizzle 之后行方向（s(0,0..7)）还是连续的
//   C: swizzle 之后 s(r,c) 到偏移的映射还是双射
// ===========================================================================
constexpr bool EX3_A = true;   // TODO
constexpr bool EX3_B = false;  // TODO
constexpr bool EX3_C = false;  // TODO

void ex3() {
    printf("\n--- 练习 3: Swizzle 的不变量 ---\n");
    auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto swz = composition(Swizzle<5, 0, 5>{}, plain);

    bool a = int(cosize(swz)) > int(cosize(plain));
    bool b = true;
    for (int c = 0; c < 8; ++c)
        if (int(swz(0, c)) != c) b = false;
    static int seen[1024];
    for (int i = 0; i < 1024; ++i) seen[i] = 0;
    bool c_ = true;
    for (int r = 0; r < 32; ++r)
        for (int cc = 0; cc < 32; ++cc) {
            int o = int(swz(r, cc));
            if (o < 0 || o >= 1024 || seen[o]++) c_ = false;
        }

    printf("  实际: cosize 变大 = %s, 行连续 = %s, 双射 = %s\n", a ? "true" : "false",
           b ? "true" : "false", c_ ? "true" : "false");
    expect("EX3_A", EX3_A == a);
    expect("EX3_B", EX3_B == b);
    expect("EX3_C", EX3_C == c_);
}

// ===========================================================================
// 练习 4 — 选对 GMMA swizzle 原子 ★★☆
//
// 你要给 WGMMA 准备一块 (BM=128, BK=32) 的 half smem。
// 四个候选原子的 K 方向长度: SW128->64, SW64->32, SW32->16, INTER->8。
// tile_to_shape 要求 BK 能被原子的 K 长度整除。
//
// 哪些原子可以用？填一个 4 位的 bitmask:
//   bit0 = SW128 可用, bit1 = SW64, bit2 = SW32, bit3 = INTER
// 例如"只有 SW64 和 SW32 可用" -> 0b0110 = 6
// ===========================================================================
constexpr int EX4_MASK = 0;  // TODO

void ex4() {
    printf("\n--- 练习 4: 选 GMMA swizzle 原子 ---\n");
    constexpr int BK = 32;
    int klen[4] = {64, 32, 16, 8};
    const char* nm[4] = {"SW128", "SW64", "SW32", "INTER"};
    int mask = 0;
    for (int i = 0; i < 4; ++i)
        if (BK % klen[i] == 0) mask |= (1 << i);

    printf("  BK = %d:  ", BK);
    for (int i = 0; i < 4; ++i) printf("%s=%s ", nm[i], (BK % klen[i] == 0) ? "可用" : "不可用");
    printf("\n  实际 mask = 0b%d%d%d%d = %d\n", (mask >> 3) & 1, (mask >> 2) & 1, (mask >> 1) & 1,
           mask & 1, mask);

    expect("EX4_MASK 正确", EX4_MASK == mask);

    // 顺手验证一下真的能编译过（用最大的可用原子）
    auto s = tile_to_shape(GMMA::Layout_K_SW64_Atom<half_t>{}, make_shape(Int<128>{}, Int<BK>{}));
    printf("  tile_to_shape(SW64, (128,32)) 成功, cosize = %d\n", int(cosize(s)));
    expect("SW64 铺 (128,32) 的 cosize == 4096", int(cosize(s)) == 4096);
}

// ===========================================================================
// 练习 5 — ldmatrix 的三件套 ★★☆
//
// 不运行代码先答: Copy_Atom<SM75_U32x4_LDSM_N, half_t> 一次操作
// 总共搬多少个 half？
// 提示: 总量 = ThrID * NumValSrc。ldmatrix 是 32 线程协作的。
// ===========================================================================
constexpr int EX5_TOTAL = 0;  // TODO

void ex5() {
    printf("\n--- 练习 5: ldmatrix 搬多少 ---\n");
    using A = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    int thr = int(size(typename A::ThrID{}));
    int total = thr * A::NumValSrc;
    printf("  ThrID = %d, NumValSrc = %d, 总量 = %d 个 half\n", thr, A::NumValSrc, total);
    expect("EX5_TOTAL 正确", EX5_TOTAL == total);
}

// ===========================================================================
// 练习 6 — 修一个 smem layout bug ★★★
//
// 下面的 kernel 想把 gmem 的一块 32x32 搬进 smem 再按列读出来。
// 它结果正确，但 smem 读是 32-way conflict。
// 只改 smem layout（不改任何访存代码），把冲突消掉。
//
// 注意: 数组已经按 33*32 开好了，所以 padding 和 swizzle 都放得下。
// ===========================================================================
__global__ void ex6_kernel(const float* in, float* out, int* conflict_probe) {
    __shared__ __align__(128) float raw[32 * 33];

    // TODO: 这个 layout 导致按列读时 32-way conflict。改掉它。
    //       两条路都可以: padding (stride 33) 或 composition(Swizzle<5,0,5>{}, ...)
    auto slay = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));

    auto s = make_tensor(make_smem_ptr(raw), slay);
    int tx = threadIdx.x % 32, ty = threadIdx.x / 32;
    for (int r = ty; r < 32; r += blockDim.x / 32) s(r, tx) = in[r * 32 + tx];
    __syncthreads();
    for (int r = ty; r < 32; r += blockDim.x / 32) out[r * 32 + tx] = s(tx, r);

    // 探针: 记录 lane l 读 s(l, 0) 时的偏移，host 侧据此算冲突
    if (threadIdx.x < 32) conflict_probe[threadIdx.x] = int(&s(threadIdx.x, 0) - raw);
}

void ex6() {
    printf("\n--- 练习 6: 修 smem layout ---\n");
    float *d_in, *d_out;
    int* d_probe;
    CUDA_CHECK(cudaMalloc(&d_in, 32 * 32 * 4));
    CUDA_CHECK(cudaMalloc(&d_out, 32 * 32 * 4));
    CUDA_CHECK(cudaMalloc(&d_probe, 32 * 4));

    float h_in[32 * 32], h_out[32 * 32];
    for (int i = 0; i < 32 * 32; ++i) h_in[i] = float(i);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));

    ex6_kernel<<<1, 256>>>(d_in, d_out, d_probe);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    int probe[32];
    CUDA_CHECK(cudaMemcpy(probe, d_probe, sizeof(probe), cudaMemcpyDeviceToHost));
    int worst = max_bank_requests(32, [&](int l) { return probe[l] * 4; });

    bool correct = true;
    for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 32; ++c)
            if (h_out[r * 32 + c] != h_in[c * 32 + r]) correct = false;

    printf("  转置结果正确 = %s, 列读冲突 = %d-way\n", correct ? "是" : "否", worst);
    expect("转置结果仍然正确", correct);
    expect("列读无 bank conflict", worst == 1);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_probe));
}

int main() {
    printf("========== cute_04 练习 ==========\n");
    ex1();
    ex2();
    ex3();
    ex4();
    ex5();
    ex6();
    printf("\n===== 结果: %d PASS, %d FAIL =====\n", g_pass, g_fail);
    if (g_fail > 0) printf("还有 TODO 没填 —— 打开 ex.cu 搜 TODO。\n");
    return g_fail == 0 ? 0 : 1;
}
