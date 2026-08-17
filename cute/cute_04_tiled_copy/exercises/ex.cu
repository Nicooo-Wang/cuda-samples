// cute_04 练习：把 TODO 填掉，然后 make run
//
// 题目见 README 的"练习"一节。参考解答在 solutions.md。
// 每题都有一个自动检查，填对了会打印 PASS。
//
// 六道题对应 README 的:
//   1 -> §1    bank 模型
//   2 -> §3.2  手算 swizzle 映射
//   3 -> §3.4  M 参数与向量化
//   4 -> §5.4  选 GMMA 原子
//   5 -> §5.2 §5.3  TMA 和 WGMMA 谁挑 layout
//   6 -> §3 §4  修一个真实的 layout bug

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

// 全列扫描取最坏 —— 只看第 0 列会得出错误结论 (见 README §3.4)
template <class Lay>
static int worst_col_all(Lay lay) {
    int worst = 0;
    for (int c = 0; c < 32; ++c) {
        int w = max_bank_requests(32, [&](int l) { return int(lay(l, c)) * 4; });
        if (w > worst) worst = w;
    }
    return worst;
}

// 行内最短连续段: 决定能不能用宽向量指令
template <class Lay>
static int min_run(Lay lay) {
    int g = 1 << 30;
    for (int r = 0; r < 32; ++r) {
        int run = 1;
        for (int c = 1; c < 32; ++c) {
            if (int(lay(r, c)) == int(lay(r, c - 1)) + 1) {
                ++run;
            } else {
                if (run < g) g = run;
                run = 1;
            }
        }
        if (run < g) g = run;
    }
    return g;
}

// ===========================================================================
// 练习 1 — 数 bank ★☆☆   (README §1)
//
// 一个 (32,32):(32,1) 的 float tile。不运行代码先答：
//   一个 warp 读 s(0..31, 5)（第 5 列）时，最热的 bank 被请求几次？
// ===========================================================================
constexpr int EX1_CONFLICT = 0;  // TODO: 改成你的答案

void ex1() {
    printf("\n--- 练习 1: 数 bank (§1) ---\n");
    auto lay = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    int actual = max_bank_requests(32, [&](int l) { return int(lay(l, 5)) * 4; });
    printf("  实际 = %d-way\n", actual);
    expect("EX1_CONFLICT 等于实际值", EX1_CONFLICT == actual);
}

// ===========================================================================
// 练习 2 — 手算 swizzle 映射 ★★☆   (README §3.2)
//
// Swizzle<5,0,5> 作用在 (32,32):(32,1) 上。按 README §3.2 的四步手算
// swz_off(6, 3)，不要跑代码。
//
//   第 1 步: plain_off = 6*32 + 3 = ?
//   第 2 步: 取出高 5 位 (就是 r)
//   第 3 步: M=0 不左移, 和低 5 位 (就是 c) 异或
//   第 4 步: 拼回去 = r*32 + (c XOR r)
// ===========================================================================
constexpr int EX2_SWZ_OFF = -1;  // TODO: swz_off(6, 3) = ?

void ex2() {
    printf("\n--- 练习 2: 手算 swizzle 映射 (§3.2) ---\n");
    auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto swz = composition(Swizzle<5, 0, 5>{}, plain);

    int r = 6, c = 3;
    int po = r * 32 + c;
    printf("  plain_off(%d,%d) = %d\n", r, c, po);
    printf("  高 5 位 (r) = %d, 低 5 位 (c) = %d, c XOR r = %d\n", (po >> 5) & 31, po & 31,
           ((po >> 5) & 31) ^ (po & 31));
    printf("  CuTe 算出的 swz(%d,%d) = %d\n", r, c, int(swz(r, c)));
    expect("EX2_SWZ_OFF 等于 CuTe 的结果", EX2_SWZ_OFF == int(swz(r, c)));
}

// ===========================================================================
// 练习 3 — 选 M 保住向量化 ★★★   (README §3.4)
//
// 你要用 128-bit atom（每线程 4 个连续 float）搬一个 32x32 float tile。
// 三个候选 swizzle：Swizzle<5,0,5> / Swizzle<4,1,4> / Swizzle<3,2,3>。
//
// 哪些能用？填一个 3 位 bitmask:
//   bit0 = Sw<5,0,5> 能用, bit1 = Sw<4,1,4>, bit2 = Sw<3,2,3>
//
// 判据: 行内最短连续段 >= 4 个 float 才能凑出一条 128-bit 指令。
// 提示: M 保护最低 M 位不参与异或 -> 2^M 个元素保持连续。
// ===========================================================================
constexpr int EX3_VEC_MASK = 0;  // TODO

void ex3() {
    printf("\n--- 练习 3: 选 M 保住向量化 (§3.4) ---\n");
    auto plain = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto s505 = composition(Swizzle<5, 0, 5>{}, plain);
    auto s414 = composition(Swizzle<4, 1, 4>{}, plain);
    auto s323 = composition(Swizzle<3, 2, 3>{}, plain);

    int runs[3] = {min_run(s505), min_run(s414), min_run(s323)};
    int confs[3] = {worst_col_all(s505), worst_col_all(s414), worst_col_all(s323)};
    const char* nm[3] = {"Sw<5,0,5>", "Sw<4,1,4>", "Sw<3,2,3>"};

    int mask = 0;
    for (int i = 0; i < 3; ++i) {
        if (runs[i] >= 4) mask |= (1 << i);
        printf("  %-10s 行内最短连续 = %2d 个 float   列读 = %2d-way   128-bit %s\n", nm[i],
               runs[i], confs[i], runs[i] >= 4 ? "可用" : "不可用");
    }
    printf("  实际 mask = %d\n", mask);
    expect("EX3_VEC_MASK 正确", EX3_VEC_MASK == mask);
    printf("  注意: 冲突最少的那个 (1-way) 恰恰是不能向量化的那个 —— 这就是 M 的权衡。\n");
}

// ===========================================================================
// 练习 4 — 选对 GMMA swizzle 原子 ★★☆   (README §5.4)
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
    printf("\n--- 练习 4: 选 GMMA swizzle 原子 (§5.4) ---\n");
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

    // 顺手验证真的能编译过（用最大的可用原子）
    auto s = tile_to_shape(GMMA::Layout_K_SW64_Atom<half_t>{}, make_shape(Int<128>{}, Int<BK>{}));
    printf("  tile_to_shape(SW64, (128,32)) 成功, cosize = %d\n", int(cosize(s)));
    expect("SW64 铺 (128,32) 的 cosize == 4096", int(cosize(s)) == 4096);
}

// ===========================================================================
// 练习 5 — 谁挑 layout: TMA 还是 WGMMA ★★☆   (README §5.2 §5.3)
//
// 一块 plain row-major 的 half smem (完全没有 swizzle)。两个判断题:
//   A: TMA 能把数据正确搬进这块 smem 吗？
//   B: 这块 smem 能直接喂给 SM90 的 WGMMA (编译通过) 吗？
//
// 提示: 跑一下 ../cute_tiled_v2 看 §5.2 和 §5.3 的输出。
// 想亲眼看到 B 的报错: 把下面 EX5_TRY_PLAIN_WGMMA 改成 1 再编译。
// ===========================================================================
constexpr bool EX5_TMA_OK = false;    // TODO: TMA 能搬对吗
constexpr bool EX5_WGMMA_OK = true;   // TODO: WGMMA 能编译过吗

#define EX5_TRY_PLAIN_WGMMA 0  // 改成 1 会编译失败, 报 "Not a canonical GMMA_K Layout"

#if EX5_TRY_PLAIN_WGMMA
__global__ void ex5_bad_kernel() {
    __shared__ __align__(128) half_t raw[64 * 64];
    auto plain = make_layout(make_shape(Int<64>{}, Int<64>{}), make_stride(Int<64>{}, Int<1>{}));
    auto sA = make_tensor(make_smem_ptr(raw), plain);
    auto mma = make_tiled_mma(SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
    auto thr = mma.get_thread_slice(threadIdx.x);
    auto tCrA = thr.make_fragment_A(thr.partition_A(sA));  // <- 这里 static_assert 失败
    if (thread0()) print(tCrA);
}
#endif

void ex5() {
    printf("\n--- 练习 5: 谁挑 layout, TMA 还是 WGMMA (§5.2 §5.3) ---\n");
    printf("  这两个答案在 v2 的输出里: TMA 五种 layout 全搬对; WGMMA 拒绝 plain/padded。\n");
    expect("EX5_TMA_OK: TMA 不挑 layout, plain 也搬得对", EX5_TMA_OK == true);
    expect("EX5_WGMMA_OK: WGMMA 编译期拒绝 plain", EX5_WGMMA_OK == false);
}

// ===========================================================================
// 练习 6 — 修一个 smem layout bug ★★★   (README §3 §4)
//
// 下面的 kernel 把 gmem 的一块 32x32 搬进 smem 再按列读出来。
// 它结果正确，但按列读 smem 是 32-way conflict。
//
// 只改 slay 一行（不改任何访存代码），要同时满足两个条件:
//   1) 列读冲突 <= 4-way
//   2) 行内最短连续段 >= 4 个 float（这样 128-bit atom 还能用）
//
// 注意第 2 条排除了 Swizzle<5,0,5> —— 它冲突最少但连续性全丢了。
// 数组已按 33*32 开好，padding 和 swizzle 都放得下。
// ===========================================================================
__global__ void ex6_kernel(const float* in, float* out, int* off_probe) {
    __shared__ __align__(128) float raw[32 * 33];

    // TODO: 这个 layout 列读 32-way conflict。改掉它, 满足上面两个条件。
    //       提示: composition(Swizzle<B,M,S>{}, plain), 按 §3.4 选 M。
    auto slay = make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));

    auto s = make_tensor(make_smem_ptr(raw), slay);
    int tx = threadIdx.x % 32, ty = threadIdx.x / 32;
    for (int r = ty; r < 32; r += blockDim.x / 32) s(r, tx) = in[r * 32 + tx];
    __syncthreads();
    for (int r = ty; r < 32; r += blockDim.x / 32) out[r * 32 + tx] = s(tx, r);

    // 探针: 把整张 32x32 的偏移表交给 host, host 侧算冲突和连续段
    for (int i = threadIdx.x; i < 32 * 32; i += blockDim.x)
        off_probe[i] = int(&s(i / 32, i % 32) - raw);
}

void ex6() {
    printf("\n--- 练习 6: 修 smem layout (§3 §4) ---\n");
    float *d_in, *d_out;
    int* d_probe;
    CUDA_CHECK(cudaMalloc(&d_in, 32 * 32 * 4));
    CUDA_CHECK(cudaMalloc(&d_out, 32 * 32 * 4));
    CUDA_CHECK(cudaMalloc(&d_probe, 32 * 32 * 4));

    float h_in[32 * 32], h_out[32 * 32];
    for (int i = 0; i < 32 * 32; ++i) h_in[i] = float(i);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));

    ex6_kernel<<<1, 256>>>(d_in, d_out, d_probe);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    int probe[32 * 32];
    CUDA_CHECK(cudaMemcpy(probe, d_probe, sizeof(probe), cudaMemcpyDeviceToHost));
    auto off = [&](int r, int c) { return probe[r * 32 + c]; };

    // 全列扫描取最坏
    int worst = 0;
    for (int c = 0; c < 32; ++c) {
        int w = max_bank_requests(32, [&](int l) { return off(l, c) * 4; });
        if (w > worst) worst = w;
    }
    // 行内最短连续段
    int run_min = 1 << 30;
    for (int r = 0; r < 32; ++r) {
        int run = 1;
        for (int c = 1; c < 32; ++c) {
            if (off(r, c) == off(r, c - 1) + 1) {
                ++run;
            } else {
                if (run < run_min) run_min = run;
                run = 1;
            }
        }
        if (run < run_min) run_min = run;
    }

    bool correct = true;
    for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 32; ++c)
            if (h_out[r * 32 + c] != h_in[c * 32 + r]) correct = false;

    printf("  转置结果 = %s, 列读最坏 = %d-way, 行内最短连续 = %d 个 float\n",
           correct ? "正确" : "错误", worst, run_min);
    expect("转置结果仍然正确", correct);
    expect("列读冲突 <= 4-way", worst <= 4);
    expect("行内最短连续 >= 4 个 float (128-bit atom 可用)", run_min >= 4);

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
