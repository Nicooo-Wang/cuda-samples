// rmsnorm 四个版本共用的样板：错误检查、随机填充、CPU 参考、容差校验。
// 抽出来是为了让每个 .cu 只剩下"这一版 kernel 到底做了什么"这一件事（同 transpose/common.h 的做法）。
//
// RMSNorm 数学（对每一行 x，长度 N）：
//   ms  = mean(x_i^2) = sum(x_i^2) / N
//   rms = sqrt(ms + eps)
//   out_i = (x_i / rms) * w_i            // w 是逐 hidden 维的缩放权重，所有行共享
// 注意：必须先算完整行的 ms 才能除以 rms，所以本质上是"先归约、再归一化"两步。
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

#define CEIL(a, b) (((a) + (b) - 1) / (b))

constexpr int   M_ROWS   = 1 << 16;  // 行数（写死）
constexpr int   N_HIDDEN = 1024;     // 行宽 / hidden dim（写死，2 的幂，一个 block 处理一行）
constexpr float EPS_RMS  = 1e-5f;    // 防 rms=0 的小常数

// 用固定种子填 [-1, 1) 随机数，保证四个版本在同一组输入上跑，结果可横向对比。
inline void fill_random(float* in, size_t count) {
    srand(0);
    for (size_t i = 0; i < count; ++i) in[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// 权重填 [0.5, 1.5)，模拟真实的逐维缩放（默认 1.0 附近）。另起一个种子，和输入独立。
inline void fill_weights(float* w, size_t count) {
    srand(1);
    for (size_t i = 0; i < count; ++i) w[i] = 0.5f + (float)rand() / RAND_MAX;
}

// CPU 参考：逐行算 RMSNorm，sum(x^2) 用 double 累加（参考值要比被测的 float 更准）。
inline void cpu_rmsnorm(const float* x, const float* w, float* out, int M, int N, float eps) {
    for (int r = 0; r < M; ++r) {
        const float* xr = x + (size_t)r * N;
        float* outr = out + (size_t)r * N;
        double sumsq = 0.0;
        for (int c = 0; c < N; ++c) sumsq += (double)xr[c] * xr[c];
        double inv_rms = 1.0 / sqrt(sumsq / N + eps);  // 提前算倒数，后面只用乘法
        for (int c = 0; c < N; ++c) outr[c] = (float)((double)xr[c] * inv_rms * w[c]);
    }
}

// 精度验证：逐元素相对误差。denom 取 max(|ref|, 1e-6) 避免 ref≈0 时除爆。
// RMSNorm 数值良态（不像 softmax 那样要除以很小的 sum），fp32 vs double 通常 < 1e-6。
inline bool validate_rmsnorm(const float* out, const float* ref, int M, int N) {
    double max_rel = 0.0;
    size_t bad_idx = 0;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            size_t i = (size_t)r * N + c;
            double denom = fabs(ref[i]) > 1e-6 ? fabs(ref[i]) : 1e-6;
            double rel = fabs(out[i] - ref[i]) / denom;
            if (rel > max_rel) { max_rel = rel; bad_idx = i; }
        }
    }
    bool pass = max_rel < 1e-5;
    printf("  max rel error: %.3e (at i=%zu)\n", max_rel, bad_idx);
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}
