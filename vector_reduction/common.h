// vector_reduction 各版本共用的样板：数组规模、错误检查、随机填充、CPU 参考、校验、计时。
// 只放这些"每个版本都一样"的东西；BLOCK_SIZE、每线程元素数、grid 形状、kernel 怎么串、
// 显存申请都留在各自的 .cu 里，因为那些正是每一版要讲的内容。
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// 16M 个 float = 64MB。各版本跑同一规模，带宽数字可以直接横向比。
constexpr int N = 1 << 24;

// 用随机输入而不是全 1.0：全 1 时和恰好是 2^24，float 能精确表示，
// 任何加法顺序都得到同一个精确值，归约顺序写错了也照样"通过"——
// 那种判据只能抓漏加元素，抓不到精度问题。
// 随机值下树形归约和串行累加的舍入路径不同，才真正在检查数值精度。
inline void fill_inputs(float* in, int n) {
    srand(0);
    for (int i = 0; i < n; ++i) in[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// CPU 参考：用 double 累加。串行加 16M 个 float，float 累加器自身的误差会大到
// 淹没被测对象，参考值必须比被测值更准。
inline double cpu_reference(const float* in, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += in[i];
    return sum;
}

// 一次归约读一遍输入（N 个 float = 64MB）。
//
// 参照值（这台机器，N=16M）：纯读 64MB 是 0.021ms / 3135 GB/s。
inline void report_perf(const char* tag, float ms) {
    printf("  %-26s %.4f ms   %6.0f GB/s\n", tag, ms, (double)N * sizeof(float) / (ms * 1e-3) / 1e9);
}

// 精度校验：打印结果并返回是否通过。
// 输入零均值，cpu_sum 本身会相消到接近 0，拿它当相对误差的分母会炸。
// 用 sum|x_i| 作尺度：它是这次归约里被舍入的总量级，
// 树形归约的误差上界正比于它（约 eps * log2(N) 倍）。
inline bool check_result(float gpu_sum, double cpu_sum, const float* in, int n) {
    double scale = 0.0;
    for (int i = 0; i < n; ++i) scale += fabs((double)in[i]);
    const double tol = 1e-9 * scale;  // 实测误差 ~4e-4，scale ~8.4e6，留约一个量级余量
    const double diff = fabs((double)gpu_sum - cpu_sum);
    const bool pass = diff <= tol;

    printf("  cpu = %.6f, gpu = %.6f\n", cpu_sum, (double)gpu_sum);
    printf("  abs diff = %.3e (tol %.3e)  ->  %s\n", diff, tol, pass ? "PASS" : "FAIL");
    return pass;
}

// 计时：先热身一次，再取 10 次里最快的。
//
// 为什么要热身：第一次 launch 要付内核加载、页表建立等一次性开销，
// 实测能比稳定态慢一倍以上——只测一次冷启动，各版本之间就没法比了。
// 取最快值而不是平均值：把偶发的调度抖动（别的进程抢 SM）排除掉。
//
// 为什么不用 matmul 的单次冷跑：reduce 单次只有 0.04ms 且要启动多趟 kernel，
// 冷启动开销和调度抖动足以盖掉版本间的差别（v3 和 v4 之间只差 3%）。
// matmul 单次 0.07ms 尚可是因为它只启动一个 kernel。
//
// 代价是 ncu profile 到的是热身之后的某一次，和这里打印的数字不是同一次；
// 但两者稳定态一致，对照看没有问题。
template <typename Launch>
float benchmark(Launch launch) {
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    launch();  // 热身，不计时
    CUDA_CHECK(cudaDeviceSynchronize());

    float best = 1e30f;
    for (int r = 0; r < 10; ++r) {
        CUDA_CHECK(cudaEventRecord(ev0));
        launch();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return best;
}
