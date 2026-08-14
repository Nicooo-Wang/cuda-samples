// v1: 消除 warp divergence（contiguous addressing）
//
// v0 的活跃线程是 tid % (2*stride) == 0，散布在整个 block 里；
// 这里改成 tid < stride，活跃线程永远是前面连续的一段。
//
// 收益：同样是每轮减半的工作量，但现在闲置的 warp 是整个闲置，
// 可以直接退出调度，而不是每个 warp 里都有一半线程陪跑。
// 前 128 个线程活跃时正好是前 4 个 warp 满负荷，后 4 个 warp 完全不参与。
//
// 副作用：访问 s[tid] 和 s[tid+stride] 变成连续的，shared memory bank 冲突也更少。
//
// 实测 0.255 -> 0.106 ms（263 -> 631 GB/s），2.4 倍。改的只是一个判断条件。
#include "common.h"

constexpr int BLOCK_SIZE = 256;
constexpr int ELEMS_PER_BLOCK = BLOCK_SIZE;

__global__ void reduce_v1(const float* in, float* out, int n) {
    __shared__ float s[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    s[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    // stride 从大到小，活跃线程始终是 [0, stride) 这一段连续区间
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = s[0];
}

int main() {
    const size_t bytes = (size_t)N * sizeof(float);
    float* h_in = (float*)malloc(bytes);
    fill_inputs(h_in, N);
    double cpu_sum = cpu_reference(h_in, N);

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, (size_t)CEIL(N, ELEMS_PER_BLOCK) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_in, bytes, cudaMemcpyHostToDevice));

    printf("v1 contiguous  N = %d\n", N);

    float ms = benchmark([&] {
        int cur = N;
        float *src = d_a, *dst = d_b;
        while (cur > 1) {
            const int grid = CEIL(cur, ELEMS_PER_BLOCK);
            reduce_v1<<<grid, BLOCK_SIZE>>>(src, dst, cur);
            cur = grid;
            float* t = src; src = dst; dst = t;
        }
    });

    // benchmark 把 d_a/d_b 里的数据弄乱了，重新跑一次完整的归约（不计时）获取正确结果
    CUDA_CHECK(cudaMemcpy(d_a, h_in, bytes, cudaMemcpyHostToDevice));
    int cur = N;
    float *src = d_a, *dst = d_b;
    while (cur > 1) {
        const int grid = CEIL(cur, ELEMS_PER_BLOCK);
        reduce_v1<<<grid, BLOCK_SIZE>>>(src, dst, cur);
        cur = grid;
        float* t = src; src = dst; dst = t;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, src, sizeof(float), cudaMemcpyDeviceToHost));

    report_perf("time", ms);
    bool pass = check_result(gpu_sum, cpu_sum, h_in, N);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_in);
    return pass ? 0 : 1;
}

