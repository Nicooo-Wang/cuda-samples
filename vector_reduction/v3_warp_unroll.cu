// v3: 最后 32 个元素用 warp 内展开，去掉多余的 __syncthreads
//
// 当 stride <= 32 时，活跃线程只剩一个 warp。warp 内的线程本来就是锁步执行的
// （同一条指令一起走），不需要 __syncthreads 来同步 —— 那是 block 级的栅栏，
// 对单个 warp 来说纯属浪费。
//
// 关键：必须用 __syncwarp()，不能什么都不加。
// 老教程里常见的写法是把 shared 指针声明成 volatile 然后裸展开，
// 那在 Volta 之前成立；Volta 引入独立线程调度（independent thread scheduling）后，
// warp 内的线程可以真的走散，必须显式 __syncwarp() 才能保证读到别人写的值。
//
// 实测 0.062 -> 0.045 ms（1075 -> 1507 GB/s），1.4 倍。
// 省掉的是最后 5 轮的 block 级 barrier：__syncthreads() 要等整个 block 的
// 8 个 warp 都到齐，而那时只有 1 个 warp 在干活，其余 7 个纯粹在陪等。
#include "common.h"

constexpr int BLOCK_SIZE = 256;
constexpr int ELEMS_PER_BLOCK = 512;

__global__ void reduce_v3(const float* in, float* out, int n) {
    __shared__ float s[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * ELEMS_PER_BLOCK + tid;

    float sum = (idx < n) ? in[idx] : 0.0f;
    if (idx + blockDim.x < n) sum += in[idx + blockDim.x];
    s[tid] = sum;
    __syncthreads();

    // 先归约到只剩 32 个元素，这一段跨 warp，需要 block 级同步
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    // 剩下的 32 个由第 0 个 warp 独自完成，展开成 6 步，只需 warp 级同步
    if (tid < 32) {
        float v = s[tid] + s[tid + 32];
        __syncwarp();
        s[tid] = v;
        __syncwarp();
        v += s[tid + 16]; __syncwarp(); s[tid] = v; __syncwarp();
        v += s[tid + 8];  __syncwarp(); s[tid] = v; __syncwarp();
        v += s[tid + 4];  __syncwarp(); s[tid] = v; __syncwarp();
        v += s[tid + 2];  __syncwarp(); s[tid] = v; __syncwarp();
        v += s[tid + 1];
        if (tid == 0) out[blockIdx.x] = v;
    }
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

    printf("v3 warp unroll  N = %d\n", N);

    float ms = benchmark([&] {
        int cur = N;
        float *src = d_a, *dst = d_b;
        while (cur > 1) {
            const int grid = CEIL(cur, ELEMS_PER_BLOCK);
            reduce_v3<<<grid, BLOCK_SIZE>>>(src, dst, cur);
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
        reduce_v3<<<grid, BLOCK_SIZE>>>(src, dst, cur);
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

