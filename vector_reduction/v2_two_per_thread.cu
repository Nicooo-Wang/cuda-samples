// v2: 每个线程先加两个元素（more work per thread）
//
// v1 里每个 block 只处理 BLOCK_SIZE 个元素，而 kernel 一启动就有一半线程
// （stride = 128 那轮之后）逐渐闲下来。既然如此，不如在载入阶段就让每个线程
// 多干一份活：一个 block 覆盖 2 * BLOCK_SIZE 个元素，读的时候顺手加一次。
//
// 收益：block 数量减半 -> kernel launch 次数和总的 __syncthreads 次数都减半，
// 而这一次额外的加法几乎免费（本来就要从 global 读数据，带宽是瓶颈）。
//
// 实测 0.106 -> 0.062 ms（631 -> 1075 GB/s），1.7 倍。
// 这个思路可以继续推：每线程 4 个、8 个……但收益会递减，因为省下的是
// launch 和同步开销，而它们在总时间里的占比越来越小。v5 会把它推到极致
// （每线程吃掉 N/grid 个元素），那时结构本身也跟着变了。
#include "common.h"

constexpr int BLOCK_SIZE = 256;
constexpr int ELEMS_PER_BLOCK = 512;

__global__ void reduce_v2(const float* in, float* out, int n) {
    __shared__ float s[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * ELEMS_PER_BLOCK + tid;   // 注意步长是 2*blockDim.x

    // 载入时就把相隔 blockDim.x 的两个元素加起来。
    // 相隔 blockDim.x 而不是相邻，是为了让同一个 warp 的 32 个线程
    // 访问连续的 32 个 float，保持合并访存。
    float sum = (idx < n) ? in[idx] : 0.0f;
    if (idx + blockDim.x < n) sum += in[idx + blockDim.x];
    s[tid] = sum;
    __syncthreads();

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

    printf("v2 two per thread  N = %d\n", N);

    float ms = benchmark([&] {
        int cur = N;
        float *src = d_a, *dst = d_b;
        while (cur > 1) {
            const int grid = CEIL(cur, ELEMS_PER_BLOCK);
            reduce_v2<<<grid, BLOCK_SIZE>>>(src, dst, cur);
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
        reduce_v2<<<grid, BLOCK_SIZE>>>(src, dst, cur);
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

