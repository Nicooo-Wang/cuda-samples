// v7: atomicAdd 收尾 —— 一个 kernel 干完，去掉第二趟。
//
// v6 已经很接近带宽上限了，还剩的那点开销在结构上：第二趟 kernel 只为了把
// grid = 1056 个部分和加成一个数。这 1056 个 float = 4 KB，相比 64 MB 的输入
// 完全可以忽略，但它仍然要付一次完整的 kernel launch（约 3~5 微秒），
// 而整个 v6 也才 25.6 微秒——光启动开销就占了十分之一。
//
// 这一版让每个 block 算完自己那份之后，直接用 atomicAdd 把结果加到全局的那一个
// float 上。1056 次原子加，硬件在 L2 上做，几乎不花时间。
//
// 实测 0.0256 -> 0.0235 ms（2618 -> 2857 GB/s），达到纯读上限 3135 GB/s 的 91%。
// 剩下那 9% 是 kernel 启动、尾部 wave 不齐、以及 HBM 本身达不到理论峰值。
//
// ⚠️ 代价：求和顺序不再确定
//   1056 个 block 的 atomicAdd 谁先谁后取决于调度，每次跑的加法顺序都可能不同。
//   float 加法不满足结合律（(a+b)+c ≠ a+(b+c)），所以【同一个程序跑两次，
//   结果可能差最后一两位】。前面 v0~v6 都是确定性的，每次跑结果逐位相同。
//
//   实测把同一个二进制连跑 5 次（cpu 参考 = 953.241097）：
//     v7 (本版)          v6 (上一版)
//     953.240906         953.241455
//     953.240845         953.241455
//     953.241577         953.241455
//     953.240417         953.241455
//     953.240479         953.241455
//   v7 每次都不一样，v6 逐位相同。自己跑一遍就能复现。
//
//   这正好和 v0 文件头讲的那件事对上：那里说"不要用全 1.0 输入，否则和恰好是 2^24，
//   float 能精确表示，任何加法顺序都得到同一个精确值"——那种输入下这一版的
//   不确定性会被完全掩盖，你根本看不出它和 v6 有什么区别。
//   用随机输入才能暴露出来。
//
//   什么时候可以接受：绝大多数深度学习场景（梯度累加、loss 求和）都可以，
//   误差量级远小于训练本身的噪声。什么时候不行：需要逐位复现的场景
//   （回归测试、金融计算、调试时对比两次运行）。
//   这种时候要么退回 v6，要么用确定性的 atomic 方案（比如按 blockIdx 顺序串行提交）。
//
// ⚠️ 另一个坑：输出必须先清零
//   atomicAdd 是"加到已有值上"，所以每次 launch 前必须把那个 float 清成 0。
//   这里放在 benchmark 的 lambda 里，和 kernel 一起计时——它是这个方案的固有成本。
//   （cudaMemsetAsync 4 个字节，开销可以忽略，但不能忘。）
#include "common.h"

constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
static_assert(N % 4 == 0, "float4 载入要求 N 是 4 的整数倍");

__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduce_v7(const float4* __restrict__ in, float* __restrict__ out, int n4) {
    __shared__ float s[WARPS_PER_BLOCK];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;

    // 载入部分和 v6 一字不差
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n4; i += gridDim.x * blockDim.x) {
        const float4 v = in[i];
        sum += v.x + v.y + v.z + v.w;
    }

    sum = warp_reduce(sum);
    if (lane == 0) s[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < WARPS_PER_BLOCK) ? s[lane] : 0.0f;
        sum = warp_reduce(sum);
        // 唯一的改动：不写 out[blockIdx.x] 让第二趟去收，直接原子加到那一个数上。
        // 整个 grid 一共只有 gridDim.x = 1056 次原子操作。
        if (lane == 0) atomicAdd(out, sum);
    }
}

int main() {
    const size_t bytes = (size_t)N * sizeof(float);
    float* h_in = (float*)malloc(bytes);
    fill_inputs(h_in, N);
    double cpu_sum = cpu_reference(h_in, N);

    int dev = 0, num_sm = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev));
    const int grid = num_sm * 8;

    float *d_in, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));  // 整个输出就这一个 float
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("v7 float4 + atomicAdd  N = %d, grid = %d (%d SMs x 8)\n", N, grid, num_sm);

    float ms = benchmark([&] {
        // atomicAdd 是累加，每次 launch 前必须清零
        CUDA_CHECK(cudaMemsetAsync(d_sum, 0, sizeof(float)));
        reduce_v7<<<grid, BLOCK_SIZE>>>((const float4*)d_in, d_sum, N / 4);
    });

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    report_perf("time (1 kernel)", ms);
    bool pass = check_result(gpu_sum, cpu_sum, h_in, N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_sum));
    free(h_in);
    return pass ? 0 : 1;
}
