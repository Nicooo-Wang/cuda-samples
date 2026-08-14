// v6: cuBLAS，作为整条优化阶梯的上界参考。
//
// 前面 v1~v5 用 wmma 从 24 走到 199 TFLOP/s，这里看看厂商库的水平。
//
// 需要注意计时的口径。cuBLAS 的首次调用要做 heuristic 选择（在几百个预编译 kernel 里
// 挑一个）、加载 kernel 模块、分配 workspace，这些开销比 GEMM 本身大几个数量级：
//
//   完全冷启动的第一次调用     37 ms      0.5 TFLOP/s
//   先用小 shape 热一下再调     2.5 ms      6.7
//   稳定之后                   0.03 ms    500+
//
// 差 700 倍。所以这一版和前面几版不一样，必须预热后再计时，否则量到的全是初始化开销。
// v1~v5 那种"单次冷跑"的口径对自研 kernel 成立（实测抖动 3% 以内），对 cuBLAS 不成立。
//
// 换个角度说：如果你的应用只调一次 GEMM 就退出，cuBLAS 真的比 v1 还慢。
// 库的优势要摊在成千上万次调用上才体现出来。
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "common.h"

int main() {
    const size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N, c_elems = (size_t)M * N;

    half* h_A = (half*)malloc(a_elems * sizeof(half));
    half* h_B = (half*)malloc(b_elems * sizeof(half));
    float* h_C = (float*)malloc(c_elems * sizeof(float));
    float* h_ref = (float*)malloc(c_elems * sizeof(float));
    fill_inputs(h_A, a_elems, h_B, b_elems);

    printf("v6 cuBLAS GemmEx TC  (M=%d, N=%d, K=%d)\n", M, N, K);

    half *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, a_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, b_elems * sizeof(half), cudaMemcpyHostToDevice));

    // 参考结果也是 cuBLAS 算的，所以这一版的校验只是自我一致（必然 PASS）。
    // 留着是为了六个版本的输出格式统一。
    cublas_reference(d_A, d_B, d_C, h_ref);
    CUDA_CHECK(cudaMemset(d_C, 0x7F, c_elems * sizeof(float)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    auto gemm = [&] {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B,
                                  CUDA_R_16F, N, d_A, CUDA_R_16F, K, &beta, d_C, CUDA_R_32F, N,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    // 预热：把 heuristic 选择、模块加载、workspace 分配都赶到计时区间外
    for (int i = 0; i < 5; ++i) gemm();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    gemm();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    printf("  预热后（口径和 v1~v5 不同，见文件头注释）：\n");
    report_perf(ms);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
    const bool pass = check_result(h_C, h_ref);

    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return pass ? 0 : 1;
}
