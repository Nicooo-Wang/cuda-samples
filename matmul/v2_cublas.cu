// v2: cuBLAS 参考，作为性能上界。自己写的版本和它的差距就是剩余优化空间。
#include "matmul_common.cuh"

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const int rc = run_matmul("v2 cuBLAS GemmEx TC",
                              [handle](const half* A, const half* B, float* C, int m, int n, int k) {
                                  cublas_reference(handle, A, B, C, m, n, k);
                              });
    CUBLAS_CHECK(cublasDestroy(handle));
    return rc;
}
