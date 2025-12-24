#include "../../common/polybench.h"
#include "jacobi1D.cuh"

__global__ void runJacobiCUDA_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i > 0) && (i < (_PB_N - 1)))
        B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
}

__global__ void runJacobiCUDA_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i > 0) && (i < (_PB_N - 1))) A[i] = B[i];
}