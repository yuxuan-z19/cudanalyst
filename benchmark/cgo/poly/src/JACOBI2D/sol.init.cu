#include "../../common/polybench.h"
#include "jacobi2D.cuh"

__global__ void runJacobiCUDA_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i >= 1) && (i < (_PB_N - 1)) && (j >= 1) && (j < (_PB_N - 1)))
        B[i * N + j] =
            0.2f * (A[i * N + j] + A[i * N + (j - 1)] + A[i * N + (1 + j)] +
                    A[(1 + i) * N + j] + A[(i - 1) * N + j]);
}

__global__ void runJacobiCUDA_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i >= 1) && (i < (_PB_N - 1)) && (j >= 1) && (j < (_PB_N - 1)))
        A[i * N + j] = B[i * N + j];
}