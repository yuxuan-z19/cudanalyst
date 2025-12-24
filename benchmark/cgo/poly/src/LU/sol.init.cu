#include "../../common/polybench.h"
#include "lu.cuh"

__global__ void lu_kernel1(int n, DATA_TYPE* A, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((j > k) && (j < _PB_N)) A[k * N + j] = A[k * N + j] / A[k * N + k];
}

__global__ void lu_kernel2(int n, DATA_TYPE* A, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i > k) && (j > k) && (i < _PB_N) && (j < _PB_N))
        A[i * N + j] = A[i * N + j] - A[i * N + k] * A[k * N + j];
}