#include "../../common/polybench.h"
#include "gemm.cuh"

__global__ void gemm_kernel(int ni, int nj, int nk, DATA_TYPE alpha,
                            DATA_TYPE beta, DATA_TYPE* a, DATA_TYPE* b,
                            DATA_TYPE* c) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < _PB_NI) && (j < _PB_NJ)) {
        c[i * NJ + j] *= beta;
        for (int k = 0; k < _PB_NK; k++)
            c[i * NJ + j] += alpha * a[i * NK + k] * b[k * NJ + j];
    }
}