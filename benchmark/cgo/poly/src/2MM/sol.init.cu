#include "../../common/polybench.h"
#include "2mm.cuh"

__global__ void mm2_kernel1(int ni, int nj, int nk, int nl, DATA_TYPE alpha,
                            DATA_TYPE beta, DATA_TYPE* tmp, DATA_TYPE* A,
                            DATA_TYPE* B) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < _PB_NI) && (j < _PB_NJ)) {
        tmp[i * NJ + j] = 0;
        for (int k = 0; k < _PB_NK; k++)
            tmp[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
    }
}

__global__ void mm2_kernel2(int ni, int nj, int nk, int nl, DATA_TYPE alpha,
                            DATA_TYPE beta, DATA_TYPE* tmp, DATA_TYPE* C,
                            DATA_TYPE* D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < _PB_NI) && (j < _PB_NL)) {
        D[i * NL + j] *= beta;
        for (int k = 0; k < _PB_NJ; k++)
            D[i * NL + j] += tmp[i * NJ + k] * C[k * NL + j];
    }
}
