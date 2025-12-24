#include "../../common/polybench.h"
#include "3mm.cuh"

__global__ void mm3_kernel1(int ni, int nj, int nk, int nl, int nm,
                            DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* E) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < _PB_NI) && (j < _PB_NJ)) {
        E[i * NJ + j] = 0;
        for (int k = 0; k < _PB_NK; k++) {
            E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
        }
    }
}

__global__ void mm3_kernel2(int ni, int nj, int nk, int nl, int nm,
                            DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* F) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < _PB_NJ) && (j < _PB_NL)) {
        F[i * NL + j] = 0;
        for (int k = 0; k < _PB_NM; k++) {
            F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
        }
    }
}

__global__ void mm3_kernel3(int ni, int nj, int nk, int nl, int nm,
                            DATA_TYPE* E, DATA_TYPE* F, DATA_TYPE* G) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < _PB_NI) && (j < _PB_NL)) {
        G[i * NL + j] = 0;
        for (int k = 0; k < _PB_NJ; k++) {
            G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
        }
    }
}