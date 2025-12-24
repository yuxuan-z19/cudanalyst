#include "../../common/polybench.h"
#include "atax.cuh"

__global__ void atax_kernel1(int nx, int ny, DATA_TYPE* A, DATA_TYPE* x,
                             DATA_TYPE* tmp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < _PB_NX) {
        tmp[i] = 0;
        for (int j = 0; j < _PB_NY; j++) tmp[i] += A[i * NY + j] * x[j];
    }
}

__global__ void atax_kernel2(int nx, int ny, DATA_TYPE* A, DATA_TYPE* y,
                             DATA_TYPE* tmp) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < _PB_NY) {
        y[j] = 0;
        for (int i = 0; i < _PB_NX; i++) y[j] += A[i * NY + j] * tmp[i];
    }
}