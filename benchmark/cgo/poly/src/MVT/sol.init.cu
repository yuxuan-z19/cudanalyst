#include "../../common/polybench.h"
#include "mvt.cuh"

__global__ void mvt_kernel1(int n, DATA_TYPE* a, DATA_TYPE* x1,
                            DATA_TYPE* y_1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < _PB_N)
        for (int j = 0; j < _PB_N; j++) x1[i] += a[i * N + j] * y_1[j];
}

__global__ void mvt_kernel2(int n, DATA_TYPE* a, DATA_TYPE* x2,
                            DATA_TYPE* y_2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < _PB_N)
        for (int j = 0; j < _PB_N; j++) x2[i] += a[j * N + i] * y_2[j];
}