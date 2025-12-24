#include "../../common/polybench.h"
#include "gemver.cuh"

__global__ void gemver_kernel1(int n, DATA_TYPE alpha, DATA_TYPE beta,
                               DATA_TYPE* a, DATA_TYPE* v1, DATA_TYPE* v2,
                               DATA_TYPE* u1, DATA_TYPE* u2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < _PB_N) && (j < _PB_N))
        a[i * N + j] += u1[i] * v1[j] + u2[i] * v2[j];
}

__global__ void gemver_kernel2(int n, DATA_TYPE alpha, DATA_TYPE beta,
                               DATA_TYPE* a, DATA_TYPE* x, DATA_TYPE* y,
                               DATA_TYPE* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < _PB_N) {
        for (int j = 0; j < _PB_N; j++) x[i] += beta * a[j * N + i] * y[j];
        x[i] += z[i];
    }
}

__global__ void gemver_kernel3(int n, DATA_TYPE alpha, DATA_TYPE beta,
                               DATA_TYPE* a, DATA_TYPE* x, DATA_TYPE* w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i >= 0) && (i < _PB_N))
        for (int j = 0; j < _PB_N; j++) w[i] += alpha * a[i * N + j] * x[j];
}