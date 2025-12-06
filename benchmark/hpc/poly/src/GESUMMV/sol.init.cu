#include "../../common/polybench.h"
#include "gesummv.cuh"

__global__ void gesummv_kernel(int n, DATA_TYPE alpha, DATA_TYPE beta,
                               DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* tmp,
                               DATA_TYPE* x, DATA_TYPE* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < _PB_N) {
        for (int j = 0; j < _PB_N; j++) {
            tmp[i] += A[i * N + j] * x[j];
            y[i] += B[i * N + j] * x[j];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}