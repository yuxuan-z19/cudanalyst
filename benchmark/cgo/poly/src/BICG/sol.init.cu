#include "../../common/polybench.h"
#include "bicg.cuh"

// Distributed (split) from initial loop and permuted into reverse order to
// allow parallelism...
__global__ void bicg_kernel1(int nx, int ny, DATA_TYPE* A, DATA_TYPE* r,
                             DATA_TYPE* s) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < _PB_NY) {
        s[j] = 0.0f;
        for (int i = 0; i < _PB_NX; i++) s[j] += r[i] * A[i * NY + j];
    }
}

// Distributed (split) from initial loop to allow parallelism
__global__ void bicg_kernel2(int nx, int ny, DATA_TYPE* A, DATA_TYPE* p,
                             DATA_TYPE* q) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < _PB_NX) {
        q[i] = 0.0f;
        for (int j = 0; j < _PB_NY; j++) q[i] += A[i * NY + j] * p[j];
    }
}