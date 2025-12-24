#include "../../common/polybench.h"
#include "syr2k.cuh"

__global__ void syr2k_kernel(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta,
                             DATA_TYPE* a, DATA_TYPE* b, DATA_TYPE* c) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < NI) && (j < NI)) {
        c[i * NI + j] *= beta;
        for (int k = 0; k < NJ; k++)
            c[i * NI + j] += alpha * a[i * NJ + k] * b[j * NJ + k] +
                             alpha * b[i * NJ + k] * a[j * NJ + k];
    }
}