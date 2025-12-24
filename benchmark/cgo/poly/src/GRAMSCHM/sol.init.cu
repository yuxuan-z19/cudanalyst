#include "../../common/polybench.h"
#include "gramschmidt.cuh"

__global__ void gramschmidt_kernel1(int ni, int nj, DATA_TYPE* a, DATA_TYPE* r,
                                    DATA_TYPE* q, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        DATA_TYPE nrm = 0.0;
        for (int i = 0; i < _PB_NI; i++) nrm += a[i * NJ + k] * a[i * NJ + k];
        r[k * NJ + k] = sqrt(nrm);
    }
}

__global__ void gramschmidt_kernel2(int ni, int nj, DATA_TYPE* a, DATA_TYPE* r,
                                    DATA_TYPE* q, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < _PB_NI) q[i * NJ + k] = a[i * NJ + k] / r[k * NJ + k];
}

__global__ void gramschmidt_kernel3(int ni, int nj, DATA_TYPE* a, DATA_TYPE* r,
                                    DATA_TYPE* q, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((j > k) && (j < _PB_NJ)) {
        r[k * NJ + j] = 0.0;
        for (int i = 0; i < _PB_NI; i++)
            r[k * NJ + j] += q[i * NJ + k] * a[i * NJ + j];
        for (int i = 0; i < _PB_NI; i++)
            a[i * NJ + j] -= q[i * NJ + k] * r[k * NJ + j];
    }
}