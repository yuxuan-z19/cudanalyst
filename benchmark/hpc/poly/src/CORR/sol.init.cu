#include "../../common/polybench.h"
#include "correlation.cuh"

__global__ void mean_kernel(int m, int n, DATA_TYPE* mean, DATA_TYPE* data) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < _PB_M) {
        mean[j] = 0.0;
        for (int i = 0; i < _PB_N; i++) mean[j] += data[i * M + j];
        mean[j] /= (DATA_TYPE)FLOAT_N;
    }
}

__global__ void std_kernel(int m, int n, DATA_TYPE* mean, DATA_TYPE* std,
                           DATA_TYPE* data) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < _PB_M) {
        std[j] = 0.0;
        for (int i = 0; i < _PB_N; i++)
            std[j] += (data[i * M + j] - mean[j]) * (data[i * M + j] - mean[j]);
        std[j] /= (FLOAT_N);
        std[j] = sqrt(std[j]);
        if (std[j] <= EPS) std[j] = 1.0;
    }
}

__global__ void reduce_kernel(int m, int n, DATA_TYPE* mean, DATA_TYPE* std,
                              DATA_TYPE* data) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < _PB_N) && (j < _PB_M)) {
        data[i * M + j] -= mean[j];
        data[i * M + j] /= (sqrt(FLOAT_N) * std[j]);
    }
}

__global__ void corr_kernel(int m, int n, DATA_TYPE* symmat, DATA_TYPE* data) {
    int j1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (j1 < (_PB_M - 1)) {
        symmat[j1 * M + j1] = 1.0;
        for (int j2 = (j1 + 1); j2 < _PB_M; j2++) {
            symmat[j1 * M + j2] = 0.0;
            for (int i = 0; i < _PB_N; i++)
                symmat[j1 * M + j2] += data[i * M + j1] * data[i * M + j2];
            symmat[j2 * M + j1] = symmat[j1 * M + j2];
        }
    }
}