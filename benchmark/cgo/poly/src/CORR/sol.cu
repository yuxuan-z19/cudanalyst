#include "../../common/polybench.h"
#include "correlation.cuh"

__global__ void mean_kernel(int m, int n, DATA_TYPE* mean, DATA_TYPE* data) {
    // TODO:
}

__global__ void std_kernel(int m, int n, DATA_TYPE* mean, DATA_TYPE* std,
                           DATA_TYPE* data) {
    // TODO:
}

__global__ void reduce_kernel(int m, int n, DATA_TYPE* mean, DATA_TYPE* std,
                              DATA_TYPE* data) {
    // TODO:
}

__global__ void corr_kernel(int m, int n, DATA_TYPE* symmat, DATA_TYPE* data) {
    // TODO:
}