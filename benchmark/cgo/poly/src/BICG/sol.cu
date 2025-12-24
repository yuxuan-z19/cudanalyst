#include "../../common/polybench.h"
#include "bicg.cuh"

// Distributed (split) from initial loop and permuted into reverse order to
// allow parallelism...
__global__ void bicg_kernel1(int nx, int ny, DATA_TYPE* A, DATA_TYPE* r,
                             DATA_TYPE* s) {
    // TODO:
}

// Distributed (split) from initial loop to allow parallelism
__global__ void bicg_kernel2(int nx, int ny, DATA_TYPE* A, DATA_TYPE* p,
                             DATA_TYPE* q) {
    // TODO:
}