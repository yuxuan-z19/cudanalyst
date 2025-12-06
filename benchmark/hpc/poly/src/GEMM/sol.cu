#include "../../common/polybench.h"
#include "gemm.cuh"

__global__ void gemm_kernel(int ni, int nj, int nk, DATA_TYPE alpha,
                            DATA_TYPE beta, DATA_TYPE* a, DATA_TYPE* b,
                            DATA_TYPE* c) {
    // TODO:
}