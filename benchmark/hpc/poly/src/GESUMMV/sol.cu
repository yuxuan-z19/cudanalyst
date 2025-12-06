#include "../../common/polybench.h"
#include "gesummv.cuh"

__global__ void gesummv_kernel(int n, DATA_TYPE alpha, DATA_TYPE beta,
                               DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* tmp,
                               DATA_TYPE* x, DATA_TYPE* y) {
    // TODO:
}