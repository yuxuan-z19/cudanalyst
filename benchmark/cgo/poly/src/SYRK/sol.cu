#include "../../common/polybench.h"
#include "syrk.cuh"

__global__ void syrk_kernel(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta,
                            DATA_TYPE* a, DATA_TYPE* c) {
    // TODO:
}