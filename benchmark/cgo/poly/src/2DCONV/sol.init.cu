#include "../../common/polybench.h"
#include "2DConvolution.cuh"

__global__ void convolution2D_kernel(int ni, int nj, DATA_TYPE* A,
                                     DATA_TYPE* B) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NI - 1) && (j < _PB_NJ - 1) && (i > 0) && (j > 0)) {
        B[i * NJ + j] =
            c11 * A[(i - 1) * NJ + (j - 1)] + c21 * A[(i - 1) * NJ + (j + 0)] +
            c31 * A[(i - 1) * NJ + (j + 1)] + c12 * A[(i + 0) * NJ + (j - 1)] +
            c22 * A[(i + 0) * NJ + (j + 0)] + c32 * A[(i + 0) * NJ + (j + 1)] +
            c13 * A[(i + 1) * NJ + (j - 1)] + c23 * A[(i + 1) * NJ + (j + 0)] +
            c33 * A[(i + 1) * NJ + (j + 1)];
    }
}
