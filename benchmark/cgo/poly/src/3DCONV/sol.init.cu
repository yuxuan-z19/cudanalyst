#include "../../common/polybench.h"
#include "3DConvolution.cuh"

__global__ void convolution3D_kernel(int ni, int nj, int nk, DATA_TYPE* A,
                                     DATA_TYPE* B, int i) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < (_PB_NI - 1)) && (j < (_PB_NJ - 1)) && (k < (_PB_NK - 1)) &&
        (i > 0) && (j > 0) && (k > 0))
        B[i * (NK * NJ) + j * NK + k] =
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] +
            c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] +
            c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] +
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
}