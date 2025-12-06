#include "../../common/polybench.h"
#include "doitgen.cuh"

__global__ void doitgen_kernel1(DATA_TYPE* sum, DATA_TYPE* A, DATA_TYPE* C4,
                                int r) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    if ((p < NP) && (q < NQ)) {
        sum[r * (NQ * NP) + q * NP + p] = (DATA_TYPE)0.0;
        for (int s = 0; s < NP; s++) {
            sum[r * (NQ * NP) + q * NP + p] =
                sum[r * (NQ * NP) + q * NP + p] +
                A[r * (NQ * NP) + q * NP + s] * C4[s * NP + p];
        }
    }
}

__global__ void doitgen_kernel2(DATA_TYPE* sum, DATA_TYPE* A, DATA_TYPE* C4,
                                int r) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    if ((p < NP) && (q < NQ))
        A[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p];
}