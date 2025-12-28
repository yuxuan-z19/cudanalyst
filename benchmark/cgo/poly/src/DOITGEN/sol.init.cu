#include "../../common/polybench.h"
#include "doitgen.cuh"

__global__ void doitgen_kernel1(int nr, int nq, int np, DATA_TYPE* sum,
                                DATA_TYPE* A, DATA_TYPE* C4, int r) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;

    if ((p < np) && (q < nq)) {
        sum[r * (nq * np) + q * np + p] = (DATA_TYPE)0.0;

        for (int s = 0; s < np; s++)
            sum[r * (nq * np) + q * np + p] =
                sum[r * (nq * np) + q * np + p] +
                A[r * (nq * np) + q * np + s] * C4[s * np + p];
    }
}

__global__ void doitgen_kernel2(int nr, int nq, int np, DATA_TYPE* sum,
                                DATA_TYPE* A, DATA_TYPE* C4, int r) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;

    if ((p < np) && (q < nq))
        A[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p];
}
