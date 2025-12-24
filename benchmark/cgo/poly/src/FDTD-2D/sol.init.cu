#include "../../common/polybench.h"
#include "fdtd2d.cuh"

__global__ void fdtd_step1_kernel(int nx, int ny, DATA_TYPE* _fict_,
                                  DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz,
                                  int t) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < _PB_NX) && (j < _PB_NY)) {
        if (i == 0)
            ey[i * NY + j] = _fict_[t];
        else
            ey[i * NY + j] =
                ey[i * NY + j] - 0.5f * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
    }
}

__global__ void fdtd_step2_kernel(int nx, int ny, DATA_TYPE* ex, DATA_TYPE* ey,
                                  DATA_TYPE* hz, int t) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < _PB_NX) && (j < _PB_NY) && (j > 0))
        ex[i * NY + j] =
            ex[i * NY + j] - 0.5f * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
}

__global__ void fdtd_step3_kernel(int nx, int ny, DATA_TYPE* ex, DATA_TYPE* ey,
                                  DATA_TYPE* hz, int t) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < (_PB_NX - 1)) && (j < (_PB_NY - 1)))
        hz[i * NY + j] =
            hz[i * NY + j] - 0.7f * (ex[i * NY + (j + 1)] - ex[i * NY + j] +
                                     ey[(i + 1) * NY + j] - ey[i * NY + j]);
}