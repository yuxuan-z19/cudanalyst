#include "../../common/polybench.h"
#include "fdtd2d.cuh"

__global__ void fdtd_step1_kernel(int nx, int ny, DATA_TYPE* _fict_,
                                  DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz,
                                  int t) {
    // TODO:
}

__global__ void fdtd_step2_kernel(int nx, int ny, DATA_TYPE* ex, DATA_TYPE* ey,
                                  DATA_TYPE* hz, int t) {
    // TODO:
}

__global__ void fdtd_step3_kernel(int nx, int ny, DATA_TYPE* ex, DATA_TYPE* ey,
                                  DATA_TYPE* hz, int t) {
    // TODO:
}