#include "ft.cuh"

/*
 * ----------------------------------------------------------------------
 * y0[z][x][y] = x_in[z][y][x]
 *
 * y0[y + x*NY + z*NX*NY] = x_in[x + y*NX + z*NX*NY]
 * ----------------------------------------------------------------------
 */
__global__ void cffts1_gpu_kernel_1(dcomplex x_in[], dcomplex y0[]) {
    // TODO:
}

/*
 * ----------------------------------------------------------------------
 * pattern = j + variable*NY + k*NX*NY | variable is i and transforms x axis
 * ----------------------------------------------------------------------
 */
__global__ void cffts1_gpu_kernel_2(const int is, dcomplex gty1[],
                                    dcomplex gty2[], dcomplex u_device[]) {
    // TODO:
}

/*
 * ----------------------------------------------------------------------
 * x_out[z][y][x] = y0[z][x][y]
 *
 * x_out[x + y*NX + z*NX*NY] = y0[y + x*NY + z*NX*NY]
 * ----------------------------------------------------------------------
 */
__global__ void cffts1_gpu_kernel_3(dcomplex x_out[], dcomplex y0[]) {
    // TODO:
}

/*
 * ----------------------------------------------------------------------
 * y0[z][y][x] = x_in[z][y][x]
 *
 * y0[x + y*NX + z*NX*NY]  = x_in[x + y*NX + z*NX*NY]
 * ----------------------------------------------------------------------
 */
__global__ void cffts2_gpu_kernel_1(dcomplex x_in[], dcomplex y0[]) {
    // TODO:
}

/*
 * ----------------------------------------------------------------------
 * pattern = i + variable*NX + k*NX*NY | variable is j and transforms y axis
 * ----------------------------------------------------------------------
 */
__global__ void cffts2_gpu_kernel_2(const int is, dcomplex gty1[],
                                    dcomplex gty2[], dcomplex u_device[]) {
    // TODO:
}

/*
 * ----------------------------------------------------------------------
 * x_out[z][y][x] = y0[z][y][x]
 *
 * x_out[x + y*NX + z*NX*NY] = y0[x + y*NX + z*NX*NY]
 * ----------------------------------------------------------------------
 */
__global__ void cffts2_gpu_kernel_3(dcomplex x_out[], dcomplex y0[]) {
    // TODO:
}

/*
 * ----------------------------------------------------------------------
 * pattern = i + j*NX + variable*NX*NY | variable is z and transforms z axis
 *
 * index_arg = i + j*NX
 *
 * size_arg = NX*NY
 * ----------------------------------------------------------------------
 */
__device__ void cffts3_gpu_cfftz_device(const int is, int m, int n,
                                        dcomplex x[], dcomplex y[],
                                        dcomplex u_device[], int index_arg,
                                        int size_arg) {
    // TODO:
}

/*
 * ----------------------------------------------------------------------
 * pattern = i + j*NX + variable*NX*NY | variable is z and transforms z axis
 *
 * index_arg = i + j*NX
 *
 * size_arg = NX*NY
 * ----------------------------------------------------------------------
 */
__device__ void cffts3_gpu_fftz2_device(const int is, int l, int m, int n,
                                        dcomplex u[], dcomplex x[],
                                        dcomplex y[], int index_arg,
                                        int size_arg) {
    // TODO:
}

/*
 * ----------------------------------------------------------------------
 * y0[z][y][x] = x_in[z][y][x]
 *
 * y0[x + y*NX + z*NX*NY]  = x_in[x + y*NX + z*NX*NY]
 * ----------------------------------------------------------------------
 */
__global__ void cffts3_gpu_kernel_1(dcomplex x_in[], dcomplex y0[]) {
    // TODO:
}

/*
 * ----------------------------------------------------------------------
 * pattern = i + j*NX + variable*NX*NY | variable is z and transforms z axis
 * ----------------------------------------------------------------------
 */
__global__ void cffts3_gpu_kernel_2(const int is, dcomplex gty1[],
                                    dcomplex gty2[], dcomplex u_device[]) {
    // TODO:
}

/*
 * ----------------------------------------------------------------------
 * x_out[z][y][x] = y0[z][y][x]
 *
 * x_out[x + y*NX + z*NX*NY] = y0[x + y*NX + z*NX*NY]
 * ----------------------------------------------------------------------
 */
__global__ void cffts3_gpu_kernel_3(dcomplex x_out[], dcomplex y0[]) {
    // TODO:
}

__global__ void evolve_gpu_kernel(dcomplex u0[], dcomplex u1[],
                                  double twiddle[]) {
    // TODO:
}

__global__ void checksum_gpu_kernel(int iteration, dcomplex u1[],
                                    dcomplex sums[]) {
    // TODO:
}
