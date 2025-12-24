#include "mg.cuh"

__global__ void resid_gpu_kernel(double* u, double* v, double* r, double* a,
                                 int n1, int n2, int n3, int amount_of_work) {
    // TODO:
}

__global__ void comm3_gpu_kernel_1(double* u, int n1, int n2, int n3,
                                   int amount_of_work) {
    // TODO:
}

__global__ void comm3_gpu_kernel_2(double* u, int n1, int n2, int n3,
                                   int amount_of_work) {
    // TODO:
}

__global__ void comm3_gpu_kernel_3(double* u, int n1, int n2, int n3,
                                   int amount_of_work) {
    // TODO:
}

__global__ void norm2u3_gpu_kernel(double* r, const int n1, const int n2,
                                   const int n3, double* res_sum,
                                   double* res_max,
                                   int number_of_blocks_on_x_axis,
                                   int amount_of_work) {
    // TODO:
}

__global__ void rprj3_gpu_kernel(double* r_device, double* s_device, int m1k,
                                 int m2k, int m3k, int m1j, int m2j, int m3j,
                                 int d1, int d2, int d3, int amount_of_work) {
    // TODO:
}

__global__ void psinv_gpu_kernel(double* r, double* u, double* c, int n1,
                                 int n2, int n3, int amount_of_work) {
    // TODO:
}

__global__ void zero3_gpu_kernel(double* z, int n1, int n2, int n3,
                                 int amount_of_work) {
    // TODO:
}

__global__ void interp_gpu_kernel(double* z_device, double* u_device, int mm1,
                                  int mm2, int mm3, int n1, int n2, int n3,
                                  int amount_of_work) {
    // TODO:
}