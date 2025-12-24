#include "lu.cuh"

__global__ void jacld_blts_gpu_kernel(const int plane, const int klower,
                                      const int jlower, const double* u,
                                      const double* rho_i, const double* qs,
                                      double* v, const int nx, const int ny,
                                      const int nz) {
    // TODO:
}

__global__ void jacu_buts_gpu_kernel(const int plane, const int klower,
                                     const int jlower, const double* u,
                                     const double* rho_i, const double* qs,
                                     double* v, const int nx, const int ny,
                                     const int nz) {
    // TODO:
}

__global__ void l2norm_gpu_kernel(const double* v, double* sum, const int nx,
                                  const int ny, const int nz) {
    // TODO:
}

__global__ void rhs_gpu_kernel_1(const double* u, double* rsd,
                                 const double* frct, double* qs, double* rho_i,
                                 const int nx, const int ny, const int nz) {
    // TODO:
}

__global__ void rhs_gpu_kernel_2(const double* u, double* rsd, const double* qs,
                                 const double* rho_i, const int nx,
                                 const int ny, const int nz) {
    // TODO:
}

__global__ void rhs_gpu_kernel_3(const double* u, double* rsd, const double* qs,
                                 const double* rho_i, const int nx,
                                 const int ny, const int nz) {
    // TODO:
}

__global__ void rhs_gpu_kernel_4(const double* u, double* rsd, const double* qs,
                                 const double* rho_i, const int nx,
                                 const int ny, const int nz) {
    // TODO:
}

__global__ void ssor_gpu_kernel_1(double* rsd, const int nx, const int ny,
                                  const int nz) {
    // TODO:
}

__global__ void ssor_gpu_kernel_2(double* u, double* rsd, const double tmp,
                                  const int nx, const int ny, const int nz) {
    // TODO:
}
