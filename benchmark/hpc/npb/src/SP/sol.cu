#include "sp.cuh"

__global__ void add_gpu_kernel(double* u, const double* rhs, const int nx,
                               const int ny, const int nz) {
    // TODO:
}

__global__ void compute_rhs_gpu_kernel_1(double* rho_i, double* us, double* vs,
                                         double* ws, double* speed, double* qs,
                                         double* square, const double* u,
                                         const int nx, const int ny,
                                         const int nz) {
    // TODO:
}

__global__ void compute_rhs_gpu_kernel_2(const double* rho_i, const double* us,
                                         const double* vs, const double* ws,
                                         const double* qs, const double* square,
                                         double* rhs, const double* forcing,
                                         const double* u, const int nx,
                                         const int ny, const int nz) {
    // TODO:
}

__global__ void txinvr_gpu_kernel(const double* rho_i, const double* us,
                                  const double* vs, const double* ws,
                                  const double* speed, const double* qs,
                                  double* rhs, const int nx, const int ny,
                                  const int nz) {
    // TODO:
}

__global__ void x_solve_gpu_kernel(const double* rho_i, const double* us,
                                   const double* speed, double* rhs,
                                   double* lhs, double* rhstmp, const int nx,
                                   const int ny, const int nz) {
    // TODO:
}

__global__ void y_solve_gpu_kernel(const double* rho_i, const double* vs,
                                   const double* speed, double* rhs,
                                   double* lhs, double* rhstmp, const int nx,
                                   const int ny, const int nz) {
    // TODO:
}

__global__ void z_solve_gpu_kernel(const double* rho_i, const double* us,
                                   const double* vs, const double* ws,
                                   const double* speed, const double* qs,
                                   const double* u, double* rhs, double* lhs,
                                   double* rhstmp, const int nx, const int ny,
                                   const int nz) {
    // TODO:
}
