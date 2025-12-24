#include "bt.cuh"

__global__ void add_gpu_kernel(double* u_device, double* rhs_device) {
    // TODO:
}

/*
 * ---------------------------------------------------------------------
 * compute the reciprocal of density, and the kinetic energy,
 * and the speed of sound.
 * ---------------------------------------------------------------------
 */
__global__ void compute_rhs_gpu_kernel_1(double* rho_i_device,
                                         double* us_device, double* vs_device,
                                         double* ws_device, double* qs_device,
                                         double* square_device,
                                         double* u_device) {
    // TODO:
}

/*
 * ---------------------------------------------------------------------
 * copy the exact forcing term to the right hand side; because
 * this forcing term is known, we can store it on the whole grid
 * including the boundary
 * ---------------------------------------------------------------------
 */
__global__ void compute_rhs_gpu_kernel_2(double* rhs_device,
                                         double* forcing_device) {
    // TODO:
}

/*
 * ---------------------------------------------------------------------
 * compute xi-direction fluxes
 * ---------------------------------------------------------------------
 */
__global__ void compute_rhs_gpu_kernel_3(double* us_device, double* vs_device,
                                         double* ws_device, double* qs_device,
                                         double* rho_i_device,
                                         double* square_device,
                                         double* u_device, double* rhs_device) {
    // TODO:
}

__global__ void compute_rhs_gpu_kernel_4(double* u_device, double* rhs_device) {
    // TODO:
}

/*
 * ---------------------------------------------------------------------
 * compute eta-direction fluxes
 * ---------------------------------------------------------------------
 * Input(write buffer) - us_device, vs_device, ws_device, qs_device,
 * rho_i_device, square_device, u_device, rhs_device
 * ---------------------------------------------------------------------
 * Input(write buffer) - us_device, vs_device, ws_device, qs_device,
 * rho_i_device, square_device, u_device, rhs_device
 * ---------------------------------------------------------------------
 * Output(read buffer) - rhs_device
 * ---------------------------------------------------------------------
 */
__global__ void compute_rhs_gpu_kernel_5(double* us_device, double* vs_device,
                                         double* ws_device, double* qs_device,
                                         double* rho_i_device,
                                         double* square_device,
                                         double* u_device, double* rhs_device) {
    // TODO:
}

__global__ void compute_rhs_gpu_kernel_6(double* u_device, double* rhs_device) {
    // TODO:
}

__global__ void compute_rhs_gpu_kernel_7(double* us_device, double* vs_device,
                                         double* ws_device, double* qs_device,
                                         double* rho_i_device,
                                         double* square_device,
                                         double* u_device, double* rhs_device) {
    // TODO:
}

__global__ void compute_rhs_gpu_kernel_8(double* u_device, double* rhs_device) {
    // TODO:
}

__global__ void compute_rhs_gpu_kernel_9(double* rhs_device) {
    // TODO:
}

/*
 * ---------------------------------------------------------------------
 * this function computes the left hand side in the xi-direction
 * ---------------------------------------------------------------------
 */
__device__ void x_solve_gpu_device_fjac(double fjac[5][5], double t_u[5],
                                        double rho_i, double qs,
                                        double square) {
    // TODO:
}

/*
 * ---------------------------------------------------------------------
 * this function computes the left hand side in the xi-direction
 * ---------------------------------------------------------------------
 */
__global__ void x_solve_gpu_kernel_1(double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device) {
    // TODO:
}

__global__ void x_solve_gpu_kernel_2(double* qs_device, double* rho_i_device,
                                     double* square_device, double* u_device,
                                     double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device) {
    // TODO:
}

/*
 * ---------------------------------------------------------------------
 * this function computes the left hand side in the xi-direction
 * ---------------------------------------------------------------------
 */
__global__ void x_solve_gpu_kernel_3(double* rhs_device, double* lhsA_device,
                                     double* lhsB_device, double* lhsC_device) {
    // TODO:
}

__device__ void y_solve_gpu_device_fjac(double fjac[5][5], double t_u[5],
                                        double rho_i, double square,
                                        double qs) {
    // TODO:
}

__device__ void y_solve_gpu_device_njac(double njac[5][5], double t_u[5],
                                        double rho_i) {
    // TODO:
}

__global__ void y_solve_gpu_kernel_1(double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device) {
    // TODO:
}

__global__ void y_solve_gpu_kernel_2(double* qs_device, double* rho_i_device,
                                     double* square_device, double* u_device,
                                     double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device) {
    // TODO:
}

__global__ void y_solve_gpu_kernel_3(double* rhs_device, double* lhsA_device,
                                     double* lhsB_device, double* lhsC_device) {
    // TODO:
}

/*
 * ---------------------------------------------------------------------
 * this function computes the left hand side for the three z-factors
 * ---------------------------------------------------------------------
 */
__device__ void z_solve_gpu_device_fjac(double l_fjac[5][5], double t_u[5],
                                        double square, double qs) {
    // TODO:
}

__device__ void z_solve_gpu_device_njac(double l_njac[5][5], double t_u[5]) {
    // TODO:
}

/*
 * ---------------------------------------------------------------------
 * this function computes the left hand side for the three z-factors
 * ---------------------------------------------------------------------
 */
__global__ void z_solve_gpu_kernel_1(double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device) {
    // TODO:
}

__global__ void z_solve_gpu_kernel_2(double* qs_device, double* square_device,
                                     double* u_device, double* lhsA_device,
                                     double* lhsB_device, double* lhsC_device) {
    // TODO:
}

/*
 * ---------------------------------------------------------------------
 * this function computes the left hand side for the three z-factors
 * ---------------------------------------------------------------------
 */
__global__ void z_solve_gpu_kernel_3(double* rhs_device, double* lhsA_device,
                                     double* lhsB_device, double* lhsC_device) {
    // TODO:
}
