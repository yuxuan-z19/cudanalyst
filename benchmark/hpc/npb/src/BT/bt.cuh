#pragma once

#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

#define IMAX (PROBLEM_SIZE)
#define JMAX (PROBLEM_SIZE)
#define KMAX (PROBLEM_SIZE)
#define IMAXP (IMAX / 2 * 2)
#define JMAXP (JMAX / 2 * 2)
#define AA (0)
#define BB (1)
#define CC (2)
#define M_SIZE (5)
#define PROFILING_TOTAL_TIME (0)
/* new */
#define PROFILING_ADD (1)
#define PROFILING_RHS_1 (2)
#define PROFILING_RHS_2 (3)
#define PROFILING_RHS_3 (4)
#define PROFILING_RHS_4 (5)
#define PROFILING_RHS_5 (6)
#define PROFILING_RHS_6 (7)
#define PROFILING_RHS_7 (8)
#define PROFILING_RHS_8 (9)
#define PROFILING_RHS_9 (10)
#define PROFILING_X_SOLVE_1 (11)
#define PROFILING_X_SOLVE_2 (12)
#define PROFILING_X_SOLVE_3 (13)
#define PROFILING_Y_SOLVE_1 (14)
#define PROFILING_Y_SOLVE_2 (15)
#define PROFILING_Y_SOLVE_3 (16)
#define PROFILING_Z_SOLVE_1 (17)
#define PROFILING_Z_SOLVE_2 (18)
#define PROFILING_Z_SOLVE_3 (19)
/* old */
#define PROFILING_EXACT_RHS_1 (20)
#define PROFILING_EXACT_RHS_2 (21)
#define PROFILING_EXACT_RHS_3 (22)
#define PROFILING_EXACT_RHS_4 (23)
#define PROFILING_ERROR_NORM_1 (24)
#define PROFILING_ERROR_NORM_2 (25)
#define PROFILING_INITIALIZE (26)
#define PROFILING_RHS_NORM_1 (27)
#define PROFILING_RHS_NORM_2 (28)

extern __shared__ double extern_share_data[];

namespace constants_device {
extern __constant__ double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, dx1,
    dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4, dy5, dz1, dz2, dz3, dz4, dz5, dssp,
    dt, dxmax, dymax, dzmax, xxcon1, xxcon2, xxcon3, xxcon4, xxcon5, dx1tx1,
    dx2tx1, dx3tx1, dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4, yycon5,
    dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, zzcon1, zzcon2, zzcon3, zzcon4,
    zzcon5, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, dIMAXm1, dJMAXm1, dKMAXm1,
    c1c2, c1c5, c3c4, c1345, coKMAX1, c1, c2, c3, c4, c5, c4dssp, c5dssp,
    dtdssp, dttx1, dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, c2dtty1, c2dttz1,
    comz1, comz4, comz5, comz6, c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16,
    ce[5][13];
}

void add_gpu();
__global__ void add_gpu_kernel(double* u_device, double* rhs_device);
void adi_gpu();
void compute_rhs_gpu();
__global__ void compute_rhs_gpu_kernel_1(double* rho_i_device,
                                         double* us_device, double* vs_device,
                                         double* ws_device, double* qs_device,
                                         double* square_device,
                                         double* u_device);
__global__ void compute_rhs_gpu_kernel_2(double* rhs_device,
                                         double* forcing_device);
__global__ void compute_rhs_gpu_kernel_3(double* us_device, double* vs_device,
                                         double* ws_device, double* qs_device,
                                         double* rho_i_device,
                                         double* square_device,
                                         double* u_device, double* rhs_device);
__global__ void compute_rhs_gpu_kernel_4(double* u_device, double* rhs_device);
__global__ void compute_rhs_gpu_kernel_5(double* us_device, double* vs_device,
                                         double* ws_device, double* m_q,
                                         double* rho_i_device,
                                         double* square_device,
                                         double* u_device, double* rhs_device);
__global__ void compute_rhs_gpu_kernel_6(double* u_device, double* rhs_device);
__global__ void compute_rhs_gpu_kernel_7(double* us_device, double* vs_device,
                                         double* ws_device, double* qs_device,
                                         double* rho_i_device,
                                         double* square_device,
                                         double* u_device, double* rhs_device);
__global__ void compute_rhs_gpu_kernel_8(double* u_device, double* rhs_device);
__global__ void compute_rhs_gpu_kernel_9(double* rhs_device);
void error_norm(double rms[5]);
void exact_rhs();
void exact_solution(double xi, double eta, double zeta, double dtemp[5]);
void initialize();
void release_gpu();
void rhs_norm(double rms[5]);
size_t round_amount_of_work(size_t amount_of_work, size_t amount_of_threads);
void set_constants();
void setup_gpu();
void verify(int no_time_steps, char* class_npb, boolean* verified);
void x_solve_gpu();
__device__ void x_solve_gpu_device_fjac(double fjac[5][5], double t_u[5],
                                        double rho_i, double qs, double square);
__device__ void x_solve_gpu_device_njac(double njac[5][5], double t_u[5],
                                        double rho_i);
__global__ void x_solve_gpu_kernel_1(double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device);
__global__ void x_solve_gpu_kernel_2(double* qs_device, double* rho_i_device,
                                     double* square_device, double* u_device,
                                     double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device);
__global__ void x_solve_gpu_kernel_3(double* rhs_device, double* lhsA_device,
                                     double* lhsB_device, double* lhsC_device);
void y_solve_gpu();
__device__ void y_solve_gpu_device_fjac(double fjac[5][5], double t_u[5],
                                        double rho_i, double square, double qs);
__device__ void y_solve_gpu_device_njac(double njac[5][5], double t_u[5],
                                        double rho_i);
__global__ void y_solve_gpu_kernel_1(double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device);
__global__ void y_solve_gpu_kernel_2(double* qs_device, double* rho_i_device,
                                     double* square_device, double* u_device,
                                     double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device);
__global__ void y_solve_gpu_kernel_3(double* rhs_device, double* lhsA_device,
                                     double* lhsB_device, double* lhsC_device);
void z_solve_gpu();
__device__ void z_solve_gpu_device_fjac(double l_fjac[5][5], double t_u[5],
                                        double square, double qs);
__device__ void z_solve_gpu_device_njac(double l_njac[5][5], double t_u[5]);
__global__ void z_solve_gpu_kernel_1(double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device);
__global__ void z_solve_gpu_kernel_2(double* qs_device, double* square_device,
                                     double* u_device, double* lhsA_device,
                                     double* lhsB_device, double* lhsC_device);
__global__ void z_solve_gpu_kernel_3(double* rhs_device, double* lhsA_device,
                                     double* lhsB_device, double* lhsC_device);