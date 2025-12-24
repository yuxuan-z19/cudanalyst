#pragma once

#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

/*
 * ---------------------------------------------------------------------
 * driver for the performance evaluation of the solver for
 * five coupled parabolic/elliptic partial differential equations
 * ---------------------------------------------------------------------
 * parameters which can be overridden in runtime config file
 * isiz1,isiz2,isiz3 give the maximum size
 * ipr = 1 to print out verbose information
 * omega = 2.0 is correct for all classes
 * tolrsd is tolerance levels for steady state residuals
 * ---------------------------------------------------------------------
 * field variables and residuals
 * to improve cache performance, second two dimensions padded by 1
 * for even number sizes only.
 * note: corresponding array (called "v") in routines blts, buts,
 * and l2norm are similarly padded
 * ---------------------------------------------------------------------
 */
#define IPR_DEFAULT (1)
#define OMEGA_DEFAULT (1.2)
#define TOLRSD1_DEF (1.0e-08)
#define TOLRSD2_DEF (1.0e-08)
#define TOLRSD3_DEF (1.0e-08)
#define TOLRSD4_DEF (1.0e-08)
#define TOLRSD5_DEF (1.0e-08)
#define C1 (1.40e+00)
#define C2 (0.40e+00)
#define C3 (1.00e-01)
#define C4 (1.00e+00)
#define C5 (1.40e+00)
#define PROFILING_TOTAL_TIME (0)

#define PROFILING_ERHS_1 (1)
#define PROFILING_ERHS_2 (2)
#define PROFILING_ERHS_3 (3)
#define PROFILING_ERHS_4 (4)
#define PROFILING_ERROR (5)
#define PROFILING_NORM (6)
#define PROFILING_JACLD_BLTS (7)
#define PROFILING_JACU_BUTS (8)
#define PROFILING_L2NORM (9)
#define PROFILING_PINTGR_1 (10)
#define PROFILING_PINTGR_2 (11)
#define PROFILING_PINTGR_3 (12)
#define PROFILING_PINTGR_4 (13)
#define PROFILING_RHS_1 (14)
#define PROFILING_RHS_2 (15)
#define PROFILING_RHS_3 (16)
#define PROFILING_RHS_4 (17)
#define PROFILING_SETBV_1 (18)
#define PROFILING_SETBV_2 (19)
#define PROFILING_SETBV_3 (20)
#define PROFILING_SETIV (21)
#define PROFILING_SSOR_1 (22)
#define PROFILING_SSOR_2 (23)

/* gpu linear pattern */
#define u(m, i, j, k) u[(m) + 5 * ((i) + nx * ((j) + ny * (k)))]
#define v(m, i, j, k) v[(m) + 5 * ((i) + nx * ((j) + ny * (k)))]
#define rsd(m, i, j, k) rsd[(m) + 5 * ((i) + nx * ((j) + ny * (k)))]
#define frct(m, i, j, k) frct[(m) + 5 * ((i) + nx * ((j) + ny * (k)))]
#define rho_i(i, j, k) rho_i[(i) + nx * ((j) + ny * (k))]
#define qs(i, j, k) qs[(i) + nx * ((j) + ny * (k))]

extern __shared__ double extern_share_data[];

namespace constants_device {
/* coefficients of the exact solution */
extern __constant__ double ce[13][5];
/* grid */
extern __constant__ double dxi, deta, dzeta;
extern __constant__ double tx1, tx2, tx3;
extern __constant__ double ty1, ty2, ty3;
extern __constant__ double tz1, tz2, tz3;
/* dissipation */
extern __constant__ double dx1, dx2, dx3, dx4, dx5;
extern __constant__ double dy1, dy2, dy3, dy4, dy5;
extern __constant__ double dz1, dz2, dz3, dz4, dz5;
extern __constant__ double dssp;
/* newton-raphson iteration control parameters */
extern __constant__ double dt, omega;
}  // namespace constants_device

/* function prototypes */
void erhs_gpu();
__global__ void erhs_gpu_kernel_1(double* frct, double* rsd, const int nx,
                                  const int ny, const int nz);
__global__ void erhs_gpu_kernel_2(double* frct, const double* rsd, const int nx,
                                  const int ny, const int nz);
__global__ void erhs_gpu_kernel_3(double* frct, const double* rsd, const int nx,
                                  const int ny, const int nz);
__global__ void erhs_gpu_kernel_4(double* frct, const double* rsd, const int nx,
                                  const int ny, const int nz);
void error_gpu();
__global__ void error_gpu_kernel(const double* u, double* errnm, const int nx,
                                 const int ny, const int nz);
__device__ void exact_gpu_device(const int i, const int j, const int k,
                                 double* u000ijk, const int nx, const int ny,
                                 const int nz);
__global__ void jacld_blts_gpu_kernel(const int plane, const int klower,
                                      const int jlower, const double* u,
                                      const double* rho_i, const double* qs,
                                      double* v, const int nx, const int ny,
                                      const int nz);
__global__ void jacu_buts_gpu_kernel(const int plane, const int klower,
                                     const int jlower, const double* u,
                                     const double* rho_i, const double* qs,
                                     double* v, const int nx, const int ny,
                                     const int nz);
void l2norm_gpu(const double* v, double* sum);
__global__ void l2norm_gpu_kernel(const double* v, double* sum, const int nx,
                                  const int ny, const int nz);
__global__ void norm_gpu_kernel(double* rms, const int size);
void pintgr_gpu();
__global__ void pintgr_gpu_kernel_1(const double* u, double* frc, const int nx,
                                    const int ny, const int nz);
__global__ void pintgr_gpu_kernel_2(const double* u, double* frc, const int nx,
                                    const int ny, const int nz);
__global__ void pintgr_gpu_kernel_3(const double* u, double* frc, const int nx,
                                    const int ny, const int nz);
__global__ void pintgr_gpu_kernel_4(double* frc, const int num);
void read_input();
void release_gpu();
void rhs_gpu();
__global__ void rhs_gpu_kernel_1(const double* u, double* rsd,
                                 const double* frct, double* qs, double* rho_i,
                                 const int nx, const int ny, const int nz);
__global__ void rhs_gpu_kernel_2(const double* u, double* rsd, const double* qs,
                                 const double* rho_i, const int nx,
                                 const int ny, const int nz);
__global__ void rhs_gpu_kernel_3(const double* u, double* rsd, const double* qs,
                                 const double* rho_i, const int nx,
                                 const int ny, const int nz);
__global__ void rhs_gpu_kernel_4(const double* u, double* rsd, const double* qs,
                                 const double* rho_i, const int nx,
                                 const int ny, const int nz);
void setbv_gpu();
__global__ void setbv_gpu_kernel_1(double* u, const int nx, const int ny,
                                   const int nz);
__global__ void setbv_gpu_kernel_2(double* u, const int nx, const int ny,
                                   const int nz);
__global__ void setbv_gpu_kernel_3(double* u, const int nx, const int ny,
                                   const int nz);
void setcoeff_gpu();
void setiv_gpu();
__global__ void setiv_gpu_kernel(double* u, const int nx, const int ny,
                                 const int nz);
void setup_gpu();
void ssor_gpu(int niter);
__global__ void ssor_gpu_kernel_1(double* rsd, const int nx, const int ny,
                                  const int nz);
__global__ void ssor_gpu_kernel_2(double* u, double* rsd, const double tmp,
                                  const int nx, const int ny, const int nz);
void verify_gpu(double xcr[], double xce[], double xci, char* class_npb,
                boolean* verified);
