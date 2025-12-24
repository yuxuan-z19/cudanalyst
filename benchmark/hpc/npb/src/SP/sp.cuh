#pragma once

#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

#define IMAX (PROBLEM_SIZE)
#define JMAX (PROBLEM_SIZE)
#define KMAX (PROBLEM_SIZE)
#define IMAXP (IMAX / 2 * 2)
#define JMAXP (JMAX / 2 * 2)
#define PROFILING_TOTAL_TIME (0)

#define PROFILING_ADD (1)
#define PROFILING_COMPUTE_RHS_1 (2)
#define PROFILING_COMPUTE_RHS_2 (3)
#define PROFILING_ERROR_NORM_1 (4)
#define PROFILING_ERROR_NORM_2 (5)
#define PROFILING_EXACT_RHS_1 (6)
#define PROFILING_EXACT_RHS_2 (7)
#define PROFILING_EXACT_RHS_3 (8)
#define PROFILING_EXACT_RHS_4 (9)
#define PROFILING_INITIALIZE (10)
#define PROFILING_RHS_NORM_1 (11)
#define PROFILING_RHS_NORM_2 (12)
#define PROFILING_TXINVR (13)
#define PROFILING_X_SOLVE (14)
#define PROFILING_Y_SOLVE (15)
#define PROFILING_Z_SOLVE (16)

/* gpu linear pattern */
#define u(m, i, j, k) u[(i) + nx * ((j) + ny * ((k) + nz * (m)))]
#define forcing(m, i, j, k) forcing[(i) + nx * ((j) + ny * ((k) + nz * (m)))]
#define rhs(m, i, j, k) rhs[m + (i) * 5 + (j) * 5 * nx + (k) * 5 * nx * ny]
#define rho_i(i, j, k) rho_i[i + (j) * nx + (k) * nx * ny]
#define us(i, j, k) us[i + (j) * nx + (k) * nx * ny]
#define vs(i, j, k) vs[i + (j) * nx + (k) * nx * ny]
#define ws(i, j, k) ws[i + (j) * nx + (k) * nx * ny]
#define square(i, j, k) square[i + (j) * nx + (k) * nx * ny]
#define qs(i, j, k) qs[i + (j) * nx + (k) * nx * ny]
#define speed(i, j, k) speed[i + (j) * nx + (k) * nx * ny]

extern __shared__ double extern_share_data[];

namespace constants_device {
extern __constant__ double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, dx1,
    dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4, dy5, dz1, dz2, dz3, dz4, dz5, dssp,
    dt, dxmax, dymax, dzmax, xxcon1, xxcon2, xxcon3, xxcon4, xxcon5, dx1tx1,
    dx2tx1, dx3tx1, dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4, yycon5,
    dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, zzcon1, zzcon2, zzcon3, zzcon4,
    zzcon5, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, dnxm1, dnym1, dnzm1, c1c2,
    c1c5, c3c4, c1345, conz1, c1, c2, c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1,
    bt, dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, c2dtty1, c2dttz1, comz1,
    comz4, comz5, comz6, c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16,
    ce[13][5];
}

/* function prototypes */
void add_gpu();
__global__ void add_gpu_kernel(double* u, const double* rhs, const int nx,
                               const int ny, const int nz);
void adi_gpu();
void compute_rhs_gpu();
__global__ void compute_rhs_gpu_kernel_1(double* rho_i, double* us, double* vs,
                                         double* ws, double* speed, double* qs,
                                         double* square, const double* u,
                                         const int nx, const int ny,
                                         const int nz);
__global__ void compute_rhs_gpu_kernel_2(const double* rho_i, const double* us,
                                         const double* vs, const double* ws,
                                         const double* qs, const double* square,
                                         double* rhs, const double* forcing,
                                         const double* u, const int nx,
                                         const int ny, const int nz);
void error_norm_gpu(double rms[]);
__global__ void error_norm_gpu_kernel_1(double* rms, const double* u,
                                        const int nx, const int ny,
                                        const int nz);
__global__ void error_norm_gpu_kernel_2(double* rms, const int nx, const int ny,
                                        const int nz);
void exact_rhs_gpu();
__global__ void exact_rhs_gpu_kernel_1(double* forcing, const int nx,
                                       const int ny, const int nz);
__global__ void exact_rhs_gpu_kernel_2(double* forcing, const int nx,
                                       const int ny, const int nz);
__global__ void exact_rhs_gpu_kernel_3(double* forcing, const int nx,
                                       const int ny, const int nz);
__global__ void exact_rhs_gpu_kernel_4(double* forcing, const int nx,
                                       const int ny, const int nz);
__device__ void exact_solution_gpu_device(const double xi, const double eta,
                                          const double zeta, double* dtemp);
void initialize_gpu();
__global__ void initialize_gpu_kernel(double* u, const int nx, const int ny,
                                      const int nz);
void release_gpu();
void rhs_norm_gpu(double rms[]);
__global__ void rhs_norm_gpu_kernel_1(double* rms, const double* rhs,
                                      const int nx, const int ny, const int nz);
__global__ void rhs_norm_gpu_kernel_2(double* rms, const int nx, const int ny,
                                      const int nz);
void set_constants();
void setup_gpu();
void txinvr_gpu();
__global__ void txinvr_gpu_kernel(const double* rho_i, const double* us,
                                  const double* vs, const double* ws,
                                  const double* speed, const double* qs,
                                  double* rhs, const int nx, const int ny,
                                  const int nz);
void verify_gpu(int no_time_steps, char* class_npb, boolean* verified);
void x_solve_gpu();
__global__ void x_solve_gpu_kernel(const double* rho_i, const double* us,
                                   const double* speed, double* rhs,
                                   double* lhs, double* rhstmp, const int nx,
                                   const int ny, const int nz);
void y_solve_gpu();
__global__ void y_solve_gpu_kernel(const double* rho_i, const double* vs,
                                   const double* speed, double* rhs,
                                   double* lhs, double* rhstmp, const int nx,
                                   const int ny, const int nz);
void z_solve_gpu();
__global__ void z_solve_gpu_kernel(const double* rho_i, const double* us,
                                   const double* vs, const double* ws,
                                   const double* speed, const double* qs,
                                   const double* u, double* rhs, double* lhs,
                                   double* rhstmp, const int nx, const int ny,
                                   const int nz);
