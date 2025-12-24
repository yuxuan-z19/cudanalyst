#pragma once

#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

/*
 * ---------------------------------------------------------------------
 * note: please observe that in the routine conj_grad three
 * implementations of the sparse matrix-vector multiply have
 * been supplied. the default matrix-vector multiply is not
 * loop unrolled. the alternate implementations are unrolled
 * to a depth of 2 and unrolled to a depth of 8. please
 * experiment with these to find the fastest for your particular
 * architecture. if reporting timing results, any of these three may
 * be used without penalty.
 * ---------------------------------------------------------------------
 * class specific parameters:
 * it appears here for reference only.
 * these are their values, however, this info is imported in the npbparams.h
 * include file, which is written by the sys/setparams.c program.
 * ---------------------------------------------------------------------
 */
#define NZ (NA * (NONZER + 1) * (NONZER + 1))
#define NAZ (NA * (NONZER + 1))
#define PROFILING_TOTAL_TIME (0)
#define PROFILING_KERNEL_ONE (1)
#define PROFILING_KERNEL_TWO (2)
#define PROFILING_KERNEL_THREE (3)
#define PROFILING_KERNEL_FOUR (4)
#define PROFILING_KERNEL_FIVE (5)
#define PROFILING_KERNEL_SIX (6)
#define PROFILING_KERNEL_SEVEN (7)
#define PROFILING_KERNEL_EIGHT (8)
#define PROFILING_KERNEL_NINE (9)
#define PROFILING_KERNEL_TEN (10)
#define PROFILING_KERNEL_ELEVEN (11)

extern __shared__ double extern_share_data[];

/* function prototypes */
static void conj_grad(int colidx[], int rowstr[], double x[], double z[],
                      double a[], double p[], double q[], double r[],
                      double* rnorm);
static void conj_grad_gpu(double* rnorm);
static void gpu_kernel_one();
__global__ void gpu_kernel_one(double p[], double q[], double r[], double x[],
                               double z[]);
static void gpu_kernel_two(double* rho_host);
__global__ void gpu_kernel_two(double r[], double* rho, double global_data[]);
static void gpu_kernel_three();
__global__ void gpu_kernel_three(int colidx[], int rowstr[], double a[],
                                 double p[], double q[]);
static void gpu_kernel_four(double* d_host);
__global__ void gpu_kernel_four(double* d, double* p, double* q,
                                double global_data[]);
static void gpu_kernel_five(double alpha_host);
__global__ void gpu_kernel_five_1(double alpha, double* p, double* z);
__global__ void gpu_kernel_five_2(double alpha, double* q, double* r);
static void gpu_kernel_six(double* rho_host);
__global__ void gpu_kernel_six(double r[], double global_data[]);
static void gpu_kernel_seven(double beta_host);
__global__ void gpu_kernel_seven(double beta, double* p, double* r);
static void gpu_kernel_eight();
__global__ void gpu_kernel_eight(int colidx[], int rowstr[], double a[],
                                 double r[], double* z);
static void gpu_kernel_nine(double* sum_host);
__global__ void gpu_kernel_nine(double r[], double x[], double* sum,
                                double global_data[]);
static void gpu_kernel_ten(double* norm_temp1, double* norm_temp2);
__global__ void gpu_kernel_ten_1(double* norm_temp, double x[], double z[]);
__global__ void gpu_kernel_ten_2(double* norm_temp, double x[], double z[]);
static void gpu_kernel_eleven(double norm_temp2);
__global__ void gpu_kernel_eleven(double norm_temp2, double x[], double z[]);
static int icnvrt(double x, int ipwr2);
static void makea(int n, int nz, double a[], int colidx[], int rowstr[],
                  int firstrow, int lastrow, int firstcol, int lastcol,
                  int arow[], int acol[][NONZER + 1], double aelt[][NONZER + 1],
                  int iv[]);
static void release_gpu();
static void setup_gpu();
static void sparse(double a[], int colidx[], int rowstr[], int n, int nz,
                   int nozer, int arow[], int acol[][NONZER + 1],
                   double aelt[][NONZER + 1], int firstrow, int lastrow,
                   int nzloc[], double rcond, double shift);
static void sprnvc(int n, int nz, int nn1, double v[], int iv[]);
static void vecset(int n, double v[], int iv[], int* nzv, int i, double val);
