#pragma once

#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

#define NM \
    (2 +   \
     (1    \
      << LM)) /* actual dimension including ghost cells for communications */
#define NV                                           \
    (ONE * (2 + (1 << NDIM1)) * (2 + (1 << NDIM2)) * \
     (2 + (1 << NDIM3))) /* size of rhs array */
#define NR                                        \
    (((NV + NM * NM + 5 * NM + 7 * LM + 6) / 7) * \
     8)                           /* size of residual array */
#define MAXLEVEL (LT_DEFAULT + 1) /* maximum number of levels */
#define M (NM + 1) /* set at m=1024, can handle cases up to 1024^3 case */
#define MM (10)
#define A (pow(5.0, 13.0))
#define X (314159265.0)
#define PROFILING_TOTAL_TIME (0)
#define PROFILING_COMM3 (1)
#define PROFILING_INTERP (2)
#define PROFILING_NORM2U3 (3)
#define PROFILING_PSINV (4)
#define PROFILING_RESID (5)
#define PROFILING_RPRJ3 (6)
#define PROFILING_ZERO3 (7)

extern __shared__ double extern_share_data[];

/* function prototypes */
static void bubble(double ten[][MM], int j1[][MM], int j2[][MM], int j3[][MM],
                   int m, int ind);
static void comm3(void* pointer_u, int n1, int n2, int n3, int kk);
static void comm3_gpu(double* u_device, int n1, int n2, int n3, int kk);
__global__ void comm3_gpu_kernel_1(double* u, int n1, int n2, int n3,
                                   int amount_of_work);
__global__ void comm3_gpu_kernel_2(double* u, int n1, int n2, int n3,
                                   int amount_of_work);
__global__ void comm3_gpu_kernel_3(double* u, int n1, int n2, int n3,
                                   int amount_of_work);
static void interp(void* pointer_z, int mm1, int mm2, int mm3, void* pointer_u,
                   int n1, int n2, int n3, int k);
static void interp_gpu(double* z_device, int mm1, int mm2, int mm3,
                       double* u_device, int n1, int n2, int n3, int k);
__global__ void interp_gpu_kernel(double* z_device, double* u_device, int mm1,
                                  int mm2, int mm3, int n1, int n2, int n3,
                                  int amount_of_work);
static void mg3P(double u[], double v[], double r[], double a[4], double c[4],
                 int n1, int n2, int n3, int k);
static void mg3P_gpu(double* u_device, double* v_device, double* r_device,
                     double a[4], double c[4], int n1, int n2, int n3, int k);
static void norm2u3(void* pointer_r, int n1, int n2, int n3, double* rnm2,
                    double* rnmu, int nx, int ny, int nz);
static void norm2u3_gpu(double* r_device, int n1, int n2, int n3, double* rnm2,
                        double* rnmu, int nx, int ny, int nz);
__global__ void norm2u3_gpu_kernel(double* r, const int n1, const int n2,
                                   const int n3, double* res_sum,
                                   double* res_max, int number_of_blocks,
                                   int amount_of_work);
static double power(double a, int n);
static void psinv(void* pointer_r, void* pointer_u, int n1, int n2, int n3,
                  double c[4], int k);
static void psinv_gpu(double* r_device, double* u_device, int n1, int n2,
                      int n3, double* c_device, int k);
__global__ void psinv_gpu_kernel(double* r, double* u, double* c, int n1,
                                 int n2, int n3, int amount_of_work);
static void release_gpu();
static void rep_nrm(void* pointer_u, int n1, int n2, int n3, char* title,
                    int kk);
static void resid(void* pointer_u, void* pointer_v, void* pointer_r, int n1,
                  int n2, int n3, double a[4], int k);
static void resid_gpu(double* u_device, double* v_device, double* r_device,
                      int n1, int n2, int n3, double* a_device, int k);
__global__ void resid_gpu_kernel(double* r, double* u, double* v, double* a,
                                 int n1, int n2, int n3, int amount_of_work);
static void rprj3(void* pointer_r, int m1k, int m2k, int m3k, void* pointer_s,
                  int m1j, int m2j, int m3j, int k);
static void rprj3_gpu(double* r_device, int m1k, int m2k, int m3k,
                      double* s_device, int m1j, int m2j, int m3j, int k);
__global__ void rprj3_gpu_kernel(double* r_device, double* s_device, int m1k,
                                 int m2k, int m3k, int m1j, int m2j, int m3j,
                                 int d1, int d2, int d3, int amount_of_work);
static void setup(int* n1, int* n2, int* n3, int k);
static void setup_gpu(double* a, double* c);
static void showall(void* pointer_z, int n1, int n2, int n3);
static void zero3_gpu(double* z_device, int n1, int n2, int n3);
__global__ void zero3_gpu_kernel(double* z, int n1, int n2, int n3,
                                 int amount_of_work);
static void zero3(void* pointer_z, int n1, int n2, int n3);
static void zran3(void* pointer_z, int n1, int n2, int n3, int nx, int ny,
                  int k);