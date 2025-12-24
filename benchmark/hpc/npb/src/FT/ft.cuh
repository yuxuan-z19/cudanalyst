#pragma once

#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

/*
 * ---------------------------------------------------------------------
 * u0, u1, u2 are the main arrays in the problem.
 * depending on the decomposition, these arrays will have different
 * dimensions. to accomodate all possibilities, we allocate them as
 * one-dimensional arrays and pass them to subroutines for different
 * views
 * - u0 contains the initial (transformed) initial condition
 * - u1 and u2 are working arrays
 * - twiddle contains exponents for the time evolution operator.
 * ---------------------------------------------------------------------
 * large arrays are in common so that they are allocated on the
 * heap rather than the stack. this common block is not
 * referenced directly anywhere else. padding is to avoid accidental
 * cache problems, since all array sizes are powers of two.
 * ---------------------------------------------------------------------
 * we need a bunch of logic to keep track of how
 * arrays are laid out.
 *
 * note: this serial version is the derived from the parallel 0D case
 * of the ft NPB.
 * the computation proceeds logically as
 *
 * set up initial conditions
 * fftx(1)
 * transpose (1->2)
 * ffty(2)
 * transpose (2->3)
 * fftz(3)
 * time evolution
 * fftz(3)
 * transpose (3->2)
 * ffty(2)
 * transpose (2->1)
 * fftx(1)
 * compute residual(1)
 *
 * for the 0D, 1D, 2D strategies, the layouts look like xxx
 *
 *            0D        1D        2D
 * 1:        xyz       xyz       xyz
 * 2:        xyz       xyz       yxz
 * 3:        xyz       zyx       zxy
 * the array dimensions are stored in dims(coord, phase)
 * ---------------------------------------------------------------------
 * if processor array is 1x1 -> 0D grid decomposition
 *
 * cache blocking params. these values are good for most
 * RISC processors.
 * FFT parameters:
 * fftblock controls how many ffts are done at a time.
 * the default is appropriate for most cache-based machines
 * on vector machines, the FFT can be vectorized with vector
 * length equal to the block size, so the block size should
 * be as large as possible. this is the size of the smallest
 * dimension of the problem: 128 for class A, 256 for class B
 * and 512 for class C.
 * ---------------------------------------------------------------------
 */
#define FFTBLOCK_DEFAULT (DEFAULT_BEHAVIOR)
#define FFTBLOCKPAD_DEFAULT (DEFAULT_BEHAVIOR)
#define FFTBLOCK (FFTBLOCK_DEFAULT)
#define FFTBLOCKPAD (FFTBLOCKPAD_DEFAULT)
#define SEED (314159265.0)
#define A (1220703125.0)
#define PI (3.141592653589793238)
#define ALPHA (1.0e-6)
#define AP (-4.0 * ALPHA * PI * PI)
#define OMP_THREADS (3)
#define TASK_INDEXMAP (0)
#define TASK_INITIAL_CONDITIONS (1)
#define TASK_INIT_UI (2)
#define PROFILING_TOTAL_TIME (0)
#define PROFILING_INDEXMAP (1)
#define PROFILING_INITIAL_CONDITIONS (2)
#define PROFILING_INIT_UI (3)
#define PROFILING_EVOLVE (4)
#define PROFILING_FFTX_1 (5)
#define PROFILING_FFTX_2 (6)
#define PROFILING_FFTX_3 (7)
#define PROFILING_FFTY_1 (8)
#define PROFILING_FFTY_2 (9)
#define PROFILING_FFTY_3 (10)
#define PROFILING_FFTZ_1 (11)
#define PROFILING_FFTZ_2 (12)
#define PROFILING_FFTZ_3 (13)
#define PROFILING_CHECKSUM (14)
#define PROFILING_INIT (15)
#define CHECKSUM_TASKS (1024)

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static dcomplex sums[NITER_DEFAULT + 1];
static double twiddle[NTOTAL];
static dcomplex u[MAXDIM];
static dcomplex u0[NTOTAL];
static dcomplex u1[NTOTAL];
static int dims[3];
#else
static dcomplex(*sums) = (dcomplex*)malloc(sizeof(dcomplex) *
                                           (NITER_DEFAULT + 1));
static double(*twiddle) = (double*)malloc(sizeof(double) * (NTOTAL));
static dcomplex(*u) = (dcomplex*)malloc(sizeof(dcomplex) * (MAXDIM));
static dcomplex(*u0) = (dcomplex*)malloc(sizeof(dcomplex) * (NTOTAL));
static dcomplex(*u1) = (dcomplex*)malloc(sizeof(dcomplex) * (NTOTAL));
static int(*dims) = (int*)malloc(sizeof(int) * (3));
#endif

extern __shared__ double extern_share_data[];

/* function declarations */
static void cffts1_gpu(const int is, dcomplex u[], dcomplex x_in[],
                       dcomplex x_out[], dcomplex y0[], dcomplex y1[]);
__global__ void cffts1_gpu_kernel_1(dcomplex x_in[], dcomplex y0[]);
__global__ void cffts1_gpu_kernel_2(const int is, dcomplex y0[], dcomplex y1[],
                                    dcomplex u_device[]);
__global__ void cffts1_gpu_kernel_3(dcomplex x_out[], dcomplex y0[]);
static void cffts2_gpu(int is, dcomplex u[], dcomplex x_in[], dcomplex x_out[],
                       dcomplex y0[], dcomplex y1[]);
__global__ void cffts2_gpu_kernel_1(dcomplex x_in[], dcomplex y0[]);
__global__ void cffts2_gpu_kernel_2(const int is, dcomplex y0[], dcomplex y1[],
                                    dcomplex u_device[]);
__global__ void cffts2_gpu_kernel_3(dcomplex x_out[], dcomplex y0[]);
static void cffts3_gpu(int is, dcomplex u[], dcomplex x_in[], dcomplex x_out[],
                       dcomplex y0[], dcomplex y1[]);
__device__ void cffts3_gpu_cfftz_device(const int is, int m, int n,
                                        dcomplex x[], dcomplex y[],
                                        dcomplex u_device[], int index_arg,
                                        int size_arg);
__device__ void cffts3_gpu_fftz2_device(const int is, int l, int m, int n,
                                        dcomplex u[], dcomplex x[],
                                        dcomplex y[], int index_arg,
                                        int size_arg);
__global__ void cffts3_gpu_kernel_1(dcomplex x_in[], dcomplex y0[]);
__global__ void cffts3_gpu_kernel_2(const int is, dcomplex y0[], dcomplex y1[],
                                    dcomplex u_device[]);
__global__ void cffts3_gpu_kernel_3(dcomplex x_out[], dcomplex y0[]);
static void checksum_gpu(int iteration, dcomplex u1[]);
__global__ void checksum_gpu_kernel(int iteration, dcomplex u1[],
                                    dcomplex sums[]);
static void compute_indexmap_gpu(double twiddle[]);
__global__ void compute_indexmap_gpu_kernel(double twiddle[]);
static void compute_initial_conditions_gpu(dcomplex u0[]);
__global__ void compute_initial_conditions_gpu_kernel(dcomplex u0[],
                                                      double starts[]);
static void evolve_gpu(dcomplex u0[], dcomplex u1[], double twiddle[]);
__global__ void evolve_gpu_kernel(dcomplex u0[], dcomplex u1[],
                                  double twiddle[]);
static void fft_gpu(int dir, dcomplex x1[], dcomplex x2[]);
static void fft_init_gpu(int n);
static int ilog2(int n);
__device__ int ilog2_device(int n);
static void init_ui_gpu(dcomplex u0[], dcomplex u1[], double twiddle[]);
__global__ void init_ui_gpu_kernel(dcomplex u0[], dcomplex u1[],
                                   double twiddle[]);
static void ipow46(double a, int exponent, double* result);
__device__ void ipow46_device(double a, int exponent, double* result);
__device__ double randlc_device(double* x, double a);
static void release_gpu();
static void setup();
static void setup_gpu();
static void verify(int d1, int d2, int d3, int nt, boolean* verified,
                   char* class_npb);
__device__ void vranlc_device(int n, double* x_seed, double a, double y[]);
