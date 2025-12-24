#pragma once

#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

#define MK (16)
#define MM (M - MK)
#define NN (1 << MM)
#define NK (1 << MK)
#define NQ (10)
#define EPSILON (1.0e-8)
#define A (1220703125.0)
#define S (271828183.0)
#define NK_PLUS ((2 * NK) + 1)
#define RECOMPUTATION (128)
#define PROFILING_TOTAL_TIME (0)

/* function declarations */
__global__ void gpu_kernel(double* q_device, double* sx_device,
                           double* sy_device, double an);
__device__ double randlc_device(double* x, double a);
static void release_gpu();
static void setup_gpu();
__device__ void vranlc_device(int n, double* x_seed, double a, double* y);
