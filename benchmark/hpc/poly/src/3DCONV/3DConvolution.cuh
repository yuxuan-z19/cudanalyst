/**
 * 3DConvolution.cuh: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#ifndef THREEDCONV_H
#define THREEDCONV_H

/* Default to STANDARD_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && \
    !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define STANDARD_DATASET
#endif

/* Do not define anything if the user manually defines the size. */
#if !defined(N)
/* Define the possible dataset sizes. */
#ifdef MINI_DATASET
#define NI 256
#define NJ 256
#define NK 256
#endif

#ifdef SMALL_DATASET
#define NI 256
#define NJ 256
#define NK 256
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
#define NI 256
#define NJ 256
#define NK 256
#endif

#ifdef LARGE_DATASET
#define NI 256
#define NJ 256
#define NK 256
#endif

#ifdef EXTRALARGE_DATASET
#define NI 256
#define NJ 256
#define NK 256
#endif
#endif /* !N */

#define _PB_NI POLYBENCH_LOOP_BOUND(NI, ni)
#define _PB_NJ POLYBENCH_LOOP_BOUND(NJ, nj)
#define _PB_NK POLYBENCH_LOOP_BOUND(NK, nk)

#ifndef DATA_TYPE
#define DATA_TYPE float
#define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Coefficient */
#define c11 (DATA_TYPE)(+2)
#define c21 (DATA_TYPE)(+5)
#define c31 (DATA_TYPE)(-8)
#define c12 (DATA_TYPE)(-3)
#define c22 (DATA_TYPE)(+6)
#define c32 (DATA_TYPE)(-9)
#define c13 (DATA_TYPE)(+4)
#define c23 (DATA_TYPE)(+7)
#define c33 (DATA_TYPE)(+10)

#endif /* !THREEDCONV*/