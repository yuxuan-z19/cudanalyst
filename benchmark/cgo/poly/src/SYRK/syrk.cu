/**
 * syrk.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <assert.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define POLYBENCH_TIME 1

#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"
#include "syrk.cuh"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

void init_arrays(int ni, int nj, DATA_TYPE* alpha, DATA_TYPE* beta,
                 DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni),
                 DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj)) {
    *alpha = ALPHA;
    *beta = BETA;
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nj; j++) A[i][j] = ((DATA_TYPE)i * j) / ni;
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < ni; j++) C[i][j] = ((DATA_TYPE)i * j) / ni;
}

void syrk(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta,
          DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
          DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni)) {
    int i, j, k;
    /* C := alpha*A*A' + beta*C */
    for (i = 0; i < _PB_NI; i++)
        for (j = 0; j < _PB_NI; j++) C[i][j] *= beta;

    for (i = 0; i < _PB_NI; i++)
        for (j = 0; j < _PB_NI; j++)
            for (k = 0; k < _PB_NJ; k++) C[i][j] += alpha * A[i][k] * A[j][k];
}

void compareResults(int ni, DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni),
                    DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NI, ni, ni)) {
    int fail = 0;

    // Compare C with D
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < ni; j++)
            fail += percentDiff(C[i][j], C_outputFromGpu[i][j]) >
                    PERCENT_DIFF_ERROR_THRESHOLD;

    // print results
    printf("Mismatch_Count=%d\n", fail);
}

void GPU_argv_init() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    // printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}

__global__ void syrk_kernel(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta,
                            DATA_TYPE* a, DATA_TYPE* c);

void syrkCuda(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta,
              DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
              DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni),
              DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NI, ni, ni)) {
    DATA_TYPE* A_gpu;
    DATA_TYPE* C_gpu;

    cudaMalloc((void**)&A_gpu, sizeof(DATA_TYPE) * NI * NJ);
    cudaMalloc((void**)&C_gpu, sizeof(DATA_TYPE) * NI * NI);
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NI, cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((size_t)(ceil(((float)NI) / ((float)DIM_THREAD_BLOCK_X))),
              (size_t)ceil(((float)NI) / ((float)DIM_THREAD_BLOCK_Y)));

    /* Start timer. */
    polybench_start_instruments;

    syrk_kernel<<<grid, block>>>(ni, nj, alpha, beta, A_gpu, C_gpu);
    cudaDeviceSynchronize();

    /* Stop and print timer. */
    printf("GPU_Seconds=");
    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NI,
               cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(C_gpu);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni)) {
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < ni; j++) {
            fprintf(stderr, DATA_PRINTF_MODIFIER, C[i][j]);
            if ((i * ni + j) % 20 == 0) fprintf(stderr, "\n");
        }
    fprintf(stderr, "\n");
}

int main() {
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NI, ni, ni);
    POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu, DATA_TYPE, NI, NI, ni, ni);

    init_arrays(ni, nj, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

    GPU_argv_init();
    syrkCuda(ni, nj, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C),
             POLYBENCH_ARRAY(C_outputFromGpu));

#ifdef RUN_ON_CPU

    /* Start timer. */
    polybench_start_instruments;

    syrk(ni, nj, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C));

    /* Stop and print timer. */
    printf("CPU_Seconds=");
    polybench_stop_instruments;
    polybench_print_instruments;

    compareResults(ni, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

#else  // print output to stderr so no dead code elimination

    print_array(ni, POLYBENCH_ARRAY(C_outputFromGpu));

#endif  // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(C_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
