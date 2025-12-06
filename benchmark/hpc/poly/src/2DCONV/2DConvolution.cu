/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define POLYBENCH_TIME 1

#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"
#include "2DConvolution.cuh"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

void conv2D(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
            DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj)) {
    for (int i = 1; i < _PB_NI - 1; ++i)      // 0
        for (int j = 1; j < _PB_NJ - 1; ++j)  // 1
            B[i][j] = c11 * A[(i - 1)][(j - 1)] + c12 * A[(i + 0)][(j - 1)] +
                      c13 * A[(i + 1)][(j - 1)] + c21 * A[(i - 1)][(j + 0)] +
                      c22 * A[(i + 0)][(j + 0)] + c23 * A[(i + 1)][(j + 0)] +
                      c31 * A[(i - 1)][(j + 1)] + c32 * A[(i + 0)][(j + 1)] +
                      c33 * A[(i + 1)][(j + 1)];
}

void init(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj)) {
    for (int i = 0; i < ni; ++i)
        for (int j = 0; j < nj; ++j) A[i][j] = (float)rand() / RAND_MAX;
}

void compareResults(int ni, int nj, DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj),
                    DATA_TYPE POLYBENCH_2D(B_outputFromGpu, NI, NJ, ni, nj)) {
    int fail = 0;

    // Compare outputs from CPU and GPU
    for (int i = 1; i < (ni - 1); i++)
        for (int j = 1; j < (nj - 1); j++)
            fail += (percentDiff(B[i][j], B_outputFromGpu[i][j]) >
                     PERCENT_DIFF_ERROR_THRESHOLD);

    // Print results
    printf("Mismatch_Count=%d\n", fail);
}

void GPU_argv_init() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    // printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}

__global__ void convolution2D_kernel(int ni, int nj, DATA_TYPE* A,
                                     DATA_TYPE* B);

void convolution2DCuda(int ni, int nj,
                       DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(B_outputFromGpu, NI, NJ, ni,
                                              nj)) {
    DATA_TYPE* A_gpu;
    DATA_TYPE* B_gpu;

    cudaMalloc((void**)&A_gpu, sizeof(DATA_TYPE) * NI * NJ);
    cudaMalloc((void**)&B_gpu, sizeof(DATA_TYPE) * NI * NJ);
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((size_t)ceil(((float)NI) / ((float)block.x)),
              (size_t)ceil(((float)NJ) / ((float)block.y)));

    polybench_start_instruments;

    convolution2D_kernel<<<grid, block>>>(ni, nj, A_gpu, B_gpu);
    cudaDeviceSynchronize();

    /* Stop and print timer. */
    printf("GPU_Seconds=");

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ,
               cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj,
                        DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj)) {
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nj; j++) {
            fprintf(stderr, DATA_PRINTF_MODIFIER, B[i][j]);
            if ((i * ni + j) % 20 == 0) fprintf(stderr, "\n");
        }
    fprintf(stderr, "\n");
}

int main(int argc, char* argv[]) {
    /* Retrieve problem size */
    int ni = NI;
    int nj = NJ;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(B_outputFromGpu, DATA_TYPE, NI, NJ, ni, nj);

    // initialize the arrays
    init(ni, nj, POLYBENCH_ARRAY(A));

    GPU_argv_init();

    convolution2DCuda(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
                      POLYBENCH_ARRAY(B_outputFromGpu));

#ifdef RUN_ON_CPU

    /* Start timer. */
    polybench_start_instruments;

    conv2D(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    /* Stop and print timer. */
    printf("CPU_Seconds=");
    polybench_stop_instruments;
    polybench_print_instruments;

    compareResults(ni, nj, POLYBENCH_ARRAY(B),
                   POLYBENCH_ARRAY(B_outputFromGpu));

#else  // print output to stderr so no dead code elimination

    print_array(ni, nj, POLYBENCH_ARRAY(B_outputFromGpu));

#endif  // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(B_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
