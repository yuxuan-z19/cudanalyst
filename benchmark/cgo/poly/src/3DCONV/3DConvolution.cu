/**
 * 3DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <cuda.h>
#include <nvtx3/nvToolsExt.h>
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
#include "3DConvolution.cuh"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

#define RUN_ON_CPU

void conv3D(int ni, int nj, int nk,
            DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk),
            DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk)) {
    for (int i = 1; i < _PB_NI - 1; ++i)          // 0
        for (int j = 1; j < _PB_NJ - 1; ++j)      // 1
            for (int k = 1; k < _PB_NK - 1; ++k)  // 2
                B[i][j][k] = c11 * A[(i - 1)][(j - 1)][(k - 1)] +
                             c13 * A[(i + 1)][(j - 1)][(k - 1)] +
                             c21 * A[(i - 1)][(j - 1)][(k - 1)] +
                             c23 * A[(i + 1)][(j - 1)][(k - 1)] +
                             c31 * A[(i - 1)][(j - 1)][(k - 1)] +
                             c33 * A[(i + 1)][(j - 1)][(k - 1)] +
                             c12 * A[(i + 0)][(j - 1)][(k + 0)] +
                             c22 * A[(i + 0)][(j + 0)][(k + 0)] +
                             c32 * A[(i + 0)][(j + 1)][(k + 0)] +
                             c11 * A[(i - 1)][(j - 1)][(k + 1)] +
                             c13 * A[(i + 1)][(j - 1)][(k + 1)] +
                             c21 * A[(i - 1)][(j + 0)][(k + 1)] +
                             c23 * A[(i + 1)][(j + 0)][(k + 1)] +
                             c31 * A[(i - 1)][(j + 1)][(k + 1)] +
                             c33 * A[(i + 1)][(j + 1)][(k + 1)];
}

void init(int ni, int nj, int nk,
          DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk)) {
    for (int i = 0; i < ni; ++i)
        for (int j = 0; j < nj; ++j)
            for (int k = 0; k < nk; ++k)
                A[i][j][k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
}

void compareResults(int ni, int nj, int nk,
                    DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk),
                    DATA_TYPE POLYBENCH_3D(B_outputFromGpu, NI, NJ, NK, ni, nj,
                                           nk)) {
    int fail = 0;

    // Compare result from cpu and gpu
    for (int i = 1; i < ni - 1; ++i)          // 0
        for (int j = 1; j < nj - 1; ++j)      // 1
            for (int k = 1; k < nk - 1; ++k)  // 2
                fail += percentDiff(B[i][j][k], B_outputFromGpu[i][j][k]) >
                        PERCENT_DIFF_ERROR_THRESHOLD;

    // Print results
    printf("Mismatch_Count=%d\n", fail);
}

void GPU_argv_init() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    // printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}

__global__ void convolution3D_kernel(int ni, int nj, int nk, DATA_TYPE* A,
                                     DATA_TYPE* B, int i);

void convolution3DCuda(int ni, int nj, int nk,
                       DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk),
                       DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk),
                       DATA_TYPE POLYBENCH_3D(B_outputFromGpu, NI, NJ, NK, ni,
                                              nj, nk)) {
    DATA_TYPE* A_gpu;
    DATA_TYPE* B_gpu;

    cudaMalloc((void**)&A_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
    cudaMalloc((void**)&B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ * NK,
               cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NI * NJ * NK,
               cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((size_t)(ceil(((float)NK) / ((float)block.x))),
              (size_t)(ceil(((float)NJ) / ((float)block.y))));

    /* Start timer. */
    nvtxRangePushA("cugedit");
    polybench_start_instruments;
    for (int i = 1; i < _PB_NI - 1; ++i)  // 0
        convolution3D_kernel<<<grid, block>>>(ni, nj, nk, A_gpu, B_gpu, i);
    cudaDeviceSynchronize();
    printf("GPU_Seconds=");
    polybench_stop_instruments;
    polybench_print_instruments;
    nvtxRangePop();

    cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK,
               cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj, int nk,
                        DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk)) {
    int i, j, k;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++)
            for (k = 0; k < nk; k++) {
                fprintf(stderr, DATA_PRINTF_MODIFIER, B[i][j][k]);
                if ((i * (nj * nk) + j * nk + k) % 20 == 0)
                    fprintf(stderr, "\n");
            }
    fprintf(stderr, "\n");
}

int main(int argc, char* argv[]) {
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, NK, ni, nj, nk);
    POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, NI, NJ, NK, ni, nj, nk);
    POLYBENCH_3D_ARRAY_DECL(B_outputFromGpu, DATA_TYPE, NI, NJ, NK, ni, nj, nk);

    init(ni, nj, nk, POLYBENCH_ARRAY(A));

    GPU_argv_init();

    convolution3DCuda(ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
                      POLYBENCH_ARRAY(B_outputFromGpu));

#ifdef RUN_ON_CPU

    /* Start timer. */
    polybench_start_instruments;

    conv3D(ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    printf("CPU_Seconds=");
    polybench_stop_instruments;
    polybench_print_instruments;

    compareResults(ni, nj, nk, POLYBENCH_ARRAY(B),
                   POLYBENCH_ARRAY(B_outputFromGpu));

#else  // print output to stderr so no dead code elimination

    print_array(ni, nj, nk, POLYBENCH_ARRAY(B_outputFromGpu));

#endif  // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(B_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
