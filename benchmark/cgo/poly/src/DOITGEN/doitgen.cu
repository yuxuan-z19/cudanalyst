/*********************************************************************************/
//
// Polybench kernels implementation on CUDA GPU
//
// Computer & Information Science, University of Delaware
// Author(s):   Sudhee Ayalasomayajula (sudhee1@gmail.com)
//              John Cavazos (cavazos@cis.udel.edu)
//		Scott Grauer Gray(sgrauerg@gmail.com)
//              Robert Searles (rsearles35@aol.com)
//              Lifan Xu (xulifan@udel.edu)
//
// Contact(s): Lifan Xu (xulifan@udel.edu)
// Reference(s):
//
/*********************************************************************************/

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
#include "doitgen.cuh"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

void doitgenCPU(DATA_TYPE* sum, DATA_TYPE* A, DATA_TYPE* C4) {
    for (int r = 0; r < NR; r++)
        for (int q = 0; q < NQ; q++) {
            for (int p = 0; p < NP; p++) {
                sum[r * (NQ * NP) + q * NP + p] = (DATA_TYPE)0.0;
                for (int s = 0; s < NP; s++)
                    sum[r * (NQ * NP) + q * NP + p] =
                        sum[r * (NQ * NP) + q * NP + p] +
                        A[r * (NQ * NP) + q * NP + s] * C4[s * NP + p];
            }

            for (int p = 0; p < NP; p++)
                A[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p];
        }
}

void init_array(DATA_TYPE* A, DATA_TYPE* C4) {
    for (int i = 0; i < NR; i++)
        for (int j = 0; j < NQ; j++)
            for (int k = 0; k < NP; k++)
                A[i * (NQ * NP) + j * NP + k] = ((DATA_TYPE)i * j + k) / NP;

    for (int i = 0; i < NP; i++)
        for (int j = 0; j < NP; j++) C4[i * NP + j] = ((DATA_TYPE)i * j) / NP;
}

void compareResults(DATA_TYPE* sum, DATA_TYPE* sum_outputFromGpu) {
    int fail = 0;

    for (int r = 0; r < NR; r++)
        for (int q = 0; q < NQ; q++)
            for (int p = 0; p < NP; p++)
                fail +=
                    percentDiff(sum[r * (NQ * NP) + q * NP + p],
                                sum_outputFromGpu[r * (NQ * NP) + q * NP + p]) >
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

__global__ void doitgen_kernel1(DATA_TYPE* sum, DATA_TYPE* A, DATA_TYPE* C4,
                                int r);

__global__ void doitgen_kernel2(DATA_TYPE* sum, DATA_TYPE* A, DATA_TYPE* C4,
                                int r);

void doitgenCuda(DATA_TYPE* A, DATA_TYPE* C4, DATA_TYPE* sum,
                 DATA_TYPE* sum_outputFromGpu) {
    DATA_TYPE* AGpu;
    DATA_TYPE* C4Gpu;
    DATA_TYPE* sumGpu;

    cudaMalloc(&AGpu, NR * NQ * NP * sizeof(DATA_TYPE));
    cudaMalloc(&C4Gpu, NP * NP * sizeof(DATA_TYPE));
    cudaMalloc(&sumGpu, NR * NQ * NP * sizeof(DATA_TYPE));

    cudaMemcpy(AGpu, A, NR * NQ * NP * sizeof(DATA_TYPE),
               cudaMemcpyHostToDevice);
    cudaMemcpy(C4Gpu, C4, NP * NP * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(sumGpu, sum, NR * NQ * NP * sizeof(DATA_TYPE),
               cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((unsigned int)ceil(((float)NP) / ((float)block.x)),
              (unsigned int)ceil(((float)NR) / ((float)block.y)));

    polybench_start_instruments;

    for (int r = 0; r < NR; r++) {
        doitgen_kernel1<<<grid, block>>>(sumGpu, AGpu, C4Gpu, r);
        cudaDeviceSynchronize();
        doitgen_kernel2<<<grid, block>>>(sumGpu, AGpu, C4Gpu, r);
        cudaDeviceSynchronize();
    }

    printf("GPU_Seconds=");
    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(sum_outputFromGpu, sumGpu, NR * NQ * NP * sizeof(DATA_TYPE),
               cudaMemcpyDeviceToHost);

    cudaFree(AGpu);
    cudaFree(C4Gpu);
    cudaFree(sumGpu);
}

int main(int argc, char* argv[]) {
    DATA_TYPE* A = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));
    DATA_TYPE* C4 = (DATA_TYPE*)malloc(NP * NP * sizeof(DATA_TYPE));
    DATA_TYPE* sum = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));
    DATA_TYPE* sum_outputFromGpu =
        (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));

    init_array(A, C4);

    doitgenCuda(A, C4, sum, sum_outputFromGpu);

    polybench_start_instruments;

    doitgenCPU(sum, A, C4);

    printf("CPU_Seconds=");
    polybench_stop_instruments;
    polybench_print_instruments;

    compareResults(sum, sum_outputFromGpu);

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(C4);
    POLYBENCH_FREE_ARRAY(sum);
    POLYBENCH_FREE_ARRAY(sum_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"