#include "is.cuh"

__global__ void rank_gpu_kernel_1(INT_TYPE* key_array,
                                  INT_TYPE* partial_verify_vals,
                                  INT_TYPE* test_index_array,
                                  INT_TYPE iteration, INT_TYPE number_of_blocks,
                                  INT_TYPE amount_of_work) {
    key_array[iteration] = iteration;
    key_array[iteration + MAX_ITERATIONS] = MAX_KEY - iteration;
    /*
     * --------------------------------------------------------------------
     * determine where the partial verify test keys are,
     * --------------------------------------------------------------------
     * load into top of array bucket_size
     * --------------------------------------------------------------------
     */
#pragma unroll
    for (INT_TYPE i = 0; i < TEST_ARRAY_SIZE; i++)
        partial_verify_vals[i] = key_array[test_index_array[i]];
}

__global__ void rank_gpu_kernel_2(INT_TYPE* key_buff1,
                                  INT_TYPE number_of_blocks,
                                  INT_TYPE amount_of_work) {
    key_buff1[blockIdx.x * blockDim.x + threadIdx.x] = 0;
}

__global__ void rank_gpu_kernel_3(INT_TYPE* key_buff_ptr,
                                  INT_TYPE* key_buff_ptr2,
                                  INT_TYPE number_of_blocks,
                                  INT_TYPE amount_of_work) {
/*
 * --------------------------------------------------------------------
 * in this section, the keys themselves are used as their
 * own indexes to determine how many of each there are: their
 * individual population
 * --------------------------------------------------------------------
 */
#if CLASS == 'D'
    atomicAdd((unsigned long long int*)&key_buff_ptr
                  [key_buff_ptr2[blockIdx.x * blockDim.x + threadIdx.x]],
              (unsigned long long int)1);
#else
    atomicAdd(
        &key_buff_ptr[key_buff_ptr2[blockIdx.x * blockDim.x + threadIdx.x]], 1);
#endif
}

__global__ void rank_gpu_kernel_4(INT_TYPE* source, INT_TYPE* destiny,
                                  INT_TYPE* sum, INT_TYPE number_of_blocks,
                                  INT_TYPE amount_of_work) {
    INT_TYPE* shared_data = (INT_TYPE*)(extern_share_data);

    shared_data[threadIdx.x] = 0;
    INT_TYPE position = blockDim.x + threadIdx.x;

    INT_TYPE factor = MAX_KEY / number_of_blocks;
    INT_TYPE start = factor * blockIdx.x;
    INT_TYPE end = start + factor;

    for (INT_TYPE i = start; i < end; i += blockDim.x) {
        shared_data[position] = source[i + threadIdx.x];

        for (INT_TYPE offset = 1; offset < blockDim.x; offset <<= 1) {
            __syncthreads();
            INT_TYPE t = shared_data[position] + shared_data[position - offset];
            __syncthreads();
            shared_data[position] = t;
        }

        INT_TYPE prv_val = (i == start) ? 0 : destiny[i - 1];
        destiny[i + threadIdx.x] = shared_data[position] + prv_val;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        sum[blockIdx.x] = destiny[end - 1];
    }
}

__global__ void rank_gpu_kernel_5(INT_TYPE* source, INT_TYPE* destiny,
                                  INT_TYPE number_of_blocks,
                                  INT_TYPE amount_of_work) {
    INT_TYPE* shared_data = (INT_TYPE*)(extern_share_data);

    shared_data[threadIdx.x] = 0;
    INT_TYPE position = blockDim.x + threadIdx.x;
    shared_data[position] = source[threadIdx.x];

    for (INT_TYPE offset = 1; offset < blockDim.x; offset <<= 1) {
        __syncthreads();
        INT_TYPE t = shared_data[position] + shared_data[position - offset];
        __syncthreads();
        shared_data[position] = t;
    }

    __syncthreads();

    destiny[threadIdx.x] = shared_data[position - 1];
}

__global__ void rank_gpu_kernel_6(INT_TYPE* source, INT_TYPE* destiny,
                                  INT_TYPE* offset, INT_TYPE number_of_blocks,
                                  INT_TYPE amount_of_work) {
    INT_TYPE factor = MAX_KEY / number_of_blocks;
    INT_TYPE start = factor * blockIdx.x;
    INT_TYPE end = start + factor;
    INT_TYPE sum = offset[blockIdx.x];
    for (INT_TYPE i = start; i < end; i += blockDim.x)
        destiny[i + threadIdx.x] = source[i + threadIdx.x] + sum;
}
