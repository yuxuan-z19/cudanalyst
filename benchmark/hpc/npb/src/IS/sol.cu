#include "is.cuh"

__global__ void rank_gpu_kernel_1(INT_TYPE* key_array,
                                  INT_TYPE* partial_verify_vals,
                                  INT_TYPE* test_index_array,
                                  INT_TYPE iteration, INT_TYPE number_of_blocks,
                                  INT_TYPE amount_of_work) {
    // TODO:
}

__global__ void rank_gpu_kernel_2(INT_TYPE* key_buff1,
                                  INT_TYPE number_of_blocks,
                                  INT_TYPE amount_of_work) {
    // TODO:
}

__global__ void rank_gpu_kernel_3(INT_TYPE* key_buff_ptr,
                                  INT_TYPE* key_buff_ptr2,
                                  INT_TYPE number_of_blocks,
                                  INT_TYPE amount_of_work) {
    // TODO:
}

__global__ void rank_gpu_kernel_4(INT_TYPE* source, INT_TYPE* destiny,
                                  INT_TYPE* sum, INT_TYPE number_of_blocks,
                                  INT_TYPE amount_of_work) {
    // TODO:
}

__global__ void rank_gpu_kernel_5(INT_TYPE* source, INT_TYPE* destiny,
                                  INT_TYPE number_of_blocks,
                                  INT_TYPE amount_of_work) {
    // TODO:
}

__global__ void rank_gpu_kernel_6(INT_TYPE* source, INT_TYPE* destiny,
                                  INT_TYPE* offset, INT_TYPE number_of_blocks,
                                  INT_TYPE amount_of_work) {
    // TODO:
}

// ! rank_gpu_kernel_7 is part of the verification test, should not appear here
