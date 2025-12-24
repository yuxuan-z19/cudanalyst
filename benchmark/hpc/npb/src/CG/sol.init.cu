#include "cg.cuh"

__global__ void gpu_kernel_one(double p[], double q[], double r[], double x[],
                               double z[]) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= NA) return;
    q[thread_id] = 0.0;
    z[thread_id] = 0.0;
    double x_value = x[thread_id];
    r[thread_id] = x_value;
    p[thread_id] = x_value;
}

__global__ void gpu_kernel_two(double r[], double* rho, double global_data[]) {
    double* share_data = (double*)extern_share_data;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    share_data[local_id] = 0.0;

    if (thread_id >= NA) return;

    double r_value = r[thread_id];
    share_data[local_id] = r_value * r_value;

    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (local_id < i) share_data[local_id] += share_data[local_id + i];
        __syncthreads();
    }
    if (local_id == 0) global_data[blockIdx.x] = share_data[0];
}

__global__ void gpu_kernel_three(int colidx[], int rowstr[], double a[],
                                 double p[], double q[]) {
    double* share_data = (double*)extern_share_data;

    int j = (int)((blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x);
    int local_id = threadIdx.x;

    int begin = rowstr[j];
    int end = rowstr[j + 1];
    double sum = 0.0;
    for (int k = begin + local_id; k < end; k += blockDim.x)
        sum = sum + a[k] * p[colidx[k]];
    share_data[local_id] = sum;

    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (local_id < i) share_data[local_id] += share_data[local_id + i];
        __syncthreads();
    }
    if (local_id == 0) q[j] = share_data[0];
}

__global__ void gpu_kernel_four(double* d, double* p, double* q,
                                double global_data[]) {
    double* share_data = (double*)extern_share_data;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    share_data[local_id] = 0.0;

    if (thread_id >= NA) return;

    share_data[threadIdx.x] = p[thread_id] * q[thread_id];

    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (local_id < i) share_data[local_id] += share_data[local_id + i];
        __syncthreads();
    }
    if (local_id == 0) global_data[blockIdx.x] = share_data[0];
}

__global__ void gpu_kernel_five_1(double alpha, double* p, double* z) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= NA) return;
    z[j] += alpha * p[j];
}

__global__ void gpu_kernel_five_2(double alpha, double* q, double* r) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= NA) return;
    r[j] -= alpha * q[j];
}

__global__ void gpu_kernel_six(double r[], double global_data[]) {
    double* share_data = (double*)extern_share_data;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;
    share_data[local_id] = 0.0;
    if (thread_id >= NA) return;
    double r_value = r[thread_id];
    share_data[local_id] = r_value * r_value;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (local_id < i) share_data[local_id] += share_data[local_id + i];
        __syncthreads();
    }
    if (local_id == 0) global_data[blockIdx.x] = share_data[0];
}

__global__ void gpu_kernel_seven(double beta, double* p, double* r) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= NA) return;
    p[j] = r[j] + beta * p[j];
}

__global__ void gpu_kernel_eight(int colidx[], int rowstr[], double a[],
                                 double r[], double* z) {
    double* share_data = (double*)extern_share_data;

    int j = (int)((blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x);
    int local_id = threadIdx.x;

    int begin = rowstr[j];
    int end = rowstr[j + 1];
    double sum = 0.0;
    for (int k = begin + local_id; k < end; k += blockDim.x)
        sum = sum + a[k] * z[colidx[k]];
    share_data[local_id] = sum;

    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (local_id < i) share_data[local_id] += share_data[local_id + i];
        __syncthreads();
    }
    if (local_id == 0) r[j] = share_data[0];
}

__global__ void gpu_kernel_nine(double r[], double x[], double* sum,
                                double global_data[]) {
    double* share_data = (double*)extern_share_data;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    share_data[local_id] = 0.0;

    if (thread_id >= NA) return;

    share_data[local_id] = x[thread_id] - r[thread_id];
    share_data[local_id] = share_data[local_id] * share_data[local_id];

    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (local_id < i) share_data[local_id] += share_data[local_id + i];
        __syncthreads();
    }
    if (local_id == 0) global_data[blockIdx.x] = share_data[0];
}

__global__ void gpu_kernel_ten_1(double* norm_temp, double x[], double z[]) {
    double* share_data = (double*)extern_share_data;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    share_data[threadIdx.x] = 0.0;

    if (thread_id >= NA) return;

    share_data[threadIdx.x] = x[thread_id] * z[thread_id];

    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (local_id < i) share_data[local_id] += share_data[local_id + i];
        __syncthreads();
    }
    if (local_id == 0) norm_temp[blockIdx.x] = share_data[0];
}

__global__ void gpu_kernel_ten_2(double* norm_temp, double x[], double z[]) {
    double* share_data = (double*)extern_share_data;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    share_data[threadIdx.x] = 0.0;

    if (thread_id >= NA) return;

    share_data[threadIdx.x] = z[thread_id] * z[thread_id];

    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (local_id < i) share_data[local_id] += share_data[local_id + i];
        __syncthreads();
    }
    if (local_id == 0) norm_temp[blockIdx.x] = share_data[0];
}

__global__ void gpu_kernel_eleven(double norm_temp2, double x[], double z[]) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= NA) return;
    x[j] = norm_temp2 * z[j];
}