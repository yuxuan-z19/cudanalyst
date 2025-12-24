#include "ft.cuh"

/*
 * ----------------------------------------------------------------------
 * y0[z][x][y] = x_in[z][y][x]
 *
 * y0[y + x*NY + z*NX*NY] = x_in[x + y*NX + z*NX*NY]
 * ----------------------------------------------------------------------
 */
__global__ void cffts1_gpu_kernel_1(dcomplex x_in[], dcomplex y0[]) {
    int x_y_z = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_y_z >= (NX * NY * NZ)) return;
    int x = x_y_z % NX;
    int y = (x_y_z / NX) % NY;
    int z = x_y_z / (NX * NY);
    y0[y + (x * NY) + (z * NX * NY)].real = x_in[x_y_z].real;
    y0[y + (x * NY) + (z * NX * NY)].imag = x_in[x_y_z].imag;
}

/*
 * ----------------------------------------------------------------------
 * pattern = j + variable*NY + k*NX*NY | variable is i and transforms x axis
 * ----------------------------------------------------------------------
 */
__global__ void cffts1_gpu_kernel_2(const int is, dcomplex gty1[],
                                    dcomplex gty2[], dcomplex u_device[]) {
    int y_z = blockIdx.x * blockDim.x + threadIdx.x;
    if (y_z >= (NY * NZ)) return;

    int j, k;
    int l, j1, i1, k1;
    int n1, li, lj, lk, ku, i11, i12, i21, i22;

    j = y_z % NY;        /* j = y */
    k = (y_z / NY) % NZ; /* k = z */

    const int logd1 = ilog2_device(NX);

    double uu1_real, x11_real, x21_real;
    double uu1_imag, x11_imag, x21_imag;
    double uu2_real, x12_real, x22_real;
    double uu2_imag, x12_imag, x22_imag;
    double temp_real, temp2_real;
    double temp_imag, temp2_imag;

    for (l = 1; l <= logd1; l += 2) {
        n1 = NX / 2;
        lk = 1 << (l - 1);
        li = 1 << (logd1 - l);
        lj = 2 * lk;
        ku = li;
        for (i1 = 0; i1 <= li - 1; i1++) {
            for (k1 = 0; k1 <= lk - 1; k1++) {
                i11 = i1 * lk;
                i12 = i11 + n1;
                i21 = i1 * lj;
                i22 = i21 + lk;

                uu1_real = u_device[ku + i1].real;
                uu1_imag = is * u_device[ku + i1].imag;

                /* gty1[k][i11+k1][j] */
                x11_real = gty1[j + (i11 + k1) * NY + k * NX * NY].real;
                x11_imag = gty1[j + (i11 + k1) * NY + k * NX * NY].imag;

                /* gty1[k][i12+k1][j] */
                x21_real = gty1[j + (i12 + k1) * NY + k * NX * NY].real;
                x21_imag = gty1[j + (i12 + k1) * NY + k * NX * NY].imag;

                /* gty2[k][i21+k1][j] */
                gty2[j + (i21 + k1) * NY + k * NX * NY].real =
                    x11_real + x21_real;
                gty2[j + (i21 + k1) * NY + k * NX * NY].imag =
                    x11_imag + x21_imag;

                temp_real = x11_real - x21_real;
                temp_imag = x11_imag - x21_imag;

                /* gty2[k][i22+k1][j] */
                gty2[j + (i22 + k1) * NY + k * NX * NY].real =
                    (uu1_real) * (temp_real) - (uu1_imag) * (temp_imag);
                gty2[j + (i22 + k1) * NY + k * NX * NY].imag =
                    (uu1_real) * (temp_imag) + (uu1_imag) * (temp_real);
            }
        }
        if (l == logd1) {
            for (j1 = 0; j1 < NX; j1++) {
                /* gty1[k][j1][j] */
                gty1[j + j1 * NY + k * NX * NY].real =
                    gty2[j + j1 * NY + k * NX * NY].real;
                gty1[j + j1 * NY + k * NX * NY].imag =
                    gty2[j + j1 * NY + k * NX * NY].imag;
            }
        } else {
            n1 = NX / 2;
            lk = 1 << (l + 1 - 1);
            li = 1 << (logd1 - (l + 1));
            lj = 2 * lk;
            ku = li;
            for (i1 = 0; i1 <= li - 1; i1++) {
                for (k1 = 0; k1 <= lk - 1; k1++) {
                    i11 = i1 * lk;
                    i12 = i11 + n1;
                    i21 = i1 * lj;
                    i22 = i21 + lk;

                    uu2_real = u_device[ku + i1].real;
                    uu2_imag = is * u_device[ku + i1].imag;

                    /* gty2[k][i11+k1][j] */
                    x12_real = gty2[j + (i11 + k1) * NY + k * NX * NY].real;
                    x12_imag = gty2[j + (i11 + k1) * NY + k * NX * NY].imag;

                    /* gty2[k][i12+k1][j] */
                    x22_real = gty2[j + (i12 + k1) * NY + k * NX * NY].real;
                    x22_imag = gty2[j + (i12 + k1) * NY + k * NX * NY].imag;

                    /* gty2[k][i21+k1][j] */
                    gty1[j + (i21 + k1) * NY + k * NX * NY].real =
                        x12_real + x22_real;
                    gty1[j + (i21 + k1) * NY + k * NX * NY].imag =
                        x12_imag + x22_imag;

                    temp2_real = x12_real - x22_real;
                    temp2_imag = x12_imag - x22_imag;

                    /* gty1[k][i22+k1][j] */
                    gty1[j + (i22 + k1) * NY + k * NX * NY].real =
                        (uu2_real) * (temp2_real) - (uu2_imag) * (temp2_imag);
                    gty1[j + (i22 + k1) * NY + k * NX * NY].imag =
                        (uu2_real) * (temp2_imag) + (uu2_imag) * (temp2_real);
                }
            }
        }
    }
}

/*
 * ----------------------------------------------------------------------
 * x_out[z][y][x] = y0[z][x][y]
 *
 * x_out[x + y*NX + z*NX*NY] = y0[y + x*NY + z*NX*NY]
 * ----------------------------------------------------------------------
 */
__global__ void cffts1_gpu_kernel_3(dcomplex x_out[], dcomplex y0[]) {
    int x_y_z = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_y_z >= (NX * NY * NZ)) return;
    int x = x_y_z % NX;
    int y = (x_y_z / NX) % NY;
    int z = x_y_z / (NX * NY);
    x_out[x_y_z].real = y0[y + (x * NY) + (z * NX * NY)].real;
    x_out[x_y_z].imag = y0[y + (x * NY) + (z * NX * NY)].imag;
}

/*
 * ----------------------------------------------------------------------
 * y0[z][y][x] = x_in[z][y][x]
 *
 * y0[x + y*NX + z*NX*NY]  = x_in[x + y*NX + z*NX*NY]
 * ----------------------------------------------------------------------
 */
__global__ void cffts2_gpu_kernel_1(dcomplex x_in[], dcomplex y0[]) {
    int x_y_z = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_y_z >= (NX * NY * NZ)) return;
    y0[x_y_z].real = x_in[x_y_z].real;
    y0[x_y_z].imag = x_in[x_y_z].imag;
}

/*
 * ----------------------------------------------------------------------
 * pattern = i + variable*NX + k*NX*NY | variable is j and transforms y axis
 * ----------------------------------------------------------------------
 */
__global__ void cffts2_gpu_kernel_2(const int is, dcomplex gty1[],
                                    dcomplex gty2[], dcomplex u_device[]) {
    int x_z = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_z >= (NX * NZ)) return;

    int i, k;
    int l, j1, i1, k1;
    int n1, li, lj, lk, ku, i11, i12, i21, i22;

    i = x_z % NX;        /* i = x */
    k = (x_z / NX) % NZ; /* k = z */

    const int logd2 = ilog2_device(NY);

    double uu1_real, x11_real, x21_real;
    double uu1_imag, x11_imag, x21_imag;
    double uu2_real, x12_real, x22_real;
    double uu2_imag, x12_imag, x22_imag;
    double temp_real, temp2_real;
    double temp_imag, temp2_imag;

    for (l = 1; l <= logd2; l += 2) {
        n1 = NY / 2;
        lk = 1 << (l - 1);
        li = 1 << (logd2 - l);
        lj = 2 * lk;
        ku = li;
        for (i1 = 0; i1 <= li - 1; i1++) {
            for (k1 = 0; k1 <= lk - 1; k1++) {
                i11 = i1 * lk;
                i12 = i11 + n1;
                i21 = i1 * lj;
                i22 = i21 + lk;

                uu1_real = u_device[ku + i1].real;
                uu1_imag = is * u_device[ku + i1].imag;

                /* gty1[k][i11+k1][i] */
                x11_real = gty1[i + (i11 + k1) * NX + k * NX * NY].real;
                x11_imag = gty1[i + (i11 + k1) * NX + k * NX * NY].imag;

                /* gty1[k][i12+k1][i] */
                x21_real = gty1[i + (i12 + k1) * NX + k * NX * NY].real;
                x21_imag = gty1[i + (i12 + k1) * NX + k * NX * NY].imag;

                /* gty2[k][i21+k1][i] */
                gty2[i + (i21 + k1) * NX + k * NX * NY].real =
                    x11_real + x21_real;
                gty2[i + (i21 + k1) * NX + k * NX * NY].imag =
                    x11_imag + x21_imag;

                temp_real = x11_real - x21_real;
                temp_imag = x11_imag - x21_imag;

                /* gty2[k][i22+k1][i] */
                gty2[i + (i22 + k1) * NX + k * NX * NY].real =
                    (uu1_real) * (temp_real) - (uu1_imag) * (temp_imag);
                gty2[i + (i22 + k1) * NX + k * NX * NY].imag =
                    (uu1_real) * (temp_imag) + (uu1_imag) * (temp_real);
            }
        }
        if (l == logd2) {
            for (j1 = 0; j1 < NY; j1++) {
                /* gty1[k][j1][i] */
                gty1[i + j1 * NX + k * NX * NY].real =
                    gty2[i + j1 * NX + k * NX * NY].real;
                gty1[i + j1 * NX + k * NX * NY].imag =
                    gty2[i + j1 * NX + k * NX * NY].imag;
            }
        } else {
            n1 = NY / 2;
            lk = 1 << (l + 1 - 1);
            li = 1 << (logd2 - (l + 1));
            lj = 2 * lk;
            ku = li;
            for (i1 = 0; i1 <= li - 1; i1++) {
                for (k1 = 0; k1 <= lk - 1; k1++) {
                    i11 = i1 * lk;
                    i12 = i11 + n1;
                    i21 = i1 * lj;
                    i22 = i21 + lk;

                    uu2_real = u_device[ku + i1].real;
                    uu2_imag = is * u_device[ku + i1].imag;

                    /* gty2[k][i11+k1][i] */
                    x12_real = gty2[i + (i11 + k1) * NX + k * NX * NY].real;
                    x12_imag = gty2[i + (i11 + k1) * NX + k * NX * NY].imag;

                    /* gty2[k][i12+k1][i] */
                    x22_real = gty2[i + (i12 + k1) * NX + k * NX * NY].real;
                    x22_imag = gty2[i + (i12 + k1) * NX + k * NX * NY].imag;

                    /* gty1[k][i21+k1][i] */
                    gty1[i + (i21 + k1) * NX + k * NX * NY].real =
                        x12_real + x22_real;
                    gty1[i + (i21 + k1) * NX + k * NX * NY].imag =
                        x12_imag + x22_imag;

                    temp2_real = x12_real - x22_real;
                    temp2_imag = x12_imag - x22_imag;

                    /* gty1[k][i22+k1][i] */
                    gty1[i + (i22 + k1) * NX + k * NX * NY].real =
                        (uu2_real) * (temp2_real) - (uu2_imag) * (temp2_imag);
                    gty1[i + (i22 + k1) * NX + k * NX * NY].imag =
                        (uu2_real) * (temp2_imag) + (uu2_imag) * (temp2_real);
                }
            }
        }
    }
}

/*
 * ----------------------------------------------------------------------
 * x_out[z][y][x] = y0[z][y][x]
 *
 * x_out[x + y*NX + z*NX*NY] = y0[x + y*NX + z*NX*NY]
 * ----------------------------------------------------------------------
 */
__global__ void cffts2_gpu_kernel_3(dcomplex x_out[], dcomplex y0[]) {
    int x_y_z = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_y_z >= (NX * NY * NZ)) return;
    x_out[x_y_z].real = y0[x_y_z].real;
    x_out[x_y_z].imag = y0[x_y_z].imag;
}

/*
 * ----------------------------------------------------------------------
 * pattern = i + j*NX + variable*NX*NY | variable is z and transforms z axis
 *
 * index_arg = i + j*NX
 *
 * size_arg = NX*NY
 * ----------------------------------------------------------------------
 */
__device__ void cffts3_gpu_cfftz_device(const int is, int m, int n,
                                        dcomplex x[], dcomplex y[],
                                        dcomplex u_device[], int index_arg,
                                        int size_arg) {
    int j, l;
    /*
     * ---------------------------------------------------------------------
     * perform one variant of the Stockham FFT.
     * ---------------------------------------------------------------------
     */
    for (l = 1; l <= m; l += 2) {
        cffts3_gpu_fftz2_device(is, l, m, n, u_device, x, y, index_arg,
                                size_arg);
        if (l == m) break;
        cffts3_gpu_fftz2_device(is, l + 1, m, n, u_device, y, x, index_arg,
                                size_arg);
    }
    /*
     * ---------------------------------------------------------------------
     * copy Y to X.
     * ---------------------------------------------------------------------
     */
    if (m % 2 == 1) {
        for (j = 0; j < n; j++) {
            x[j * size_arg + index_arg].real = y[j * size_arg + index_arg].real;
            x[j * size_arg + index_arg].imag = y[j * size_arg + index_arg].imag;
        }
    }
}

/*
 * ----------------------------------------------------------------------
 * pattern = i + j*NX + variable*NX*NY | variable is z and transforms z axis
 *
 * index_arg = i + j*NX
 *
 * size_arg = NX*NY
 * ----------------------------------------------------------------------
 */
__device__ void cffts3_gpu_fftz2_device(const int is, int l, int m, int n,
                                        dcomplex u[], dcomplex x[],
                                        dcomplex y[], int index_arg,
                                        int size_arg) {
    int k, n1, li, lj, lk, ku, i, i11, i12, i21, i22;
    double x11real, x11imag;
    double x21real, x21imag;
    dcomplex u1;
    /*
     * ---------------------------------------------------------------------
     * set initial parameters.
     * ---------------------------------------------------------------------
     */
    n1 = n / 2;
    lk = 1 << (l - 1);
    li = 1 << (m - l);
    lj = 2 * lk;
    ku = li;
    for (i = 0; i < li; i++) {
        i11 = i * lk;
        i12 = i11 + n1;
        i21 = i * lj;
        i22 = i21 + lk;
        if (is >= 1) {
            u1.real = u[ku + i].real;
            u1.imag = u[ku + i].imag;
        } else {
            u1.real = u[ku + i].real;
            u1.imag = -u[ku + i].imag;
        }
        for (k = 0; k < lk; k++) {
            x11real = x[(i11 + k) * size_arg + index_arg].real;
            x11imag = x[(i11 + k) * size_arg + index_arg].imag;
            x21real = x[(i12 + k) * size_arg + index_arg].real;
            x21imag = x[(i12 + k) * size_arg + index_arg].imag;
            y[(i21 + k) * size_arg + index_arg].real = x11real + x21real;
            y[(i21 + k) * size_arg + index_arg].imag = x11imag + x21imag;
            y[(i22 + k) * size_arg + index_arg].real =
                u1.real * (x11real - x21real) - u1.imag * (x11imag - x21imag);
            y[(i22 + k) * size_arg + index_arg].imag =
                u1.real * (x11imag - x21imag) + u1.imag * (x11real - x21real);
        }
    }
}

/*
 * ----------------------------------------------------------------------
 * y0[z][y][x] = x_in[z][y][x]
 *
 * y0[x + y*NX + z*NX*NY]  = x_in[x + y*NX + z*NX*NY]
 * ----------------------------------------------------------------------
 */
__global__ void cffts3_gpu_kernel_1(dcomplex x_in[], dcomplex y0[]) {
    int x_y_z = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_y_z >= (NX * NY * NZ)) return;
    y0[x_y_z].real = x_in[x_y_z].real;
    y0[x_y_z].imag = x_in[x_y_z].imag;
}

/*
 * ----------------------------------------------------------------------
 * pattern = i + j*NX + variable*NX*NY | variable is z and transforms z axis
 * ----------------------------------------------------------------------
 */
__global__ void cffts3_gpu_kernel_2(const int is, dcomplex gty1[],
                                    dcomplex gty2[], dcomplex u_device[]) {
    int x_y = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_y >= (NX * NY)) return;
    cffts3_gpu_cfftz_device(is, ilog2_device(NZ), NZ, gty1, gty2, u_device,
                            x_y /* index_arg */, NX * NY /* size_arg */);
}

/*
 * ----------------------------------------------------------------------
 * x_out[z][y][x] = y0[z][y][x]
 *
 * x_out[x + y*NX + z*NX*NY] = y0[x + y*NX + z*NX*NY]
 * ----------------------------------------------------------------------
 */
__global__ void cffts3_gpu_kernel_3(dcomplex x_out[], dcomplex y0[]) {
    int x_y_z = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_y_z >= (NX * NY * NZ)) return;
    x_out[x_y_z].real = y0[x_y_z].real;
    x_out[x_y_z].imag = y0[x_y_z].imag;
}

__global__ void evolve_gpu_kernel(dcomplex u0[], dcomplex u1[],
                                  double twiddle[]) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id >= (NZ * NY * NX)) return;

    u0[thread_id] = dcomplex_mul2(u0[thread_id], twiddle[thread_id]);
    u1[thread_id] = u0[thread_id];
}

__global__ void checksum_gpu_kernel(int iteration, dcomplex u1[],
                                    dcomplex sums[]) {
    dcomplex* share_sums = (dcomplex*)(extern_share_data);
    int j = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
    int q, r, s;

    if (j <= CHECKSUM_TASKS) {
        q = j % NX;
        r = 3 * j % NY;
        s = 5 * j % NZ;
        share_sums[threadIdx.x] = u1[q + r * NX + s * NX * NY];
    } else
        share_sums[threadIdx.x] = dcomplex_create(0.0, 0.0);

    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i)
            share_sums[threadIdx.x] = dcomplex_add(share_sums[threadIdx.x],
                                                   share_sums[threadIdx.x + i]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        share_sums[0].real = share_sums[0].real / (double)(NTOTAL);
        atomicAdd(&sums[iteration].real, share_sums[0].real);
        share_sums[0].imag = share_sums[0].imag / (double)(NTOTAL);
        atomicAdd(&sums[iteration].imag, share_sums[0].imag);
    }
}
