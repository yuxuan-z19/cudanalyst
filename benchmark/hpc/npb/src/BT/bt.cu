/*
 * ------------------------------------------------------------------------------
 *
 * MIT License
 *
 * Copyright (c) 2021 Parallel Applications Modelling Group - GMAP
 *      GMAP website: https://gmap.pucrs.br
 *
 * Pontifical Catholic University of Rio Grande do Sul (PUCRS)
 * Av. Ipiranga, 6681, Porto Alegre - Brazil, 90619-900
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ------------------------------------------------------------------------------
 *
 * The original NPB 3.4 version was written in Fortran and belongs to:
 *      http://www.nas.nasa.gov/Software/NPB/
 *
 * Authors of the Fortran code:
 *      R. Van der Wijngaart
 *      T. Harris
 *      M. Yarrow
 *      H. Jin
 *
 * ------------------------------------------------------------------------------
 *
 * The serial C++ version is a translation of the original NPB 3.4
 * Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER
 *
 * Authors of the C++ code:
 *      Dalvan Griebler <dalvangriebler@gmail.com>
 *      Gabriell Araujo <hexenoften@gmail.com>
 *      Júnior Löff <loffjh@gmail.com>
 *
 * ------------------------------------------------------------------------------
 *
 * The CUDA version is a parallel implementation of the serial C++ version
 * CUDA version: https://github.com/GMAP/NPB-GPU/tree/master/CUDA
 *
 * Authors of the CUDA code:
 *      Gabriell Araujo <hexenoften@gmail.com>
 *
 * ------------------------------------------------------------------------------
 */

#include <cuda.h>

#include "bt.cuh"

namespace constants_device {
__constant__ double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, dx1, dx2, dx3,
    dx4, dx5, dy1, dy2, dy3, dy4, dy5, dz1, dz2, dz3, dz4, dz5, dssp, dt, dxmax,
    dymax, dzmax, xxcon1, xxcon2, xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1,
    dx3tx1, dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4, yycon5, dy1ty1,
    dy2ty1, dy3ty1, dy4ty1, dy5ty1, zzcon1, zzcon2, zzcon3, zzcon4, zzcon5,
    dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, dIMAXm1, dJMAXm1, dKMAXm1, c1c2,
    c1c5, c3c4, c1345, coKMAX1, c1, c2, c3, c4, c5, c4dssp, c5dssp, dtdssp,
    dttx1, dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, c2dtty1, c2dttz1, comz1,
    comz4, comz5, comz6, c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16,
    ce[5][13];
}

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
double us[KMAX][JMAXP + 1][IMAXP + 1];
double vs[KMAX][JMAXP + 1][IMAXP + 1];
double ws[KMAX][JMAXP + 1][IMAXP + 1];
double qs[KMAX][JMAXP + 1][IMAXP + 1];
double rho_i[KMAX][JMAXP + 1][IMAXP + 1];
double square[KMAX][JMAXP + 1][IMAXP + 1];
double forcing[KMAX][JMAXP + 1][IMAXP + 1][5];
double u[KMAX][JMAXP + 1][IMAXP + 1][5];
double rhs[KMAX][JMAXP + 1][IMAXP + 1][5];
double cuf[PROBLEM_SIZE + 1];
double q[PROBLEM_SIZE + 1];
double ue[PROBLEM_SIZE + 1][5];
double buf[PROBLEM_SIZE + 1][5];
double fjac[PROBLEM_SIZE + 1][5][5];
double njac[PROBLEM_SIZE + 1][5][5];
double lhs[PROBLEM_SIZE + 1][3][5][5];
double ce[5][13];
#else
static double (*us)[JMAXP + 1][IMAXP + 1] = (double (*)[JMAXP + 1][IMAXP + 1])
    malloc(sizeof(double) * ((KMAX) * (JMAXP + 1) * (IMAXP + 1)));
static double (*vs)[JMAXP + 1][IMAXP + 1] = (double (*)[JMAXP + 1][IMAXP + 1])
    malloc(sizeof(double) * ((KMAX) * (JMAXP + 1) * (IMAXP + 1)));
static double (*ws)[JMAXP + 1][IMAXP + 1] = (double (*)[JMAXP + 1][IMAXP + 1])
    malloc(sizeof(double) * ((KMAX) * (JMAXP + 1) * (IMAXP + 1)));
static double (*qs)[JMAXP + 1][IMAXP + 1] = (double (*)[JMAXP + 1][IMAXP + 1])
    malloc(sizeof(double) * ((KMAX) * (JMAXP + 1) * (IMAXP + 1)));
static double (*rho_i)[JMAXP + 1][IMAXP + 1] =
    (double (*)[JMAXP + 1][IMAXP + 1])
        malloc(sizeof(double) * ((KMAX) * (JMAXP + 1) * (IMAXP + 1)));
static double (*square)[JMAXP + 1][IMAXP + 1] =
    (double (*)[JMAXP + 1][IMAXP + 1])
        malloc(sizeof(double) * ((KMAX) * (JMAXP + 1) * (IMAXP + 1)));
static double (*forcing)[JMAXP + 1][IMAXP + 1][5] =
    (double (*)[JMAXP + 1][IMAXP + 1][5])
        malloc(sizeof(double) * ((KMAX) * (JMAXP + 1) * (IMAXP + 1) * (5)));
static double (*u)[JMAXP + 1][IMAXP + 1][5] =
    (double (*)[JMAXP + 1][IMAXP + 1][5])
        malloc(sizeof(double) * ((KMAX) * (JMAXP + 1) * (IMAXP + 1) * (5)));
static double (*rhs)[JMAXP + 1][IMAXP + 1][5] =
    (double (*)[JMAXP + 1][IMAXP + 1][5])
        malloc(sizeof(double) * ((KMAX) * (JMAXP + 1) * (IMAXP + 1) * (5)));
static double(*cuf) = (double*)malloc(sizeof(double) * (PROBLEM_SIZE + 1));
static double(*q) = (double*)malloc(sizeof(double) * (PROBLEM_SIZE + 1));
static double (*ue)[5] = (double (*)[5])malloc(sizeof(double) *
                                               ((PROBLEM_SIZE + 1) * (5)));
static double (*buf)[5] = (double (*)[5])malloc(sizeof(double) *
                                                ((PROBLEM_SIZE + 1) * (5)));
static double (*fjac)[5][5] = (double (*)[5][5])
    malloc(sizeof(double) * ((PROBLEM_SIZE + 1) * (5) * (5)));
static double (*njac)[5][5] = (double (*)[5][5])
    malloc(sizeof(double) * ((PROBLEM_SIZE + 1) * (5) * (5)));
double (*lhs)[3][5][5] = (double (*)[3][5][5])
    malloc(sizeof(double) * ((PROBLEM_SIZE + 1) * (3) * (5) * (5)));
static double (*ce)[13] = (double (*)[13])malloc(sizeof(double) * ((5) * (13)));
#endif
int grid_points[3];
/* gpu variables */
static double* us_device;
static double* vs_device;
static double* ws_device;
static double* qs_device;
static double* rho_i_device;
static double* square_device;
static double* forcing_device;
static double* u_device;
static double* rhs_device;
static double* lhsA_device;
static double* lhsB_device;
static double* lhsC_device;
static size_t size_u;
static size_t size_forcing;
static size_t size_rhs;
static size_t size_qs;
static size_t size_square;
static size_t size_rho_i;
/* new */
static int THREADS_PER_BLOCK_ON_ADD;
static int THREADS_PER_BLOCK_ON_RHS_1;
static int THREADS_PER_BLOCK_ON_RHS_2;
static int THREADS_PER_BLOCK_ON_RHS_3;
static int THREADS_PER_BLOCK_ON_RHS_4;
static int THREADS_PER_BLOCK_ON_RHS_5;
static int THREADS_PER_BLOCK_ON_RHS_6;
static int THREADS_PER_BLOCK_ON_RHS_7;
static int THREADS_PER_BLOCK_ON_RHS_8;
static int THREADS_PER_BLOCK_ON_RHS_9;
static int THREADS_PER_BLOCK_ON_X_SOLVE_1;
static int THREADS_PER_BLOCK_ON_X_SOLVE_2;
static int THREADS_PER_BLOCK_ON_X_SOLVE_3;
static int THREADS_PER_BLOCK_ON_Y_SOLVE_1;
static int THREADS_PER_BLOCK_ON_Y_SOLVE_2;
static int THREADS_PER_BLOCK_ON_Y_SOLVE_3;
static int THREADS_PER_BLOCK_ON_Z_SOLVE_1;
static int THREADS_PER_BLOCK_ON_Z_SOLVE_2;
static int THREADS_PER_BLOCK_ON_Z_SOLVE_3;
/* old */
static int THREADS_PER_BLOCK_ON_EXACT_RHS_1;
static int THREADS_PER_BLOCK_ON_EXACT_RHS_2;
static int THREADS_PER_BLOCK_ON_EXACT_RHS_3;
static int THREADS_PER_BLOCK_ON_EXACT_RHS_4;
static int THREADS_PER_BLOCK_ON_ERROR_NORM_1;
static int THREADS_PER_BLOCK_ON_ERROR_NORM_2;
static int THREADS_PER_BLOCK_ON_INITIALIZE;
static int THREADS_PER_BLOCK_ON_RHS_NORM_1;
static int THREADS_PER_BLOCK_ON_RHS_NORM_2;
int gpu_device_id;
int total_devices;
cudaDeviceProp gpu_device_properties;

/* constants */
double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, dx1, dx2, dx3, dx4, dx5,
    dy1, dy2, dy3, dy4, dy5, dz1, dz2, dz3, dz4, dz5, dssp, dt, dxmax, dymax,
    dzmax, xxcon1, xxcon2, xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1,
    dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4, yycon5, dy1ty1, dy2ty1,
    dy3ty1, dy4ty1, dy5ty1, zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1,
    dz2tz1, dz3tz1, dz4tz1, dz5tz1, dIMAXm1, dJMAXm1, dKMAXm1, c1c2, c1c5, c3c4,
    c1345, coKMAX1, c1, c2, c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1, dttx2,
    dtty1, dtty2, dttz1, dttz2, c2dttx1, c2dtty1, c2dttz1, comz1, comz4, comz5,
    comz6, c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16;

/* bt */
int main(int argc, char* argv[]) {
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
    printf(
        " DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION "
        "mode on\n");
#endif
#if defined(PROFILING)
    printf(" PROFILING mode on\n");
#endif
    int niter, step;
    double navg, mflops, n3;
    double tmax;
    boolean verified;
    char class_npb;

    /*
     * ---------------------------------------------------------------------
     * root node reads input file (if it exists) else takes
     * defaults from parameters
     * ---------------------------------------------------------------------
     */
    FILE* fp;
    if ((fp = fopen("inputbt.data", "r")) != NULL) {
        int avoid_warning;
        // printf(" Readi_gpung from input file inputbt.data\n");
        avoid_warning = fscanf(fp, "%d", &niter);
        while (fgetc(fp) != '\n');
        avoid_warning = fscanf(fp, "%lf", &dt);
        while (fgetc(fp) != '\n');
        avoid_warning = fscanf(fp, "%d%d%d\n", &grid_points[0], &grid_points[1],
                               &grid_points[2]);
        avoid_warning++;
        fclose(fp);
    } else {
        // printf(" No input file inputbt.data. Using compiled defaults\n");
        niter = NITER_DEFAULT;
        dt = DT_DEFAULT;
        grid_points[0] = PROBLEM_SIZE;
        grid_points[1] = PROBLEM_SIZE;
        grid_points[2] = PROBLEM_SIZE;
    }
    timer_clear(PROFILING_TOTAL_TIME);
#if defined(PROFILING)
    /* new */
    timer_clear(PROFILING_ADD);
    timer_clear(PROFILING_RHS_1);
    timer_clear(PROFILING_RHS_2);
    timer_clear(PROFILING_RHS_3);
    timer_clear(PROFILING_RHS_4);
    timer_clear(PROFILING_RHS_5);
    timer_clear(PROFILING_RHS_6);
    timer_clear(PROFILING_RHS_7);
    timer_clear(PROFILING_RHS_8);
    timer_clear(PROFILING_RHS_9);
    timer_clear(PROFILING_X_SOLVE_1);
    timer_clear(PROFILING_X_SOLVE_2);
    timer_clear(PROFILING_X_SOLVE_3);
    timer_clear(PROFILING_Y_SOLVE_1);
    timer_clear(PROFILING_Y_SOLVE_2);
    timer_clear(PROFILING_Y_SOLVE_3);
    timer_clear(PROFILING_Z_SOLVE_1);
    timer_clear(PROFILING_Z_SOLVE_2);
    timer_clear(PROFILING_Z_SOLVE_3);
    /* old */
    timer_clear(PROFILING_EXACT_RHS_1);
    timer_clear(PROFILING_EXACT_RHS_2);
    timer_clear(PROFILING_EXACT_RHS_3);
    timer_clear(PROFILING_EXACT_RHS_4);
    timer_clear(PROFILING_ERROR_NORM_1);
    timer_clear(PROFILING_ERROR_NORM_2);
    timer_clear(PROFILING_INITIALIZE);
    timer_clear(PROFILING_RHS_NORM_1);
    timer_clear(PROFILING_RHS_NORM_2);
#endif
    // printf(
    //     "\n\n NAS Parallel Benchmarks 4.1 CUDA C++ version - BT
    //     Benchmark\n\n");
    // printf(" Size: %4dx%4dx%4d\n", grid_points[0], grid_points[1],
    //        grid_points[2]);
    // printf(" Iterations: %4d    dt: %10.6f\n", niter, dt);
    // printf("\n");
    if ((grid_points[0] > IMAX) || (grid_points[1] > JMAX) ||
        (grid_points[2] > KMAX)) {
        // printf(" %d, %d, %d\n", grid_points[0], grid_points[1],
        // grid_points[2]); printf(" Problem size too big for compiled array
        // sizes\n");
        return 0;
    }

    setup_gpu();
    set_constants();
    initialize();
    exact_rhs();

    /*
     * ---------------------------------------------------------------------
     * do one time step to touch all code, and reinitialize_gpu
     * ---------------------------------------------------------------------
     */
    cudaMemcpy(u_device, u, size_u, cudaMemcpyHostToDevice);
    cudaMemcpy(forcing_device, forcing, size_forcing, cudaMemcpyHostToDevice);

    adi_gpu();

    cudaMemcpy(qs, qs_device, size_qs, cudaMemcpyDeviceToHost);
    cudaMemcpy(square, square_device, size_square, cudaMemcpyDeviceToHost);
    cudaMemcpy(rho_i, rho_i_device, size_rho_i, cudaMemcpyDeviceToHost);
    cudaMemcpy(rhs, rhs_device, size_rhs, cudaMemcpyDeviceToHost);
    cudaMemcpy(u, u_device, size_u, cudaMemcpyDeviceToHost);

    initialize();

    cudaMemcpy(u_device, u, size_u, cudaMemcpyHostToDevice);
    cudaMemcpy(forcing_device, forcing, size_forcing, cudaMemcpyHostToDevice);

    timer_clear();

    timer_start();

    for (step = 1; step <= niter; step++) {
        // if ((step % 20) == 0 || step == 1) {
        //     printf(" Time step %4d\n", step);
        // }
        adi_gpu();
    }

    timer_stop();
    tmax = timer_read();

    cudaMemcpy(rhs, rhs_device, size_rhs, cudaMemcpyDeviceToHost);
    cudaMemcpy(u, u_device, size_u, cudaMemcpyDeviceToHost);

    verify(niter, &class_npb, &verified);

    n3 = 1.0 * grid_points[0] * grid_points[1] * grid_points[2];
    navg = (grid_points[0] + grid_points[1] + grid_points[2]) / 3.0;
    if (tmax != 0.0) {
        mflops = 1.0e-6 * (double)niter *
                 (3478.8 * n3 - 17655.7 * (navg * navg) + 28023.7 * navg) /
                 tmax;
    } else {
        mflops = 0.0;
    }

    //     char gpu_config[512];
    //     char gpu_config_string[4096];
    // #if defined(PROFILING)
    //     /* new */
    //     sprintf(gpu_config, "%5s\t%25s\t%25s\t%25s\n", "GPU Kernel",
    //             "Threads Per Block", "Time in Seconds", "Time in
    //             Percentage");
    //     strcpy(gpu_config_string, gpu_config);
    //     sprintf(
    //         gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-add",
    //         THREADS_PER_BLOCK_ON_ADD, timer_read(PROFILING_ADD),
    //         (timer_read(PROFILING_ADD) * 100 /
    //         timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(
    //         gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-rhs-1",
    //         THREADS_PER_BLOCK_ON_RHS_1, timer_read(PROFILING_RHS_1),
    //         (timer_read(PROFILING_RHS_1) * 100 /
    //         timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(
    //         gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-rhs-2",
    //         THREADS_PER_BLOCK_ON_RHS_2, timer_read(PROFILING_RHS_2),
    //         (timer_read(PROFILING_RHS_2) * 100 /
    //         timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(
    //         gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-rhs-3",
    //         THREADS_PER_BLOCK_ON_RHS_3, timer_read(PROFILING_RHS_3),
    //         (timer_read(PROFILING_RHS_3) * 100 /
    //         timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(
    //         gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-rhs-4",
    //         THREADS_PER_BLOCK_ON_RHS_4, timer_read(PROFILING_RHS_4),
    //         (timer_read(PROFILING_RHS_4) * 100 /
    //         timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(
    //         gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-rhs-5",
    //         THREADS_PER_BLOCK_ON_RHS_5, timer_read(PROFILING_RHS_5),
    //         (timer_read(PROFILING_RHS_5) * 100 /
    //         timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(
    //         gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-rhs-6",
    //         THREADS_PER_BLOCK_ON_RHS_6, timer_read(PROFILING_RHS_6),
    //         (timer_read(PROFILING_RHS_6) * 100 /
    //         timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(
    //         gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-rhs-7",
    //         THREADS_PER_BLOCK_ON_RHS_7, timer_read(PROFILING_RHS_7),
    //         (timer_read(PROFILING_RHS_7) * 100 /
    //         timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(
    //         gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-rhs-8",
    //         THREADS_PER_BLOCK_ON_RHS_8, timer_read(PROFILING_RHS_8),
    //         (timer_read(PROFILING_RHS_8) * 100 /
    //         timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(
    //         gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-rhs-9",
    //         THREADS_PER_BLOCK_ON_RHS_9, timer_read(PROFILING_RHS_9),
    //         (timer_read(PROFILING_RHS_9) * 100 /
    //         timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-x-solve-1",
    //             THREADS_PER_BLOCK_ON_X_SOLVE_1,
    //             timer_read(PROFILING_X_SOLVE_1),
    //             (timer_read(PROFILING_X_SOLVE_1) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-x-solve-2",
    //             THREADS_PER_BLOCK_ON_X_SOLVE_2,
    //             timer_read(PROFILING_X_SOLVE_2),
    //             (timer_read(PROFILING_X_SOLVE_2) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-x-solve-3",
    //             THREADS_PER_BLOCK_ON_X_SOLVE_3,
    //             timer_read(PROFILING_X_SOLVE_3),
    //             (timer_read(PROFILING_X_SOLVE_3) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-y-solve-1",
    //             THREADS_PER_BLOCK_ON_Y_SOLVE_1,
    //             timer_read(PROFILING_Y_SOLVE_1),
    //             (timer_read(PROFILING_Y_SOLVE_1) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-y-solve-2",
    //             THREADS_PER_BLOCK_ON_Y_SOLVE_2,
    //             timer_read(PROFILING_Y_SOLVE_2),
    //             (timer_read(PROFILING_Y_SOLVE_2) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-y-solve-3",
    //             THREADS_PER_BLOCK_ON_Y_SOLVE_3,
    //             timer_read(PROFILING_Y_SOLVE_3),
    //             (timer_read(PROFILING_Y_SOLVE_3) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-z-solve-1",
    //             THREADS_PER_BLOCK_ON_Z_SOLVE_1,
    //             timer_read(PROFILING_Z_SOLVE_1),
    //             (timer_read(PROFILING_Z_SOLVE_1) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-z-solve-2",
    //             THREADS_PER_BLOCK_ON_Z_SOLVE_2,
    //             timer_read(PROFILING_Z_SOLVE_2),
    //             (timer_read(PROFILING_Z_SOLVE_2) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-z-solve-3",
    //             THREADS_PER_BLOCK_ON_Z_SOLVE_3,
    //             timer_read(PROFILING_Z_SOLVE_3),
    //             (timer_read(PROFILING_Z_SOLVE_3) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     /* old */
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", "
    //     bt-exact-rhs-1",
    //             THREADS_PER_BLOCK_ON_EXACT_RHS_1,
    //             timer_read(PROFILING_EXACT_RHS_1),
    //             (timer_read(PROFILING_EXACT_RHS_1) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", "
    //     bt-exact-rhs-2",
    //             THREADS_PER_BLOCK_ON_EXACT_RHS_2,
    //             timer_read(PROFILING_EXACT_RHS_2),
    //             (timer_read(PROFILING_EXACT_RHS_2) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", "
    //     bt-exact-rhs-3",
    //             THREADS_PER_BLOCK_ON_EXACT_RHS_3,
    //             timer_read(PROFILING_EXACT_RHS_3),
    //             (timer_read(PROFILING_EXACT_RHS_3) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", "
    //     bt-exact-rhs-4",
    //             THREADS_PER_BLOCK_ON_EXACT_RHS_4,
    //             timer_read(PROFILING_EXACT_RHS_4),
    //             (timer_read(PROFILING_EXACT_RHS_4) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", "
    //     bt-error-norm-1",
    //             THREADS_PER_BLOCK_ON_ERROR_NORM_1,
    //             timer_read(PROFILING_ERROR_NORM_1),
    //             (timer_read(PROFILING_ERROR_NORM_1) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", "
    //     bt-error-norm-2",
    //             THREADS_PER_BLOCK_ON_ERROR_NORM_2,
    //             timer_read(PROFILING_ERROR_NORM_2),
    //             (timer_read(PROFILING_ERROR_NORM_2) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-initialize",
    //             THREADS_PER_BLOCK_ON_INITIALIZE,
    //             timer_read(PROFILING_INITIALIZE),
    //             (timer_read(PROFILING_INITIALIZE) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-rhs-norm-1",
    //             THREADS_PER_BLOCK_ON_RHS_NORM_1,
    //             timer_read(PROFILING_RHS_NORM_1),
    //             (timer_read(PROFILING_RHS_NORM_1) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\t%25f\t%24.2f%%\n", " bt-rhs-norm-2",
    //             THREADS_PER_BLOCK_ON_RHS_NORM_2,
    //             timer_read(PROFILING_RHS_NORM_2),
    //             (timer_read(PROFILING_RHS_NORM_2) * 100 /
    //              timer_read(PROFILING_TOTAL_TIME)));
    //     strcat(gpu_config_string, gpu_config);
    // #else
    //     /* new */
    //     sprintf(gpu_config, "%5s\t%25s\n", "GPU Kernel", "Threads Per
    //     Block"); strcpy(gpu_config_string, gpu_config); sprintf(gpu_config,
    //     "%29s\t%25d\n", " bt-add", THREADS_PER_BLOCK_ON_ADD);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-rhs-1",
    //             THREADS_PER_BLOCK_ON_RHS_1);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-rhs-2",
    //             THREADS_PER_BLOCK_ON_RHS_2);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-rhs-3",
    //             THREADS_PER_BLOCK_ON_RHS_3);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-rhs-4",
    //             THREADS_PER_BLOCK_ON_RHS_4);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-rhs-5",
    //             THREADS_PER_BLOCK_ON_RHS_5);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-rhs-6",
    //             THREADS_PER_BLOCK_ON_RHS_6);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-rhs-7",
    //             THREADS_PER_BLOCK_ON_RHS_7);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-rhs-8",
    //             THREADS_PER_BLOCK_ON_RHS_8);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-rhs-9",
    //             THREADS_PER_BLOCK_ON_RHS_9);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-x-solve-1",
    //             THREADS_PER_BLOCK_ON_X_SOLVE_1);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-x-solve-2",
    //             THREADS_PER_BLOCK_ON_X_SOLVE_2);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-x-solve-3",
    //             THREADS_PER_BLOCK_ON_X_SOLVE_3);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-y-solve-1",
    //             THREADS_PER_BLOCK_ON_Y_SOLVE_1);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-y-solve-2",
    //             THREADS_PER_BLOCK_ON_Y_SOLVE_2);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-y-solve-3",
    //             THREADS_PER_BLOCK_ON_Y_SOLVE_3);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-z-solve-1",
    //             THREADS_PER_BLOCK_ON_Z_SOLVE_1);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-z-solve-2",
    //             THREADS_PER_BLOCK_ON_Z_SOLVE_2);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-z-solve-3",
    //             THREADS_PER_BLOCK_ON_Z_SOLVE_3);
    //     strcat(gpu_config_string, gpu_config);
    //     /* old */
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-exact-rhs-1",
    //             THREADS_PER_BLOCK_ON_EXACT_RHS_1);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-exact-rhs-2",
    //             THREADS_PER_BLOCK_ON_EXACT_RHS_2);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-exact-rhs-3",
    //             THREADS_PER_BLOCK_ON_EXACT_RHS_3);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-exact-rhs-4",
    //             THREADS_PER_BLOCK_ON_EXACT_RHS_4);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-error-norm-1",
    //             THREADS_PER_BLOCK_ON_ERROR_NORM_1);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-error-norm-2",
    //             THREADS_PER_BLOCK_ON_ERROR_NORM_2);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-initialize",
    //             THREADS_PER_BLOCK_ON_INITIALIZE);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-rhs-norm-1",
    //             THREADS_PER_BLOCK_ON_RHS_NORM_1);
    //     strcat(gpu_config_string, gpu_config);
    //     sprintf(gpu_config, "%29s\t%25d\n", " bt-rhs-norm-2",
    //             THREADS_PER_BLOCK_ON_RHS_NORM_2);
    //     strcat(gpu_config_string, gpu_config);
    // #endif

    printf("verified=%d\nMops=%.6f\n", verified, mflops);

    // c_print_results(
    //     (char*)"BT", class_npb, grid_points[0], grid_points[1],
    //     grid_points[2], niter, tmax, mflops, (char*)"          floating
    //     point", verified, (char*)NPBVERSION, (char*)COMPILETIME,
    //     (char*)COMPILERVERSION, (char*)LIBVERSION, (char*)CPU_MODEL,
    //     (char*)gpu_device_properties.name, gpu_config_string, (char*)CS1,
    //     (char*)CS2, (char*)CS3, (char*)CS4, (char*)CS5, (char*)CS6,
    //     (char*)"(none)");

    release_gpu();

    return 0;
}

/*
 * ---------------------------------------------------------------------
 * addition of update to the vector u
 * ---------------------------------------------------------------------
 */
void add_gpu() {
    size_t amount_of_threads[3];
    size_t amount_of_work[3];

    amount_of_threads[2] = 1;
    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_ADD;

    amount_of_work[2] = grid_points[2] - 2;
    amount_of_work[1] = PROBLEM_SIZE;
    amount_of_work[0] = grid_points[0] - 2;
    amount_of_work[0] *= 5;

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    dim3 blockSize(amount_of_work[0] / amount_of_threads[0],
                   amount_of_work[1] / amount_of_threads[1],
                   amount_of_work[2] / amount_of_threads[2]);
    dim3 threadSize(amount_of_threads[0], amount_of_threads[1],
                    amount_of_threads[2]);

#if defined(PROFILING)
    timer_start(PROFILING_ADD);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    add_gpu_kernel<<<blockSize, threadSize, 0>>>(u_device, rhs_device);
#if defined(PROFILING)
    timer_stop(PROFILING_ADD);
#endif
}

void adi_gpu() {
    /*
     * ---------------------------------------------------------------------
     * compute the reciprocal of density, and the kinetic energy,
     * and the speed of sound.
     * ---------------------------------------------------------------------
     */
    compute_rhs_gpu();
    x_solve_gpu();
    y_solve_gpu();
    z_solve_gpu();
    add_gpu();
}

void compute_rhs_gpu() {
    int work_base = 0;
    int work_num_item = min(PROBLEM_SIZE, grid_points[2] - work_base);
    int copy_num_item = min(PROBLEM_SIZE, grid_points[2] - work_base);

    size_t amount_of_threads[3];
    size_t amount_of_work[3];
    dim3 blockSize;
    dim3 threadSize;

    amount_of_threads[2] = 1;
    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_RHS_1;

    amount_of_work[2] = (size_t)copy_num_item;
    amount_of_work[1] = (size_t)grid_points[1];
    amount_of_work[0] = (size_t)grid_points[0];

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = amount_of_work[2] / amount_of_threads[2];
    threadSize.z = amount_of_threads[2];

#if defined(PROFILING)
    timer_start(PROFILING_RHS_1);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    compute_rhs_gpu_kernel_1<<<blockSize, threadSize, 0>>>(
        rho_i_device, us_device, vs_device, ws_device, qs_device, square_device,
        u_device);
#if defined(PROFILING)
    timer_stop(PROFILING_RHS_1);
#endif

    /*
     * ---------------------------------------------------------------------
     * copy the exact forcing term to the right hand side; because
     * this forcing term is known, we can store it on the whole grid
     * including the boundary
     * ---------------------------------------------------------------------
     */
    amount_of_threads[2] = 1;
    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_RHS_2;

    amount_of_work[2] = work_num_item;
    amount_of_work[1] = (size_t)grid_points[1];
    amount_of_work[0] = (size_t)grid_points[0];
    amount_of_work[0] *= 5;

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = amount_of_work[2] / amount_of_threads[2];
    threadSize.z = amount_of_threads[2];

#if defined(PROFILING)
    timer_start(PROFILING_RHS_2);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    compute_rhs_gpu_kernel_2<<<blockSize, threadSize, 0>>>(rhs_device,
                                                           forcing_device);
#if defined(PROFILING)
    timer_stop(PROFILING_RHS_2);
#endif

    /*
     * ---------------------------------------------------------------------
     * compute xi-direction fluxes
     * ---------------------------------------------------------------------
     */
    amount_of_threads[2] = 1;
    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_RHS_3;

    amount_of_work[2] = work_num_item;
    amount_of_work[1] = grid_points[1] - 2;
    amount_of_work[0] = grid_points[0] - 2;

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = amount_of_work[2] / amount_of_threads[2];
    threadSize.z = amount_of_threads[2];

#if defined(PROFILING)
    timer_start(PROFILING_RHS_3);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    compute_rhs_gpu_kernel_3<<<blockSize, threadSize, 0>>>(
        us_device, vs_device, ws_device, qs_device, rho_i_device, square_device,
        u_device, rhs_device);
#if defined(PROFILING)
    timer_stop(PROFILING_RHS_3);
#endif

    amount_of_threads[2] = 1;
    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_RHS_4;

    amount_of_work[2] = work_num_item;
    amount_of_work[1] = grid_points[1] - 2;
    amount_of_work[0] = grid_points[0] - 2;
    amount_of_work[0] *= 5;

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = amount_of_work[2] / amount_of_threads[2];
    threadSize.z = amount_of_threads[2];

#if defined(PROFILING)
    timer_start(PROFILING_RHS_4);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    compute_rhs_gpu_kernel_4<<<blockSize, threadSize, 0>>>(u_device,
                                                           rhs_device);
#if defined(PROFILING)
    timer_stop(PROFILING_RHS_4);
#endif

    /*
     * ---------------------------------------------------------------------
     * compute eta-direction fluxes
     * ---------------------------------------------------------------------
     */
    amount_of_threads[2] = 1;
    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_RHS_5;

    amount_of_work[2] = work_num_item;
    amount_of_work[1] = grid_points[1] - 2;
    amount_of_work[0] = grid_points[0] - 2;

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = amount_of_work[2] / amount_of_threads[2];
    threadSize.z = amount_of_threads[2];

#if defined(PROFILING)
    timer_start(PROFILING_RHS_5);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    compute_rhs_gpu_kernel_5<<<blockSize, threadSize, 0>>>(
        us_device, vs_device, ws_device, qs_device, rho_i_device, square_device,
        u_device, rhs_device);
#if defined(PROFILING)
    timer_stop(PROFILING_RHS_5);
#endif

    amount_of_threads[2] = 1;
    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_RHS_6;

    amount_of_work[2] = work_num_item;
    amount_of_work[1] = grid_points[1] - 2;
    amount_of_work[0] = grid_points[0] - 2;
    amount_of_work[0] *= 5;

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = amount_of_work[2] / amount_of_threads[2];
    threadSize.z = amount_of_threads[2];

#if defined(PROFILING)
    timer_start(PROFILING_RHS_6);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    compute_rhs_gpu_kernel_6<<<blockSize, threadSize, 0>>>(u_device,
                                                           rhs_device);
#if defined(PROFILING)
    timer_stop(PROFILING_RHS_6);
#endif

    amount_of_threads[2] = 1;
    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_RHS_7;

    amount_of_work[2] = (size_t)work_num_item;
    amount_of_work[1] = (size_t)grid_points[1] - 2;
    amount_of_work[0] = (size_t)grid_points[0] - 2;

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = amount_of_work[2] / amount_of_threads[2];
    threadSize.z = amount_of_threads[2];

#if defined(PROFILING)
    timer_start(PROFILING_RHS_7);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    compute_rhs_gpu_kernel_7<<<blockSize, threadSize, 0>>>(
        us_device, vs_device, ws_device, qs_device, rho_i_device, square_device,
        u_device, rhs_device);
#if defined(PROFILING)
    timer_stop(PROFILING_RHS_7);
#endif

    amount_of_threads[2] = 1;
    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_RHS_8;

    amount_of_work[2] = work_num_item;
    amount_of_work[1] = grid_points[1] - 2;
    amount_of_work[0] = grid_points[0] - 2;
    amount_of_work[0] *= 5;

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = amount_of_work[2] / amount_of_threads[2];
    threadSize.z = amount_of_threads[2];

#if defined(PROFILING)
    timer_start(PROFILING_RHS_8);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    compute_rhs_gpu_kernel_8<<<blockSize, threadSize, 0>>>(u_device,
                                                           rhs_device);
#if defined(PROFILING)
    timer_stop(PROFILING_RHS_8);
#endif

    amount_of_threads[2] = 1;
    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_RHS_9;

    amount_of_work[2] = work_num_item;
    amount_of_work[1] = grid_points[1] - 2;
    amount_of_work[0] = grid_points[0] - 2;
    amount_of_work[0] *= 5;

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = amount_of_work[2] / amount_of_threads[2];
    threadSize.z = amount_of_threads[2];

#if defined(PROFILING)
    timer_start(PROFILING_RHS_9);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    compute_rhs_gpu_kernel_9<<<blockSize, threadSize, 0>>>(rhs_device);
#if defined(PROFILING)
    timer_stop(PROFILING_RHS_9);
#endif
}

/*
 * ---------------------------------------------------------------------
 * this function computes the norm of the difference between the
 * computed solution and the exact solution
 * ---------------------------------------------------------------------
 */
void error_norm(double rms[5]) {
    int i, j, k, m, d;
    double xi, eta, zeta, u_exact[5], add;

    for (m = 0; m < 5; m++) {
        rms[m] = 0.0;
    }
    for (k = 0; k <= grid_points[2] - 1; k++) {
        zeta = (double)(k)*dKMAXm1;
        for (j = 0; j <= grid_points[1] - 1; j++) {
            eta = (double)(j)*dJMAXm1;
            for (i = 0; i <= grid_points[0] - 1; i++) {
                xi = (double)(i)*dIMAXm1;
                exact_solution(xi, eta, zeta, u_exact);

                for (m = 0; m < 5; m++) {
                    add = u[k][j][i][m] - u_exact[m];
                    rms[m] = rms[m] + add * add;
                }
            }
        }
    }

    for (m = 0; m < 5; m++) {
        for (d = 0; d < 3; d++) {
            rms[m] = rms[m] / (double)(grid_points[d] - 2);
        }
        rms[m] = sqrt(rms[m]);
    }
}

/*
 * ---------------------------------------------------------------------
 * compute the right hand side based on exact solution
 * ---------------------------------------------------------------------
 * initialize
 * ---------------------------------------------------------------------
 */
void exact_rhs() {
    double dtemp[5], xi, eta, zeta, dtpp;
    int m, i, j, k, ip1, im1, jp1, jm1, km1, kp1;

    /*
     * ---------------------------------------------------------------------
     * initialize
     * ---------------------------------------------------------------------
     */
    for (k = 0; k <= grid_points[2] - 1; k++) {
        for (j = 0; j <= grid_points[1] - 1; j++) {
            for (i = 0; i <= grid_points[0] - 1; i++) {
                for (m = 0; m < 5; m++) {
                    forcing[k][j][i][m] = 0.0;
                }
            }
        }
    }

    /*
     * ---------------------------------------------------------------------
     * xi-direction flux differences
     * ---------------------------------------------------------------------
     */
    for (k = 1; k <= grid_points[2] - 2; k++) {
        zeta = (double)(k)*dKMAXm1;
        for (j = 1; j <= grid_points[1] - 2; j++) {
            eta = (double)(j)*dJMAXm1;

            for (i = 0; i <= grid_points[0] - 1; i++) {
                xi = (double)(i)*dIMAXm1;

                exact_solution(xi, eta, zeta, dtemp);
                for (m = 0; m < 5; m++) {
                    ue[i][m] = dtemp[m];
                }

                dtpp = 1.0 / dtemp[0];

                for (m = 1; m < 5; m++) {
                    buf[i][m] = dtpp * dtemp[m];
                }

                cuf[i] = buf[i][1] * buf[i][1];
                buf[i][0] =
                    cuf[i] + buf[i][2] * buf[i][2] + buf[i][3] * buf[i][3];
                q[i] = 0.5 * (buf[i][1] * ue[i][1] + buf[i][2] * ue[i][2] +
                              buf[i][3] * ue[i][3]);
            }

            for (i = 1; i <= grid_points[0] - 2; i++) {
                im1 = i - 1;
                ip1 = i + 1;

                forcing[k][j][i][0] =
                    forcing[k][j][i][0] - tx2 * (ue[ip1][1] - ue[im1][1]) +
                    dx1tx1 * (ue[ip1][0] - 2.0 * ue[i][0] + ue[im1][0]);

                forcing[k][j][i][1] =
                    forcing[k][j][i][1] -
                    tx2 * ((ue[ip1][1] * buf[ip1][1] +
                            c2 * (ue[ip1][4] - q[ip1])) -
                           (ue[im1][1] * buf[im1][1] +
                            c2 * (ue[im1][4] - q[im1]))) +
                    xxcon1 * (buf[ip1][1] - 2.0 * buf[i][1] + buf[im1][1]) +
                    dx2tx1 * (ue[ip1][1] - 2.0 * ue[i][1] + ue[im1][1]);

                forcing[k][j][i][2] =
                    forcing[k][j][i][2] -
                    tx2 *
                        (ue[ip1][2] * buf[ip1][1] - ue[im1][2] * buf[im1][1]) +
                    xxcon2 * (buf[ip1][2] - 2.0 * buf[i][2] + buf[im1][2]) +
                    dx3tx1 * (ue[ip1][2] - 2.0 * ue[i][2] + ue[im1][2]);

                forcing[k][j][i][3] =
                    forcing[k][j][i][3] -
                    tx2 *
                        (ue[ip1][3] * buf[ip1][1] - ue[im1][3] * buf[im1][1]) +
                    xxcon2 * (buf[ip1][3] - 2.0 * buf[i][3] + buf[im1][3]) +
                    dx4tx1 * (ue[ip1][3] - 2.0 * ue[i][3] + ue[im1][3]);

                forcing[k][j][i][4] =
                    forcing[k][j][i][4] -
                    tx2 * (buf[ip1][1] * (c1 * ue[ip1][4] - c2 * q[ip1]) -
                           buf[im1][1] * (c1 * ue[im1][4] - c2 * q[im1])) +
                    0.5 * xxcon3 *
                        (buf[ip1][0] - 2.0 * buf[i][0] + buf[im1][0]) +
                    xxcon4 * (cuf[ip1] - 2.0 * cuf[i] + cuf[im1]) +
                    xxcon5 * (buf[ip1][4] - 2.0 * buf[i][4] + buf[im1][4]) +
                    dx5tx1 * (ue[ip1][4] - 2.0 * ue[i][4] + ue[im1][4]);
            }

            /*
             * ---------------------------------------------------------------------
             * fourth-order dissipation
             * ---------------------------------------------------------------------
             */
            for (m = 0; m < 5; m++) {
                i = 1;
                forcing[k][j][i][m] =
                    forcing[k][j][i][m] -
                    dssp * (5.0 * ue[i][m] - 4.0 * ue[i + 1][m] + ue[i + 2][m]);
                i = 2;
                forcing[k][j][i][m] =
                    forcing[k][j][i][m] -
                    dssp * (-4.0 * ue[i - 1][m] + 6.0 * ue[i][m] -
                            4.0 * ue[i + 1][m] + ue[i + 2][m]);
            }

            for (i = 3; i <= grid_points[0] - 4; i++) {
                for (m = 0; m < 5; m++) {
                    forcing[k][j][i][m] =
                        forcing[k][j][i][m] -
                        dssp * (ue[i - 2][m] - 4.0 * ue[i - 1][m] +
                                6.0 * ue[i][m] - 4.0 * ue[i + 1][m] +
                                ue[i + 2][m]);
                }
            }

            for (m = 0; m < 5; m++) {
                i = grid_points[0] - 3;
                forcing[k][j][i][m] =
                    forcing[k][j][i][m] -
                    dssp * (ue[i - 2][m] - 4.0 * ue[i - 1][m] + 6.0 * ue[i][m] -
                            4.0 * ue[i + 1][m]);
                i = grid_points[0] - 2;
                forcing[k][j][i][m] =
                    forcing[k][j][i][m] -
                    dssp * (ue[i - 2][m] - 4.0 * ue[i - 1][m] + 5.0 * ue[i][m]);
            }
        }
    }

    /*
     * ---------------------------------------------------------------------
     * eta-direction flux differences
     * ---------------------------------------------------------------------
     */
    for (k = 1; k <= grid_points[2] - 2; k++) {
        zeta = (double)(k)*dKMAXm1;
        for (i = 1; i <= grid_points[0] - 2; i++) {
            xi = (double)(i)*dIMAXm1;

            for (j = 0; j <= grid_points[1] - 1; j++) {
                eta = (double)(j)*dJMAXm1;

                exact_solution(xi, eta, zeta, dtemp);
                for (m = 0; m < 5; m++) {
                    ue[j][m] = dtemp[m];
                }

                dtpp = 1.0 / dtemp[0];

                for (m = 1; m < 5; m++) {
                    buf[j][m] = dtpp * dtemp[m];
                }

                cuf[j] = buf[j][2] * buf[j][2];
                buf[j][0] =
                    cuf[j] + buf[j][1] * buf[j][1] + buf[j][3] * buf[j][3];
                q[j] = 0.5 * (buf[j][1] * ue[j][1] + buf[j][2] * ue[j][2] +
                              buf[j][3] * ue[j][3]);
            }

            for (j = 1; j <= grid_points[1] - 2; j++) {
                jm1 = j - 1;
                jp1 = j + 1;

                forcing[k][j][i][0] =
                    forcing[k][j][i][0] - ty2 * (ue[jp1][2] - ue[jm1][2]) +
                    dy1ty1 * (ue[jp1][0] - 2.0 * ue[j][0] + ue[jm1][0]);

                forcing[k][j][i][1] =
                    forcing[k][j][i][1] -
                    ty2 *
                        (ue[jp1][1] * buf[jp1][2] - ue[jm1][1] * buf[jm1][2]) +
                    yycon2 * (buf[jp1][1] - 2.0 * buf[j][1] + buf[jm1][1]) +
                    dy2ty1 * (ue[jp1][1] - 2.0 * ue[j][1] + ue[jm1][1]);

                forcing[k][j][i][2] =
                    forcing[k][j][i][2] -
                    ty2 * ((ue[jp1][2] * buf[jp1][2] +
                            c2 * (ue[jp1][4] - q[jp1])) -
                           (ue[jm1][2] * buf[jm1][2] +
                            c2 * (ue[jm1][4] - q[jm1]))) +
                    yycon1 * (buf[jp1][2] - 2.0 * buf[j][2] + buf[jm1][2]) +
                    dy3ty1 * (ue[jp1][2] - 2.0 * ue[j][2] + ue[jm1][2]);

                forcing[k][j][i][3] =
                    forcing[k][j][i][3] -
                    ty2 *
                        (ue[jp1][3] * buf[jp1][2] - ue[jm1][3] * buf[jm1][2]) +
                    yycon2 * (buf[jp1][3] - 2.0 * buf[j][3] + buf[jm1][3]) +
                    dy4ty1 * (ue[jp1][3] - 2.0 * ue[j][3] + ue[jm1][3]);

                forcing[k][j][i][4] =
                    forcing[k][j][i][4] -
                    ty2 * (buf[jp1][2] * (c1 * ue[jp1][4] - c2 * q[jp1]) -
                           buf[jm1][2] * (c1 * ue[jm1][4] - c2 * q[jm1])) +
                    0.5 * yycon3 *
                        (buf[jp1][0] - 2.0 * buf[j][0] + buf[jm1][0]) +
                    yycon4 * (cuf[jp1] - 2.0 * cuf[j] + cuf[jm1]) +
                    yycon5 * (buf[jp1][4] - 2.0 * buf[j][4] + buf[jm1][4]) +
                    dy5ty1 * (ue[jp1][4] - 2.0 * ue[j][4] + ue[jm1][4]);
            }

            /*
             * ---------------------------------------------------------------------
             * fourth-order dissipation
             * ---------------------------------------------------------------------
             */
            for (m = 0; m < 5; m++) {
                j = 1;
                forcing[k][j][i][m] =
                    forcing[k][j][i][m] -
                    dssp * (5.0 * ue[j][m] - 4.0 * ue[j + 1][m] + ue[j + 2][m]);
                j = 2;
                forcing[k][j][i][m] =
                    forcing[k][j][i][m] -
                    dssp * (-4.0 * ue[j - 1][m] + 6.0 * ue[j][m] -
                            4.0 * ue[j + 1][m] + ue[j + 2][m]);
            }

            for (j = 3; j <= grid_points[1] - 4; j++) {
                for (m = 0; m < 5; m++) {
                    forcing[k][j][i][m] =
                        forcing[k][j][i][m] -
                        dssp * (ue[j - 2][m] - 4.0 * ue[j - 1][m] +
                                6.0 * ue[j][m] - 4.0 * ue[j + 1][m] +
                                ue[j + 2][m]);
                }
            }

            for (m = 0; m < 5; m++) {
                j = grid_points[1] - 3;
                forcing[k][j][i][m] =
                    forcing[k][j][i][m] -
                    dssp * (ue[j - 2][m] - 4.0 * ue[j - 1][m] + 6.0 * ue[j][m] -
                            4.0 * ue[j + 1][m]);
                j = grid_points[1] - 2;
                forcing[k][j][i][m] =
                    forcing[k][j][i][m] -
                    dssp * (ue[j - 2][m] - 4.0 * ue[j - 1][m] + 5.0 * ue[j][m]);
            }
        }
    }

    /*
     * ---------------------------------------------------------------------
     * zeta-direction flux differences
     * ---------------------------------------------------------------------
     */
    for (j = 1; j <= grid_points[1] - 2; j++) {
        eta = (double)(j)*dJMAXm1;
        for (i = 1; i <= grid_points[0] - 2; i++) {
            xi = (double)(i)*dIMAXm1;

            for (k = 0; k <= grid_points[2] - 1; k++) {
                zeta = (double)(k)*dKMAXm1;

                exact_solution(xi, eta, zeta, dtemp);
                for (m = 0; m < 5; m++) {
                    ue[k][m] = dtemp[m];
                }

                dtpp = 1.0 / dtemp[0];

                for (m = 1; m < 5; m++) {
                    buf[k][m] = dtpp * dtemp[m];
                }

                cuf[k] = buf[k][3] * buf[k][3];
                buf[k][0] =
                    cuf[k] + buf[k][1] * buf[k][1] + buf[k][2] * buf[k][2];
                q[k] = 0.5 * (buf[k][1] * ue[k][1] + buf[k][2] * ue[k][2] +
                              buf[k][3] * ue[k][3]);
            }

            for (k = 1; k <= grid_points[2] - 2; k++) {
                km1 = k - 1;
                kp1 = k + 1;

                forcing[k][j][i][0] =
                    forcing[k][j][i][0] - tz2 * (ue[kp1][3] - ue[km1][3]) +
                    dz1tz1 * (ue[kp1][0] - 2.0 * ue[k][0] + ue[km1][0]);

                forcing[k][j][i][1] =
                    forcing[k][j][i][1] -
                    tz2 *
                        (ue[kp1][1] * buf[kp1][3] - ue[km1][1] * buf[km1][3]) +
                    zzcon2 * (buf[kp1][1] - 2.0 * buf[k][1] + buf[km1][1]) +
                    dz2tz1 * (ue[kp1][1] - 2.0 * ue[k][1] + ue[km1][1]);

                forcing[k][j][i][2] =
                    forcing[k][j][i][2] -
                    tz2 *
                        (ue[kp1][2] * buf[kp1][3] - ue[km1][2] * buf[km1][3]) +
                    zzcon2 * (buf[kp1][2] - 2.0 * buf[k][2] + buf[km1][2]) +
                    dz3tz1 * (ue[kp1][2] - 2.0 * ue[k][2] + ue[km1][2]);

                forcing[k][j][i][3] =
                    forcing[k][j][i][3] -
                    tz2 * ((ue[kp1][3] * buf[kp1][3] +
                            c2 * (ue[kp1][4] - q[kp1])) -
                           (ue[km1][3] * buf[km1][3] +
                            c2 * (ue[km1][4] - q[km1]))) +
                    zzcon1 * (buf[kp1][3] - 2.0 * buf[k][3] + buf[km1][3]) +
                    dz4tz1 * (ue[kp1][3] - 2.0 * ue[k][3] + ue[km1][3]);

                forcing[k][j][i][4] =
                    forcing[k][j][i][4] -
                    tz2 * (buf[kp1][3] * (c1 * ue[kp1][4] - c2 * q[kp1]) -
                           buf[km1][3] * (c1 * ue[km1][4] - c2 * q[km1])) +
                    0.5 * zzcon3 *
                        (buf[kp1][0] - 2.0 * buf[k][0] + buf[km1][0]) +
                    zzcon4 * (cuf[kp1] - 2.0 * cuf[k] + cuf[km1]) +
                    zzcon5 * (buf[kp1][4] - 2.0 * buf[k][4] + buf[km1][4]) +
                    dz5tz1 * (ue[kp1][4] - 2.0 * ue[k][4] + ue[km1][4]);
            }

            /*
             * ---------------------------------------------------------------------
             * fourth-order dissipation
             * ---------------------------------------------------------------------
             */
            for (m = 0; m < 5; m++) {
                k = 1;
                forcing[k][j][i][m] =
                    forcing[k][j][i][m] -
                    dssp * (5.0 * ue[k][m] - 4.0 * ue[k + 1][m] + ue[k + 2][m]);
                k = 2;
                forcing[k][j][i][m] =
                    forcing[k][j][i][m] -
                    dssp * (-4.0 * ue[k - 1][m] + 6.0 * ue[k][m] -
                            4.0 * ue[k + 1][m] + ue[k + 2][m]);
            }

            for (k = 3; k <= grid_points[2] - 4; k++) {
                for (m = 0; m < 5; m++) {
                    forcing[k][j][i][m] =
                        forcing[k][j][i][m] -
                        dssp * (ue[k - 2][m] - 4.0 * ue[k - 1][m] +
                                6.0 * ue[k][m] - 4.0 * ue[k + 1][m] +
                                ue[k + 2][m]);
                }
            }

            for (m = 0; m < 5; m++) {
                k = grid_points[2] - 3;
                forcing[k][j][i][m] =
                    forcing[k][j][i][m] -
                    dssp * (ue[k - 2][m] - 4.0 * ue[k - 1][m] + 6.0 * ue[k][m] -
                            4.0 * ue[k + 1][m]);
                k = grid_points[2] - 2;
                forcing[k][j][i][m] =
                    forcing[k][j][i][m] -
                    dssp * (ue[k - 2][m] - 4.0 * ue[k - 1][m] + 5.0 * ue[k][m]);
            }
        }
    }

    /*
     * ---------------------------------------------------------------------
     * now change the sign of the forcing function
     * ---------------------------------------------------------------------
     */
    for (k = 1; k <= grid_points[2] - 2; k++) {
        for (j = 1; j <= grid_points[1] - 2; j++) {
            for (i = 1; i <= grid_points[0] - 2; i++) {
                for (m = 0; m < 5; m++) {
                    forcing[k][j][i][m] = -1.0 * forcing[k][j][i][m];
                }
            }
        }
    }
}

void exact_solution(double xi, double eta, double zeta, double dtemp[5]) {
    int m;

    for (m = 0; m < 5; m++) {
        dtemp[m] =
            ce[m][0] +
            xi * (ce[m][1] +
                  xi * (ce[m][4] + xi * (ce[m][7] + xi * ce[m][10]))) +
            eta * (ce[m][2] +
                   eta * (ce[m][5] + eta * (ce[m][8] + eta * ce[m][11]))) +
            zeta * (ce[m][3] +
                    zeta * (ce[m][6] + zeta * (ce[m][9] + zeta * ce[m][12])));
    }
}

/*
 * ---------------------------------------------------------------------
 * this subroutine initializes the field variable u using
 * tri-linear transfinite interpolation of the boundary values
 * ---------------------------------------------------------------------
 */
void initialize() {
    int i, j, k, m, ix, iy, iz;
    double xi, eta, zeta, Pface[2][3][5], Pxi, Peta, Pzeta, temp[5];

    /*
     * ---------------------------------------------------------------------
     * later (in compute_rhs) we compute 1/u for every element. a few of
     * the corner elements are not used, but it convenient (and faster)
     * to compute the whole thing with a simple loop. make sure those
     * values are noKMAXero by initializing the whole thing here.
     * ---------------------------------------------------------------------
     */
    for (k = 0; k <= grid_points[2] - 1; k++) {
        for (j = 0; j <= grid_points[1] - 1; j++) {
            for (i = 0; i <= grid_points[0] - 1; i++) {
                for (m = 0; m < 5; m++) {
                    u[k][j][i][m] = 1.0;
                }
            }
        }
    }

    /*
     * ---------------------------------------------------------------------
     * first store the "interpolated" values everywhere on the grid
     * ---------------------------------------------------------------------
     */
    for (k = 0; k <= grid_points[2] - 1; k++) {
        zeta = (double)(k)*dKMAXm1;
        for (j = 0; j <= grid_points[1] - 1; j++) {
            eta = (double)(j)*dJMAXm1;
            for (i = 0; i <= grid_points[0] - 1; i++) {
                xi = (double)(i)*dIMAXm1;

                for (ix = 0; ix < 2; ix++) {
                    exact_solution((double)ix, eta, zeta, &Pface[ix][0][0]);
                }

                for (iy = 0; iy < 2; iy++) {
                    exact_solution(xi, (double)iy, zeta, &Pface[iy][1][0]);
                }

                for (iz = 0; iz < 2; iz++) {
                    exact_solution(xi, eta, (double)iz, &Pface[iz][2][0]);
                }

                for (m = 0; m < 5; m++) {
                    Pxi = xi * Pface[1][0][m] + (1.0 - xi) * Pface[0][0][m];
                    Peta = eta * Pface[1][1][m] + (1.0 - eta) * Pface[0][1][m];
                    Pzeta =
                        zeta * Pface[1][2][m] + (1.0 - zeta) * Pface[0][2][m];

                    u[k][j][i][m] = Pxi + Peta + Pzeta - Pxi * Peta -
                                    Pxi * Pzeta - Peta * Pzeta +
                                    Pxi * Peta * Pzeta;
                }
            }
        }
    }

    /*
     * ---------------------------------------------------------------------
     * now store the exact values on the boundaries
     * ---------------------------------------------------------------------
     * west face
     * ---------------------------------------------------------------------
     */
    i = 0;
    xi = 0.0;
    for (k = 0; k <= grid_points[2] - 1; k++) {
        zeta = (double)(k)*dKMAXm1;
        for (j = 0; j <= grid_points[1] - 1; j++) {
            eta = (double)(j)*dJMAXm1;
            exact_solution(xi, eta, zeta, temp);
            for (m = 0; m < 5; m++) {
                u[k][j][i][m] = temp[m];
            }
        }
    }

    /*
     * ---------------------------------------------------------------------
     * east face
     * ---------------------------------------------------------------------
     */
    i = grid_points[0] - 1;
    xi = 1.0;
    for (k = 0; k <= grid_points[2] - 1; k++) {
        zeta = (double)(k)*dKMAXm1;
        for (j = 0; j <= grid_points[1] - 1; j++) {
            eta = (double)(j)*dJMAXm1;
            exact_solution(xi, eta, zeta, temp);
            for (m = 0; m < 5; m++) {
                u[k][j][i][m] = temp[m];
            }
        }
    }

    /*
     * ---------------------------------------------------------------------
     * south face
     * ---------------------------------------------------------------------
     */
    j = 0;
    eta = 0.0;
    for (k = 0; k <= grid_points[2] - 1; k++) {
        zeta = (double)(k)*dKMAXm1;
        for (i = 0; i <= grid_points[0] - 1; i++) {
            xi = (double)(i)*dIMAXm1;
            exact_solution(xi, eta, zeta, temp);
            for (m = 0; m < 5; m++) {
                u[k][j][i][m] = temp[m];
            }
        }
    }

    /*
     * ---------------------------------------------------------------------
     * north face
     * ---------------------------------------------------------------------
     */
    j = grid_points[1] - 1;
    eta = 1.0;
    for (k = 0; k <= grid_points[2] - 1; k++) {
        zeta = (double)(k)*dKMAXm1;
        for (i = 0; i <= grid_points[0] - 1; i++) {
            xi = (double)(i)*dIMAXm1;
            exact_solution(xi, eta, zeta, temp);
            for (m = 0; m < 5; m++) {
                u[k][j][i][m] = temp[m];
            }
        }
    }

    /*
     * ---------------------------------------------------------------------
     * bottom face
     * ---------------------------------------------------------------------
     */
    k = 0;
    zeta = 0.0;
    for (j = 0; j <= grid_points[1] - 1; j++) {
        eta = (double)(j)*dJMAXm1;
        for (i = 0; i <= grid_points[0] - 1; i++) {
            xi = (double)(i)*dIMAXm1;
            exact_solution(xi, eta, zeta, temp);
            for (m = 0; m < 5; m++) {
                u[k][j][i][m] = temp[m];
            }
        }
    }

    /*
     * ---------------------------------------------------------------------
     * top face
     * ---------------------------------------------------------------------
     */
    k = grid_points[2] - 1;
    zeta = 1.0;
    for (j = 0; j <= grid_points[1] - 1; j++) {
        eta = (double)(j)*dJMAXm1;
        for (i = 0; i <= grid_points[0] - 1; i++) {
            xi = (double)(i)*dIMAXm1;
            exact_solution(xi, eta, zeta, temp);
            for (m = 0; m < 5; m++) {
                u[k][j][i][m] = temp[m];
            }
        }
    }
}

void release_gpu() {
    cudaFree(forcing_device);
    cudaFree(u_device);
    cudaFree(rhs_device);
    cudaFree(us_device);
    cudaFree(vs_device);
    cudaFree(ws_device);
    cudaFree(qs_device);
    cudaFree(rho_i_device);
    cudaFree(square_device);
    cudaFree(lhsA_device);
    cudaFree(lhsB_device);
    cudaFree(lhsC_device);
}

void rhs_norm(double rms[5]) {
    int i, j, k, d, m;
    double add;

    for (m = 0; m < 5; m++) {
        rms[m] = 0.0;
    }
    for (k = 1; k <= grid_points[2] - 2; k++) {
        for (j = 1; j <= grid_points[1] - 2; j++) {
            for (i = 1; i <= grid_points[0] - 2; i++) {
                for (m = 0; m < 5; m++) {
                    add = rhs[k][j][i][m];
                    rms[m] = rms[m] + add * add;
                }
            }
        }
    }
    for (m = 0; m < 5; m++) {
        for (d = 0; d < 3; d++) {
            rms[m] = rms[m] / (double)(grid_points[d] - 2);
        }
        rms[m] = sqrt(rms[m]);
    }
}

size_t round_amount_of_work(size_t amount_of_work, size_t amount_of_threads) {
    size_t rest = amount_of_work % amount_of_threads;
    return (rest == 0) ? amount_of_work
                       : (amount_of_work + amount_of_threads - rest);
}

void set_constants() {
    ce[0][0] = 2.0;
    ce[0][1] = 0.0;
    ce[0][2] = 0.0;
    ce[0][3] = 4.0;
    ce[0][4] = 5.0;
    ce[0][5] = 3.0;
    ce[0][6] = 0.5;
    ce[0][7] = 0.02;
    ce[0][8] = 0.01;
    ce[0][9] = 0.03;
    ce[0][10] = 0.5;
    ce[0][11] = 0.4;
    ce[0][12] = 0.3;

    ce[1][0] = 1.0;
    ce[1][1] = 0.0;
    ce[1][2] = 0.0;
    ce[1][3] = 0.0;
    ce[1][4] = 1.0;
    ce[1][5] = 2.0;
    ce[1][6] = 3.0;
    ce[1][7] = 0.01;
    ce[1][8] = 0.03;
    ce[1][9] = 0.02;
    ce[1][10] = 0.4;
    ce[1][11] = 0.3;
    ce[1][12] = 0.5;

    ce[2][0] = 2.0;
    ce[2][1] = 2.0;
    ce[2][2] = 0.0;
    ce[2][3] = 0.0;
    ce[2][4] = 0.0;
    ce[2][5] = 2.0;
    ce[2][6] = 3.0;
    ce[2][7] = 0.04;
    ce[2][8] = 0.03;
    ce[2][9] = 0.05;
    ce[2][10] = 0.3;
    ce[2][11] = 0.5;
    ce[2][12] = 0.4;

    ce[3][0] = 2.0;
    ce[3][1] = 2.0;
    ce[3][2] = 0.0;
    ce[3][3] = 0.0;
    ce[3][4] = 0.0;
    ce[3][5] = 2.0;
    ce[3][6] = 3.0;
    ce[3][7] = 0.03;
    ce[3][8] = 0.05;
    ce[3][9] = 0.04;
    ce[3][10] = 0.2;
    ce[3][11] = 0.1;
    ce[3][12] = 0.3;

    ce[4][0] = 5.0;
    ce[4][1] = 4.0;
    ce[4][2] = 3.0;
    ce[4][3] = 2.0;
    ce[4][4] = 0.1;
    ce[4][5] = 0.4;
    ce[4][6] = 0.3;
    ce[4][7] = 0.05;
    ce[4][8] = 0.04;
    ce[4][9] = 0.03;
    ce[4][10] = 0.1;
    ce[4][11] = 0.3;
    ce[4][12] = 0.2;

    c1 = 1.4;
    c2 = 0.4;
    c3 = 0.1;
    c4 = 1.0;
    c5 = 1.4;

    dIMAXm1 = 1.0 / (double)(grid_points[0] - 1);
    dJMAXm1 = 1.0 / (double)(grid_points[1] - 1);
    dKMAXm1 = 1.0 / (double)(grid_points[2] - 1);

    c1c2 = c1 * c2;
    c1c5 = c1 * c5;
    c3c4 = c3 * c4;
    c1345 = c1c5 * c3c4;

    coKMAX1 = (1.0 - c1c5);

    tx1 = 1.0 / (dIMAXm1 * dIMAXm1);
    tx2 = 1.0 / (2.0 * dIMAXm1);
    tx3 = 1.0 / dIMAXm1;

    ty1 = 1.0 / (dJMAXm1 * dJMAXm1);
    ty2 = 1.0 / (2.0 * dJMAXm1);
    ty3 = 1.0 / dJMAXm1;

    tz1 = 1.0 / (dKMAXm1 * dKMAXm1);
    tz2 = 1.0 / (2.0 * dKMAXm1);
    tz3 = 1.0 / dKMAXm1;

    dx1 = 0.75;
    dx2 = 0.75;
    dx3 = 0.75;
    dx4 = 0.75;
    dx5 = 0.75;

    dy1 = 0.75;
    dy2 = 0.75;
    dy3 = 0.75;
    dy4 = 0.75;
    dy5 = 0.75;

    dz1 = 1.0;
    dz2 = 1.0;
    dz3 = 1.0;
    dz4 = 1.0;
    dz5 = 1.0;

    dxmax = max(dx3, dx4);
    dymax = max(dy2, dy4);
    dzmax = max(dz2, dz3);

    dssp = 0.25 * max(dx1, max(dy1, dz1));

    c4dssp = 4.0 * dssp;
    c5dssp = 5.0 * dssp;

    dttx1 = dt * tx1;
    dttx2 = dt * tx2;
    dtty1 = dt * ty1;
    dtty2 = dt * ty2;
    dttz1 = dt * tz1;
    dttz2 = dt * tz2;

    c2dttx1 = 2.0 * dttx1;
    c2dtty1 = 2.0 * dtty1;
    c2dttz1 = 2.0 * dttz1;

    dtdssp = dt * dssp;

    comz1 = dtdssp;
    comz4 = 4.0 * dtdssp;
    comz5 = 5.0 * dtdssp;
    comz6 = 6.0 * dtdssp;

    c3c4tx3 = c3c4 * tx3;
    c3c4ty3 = c3c4 * ty3;
    c3c4tz3 = c3c4 * tz3;

    dx1tx1 = dx1 * tx1;
    dx2tx1 = dx2 * tx1;
    dx3tx1 = dx3 * tx1;
    dx4tx1 = dx4 * tx1;
    dx5tx1 = dx5 * tx1;

    dy1ty1 = dy1 * ty1;
    dy2ty1 = dy2 * ty1;
    dy3ty1 = dy3 * ty1;
    dy4ty1 = dy4 * ty1;
    dy5ty1 = dy5 * ty1;

    dz1tz1 = dz1 * tz1;
    dz2tz1 = dz2 * tz1;
    dz3tz1 = dz3 * tz1;
    dz4tz1 = dz4 * tz1;
    dz5tz1 = dz5 * tz1;

    c2iv = 2.5;
    con43 = 4.0 / 3.0;
    con16 = 1.0 / 6.0;

    xxcon1 = c3c4tx3 * con43 * tx3;
    xxcon2 = c3c4tx3 * tx3;
    xxcon3 = c3c4tx3 * coKMAX1 * tx3;
    xxcon4 = c3c4tx3 * con16 * tx3;
    xxcon5 = c3c4tx3 * c1c5 * tx3;

    yycon1 = c3c4ty3 * con43 * ty3;
    yycon2 = c3c4ty3 * ty3;
    yycon3 = c3c4ty3 * coKMAX1 * ty3;
    yycon4 = c3c4ty3 * con16 * ty3;
    yycon5 = c3c4ty3 * c1c5 * ty3;

    zzcon1 = c3c4tz3 * con43 * tz3;
    zzcon2 = c3c4tz3 * tz3;
    zzcon3 = c3c4tz3 * coKMAX1 * tz3;
    zzcon4 = c3c4tz3 * con16 * tz3;
    zzcon5 = c3c4tz3 * c1c5 * tz3;

    cudaMemcpyToSymbol(constants_device::ce, &ce, 13 * 5 * sizeof(double));
    cudaMemcpyToSymbol(constants_device::dt, &dt, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c1, &c1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c2, &c2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c3, &c3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c4, &c4, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c5, &c5, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dIMAXm1, &dIMAXm1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dJMAXm1, &dJMAXm1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dKMAXm1, &dKMAXm1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c1c2, &c1c2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c1c5, &c1c5, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c3c4, &c3c4, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c1345, &c1345, sizeof(double));
    cudaMemcpyToSymbol(constants_device::coKMAX1, &coKMAX1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::tx1, &tx1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::tx2, &tx2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::tx3, &tx3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::ty1, &ty1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::ty2, &ty2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::ty3, &ty3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::tz1, &tz1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::tz2, &tz2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::tz3, &tz3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx1, &dx1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx2, &dx2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx3, &dx3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx4, &dx4, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx5, &dx5, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy1, &dy1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy2, &dy2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy3, &dy3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy4, &dy4, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy5, &dy5, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz1, &dz1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz2, &dz2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz3, &dz3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz4, &dz4, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz5, &dz5, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dxmax, &dxmax, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dymax, &dymax, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dzmax, &dzmax, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dssp, &dssp, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c4dssp, &c4dssp, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c5dssp, &c5dssp, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dttx1, &dttx1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dttx2, &dttx2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dtty1, &dtty1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dtty2, &dtty2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dttz1, &dttz1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dttz2, &dttz2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c2dttx1, &c2dttx1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c2dtty1, &c2dtty1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c2dttz1, &c2dttz1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dtdssp, &dtdssp, sizeof(double));
    cudaMemcpyToSymbol(constants_device::comz1, &comz1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::comz4, &comz4, sizeof(double));
    cudaMemcpyToSymbol(constants_device::comz5, &comz5, sizeof(double));
    cudaMemcpyToSymbol(constants_device::comz6, &comz6, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c3c4tx3, &c3c4tx3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c3c4ty3, &c3c4ty3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c3c4tz3, &c3c4tz3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx1tx1, &dx1tx1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx2tx1, &dx2tx1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx3tx1, &dx3tx1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx4tx1, &dx4tx1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dx5tx1, &dx5tx1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy1ty1, &dy1ty1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy2ty1, &dy2ty1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy3ty1, &dy3ty1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy4ty1, &dy4ty1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dy5ty1, &dy5ty1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz1tz1, &dz1tz1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz2tz1, &dz2tz1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz3tz1, &dz3tz1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz4tz1, &dz4tz1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::dz5tz1, &dz5tz1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::c2iv, &c2iv, sizeof(double));
    cudaMemcpyToSymbol(constants_device::con43, &con43, sizeof(double));
    cudaMemcpyToSymbol(constants_device::con16, &con16, sizeof(double));
    cudaMemcpyToSymbol(constants_device::xxcon1, &xxcon1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::xxcon2, &xxcon2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::xxcon3, &xxcon3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::xxcon4, &xxcon4, sizeof(double));
    cudaMemcpyToSymbol(constants_device::xxcon5, &xxcon5, sizeof(double));
    cudaMemcpyToSymbol(constants_device::yycon1, &yycon1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::yycon2, &yycon2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::yycon3, &yycon3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::yycon4, &yycon4, sizeof(double));
    cudaMemcpyToSymbol(constants_device::yycon5, &yycon5, sizeof(double));
    cudaMemcpyToSymbol(constants_device::zzcon1, &zzcon1, sizeof(double));
    cudaMemcpyToSymbol(constants_device::zzcon2, &zzcon2, sizeof(double));
    cudaMemcpyToSymbol(constants_device::zzcon3, &zzcon3, sizeof(double));
    cudaMemcpyToSymbol(constants_device::zzcon4, &zzcon4, sizeof(double));
    cudaMemcpyToSymbol(constants_device::zzcon5, &zzcon5, sizeof(double));
}

void setup_gpu() {
    /*
     * struct cudaDeviceProp{
     *  char name[256];
     *  size_t totalGlobalMem;
     *  size_t sharedMemPerBlock;
     *  int regsPerBlock;
     *  int warpSize;
     *  size_t memPitch;
     *  int maxThreadsPerBlock;
     *  int maxThreadsDim[3];
     *  int maxGridSize[3];
     *  size_t totalConstMem;
     *  int major;
     *  int minor;
     *  int clockRate;
     *  size_t textureAlignment;
     *  int deviceOverlap;
     *  int multiProcessorCount;
     *  int kernelExecTimeoutEnabled;
     *  int integrated;
     *  int canMapHostMemory;
     *  int computeMode;
     *  int concurrentKernels;
     *  int ECCEnabled;
     *  int pciBusID;
     *  int pciDeviceID;
     *  int tccDriver;
     * }
     */
    /* amount of available devices */
    cudaGetDeviceCount(&total_devices);

    /* define gpu_device */
    if (total_devices == 0) {
        // printf("\n\n\nNo Nvidia GPU found!\n\n\n");
        exit(-1);
    } else if ((GPU_DEVICE >= 0) && (GPU_DEVICE < total_devices)) {
        gpu_device_id = GPU_DEVICE;
    } else {
        gpu_device_id = 0;
    }
    cudaSetDevice(gpu_device_id);
    cudaGetDeviceProperties(&gpu_device_properties, gpu_device_id);

    /* define threads_per_block */
    /* new */
    if ((BT_THREADS_PER_BLOCK_ON_ADD >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_ADD <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_ADD = BT_THREADS_PER_BLOCK_ON_ADD;
    } else {
        THREADS_PER_BLOCK_ON_ADD = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_RHS_1 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_RHS_1 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_RHS_1 = BT_THREADS_PER_BLOCK_ON_RHS_1;
    } else {
        THREADS_PER_BLOCK_ON_RHS_1 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_RHS_2 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_RHS_2 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_RHS_2 = BT_THREADS_PER_BLOCK_ON_RHS_2;
    } else {
        THREADS_PER_BLOCK_ON_RHS_2 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_RHS_3 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_RHS_3 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_RHS_3 = BT_THREADS_PER_BLOCK_ON_RHS_3;
    } else {
        THREADS_PER_BLOCK_ON_RHS_3 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_RHS_4 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_RHS_4 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_RHS_4 = BT_THREADS_PER_BLOCK_ON_RHS_4;
    } else {
        THREADS_PER_BLOCK_ON_RHS_4 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_RHS_5 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_RHS_5 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_RHS_5 = BT_THREADS_PER_BLOCK_ON_RHS_5;
    } else {
        THREADS_PER_BLOCK_ON_RHS_5 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_RHS_6 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_RHS_6 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_RHS_6 = BT_THREADS_PER_BLOCK_ON_RHS_6;
    } else {
        THREADS_PER_BLOCK_ON_RHS_6 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_RHS_7 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_RHS_7 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_RHS_7 = BT_THREADS_PER_BLOCK_ON_RHS_7;
    } else {
        THREADS_PER_BLOCK_ON_RHS_7 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_RHS_8 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_RHS_8 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_RHS_8 = BT_THREADS_PER_BLOCK_ON_RHS_8;
    } else {
        THREADS_PER_BLOCK_ON_RHS_8 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_RHS_9 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_RHS_9 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_RHS_9 = BT_THREADS_PER_BLOCK_ON_RHS_9;
    } else {
        THREADS_PER_BLOCK_ON_RHS_9 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_X_SOLVE_1 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_X_SOLVE_1 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_X_SOLVE_1 = BT_THREADS_PER_BLOCK_ON_X_SOLVE_1;
    } else {
        THREADS_PER_BLOCK_ON_X_SOLVE_1 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_X_SOLVE_2 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_X_SOLVE_2 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_X_SOLVE_2 = BT_THREADS_PER_BLOCK_ON_X_SOLVE_2;
    } else {
        THREADS_PER_BLOCK_ON_X_SOLVE_2 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_X_SOLVE_3 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_X_SOLVE_3 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_X_SOLVE_3 = BT_THREADS_PER_BLOCK_ON_X_SOLVE_3;
    } else {
        THREADS_PER_BLOCK_ON_X_SOLVE_3 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_Y_SOLVE_1 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_Y_SOLVE_1 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_Y_SOLVE_1 = BT_THREADS_PER_BLOCK_ON_Y_SOLVE_1;
    } else {
        THREADS_PER_BLOCK_ON_Y_SOLVE_1 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_Y_SOLVE_2 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_Y_SOLVE_2 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_Y_SOLVE_2 = BT_THREADS_PER_BLOCK_ON_Y_SOLVE_2;
    } else {
        THREADS_PER_BLOCK_ON_Y_SOLVE_2 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_Y_SOLVE_3 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_Y_SOLVE_3 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_Y_SOLVE_3 = BT_THREADS_PER_BLOCK_ON_Y_SOLVE_3;
    } else {
        THREADS_PER_BLOCK_ON_Y_SOLVE_3 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_Z_SOLVE_1 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_Z_SOLVE_1 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_Z_SOLVE_1 = BT_THREADS_PER_BLOCK_ON_Z_SOLVE_1;
    } else {
        THREADS_PER_BLOCK_ON_Z_SOLVE_1 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_Z_SOLVE_2 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_Z_SOLVE_2 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_Z_SOLVE_2 = BT_THREADS_PER_BLOCK_ON_Z_SOLVE_2;
    } else {
        THREADS_PER_BLOCK_ON_Z_SOLVE_2 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_Z_SOLVE_3 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_Z_SOLVE_3 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_Z_SOLVE_3 = BT_THREADS_PER_BLOCK_ON_Z_SOLVE_3;
    } else {
        THREADS_PER_BLOCK_ON_Z_SOLVE_3 = gpu_device_properties.warpSize;
    }
    /* old */
    if ((BT_THREADS_PER_BLOCK_ON_EXACT_RHS_1 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_EXACT_RHS_1 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_EXACT_RHS_1 = BT_THREADS_PER_BLOCK_ON_EXACT_RHS_1;
    } else {
        THREADS_PER_BLOCK_ON_EXACT_RHS_1 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_EXACT_RHS_2 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_EXACT_RHS_2 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_EXACT_RHS_2 = BT_THREADS_PER_BLOCK_ON_EXACT_RHS_2;
    } else {
        THREADS_PER_BLOCK_ON_EXACT_RHS_2 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_EXACT_RHS_3 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_EXACT_RHS_3 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_EXACT_RHS_3 = BT_THREADS_PER_BLOCK_ON_EXACT_RHS_3;
    } else {
        THREADS_PER_BLOCK_ON_EXACT_RHS_3 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_EXACT_RHS_4 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_EXACT_RHS_4 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_EXACT_RHS_4 = BT_THREADS_PER_BLOCK_ON_EXACT_RHS_4;
    } else {
        THREADS_PER_BLOCK_ON_EXACT_RHS_4 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_ERROR_NORM_1 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_ERROR_NORM_1 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_ERROR_NORM_1 =
            BT_THREADS_PER_BLOCK_ON_ERROR_NORM_1;
    } else {
        THREADS_PER_BLOCK_ON_ERROR_NORM_1 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_ERROR_NORM_2 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_ERROR_NORM_2 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_ERROR_NORM_2 =
            BT_THREADS_PER_BLOCK_ON_ERROR_NORM_2;
    } else {
        THREADS_PER_BLOCK_ON_ERROR_NORM_2 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_INITIALIZE >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_INITIALIZE <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_INITIALIZE = BT_THREADS_PER_BLOCK_ON_INITIALIZE;
    } else {
        THREADS_PER_BLOCK_ON_INITIALIZE = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_RHS_NORM_1 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_RHS_NORM_1 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_RHS_NORM_1 = BT_THREADS_PER_BLOCK_ON_RHS_NORM_1;
    } else {
        THREADS_PER_BLOCK_ON_RHS_NORM_1 = gpu_device_properties.warpSize;
    }
    if ((BT_THREADS_PER_BLOCK_ON_RHS_NORM_2 >= 1) &&
        (BT_THREADS_PER_BLOCK_ON_RHS_NORM_2 <=
         gpu_device_properties.maxThreadsPerBlock)) {
        THREADS_PER_BLOCK_ON_RHS_NORM_2 = BT_THREADS_PER_BLOCK_ON_RHS_NORM_2;
    } else {
        THREADS_PER_BLOCK_ON_RHS_NORM_2 = gpu_device_properties.warpSize;
    }

    size_u = sizeof(double) * KMAX * (JMAXP + 1) * (IMAXP + 1) * 5;
    size_forcing = sizeof(double) * KMAX * (JMAXP + 1) * (IMAXP + 1) * 5;
    size_rhs = sizeof(double) * KMAX * (JMAXP + 1) * (IMAXP + 1) * 5;
    size_qs = sizeof(double) * KMAX * (JMAXP + 1) * (IMAXP + 1);
    size_square = sizeof(double) * KMAX * (JMAXP + 1) * (IMAXP + 1);
    size_rho_i = sizeof(double) * KMAX * (JMAXP + 1) * (IMAXP + 1);

    size_t forcing_buf_size;
    size_t u_buf_size;
    size_t rhs_buf_size;
    size_t us_buf_size;
    size_t vs_buf_size;
    size_t ws_buf_size;
    size_t qs_buf_size;
    size_t rho_i_buf_size;
    size_t square_buf_size;
    size_t lhsA_buf_size;
    size_t lhsB_buf_size;
    size_t lhsC_buf_size;
    size_t lhs_buf_size;
    size_t njac_buf_size;
    size_t fjac_buf_size;

    size_t forcing_slice_size = sizeof(double) * (JMAXP + 1) * (IMAXP + 1) * 5;
    size_t u_slice_size = sizeof(double) * (JMAXP + 1) * (IMAXP + 1) * 5;
    size_t rhs_slice_size = sizeof(double) * (JMAXP + 1) * (IMAXP + 1) * 5;
    size_t us_slice_size = sizeof(double) * (JMAXP + 1) * (IMAXP + 1);
    size_t vs_slice_size = sizeof(double) * (JMAXP + 1) * (IMAXP + 1);
    size_t ws_slice_size = sizeof(double) * (JMAXP + 1) * (IMAXP + 1);
    size_t qs_slice_size = sizeof(double) * (JMAXP + 1) * (IMAXP + 1);
    size_t rho_i_slice_size = sizeof(double) * (JMAXP + 1) * (IMAXP + 1);
    size_t square_slice_size = sizeof(double) * (JMAXP + 1) * (IMAXP + 1);

    size_t lhsA_slice_size =
        sizeof(double) * (PROBLEM_SIZE - 1) * (PROBLEM_SIZE + 1) * 5 * 5;
    size_t lhsB_slice_size =
        sizeof(double) * (PROBLEM_SIZE - 1) * (PROBLEM_SIZE + 1) * 5 * 5;
    size_t lhsC_slice_size =
        sizeof(double) * (PROBLEM_SIZE - 1) * (PROBLEM_SIZE + 1) * 5 * 5;

    size_t lhs_slice_size =
        sizeof(double) * (PROBLEM_SIZE - 1) * (PROBLEM_SIZE + 1) * 3 * 5 * 5;
    size_t njac_slice_size =
        sizeof(double) * (PROBLEM_SIZE - 1) * (PROBLEM_SIZE + 1) * 5 * 5;
    size_t fjac_slice_size =
        sizeof(double) * (PROBLEM_SIZE - 1) * (PROBLEM_SIZE + 1) * 5 * 5;

    forcing_buf_size = forcing_slice_size;
    rhs_buf_size = rhs_slice_size;
    u_buf_size = u_slice_size;
    us_buf_size = us_slice_size;
    vs_buf_size = vs_slice_size;
    ws_buf_size = ws_slice_size;
    qs_buf_size = qs_slice_size;
    rho_i_buf_size = rho_i_slice_size;
    square_buf_size = square_slice_size;
    lhs_buf_size = lhs_slice_size;
    njac_buf_size = njac_slice_size;
    fjac_buf_size = fjac_slice_size;
    lhsA_buf_size = lhsA_slice_size;
    lhsB_buf_size = lhsB_slice_size;
    lhsC_buf_size = lhsC_slice_size;
    forcing_buf_size *= PROBLEM_SIZE;
    rhs_buf_size *= PROBLEM_SIZE;
    u_buf_size *= PROBLEM_SIZE;
    us_buf_size *= PROBLEM_SIZE;
    vs_buf_size *= PROBLEM_SIZE;
    ws_buf_size *= PROBLEM_SIZE;
    qs_buf_size *= PROBLEM_SIZE;
    rho_i_buf_size *= PROBLEM_SIZE;
    square_buf_size *= PROBLEM_SIZE;
    lhs_buf_size *= (JMAXP + 1);
    njac_buf_size *= (JMAXP + 1);
    fjac_buf_size *= (JMAXP + 1);
    lhsA_buf_size *= (JMAXP + 1);
    lhsB_buf_size *= (JMAXP + 1);
    lhsC_buf_size *= (JMAXP + 1);

    cudaMalloc(&forcing_device, forcing_buf_size);
    cudaMalloc(&u_device, u_buf_size);
    cudaMalloc(&rhs_device, rhs_buf_size);
    cudaMalloc(&us_device, us_buf_size);
    cudaMalloc(&vs_device, vs_buf_size);
    cudaMalloc(&ws_device, ws_buf_size);
    cudaMalloc(&qs_device, qs_buf_size);
    cudaMalloc(&rho_i_device, rho_i_buf_size);
    cudaMalloc(&square_device, square_buf_size);
    cudaMalloc(&lhsA_device, lhsA_buf_size);
    cudaMalloc(&lhsB_device, lhsB_buf_size);
    cudaMalloc(&lhsC_device, lhsC_buf_size);
}

/*
 * ---------------------------------------------------------------------
 * verification routine
 * ---------------------------------------------------------------------
 */
void verify(int no_time_steps, char* class_npb, boolean* verified) {
    double xcrref[5], xceref[5], xcrdif[5], xcedif[5];
    double epsilon, xce[5], xcr[5], dtref = 0.0;
    int m;

    /*
     * ---------------------------------------------------------------------
     * tolerance level
     * ---------------------------------------------------------------------
     */
    epsilon = 1.0e-08;

    /*
     * ---------------------------------------------------------------------
     * compute the error norm and the residual norm, and exit if not printing
     * ---------------------------------------------------------------------
     */
    error_norm(xce);

    cudaMemcpy(u_device, u,
               sizeof(double) * KMAX * (JMAXP + 1) * (IMAXP + 1) * 5,
               cudaMemcpyHostToDevice);
    cudaMemcpy(forcing_device, forcing,
               sizeof(double) * KMAX * (JMAXP + 1) * (IMAXP + 1) * 5,
               cudaMemcpyHostToDevice);

    compute_rhs_gpu();

    cudaMemcpy(rhs, rhs_device,
               sizeof(double) * KMAX * (JMAXP + 1) * (IMAXP + 1) * 5,
               cudaMemcpyDeviceToHost);

    rhs_norm(xcr);

    for (m = 0; m < 5; m++) {
        xcr[m] = xcr[m] / dt;
    }

    *class_npb = 'U';
    *verified = true;

    for (m = 0; m < 5; m++) {
        xcrref[m] = 1.0;
        xceref[m] = 1.0;
    }

    /*
     * ---------------------------------------------------------------------
     * reference data for 12X12X12 grids after 60 time steps, with DT = 1.0e-02
     * ---------------------------------------------------------------------
     */
    if ((grid_points[0] == 12) && (grid_points[1] == 12) &&
        (grid_points[2] == 12) && (no_time_steps == 60)) {
        *class_npb = 'S';
        dtref = 1.0e-2;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual.
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 1.7034283709541311e-01;
        xcrref[1] = 1.2975252070034097e-02;
        xcrref[2] = 3.2527926989486055e-02;
        xcrref[3] = 2.6436421275166801e-02;
        xcrref[4] = 1.9211784131744430e-01;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error.
         * ---------------------------------------------------------------------
         */
        xceref[0] = 4.9976913345811579e-04;
        xceref[1] = 4.5195666782961927e-05;
        xceref[2] = 7.3973765172921357e-05;
        xceref[3] = 7.3821238632439731e-05;
        xceref[4] = 8.9269630987491446e-04;

        /*
         * ---------------------------------------------------------------------
         * reference data for 24X24X24 grids after 200 time steps, with DT =
         * 0.8d-3
         * ---------------------------------------------------------------------
         */
    } else if ((grid_points[0] == 24) && (grid_points[1] == 24) &&
               (grid_points[2] == 24) && (no_time_steps == 200)) {
        *class_npb = 'W';
        dtref = 0.8e-3;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual.
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 0.1125590409344e+03;
        xcrref[1] = 0.1180007595731e+02;
        xcrref[2] = 0.2710329767846e+02;
        xcrref[3] = 0.2469174937669e+02;
        xcrref[4] = 0.2638427874317e+03;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error.
         * ---------------------------------------------------------------------
         */
        xceref[0] = 0.4419655736008e+01;
        xceref[1] = 0.4638531260002e+00;
        xceref[2] = 0.1011551749967e+01;
        xceref[3] = 0.9235878729944e+00;
        xceref[4] = 0.1018045837718e+02;

        /*
         * ---------------------------------------------------------------------
         * reference data for 64X64X64 grids after 200 time steps, with DT =
         * 0.8d-3
         * ---------------------------------------------------------------------
         */
    } else if ((grid_points[0] == 64) && (grid_points[1] == 64) &&
               (grid_points[2] == 64) && (no_time_steps == 200)) {
        *class_npb = 'A';
        dtref = 0.8e-3;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual.
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 1.0806346714637264e+02;
        xcrref[1] = 1.1319730901220813e+01;
        xcrref[2] = 2.5974354511582465e+01;
        xcrref[3] = 2.3665622544678910e+01;
        xcrref[4] = 2.5278963211748344e+02;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error.
         * ---------------------------------------------------------------------
         */
        xceref[0] = 4.2348416040525025e+00;
        xceref[1] = 4.4390282496995698e-01;
        xceref[2] = 9.6692480136345650e-01;
        xceref[3] = 8.8302063039765474e-01;
        xceref[4] = 9.7379901770829278e+00;

        /*
         * ---------------------------------------------------------------------
         * reference data for 102X102X102 grids after 200 time steps,
         * with DT = 3.0e-04
         * ---------------------------------------------------------------------
         */
    } else if ((grid_points[0] == 102) && (grid_points[1] == 102) &&
               (grid_points[2] == 102) && (no_time_steps == 200)) {
        *class_npb = 'B';
        dtref = 3.0e-4;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual.
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 1.4233597229287254e+03;
        xcrref[1] = 9.9330522590150238e+01;
        xcrref[2] = 3.5646025644535285e+02;
        xcrref[3] = 3.2485447959084092e+02;
        xcrref[4] = 3.2707541254659363e+03;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error.
         * ---------------------------------------------------------------------
         */
        xceref[0] = 5.2969847140936856e+01;
        xceref[1] = 4.4632896115670668e+00;
        xceref[2] = 1.3122573342210174e+01;
        xceref[3] = 1.2006925323559144e+01;
        xceref[4] = 1.2459576151035986e+02;

        /*
         * ---------------------------------------------------------------------
         * reference data for 162X162X162 grids after 200 time steps,
         * with DT = 1.0e-04
         * ---------------------------------------------------------------------
         */
    } else if ((grid_points[0] == 162) && (grid_points[1] == 162) &&
               (grid_points[2] == 162) && (no_time_steps == 200)) {
        *class_npb = 'C';
        dtref = 1.0e-4;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual.
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 0.62398116551764615e+04;
        xcrref[1] = 0.50793239190423964e+03;
        xcrref[2] = 0.15423530093013596e+04;
        xcrref[3] = 0.13302387929291190e+04;
        xcrref[4] = 0.11604087428436455e+05;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error.
         * ---------------------------------------------------------------------
         */
        xceref[0] = 0.16462008369091265e+03;
        xceref[1] = 0.11497107903824313e+02;
        xceref[2] = 0.41207446207461508e+02;
        xceref[3] = 0.37087651059694167e+02;
        xceref[4] = 0.36211053051841265e+03;

        /*
         * ---------------------------------------------------------------------
         * reference data for 408x408x408 grids after 250 time steps,
         * with DT = 0.2e-04
         * ---------------------------------------------------------------------
         */
    } else if ((grid_points[0] == 408) && (grid_points[1] == 408) &&
               (grid_points[2] == 408) && (no_time_steps == 250)) {
        *class_npb = 'D';
        dtref = 0.2e-4;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual.
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 0.2533188551738e+05;
        xcrref[1] = 0.2346393716980e+04;
        xcrref[2] = 0.6294554366904e+04;
        xcrref[3] = 0.5352565376030e+04;
        xcrref[4] = 0.3905864038618e+05;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error.
         * ---------------------------------------------------------------------
         */
        xceref[0] = 0.3100009377557e+03;
        xceref[1] = 0.2424086324913e+02;
        xceref[2] = 0.7782212022645e+02;
        xceref[3] = 0.6835623860116e+02;
        xceref[4] = 0.6065737200368e+03;

        /*
         * ---------------------------------------------------------------------
         * reference data for 1020x1020x1020 grids after 250 time steps,
         * with DT = 0.4e-05
         * ---------------------------------------------------------------------
         */
    } else if ((grid_points[0] == 1020) && (grid_points[1] == 1020) &&
               (grid_points[2] == 1020) && (no_time_steps == 250)) {
        *class_npb = 'E';
        dtref = 0.4e-5;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of residual.
         * ---------------------------------------------------------------------
         */
        xcrref[0] = 0.9795372484517e+05;
        xcrref[1] = 0.9739814511521e+04;
        xcrref[2] = 0.2467606342965e+05;
        xcrref[3] = 0.2092419572860e+05;
        xcrref[4] = 0.1392138856939e+06;

        /*
         * ---------------------------------------------------------------------
         * reference values of RMS-norms of solution error.
         * ---------------------------------------------------------------------
         */
        xceref[0] = 0.4327562208414e+03;
        xceref[1] = 0.3699051964887e+02;
        xceref[2] = 0.1089845040954e+03;
        xceref[3] = 0.9462517622043e+02;
        xceref[4] = 0.7765512765309e+03;

    } else {
        *verified = false;
    }

    /*
     * ---------------------------------------------------------------------
     * verification test for residuals if gridsize is one of
     * the defined grid sizes above (*class_npb != 'U')
     * ---------------------------------------------------------------------
     * compute the difference of solution values and the known reference values.
     * ---------------------------------------------------------------------
     */
    for (m = 0; m < 5; m++) {
        xcrdif[m] = fabs((xcr[m] - xcrref[m]) / xcrref[m]);
        xcedif[m] = fabs((xce[m] - xceref[m]) / xceref[m]);
    }

    /*
     * ---------------------------------------------------------------------
     * output the comparison of computed results to known cases.
     * ---------------------------------------------------------------------
     */
    if (*class_npb != 'U') {
        // printf(" Verification being performed for class %c\n", *class_npb);
        // printf(" accuracy setting for epsilon = %20.13E\n", epsilon);
        *verified = (fabs(dt - dtref) <= epsilon);
        if (!(*verified)) {
            *class_npb = 'U';
            // printf(" DT does not match the reference value of %15.8E\n",
            // dtref);
        }
    } else {
        // printf(" Unknown class_npb\n");
    }

    // if (*class_npb != 'U') {
    //     printf(" Comparison of RMS-norms of residual\n");
    // } else {
    //     printf(" RMS-norms of residual\n");
    // }

    for (m = 0; m < 5; m++) {
        if (*class_npb == 'U') {
            // printf("          %2d%20.13E\n", m + 1, xcr[m]);
        } else if (xcrdif[m] <= epsilon) {
            // printf("          %2d%20.13E%20.13E%20.13E\n", m + 1, xcr[m],
            //        xcrref[m], xcrdif[m]);
        } else {
            *verified = false;
            // printf(" FAILURE: %2d%20.13E%20.13E%20.13E\n", m + 1, xcr[m],
            //        xcrref[m], xcrdif[m]);
        }
    }

    // if (*class_npb != 'U') {
    //     printf(" Comparison of RMS-norms of solution error\n");
    // } else {
    //     printf(" RMS-norms of solution error\n");
    // }

    for (m = 0; m < 5; m++) {
        if (*class_npb == 'U') {
            // printf("          %2d%20.13E\n", m + 1, xce[m]);
        } else if (xcedif[m] <= epsilon) {
            // printf("          %2d%20.13E%20.13E%20.13E\n", m + 1, xce[m],
            //        xceref[m], xcedif[m]);
        } else {
            *verified = false;
            // printf(" FAILURE: %2d%20.13E%20.13E%20.13E\n", m + 1, xce[m],
            //        xceref[m], xcedif[m]);
        }
    }

    // if (*class_npb == 'U') {
    //     printf(" No reference values provided\n");
    //     printf(" No verification performed\n");
    // } else if (*verified) {
    //     printf(" Verification Successful\n");
    // } else {
    //     printf(" Verification failed\n");
    // }
}

/*
 * ---------------------------------------------------------------------
 * performs line solves in X direction by first factoring
 * the block-tridiagonal matrix into an upper triangular matrix,
 * and then performing back substitution to solve for the unknow
 * vectors of each line.
 *
 * make sure we treat elements zero to cell_size in the direction
 * of the sweep.
 * ---------------------------------------------------------------------
 */
void x_solve_gpu() {
    size_t amount_of_threads[3];
    size_t amount_of_work[3];
    /*
     * ---------------------------------------------------------------------
     * this function computes the left hand side in the xi-direction
     * ---------------------------------------------------------------------
     * determine a (labeled f) and n jacobians
     * ---------------------------------------------------------------------
     */
    amount_of_work[1] = PROBLEM_SIZE * 5 * 5;
    amount_of_work[0] = grid_points[1] - 2;

    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_X_SOLVE_1;

    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    dim3 blockSize(amount_of_work[0] / amount_of_threads[0],
                   amount_of_work[1] / amount_of_threads[1]);
    dim3 threadSize(amount_of_threads[0], amount_of_threads[1]);

#if defined(PROFILING)
    timer_start(PROFILING_X_SOLVE_1);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    x_solve_gpu_kernel_1<<<blockSize, threadSize, 0>>>(lhsA_device, lhsB_device,
                                                       lhsC_device);
#if defined(PROFILING)
    timer_stop(PROFILING_X_SOLVE_1);
#endif

    amount_of_work[2] = (size_t)PROBLEM_SIZE;
    amount_of_work[1] = (size_t)grid_points[1] - 2;
    amount_of_work[0] = (size_t)grid_points[0] - 2;

    amount_of_threads[2] = 1;
    amount_of_threads[1] = THREADS_PER_BLOCK_ON_X_SOLVE_2;
    amount_of_threads[0] = 1;

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = amount_of_work[2] / amount_of_threads[2];
    threadSize.z = amount_of_threads[2];

#if defined(PROFILING)
    timer_start(PROFILING_X_SOLVE_2);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    x_solve_gpu_kernel_2<<<blockSize, threadSize, 0>>>(
        qs_device, rho_i_device, square_device, u_device, lhsA_device,
        lhsB_device, lhsC_device);
#if defined(PROFILING)
    timer_stop(PROFILING_X_SOLVE_2);
#endif

    size_t max_amount_of_threads_j =
        min(THREADS_PER_BLOCK_ON_X_SOLVE_3 / 5,
            gpu_device_properties.sharedMemPerBlock /
                (sizeof(double) * (3 * 5 * 5 + 2 * 5)));
    max_amount_of_threads_j /= 2;

    amount_of_threads[1] = 5;
    amount_of_threads[0] = max_amount_of_threads_j;

    amount_of_work[1] = (size_t)PROBLEM_SIZE;
    amount_of_work[1] *= 5;
    amount_of_work[0] = (size_t)grid_points[1] - 2;

    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = 1;
    threadSize.z = 1;

#if defined(PROFILING)
    timer_start(PROFILING_X_SOLVE_3);
#endif
    // printf("threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    x_solve_gpu_kernel_3<<<blockSize, threadSize,
                           sizeof(double) * max_amount_of_threads_j *
                               (3 * 5 * 5 + 2 * 5)>>>(rhs_device, lhsA_device,
                                                      lhsB_device, lhsC_device);
#if defined(PROFILING)
    timer_stop(PROFILING_X_SOLVE_3);
#endif
}

/*
 * ---------------------------------------------------------------------
 * performs line solves in y direction by first factoring
 * the block-tridiagonal matrix into an upper triangular matrix,
 * and then performing back substitution to solve for the unknow
 * vectors of each line.
 *
 * make sure we treat elements zero to cell_size in the direction
 * of the sweep.
 * ---------------------------------------------------------------------
 */
void y_solve_gpu() {
    size_t amount_of_threads[3];
    size_t amount_of_work[3];
    /*
     * ---------------------------------------------------------------------
     * this function computes the left hand side for the three y-factors
     * ---------------------------------------------------------------------
     * compute the indices for storing the tri-diagonal matrix;
     * determine a (labeled f) and n jacobians for cell c
     * ---------------------------------------------------------------------
     */

    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_Y_SOLVE_1;

    amount_of_work[1] = (size_t)PROBLEM_SIZE;
    amount_of_work[1] *= 25;
    amount_of_work[0] = (size_t)grid_points[0] - 2;

    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    dim3 blockSize(amount_of_work[0] / amount_of_threads[0],
                   amount_of_work[1] / amount_of_threads[1]);
    dim3 threadSize(amount_of_threads[0], amount_of_threads[1]);

#if defined(PROFILING)
    timer_start(PROFILING_Y_SOLVE_1);
#endif
    // printf("1threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    y_solve_gpu_kernel_1<<<blockSize, threadSize, 0>>>(lhsA_device, lhsB_device,
                                                       lhsC_device);
#if defined(PROFILING)
    timer_stop(PROFILING_Y_SOLVE_1);
#endif

    amount_of_work[2] = (size_t)PROBLEM_SIZE;
    amount_of_work[1] = (size_t)grid_points[1] - 2;
    amount_of_work[0] = (size_t)grid_points[0] - 2;

    amount_of_threads[2] = 1;
    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_Y_SOLVE_2;

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = amount_of_work[2] / amount_of_threads[2];
    threadSize.z = amount_of_threads[2];

#if defined(PROFILING)
    timer_start(PROFILING_Y_SOLVE_2);
#endif
    // printf("2threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    y_solve_gpu_kernel_2<<<blockSize, threadSize, 0>>>(
        qs_device, rho_i_device, square_device, u_device, lhsA_device,
        lhsB_device, lhsC_device);
#if defined(PROFILING)
    timer_stop(PROFILING_Y_SOLVE_2);
#endif

    size_t max_amount_of_threads_i =
        min(THREADS_PER_BLOCK_ON_Y_SOLVE_3 / 5,
            gpu_device_properties.sharedMemPerBlock /
                (sizeof(double) * (3 * 5 * 5 + 2 * 5)));
    max_amount_of_threads_i /= 2;

    amount_of_threads[1] = 5;
    amount_of_threads[0] = max_amount_of_threads_i;

    amount_of_work[1] = (size_t)PROBLEM_SIZE;
    amount_of_work[1] *= 5;
    amount_of_work[0] = (size_t)grid_points[0] - 2;

    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = 1;
    threadSize.z = 1;

#if defined(PROFILING)
    timer_start(PROFILING_Y_SOLVE_3);
#endif
    // printf("3threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    y_solve_gpu_kernel_3<<<blockSize, threadSize,
                           sizeof(double) * max_amount_of_threads_i *
                               (3 * 5 * 5 + 2 * 5)>>>(rhs_device, lhsA_device,
                                                      lhsB_device, lhsC_device);
#if defined(PROFILING)
    timer_stop(PROFILING_Y_SOLVE_3);
#endif
}

/*
 * ---------------------------------------------------------------------
 * performs line solves in Z direction by first factoring
 * the block-tridiagonal matrix into an upper triangular matrix,
 * and then performing back substitution to solve for the unknow
 * vectors of each line.
 *
 * make sure we treat elements zero to cell_size in the direction
 * of the sweep.
 * ---------------------------------------------------------------------
 */
void z_solve_gpu() {
    size_t amount_of_threads[3];
    size_t amount_of_work[3];

    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_Z_SOLVE_1;

    amount_of_work[1] = PROBLEM_SIZE * 25;
    amount_of_work[0] = grid_points[0] - 2;

    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    dim3 blockSize(amount_of_work[0] / amount_of_threads[0],
                   amount_of_work[1] / amount_of_threads[1]);
    dim3 threadSize(amount_of_threads[0], amount_of_threads[1]);

#if defined(PROFILING)
    timer_start(PROFILING_Z_SOLVE_1);
#endif
    // printf("1threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    z_solve_gpu_kernel_1<<<blockSize, threadSize, 0>>>(lhsA_device, lhsB_device,
                                                       lhsC_device);
#if defined(PROFILING)
    timer_stop(PROFILING_Z_SOLVE_1);
#endif

    amount_of_threads[2] = 1;
    amount_of_threads[1] = 1;
    amount_of_threads[0] = THREADS_PER_BLOCK_ON_Z_SOLVE_2;

    amount_of_work[2] = (size_t)grid_points[2] - 2;
    amount_of_work[1] = (size_t)PROBLEM_SIZE;
    amount_of_work[0] = (size_t)grid_points[0] - 2;

    amount_of_work[2] =
        round_amount_of_work(amount_of_work[2], amount_of_threads[2]);
    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = amount_of_work[2] / amount_of_threads[2];
    threadSize.z = amount_of_threads[2];

#if defined(PROFILING)
    timer_start(PROFILING_Z_SOLVE_2);
#endif
    // printf("2threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    z_solve_gpu_kernel_2<<<blockSize, threadSize, 0>>>(
        qs_device, square_device, u_device, lhsA_device, lhsB_device,
        lhsC_device);
#if defined(PROFILING)
    timer_stop(PROFILING_Z_SOLVE_2);
#endif

    size_t max_amount_of_threads_i =
        min(THREADS_PER_BLOCK_ON_Z_SOLVE_3 / 5,
            gpu_device_properties.sharedMemPerBlock /
                (sizeof(double) * (3 * 5 * 5 + 2 * 5)));

    amount_of_threads[1] = 5;
    amount_of_threads[0] = max_amount_of_threads_i;

    amount_of_work[1] = (size_t)PROBLEM_SIZE;
    amount_of_work[1] *= 5;
    amount_of_work[0] = (size_t)grid_points[0] - 2;

    amount_of_work[1] =
        round_amount_of_work(amount_of_work[1], amount_of_threads[1]);
    amount_of_work[0] =
        round_amount_of_work(amount_of_work[0], amount_of_threads[0]);

    blockSize.x = amount_of_work[0] / amount_of_threads[0];
    threadSize.x = amount_of_threads[0];
    blockSize.y = amount_of_work[1] / amount_of_threads[1];
    threadSize.y = amount_of_threads[1];
    blockSize.z = 1;
    threadSize.z = 1;

#if defined(PROFILING)
    timer_start(PROFILING_Z_SOLVE_3);
#endif
    // printf("3threadSize=[%d, %d, %d]\n", threadSize.x, threadSize.y,
    // threadSize.z);
    z_solve_gpu_kernel_3<<<blockSize, threadSize,
                           sizeof(double) * max_amount_of_threads_i *
                               (3 * 5 * 5 + 2 * 5)>>>(rhs_device, lhsA_device,
                                                      lhsB_device, lhsC_device);
#if defined(PROFILING)
    timer_stop(PROFILING_Z_SOLVE_3);
#endif
}
