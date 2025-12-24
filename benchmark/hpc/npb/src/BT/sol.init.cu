#include "bt.cuh"

__global__ void add_gpu_kernel(double* u_device, double* rhs_device) {
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    int i = t_i / 5 + 1;
    int m = t_i % 5;

    if (k > KMAX - 2 || j + 1 < 1 || j + 1 > JMAX - 2 || j >= PROBLEM_SIZE ||
        i > IMAX - 2)
        return;

    j++;

    u_device[((k * (JMAXP + 1) + j) * (IMAXP + 1) + i) * 5 + m] +=
        rhs_device[((k * (JMAXP + 1) + j) * (IMAXP + 1) + i) * 5 + m];
}

/*
 * ---------------------------------------------------------------------
 * compute the reciprocal of density, and the kinetic energy,
 * and the speed of sound.
 * ---------------------------------------------------------------------
 */
__global__ void compute_rhs_gpu_kernel_1(double* rho_i_device,
                                         double* us_device, double* vs_device,
                                         double* ws_device, double* qs_device,
                                         double* square_device,
                                         double* u_device) {
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= KMAX || j > JMAX - 1 || i > IMAX - 1) return;

    double (*rho_i)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) rho_i_device;

    double (*us)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) us_device;

    double (*vs)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) vs_device;

    double (*ws)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) ws_device;

    double (*qs)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) qs_device;

    double (*square)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) square_device;

    double (*u)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) u_device;

    double rho_inv;
    double t_u[4];
    int m;

    for (m = 0; m < 4; m++) t_u[m] = u[k][j][i][m];

    rho_inv = 1.0 / t_u[0];
    rho_i[k][j][i] = rho_inv;
    us[k][j][i] = t_u[1] * rho_inv;
    vs[k][j][i] = t_u[2] * rho_inv;
    ws[k][j][i] = t_u[3] * rho_inv;
    square[k][j][i] =
        0.5 * (t_u[1] * t_u[1] + t_u[2] * t_u[2] + t_u[3] * t_u[3]) * rho_inv;
    qs[k][j][i] = square[k][j][i] * rho_inv;
}

/*
 * ---------------------------------------------------------------------
 * copy the exact forcing term to the right hand side; because
 * this forcing term is known, we can store it on the whole grid
 * including the boundary
 * ---------------------------------------------------------------------
 */
__global__ void compute_rhs_gpu_kernel_2(double* rhs_device,
                                         double* forcing_device) {
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    int i = t_i / 5;
    int m = t_i % 5;

    if (k + 0 > KMAX - 1 || k >= KMAX || j > JMAX - 1 || i > IMAX - 1) return;

    double (*rhs)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) rhs_device;
    double (*forcing)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) forcing_device;

    rhs[k][j][i][m] = forcing[k][j][i][m];
}

/*
 * ---------------------------------------------------------------------
 * compute xi-direction fluxes
 * ---------------------------------------------------------------------
 */
__global__ void compute_rhs_gpu_kernel_3(double* us_device, double* vs_device,
                                         double* ws_device, double* qs_device,
                                         double* rho_i_device,
                                         double* square_device,
                                         double* u_device, double* rhs_device) {
    /* 1 <= k <= KMAX-2 */
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= KMAX || j > JMAX - 2 ||
        i > IMAX - 2)
        return;

    double uijk, up1, um1;

    double (*us)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) us_device;
    double (*vs)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) vs_device;
    double (*ws)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) ws_device;
    double (*qs)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) qs_device;
    double (*rho_i)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) rho_i_device;
    double (*square)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) square_device;
    double (*u)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) u_device;
    double (*rhs)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) rhs_device;

    uijk = us[k][j][i];
    up1 = us[k][j][i + 1];
    um1 = us[k][j][i - 1];

    rhs[k][j][i][0] =
        rhs[k][j][i][0] +
        constants_device::dx1tx1 *
            (u[k][j][i + 1][0] - 2.0 * u[k][j][i][0] + u[k][j][i - 1][0]) -
        constants_device::tx2 * (u[k][j][i + 1][1] - u[k][j][i - 1][1]);

    rhs[k][j][i][1] =
        rhs[k][j][i][1] +
        constants_device::dx2tx1 *
            (u[k][j][i + 1][1] - 2.0 * u[k][j][i][1] + u[k][j][i - 1][1]) +
        constants_device::xxcon2 * constants_device::con43 *
            (up1 - 2.0 * uijk + um1) -
        constants_device::tx2 *
            (u[k][j][i + 1][1] * up1 - u[k][j][i - 1][1] * um1 +
             (u[k][j][i + 1][4] - square[k][j][i + 1] - u[k][j][i - 1][4] +
              square[k][j][i - 1]) *
                 constants_device::c2);

    rhs[k][j][i][2] =
        rhs[k][j][i][2] +
        constants_device::dx3tx1 *
            (u[k][j][i + 1][2] - 2.0 * u[k][j][i][2] + u[k][j][i - 1][2]) +
        constants_device::xxcon2 *
            (vs[k][j][i + 1] - 2.0 * vs[k][j][i] + vs[k][j][i - 1]) -
        constants_device::tx2 *
            (u[k][j][i + 1][2] * up1 - u[k][j][i - 1][2] * um1);

    rhs[k][j][i][3] =
        rhs[k][j][i][3] +
        constants_device::dx4tx1 *
            (u[k][j][i + 1][3] - 2.0 * u[k][j][i][3] + u[k][j][i - 1][3]) +
        constants_device::xxcon2 *
            (ws[k][j][i + 1] - 2.0 * ws[k][j][i] + ws[k][j][i - 1]) -
        constants_device::tx2 *
            (u[k][j][i + 1][3] * up1 - u[k][j][i - 1][3] * um1);

    rhs[k][j][i][4] =
        rhs[k][j][i][4] +
        constants_device::dx5tx1 *
            (u[k][j][i + 1][4] - 2.0 * u[k][j][i][4] + u[k][j][i - 1][4]) +
        constants_device::xxcon3 *
            (qs[k][j][i + 1] - 2.0 * qs[k][j][i] + qs[k][j][i - 1]) +
        constants_device::xxcon4 * (up1 * up1 - 2.0 * uijk * uijk + um1 * um1) +
        constants_device::xxcon5 * (u[k][j][i + 1][4] * rho_i[k][j][i + 1] -
                                    2.0 * u[k][j][i][4] * rho_i[k][j][i] +
                                    u[k][j][i - 1][4] * rho_i[k][j][i - 1]) -
        constants_device::tx2 * ((constants_device::c1 * u[k][j][i + 1][4] -
                                  constants_device::c2 * square[k][j][i + 1]) *
                                     up1 -
                                 (constants_device::c1 * u[k][j][i - 1][4] -
                                  constants_device::c2 * square[k][j][i - 1]) *
                                     um1);
}

__global__ void compute_rhs_gpu_kernel_4(double* u_device, double* rhs_device) {
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    int i = t_i / 5 + 1;
    int m = t_i % 5;

    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= KMAX || j > JMAX - 2 ||
        i > IMAX - 2)
        return;

    double (*u)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) u_device;
    double (*rhs)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) rhs_device;

    if (i == 1) {
        rhs[k][j][i][m] =
            rhs[k][j][i][m] - constants_device::dssp *
                                  (5.0 * u[k][j][i][m] -
                                   4.0 * u[k][j][i + 1][m] + u[k][j][i + 2][m]);
    } else if (i == 2) {
        rhs[k][j][i][m] = rhs[k][j][i][m] -
                          constants_device::dssp *
                              (-4.0 * u[k][j][i - 1][m] + 6.0 * u[k][j][i][m] -
                               4.0 * u[k][j][i + 1][m] + u[k][j][i + 2][m]);
    } else if (i == IMAX - 3) {
        rhs[k][j][i][m] = rhs[k][j][i][m] -
                          constants_device::dssp *
                              (u[k][j][i - 2][m] - 4.0 * u[k][j][i - 1][m] +
                               6.0 * u[k][j][i][m] - 4.0 * u[k][j][i + 1][m]);
    } else if (i == IMAX - 2) {
        rhs[k][j][i][m] =
            rhs[k][j][i][m] - constants_device::dssp *
                                  (u[k][j][i - 2][m] - 4.0 * u[k][j][i - 1][m] +
                                   5.0 * u[k][j][i][m]);
    } else {
        rhs[k][j][i][m] =
            rhs[k][j][i][m] - constants_device::dssp *
                                  (u[k][j][i - 2][m] - 4.0 * u[k][j][i - 1][m] +
                                   6.0 * u[k][j][i][m] -
                                   4.0 * u[k][j][i + 1][m] + u[k][j][i + 2][m]);
    }
}

/*
 * ---------------------------------------------------------------------
 * compute eta-direction fluxes
 * ---------------------------------------------------------------------
 * Input(write buffer) - us_device, vs_device, ws_device, qs_device,
 * rho_i_device, square_device, u_device, rhs_device
 * ---------------------------------------------------------------------
 * Input(write buffer) - us_device, vs_device, ws_device, qs_device,
 * rho_i_device, square_device, u_device, rhs_device
 * ---------------------------------------------------------------------
 * Output(read buffer) - rhs_device
 * ---------------------------------------------------------------------
 */
__global__ void compute_rhs_gpu_kernel_5(double* us_device, double* vs_device,
                                         double* ws_device, double* qs_device,
                                         double* rho_i_device,
                                         double* square_device,
                                         double* u_device, double* rhs_device) {
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= KMAX || j > JMAX - 2 ||
        i > IMAX - 2)
        return;

    double vijk, vp1, vm1;

    double (*us)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) us_device;
    double (*vs)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) vs_device;
    double (*ws)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) ws_device;
    double (*qs)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) qs_device;
    double (*rho_i)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) rho_i_device;
    double (*square)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) square_device;
    double (*u)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) u_device;
    double (*rhs)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) rhs_device;

    vijk = vs[k][j][i];
    vp1 = vs[k][j + 1][i];
    vm1 = vs[k][j - 1][i];
    rhs[k][j][i][0] =
        rhs[k][j][i][0] +
        constants_device::dy1ty1 *
            (u[k][j + 1][i][0] - 2.0 * u[k][j][i][0] + u[k][j - 1][i][0]) -
        constants_device::ty2 * (u[k][j + 1][i][2] - u[k][j - 1][i][2]);
    rhs[k][j][i][1] =
        rhs[k][j][i][1] +
        constants_device::dy2ty1 *
            (u[k][j + 1][i][1] - 2.0 * u[k][j][i][1] + u[k][j - 1][i][1]) +
        constants_device::yycon2 *
            (us[k][j + 1][i] - 2.0 * us[k][j][i] + us[k][j - 1][i]) -
        constants_device::ty2 *
            (u[k][j + 1][i][1] * vp1 - u[k][j - 1][i][1] * vm1);
    rhs[k][j][i][2] =
        rhs[k][j][i][2] +
        constants_device::dy3ty1 *
            (u[k][j + 1][i][2] - 2.0 * u[k][j][i][2] + u[k][j - 1][i][2]) +
        constants_device::yycon2 * constants_device::con43 *
            (vp1 - 2.0 * vijk + vm1) -
        constants_device::ty2 *
            (u[k][j + 1][i][2] * vp1 - u[k][j - 1][i][2] * vm1 +
             (u[k][j + 1][i][4] - square[k][j + 1][i] - u[k][j - 1][i][4] +
              square[k][j - 1][i]) *
                 constants_device::c2);
    rhs[k][j][i][3] =
        rhs[k][j][i][3] +
        constants_device::dy4ty1 *
            (u[k][j + 1][i][3] - 2.0 * u[k][j][i][3] + u[k][j - 1][i][3]) +
        constants_device::yycon2 *
            (ws[k][j + 1][i] - 2.0 * ws[k][j][i] + ws[k][j - 1][i]) -
        constants_device::ty2 *
            (u[k][j + 1][i][3] * vp1 - u[k][j - 1][i][3] * vm1);
    rhs[k][j][i][4] =
        rhs[k][j][i][4] +
        constants_device::dy5ty1 *
            (u[k][j + 1][i][4] - 2.0 * u[k][j][i][4] + u[k][j - 1][i][4]) +
        constants_device::yycon3 *
            (qs[k][j + 1][i] - 2.0 * qs[k][j][i] + qs[k][j - 1][i]) +
        constants_device::yycon4 * (vp1 * vp1 - 2.0 * vijk * vijk + vm1 * vm1) +
        constants_device::yycon5 * (u[k][j + 1][i][4] * rho_i[k][j + 1][i] -
                                    2.0 * u[k][j][i][4] * rho_i[k][j][i] +
                                    u[k][j - 1][i][4] * rho_i[k][j - 1][i]) -
        constants_device::ty2 * ((constants_device::c1 * u[k][j + 1][i][4] -
                                  constants_device::c2 * square[k][j + 1][i]) *
                                     vp1 -
                                 (constants_device::c1 * u[k][j - 1][i][4] -
                                  constants_device::c2 * square[k][j - 1][i]) *
                                     vm1);
}

__global__ void compute_rhs_gpu_kernel_6(double* u_device, double* rhs_device) {
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    int i = t_i / 5 + 1;
    int m = t_i % 5;

    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= KMAX || j > JMAX - 2 ||
        i > IMAX - 2)
        return;

    double (*u)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) u_device;
    double (*rhs)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) rhs_device;

    if (j == 1) {
        rhs[k][j][i][m] =
            rhs[k][j][i][m] - constants_device::dssp *
                                  (5.0 * u[k][j][i][m] -
                                   4.0 * u[k][j + 1][i][m] + u[k][j + 2][i][m]);
    } else if (j == 2) {
        rhs[k][j][i][m] = rhs[k][j][i][m] -
                          constants_device::dssp *
                              (-4.0 * u[k][j - 1][i][m] + 6.0 * u[k][j][i][m] -
                               4.0 * u[k][j + 1][i][m] + u[k][j + 2][i][m]);
    } else if (j == JMAX - 3) {
        rhs[k][j][i][m] = rhs[k][j][i][m] -
                          constants_device::dssp *
                              (u[k][j - 2][i][m] - 4.0 * u[k][j - 1][i][m] +
                               6.0 * u[k][j][i][m] - 4.0 * u[k][j + 1][i][m]);
    } else if (j == JMAX - 2) {
        rhs[k][j][i][m] =
            rhs[k][j][i][m] - constants_device::dssp *
                                  (u[k][j - 2][i][m] - 4. * u[k][j - 1][i][m] +
                                   5. * u[k][j][i][m]);
    } else {
        rhs[k][j][i][m] =
            rhs[k][j][i][m] - constants_device::dssp *
                                  (u[k][j - 2][i][m] - 4.0 * u[k][j - 1][i][m] +
                                   6.0 * u[k][j][i][m] -
                                   4.0 * u[k][j + 1][i][m] + u[k][j + 2][i][m]);
    }
}

__global__ void compute_rhs_gpu_kernel_7(double* us_device, double* vs_device,
                                         double* ws_device, double* qs_device,
                                         double* rho_i_device,
                                         double* square_device,
                                         double* u_device, double* rhs_device) {
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= KMAX || j > JMAX - 2 ||
        i > IMAX - 2)
        return;

    double wijk, wp1, wm1;

    double (*us)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) us_device;
    double (*vs)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) vs_device;
    double (*ws)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) ws_device;
    double (*qs)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) qs_device;
    double (*rho_i)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) rho_i_device;
    double (*square)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) square_device;
    double (*u)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) u_device;
    double (*rhs)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) rhs_device;

    wijk = ws[k][j][i];
    wp1 = ws[k + 1][j][i];
    wm1 = ws[k - 1][j][i];

    rhs[k][j][i][0] =
        rhs[k][j][i][0] +
        constants_device::dz1tz1 *
            (u[k + 1][j][i][0] - 2.0 * u[k][j][i][0] + u[k - 1][j][i][0]) -
        constants_device::tz2 * (u[k + 1][j][i][3] - u[k - 1][j][i][3]);
    rhs[k][j][i][1] =
        rhs[k][j][i][1] +
        constants_device::dz2tz1 *
            (u[k + 1][j][i][1] - 2.0 * u[k][j][i][1] + u[k - 1][j][i][1]) +
        constants_device::zzcon2 *
            (us[k + 1][j][i] - 2.0 * us[k][j][i] + us[k - 1][j][i]) -
        constants_device::tz2 *
            (u[k + 1][j][i][1] * wp1 - u[k - 1][j][i][1] * wm1);
    rhs[k][j][i][2] =
        rhs[k][j][i][2] +
        constants_device::dz3tz1 *
            (u[k + 1][j][i][2] - 2.0 * u[k][j][i][2] + u[k - 1][j][i][2]) +
        constants_device::zzcon2 *
            (vs[k + 1][j][i] - 2.0 * vs[k][j][i] + vs[k - 1][j][i]) -
        constants_device::tz2 *
            (u[k + 1][j][i][2] * wp1 - u[k - 1][j][i][2] * wm1);
    rhs[k][j][i][3] =
        rhs[k][j][i][3] +
        constants_device::dz4tz1 *
            (u[k + 1][j][i][3] - 2.0 * u[k][j][i][3] + u[k - 1][j][i][3]) +
        constants_device::zzcon2 * constants_device::con43 *
            (wp1 - 2.0 * wijk + wm1) -
        constants_device::tz2 *
            (u[k + 1][j][i][3] * wp1 - u[k - 1][j][i][3] * wm1 +
             (u[k + 1][j][i][4] - square[k + 1][j][i] - u[k - 1][j][i][4] +
              square[k - 1][j][i]) *
                 constants_device::c2);
    rhs[k][j][i][4] =
        rhs[k][j][i][4] +
        constants_device::dz5tz1 *
            (u[k + 1][j][i][4] - 2.0 * u[k][j][i][4] + u[k - 1][j][i][4]) +
        constants_device::zzcon3 *
            (qs[k + 1][j][i] - 2.0 * qs[k][j][i] + qs[k - 1][j][i]) +
        constants_device::zzcon4 * (wp1 * wp1 - 2.0 * wijk * wijk + wm1 * wm1) +
        constants_device::zzcon5 * (u[k + 1][j][i][4] * rho_i[k + 1][j][i] -
                                    2.0 * u[k][j][i][4] * rho_i[k][j][i] +
                                    u[k - 1][j][i][4] * rho_i[k - 1][j][i]) -
        constants_device::tz2 * ((constants_device::c1 * u[k + 1][j][i][4] -
                                  constants_device::c2 * square[k + 1][j][i]) *
                                     wp1 -
                                 (constants_device::c1 * u[k - 1][j][i][4] -
                                  constants_device::c2 * square[k - 1][j][i]) *
                                     wm1);
}

__global__ void compute_rhs_gpu_kernel_8(double* u_device, double* rhs_device) {
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    int i = t_i / 5 + 1;
    int m = t_i % 5;
    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= KMAX || j > JMAX - 2 ||
        i > IMAX - 2)
        return;

    double (*u)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) u_device;
    double (*rhs)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) rhs_device;

    if (k == 1) {
        rhs[k][j][i][m] =
            rhs[k][j][i][m] - constants_device::dssp *
                                  (5.0 * u[k][j][i][m] -
                                   4.0 * u[k + 1][j][i][m] + u[k + 2][j][i][m]);
    } else if (k == 2) {
        rhs[k][j][i][m] = rhs[k][j][i][m] -
                          constants_device::dssp *
                              (-4.0 * u[k - 1][j][i][m] + 6.0 * u[k][j][i][m] -
                               4.0 * u[k + 1][j][i][m] + u[k + 2][j][i][m]);
    } else if (k == KMAX - 3) {
        rhs[k][j][i][m] = rhs[k][j][i][m] -
                          constants_device::dssp *
                              (u[k - 2][j][i][m] - 4.0 * u[k - 1][j][i][m] +
                               6.0 * u[k][j][i][m] - 4.0 * u[k + 1][j][i][m]);
    } else if (k == KMAX - 2) {
        rhs[k][j][i][m] =
            rhs[k][j][i][m] - constants_device::dssp *
                                  (u[k - 2][j][i][m] - 4. * u[k - 1][j][i][m] +
                                   5. * u[k][j][i][m]);
    } else {
        rhs[k][j][i][m] =
            rhs[k][j][i][m] - constants_device::dssp *
                                  (u[k - 2][j][i][m] - 4.0 * u[k - 1][j][i][m] +
                                   6.0 * u[k][j][i][m] -
                                   4.0 * u[k + 1][j][i][m] + u[k + 2][j][i][m]);
    }
}

__global__ void compute_rhs_gpu_kernel_9(double* rhs_device) {
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    int i = t_i / 5 + 1;
    int m = t_i % 5;

    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= KMAX || j > JMAX - 2 ||
        i > IMAX - 2)
        return;

    double (*rhs)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) rhs_device;

    rhs[k][j][i][m] = rhs[k][j][i][m] * constants_device::dt;
}

/*
 * ---------------------------------------------------------------------
 * this function computes the left hand side in the xi-direction
 * ---------------------------------------------------------------------
 */
__device__ void x_solve_gpu_device_fjac(double fjac[5][5], double t_u[5],
                                        double rho_i, double qs,
                                        double square) {
    double tmp1, tmp2;

    /*
     * ---------------------------------------------------------------------
     * determine a (labeled f) and n jacobians
     * ---------------------------------------------------------------------
     */
    tmp1 = rho_i;
    tmp2 = tmp1 * tmp1;

    fjac[0][0] = 0.0;
    fjac[1][0] = 1.0;
    fjac[2][0] = 0.0;
    fjac[3][0] = 0.0;
    fjac[4][0] = 0.0;

    fjac[0][1] = -(t_u[1] * tmp2 * t_u[1]) + constants_device::c2 * qs;
    fjac[1][1] = (2.0 - constants_device::c2) * (t_u[1] / t_u[0]);
    fjac[2][1] = -constants_device::c2 * (t_u[2] * tmp1);
    fjac[3][1] = -constants_device::c2 * (t_u[3] * tmp1);
    fjac[4][1] = constants_device::c2;

    fjac[0][2] = -(t_u[1] * t_u[2]) * tmp2;
    fjac[1][2] = t_u[2] * tmp1;
    fjac[2][2] = t_u[1] * tmp1;
    fjac[3][2] = 0.0;
    fjac[4][2] = 0.0;

    fjac[0][3] = -(t_u[1] * t_u[3]) * tmp2;
    fjac[1][3] = t_u[3] * tmp1;
    fjac[2][3] = 0.0;
    fjac[3][3] = t_u[1] * tmp1;
    fjac[4][3] = 0.0;

    fjac[0][4] =
        (constants_device::c2 * 2.0 * square - constants_device::c1 * t_u[4]) *
        (t_u[1] * tmp2);
    fjac[1][4] = constants_device::c1 * t_u[4] * tmp1 -
                 constants_device::c2 * (t_u[1] * t_u[1] * tmp2 + qs);
    fjac[2][4] = -constants_device::c2 * (t_u[2] * t_u[1]) * tmp2;
    fjac[3][4] = -constants_device::c2 * (t_u[3] * t_u[1]) * tmp2;
    fjac[4][4] = constants_device::c1 * (t_u[1] * tmp1);
}

__device__ void x_solve_gpu_device_njac(double njac[5][5], double t_u[5],
                                        double rho_i) {
    double tmp1, tmp2, tmp3;

    /*
     * ---------------------------------------------------------------------
     * determine a (labeled f) and n jacobians
     * ---------------------------------------------------------------------
     */
    tmp1 = rho_i;
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;

    njac[0][0] = 0.0;
    njac[1][0] = 0.0;
    njac[2][0] = 0.0;
    njac[3][0] = 0.0;
    njac[4][0] = 0.0;

    njac[0][1] =
        -constants_device::con43 * constants_device::c3c4 * tmp2 * t_u[1];
    njac[1][1] = constants_device::con43 * constants_device::c3c4 * tmp1;
    njac[2][1] = 0.0;
    njac[3][1] = 0.0;
    njac[4][1] = 0.0;

    njac[0][2] = -constants_device::c3c4 * tmp2 * t_u[2];
    njac[1][2] = 0.0;
    njac[2][2] = constants_device::c3c4 * tmp1;
    njac[3][2] = 0.0;
    njac[4][2] = 0.0;

    njac[0][3] = -constants_device::c3c4 * tmp2 * t_u[3];
    njac[1][3] = 0.0;
    njac[2][3] = 0.0;
    njac[3][3] = constants_device::c3c4 * tmp1;
    njac[4][3] = 0.0;

    njac[0][4] = -(constants_device::con43 * constants_device::c3c4 -
                   constants_device::c1345) *
                     tmp3 * (t_u[1] * t_u[1]) -
                 (constants_device::c3c4 - constants_device::c1345) * tmp3 *
                     (t_u[2] * t_u[2]) -
                 (constants_device::c3c4 - constants_device::c1345) * tmp3 *
                     (t_u[3] * t_u[3]) -
                 constants_device::c1345 * tmp2 * t_u[4];

    njac[1][4] = (constants_device::con43 * constants_device::c3c4 -
                  constants_device::c1345) *
                 tmp2 * t_u[1];
    njac[2][4] =
        (constants_device::c3c4 - constants_device::c1345) * tmp2 * t_u[2];
    njac[3][4] =
        (constants_device::c3c4 - constants_device::c1345) * tmp2 * t_u[3];
    njac[4][4] = (constants_device::c1345)*tmp1;
}

/*
 * ---------------------------------------------------------------------
 * this function computes the left hand side in the xi-direction
 * ---------------------------------------------------------------------
 */
__global__ void x_solve_gpu_kernel_1(double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device) {
    int t_k = blockDim.y * blockIdx.y + threadIdx.y;
    int mn = t_k / PROBLEM_SIZE;
    int k = t_k % PROBLEM_SIZE;
    int m = mn / 5;
    int n = mn % 5;
    int j = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= PROBLEM_SIZE || j > JMAX - 2 ||
        m >= 5)
        return;

    k += 0;

    int isize;

    /* memory coalescing, index from k,j,i,m,n to m,n,k,i,j */
#define lhsA(a, b, c, d, e)                                                    \
    lhsA_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]
#define lhsB(a, b, c, d, e)                                                    \
    lhsB_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]
#define lhsC(a, b, c, d, e)                                                    \
    lhsC_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]

    isize = IMAX - 1;

    /*
     * ---------------------------------------------------------------------
     * now jacobians set, so form left hand side in x direction
     * ---------------------------------------------------------------------
     */
    lhsA(m, n, k, 0, j - 1) = 0.0;
    lhsB(m, n, k, 0, j - 1) = (m == n) ? 1.0 : 0.0;
    lhsC(m, n, k, 0, j - 1) = 0.0;

    lhsA(m, n, k, isize, j - 1) = 0.0;
    lhsB(m, n, k, isize, j - 1) = (m == n) ? 1.0 : 0.0;
    lhsC(m, n, k, isize, j - 1) = 0.0;

#undef lhsA
#undef lhsB
#undef lhsC
}

__global__ void x_solve_gpu_kernel_2(double* qs_device, double* rho_i_device,
                                     double* square_device, double* u_device,
                                     double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device) {
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= PROBLEM_SIZE || j > JMAX - 2 ||
        i > IMAX - 2) {
        return;
    }

    double (*qs)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) qs_device;
    double (*rho_i)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) rho_i_device;
    double (*square)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) square_device;
    double (*u)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) u_device;

    /* memory coalescing, index from k,j,i,m,n to m,n,k,i,j */
#define lhsA(a, b, c, d, e)                                                    \
    lhsA_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]
#define lhsB(a, b, c, d, e)                                                    \
    lhsB_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]
#define lhsC(a, b, c, d, e)                                                    \
    lhsC_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]

    double fjac[5][5];
    double njac[5][5];

    double t_u[5];

    int m;

    double tmp1, tmp2;

    tmp1 = constants_device::dt * constants_device::tx1;
    tmp2 = constants_device::dt * constants_device::tx2;

    for (m = 0; m < 5; m++) t_u[m] = u[k][j][i - 1][m];
    x_solve_gpu_device_fjac(fjac, t_u, rho_i[k][j][i - 1], qs[k][j][i - 1],
                            square[k][j][i - 1]);
    x_solve_gpu_device_njac(njac, t_u, rho_i[k][j][i - 1]);

    lhsA(0, 0, k, i, j - 1) =
        -tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * constants_device::dx1;
    lhsA(1, 0, k, i, j - 1) = -tmp2 * fjac[1][0] - tmp1 * njac[1][0];
    lhsA(2, 0, k, i, j - 1) = -tmp2 * fjac[2][0] - tmp1 * njac[2][0];
    lhsA(3, 0, k, i, j - 1) = -tmp2 * fjac[3][0] - tmp1 * njac[3][0];
    lhsA(4, 0, k, i, j - 1) = -tmp2 * fjac[4][0] - tmp1 * njac[4][0];

    lhsA(0, 1, k, i, j - 1) = -tmp2 * fjac[0][1] - tmp1 * njac[0][1];
    lhsA(1, 1, k, i, j - 1) =
        -tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * constants_device::dx2;
    lhsA(2, 1, k, i, j - 1) = -tmp2 * fjac[2][1] - tmp1 * njac[2][1];
    lhsA(3, 1, k, i, j - 1) = -tmp2 * fjac[3][1] - tmp1 * njac[3][1];
    lhsA(4, 1, k, i, j - 1) = -tmp2 * fjac[4][1] - tmp1 * njac[4][1];

    lhsA(0, 2, k, i, j - 1) = -tmp2 * fjac[0][2] - tmp1 * njac[0][2];
    lhsA(1, 2, k, i, j - 1) = -tmp2 * fjac[1][2] - tmp1 * njac[1][2];
    lhsA(2, 2, k, i, j - 1) =
        -tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * constants_device::dx3;
    lhsA(3, 2, k, i, j - 1) = -tmp2 * fjac[3][2] - tmp1 * njac[3][2];
    lhsA(4, 2, k, i, j - 1) = -tmp2 * fjac[4][2] - tmp1 * njac[4][2];

    lhsA(0, 3, k, i, j - 1) = -tmp2 * fjac[0][3] - tmp1 * njac[0][3];
    lhsA(1, 3, k, i, j - 1) = -tmp2 * fjac[1][3] - tmp1 * njac[1][3];
    lhsA(2, 3, k, i, j - 1) = -tmp2 * fjac[2][3] - tmp1 * njac[2][3];
    lhsA(3, 3, k, i, j - 1) =
        -tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * constants_device::dx4;
    lhsA(4, 3, k, i, j - 1) = -tmp2 * fjac[4][3] - tmp1 * njac[4][3];

    lhsA(0, 4, k, i, j - 1) = -tmp2 * fjac[0][4] - tmp1 * njac[0][4];
    lhsA(1, 4, k, i, j - 1) = -tmp2 * fjac[1][4] - tmp1 * njac[1][4];
    lhsA(2, 4, k, i, j - 1) = -tmp2 * fjac[2][4] - tmp1 * njac[2][4];
    lhsA(3, 4, k, i, j - 1) = -tmp2 * fjac[3][4] - tmp1 * njac[3][4];
    lhsA(4, 4, k, i, j - 1) =
        -tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * constants_device::dx5;

    for (m = 0; m < 5; m++) {
        t_u[m] = u[k][j][i][m];
    }
    x_solve_gpu_device_njac(fjac, t_u, rho_i[k][j][i]);

    lhsB(0, 0, k, i, j - 1) =
        1.0 + tmp1 * 2.0 * fjac[0][0] + tmp1 * 2.0 * constants_device::dx1;
    lhsB(1, 0, k, i, j - 1) = tmp1 * 2.0 * fjac[1][0];
    lhsB(2, 0, k, i, j - 1) = tmp1 * 2.0 * fjac[2][0];
    lhsB(3, 0, k, i, j - 1) = tmp1 * 2.0 * fjac[3][0];
    lhsB(4, 0, k, i, j - 1) = tmp1 * 2.0 * fjac[4][0];

    lhsB(0, 1, k, i, j - 1) = tmp1 * 2.0 * fjac[0][1];
    lhsB(1, 1, k, i, j - 1) =
        1.0 + tmp1 * 2.0 * fjac[1][1] + tmp1 * 2.0 * constants_device::dx2;
    lhsB(2, 1, k, i, j - 1) = tmp1 * 2.0 * fjac[2][1];
    lhsB(3, 1, k, i, j - 1) = tmp1 * 2.0 * fjac[3][1];
    lhsB(4, 1, k, i, j - 1) = tmp1 * 2.0 * fjac[4][1];

    lhsB(0, 2, k, i, j - 1) = tmp1 * 2.0 * fjac[0][2];
    lhsB(1, 2, k, i, j - 1) = tmp1 * 2.0 * fjac[1][2];
    lhsB(2, 2, k, i, j - 1) =
        1.0 + tmp1 * 2.0 * fjac[2][2] + tmp1 * 2.0 * constants_device::dx3;
    lhsB(3, 2, k, i, j - 1) = tmp1 * 2.0 * fjac[3][2];
    lhsB(4, 2, k, i, j - 1) = tmp1 * 2.0 * fjac[4][2];

    lhsB(0, 3, k, i, j - 1) = tmp1 * 2.0 * fjac[0][3];
    lhsB(1, 3, k, i, j - 1) = tmp1 * 2.0 * fjac[1][3];
    lhsB(2, 3, k, i, j - 1) = tmp1 * 2.0 * fjac[2][3];
    lhsB(3, 3, k, i, j - 1) =
        1.0 + tmp1 * 2.0 * fjac[3][3] + tmp1 * 2.0 * constants_device::dx4;
    lhsB(4, 3, k, i, j - 1) = tmp1 * 2.0 * fjac[4][3];

    lhsB(0, 4, k, i, j - 1) = tmp1 * 2.0 * fjac[0][4];
    lhsB(1, 4, k, i, j - 1) = tmp1 * 2.0 * fjac[1][4];
    lhsB(2, 4, k, i, j - 1) = tmp1 * 2.0 * fjac[2][4];
    lhsB(3, 4, k, i, j - 1) = tmp1 * 2.0 * fjac[3][4];
    lhsB(4, 4, k, i, j - 1) =
        1.0 + tmp1 * 2.0 * fjac[4][4] + tmp1 * 2.0 * constants_device::dx5;

    for (m = 0; m < 5; m++) t_u[m] = u[k][j][i + 1][m];
    x_solve_gpu_device_fjac(fjac, t_u, rho_i[k][j][i + 1], qs[k][j][i + 1],
                            square[k][j][i + 1]);
    x_solve_gpu_device_njac(njac, t_u, rho_i[k][j][i + 1]);

    lhsC(0, 0, k, i, j - 1) =
        tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * constants_device::dx1;
    lhsC(1, 0, k, i, j - 1) = tmp2 * fjac[1][0] - tmp1 * njac[1][0];
    lhsC(2, 0, k, i, j - 1) = tmp2 * fjac[2][0] - tmp1 * njac[2][0];
    lhsC(3, 0, k, i, j - 1) = tmp2 * fjac[3][0] - tmp1 * njac[3][0];
    lhsC(4, 0, k, i, j - 1) = tmp2 * fjac[4][0] - tmp1 * njac[4][0];

    lhsC(0, 1, k, i, j - 1) = tmp2 * fjac[0][1] - tmp1 * njac[0][1];
    lhsC(1, 1, k, i, j - 1) =
        tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * constants_device::dx2;
    lhsC(2, 1, k, i, j - 1) = tmp2 * fjac[2][1] - tmp1 * njac[2][1];
    lhsC(3, 1, k, i, j - 1) = tmp2 * fjac[3][1] - tmp1 * njac[3][1];
    lhsC(4, 1, k, i, j - 1) = tmp2 * fjac[4][1] - tmp1 * njac[4][1];

    lhsC(0, 2, k, i, j - 1) = tmp2 * fjac[0][2] - tmp1 * njac[0][2];
    lhsC(1, 2, k, i, j - 1) = tmp2 * fjac[1][2] - tmp1 * njac[1][2];
    lhsC(2, 2, k, i, j - 1) =
        tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * constants_device::dx3;
    lhsC(3, 2, k, i, j - 1) = tmp2 * fjac[3][2] - tmp1 * njac[3][2];
    lhsC(4, 2, k, i, j - 1) = tmp2 * fjac[4][2] - tmp1 * njac[4][2];

    lhsC(0, 3, k, i, j - 1) = tmp2 * fjac[0][3] - tmp1 * njac[0][3];
    lhsC(1, 3, k, i, j - 1) = tmp2 * fjac[1][3] - tmp1 * njac[1][3];
    lhsC(2, 3, k, i, j - 1) = tmp2 * fjac[2][3] - tmp1 * njac[2][3];
    lhsC(3, 3, k, i, j - 1) =
        tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * constants_device::dx4;
    lhsC(4, 3, k, i, j - 1) = tmp2 * fjac[4][3] - tmp1 * njac[4][3];

    lhsC(0, 4, k, i, j - 1) = tmp2 * fjac[0][4] - tmp1 * njac[0][4];
    lhsC(1, 4, k, i, j - 1) = tmp2 * fjac[1][4] - tmp1 * njac[1][4];
    lhsC(2, 4, k, i, j - 1) = tmp2 * fjac[2][4] - tmp1 * njac[2][4];
    lhsC(3, 4, k, i, j - 1) = tmp2 * fjac[3][4] - tmp1 * njac[3][4];
    lhsC(4, 4, k, i, j - 1) =
        tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * constants_device::dx5;

#undef lhsA
#undef lhsB
#undef lhsC
}

/*
 * ---------------------------------------------------------------------
 * this function computes the left hand side in the xi-direction
 * ---------------------------------------------------------------------
 */
__global__ void x_solve_gpu_kernel_3(double* rhs_device, double* lhsA_device,
                                     double* lhsB_device, double* lhsC_device) {
    extern __shared__ double tmp_l_lhs[];
    double* tmp_l_r = &tmp_l_lhs[blockDim.x * 3 * 5 * 5];

    int k = (blockDim.y * blockIdx.y + threadIdx.y) / 5;
    int m = (blockDim.y * blockIdx.y + threadIdx.y) % 5;
    int j = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int l_j = threadIdx.x;
    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= PROBLEM_SIZE || j > JMAX - 2)
        return;

    int i, n, p, isize;

    double (*rhs)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) rhs_device;

    /* memory coalescing, index from k,j,i,m,n to m,n,k,i,j */
#define lhsA(a, b, c, d, e)                                                    \
    lhsA_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]
#define lhsB(a, b, c, d, e)                                                    \
    lhsB_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]
#define lhsC(a, b, c, d, e)                                                    \
    lhsC_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]

    double (*tmp2_l_lhs)[3][5][5] = (double (*)[3][5][5])tmp_l_lhs;
    double (*l_lhs)[5][5] = tmp2_l_lhs[l_j];
    double (*tmp2_l_r)[2][5] = (double (*)[2][5])tmp_l_r;
    double (*l_r)[5] = tmp2_l_r[l_j];

    double pivot, coeff;

    isize = IMAX - 1;

    /*
     * ---------------------------------------------------------------------
     * performs guaussian elimination on this cell.
     * ---------------------------------------------------------------------
     * assumes that unpacking routines for non-first cells
     * preload C' and rhs' from previous cell.
     * ---------------------------------------------------------------------
     * assumed send happens outside this routine, but that
     * c'(IMAX) and rhs'(IMAX) will be sent to next cell
     * ---------------------------------------------------------------------
     * outer most do loops - sweeping in i direction
     * ---------------------------------------------------------------------
     */
    /* load data */
    for (p = 0; p < 5; p++) {
        l_lhs[BB][p][m] = lhsB(p, m, k, 0, j - 1);
        l_lhs[CC][p][m] = lhsC(p, m, k, 0, j - 1);
    }

    l_r[1][m] = rhs[k][j][0][m];

    __syncthreads();

    /*
     * ---------------------------------------------------------------------
     * multiply c[k][j][0] by b_inverse and copy back to c
     * multiply rhs(0) by b_inverse(0) and copy to rhs
     * ---------------------------------------------------------------------
     */
    for (p = 0; p < 5; p++) {
        pivot = 1.00 / l_lhs[BB][p][p];
        if (m > p && m < 5) l_lhs[BB][m][p] = l_lhs[BB][m][p] * pivot;
        if (m < 5) l_lhs[CC][m][p] = l_lhs[CC][m][p] * pivot;

        if (p == m) l_r[1][p] = l_r[1][p] * pivot;

        __syncthreads();

        if (p != m) {
            coeff = l_lhs[BB][p][m];
            for (n = p + 1; n < 5; n++)
                l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff * l_lhs[BB][n][p];
            for (n = 0; n < 5; n++)
                l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff * l_lhs[CC][n][p];
            l_r[1][m] = l_r[1][m] - coeff * l_r[1][p];
        }
        __syncthreads();
    }

    /* update data */
    rhs[k][j][0][m] = l_r[1][m];

    /*
     * ---------------------------------------------------------------------
     * begin inner most do loop
     * do all the elements of the cell unless last
     * ---------------------------------------------------------------------
     */
    for (i = 1; i <= isize - 1; i++) {
        for (n = 0; n < 5; n++) {
            l_lhs[AA][n][m] = lhsA(n, m, k, i, j - 1);
            l_lhs[BB][n][m] = lhsB(n, m, k, i, j - 1);
        }
        l_r[0][m] = l_r[1][m];
        l_r[1][m] = rhs[k][j][i][m];

        __syncthreads();

        /*
         * ---------------------------------------------------------------------
         * rhs(i) = rhs(i) - A*rhs(i-1)
         * ---------------------------------------------------------------------
         */
        l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m] * l_r[0][0] -
                    l_lhs[AA][1][m] * l_r[0][1] - l_lhs[AA][2][m] * l_r[0][2] -
                    l_lhs[AA][3][m] * l_r[0][3] - l_lhs[AA][4][m] * l_r[0][4];

        /*
         * ---------------------------------------------------------------------
         * B(i) = B(i) - C(i-1)*A(i)
         * ---------------------------------------------------------------------
         */
        for (p = 0; p < 5; p++) {
            l_lhs[BB][m][p] = l_lhs[BB][m][p] -
                              l_lhs[AA][0][p] * l_lhs[CC][m][0] -
                              l_lhs[AA][1][p] * l_lhs[CC][m][1] -
                              l_lhs[AA][2][p] * l_lhs[CC][m][2] -
                              l_lhs[AA][3][p] * l_lhs[CC][m][3] -
                              l_lhs[AA][4][p] * l_lhs[CC][m][4];
        }

        __syncthreads();

        for (n = 0; n < 5; n++) {
            l_lhs[CC][n][m] = lhsC(n, m, k, i, j - 1);
        }

        __syncthreads();

        /*
         * ---------------------------------------------------------------------
         * multiply c[k][j][i] by b_inverse and copy back to c
         * multiply rhs[k][j][0] by b_inverse[k][j][0] and copy to rhs
         * ---------------------------------------------------------------------
         */
        for (p = 0; p < 5; p++) {
            pivot = 1.00 / l_lhs[BB][p][p];
            if (m > p) {
                l_lhs[BB][m][p] = l_lhs[BB][m][p] * pivot;
            }
            l_lhs[CC][m][p] = l_lhs[CC][m][p] * pivot;
            if (p == m) {
                l_r[1][p] = l_r[1][p] * pivot;
            }

            __syncthreads();

            if (p != m) {
                coeff = l_lhs[BB][p][m];
                for (n = p + 1; n < 5; n++)
                    l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff * l_lhs[BB][n][p];
                for (n = 0; n < 5; n++)
                    l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff * l_lhs[CC][n][p];
                l_r[1][m] = l_r[1][m] - coeff * l_r[1][p];
            }

            __syncthreads();
        }

        for (n = 0; n < 5; n++) {
            lhsC(n, m, k, i, j - 1) = l_lhs[CC][n][m];
        }

        rhs[k][j][i][m] = l_r[1][m];
    }

    for (n = 0; n < 5; n++) {
        l_lhs[AA][n][m] = lhsA(n, m, k, i, j - 1);
        l_lhs[BB][n][m] = lhsB(n, m, k, i, j - 1);
    }
    l_r[0][m] = l_r[1][m];
    l_r[1][m] = rhs[k][j][i][m];

    __syncthreads();

    /*
     * ---------------------------------------------------------------------
     * rhs(isize) = rhs(isize) - A*rhs(isize-1)
     * ---------------------------------------------------------------------
     */
    l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m] * l_r[0][0] -
                l_lhs[AA][1][m] * l_r[0][1] - l_lhs[AA][2][m] * l_r[0][2] -
                l_lhs[AA][3][m] * l_r[0][3] - l_lhs[AA][4][m] * l_r[0][3];

    /*
     * ---------------------------------------------------------------------
     * B(isize) = B(isize) - C(isize-1)*A(isize)
     * ---------------------------------------------------------------------
     */
    for (p = 0; p < 5; p++) {
        l_lhs[BB][m][p] = l_lhs[BB][m][p] - l_lhs[AA][0][p] * l_lhs[CC][m][0] -
                          l_lhs[AA][1][p] * l_lhs[CC][m][1] -
                          l_lhs[AA][2][p] * l_lhs[CC][m][2] -
                          l_lhs[AA][3][p] * l_lhs[CC][m][3] -
                          l_lhs[AA][4][p] * l_lhs[CC][m][4];
    }

    /*
     * ---------------------------------------------------------------------
     * multiply rhs() by b_inverse() and copy to rhs
     * ---------------------------------------------------------------------
     */
    for (p = 0; p < 5; p++) {
        pivot = 1.00 / l_lhs[BB][p][p];
        if (m > p && m < 5) {
            l_lhs[BB][m][p] = l_lhs[BB][m][p] * pivot;
        }
        if (p == m) {
            l_r[1][p] = l_r[1][p] * pivot;
        }

        __syncthreads();

        if (p != m) {
            coeff = l_lhs[BB][p][m];
            for (n = p + 1; n < 5; n++)
                l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff * l_lhs[BB][n][p];
            l_r[1][m] = l_r[1][m] - coeff * l_r[1][p];
        }

        __syncthreads();
    }

    rhs[k][j][i][m] = l_r[1][m];

    __syncthreads();

    /*
     * ---------------------------------------------------------------------
     * back solve: if last cell, then generate U(isize)=rhs(isize)
     * else assume U(isize) is loaded in un pack backsub_info
     * so just use it
     * after u(istart) will be sent to next cell
     * ---------------------------------------------------------------------
     */
    for (i = isize - 1; i >= 0; i--) {
        for (n = 0; n < M_SIZE; n++) {
            rhs[k][j][i][m] =
                rhs[k][j][i][m] - lhsC(n, m, k, i, j - 1) * rhs[k][j][i + 1][n];
        }
        __syncthreads();
    }

#undef lhsA
#undef lhsB
#undef lhsC
}

__device__ void y_solve_gpu_device_fjac(double fjac[5][5], double t_u[5],
                                        double rho_i, double square,
                                        double qs) {
    double tmp1, tmp2;

    tmp1 = rho_i;
    tmp2 = tmp1 * tmp1;

    fjac[0][0] = 0.0;
    fjac[1][0] = 0.0;
    fjac[2][0] = 1.0;
    fjac[3][0] = 0.0;
    fjac[4][0] = 0.0;

    fjac[0][1] = -(t_u[1] * t_u[2]) * tmp2;
    fjac[1][1] = t_u[2] * tmp1;
    fjac[2][1] = t_u[1] * tmp1;
    fjac[3][1] = 0.0;
    fjac[4][1] = 0.0;

    fjac[0][2] = -(t_u[2] * t_u[2] * tmp2) + constants_device::c2 * qs;
    fjac[1][2] = -constants_device::c2 * t_u[1] * tmp1;
    fjac[2][2] = (2.0 - constants_device::c2) * t_u[2] * tmp1;
    fjac[3][2] = -constants_device::c2 * t_u[3] * tmp1;
    fjac[4][2] = constants_device::c2;

    fjac[0][3] = -(t_u[2] * t_u[3]) * tmp2;
    fjac[1][3] = 0.0;
    fjac[2][3] = t_u[3] * tmp1;
    fjac[3][3] = t_u[2] * tmp1;
    fjac[4][3] = 0.0;

    fjac[0][4] =
        (constants_device::c2 * 2.0 * square - constants_device::c1 * t_u[4]) *
        t_u[2] * tmp2;
    fjac[1][4] = -constants_device::c2 * t_u[1] * t_u[2] * tmp2;
    fjac[2][4] = constants_device::c1 * t_u[4] * tmp1 -
                 constants_device::c2 * (qs + t_u[2] * t_u[2] * tmp2);
    fjac[3][4] = -constants_device::c2 * (t_u[2] * t_u[3]) * tmp2;
    fjac[4][4] = constants_device::c1 * t_u[2] * tmp1;
}

__device__ void y_solve_gpu_device_njac(double njac[5][5], double t_u[5],
                                        double rho_i) {
    double tmp1, tmp2, tmp3;

    tmp1 = rho_i;
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;

    njac[0][0] = 0.0;
    njac[1][0] = 0.0;
    njac[2][0] = 0.0;
    njac[3][0] = 0.0;
    njac[4][0] = 0.0;

    njac[0][1] = -constants_device::c3c4 * tmp2 * t_u[1];
    njac[1][1] = constants_device::c3c4 * tmp1;
    njac[2][1] = 0.0;
    njac[3][1] = 0.0;
    njac[4][1] = 0.0;

    njac[0][2] =
        -constants_device::con43 * constants_device::c3c4 * tmp2 * t_u[2];
    njac[1][2] = 0.0;
    njac[2][2] = constants_device::con43 * constants_device::c3c4 * tmp1;
    njac[3][2] = 0.0;
    njac[4][2] = 0.0;

    njac[0][3] = -constants_device::c3c4 * tmp2 * t_u[3];
    njac[1][3] = 0.0;
    njac[2][3] = 0.0;
    njac[3][3] = constants_device::c3c4 * tmp1;
    njac[4][3] = 0.0;

    njac[0][4] = -(constants_device::c3c4 - constants_device::c1345) * tmp3 *
                     (t_u[1] * t_u[1]) -
                 (constants_device::con43 * constants_device::c3c4 -
                  constants_device::c1345) *
                     tmp3 * (t_u[2] * t_u[2]) -
                 (constants_device::c3c4 - constants_device::c1345) * tmp3 *
                     (t_u[3] * t_u[3]) -
                 constants_device::c1345 * tmp2 * t_u[4];

    njac[1][4] =
        (constants_device::c3c4 - constants_device::c1345) * tmp2 * t_u[1];
    njac[2][4] = (constants_device::con43 * constants_device::c3c4 -
                  constants_device::c1345) *
                 tmp2 * t_u[2];
    njac[3][4] =
        (constants_device::c3c4 - constants_device::c1345) * tmp2 * t_u[3];
    njac[4][4] = (constants_device::c1345)*tmp1;
}

__global__ void y_solve_gpu_kernel_1(double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device) {
    int t_k = blockDim.y * blockIdx.y + threadIdx.y;
    int k = t_k % PROBLEM_SIZE;
    int mn = t_k / PROBLEM_SIZE;
    int m = mn / 5;
    int n = mn % 5;
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= PROBLEM_SIZE || i > IMAX - 2 ||
        m >= 5) {
        return;
    }

    int jsize;

#define lhsA(a, b, c, d, e)                                                    \
    lhsA_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]
#define lhsB(a, b, c, d, e)                                                    \
    lhsB_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]
#define lhsC(a, b, c, d, e)                                                    \
    lhsC_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]

    jsize = JMAX - 1;

    /*
     * ---------------------------------------------------------------------
     * now joacobians set, so form left hand side in y direction
     * ---------------------------------------------------------------------
     */

    lhsA(m, n, k, 0, i - 1) = 0.0;
    lhsB(m, n, k, 0, i - 1) = (m == n) ? 1.0 : 0.0;
    lhsC(m, n, k, 0, i - 1) = 0.0;

    lhsA(m, n, k, jsize, i - 1) = 0.0;
    lhsB(m, n, k, jsize, i - 1) = (m == n) ? 1.0 : 0.0;
    lhsC(m, n, k, jsize, i - 1) = 0.0;

#undef lhsA
#undef lhsB
#undef lhsC
}

__global__ void y_solve_gpu_kernel_2(double* qs_device, double* rho_i_device,
                                     double* square_device, double* u_device,
                                     double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device) {
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= PROBLEM_SIZE || j > JMAX - 2 ||
        i > IMAX - 2) {
        return;
    }

    int m;
    double tmp1, tmp2;

    double (*qs)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) qs_device;
    double (*rho_i)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) rho_i_device;
    double (*square)[JMAXP + 1][IMAXP + 1] =
        (double (*)[JMAXP + 1][IMAXP + 1]) square_device;
    double (*u)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) u_device;

#define lhsA(a, b, c, d, e)                                                    \
    lhsA_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]
#define lhsB(a, b, c, d, e)                                                    \
    lhsB_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]
#define lhsC(a, b, c, d, e)                                                    \
    lhsC_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]

    double fjac[5][5], njac[5][5];

    double t_u[5];

    /*
     * ---------------------------------------------------------------------
     * this function computes the left hand side for the three y-factors
     * ---------------------------------------------------------------------
     * compute the indices for storing the tri-diagonal matrix;
     * determine a (labeled f) and n jacobians for cell c
     * ---------------------------------------------------------------------
     */
    tmp1 = constants_device::dt * constants_device::ty1;
    tmp2 = constants_device::dt * constants_device::ty2;

    for (m = 0; m < 5; m++) {
        t_u[m] = u[k][j - 1][i][m];
    }
    y_solve_gpu_device_fjac(fjac, t_u, rho_i[k][j - 1][i], square[k][j - 1][i],
                            qs[k][j - 1][i]);
    y_solve_gpu_device_njac(njac, t_u, rho_i[k][j - 1][i]);

    lhsA(0, 0, k, j, i - 1) =
        -tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * constants_device::dy1;
    lhsA(1, 0, k, j, i - 1) = -tmp2 * fjac[1][0] - tmp1 * njac[1][0];
    lhsA(2, 0, k, j, i - 1) = -tmp2 * fjac[2][0] - tmp1 * njac[2][0];
    lhsA(3, 0, k, j, i - 1) = -tmp2 * fjac[3][0] - tmp1 * njac[3][0];
    lhsA(4, 0, k, j, i - 1) = -tmp2 * fjac[4][0] - tmp1 * njac[4][0];

    lhsA(0, 1, k, j, i - 1) = -tmp2 * fjac[0][1] - tmp1 * njac[0][1];
    lhsA(1, 1, k, j, i - 1) =
        -tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * constants_device::dy2;
    lhsA(2, 1, k, j, i - 1) = -tmp2 * fjac[2][1] - tmp1 * njac[2][1];
    lhsA(3, 1, k, j, i - 1) = -tmp2 * fjac[3][1] - tmp1 * njac[3][1];
    lhsA(4, 1, k, j, i - 1) = -tmp2 * fjac[4][1] - tmp1 * njac[4][1];

    lhsA(0, 2, k, j, i - 1) = -tmp2 * fjac[0][2] - tmp1 * njac[0][2];
    lhsA(1, 2, k, j, i - 1) = -tmp2 * fjac[1][2] - tmp1 * njac[1][2];
    lhsA(2, 2, k, j, i - 1) =
        -tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * constants_device::dy3;
    lhsA(3, 2, k, j, i - 1) = -tmp2 * fjac[3][2] - tmp1 * njac[3][2];
    lhsA(4, 2, k, j, i - 1) = -tmp2 * fjac[4][2] - tmp1 * njac[4][2];

    lhsA(0, 3, k, j, i - 1) = -tmp2 * fjac[0][3] - tmp1 * njac[0][3];
    lhsA(1, 3, k, j, i - 1) = -tmp2 * fjac[1][3] - tmp1 * njac[1][3];
    lhsA(2, 3, k, j, i - 1) = -tmp2 * fjac[2][3] - tmp1 * njac[2][3];
    lhsA(3, 3, k, j, i - 1) =
        -tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * constants_device::dy4;
    lhsA(4, 3, k, j, i - 1) = -tmp2 * fjac[4][3] - tmp1 * njac[4][3];

    lhsA(0, 4, k, j, i - 1) = -tmp2 * fjac[0][4] - tmp1 * njac[0][4];
    lhsA(1, 4, k, j, i - 1) = -tmp2 * fjac[1][4] - tmp1 * njac[1][4];
    lhsA(2, 4, k, j, i - 1) = -tmp2 * fjac[2][4] - tmp1 * njac[2][4];
    lhsA(3, 4, k, j, i - 1) = -tmp2 * fjac[3][4] - tmp1 * njac[3][4];
    lhsA(4, 4, k, j, i - 1) =
        -tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * constants_device::dy5;

    for (m = 0; m < 5; m++) {
        t_u[m] = u[k][j][i][m];
    }
    y_solve_gpu_device_njac(njac, t_u, rho_i[k][j][i]);

    lhsB(0, 0, k, j, i - 1) =
        1.0 + tmp1 * 2.0 * njac[0][0] + tmp1 * 2.0 * constants_device::dy1;
    lhsB(1, 0, k, j, i - 1) = tmp1 * 2.0 * njac[1][0];
    lhsB(2, 0, k, j, i - 1) = tmp1 * 2.0 * njac[2][0];
    lhsB(3, 0, k, j, i - 1) = tmp1 * 2.0 * njac[3][0];
    lhsB(4, 0, k, j, i - 1) = tmp1 * 2.0 * njac[4][0];

    lhsB(0, 1, k, j, i - 1) = tmp1 * 2.0 * njac[0][1];
    lhsB(1, 1, k, j, i - 1) =
        1.0 + tmp1 * 2.0 * njac[1][1] + tmp1 * 2.0 * constants_device::dy2;
    lhsB(2, 1, k, j, i - 1) = tmp1 * 2.0 * njac[2][1];
    lhsB(3, 1, k, j, i - 1) = tmp1 * 2.0 * njac[3][1];
    lhsB(4, 1, k, j, i - 1) = tmp1 * 2.0 * njac[4][1];

    lhsB(0, 2, k, j, i - 1) = tmp1 * 2.0 * njac[0][2];
    lhsB(1, 2, k, j, i - 1) = tmp1 * 2.0 * njac[1][2];
    lhsB(2, 2, k, j, i - 1) =
        1.0 + tmp1 * 2.0 * njac[2][2] + tmp1 * 2.0 * constants_device::dy3;
    lhsB(3, 2, k, j, i - 1) = tmp1 * 2.0 * njac[3][2];
    lhsB(4, 2, k, j, i - 1) = tmp1 * 2.0 * njac[4][2];

    lhsB(0, 3, k, j, i - 1) = tmp1 * 2.0 * njac[0][3];
    lhsB(1, 3, k, j, i - 1) = tmp1 * 2.0 * njac[1][3];
    lhsB(2, 3, k, j, i - 1) = tmp1 * 2.0 * njac[2][3];
    lhsB(3, 3, k, j, i - 1) =
        1.0 + tmp1 * 2.0 * njac[3][3] + tmp1 * 2.0 * constants_device::dy4;
    lhsB(4, 3, k, j, i - 1) = tmp1 * 2.0 * njac[4][3];

    lhsB(0, 4, k, j, i - 1) = tmp1 * 2.0 * njac[0][4];
    lhsB(1, 4, k, j, i - 1) = tmp1 * 2.0 * njac[1][4];
    lhsB(2, 4, k, j, i - 1) = tmp1 * 2.0 * njac[2][4];
    lhsB(3, 4, k, j, i - 1) = tmp1 * 2.0 * njac[3][4];
    lhsB(4, 4, k, j, i - 1) =
        1.0 + tmp1 * 2.0 * njac[4][4] + tmp1 * 2.0 * constants_device::dy5;

    for (m = 0; m < 5; m++) {
        t_u[m] = u[k][j + 1][i][m];
    }

    y_solve_gpu_device_fjac(fjac, t_u, rho_i[k][j + 1][i], square[k][j + 1][i],
                            qs[k][j + 1][i]);
    y_solve_gpu_device_njac(njac, t_u, rho_i[k][j + 1][i]);

    lhsC(0, 0, k, j, i - 1) =
        tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * constants_device::dy1;
    lhsC(1, 0, k, j, i - 1) = tmp2 * fjac[1][0] - tmp1 * njac[1][0];
    lhsC(2, 0, k, j, i - 1) = tmp2 * fjac[2][0] - tmp1 * njac[2][0];
    lhsC(3, 0, k, j, i - 1) = tmp2 * fjac[3][0] - tmp1 * njac[3][0];
    lhsC(4, 0, k, j, i - 1) = tmp2 * fjac[4][0] - tmp1 * njac[4][0];

    lhsC(0, 1, k, j, i - 1) = tmp2 * fjac[0][1] - tmp1 * njac[0][1];
    lhsC(1, 1, k, j, i - 1) =
        tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * constants_device::dy2;
    lhsC(2, 1, k, j, i - 1) = tmp2 * fjac[2][1] - tmp1 * njac[2][1];
    lhsC(3, 1, k, j, i - 1) = tmp2 * fjac[3][1] - tmp1 * njac[3][1];
    lhsC(4, 1, k, j, i - 1) = tmp2 * fjac[4][1] - tmp1 * njac[4][1];

    lhsC(0, 2, k, j, i - 1) = tmp2 * fjac[0][2] - tmp1 * njac[0][2];
    lhsC(1, 2, k, j, i - 1) = tmp2 * fjac[1][2] - tmp1 * njac[1][2];
    lhsC(2, 2, k, j, i - 1) =
        tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * constants_device::dy3;
    lhsC(3, 2, k, j, i - 1) = tmp2 * fjac[3][2] - tmp1 * njac[3][2];
    lhsC(4, 2, k, j, i - 1) = tmp2 * fjac[4][2] - tmp1 * njac[4][2];

    lhsC(0, 3, k, j, i - 1) = tmp2 * fjac[0][3] - tmp1 * njac[0][3];
    lhsC(1, 3, k, j, i - 1) = tmp2 * fjac[1][3] - tmp1 * njac[1][3];
    lhsC(2, 3, k, j, i - 1) = tmp2 * fjac[2][3] - tmp1 * njac[2][3];
    lhsC(3, 3, k, j, i - 1) =
        tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * constants_device::dy4;
    lhsC(4, 3, k, j, i - 1) = tmp2 * fjac[4][3] - tmp1 * njac[4][3];

    lhsC(0, 4, k, j, i - 1) = tmp2 * fjac[0][4] - tmp1 * njac[0][4];
    lhsC(1, 4, k, j, i - 1) = tmp2 * fjac[1][4] - tmp1 * njac[1][4];
    lhsC(2, 4, k, j, i - 1) = tmp2 * fjac[2][4] - tmp1 * njac[2][4];
    lhsC(3, 4, k, j, i - 1) = tmp2 * fjac[3][4] - tmp1 * njac[3][4];
    lhsC(4, 4, k, j, i - 1) =
        tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * constants_device::dy5;

#undef lhsA
#undef lhsB
#undef lhsC
}

__global__ void y_solve_gpu_kernel_3(double* rhs_device, double* lhsA_device,
                                     double* lhsB_device, double* lhsC_device) {
    extern __shared__ double tmp_l_lhs[];
    double* tmp_l_r = &tmp_l_lhs[blockDim.x * 3 * 5 * 5];

    int k = (blockDim.y * blockIdx.y + threadIdx.y) / 5;
    int m = (blockDim.y * blockIdx.y + threadIdx.y) % 5;
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int l_i = threadIdx.x;
    if (k + 0 < 1 || k + 0 > KMAX - 2 || k >= PROBLEM_SIZE || i > IMAX - 2) {
        return;
    }

    int j, n, p, jsize;

    double (*rhs)[JMAXP + 1][IMAXP + 1][5] =
        (double (*)[JMAXP + 1][IMAXP + 1][5]) rhs_device;

#define lhsA(a, b, c, d, e)                                                    \
    lhsA_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]
#define lhsB(a, b, c, d, e)                                                    \
    lhsB_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]
#define lhsC(a, b, c, d, e)                                                    \
    lhsC_device[((((a) * 5 + (b)) * PROBLEM_SIZE + (c)) * (PROBLEM_SIZE + 1) + \
                 (d)) *                                                        \
                    (PROBLEM_SIZE - 1) +                                       \
                (e)]

    double (*tmp2_l_lhs)[3][5][5] = (double (*)[3][5][5])tmp_l_lhs;
    double (*l_lhs)[5][5] = tmp2_l_lhs[l_i];
    double (*tmp2_l_r)[2][5] = (double (*)[2][5])tmp_l_r;
    double (*l_r)[5] = tmp2_l_r[l_i];

    double pivot, coeff;

    jsize = JMAX - 1;

    /*
     * ---------------------------------------------------------------------
     * performs guaussian elimination on this cell.
     * ---------------------------------------------------------------------
     * assumes that unpacking routines for non-first cells
     * preload C' and rhs' from previous cell.
     * ---------------------------------------------------------------------
     * assumed send happens outside this routine, but that
     * c'(JMAX) and rhs'(JMAX) will be sent to next cell
     * ---------------------------------------------------------------------
     */
    /* load data */
    for (p = 0; p < 5; p++) {
        l_lhs[BB][p][m] = lhsB(p, m, k, 0, i - 1);
        l_lhs[CC][p][m] = lhsC(p, m, k, 0, i - 1);
    }

    l_r[1][m] = rhs[k][0][i][m];

    __syncthreads();

    /*
     * ---------------------------------------------------------------------
     * multiply c[k][0][i] by b_inverse and copy back to c
     * multiply rhs(0) by b_inverse(0) and copy to rhs
     * ---------------------------------------------------------------------
     */
    for (p = 0; p < 5; p++) {
        pivot = 1.00 / l_lhs[BB][p][p];
        if (m > p && m < 5) {
            l_lhs[BB][m][p] = l_lhs[BB][m][p] * pivot;
        }
        if (m < 5) {
            l_lhs[CC][m][p] = l_lhs[CC][m][p] * pivot;
        }
        if (p == m) {
            l_r[1][p] = l_r[1][p] * pivot;
        }

        __syncthreads();

        if (p != m) {
            coeff = l_lhs[BB][p][m];
            for (n = p + 1; n < 5; n++) {
                l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff * l_lhs[BB][n][p];
            }
            for (n = 0; n < 5; n++) {
                l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff * l_lhs[CC][n][p];
            }
            l_r[1][m] = l_r[1][m] - coeff * l_r[1][p];
        }

        __syncthreads();
    }

    /* update data */
    rhs[k][0][i][m] = l_r[1][m];

    /*
     * ---------------------------------------------------------------------
     * begin inner most do loop
     * do all the elements of the cell unless last
     * ---------------------------------------------------------------------
     */
    for (j = 1; j <= jsize - 1; j++) {
        /* load data */
        for (n = 0; n < 5; n++) {
            l_lhs[AA][n][m] = lhsA(n, m, k, j, i - 1);
            l_lhs[BB][n][m] = lhsB(n, m, k, j, i - 1);
        }
        l_r[0][m] = l_r[1][m];
        l_r[1][m] = rhs[k][j][i][m];

        __syncthreads();

        /*
         * ---------------------------------------------------------------------
         * subtract A*lhs_vector(j-1) from lhs_vector(j)
         *
         * rhs(j) = rhs(j) - A*rhs(j-1)
         * ---------------------------------------------------------------------
         */
        l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m] * l_r[0][0] -
                    l_lhs[AA][1][m] * l_r[0][1] - l_lhs[AA][2][m] * l_r[0][2] -
                    l_lhs[AA][3][m] * l_r[0][3] - l_lhs[AA][4][m] * l_r[0][4];

        /*
         * ---------------------------------------------------------------------
         * B(j) = B(j) - C(j-1)*A(j)
         * ---------------------------------------------------------------------
         */
        for (p = 0; p < 5; p++) {
            l_lhs[BB][m][p] = l_lhs[BB][m][p] -
                              l_lhs[AA][0][p] * l_lhs[CC][m][0] -
                              l_lhs[AA][1][p] * l_lhs[CC][m][1] -
                              l_lhs[AA][2][p] * l_lhs[CC][m][2] -
                              l_lhs[AA][3][p] * l_lhs[CC][m][3] -
                              l_lhs[AA][4][p] * l_lhs[CC][m][4];
        }

        __syncthreads();

        /* update data */
        for (n = 0; n < 5; n++) {
            l_lhs[CC][n][m] = lhsC(n, m, k, j, i - 1);
        }

        __syncthreads();

        /*
         * ---------------------------------------------------------------------
         * multiply c[k][j][i] by b_inverse and copy back to c
         * multiply rhs[k][0][i] by b_inverse[k][0][i] and copy to rhs
         * ---------------------------------------------------------------------
         */
        for (p = 0; p < 5; p++) {
            pivot = 1.00 / l_lhs[BB][p][p];
            if (m > p) {
                l_lhs[BB][m][p] = l_lhs[BB][m][p] * pivot;
            }
            l_lhs[CC][m][p] = l_lhs[CC][m][p] * pivot;
            if (p == m) {
                l_r[1][p] = l_r[1][p] * pivot;
            }

            __syncthreads();
            if (p != m) {
                coeff = l_lhs[BB][p][m];
                for (n = p + 1; n < 5; n++) {
                    l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff * l_lhs[BB][n][p];
                }
                for (n = 0; n < 5; n++) {
                    l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff * l_lhs[CC][n][p];
                }
                l_r[1][m] = l_r[1][m] - coeff * l_r[1][p];
            }

            __syncthreads();
        }

        /* update global memory */
        for (n = 0; n < 5; n++) {
            lhsC(n, m, k, j, i - 1) = l_lhs[CC][n][m];
        }
        rhs[k][j][i][m] = l_r[1][m];
    }

    /* load data */
    for (n = 0; n < 5; n++) {
        l_lhs[AA][n][m] = lhsA(n, m, k, j, i - 1);
        l_lhs[BB][n][m] = lhsB(n, m, k, j, i - 1);
    }
    l_r[0][m] = l_r[1][m];
    l_r[1][m] = rhs[k][j][i][m];

    __syncthreads();

    /*
     * ---------------------------------------------------------------------
     * rhs(jsize) = rhs(jsize) - A*rhs(jsize-1)
     * ---------------------------------------------------------------------
     */
    l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m] * l_r[0][0] -
                l_lhs[AA][1][m] * l_r[0][1] - l_lhs[AA][2][m] * l_r[0][2] -
                l_lhs[AA][3][m] * l_r[0][3] - l_lhs[AA][4][m] * l_r[0][4];

    /*
     * ---------------------------------------------------------------------
     * B(jsize) = B(jsize) - C(jsize-1)*A(jsize)
     * matmul_sub(AA,i,jsize,k,c, CC,i,jsize-1,k,c,BB,i,jsize,k)
     * ---------------------------------------------------------------------
     */
    for (p = 0; p < 5; p++) {
        l_lhs[BB][m][p] = l_lhs[BB][m][p] - l_lhs[AA][0][p] * l_lhs[CC][m][0] -
                          l_lhs[AA][1][p] * l_lhs[CC][m][1] -
                          l_lhs[AA][2][p] * l_lhs[CC][m][2] -
                          l_lhs[AA][3][p] * l_lhs[CC][m][3] -
                          l_lhs[AA][4][p] * l_lhs[CC][m][4];
    }

    /*
     * ---------------------------------------------------------------------
     * multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
     * ---------------------------------------------------------------------
     * binvrhs_p( lhs[jsize][BB], rhs[k][jsize][i], run_computation, m);
     * ---------------------------------------------------------------------
     */
    for (p = 0; p < 5; p++) {
        pivot = 1.00 / l_lhs[BB][p][p];
        if (m > p && m < 5) {
            l_lhs[BB][m][p] = l_lhs[BB][m][p] * pivot;
        }
        if (p == m) {
            l_r[1][p] = l_r[1][p] * pivot;
        }

        __syncthreads();

        if (p != m) {
            coeff = l_lhs[BB][p][m];
            for (n = p + 1; n < 5; n++) {
                l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff * l_lhs[BB][n][p];
            }
            l_r[1][m] = l_r[1][m] - coeff * l_r[1][p];
        }

        __syncthreads();
    }

    rhs[k][j][i][m] = l_r[1][m];

    __syncthreads();

    /*
     * ---------------------------------------------------------------------
     * back solve: if last cell, then generate U(jsize)=rhs(jsize)
     * else assume U(jsize) is loaded in un pack backsub_info
     * so just use it
     * after u(jstart) will be sent to next cell
     * ---------------------------------------------------------------------
     */
    for (j = jsize - 1; j >= 0; j--) {
        for (n = 0; n < M_SIZE; n++) {
            rhs[k][j][i][m] =
                rhs[k][j][i][m] - lhsC(n, m, k, j, i - 1) * rhs[k][j + 1][i][n];
        }
        __syncthreads();
    }

#undef lhsA
#undef lhsB
#undef lhsC
}

/*
 * ---------------------------------------------------------------------
 * this function computes the left hand side for the three z-factors
 * ---------------------------------------------------------------------
 */
__device__ void z_solve_gpu_device_fjac(double l_fjac[5][5], double t_u[5],
                                        double square, double qs) {
    double tmp1, tmp2;

    tmp1 = 1.0 / t_u[0];
    tmp2 = tmp1 * tmp1;

    l_fjac[0][0] = 0.0;
    l_fjac[1][0] = 0.0;
    l_fjac[2][0] = 0.0;
    l_fjac[3][0] = 1.0;
    l_fjac[4][0] = 0.0;

    l_fjac[0][1] = -(t_u[1] * t_u[3]) * tmp2;
    l_fjac[1][1] = t_u[3] * tmp1;
    l_fjac[2][1] = 0.0;
    l_fjac[3][1] = t_u[1] * tmp1;
    l_fjac[4][1] = 0.0;

    l_fjac[0][2] = -(t_u[2] * t_u[3]) * tmp2;
    l_fjac[1][2] = 0.0;
    l_fjac[2][2] = t_u[3] * tmp1;
    l_fjac[3][2] = t_u[2] * tmp1;
    l_fjac[4][2] = 0.0;

    l_fjac[0][3] = -(t_u[3] * t_u[3] * tmp2) + constants_device::c2 * qs;
    l_fjac[1][3] = -constants_device::c2 * t_u[1] * tmp1;
    l_fjac[2][3] = -constants_device::c2 * t_u[2] * tmp1;
    l_fjac[3][3] = (2.0 - constants_device::c2) * t_u[3] * tmp1;
    l_fjac[4][3] = constants_device::c2;

    l_fjac[0][4] =
        (constants_device::c2 * 2.0 * square - constants_device::c1 * t_u[4]) *
        t_u[3] * tmp2;
    l_fjac[1][4] = -constants_device::c2 * (t_u[1] * t_u[3]) * tmp2;
    l_fjac[2][4] = -constants_device::c2 * (t_u[2] * t_u[3]) * tmp2;
    l_fjac[3][4] = constants_device::c1 * (t_u[4] * tmp1) -
                   constants_device::c2 * (qs + t_u[3] * t_u[3] * tmp2);
    l_fjac[4][4] = constants_device::c1 * t_u[3] * tmp1;
}

__device__ void z_solve_gpu_device_njac(double l_njac[5][5], double t_u[5]) {
    double tmp1, tmp2, tmp3;

    tmp1 = 1.0 / t_u[0];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;

    l_njac[0][0] = 0.0;
    l_njac[1][0] = 0.0;
    l_njac[2][0] = 0.0;
    l_njac[3][0] = 0.0;
    l_njac[4][0] = 0.0;

    l_njac[0][1] = -constants_device::c3c4 * tmp2 * t_u[1];
    l_njac[1][1] = constants_device::c3c4 * tmp1;
    l_njac[2][1] = 0.0;
    l_njac[3][1] = 0.0;
    l_njac[4][1] = 0.0;

    l_njac[0][2] = -constants_device::c3c4 * tmp2 * t_u[2];
    l_njac[1][2] = 0.0;
    l_njac[2][2] = constants_device::c3c4 * tmp1;
    l_njac[3][2] = 0.0;
    l_njac[4][2] = 0.0;

    l_njac[0][3] =
        -constants_device::con43 * constants_device::c3c4 * tmp2 * t_u[3];
    l_njac[1][3] = 0.0;
    l_njac[2][3] = 0.0;
    l_njac[3][3] = constants_device::con43 * constants_device::c3 *
                   constants_device::c4 * tmp1;
    l_njac[4][3] = 0.0;

    l_njac[0][4] = -(constants_device::c3c4 - constants_device::c1345) * tmp3 *
                       (t_u[1] * t_u[1]) -
                   (constants_device::c3c4 - constants_device::c1345) * tmp3 *
                       (t_u[2] * t_u[2]) -
                   (constants_device::con43 * constants_device::c3c4 -
                    constants_device::c1345) *
                       tmp3 * (t_u[3] * t_u[3]) -
                   constants_device::c1345 * tmp2 * t_u[4];

    l_njac[1][4] =
        (constants_device::c3c4 - constants_device::c1345) * tmp2 * t_u[1];
    l_njac[2][4] =
        (constants_device::c3c4 - constants_device::c1345) * tmp2 * t_u[2];
    l_njac[3][4] = (constants_device::con43 * constants_device::c3c4 -
                    constants_device::c1345) *
                   tmp2 * t_u[3];
    l_njac[4][4] = (constants_device::c1345)*tmp1;
}

/*
 * ---------------------------------------------------------------------
 * this function computes the left hand side for the three z-factors
 * ---------------------------------------------------------------------
 */
__global__ void z_solve_gpu_kernel_1(double* lhsA_device, double* lhsB_device,
                                     double* lhsC_device) {
    int t_j = blockDim.y * blockIdx.y + threadIdx.y;
    int j = t_j % PROBLEM_SIZE;
    int mn = t_j / PROBLEM_SIZE;
    int m = mn / 5;
    int n = mn % 5;
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (j + 1 < 1 || j + 1 > JMAX - 2 || j >= PROBLEM_SIZE || i > IMAX - 2 ||
        m >= 5) {
        return;
    }

    j++;

    int ksize;

#define lhsA(a, b, c, d, e)                                                   \
    lhsA_device[((((a) * 5 + (b)) * (PROBLEM_SIZE + 1) + (c)) * (JMAXP + 1) + \
                 (d)) *                                                       \
                    (PROBLEM_SIZE - 1) +                                      \
                (e)]
#define lhsB(a, b, c, d, e)                                                   \
    lhsB_device[((((a) * 5 + (b)) * (PROBLEM_SIZE + 1) + (c)) * (JMAXP + 1) + \
                 (d)) *                                                       \
                    (PROBLEM_SIZE - 1) +                                      \
                (e)]
#define lhsC(a, b, c, d, e)                                                   \
    lhsC_device[((((a) * 5 + (b)) * (PROBLEM_SIZE + 1) + (c)) * (JMAXP + 1) + \
                 (d)) *                                                       \
                    (PROBLEM_SIZE - 1) +                                      \
                (e)]

    ksize = KMAX - 1;

    /*
     * ---------------------------------------------------------------------
     * now jacobians set, so form left hand side in z direction
     * ---------------------------------------------------------------------
     */
    lhsA(m, n, 0, j, i - 1) = 0.0;
    lhsB(m, n, 0, j, i - 1) = (m == n) ? 1.0 : 0.0;
    lhsC(m, n, 0, j, i - 1) = 0.0;

    lhsA(m, n, ksize, j, i - 1) = 0.0;
    lhsB(m, n, ksize, j, i - 1) = (m == n) ? 1.0 : 0.0;
    lhsC(m, n, ksize, j, i - 1) = 0.0;

#undef lhsA
#undef lhsB
#undef lhsC
}

__global__ void z_solve_gpu_kernel_2(double* qs_device, double* square_device,
                                     double* u_device, double* lhsA_device,
                                     double* lhsB_device, double* lhsC_device) {
    int k = blockDim.z * blockIdx.z + threadIdx.z + 1;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (k > KMAX - 2 || j + 1 < 1 || j + 1 > JMAX - 2 || j >= PROBLEM_SIZE ||
        i > IMAX - 2) {
        return;
    }

    j++;

    int m;
    double tmp1, tmp2;

#define qs(a, b, c) qs_device[((a) * (JMAXP + 1) + (b)) * (IMAXP + 1) + (c)]
#define square(a, b, c) \
    square_device[((a) * (JMAXP + 1) + (b)) * (IMAXP + 1) + (c)]
#define u(a, b, c, d) \
    u_device[(((a) * (JMAXP + 1) + (b)) * (IMAXP + 1) + (c)) * 5 + (d)]

#define lhsA(a, b, c, d, e)                                                   \
    lhsA_device[((((a) * 5 + (b)) * (PROBLEM_SIZE + 1) + (c)) * (JMAXP + 1) + \
                 (d)) *                                                       \
                    (PROBLEM_SIZE - 1) +                                      \
                (e)]
#define lhsB(a, b, c, d, e)                                                   \
    lhsB_device[((((a) * 5 + (b)) * (PROBLEM_SIZE + 1) + (c)) * (JMAXP + 1) + \
                 (d)) *                                                       \
                    (PROBLEM_SIZE - 1) +                                      \
                (e)]
#define lhsC(a, b, c, d, e)                                                   \
    lhsC_device[((((a) * 5 + (b)) * (PROBLEM_SIZE + 1) + (c)) * (JMAXP + 1) + \
                 (d)) *                                                       \
                    (PROBLEM_SIZE - 1) +                                      \
                (e)]

    double fjac[5][5];
    double njac[5][5];

    double t_u[5];

    /*
     * ---------------------------------------------------------------------
     * compute the indices for storing the block-diagonal matrix;
     * determine c (labeled f) and s jacobians
     * ---------------------------------------------------------------------
     */
    tmp1 = constants_device::dt * constants_device::tz1;
    tmp2 = constants_device::dt * constants_device::tz2;

    for (m = 0; m < 5; m++) {
        t_u[m] = u(k - 1, j, i, m);
    }

    z_solve_gpu_device_fjac(fjac, t_u, square(k - 1, j, i), qs(k - 1, j, i));
    z_solve_gpu_device_njac(njac, t_u);

    lhsA(0, 0, k, j, i - 1) =
        -tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * constants_device::dz1;
    lhsA(1, 0, k, j, i - 1) = -tmp2 * fjac[1][0] - tmp1 * njac[1][0];
    lhsA(2, 0, k, j, i - 1) = -tmp2 * fjac[2][0] - tmp1 * njac[2][0];
    lhsA(3, 0, k, j, i - 1) = -tmp2 * fjac[3][0] - tmp1 * njac[3][0];
    lhsA(4, 0, k, j, i - 1) = -tmp2 * fjac[4][0] - tmp1 * njac[4][0];

    lhsA(0, 1, k, j, i - 1) = -tmp2 * fjac[0][1] - tmp1 * njac[0][1];
    lhsA(1, 1, k, j, i - 1) =
        -tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * constants_device::dz2;
    lhsA(2, 1, k, j, i - 1) = -tmp2 * fjac[2][1] - tmp1 * njac[2][1];
    lhsA(3, 1, k, j, i - 1) = -tmp2 * fjac[3][1] - tmp1 * njac[3][1];
    lhsA(4, 1, k, j, i - 1) = -tmp2 * fjac[4][1] - tmp1 * njac[4][1];

    lhsA(0, 2, k, j, i - 1) = -tmp2 * fjac[0][2] - tmp1 * njac[0][2];
    lhsA(1, 2, k, j, i - 1) = -tmp2 * fjac[1][2] - tmp1 * njac[1][2];
    lhsA(2, 2, k, j, i - 1) =
        -tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * constants_device::dz3;
    lhsA(3, 2, k, j, i - 1) = -tmp2 * fjac[3][2] - tmp1 * njac[3][2];
    lhsA(4, 2, k, j, i - 1) = -tmp2 * fjac[4][2] - tmp1 * njac[4][2];

    lhsA(0, 3, k, j, i - 1) = -tmp2 * fjac[0][3] - tmp1 * njac[0][3];
    lhsA(1, 3, k, j, i - 1) = -tmp2 * fjac[1][3] - tmp1 * njac[1][3];
    lhsA(2, 3, k, j, i - 1) = -tmp2 * fjac[2][3] - tmp1 * njac[2][3];
    lhsA(3, 3, k, j, i - 1) =
        -tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * constants_device::dz4;
    lhsA(4, 3, k, j, i - 1) = -tmp2 * fjac[4][3] - tmp1 * njac[4][3];

    lhsA(0, 4, k, j, i - 1) = -tmp2 * fjac[0][4] - tmp1 * njac[0][4];
    lhsA(1, 4, k, j, i - 1) = -tmp2 * fjac[1][4] - tmp1 * njac[1][4];
    lhsA(2, 4, k, j, i - 1) = -tmp2 * fjac[2][4] - tmp1 * njac[2][4];
    lhsA(3, 4, k, j, i - 1) = -tmp2 * fjac[3][4] - tmp1 * njac[3][4];
    lhsA(4, 4, k, j, i - 1) =
        -tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * constants_device::dz5;

    for (m = 0; m < 5; m++) {
        t_u[m] = u(k, j, i, m);
    }

    z_solve_gpu_device_njac(njac, t_u);

    lhsB(0, 0, k, j, i - 1) =
        1.0 + tmp1 * 2.0 * njac[0][0] + tmp1 * 2.0 * constants_device::dz1;
    lhsB(1, 0, k, j, i - 1) = tmp1 * 2.0 * njac[1][0];
    lhsB(2, 0, k, j, i - 1) = tmp1 * 2.0 * njac[2][0];
    lhsB(3, 0, k, j, i - 1) = tmp1 * 2.0 * njac[3][0];
    lhsB(4, 0, k, j, i - 1) = tmp1 * 2.0 * njac[4][0];

    lhsB(0, 1, k, j, i - 1) = tmp1 * 2.0 * njac[0][1];
    lhsB(1, 1, k, j, i - 1) =
        1.0 + tmp1 * 2.0 * njac[1][1] + tmp1 * 2.0 * constants_device::dz2;
    lhsB(2, 1, k, j, i - 1) = tmp1 * 2.0 * njac[2][1];
    lhsB(3, 1, k, j, i - 1) = tmp1 * 2.0 * njac[3][1];
    lhsB(4, 1, k, j, i - 1) = tmp1 * 2.0 * njac[4][1];

    lhsB(0, 2, k, j, i - 1) = tmp1 * 2.0 * njac[0][2];
    lhsB(1, 2, k, j, i - 1) = tmp1 * 2.0 * njac[1][2];
    lhsB(2, 2, k, j, i - 1) =
        1.0 + tmp1 * 2.0 * njac[2][2] + tmp1 * 2.0 * constants_device::dz3;
    lhsB(3, 2, k, j, i - 1) = tmp1 * 2.0 * njac[3][2];
    lhsB(4, 2, k, j, i - 1) = tmp1 * 2.0 * njac[4][2];

    lhsB(0, 3, k, j, i - 1) = tmp1 * 2.0 * njac[0][3];
    lhsB(1, 3, k, j, i - 1) = tmp1 * 2.0 * njac[1][3];
    lhsB(2, 3, k, j, i - 1) = tmp1 * 2.0 * njac[2][3];
    lhsB(3, 3, k, j, i - 1) =
        1.0 + tmp1 * 2.0 * njac[3][3] + tmp1 * 2.0 * constants_device::dz4;
    lhsB(4, 3, k, j, i - 1) = tmp1 * 2.0 * njac[4][3];

    lhsB(0, 4, k, j, i - 1) = tmp1 * 2.0 * njac[0][4];
    lhsB(1, 4, k, j, i - 1) = tmp1 * 2.0 * njac[1][4];
    lhsB(2, 4, k, j, i - 1) = tmp1 * 2.0 * njac[2][4];
    lhsB(3, 4, k, j, i - 1) = tmp1 * 2.0 * njac[3][4];
    lhsB(4, 4, k, j, i - 1) =
        1.0 + tmp1 * 2.0 * njac[4][4] + tmp1 * 2.0 * constants_device::dz5;

    for (m = 0; m < 5; m++) {
        t_u[m] = u(k + 1, j, i, m);
    }

    z_solve_gpu_device_fjac(fjac, t_u, square(k + 1, j, i), qs(k + 1, j, i));
    z_solve_gpu_device_njac(njac, t_u);

    lhsC(0, 0, k, j, i - 1) =
        tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * constants_device::dz1;
    lhsC(1, 0, k, j, i - 1) = tmp2 * fjac[1][0] - tmp1 * njac[1][0];
    lhsC(2, 0, k, j, i - 1) = tmp2 * fjac[2][0] - tmp1 * njac[2][0];
    lhsC(3, 0, k, j, i - 1) = tmp2 * fjac[3][0] - tmp1 * njac[3][0];
    lhsC(4, 0, k, j, i - 1) = tmp2 * fjac[4][0] - tmp1 * njac[4][0];

    lhsC(0, 1, k, j, i - 1) = tmp2 * fjac[0][1] - tmp1 * njac[0][1];
    lhsC(1, 1, k, j, i - 1) =
        tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * constants_device::dz2;
    lhsC(2, 1, k, j, i - 1) = tmp2 * fjac[2][1] - tmp1 * njac[2][1];
    lhsC(3, 1, k, j, i - 1) = tmp2 * fjac[3][1] - tmp1 * njac[3][1];
    lhsC(4, 1, k, j, i - 1) = tmp2 * fjac[4][1] - tmp1 * njac[4][1];

    lhsC(0, 2, k, j, i - 1) = tmp2 * fjac[0][2] - tmp1 * njac[0][2];
    lhsC(1, 2, k, j, i - 1) = tmp2 * fjac[1][2] - tmp1 * njac[1][2];
    lhsC(2, 2, k, j, i - 1) =
        tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * constants_device::dz3;
    lhsC(3, 2, k, j, i - 1) = tmp2 * fjac[3][2] - tmp1 * njac[3][2];
    lhsC(4, 2, k, j, i - 1) = tmp2 * fjac[4][2] - tmp1 * njac[4][2];

    lhsC(0, 3, k, j, i - 1) = tmp2 * fjac[0][3] - tmp1 * njac[0][3];
    lhsC(1, 3, k, j, i - 1) = tmp2 * fjac[1][3] - tmp1 * njac[1][3];
    lhsC(2, 3, k, j, i - 1) = tmp2 * fjac[2][3] - tmp1 * njac[2][3];
    lhsC(3, 3, k, j, i - 1) =
        tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * constants_device::dz4;
    lhsC(4, 3, k, j, i - 1) = tmp2 * fjac[4][3] - tmp1 * njac[4][3];

    lhsC(0, 4, k, j, i - 1) = tmp2 * fjac[0][4] - tmp1 * njac[0][4];
    lhsC(1, 4, k, j, i - 1) = tmp2 * fjac[1][4] - tmp1 * njac[1][4];
    lhsC(2, 4, k, j, i - 1) = tmp2 * fjac[2][4] - tmp1 * njac[2][4];
    lhsC(3, 4, k, j, i - 1) = tmp2 * fjac[3][4] - tmp1 * njac[3][4];
    lhsC(4, 4, k, j, i - 1) =
        tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * constants_device::dz5;

#undef qs
#undef square
#undef u
#undef lhsA
#undef lhsB
#undef lhsC
}

/*
 * ---------------------------------------------------------------------
 * this function computes the left hand side for the three z-factors
 * ---------------------------------------------------------------------
 */
__global__ void z_solve_gpu_kernel_3(double* rhs_device, double* lhsA_device,
                                     double* lhsB_device, double* lhsC_device) {
    extern __shared__ double tmp_l_lhs[];
    double* tmp_l_r = &tmp_l_lhs[blockDim.x * 3 * 5 * 5];

    int t_j = blockDim.y * blockIdx.y + threadIdx.y;
    int j = t_j / 5;
    int m = t_j % 5;
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int l_i = threadIdx.x;
    if (j + 1 < 1 || j + 1 > JMAX - 2 || j >= PROBLEM_SIZE || i > IMAX - 2) {
        return;
    }

    j++;

    int k, n, p, ksize;

#define rhs(a, b, c, d) \
    rhs_device[(((a) * (JMAXP + 1) + (b)) * (IMAXP + 1) + (c)) * 5 + (d)]

#define lhsA(a, b, c, d, e)                                                   \
    lhsA_device[((((a) * 5 + (b)) * (PROBLEM_SIZE + 1) + (c)) * (JMAXP + 1) + \
                 (d)) *                                                       \
                    (PROBLEM_SIZE - 1) +                                      \
                (e)]
#define lhsB(a, b, c, d, e)                                                   \
    lhsB_device[((((a) * 5 + (b)) * (PROBLEM_SIZE + 1) + (c)) * (JMAXP + 1) + \
                 (d)) *                                                       \
                    (PROBLEM_SIZE - 1) +                                      \
                (e)]
#define lhsC(a, b, c, d, e)                                                   \
    lhsC_device[((((a) * 5 + (b)) * (PROBLEM_SIZE + 1) + (c)) * (JMAXP + 1) + \
                 (d)) *                                                       \
                    (PROBLEM_SIZE - 1) +                                      \
                (e)]

    double (*tmp2_l_lhs)[3][5][5] = (double (*)[3][5][5])tmp_l_lhs;
    double (*l_lhs)[5][5] = tmp2_l_lhs[l_i];
    double (*tmp2_l_r)[2][5] = (double (*)[2][5])tmp_l_r;
    double (*l_r)[5] = tmp2_l_r[l_i];

    double pivot, coeff;

    ksize = KMAX - 1;

    /*
     * ---------------------------------------------------------------------
     * compute the indices for storing the block-diagonal matrix;
     * determine c (labeled f) and s jacobians
     * ---------------------------------------------------------------------
     * performs guaussian elimination on this cell.
     * ---------------------------------------------------------------------
     * assumes that unpacking routines for non-first cells
     * preload C' and rhs' from previous cell.
     * ---------------------------------------------------------------------
     * assumed send happens outside this routine, but that
     * c'(KMAX) and rhs'(KMAX) will be sent to next cell.
     * ---------------------------------------------------------------------
     * outer most do loops - sweeping in i direction
     * ---------------------------------------------------------------------
     */

    /* load data */
    for (p = 0; p < 5; p++) {
        l_lhs[BB][p][m] = lhsB(p, m, 0, j, i - 1);
        l_lhs[CC][p][m] = lhsC(p, m, 0, j, i - 1);
    }

    l_r[1][m] = rhs(0, j, i, m);

    __syncthreads();

    /*
     * ---------------------------------------------------------------------
     * multiply c[0][j][i] by b_inverse and copy back to c
     * multiply rhs(0) by b_inverse(0) and copy to rhs
     * ---------------------------------------------------------------------
     */
    for (p = 0; p < 5; p++) {
        pivot = 1.00 / l_lhs[BB][p][p];
        if (m > p && m < 5) {
            l_lhs[BB][m][p] = l_lhs[BB][m][p] * pivot;
        }
        if (m < 5) {
            l_lhs[CC][m][p] = l_lhs[CC][m][p] * pivot;
        }
        if (p == m) {
            l_r[1][p] = l_r[1][p] * pivot;
        }

        __syncthreads();

        if (p != m) {
            coeff = l_lhs[BB][p][m];
            for (n = p + 1; n < 5; n++) {
                l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff * l_lhs[BB][n][p];
            }
            for (n = 0; n < 5; n++) {
                l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff * l_lhs[CC][n][p];
            }
            l_r[1][m] = l_r[1][m] - coeff * l_r[1][p];
        }

        __syncthreads();
    }

    /* update data */
    rhs(0, j, i, m) = l_r[1][m];

    /*
     * ---------------------------------------------------------------------
     * begin inner most do loop
     * do all the elements of the cell unless last
     * ---------------------------------------------------------------------
     */
    for (k = 1; k <= ksize - 1; k++) {
        /* load data */
        for (n = 0; n < 5; n++) {
            l_lhs[AA][n][m] = lhsA(n, m, k, j, i - 1);
            l_lhs[BB][n][m] = lhsB(n, m, k, j, i - 1);
        }
        l_r[0][m] = l_r[1][m];
        l_r[1][m] = rhs(k, j, i, m);

        __syncthreads();

        /*
         * ---------------------------------------------------------------------
         * subtract A*lhs_vector(k-1) from lhs_vector(k)
         *
         * rhs(k) = rhs(k) - A*rhs(k-1)
         * ---------------------------------------------------------------------
         */
        l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m] * l_r[0][0] -
                    l_lhs[AA][1][m] * l_r[0][1] - l_lhs[AA][2][m] * l_r[0][2] -
                    l_lhs[AA][3][m] * l_r[0][3] - l_lhs[AA][4][m] * l_r[0][4];

        /*
         * ---------------------------------------------------------------------
         * B(k) = B(k) - C(k-1)*A(k)
         * matmul_sub(AA,i,j,k,c,CC,i,j,k-1,c,BB,i,j,k)
         * ---------------------------------------------------------------------
         */
        for (p = 0; p < 5; p++) {
            l_lhs[BB][m][p] = l_lhs[BB][m][p] -
                              l_lhs[AA][0][p] * l_lhs[CC][m][0] -
                              l_lhs[AA][1][p] * l_lhs[CC][m][1] -
                              l_lhs[AA][2][p] * l_lhs[CC][m][2] -
                              l_lhs[AA][3][p] * l_lhs[CC][m][3] -
                              l_lhs[AA][4][p] * l_lhs[CC][m][4];
        }

        __syncthreads();

        /* load data */
        for (n = 0; n < 5; n++) {
            l_lhs[CC][n][m] = lhsC(n, m, k, j, i - 1);
        }

        __syncthreads();

        /*
         * ---------------------------------------------------------------------
         * multiply c[k][j][i] by b_inverse and copy back to c
         * multiply rhs[0][j][i] by b_inverse[0][j][i] and copy to rhs
         * ---------------------------------------------------------------------
         */
        for (p = 0; p < 5; p++) {
            pivot = 1.00 / l_lhs[BB][p][p];
            if (m > p && m < 5) {
                l_lhs[BB][m][p] = l_lhs[BB][m][p] * pivot;
            }
            if (m < 5) {
                l_lhs[CC][m][p] = l_lhs[CC][m][p] * pivot;
            }
            if (p == m) {
                l_r[1][p] = l_r[1][p] * pivot;
            }

            __syncthreads();

            if (p != m) {
                coeff = l_lhs[BB][p][m];
                for (n = p + 1; n < 5; n++) {
                    l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff * l_lhs[BB][n][p];
                }
                for (n = 0; n < 5; n++) {
                    l_lhs[CC][n][m] = l_lhs[CC][n][m] - coeff * l_lhs[CC][n][p];
                }
                l_r[1][m] = l_r[1][m] - coeff * l_r[1][p];
            }

            __syncthreads();
        }

        /* update data */
        for (n = 0; n < 5; n++) {
            lhsC(n, m, k, j, i - 1) = l_lhs[CC][n][m];
        }
        rhs(k, j, i, m) = l_r[1][m];
    }

    /*
     * ---------------------------------------------------------------------
     * now finish up special cases for last cell
     * ---------------------------------------------------------------------
     */
    /* load data */
    for (n = 0; n < 5; n++) {
        l_lhs[AA][n][m] = lhsA(n, m, k, j, i - 1);
        l_lhs[BB][n][m] = lhsB(n, m, k, j, i - 1);
    }
    l_r[0][m] = l_r[1][m];
    l_r[1][m] = rhs(k, j, i, m);

    __syncthreads();

    /*
     * ---------------------------------------------------------------------
     * rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
     * ---------------------------------------------------------------------
     */
    l_r[1][m] = l_r[1][m] - l_lhs[AA][0][m] * l_r[0][0] -
                l_lhs[AA][1][m] * l_r[0][1] - l_lhs[AA][2][m] * l_r[0][2] -
                l_lhs[AA][3][m] * l_r[0][3] - l_lhs[AA][4][m] * l_r[0][4];

    /*
     * ---------------------------------------------------------------------
     * B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
     * matmul_sub(AA,i,j,ksize,c,CC,i,j,ksize-1,c,BB,i,j,ksize)
     * ---------------------------------------------------------------------
     */
    for (p = 0; p < 5; p++) {
        l_lhs[BB][m][p] = l_lhs[BB][m][p] - l_lhs[AA][0][p] * l_lhs[CC][m][0] -
                          l_lhs[AA][1][p] * l_lhs[CC][m][1] -
                          l_lhs[AA][2][p] * l_lhs[CC][m][2] -
                          l_lhs[AA][3][p] * l_lhs[CC][m][3] -
                          l_lhs[AA][4][p] * l_lhs[CC][m][4];
    }

    /*
     * ---------------------------------------------------------------------
     * multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
     * ---------------------------------------------------------------------
     */
    for (p = 0; p < 5; p++) {
        pivot = 1.00 / l_lhs[BB][p][p];
        if (m > p && m < 5) {
            l_lhs[BB][m][p] = l_lhs[BB][m][p] * pivot;
        }
        if (p == m) {
            l_r[1][p] = l_r[1][p] * pivot;
        }

        __syncthreads();

        if (p != m) {
            coeff = l_lhs[BB][p][m];
            for (n = p + 1; n < 5; n++) {
                l_lhs[BB][n][m] = l_lhs[BB][n][m] - coeff * l_lhs[BB][n][p];
            }
            l_r[1][m] = l_r[1][m] - coeff * l_r[1][p];
        }

        __syncthreads();
    }

    /* update data */
    rhs(k, j, i, m) = l_r[1][m];

    __syncthreads();

    /*
     * ---------------------------------------------------------------------
     * back solve: if last cell, then generate U(ksize)=rhs(ksize)
     * else assume U(ksize) is loaded in un pack backsub_info
     * so just use it
     * after u(kstart) will be sent to next cell
     * ---------------------------------------------------------------------
     */
    for (k = ksize - 1; k >= 0; k--) {
        for (n = 0; n < M_SIZE; n++) {
            rhs(k, j, i, m) =
                rhs(k, j, i, m) - lhsC(n, m, k, j, i - 1) * rhs(k + 1, j, i, n);
        }

        __syncthreads();
    }
#undef rhs
#undef lhsA
#undef lhsB
#undef lhsC
}
