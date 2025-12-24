#include "sp.cuh"

/*
 * ---------------------------------------------------------------------
 * addition of update to the vector u
 * ---------------------------------------------------------------------
 */
__global__ void add_gpu_kernel(double* u, const double* rhs, const int nx,
                               const int ny, const int nz) {
    int i_j_k, i, j, k;

    i_j_k = blockIdx.x * blockDim.x + threadIdx.x;

    i = i_j_k % nx;
    j = (i_j_k / nx) % ny;
    k = i_j_k / (nx * ny);

    if (i_j_k >= (nx * ny * nz)) return;

    /* array(m,i,j,k) */
    u(0, i, j, k) += rhs(0, i, j, k);
    u(1, i, j, k) += rhs(1, i, j, k);
    u(2, i, j, k) += rhs(2, i, j, k);
    u(3, i, j, k) += rhs(3, i, j, k);
    u(4, i, j, k) += rhs(4, i, j, k);
}

__global__ void compute_rhs_gpu_kernel_1(double* rho_i, double* us, double* vs,
                                         double* ws, double* speed, double* qs,
                                         double* square, const double* u,
                                         const int nx, const int ny,
                                         const int nz) {
    int i_j_k, i, j, k;

    i_j_k = blockIdx.x * blockDim.x + threadIdx.x;

    i = i_j_k % nx;
    j = (i_j_k / nx) % ny;
    k = i_j_k / (nx * ny);

    if (i_j_k >= (nx * ny * nz)) return;

    using namespace constants_device;
    /*
     * ---------------------------------------------------------------------
     * compute the reciprocal of density, and the kinetic energy,
     * and the speed of sound.
     * ---------------------------------------------------------------------
     */
    double rho_inv = 1.0 / u(0, i, j, k);
    double square_ijk;
    rho_i(i, j, k) = rho_inv;
    us(i, j, k) = u(1, i, j, k) * rho_inv;
    vs(i, j, k) = u(2, i, j, k) * rho_inv;
    ws(i, j, k) = u(3, i, j, k) * rho_inv;
    square(i, j, k) = square_ijk =
        0.5 *
        (u(1, i, j, k) * u(1, i, j, k) + u(2, i, j, k) * u(2, i, j, k) +
         u(3, i, j, k) * u(3, i, j, k)) *
        rho_inv;
    qs(i, j, k) = square_ijk * rho_inv;
    /*
     * ---------------------------------------------------------------------
     * (don't need speed and ainx until the lhs computation)
     * ---------------------------------------------------------------------
     */
    speed(i, j, k) = sqrt(c1c2 * rho_inv * (u(4, i, j, k) - square_ijk));
}

__global__ void compute_rhs_gpu_kernel_2(const double* rho_i, const double* us,
                                         const double* vs, const double* ws,
                                         const double* qs, const double* square,
                                         double* rhs, const double* forcing,
                                         const double* u, const int nx,
                                         const int ny, const int nz) {
    int i, j, k, m;

    k = blockIdx.y;
    j = blockIdx.x;
    i = threadIdx.x;

    double rtmp[5];
    using namespace constants_device;
    /*
     * ---------------------------------------------------------------------
     * copy the exact forcing term to the right hand side;  because
     * this forcing term is known, we can store it on the whole grid
     * including the boundary
     * ---------------------------------------------------------------------
     */
    for (m = 0; m < 5; m++) rtmp[m] = forcing(m, i, j, k);
    /*
     * ---------------------------------------------------------------------
     * compute xi-direction fluxes
     * ---------------------------------------------------------------------
     */
    if (k >= 1 && k < nz - 1 && j >= 1 && j < ny - 1 && i >= 1 && i < nx - 1) {
        double uijk = us(i, j, k);
        double up1 = us(i + 1, j, k);
        double um1 = us(i - 1, j, k);
        rtmp[0] = rtmp[0] +
                  dx1tx1 * (u(0, i + 1, j, k) - 2.0 * u(0, i, j, k) +
                            u(0, i - 1, j, k)) -
                  tx2 * (u(1, i + 1, j, k) - u(1, i - 1, j, k));
        rtmp[1] = rtmp[1] +
                  dx2tx1 * (u(1, i + 1, j, k) - 2.0 * u(1, i, j, k) +
                            u(1, i - 1, j, k)) +
                  xxcon2 * con43 * (up1 - 2.0 * uijk + um1) -
                  tx2 * (u(1, i + 1, j, k) * up1 - u(1, i - 1, j, k) * um1 +
                         (u(4, i + 1, j, k) - square(i + 1, j, k) -
                          u(4, i - 1, j, k) + square(i - 1, j, k)) *
                             c2);
        rtmp[2] =
            rtmp[2] +
            dx3tx1 *
                (u(2, i + 1, j, k) - 2.0 * u(2, i, j, k) + u(2, i - 1, j, k)) +
            xxcon2 * (vs(i + 1, j, k) - 2.0 * vs(i, j, k) + vs(i - 1, j, k)) -
            tx2 * (u(2, i + 1, j, k) * up1 - u(2, i - 1, j, k) * um1);
        rtmp[3] =
            rtmp[3] +
            dx4tx1 *
                (u(3, i + 1, j, k) - 2.0 * u(3, i, j, k) + u(3, i - 1, j, k)) +
            xxcon2 * (ws(i + 1, j, k) - 2.0 * ws(i, j, k) + ws(i - 1, j, k)) -
            tx2 * (u(3, i + 1, j, k) * up1 - u(3, i - 1, j, k) * um1);
        rtmp[4] =
            rtmp[4] +
            dx5tx1 *
                (u(4, i + 1, j, k) - 2.0 * u(4, i, j, k) + u(4, i - 1, j, k)) +
            xxcon3 * (qs(i + 1, j, k) - 2.0 * qs(i, j, k) + qs(i - 1, j, k)) +
            xxcon4 * (up1 * up1 - 2.0 * uijk * uijk + um1 * um1) +
            xxcon5 * (u(4, i + 1, j, k) * rho_i(i + 1, j, k) -
                      2.0 * u(4, i, j, k) * rho_i(i, j, k) +
                      u(4, i - 1, j, k) * rho_i(i - 1, j, k)) -
            tx2 * ((c1 * u(4, i + 1, j, k) - c2 * square(i + 1, j, k)) * up1 -
                   (c1 * u(4, i - 1, j, k) - c2 * square(i - 1, j, k)) * um1);
        /*
         * ---------------------------------------------------------------------
         * add fourth order xi-direction dissipation
         * ---------------------------------------------------------------------
         */
        if (i == 1) {
            for (m = 0; m < 5; m++)
                rtmp[m] = rtmp[m] -
                          dssp * (5.0 * u(m, i, j, k) -
                                  4.0 * u(m, i + 1, j, k) + u(m, i + 2, j, k));
        } else if (i == 2) {
            for (m = 0; m < 5; m++)
                rtmp[m] =
                    rtmp[m] -
                    dssp * (-4.0 * u(m, i - 1, j, k) + 6.0 * u(m, i, j, k) -
                            4.0 * u(m, i + 1, j, k) + u(m, i + 2, j, k));
        } else if (i >= 3 && i < nx - 3) {
            for (m = 0; m < 5; m++)
                rtmp[m] = rtmp[m] -
                          dssp * (u(m, i - 2, j, k) - 4.0 * u(m, i - 1, j, k) +
                                  6.0 * u(m, i, j, k) -
                                  4.0 * u(m, i + 1, j, k) + u(m, i + 2, j, k));
        } else if (i == nx - 3) {
            for (m = 0; m < 5; m++)
                rtmp[m] =
                    rtmp[m] -
                    dssp * (u(m, i - 2, j, k) - 4.0 * u(m, i - 1, j, k) +
                            6.0 * u(m, i, j, k) - 4.0 * u(m, i + 1, j, k));
        } else if (i == nx - 2) {
            for (m = 0; m < 5; m++)
                rtmp[m] = rtmp[m] -
                          dssp * (u(m, i - 2, j, k) - 4.0 * u(m, i - 1, j, k) +
                                  5.0 * u(m, i, j, k));
        }
        /*
         * ---------------------------------------------------------------------
         * compute eta-direction fluxes
         * ---------------------------------------------------------------------
         */
        double vijk = vs(i, j, k);
        double vp1 = vs(i, j + 1, k);
        double vm1 = vs(i, j - 1, k);
        rtmp[0] = rtmp[0] +
                  dy1ty1 * (u(0, i, j + 1, k) - 2.0 * u(0, i, j, k) +
                            u(0, i, j - 1, k)) -
                  ty2 * (u(2, i, j + 1, k) - u(2, i, j - 1, k));
        rtmp[1] =
            rtmp[1] +
            dy2ty1 *
                (u(1, i, j + 1, k) - 2.0 * u(1, i, j, k) + u(1, i, j - 1, k)) +
            yycon2 * (us(i, j + 1, k) - 2.0 * us(i, j, k) + us(i, j - 1, k)) -
            ty2 * (u(1, i, j + 1, k) * vp1 - u(1, i, j - 1, k) * vm1);
        rtmp[2] = rtmp[2] +
                  dy3ty1 * (u(2, i, j + 1, k) - 2.0 * u(2, i, j, k) +
                            u(2, i, j - 1, k)) +
                  yycon2 * con43 * (vp1 - 2.0 * vijk + vm1) -
                  ty2 * (u(2, i, j + 1, k) * vp1 - u(2, i, j - 1, k) * vm1 +
                         (u(4, i, j + 1, k) - square(i, j + 1, k) -
                          u(4, i, j - 1, k) + square(i, j - 1, k)) *
                             c2);
        rtmp[3] =
            rtmp[3] +
            dy4ty1 *
                (u(3, i, j + 1, k) - 2.0 * u(3, i, j, k) + u(3, i, j - 1, k)) +
            yycon2 * (ws(i, j + 1, k) - 2.0 * ws(i, j, k) + ws(i, j - 1, k)) -
            ty2 * (u(3, i, j + 1, k) * vp1 - u(3, i, j - 1, k) * vm1);
        rtmp[4] =
            rtmp[4] +
            dy5ty1 *
                (u(4, i, j + 1, k) - 2.0 * u(4, i, j, k) + u(4, i, j - 1, k)) +
            yycon3 * (qs(i, j + 1, k) - 2.0 * qs(i, j, k) + qs(i, j - 1, k)) +
            yycon4 * (vp1 * vp1 - 2.0 * vijk * vijk + vm1 * vm1) +
            yycon5 * (u(4, i, j + 1, k) * rho_i(i, j + 1, k) -
                      2.0 * u(4, i, j, k) * rho_i(i, j, k) +
                      u(4, i, j - 1, k) * rho_i(i, j - 1, k)) -
            ty2 * ((c1 * u(4, i, j + 1, k) - c2 * square(i, j + 1, k)) * vp1 -
                   (c1 * u(4, i, j - 1, k) - c2 * square(i, j - 1, k)) * vm1);
        /*
         * ---------------------------------------------------------------------
         * add fourth order eta-direction dissipation
         * ---------------------------------------------------------------------
         */
        if (j == 1) {
            for (m = 0; m < 5; m++)
                rtmp[m] = rtmp[m] -
                          dssp * (5.0 * u(m, i, j, k) -
                                  4.0 * u(m, i, j + 1, k) + u(m, i, j + 2, k));
        } else if (j == 2) {
            for (m = 0; m < 5; m++)
                rtmp[m] =
                    rtmp[m] -
                    dssp * (-4.0 * u(m, i, j - 1, k) + 6.0 * u(m, i, j, k) -
                            4.0 * u(m, i, j + 1, k) + u(m, i, j + 2, k));
        } else if (j >= 3 && j < ny - 3) {
            for (m = 0; m < 5; m++)
                rtmp[m] = rtmp[m] -
                          dssp * (u(m, i, j - 2, k) - 4.0 * u(m, i, j - 1, k) +
                                  6.0 * u(m, i, j, k) -
                                  4.0 * u(m, i, j + 1, k) + u(m, i, j + 2, k));

        } else if (j == ny - 3) {
            for (m = 0; m < 5; m++)
                rtmp[m] =
                    rtmp[m] -
                    dssp * (u(m, i, j - 2, k) - 4.0 * u(m, i, j - 1, k) +
                            6.0 * u(m, i, j, k) - 4.0 * u(m, i, j + 1, k));
        } else if (j == ny - 2) {
            for (m = 0; m < 5; m++)
                rtmp[m] = rtmp[m] -
                          dssp * (u(m, i, j - 2, k) - 4.0 * u(m, i, j - 1, k) +
                                  5.0 * u(m, i, j, k));
        }
        /*
         * ---------------------------------------------------------------------
         * compute zeta-direction fluxes
         * ---------------------------------------------------------------------
         */
        double wijk = ws(i, j, k);
        double wp1 = ws(i, j, k + 1);
        double wm1 = ws(i, j, k - 1);
        rtmp[0] = rtmp[0] +
                  dz1tz1 * (u(0, i, j, k + 1) - 2.0 * u(0, i, j, k) +
                            u(0, i, j, k - 1)) -
                  tz2 * (u(3, i, j, k + 1) - u(3, i, j, k - 1));
        rtmp[1] =
            rtmp[1] +
            dz2tz1 *
                (u(1, i, j, k + 1) - 2.0 * u(1, i, j, k) + u(1, i, j, k - 1)) +
            zzcon2 * (us(i, j, k + 1) - 2.0 * us(i, j, k) + us(i, j, k - 1)) -
            tz2 * (u(1, i, j, k + 1) * wp1 - u(1, i, j, k - 1) * wm1);
        rtmp[2] =
            rtmp[2] +
            dz3tz1 *
                (u(2, i, j, k + 1) - 2.0 * u(2, i, j, k) + u(2, i, j, k - 1)) +
            zzcon2 * (vs(i, j, k + 1) - 2.0 * vs(i, j, k) + vs(i, j, k - 1)) -
            tz2 * (u(2, i, j, k + 1) * wp1 - u(2, i, j, k - 1) * wm1);
        rtmp[3] = rtmp[3] +
                  dz4tz1 * (u(3, i, j, k + 1) - 2.0 * u(3, i, j, k) +
                            u(3, i, j, k - 1)) +
                  zzcon2 * con43 * (wp1 - 2.0 * wijk + wm1) -
                  tz2 * (u(3, i, j, k + 1) * wp1 - u(3, i, j, k - 1) * wm1 +
                         (u(4, i, j, k + 1) - square(i, j, k + 1) -
                          u(4, i, j, k - 1) + square(i, j, k - 1)) *
                             c2);
        rtmp[4] =
            rtmp[4] +
            dz5tz1 *
                (u(4, i, j, k + 1) - 2.0 * u(4, i, j, k) + u(4, i, j, k - 1)) +
            zzcon3 * (qs(i, j, k + 1) - 2.0 * qs(i, j, k) + qs(i, j, k - 1)) +
            zzcon4 * (wp1 * wp1 - 2.0 * wijk * wijk + wm1 * wm1) +
            zzcon5 * (u(4, i, j, k + 1) * rho_i(i, j, k + 1) -
                      2.0 * u(4, i, j, k) * rho_i(i, j, k) +
                      u(4, i, j, k - 1) * rho_i(i, j, k - 1)) -
            tz2 * ((c1 * u(4, i, j, k + 1) - c2 * square(i, j, k + 1)) * wp1 -
                   (c1 * u(4, i, j, k - 1) - c2 * square(i, j, k - 1)) * wm1);
        /*
         * ---------------------------------------------------------------------
         * add fourth order zeta-direction dissipation
         * ---------------------------------------------------------------------
         */
        if (k == 1) {
            for (m = 0; m < 5; m++)
                rtmp[m] = rtmp[m] -
                          dssp * (5.0 * u(m, i, j, k) -
                                  4.0 * u(m, i, j, k + 1) + u(m, i, j, k + 2));
        } else if (k == 2) {
            for (m = 0; m < 5; m++)
                rtmp[m] =
                    rtmp[m] -
                    dssp * (-4.0 * u(m, i, j, k - 1) + 6.0 * u(m, i, j, k) -
                            4.0 * u(m, i, j, k + 1) + u(m, i, j, k + 2));
        } else if (k >= 3 && k < nz - 3) {
            for (m = 0; m < 5; m++)
                rtmp[m] = rtmp[m] -
                          dssp * (u(m, i, j, k - 2) - 4.0 * u(m, i, j, k - 1) +
                                  6.0 * u(m, i, j, k) -
                                  4.0 * u(m, i, j, k + 1) + u(m, i, j, k + 2));
        } else if (k == nz - 3) {
            for (m = 0; m < 5; m++)
                rtmp[m] =
                    rtmp[m] -
                    dssp * (u(m, i, j, k - 2) - 4.0 * u(m, i, j, k - 1) +
                            6.0 * u(m, i, j, k) - 4.0 * u(m, i, j, k + 1));
        } else if (k == nz - 2) {
            for (m = 0; m < 5; m++)
                rtmp[m] = rtmp[m] -
                          dssp * (u(m, i, j, k - 2) - 4.0 * u(m, i, j, k - 1) +
                                  5.0 * u(m, i, j, k));
        }
        for (m = 0; m < 5; m++) rtmp[m] *= dt;
    }
    for (m = 0; m < 5; m++) {
        rhs(m, i, j, k) = rtmp[m];
    }
}

__global__ void txinvr_gpu_kernel(const double* rho_i, const double* us,
                                  const double* vs, const double* ws,
                                  const double* speed, const double* qs,
                                  double* rhs, const int nx, const int ny,
                                  const int nz) {
    int i_j_k, i, j, k;

    i_j_k = blockIdx.x * blockDim.x + threadIdx.x;

    i = i_j_k % nx;
    j = (i_j_k / nx) % ny;
    k = i_j_k / (nx * ny);

    if (i_j_k >= (nx * ny * nz)) return;

    using namespace constants_device;
    double ru1 = rho_i(i, j, k);
    double uu = us(i, j, k);
    double vv = vs(i, j, k);
    double ww = ws(i, j, k);
    double ac = speed(i, j, k);
    double ac2inv = 1.0 / (ac * ac);
    double r1 = rhs(0, i, j, k);
    double r2 = rhs(1, i, j, k);
    double r3 = rhs(2, i, j, k);
    double r4 = rhs(3, i, j, k);
    double r5 = rhs(4, i, j, k);
    double t1 =
        c2 * ac2inv * (qs(i, j, k) * r1 - uu * r2 - vv * r3 - ww * r4 + r5);
    double t2 = bt * ru1 * (uu * r1 - r2);
    double t3 = (bt * ru1 * ac) * t1;
    rhs(0, i, j, k) = r1 - t1;
    rhs(1, i, j, k) = -ru1 * (ww * r1 - r4);
    rhs(2, i, j, k) = ru1 * (vv * r1 - r3);
    rhs(3, i, j, k) = -t2 + t3;
    rhs(4, i, j, k) = t2 + t3;
}

__global__ void x_solve_gpu_kernel(const double* rho_i, const double* us,
                                   const double* speed, double* rhs,
                                   double* lhs, double* rhstmp, const int nx,
                                   const int ny, const int nz) {
#define lhs(m, i, j, k) \
    lhs[(j - 1) + (ny - 2) * ((k - 1) + (nz - 2) * ((i) + nx * (m - 3)))]
#define lhsp(m, i, j, k) \
    lhs[(j - 1) + (ny - 2) * ((k - 1) + (nz - 2) * ((i) + nx * (m + 4)))]
#define lhsm(m, i, j, k) \
    lhs[(j - 1) + (ny - 2) * ((k - 1) + (nz - 2) * ((i) + nx * (m - 3 + 2)))]
#define rtmp(m, i, j, k) rhstmp[(j) + ny * ((k) + nz * ((i) + nx * (m)))]
    int i, j, k, m;
    double rhon[3], cv[3], _lhs[3][5], _lhsp[3][5], _rhs[3][5], fac1;

    /* coalesced */
    j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    k = blockIdx.y * blockDim.y + threadIdx.y + 1;

    /* uncoalesced */
    /* k=blockIdx.x*blockDim.x+threadIdx.x+1; */
    /* j=blockIdx.y*blockDim.y+threadIdx.y+1; */

    if ((k >= nz - 1) || (j >= ny - 1)) return;

    using namespace constants_device;
    /*
     * ---------------------------------------------------------------------
     * computes the left hand side for the three x-factors
     * ---------------------------------------------------------------------
     * first fill the lhs for the u-eigenvalue
     * ---------------------------------------------------------------------
     */
    _lhs[0][0] = lhsp(0, 0, j, k) = 0.0;
    _lhs[0][1] = lhsp(1, 0, j, k) = 0.0;
    _lhs[0][2] = lhsp(2, 0, j, k) = 1.0;
    _lhs[0][3] = lhsp(3, 0, j, k) = 0.0;
    _lhs[0][4] = lhsp(4, 0, j, k) = 0.0;
    for (i = 0; i < 3; i++) {
        fac1 = c3c4 * rho_i(i, j, k);
        rhon[i] = max(
            max(max(dx2 + con43 * fac1, dx5 + c1c5 * fac1), dxmax + fac1), dx1);
        cv[i] = us(i, j, k);
    }
    _lhs[1][0] = 0.0;
    _lhs[1][1] = -dttx2 * cv[0] - dttx1 * rhon[0];
    _lhs[1][2] = 1.0 + c2dttx1 * rhon[1];
    _lhs[1][3] = dttx2 * cv[2] - dttx1 * rhon[2];
    _lhs[1][4] = 0.0;
    _lhs[1][2] += comz5;
    _lhs[1][3] -= comz4;
    _lhs[1][4] += comz1;
    for (m = 0; m < 5; m++) lhsp(m, 1, j, k) = _lhs[1][m];

    rhon[0] = rhon[1];
    rhon[1] = rhon[2];
    cv[0] = cv[1];
    cv[1] = cv[2];
    for (m = 0; m < 3; m++) {
        _rhs[0][m] = rhs(m, 0, j, k);
        _rhs[1][m] = rhs(m, 1, j, k);
    }
    /*
     * ---------------------------------------------------------------------
     * FORWARD ELIMINATION
     * ---------------------------------------------------------------------
     * perform the thomas algorithm; first, FORWARD ELIMINATION
     * ---------------------------------------------------------------------
     */
    for (i = 0; i < nx - 2; i++) {
        /*
         * ---------------------------------------------------------------------
         * first fill the lhs for the u-eigenvalue
         * ---------------------------------------------------------------------
         */
        if ((i + 2) == (nx - 1)) {
            _lhs[2][0] = lhsp(0, i + 2, j, k) = 0.0;
            _lhs[2][1] = lhsp(1, i + 2, j, k) = 0.0;
            _lhs[2][2] = lhsp(2, i + 2, j, k) = 1.0;
            _lhs[2][3] = lhsp(3, i + 2, j, k) = 0.0;
            _lhs[2][4] = lhsp(4, i + 2, j, k) = 0.0;
        } else {
            fac1 = c3c4 * rho_i(i + 3, j, k);
            rhon[2] = max(
                max(max(dx2 + con43 * fac1, dx5 + c1c5 * fac1), dxmax + fac1),
                dx1);
            cv[2] = us(i + 3, j, k);
            _lhs[2][0] = 0.0;
            _lhs[2][1] = -dttx2 * cv[0] - dttx1 * rhon[0];
            _lhs[2][2] = 1.0 + c2dttx1 * rhon[1];
            _lhs[2][3] = dttx2 * cv[2] - dttx1 * rhon[2];
            _lhs[2][4] = 0.0;
            /*
             * ---------------------------------------------------------------------
             * add fourth order dissipation
             * ---------------------------------------------------------------------
             */
            if ((i + 2) == (2)) {
                _lhs[2][1] -= comz4;
                _lhs[2][2] += comz6;
                _lhs[2][3] -= comz4;
                _lhs[2][4] += comz1;
            } else if ((i + 2 >= 3) && (i + 2 < nx - 3)) {
                _lhs[2][0] += comz1;
                _lhs[2][1] -= comz4;
                _lhs[2][2] += comz6;
                _lhs[2][3] -= comz4;
                _lhs[2][4] += comz1;
            } else if ((i + 2) == (nx - 3)) {
                _lhs[2][0] += comz1;
                _lhs[2][1] -= comz4;
                _lhs[2][2] += comz6;
                _lhs[2][3] -= comz4;
            } else if ((i + 2) == (nx - 2)) {
                _lhs[2][0] += comz1;
                _lhs[2][1] -= comz4;
                _lhs[2][2] += comz5;
            }
            /*
             * ---------------------------------------------------------------------
             * store computed lhs for later reuse
             * ---------------------------------------------------------------------
             */
            for (m = 0; m < 5; m++) lhsp(m, i + 2, j, k) = _lhs[2][m];

            rhon[0] = rhon[1];
            rhon[1] = rhon[2];
            cv[0] = cv[1];
            cv[1] = cv[2];
        }
        /*
         * ---------------------------------------------------------------------
         * load rhs values for current iteration
         * ---------------------------------------------------------------------
         */
        for (m = 0; m < 3; m++) _rhs[2][m] = rhs(m, i + 2, j, k);

        /*
         * ---------------------------------------------------------------------
         * perform current iteration
         * ---------------------------------------------------------------------
         */
        fac1 = 1.0 / _lhs[0][2];
        _lhs[0][3] *= fac1;
        _lhs[0][4] *= fac1;
        for (m = 0; m < 3; m++) _rhs[0][m] *= fac1;

        _lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
        _lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
        for (m = 0; m < 3; m++) _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];

        _lhs[2][1] -= _lhs[2][0] * _lhs[0][3];
        _lhs[2][2] -= _lhs[2][0] * _lhs[0][4];
        for (m = 0; m < 3; m++) _rhs[2][m] -= _lhs[2][0] * _rhs[0][m];

        /*
         * ---------------------------------------------------------------------
         * store computed lhs and prepare data for next iteration
         * rhs is stored in a temp array such that write accesses are coalesced
         * ---------------------------------------------------------------------
         */
        lhs(3, i, j, k) = _lhs[0][3];
        lhs(4, i, j, k) = _lhs[0][4];
        for (m = 0; m < 5; m++) {
            _lhs[0][m] = _lhs[1][m];
            _lhs[1][m] = _lhs[2][m];
        }
        for (m = 0; m < 3; m++) {
            rtmp(m, i, j, k) = _rhs[0][m];
            _rhs[0][m] = _rhs[1][m];
            _rhs[1][m] = _rhs[2][m];
        }
    }
    /*
     * ---------------------------------------------------------------------
     * the last two rows in this zone are a bit different,
     * since they do not have two more rows available for the
     * elimination of off-diagonal entries
     * ---------------------------------------------------------------------
     */
    i = nx - 2;
    fac1 = 1.0 / _lhs[0][2];
    _lhs[0][3] *= fac1;
    _lhs[0][4] *= fac1;
    for (m = 0; m < 3; m++) _rhs[0][m] *= fac1;

    _lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
    _lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
    for (m = 0; m < 3; m++) _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];

    /*
     * ---------------------------------------------------------------------
     * scale the last row immediately
     * ---------------------------------------------------------------------
     */
    fac1 = 1.0 / _lhs[1][2];
    for (m = 0; m < 3; m++) _rhs[1][m] *= fac1;

    lhs(3, nx - 2, j, k) = _lhs[0][3];
    lhs(4, nx - 2, j, k) = _lhs[0][4];
    /*
     * ---------------------------------------------------------------------
     * subsequently, fill the other factors (u+c), (u-c)
     * ---------------------------------------------------------------------
     */
    for (i = 0; i < 3; i++) cv[i] = speed(i, j, k);

    for (m = 0; m < 5; m++) {
        _lhsp[0][m] = _lhs[0][m] = lhsp(m, 0, j, k);
        _lhsp[1][m] = _lhs[1][m] = lhsp(m, 1, j, k);
    }
    _lhsp[1][1] -= dttx2 * cv[0];
    _lhsp[1][3] += dttx2 * cv[2];
    _lhs[1][1] += dttx2 * cv[0];
    _lhs[1][3] -= dttx2 * cv[2];
    cv[0] = cv[1];
    cv[1] = cv[2];
    _rhs[0][3] = rhs(3, 0, j, k);
    _rhs[0][4] = rhs(4, 0, j, k);
    _rhs[1][3] = rhs(3, 1, j, k);
    _rhs[1][4] = rhs(4, 1, j, k);
    /*
     * ---------------------------------------------------------------------
     * do the u+c and the u-c factors
     * ---------------------------------------------------------------------
     */
    for (i = 0; i < nx - 2; i++) {
        /*
         * first, fill the other factors (u+c), (u-c)
         * ---------------------------------------------------------------------
         */
        for (m = 0; m < 5; m++) {
            _lhsp[2][m] = _lhs[2][m] = lhsp(m, i + 2, j, k);
        }
        _rhs[2][3] = rhs(3, i + 2, j, k);
        _rhs[2][4] = rhs(4, i + 2, j, k);
        if ((i + 2) < (nx - 1)) {
            cv[2] = speed(i + 3, j, k);
            _lhsp[2][1] -= dttx2 * cv[0];
            _lhsp[2][3] += dttx2 * cv[2];
            _lhs[2][1] += dttx2 * cv[0];
            _lhs[2][3] -= dttx2 * cv[2];
            cv[0] = cv[1];
            cv[1] = cv[2];
        }
        m = 3;
        fac1 = 1.0 / _lhsp[0][2];
        _lhsp[0][3] *= fac1;
        _lhsp[0][4] *= fac1;
        _rhs[0][m] *= fac1;
        _lhsp[1][2] -= _lhsp[1][1] * _lhsp[0][3];
        _lhsp[1][3] -= _lhsp[1][1] * _lhsp[0][4];
        _rhs[1][m] -= _lhsp[1][1] * _rhs[0][m];
        _lhsp[2][1] -= _lhsp[2][0] * _lhsp[0][3];
        _lhsp[2][2] -= _lhsp[2][0] * _lhsp[0][4];
        _rhs[2][m] -= _lhsp[2][0] * _rhs[0][m];
        m = 4;
        fac1 = 1.0 / _lhs[0][2];
        _lhs[0][3] *= fac1;
        _lhs[0][4] *= fac1;
        _rhs[0][m] *= fac1;
        _lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
        _lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
        _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
        _lhs[2][1] -= _lhs[2][0] * _lhs[0][3];
        _lhs[2][2] -= _lhs[2][0] * _lhs[0][4];
        _rhs[2][m] -= _lhs[2][0] * _rhs[0][m];
        /*
         * ---------------------------------------------------------------------
         * store computed lhs and prepare data for next iteration
         * rhs is stored in a temp array such that write accesses are coalesced
         * ---------------------------------------------------------------------
         */
        for (m = 3; m < 5; m++) {
            lhsp(m, i, j, k) = _lhsp[0][m];
            lhsm(m, i, j, k) = _lhs[0][m];
            rtmp(m, i, j, k) = _rhs[0][m];
            _rhs[0][m] = _rhs[1][m];
            _rhs[1][m] = _rhs[2][m];
        }
        for (m = 0; m < 5; m++) {
            _lhsp[0][m] = _lhsp[1][m];
            _lhsp[1][m] = _lhsp[2][m];
            _lhs[0][m] = _lhs[1][m];
            _lhs[1][m] = _lhs[2][m];
        }
    }
    /*
     * ---------------------------------------------------------------------
     * and again the last two rows separately
     * ---------------------------------------------------------------------
     */
    i = nx - 2;
    m = 3;
    fac1 = 1.0 / _lhsp[0][2];
    _lhsp[0][3] *= fac1;
    _lhsp[0][4] *= fac1;
    _rhs[0][m] *= fac1;
    _lhsp[1][2] -= _lhsp[1][1] * _lhsp[0][3];
    _lhsp[1][3] -= _lhsp[1][1] * _lhsp[0][4];
    _rhs[1][m] -= _lhsp[1][1] * _rhs[0][m];
    m = 4;
    fac1 = 1.0 / _lhs[0][2];
    _lhs[0][3] *= fac1;
    _lhs[0][4] *= fac1;
    _rhs[0][m] *= fac1;
    _lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
    _lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
    _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
    /*
     * ---------------------------------------------------------------------
     * scale the last row immediately
     * ---------------------------------------------------------------------
     */
    _rhs[1][3] /= _lhsp[1][2];
    _rhs[1][4] /= _lhs[1][2];
    /*
     * ---------------------------------------------------------------------
     * BACKSUBSTITUTION
     * ---------------------------------------------------------------------
     */
    for (m = 0; m < 3; m++) _rhs[0][m] -= lhs(3, nx - 2, j, k) * _rhs[1][m];

    _rhs[0][3] -= _lhsp[0][3] * _rhs[1][3];
    _rhs[0][4] -= _lhs[0][3] * _rhs[1][4];
    for (m = 0; m < 5; m++) {
        _rhs[2][m] = _rhs[1][m];
        _rhs[1][m] = _rhs[0][m];
    }
    for (i = nx - 3; i >= 0; i--) {
        /*
         * ---------------------------------------------------------------------
         * the first three factors
         * ---------------------------------------------------------------------
         */
        for (m = 0; m < 3; m++)
            _rhs[0][m] = rtmp(m, i, j, k) - lhs(3, i, j, k) * _rhs[1][m] -
                         lhs(4, i, j, k) * _rhs[2][m];

        /*
         * ---------------------------------------------------------------------
         * and the remaining two
         * ---------------------------------------------------------------------
         */
        _rhs[0][3] = rtmp(3, i, j, k) - lhsp(3, i, j, k) * _rhs[1][3] -
                     lhsp(4, i, j, k) * _rhs[2][3];
        _rhs[0][4] = rtmp(4, i, j, k) - lhsm(3, i, j, k) * _rhs[1][4] -
                     lhsm(4, i, j, k) * _rhs[2][4];
        if (i + 2 < nx - 1) {
            /*
             * ---------------------------------------------------------------------
             * do the block-diagonal inversion
             * ---------------------------------------------------------------------
             */
            double r1 = _rhs[2][0];
            double r2 = _rhs[2][1];
            double r3 = _rhs[2][2];
            double r4 = _rhs[2][3];
            double r5 = _rhs[2][4];
            double t1 = bt * r3;
            double t2 = 0.5 * (r4 + r5);
            _rhs[2][0] = -r2;
            _rhs[2][1] = r1;
            _rhs[2][2] = bt * (r4 - r5);
            _rhs[2][3] = -t1 + t2;
            _rhs[2][4] = t1 + t2;
        }
        for (m = 0; m < 5; m++) {
            rhs(m, i + 2, j, k) = _rhs[2][m];
            _rhs[2][m] = _rhs[1][m];
            _rhs[1][m] = _rhs[0][m];
        }
    }
    /*
     * ---------------------------------------------------------------------
     * do the block-diagonal inversion
     * ---------------------------------------------------------------------
     */
    double t1 = bt * _rhs[2][2];
    double t2 = 0.5 * (_rhs[2][3] + _rhs[2][4]);
    rhs(0, 1, j, k) = -_rhs[2][1];
    rhs(1, 1, j, k) = _rhs[2][0];
    rhs(2, 1, j, k) = bt * (_rhs[2][3] - _rhs[2][4]);
    rhs(3, 1, j, k) = -t1 + t2;
    rhs(4, 1, j, k) = t1 + t2;
    for (m = 0; m < 5; m++) rhs(m, 0, j, k) = _rhs[1][m];

#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp
}

__global__ void y_solve_gpu_kernel(const double* rho_i, const double* vs,
                                   const double* speed, double* rhs,
                                   double* lhs, double* rhstmp, const int nx,
                                   const int ny, const int nz) {
#define lhs(m, i, j, k) \
    lhs[(i - 1) + (nx - 2) * ((k - 1) + (nz - 2) * ((j) + ny * (m - 3)))]
#define lhsp(m, i, j, k) \
    lhs[(i - 1) + (nx - 2) * ((k - 1) + (nz - 2) * ((j) + ny * (m + 4)))]
#define lhsm(m, i, j, k) \
    lhs[(i - 1) + (nx - 2) * ((k - 1) + (nz - 2) * ((j) + ny * (m - 3 + 2)))]
#define rtmp(m, i, j, k) rhstmp[(i) + nx * ((k) + nz * ((j) + ny * (m)))]
    int i, j, k, m;
    double rhoq[3], cv[3], _lhs[3][5], _lhsp[3][5], _rhs[3][5], fac1;

    /* coalesced */
    i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    k = blockIdx.y * blockDim.y + threadIdx.y + 1;

    /* uncoalesced */
    /* k=blockIdx.x*blockDim.x+threadIdx.x+1; */
    /* i=blockIdx.y*blockDim.y+threadIdx.y+1; */

    if ((k >= (nz - 1)) || (i >= (nx - 1))) return;

    using namespace constants_device;
    /*
     * ---------------------------------------------------------------------
     * computes the left hand side for the three y-factors
     * ---------------------------------------------------------------------
     * first fill the lhs for the u-eigenvalue
     * ---------------------------------------------------------------------
     */
    _lhs[0][0] = lhsp(0, i, 0, k) = 0.0;
    _lhs[0][1] = lhsp(1, i, 0, k) = 0.0;
    _lhs[0][2] = lhsp(2, i, 0, k) = 1.0;
    _lhs[0][3] = lhsp(3, i, 0, k) = 0.0;
    _lhs[0][4] = lhsp(4, i, 0, k) = 0.0;
    for (j = 0; j < 3; j++) {
        fac1 = c3c4 * rho_i(i, j, k);
        rhoq[j] = max(
            max(max(dy3 + con43 * fac1, dy5 + c1c5 * fac1), dymax + fac1), dy1);
        cv[j] = vs(i, j, k);
    }
    _lhs[1][0] = 0.0;
    _lhs[1][1] = -dtty2 * cv[0] - dtty1 * rhoq[0];
    _lhs[1][2] = 1.0 + c2dtty1 * rhoq[1];
    _lhs[1][3] = dtty2 * cv[2] - dtty1 * rhoq[2];
    _lhs[1][4] = 0.0;
    _lhs[1][2] += comz5;
    _lhs[1][3] -= comz4;
    _lhs[1][4] += comz1;
    for (m = 0; m < 5; m++) lhsp(m, i, 1, k) = _lhs[1][m];

    rhoq[0] = rhoq[1];
    rhoq[1] = rhoq[2];
    cv[0] = cv[1];
    cv[1] = cv[2];
    for (m = 0; m < 3; m++) {
        _rhs[0][m] = rhs(m, i, 0, k);
        _rhs[1][m] = rhs(m, i, 1, k);
    }
    /*
     * ---------------------------------------------------------------------
     * FORWARD ELIMINATION
     * ---------------------------------------------------------------------
     */
    for (j = 0; j < ny - 2; j++) {
        /*
         * ---------------------------------------------------------------------
         * first fill the lhs for the u-eigenvalue
         * ---------------------------------------------------------------------
         */
        if ((j + 2) == (ny - 1)) {
            _lhs[2][0] = lhsp(0, i, j + 2, k) = 0.0;
            _lhs[2][1] = lhsp(1, i, j + 2, k) = 0.0;
            _lhs[2][2] = lhsp(2, i, j + 2, k) = 1.0;
            _lhs[2][3] = lhsp(3, i, j + 2, k) = 0.0;
            _lhs[2][4] = lhsp(4, i, j + 2, k) = 0.0;
        } else {
            fac1 = c3c4 * rho_i(i, j + 3, k);
            rhoq[2] = max(
                max(max(dy3 + con43 * fac1, dy5 + c1c5 * fac1), dymax + fac1),
                dy1);
            cv[2] = vs(i, j + 3, k);
            _lhs[2][0] = 0.0;
            _lhs[2][1] = -dtty2 * cv[0] - dtty1 * rhoq[0];
            _lhs[2][2] = 1.0 + c2dtty1 * rhoq[1];
            _lhs[2][3] = dtty2 * cv[2] - dtty1 * rhoq[2];
            _lhs[2][4] = 0.0;
            /*
             * ---------------------------------------------------------------------
             * add fourth order dissipation
             * ---------------------------------------------------------------------
             */
            if ((j + 2) == (2)) {
                _lhs[2][1] -= comz4;
                _lhs[2][2] += comz6;
                _lhs[2][3] -= comz4;
                _lhs[2][4] += comz1;
            } else if (((j + 2) >= (3)) && ((j + 2) < (ny - 3))) {
                _lhs[2][0] += comz1;
                _lhs[2][1] -= comz4;
                _lhs[2][2] += comz6;
                _lhs[2][3] -= comz4;
                _lhs[2][4] += comz1;
            } else if ((j + 2) == (ny - 3)) {
                _lhs[2][0] += comz1;
                _lhs[2][1] -= comz4;
                _lhs[2][2] += comz6;
                _lhs[2][3] -= comz4;
            } else if ((j + 2) == (ny - 2)) {
                _lhs[2][0] += comz1;
                _lhs[2][1] -= comz4;
                _lhs[2][2] += comz5;
            }
            /*
             * ---------------------------------------------------------------------
             * store computed lhs for later reuse
             * ---------------------------------------------------------------------
             */
            for (m = 0; m < 5; m++) lhsp(m, i, j + 2, k) = _lhs[2][m];

            rhoq[0] = rhoq[1];
            rhoq[1] = rhoq[2];
            cv[0] = cv[1];
            cv[1] = cv[2];
        }
        /*
         * ---------------------------------------------------------------------
         * load rhs values for current iteration
         * ---------------------------------------------------------------------
         */
        for (m = 0; m < 3; m++) _rhs[2][m] = rhs(m, i, j + 2, k);

        /*
         * ---------------------------------------------------------------------
         * perform current iteration
         * ---------------------------------------------------------------------
         */
        fac1 = 1.0 / _lhs[0][2];
        _lhs[0][3] *= fac1;
        _lhs[0][4] *= fac1;
        for (m = 0; m < 3; m++) _rhs[0][m] *= fac1;

        _lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
        _lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
        for (m = 0; m < 3; m++) _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];

        _lhs[2][1] -= _lhs[2][0] * _lhs[0][3];
        _lhs[2][2] -= _lhs[2][0] * _lhs[0][4];
        for (m = 0; m < 3; m++) _rhs[2][m] -= _lhs[2][0] * _rhs[0][m];

        /*
         * ---------------------------------------------------------------------
         * store computed lhs and prepare data for next iteration
         * rhs is stored in a temp array such that write accesses are coalesced
         * ---------------------------------------------------------------------
         */
        lhs(3, i, j, k) = _lhs[0][3];
        lhs(4, i, j, k) = _lhs[0][4];
        for (m = 0; m < 5; m++) {
            _lhs[0][m] = _lhs[1][m];
            _lhs[1][m] = _lhs[2][m];
        }
        for (m = 0; m < 3; m++) {
            rtmp(m, i, j, k) = _rhs[0][m];
            _rhs[0][m] = _rhs[1][m];
            _rhs[1][m] = _rhs[2][m];
        }
    }
    /*
     * ---------------------------------------------------------------------
     * the last two rows in this zone are a bit different,
     * since they do not have two more rows available for the
     * elimination of off-diagonal entries
     * ---------------------------------------------------------------------
     */
    j = ny - 2;
    fac1 = 1.0 / _lhs[0][2];
    _lhs[0][3] *= fac1;
    _lhs[0][4] *= fac1;
    for (m = 0; m < 3; m++) _rhs[0][m] *= fac1;

    _lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
    _lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
    for (m = 0; m < 3; m++) _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];

    /*
     * ---------------------------------------------------------------------
     * scale the last row immediately
     * ---------------------------------------------------------------------
     */
    fac1 = 1.0 / _lhs[1][2];
    for (m = 0; m < 3; m++) _rhs[1][m] *= fac1;

    lhs(3, i, ny - 2, k) = _lhs[0][3];
    lhs(4, i, ny - 2, k) = _lhs[0][4];
    /*
     * ---------------------------------------------------------------------
     * do the u+c and the u-c factors
     * ---------------------------------------------------------------------
     */
    for (j = 0; j < 3; j++) cv[j] = speed(i, j, k);

    for (m = 0; m < 5; m++) {
        _lhsp[0][m] = _lhs[0][m] = lhsp(m, i, 0, k);
        _lhsp[1][m] = _lhs[1][m] = lhsp(m, i, 1, k);
    }
    _lhsp[1][1] -= dtty2 * cv[0];
    _lhsp[1][3] += dtty2 * cv[2];
    _lhs[1][1] += dtty2 * cv[0];
    _lhs[1][3] -= dtty2 * cv[2];
    cv[0] = cv[1];
    cv[1] = cv[2];
    _rhs[0][3] = rhs(3, i, 0, k);
    _rhs[0][4] = rhs(4, i, 0, k);
    _rhs[1][3] = rhs(3, i, 1, k);
    _rhs[1][4] = rhs(4, i, 1, k);
    for (j = 0; j < ny - 2; j++) {
        for (m = 0; m < 5; m++) _lhsp[2][m] = _lhs[2][m] = lhsp(m, i, j + 2, k);

        _rhs[2][3] = rhs(3, i, j + 2, k);
        _rhs[2][4] = rhs(4, i, j + 2, k);
        if ((j + 2) < (ny - 1)) {
            cv[2] = speed(i, j + 3, k);
            _lhsp[2][1] -= dtty2 * cv[0];
            _lhsp[2][3] += dtty2 * cv[2];
            _lhs[2][1] += dtty2 * cv[0];
            _lhs[2][3] -= dtty2 * cv[2];
            cv[0] = cv[1];
            cv[1] = cv[2];
        }
        fac1 = 1.0 / _lhsp[0][2];
        m = 3;
        _lhsp[0][3] *= fac1;
        _lhsp[0][4] *= fac1;
        _rhs[0][m] *= fac1;
        _lhsp[1][2] -= _lhsp[1][1] * _lhsp[0][3];
        _lhsp[1][3] -= _lhsp[1][1] * _lhsp[0][4];
        _rhs[1][m] -= _lhsp[1][1] * _rhs[0][m];
        _lhsp[2][1] -= _lhsp[2][0] * _lhsp[0][3];
        _lhsp[2][2] -= _lhsp[2][0] * _lhsp[0][4];
        _rhs[2][m] -= _lhsp[2][0] * _rhs[0][m];
        m = 4;
        fac1 = 1.0 / _lhs[0][2];
        _lhs[0][3] *= fac1;
        _lhs[0][4] *= fac1;
        _rhs[0][m] *= fac1;
        _lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
        _lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
        _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
        _lhs[2][1] -= _lhs[2][0] * _lhs[0][3];
        _lhs[2][2] -= _lhs[2][0] * _lhs[0][4];
        _rhs[2][m] -= _lhs[2][0] * _rhs[0][m];
        /*
         * ---------------------------------------------------------------------
         * store computed lhs and prepare data for next iteration
         * rhs is stored in a temp array such that write accesses are coalesced
         * ---------------------------------------------------------------------
         */
        for (m = 3; m < 5; m++) {
            lhsp(m, i, j, k) = _lhsp[0][m];
            lhsm(m, i, j, k) = _lhs[0][m];
            rtmp(m, i, j, k) = _rhs[0][m];
            _rhs[0][m] = _rhs[1][m];
            _rhs[1][m] = _rhs[2][m];
        }
        for (m = 0; m < 5; m++) {
            _lhsp[0][m] = _lhsp[1][m];
            _lhsp[1][m] = _lhsp[2][m];
            _lhs[0][m] = _lhs[1][m];
            _lhs[1][m] = _lhs[2][m];
        }
    }
    /*
     * ---------------------------------------------------------------------
     * and again the last two rows separately
     * ---------------------------------------------------------------------
     */
    j = ny - 2;
    m = 3;
    fac1 = 1.0 / _lhsp[0][2];
    _lhsp[0][3] *= fac1;
    _lhsp[0][4] *= fac1;
    _rhs[0][m] *= fac1;
    _lhsp[1][2] -= _lhsp[1][1] * _lhsp[0][3];
    _lhsp[1][3] -= _lhsp[1][1] * _lhsp[0][4];
    _rhs[1][m] -= _lhsp[1][1] * _rhs[0][m];
    m = 4;
    fac1 = 1.0 / _lhs[0][2];
    _lhs[0][3] *= fac1;
    _lhs[0][4] *= fac1;
    _rhs[0][m] *= fac1;
    _lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
    _lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
    _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
    /*
     * ---------------------------------------------------------------------
     * scale the last row immediately
     * ---------------------------------------------------------------------
     */
    _rhs[1][3] /= _lhsp[1][2];
    _rhs[1][4] /= _lhs[1][2];
    /*
     * ---------------------------------------------------------------------
     * BACKSUBSTITUTION
     * ---------------------------------------------------------------------
     */
    for (m = 0; m < 3; m++) _rhs[0][m] -= lhs(3, i, ny - 2, k) * _rhs[1][m];

    _rhs[0][3] -= _lhsp[0][3] * _rhs[1][3];
    _rhs[0][4] -= _lhs[0][3] * _rhs[1][4];
    for (m = 0; m < 5; m++) {
        _rhs[2][m] = _rhs[1][m];
        _rhs[1][m] = _rhs[0][m];
    }
    for (j = ny - 3; j >= 0; j--) {
        /*
         * ---------------------------------------------------------------------
         * the first three factors
         * ---------------------------------------------------------------------
         */
        for (m = 0; m < 3; m++)
            _rhs[0][m] = rtmp(m, i, j, k) - lhs(3, i, j, k) * _rhs[1][m] -
                         lhs(4, i, j, k) * _rhs[2][m];

        /*
         * ---------------------------------------------------------------------
         * and the remaining two
         * ---------------------------------------------------------------------
         */
        _rhs[0][3] = rtmp(3, i, j, k) - lhsp(3, i, j, k) * _rhs[1][3] -
                     lhsp(4, i, j, k) * _rhs[2][3];
        _rhs[0][4] = rtmp(4, i, j, k) - lhsm(3, i, j, k) * _rhs[1][4] -
                     lhsm(4, i, j, k) * _rhs[2][4];
        if ((j + 2) < (ny - 1)) {
            /*
             * ---------------------------------------------------------------------
             * do the block-diagonal inversion
             * ---------------------------------------------------------------------
             */
            double r1 = _rhs[2][0];
            double r2 = _rhs[2][1];
            double r3 = _rhs[2][2];
            double r4 = _rhs[2][3];
            double r5 = _rhs[2][4];
            double t1 = bt * r1;
            double t2 = 0.5 * (r4 + r5);
            _rhs[2][0] = bt * (r4 - r5);
            _rhs[2][1] = -r3;
            _rhs[2][2] = r2;
            _rhs[2][3] = -t1 + t2;
            _rhs[2][4] = t1 + t2;
        }
        for (m = 0; m < 5; m++) {
            rhs(m, i, j + 2, k) = _rhs[2][m];
            _rhs[2][m] = _rhs[1][m];
            _rhs[1][m] = _rhs[0][m];
        }
    }
    /*
     * ---------------------------------------------------------------------
     * do the block-diagonal inversion
     * ---------------------------------------------------------------------
     */
    double t1 = bt * _rhs[2][0];
    double t2 = 0.5 * (_rhs[2][3] + _rhs[2][4]);
    rhs(0, i, 1, k) = bt * (_rhs[2][3] - _rhs[2][4]);
    rhs(1, i, 1, k) = -_rhs[2][2];
    rhs(2, i, 1, k) = _rhs[2][1];
    rhs(3, i, 1, k) = -t1 + t2;
    rhs(4, i, 1, k) = t1 + t2;
    for (m = 0; m < 5; m++) rhs(m, i, 0, k) = _rhs[1][m];

#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp
}

__global__ void z_solve_gpu_kernel(const double* rho_i, const double* us,
                                   const double* vs, const double* ws,
                                   const double* speed, const double* qs,
                                   const double* u, double* rhs, double* lhs,
                                   double* rhstmp, const int nx, const int ny,
                                   const int nz) {
#define lhs(m, i, j, k) \
    lhs[(i - 1) + (nx - 2) * ((j - 1) + (ny - 2) * ((k) + nz * (m - 3)))]
#define lhsp(m, i, j, k) \
    lhs[(i - 1) + (nx - 2) * ((j - 1) + (ny - 2) * ((k) + nz * (m + 4)))]
#define lhsm(m, i, j, k) \
    lhs[(i - 1) + (nx - 2) * ((j - 1) + (ny - 2) * ((k) + nz * (m - 3 + 2)))]
#define rtmp(m, i, j, k) rhstmp[(i) + nx * ((j) + ny * ((k) + nz * (m)))]
    int i, j, k, m;
    double rhos[3], cv[3], _lhs[3][5], _lhsp[3][5], _rhs[3][5], fac1;

    /* coalesced */
    i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    /* uncoalesced */
    /* j=blockIdx.x*blockDim.x+threadIdx.x+1; */
    /* i=blockIdx.y*blockDim.y+threadIdx.y+1; */

    if ((j >= (ny - 1)) || (i >= (nx - 1))) {
        return;
    }

    using namespace constants_device;
    /*
     * ---------------------------------------------------------------------
     * computes the left hand side for the three z-factors
     * ---------------------------------------------------------------------
     * first fill the lhs for the u-eigenvalue
     * ---------------------------------------------------------------------
     */
    _lhs[0][0] = lhsp(0, i, j, 0) = 0.0;
    _lhs[0][1] = lhsp(1, i, j, 0) = 0.0;
    _lhs[0][2] = lhsp(2, i, j, 0) = 1.0;
    _lhs[0][3] = lhsp(3, i, j, 0) = 0.0;
    _lhs[0][4] = lhsp(4, i, j, 0) = 0.0;
    for (k = 0; k < 3; k++) {
        fac1 = c3c4 * rho_i(i, j, k);
        rhos[k] = max(
            max(max(dz4 + con43 * fac1, dz5 + c1c5 * fac1), dzmax + fac1), dz1);
        cv[k] = ws(i, j, k);
    }
    _lhs[1][0] = 0.0;
    _lhs[1][1] = -dttz2 * cv[0] - dttz1 * rhos[0];
    _lhs[1][2] = 1.0 + c2dttz1 * rhos[1];
    _lhs[1][3] = dttz2 * cv[2] - dttz1 * rhos[2];
    _lhs[1][4] = 0.0;
    _lhs[1][2] += comz5;
    _lhs[1][3] -= comz4;
    _lhs[1][4] += comz1;
    for (m = 0; m < 5; m++) lhsp(m, i, j, 1) = _lhs[1][m];

    rhos[0] = rhos[1];
    rhos[1] = rhos[2];
    cv[0] = cv[1];
    cv[1] = cv[2];
    for (m = 0; m < 3; m++) {
        _rhs[0][m] = rhs(m, i, j, 0);
        _rhs[1][m] = rhs(m, i, j, 1);
    }
    /*
     * ---------------------------------------------------------------------
     * FORWARD ELIMINATION
     * ---------------------------------------------------------------------
     */
    for (k = 0; k < nz - 2; k++) {
        /*
         * ---------------------------------------------------------------------
         * first fill the lhs for the u-eigenvalue
         * ---------------------------------------------------------------------
         */
        if ((k + 2) == (nz - 1)) {
            _lhs[2][0] = lhsp(0, i, j, k + 2) = 0.0;
            _lhs[2][1] = lhsp(1, i, j, k + 2) = 0.0;
            _lhs[2][2] = lhsp(2, i, j, k + 2) = 1.0;
            _lhs[2][3] = lhsp(3, i, j, k + 2) = 0.0;
            _lhs[2][4] = lhsp(4, i, j, k + 2) = 0.0;
        } else {
            fac1 = c3c4 * rho_i(i, j, k + 3);
            rhos[2] = max(
                max(max(dz4 + con43 * fac1, dz5 + c1c5 * fac1), dzmax + fac1),
                dz1);
            cv[2] = ws(i, j, k + 3);
            _lhs[2][0] = 0.0;
            _lhs[2][1] = -dttz2 * cv[0] - dttz1 * rhos[0];
            _lhs[2][2] = 1.0 + c2dttz1 * rhos[1];
            _lhs[2][3] = dttz2 * cv[2] - dttz1 * rhos[2];
            _lhs[2][4] = 0.0;
            /*
             * ---------------------------------------------------------------------
             * add fourth order dissipation
             * ---------------------------------------------------------------------
             */
            if ((k + 2) == (2)) {
                _lhs[2][1] -= comz4;
                _lhs[2][2] += comz6;
                _lhs[2][3] -= comz4;
                _lhs[2][4] += comz1;
            } else if (((k + 2) >= (3)) && ((k + 2) < (nz - 3))) {
                _lhs[2][0] += comz1;
                _lhs[2][1] -= comz4;
                _lhs[2][2] += comz6;
                _lhs[2][3] -= comz4;
                _lhs[2][4] += comz1;
            } else if ((k + 2) == (nz - 3)) {
                _lhs[2][0] += comz1;
                _lhs[2][1] -= comz4;
                _lhs[2][2] += comz6;
                _lhs[2][3] -= comz4;
            } else if ((k + 2) == (nz - 2)) {
                _lhs[2][0] += comz1;
                _lhs[2][1] -= comz4;
                _lhs[2][2] += comz5;
            }
            /*
             * ---------------------------------------------------------------------
             * store computed lhs for later reuse
             * ---------------------------------------------------------------------
             */
            for (m = 0; m < 5; m++) lhsp(m, i, j, k + 2) = _lhs[2][m];

            rhos[0] = rhos[1];
            rhos[1] = rhos[2];
            cv[0] = cv[1];
            cv[1] = cv[2];
        }
        /*
         * ---------------------------------------------------------------------
         * load rhs values for current iteration
         * ---------------------------------------------------------------------
         */
        for (m = 0; m < 3; m++) _rhs[2][m] = rhs(m, i, j, k + 2);

        /*
         * ---------------------------------------------------------------------
         * perform current iteration
         * ---------------------------------------------------------------------
         */
        fac1 = 1.0 / _lhs[0][2];
        _lhs[0][3] *= fac1;
        _lhs[0][4] *= fac1;
        for (m = 0; m < 3; m++) _rhs[0][m] *= fac1;

        _lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
        _lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
        for (m = 0; m < 3; m++) _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];

        _lhs[2][1] -= _lhs[2][0] * _lhs[0][3];
        _lhs[2][2] -= _lhs[2][0] * _lhs[0][4];
        for (m = 0; m < 3; m++) _rhs[2][m] -= _lhs[2][0] * _rhs[0][m];

        /*
         * ---------------------------------------------------------------------
         * store computed lhs and prepare data for next iteration
         * rhs is stored in a temp array such that write accesses are coalesced
         * ---------------------------------------------------------------------
         */
        lhs(3, i, j, k) = _lhs[0][3];
        lhs(4, i, j, k) = _lhs[0][4];
        for (m = 0; m < 5; m++) {
            _lhs[0][m] = _lhs[1][m];
            _lhs[1][m] = _lhs[2][m];
        }
        for (m = 0; m < 3; m++) {
            rtmp(m, i, j, k) = _rhs[0][m];
            _rhs[0][m] = _rhs[1][m];
            _rhs[1][m] = _rhs[2][m];
        }
    }
    /*
     * ---------------------------------------------------------------------
     * the last two rows in this zone are a bit different,
     * since they do not have two more rows available for the
     * elimination of off-diagonal entries
     * ---------------------------------------------------------------------
     */
    k = nz - 2;
    fac1 = 1.0 / _lhs[0][2];
    _lhs[0][3] *= fac1;
    _lhs[0][4] *= fac1;
    for (m = 0; m < 3; m++) _rhs[0][m] *= fac1;

    _lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
    _lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
    for (m = 0; m < 3; m++) _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];

    /*
     * ---------------------------------------------------------------------
     * scale the last row immediately
     * ---------------------------------------------------------------------
     */
    fac1 = 1.0 / _lhs[1][2];
    for (m = 0; m < 3; m++) _rhs[1][m] *= fac1;

    lhs(3, i, j, k) = _lhs[0][3];
    lhs(4, i, j, k) = _lhs[0][4];
    /*
     * ---------------------------------------------------------------------
     * subsequently, fill the other factors (u+c), (u-c)
     * ---------------------------------------------------------------------
     */
    for (k = 0; k < 3; k++) cv[k] = speed(i, j, k);

    for (m = 0; m < 5; m++) {
        _lhsp[0][m] = _lhs[0][m] = lhsp(m, i, j, 0);
        _lhsp[1][m] = _lhs[1][m] = lhsp(m, i, j, 1);
    }
    _lhsp[1][1] -= dttz2 * cv[0];
    _lhsp[1][3] += dttz2 * cv[2];
    _lhs[1][1] += dttz2 * cv[0];
    _lhs[1][3] -= dttz2 * cv[2];
    cv[0] = cv[1];
    cv[1] = cv[2];
    _rhs[0][3] = rhs(3, i, j, 0);
    _rhs[0][4] = rhs(4, i, j, 0);
    _rhs[1][3] = rhs(3, i, j, 1);
    _rhs[1][4] = rhs(4, i, j, 1);
    /*
     * ---------------------------------------------------------------------
     * do the u+c and the u-c factors
     * ---------------------------------------------------------------------
     */
    for (k = 0; k < nz - 2; k++) {
        /*
         * first, fill the other factors (u+c), (u-c)
         * ---------------------------------------------------------------------
         */
        for (m = 0; m < 5; m++) _lhsp[2][m] = _lhs[2][m] = lhsp(m, i, j, k + 2);

        _rhs[2][3] = rhs(3, i, j, k + 2);
        _rhs[2][4] = rhs(4, i, j, k + 2);
        if ((k + 2) < (nz - 1)) {
            cv[2] = speed(i, j, k + 3);
            _lhsp[2][1] -= dttz2 * cv[0];
            _lhsp[2][3] += dttz2 * cv[2];
            _lhs[2][1] += dttz2 * cv[0];
            _lhs[2][3] -= dttz2 * cv[2];
            cv[0] = cv[1];
            cv[1] = cv[2];
        }
        m = 3;
        fac1 = 1.0 / _lhsp[0][2];
        _lhsp[0][3] *= fac1;
        _lhsp[0][4] *= fac1;
        _rhs[0][m] *= fac1;
        _lhsp[1][2] -= _lhsp[1][1] * _lhsp[0][3];
        _lhsp[1][3] -= _lhsp[1][1] * _lhsp[0][4];
        _rhs[1][m] -= _lhsp[1][1] * _rhs[0][m];
        _lhsp[2][1] -= _lhsp[2][0] * _lhsp[0][3];
        _lhsp[2][2] -= _lhsp[2][0] * _lhsp[0][4];
        _rhs[2][m] -= _lhsp[2][0] * _rhs[0][m];
        m = 4;
        fac1 = 1.0 / _lhs[0][2];
        _lhs[0][3] *= fac1;
        _lhs[0][4] *= fac1;
        _rhs[0][m] *= fac1;
        _lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
        _lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
        _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
        _lhs[2][1] -= _lhs[2][0] * _lhs[0][3];
        _lhs[2][2] -= _lhs[2][0] * _lhs[0][4];
        _rhs[2][m] -= _lhs[2][0] * _rhs[0][m];
        /*
         * ---------------------------------------------------------------------
         * store computed lhs and prepare data for next iteration
         * rhs is stored in a temp array such that write accesses are coalesced
         * ---------------------------------------------------------------------
         */
        for (m = 3; m < 5; m++) {
            lhsp(m, i, j, k) = _lhsp[0][m];
            lhsm(m, i, j, k) = _lhs[0][m];
            rtmp(m, i, j, k) = _rhs[0][m];
            _rhs[0][m] = _rhs[1][m];
            _rhs[1][m] = _rhs[2][m];
        }
        for (m = 0; m < 5; m++) {
            _lhsp[0][m] = _lhsp[1][m];
            _lhsp[1][m] = _lhsp[2][m];
            _lhs[0][m] = _lhs[1][m];
            _lhs[1][m] = _lhs[2][m];
        }
    }
    /*
     * ---------------------------------------------------------------------
     * and again the last two rows separately
     * ---------------------------------------------------------------------
     */
    k = nz - 2;
    m = 3;
    fac1 = 1.0 / _lhsp[0][2];
    _lhsp[0][3] *= fac1;
    _lhsp[0][4] *= fac1;
    _rhs[0][m] *= fac1;
    _lhsp[1][2] -= _lhsp[1][1] * _lhsp[0][3];
    _lhsp[1][3] -= _lhsp[1][1] * _lhsp[0][4];
    _rhs[1][m] -= _lhsp[1][1] * _rhs[0][m];
    m = 4;
    fac1 = 1.0 / _lhs[0][2];
    _lhs[0][3] *= fac1;
    _lhs[0][4] *= fac1;
    _rhs[0][m] *= fac1;
    _lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
    _lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
    _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
    /*
     * ---------------------------------------------------------------------
     * scale the last row immediately
     * ---------------------------------------------------------------------
     */
    _rhs[1][3] /= _lhsp[1][2];
    _rhs[1][4] /= _lhs[1][2];
    /*
     * ---------------------------------------------------------------------
     * BACKSUBSTITUTION
     * ---------------------------------------------------------------------
     */
    for (m = 0; m < 3; m++) _rhs[0][m] -= lhs(3, i, j, nz - 2) * _rhs[1][m];

    _rhs[0][3] -= _lhsp[0][3] * _rhs[1][3];
    _rhs[0][4] -= _lhs[0][3] * _rhs[1][4];
    for (m = 0; m < 5; m++) {
        _rhs[2][m] = _rhs[1][m];
        _rhs[1][m] = _rhs[0][m];
    }
    for (k = nz - 3; k >= 0; k--) {
        /*
         * ---------------------------------------------------------------------
         * the first three factors
         * ---------------------------------------------------------------------
         */
        for (m = 0; m < 3; m++)
            _rhs[0][m] = rtmp(m, i, j, k) - lhs(3, i, j, k) * _rhs[1][m] -
                         lhs(4, i, j, k) * _rhs[2][m];

        /*
         * ---------------------------------------------------------------------
         * and the remaining two
         * ---------------------------------------------------------------------
         */
        _rhs[0][3] = rtmp(3, i, j, k) - lhsp(3, i, j, k) * _rhs[1][3] -
                     lhsp(4, i, j, k) * _rhs[2][3];
        _rhs[0][4] = rtmp(4, i, j, k) - lhsm(3, i, j, k) * _rhs[1][4] -
                     lhsm(4, i, j, k) * _rhs[2][4];
        if ((k + 2) < (nz - 1)) {
            /*
             * ---------------------------------------------------------------------
             * do the block-diagonal inversion
             * ---------------------------------------------------------------------
             */
            double xvel = us(i, j, k + 2);
            double yvel = vs(i, j, k + 2);
            double zvel = ws(i, j, k + 2);
            double ac = speed(i, j, k + 2);
            double uzik1 = u(0, i, j, k + 2);
            double t1 = (bt * uzik1) / ac * (_rhs[2][3] + _rhs[2][4]);
            double t2 = _rhs[2][2] + t1;
            double t3 = bt * uzik1 * (_rhs[2][3] - _rhs[2][4]);
            _rhs[2][4] = uzik1 * (-xvel * _rhs[2][1] + yvel * _rhs[2][0]) +
                         qs(i, j, k + 2) * t2 + c2iv * (ac * ac) * t1 +
                         zvel * t3;
            _rhs[2][3] = zvel * t2 + t3;
            _rhs[2][2] = uzik1 * _rhs[2][0] + yvel * t2;
            _rhs[2][1] = -uzik1 * _rhs[2][1] + xvel * t2;
            _rhs[2][0] = t2;
        }
        for (m = 0; m < 5; m++) {
            rhs(m, i, j, k + 2) = _rhs[2][m];
            _rhs[2][m] = _rhs[1][m];
            _rhs[1][m] = _rhs[0][m];
        }
    }
    /*
     * ---------------------------------------------------------------------
     * do the block-diagonal inversion
     * ---------------------------------------------------------------------
     */
    double xvel = us(i, j, 1);
    double yvel = vs(i, j, 1);
    double zvel = ws(i, j, 1);
    double ac = speed(i, j, 1);
    double uzik1 = u(0, i, j, 1);
    double t1 = (bt * uzik1) / ac * (_rhs[2][3] + _rhs[2][4]);
    double t2 = _rhs[2][2] + t1;
    double t3 = bt * uzik1 * (_rhs[2][3] - _rhs[2][4]);
    rhs(4, i, j, 1) = uzik1 * (-xvel * _rhs[2][1] + yvel * _rhs[2][0]) +
                      qs(i, j, 1) * t2 + c2iv * (ac * ac) * t1 + zvel * t3;
    rhs(3, i, j, 1) = zvel * t2 + t3;
    rhs(2, i, j, 1) = uzik1 * _rhs[2][0] + yvel * t2;
    rhs(1, i, j, 1) = -uzik1 * _rhs[2][1] + xvel * t2;
    rhs(0, i, j, 1) = t2;
    for (m = 0; m < 5; m++) {
        rhs(m, i, j, 0) = _rhs[1][m];
    }
#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp
}
