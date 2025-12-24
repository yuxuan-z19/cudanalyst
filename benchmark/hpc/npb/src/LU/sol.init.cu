#include "lu.cuh"

__global__ void jacld_blts_gpu_kernel(const int plane, const int klower,
                                      const int jlower, const double* u,
                                      const double* rho_i, const double* qs,
                                      double* v, const int nx, const int ny,
                                      const int nz) {
    int i, j, k, m;
    double tmp1, tmp2, tmp3, tmat[5 * 5], tv[5];
    double r43, c1345, c34;

    k = klower + blockIdx.x + 1;
    j = jlower + threadIdx.x + 1;

    i = plane - k - j + 3;

    if ((j > (ny - 2)) || (i > (nx - 2)) || (i < 1)) {
        return;
    }

    r43 = 4.0 / 3.0;
    c1345 = C1 * C3 * C4 * C5;
    c34 = C3 * C4;
    using namespace constants_device;
    /*
     * ---------------------------------------------------------------------
     * form the first block sub-diagonal
     * ---------------------------------------------------------------------
     */
    tmp1 = rho_i(i, j, k - 1);
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;
    tmat[0 + 5 * 0] = -dt * tz1 * dz1;
    tmat[0 + 5 * 1] = 0.0;
    tmat[0 + 5 * 2] = 0.0;
    tmat[0 + 5 * 3] = -dt * tz2;
    tmat[0 + 5 * 4] = 0.0;
    tmat[1 + 5 * 0] =
        -dt * tz2 * (-(u(1, i, j, k - 1) * u(3, i, j, k - 1)) * tmp2) -
        dt * tz1 * (-c34 * tmp2 * u(1, i, j, k - 1));
    tmat[1 + 5 * 1] = -dt * tz2 * (u(3, i, j, k - 1) * tmp1) -
                      dt * tz1 * c34 * tmp1 - dt * tz1 * dz2;
    tmat[1 + 5 * 2] = 0.0;
    tmat[1 + 5 * 3] = -dt * tz2 * (u(1, i, j, k - 1) * tmp1);
    tmat[1 + 5 * 4] = 0.0;
    tmat[2 + 5 * 0] =
        -dt * tz2 * (-(u(2, i, j, k - 1) * u(3, i, j, k - 1)) * tmp2) -
        dt * tz1 * (-c34 * tmp2 * u(2, i, j, k - 1));
    tmat[2 + 5 * 1] = 0.0;
    tmat[2 + 5 * 2] = -dt * tz2 * (u(3, i, j, k - 1) * tmp1) -
                      dt * tz1 * (c34 * tmp1) - dt * tz1 * dz3;
    tmat[2 + 5 * 3] = -dt * tz2 * (u(2, i, j, k - 1) * tmp1);
    tmat[2 + 5 * 4] = 0.0;
    tmat[3 + 5 * 0] =
        -dt * tz2 *
            (-(u(3, i, j, k - 1) * tmp1) * (u(3, i, j, k - 1) * tmp1) +
             C2 * qs(i, j, k - 1) * tmp1) -
        dt * tz1 * (-r43 * c34 * tmp2 * u(3, i, j, k - 1));
    tmat[3 + 5 * 1] = -dt * tz2 * (-C2 * (u(1, i, j, k - 1) * tmp1));
    tmat[3 + 5 * 2] = -dt * tz2 * (-C2 * (u(2, i, j, k - 1) * tmp1));
    tmat[3 + 5 * 3] = -dt * tz2 * (2.0 - C2) * (u(3, i, j, k - 1) * tmp1) -
                      dt * tz1 * (r43 * c34 * tmp1) - dt * tz1 * dz4;
    tmat[3 + 5 * 4] = -dt * tz2 * C2;
    tmat[4 + 5 * 0] =
        -dt * tz2 *
            ((C2 * 2.0 * qs(i, j, k - 1) - C1 * u(4, i, j, k - 1)) *
             u(3, i, j, k - 1) * tmp2) -
        dt * tz1 *
            (-(c34 - c1345) * tmp3 * (u(1, i, j, k - 1) * u(1, i, j, k - 1)) -
             (c34 - c1345) * tmp3 * (u(2, i, j, k - 1) * u(2, i, j, k - 1)) -
             (r43 * c34 - c1345) * tmp3 *
                 (u(3, i, j, k - 1) * u(3, i, j, k - 1)) -
             c1345 * tmp2 * u(4, i, j, k - 1));
    tmat[4 + 5 * 1] =
        -dt * tz2 * (-C2 * (u(1, i, j, k - 1) * u(3, i, j, k - 1)) * tmp2) -
        dt * tz1 * (c34 - c1345) * tmp2 * u(1, i, j, k - 1);
    tmat[4 + 5 * 2] =
        -dt * tz2 * (-C2 * (u(2, i, j, k - 1) * u(3, i, j, k - 1)) * tmp2) -
        dt * tz1 * (c34 - c1345) * tmp2 * u(2, i, j, k - 1);
    tmat[4 + 5 * 3] =
        -dt * tz2 *
            (C1 * (u(4, i, j, k - 1) * tmp1) -
             C2 * (qs(i, j, k - 1) * tmp1 +
                   u(3, i, j, k - 1) * u(3, i, j, k - 1) * tmp2)) -
        dt * tz1 * (r43 * c34 - c1345) * tmp2 * u(3, i, j, k - 1);
    tmat[4 + 5 * 4] = -dt * tz2 * (C1 * (u(3, i, j, k - 1) * tmp1)) -
                      dt * tz1 * c1345 * tmp1 - dt * tz1 * dz5;
    for (m = 0; m < 5; m++) {
        tv[m] = v(m, i, j, k) - omega * (tmat[m + 5 * 0] * v(0, i, j, k - 1) +
                                         tmat[m + 5 * 1] * v(1, i, j, k - 1) +
                                         tmat[m + 5 * 2] * v(2, i, j, k - 1) +
                                         tmat[m + 5 * 3] * v(3, i, j, k - 1) +
                                         tmat[m + 5 * 4] * v(4, i, j, k - 1));
    }
    /*
     * ---------------------------------------------------------------------
     * form the second block sub-diagonal
     * ---------------------------------------------------------------------
     */
    tmp1 = rho_i(i, j - 1, k);
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;
    tmat[0 + 5 * 0] = -dt * ty1 * dy1;
    tmat[0 + 5 * 1] = 0.0;
    tmat[0 + 5 * 2] = -dt * ty2;
    tmat[0 + 5 * 3] = 0.0;
    tmat[0 + 5 * 4] = 0.0;
    tmat[1 + 5 * 0] =
        -dt * ty2 * (-(u(1, i, j - 1, k) * u(2, i, j - 1, k)) * tmp2) -
        dt * ty1 * (-c34 * tmp2 * u(1, i, j - 1, k));
    tmat[1 + 5 * 1] = -dt * ty2 * (u(2, i, j - 1, k) * tmp1) -
                      dt * ty1 * (c34 * tmp1) - dt * ty1 * dy2;
    tmat[1 + 5 * 2] = -dt * ty2 * (u(1, i, j - 1, k) * tmp1);
    tmat[1 + 5 * 3] = 0.0;
    tmat[1 + 5 * 4] = 0.0;
    tmat[2 + 5 * 0] =
        -dt * ty2 *
            (-(u(2, i, j - 1, k) * tmp1) * (u(2, i, j - 1, k) * tmp1) +
             C2 * (qs(i, j - 1, k) * tmp1)) -
        dt * ty1 * (-r43 * c34 * tmp2 * u(2, i, j - 1, k));
    tmat[2 + 5 * 1] = -dt * ty2 * (-C2 * (u(1, i, j - 1, k) * tmp1));
    tmat[2 + 5 * 2] = -dt * ty2 * ((2.0 - C2) * (u(2, i, j - 1, k) * tmp1)) -
                      dt * ty1 * (r43 * c34 * tmp1) - dt * ty1 * dy3;
    tmat[2 + 5 * 3] = -dt * ty2 * (-C2 * (u(3, i, j - 1, k) * tmp1));
    tmat[2 + 5 * 4] = -dt * ty2 * C2;
    tmat[3 + 5 * 0] =
        -dt * ty2 * (-(u(2, i, j - 1, k) * u(3, i, j - 1, k)) * tmp2) -
        dt * ty1 * (-c34 * tmp2 * u(3, i, j - 1, k));
    tmat[3 + 5 * 1] = 0.0;
    tmat[3 + 5 * 2] = -dt * ty2 * (u(3, i, j - 1, k) * tmp1);
    tmat[3 + 5 * 3] = -dt * ty2 * (u(2, i, j - 1, k) * tmp1) -
                      dt * ty1 * (c34 * tmp1) - dt * ty1 * dy4;
    tmat[3 + 5 * 4] = 0.0;
    tmat[4 + 5 * 0] =
        -dt * ty2 *
            ((C2 * 2.0 * qs(i, j - 1, k) - C1 * u(4, i, j - 1, k)) *
             (u(2, i, j - 1, k) * tmp2)) -
        dt * ty1 *
            (-(c34 - c1345) * tmp3 * (u(1, i, j - 1, k) * u(1, i, j - 1, k)) -
             (r43 * c34 - c1345) * tmp3 *
                 (u(2, i, j - 1, k) * u(2, i, j - 1, k)) -
             (c34 - c1345) * tmp3 * (u(3, i, j - 1, k) * u(3, i, j - 1, k)) -
             c1345 * tmp2 * u(4, i, j - 1, k));
    tmat[4 + 5 * 1] =
        -dt * ty2 * (-C2 * (u(1, i, j - 1, k) * u(2, i, j - 1, k)) * tmp2) -
        dt * ty1 * (c34 - c1345) * tmp2 * u(1, i, j - 1, k);
    tmat[4 + 5 * 2] =
        -dt * ty2 *
            (C1 * (u(4, i, j - 1, k) * tmp1) -
             C2 * (qs(i, j - 1, k) * tmp1 +
                   u(2, i, j - 1, k) * u(2, i, j - 1, k) * tmp2)) -
        dt * ty1 * (r43 * c34 - c1345) * tmp2 * u(2, i, j - 1, k);
    tmat[4 + 5 * 3] =
        -dt * ty2 * (-C2 * (u(2, i, j - 1, k) * u(3, i, j - 1, k)) * tmp2) -
        dt * ty1 * (c34 - c1345) * tmp2 * u(3, i, j - 1, k);
    tmat[4 + 5 * 4] = -dt * ty2 * (C1 * (u(2, i, j - 1, k) * tmp1)) -
                      dt * ty1 * c1345 * tmp1 - dt * ty1 * dy5;
    for (m = 0; m < 5; m++) {
        tv[m] = tv[m] - omega * (tmat[m + 5 * 0] * v(0, i, j - 1, k) +
                                 tmat[m + 5 * 1] * v(1, i, j - 1, k) +
                                 tmat[m + 5 * 2] * v(2, i, j - 1, k) +
                                 tmat[m + 5 * 3] * v(3, i, j - 1, k) +
                                 tmat[m + 5 * 4] * v(4, i, j - 1, k));
    }
    /*
     * ---------------------------------------------------------------------
     * form the third block sub-diagonal
     * ---------------------------------------------------------------------
     */
    tmp1 = rho_i(i - 1, j, k);
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;
    tmat[0 + 5 * 0] = -dt * tx1 * dx1;
    tmat[0 + 5 * 1] = -dt * tx2;
    tmat[0 + 5 * 2] = 0.0;
    tmat[0 + 5 * 3] = 0.0;
    tmat[0 + 5 * 4] = 0.0;
    tmat[1 + 5 * 0] =
        -dt * tx2 *
            (-(u(1, i - 1, j, k) * tmp1) * (u(1, i - 1, j, k) * tmp1) +
             C2 * qs(i - 1, j, k) * tmp1) -
        dt * tx1 * (-r43 * c34 * tmp2 * u(1, i - 1, j, k));
    tmat[1 + 5 * 1] = -dt * tx2 * ((2.0 - C2) * (u(1, i - 1, j, k) * tmp1)) -
                      dt * tx1 * (r43 * c34 * tmp1) - dt * tx1 * dx2;
    tmat[1 + 5 * 2] = -dt * tx2 * (-C2 * (u(2, i - 1, j, k) * tmp1));
    tmat[1 + 5 * 3] = -dt * tx2 * (-C2 * (u(3, i - 1, j, k) * tmp1));
    tmat[1 + 5 * 4] = -dt * tx2 * C2;
    tmat[2 + 5 * 0] =
        -dt * tx2 * (-(u(1, i - 1, j, k) * u(2, i - 1, j, k)) * tmp2) -
        dt * tx1 * (-c34 * tmp2 * u(2, i - 1, j, k));
    tmat[2 + 5 * 1] = -dt * tx2 * (u(2, i - 1, j, k) * tmp1);
    tmat[2 + 5 * 2] = -dt * tx2 * (u(1, i - 1, j, k) * tmp1) -
                      dt * tx1 * (c34 * tmp1) - dt * tx1 * dx3;
    tmat[2 + 5 * 3] = 0.0;
    tmat[2 + 5 * 4] = 0.0;
    tmat[3 + 5 * 0] =
        -dt * tx2 * (-(u(1, i - 1, j, k) * u(3, i - 1, j, k)) * tmp2) -
        dt * tx1 * (-c34 * tmp2 * u(3, i - 1, j, k));
    tmat[3 + 5 * 1] = -dt * tx2 * (u(3, i - 1, j, k) * tmp1);
    tmat[3 + 5 * 2] = 0.0;
    tmat[3 + 5 * 3] = -dt * tx2 * (u(1, i - 1, j, k) * tmp1) -
                      dt * tx1 * (c34 * tmp1) - dt * tx1 * dx4;
    tmat[3 + 5 * 4] = 0.0;
    tmat[4 + 5 * 0] =
        -dt * tx2 *
            ((C2 * 2.0 * qs(i - 1, j, k) - C1 * u(4, i - 1, j, k)) *
             u(1, i - 1, j, k) * tmp2) -
        dt * tx1 *
            (-(r43 * c34 - c1345) * tmp3 *
                 (u(1, i - 1, j, k) * u(1, i - 1, j, k)) -
             (c34 - c1345) * tmp3 * (u(2, i - 1, j, k) * u(2, i - 1, j, k)) -
             (c34 - c1345) * tmp3 * (u(3, i - 1, j, k) * u(3, i - 1, j, k)) -
             c1345 * tmp2 * u(4, i - 1, j, k));
    tmat[4 + 5 * 1] = -dt * tx2 *
                          (C1 * (u(4, i - 1, j, k) * tmp1) -
                           C2 * (u(1, i - 1, j, k) * u(1, i - 1, j, k) * tmp2 +
                                 qs(i - 1, j, k) * tmp1)) -
                      dt * tx1 * (r43 * c34 - c1345) * tmp2 * u(1, i - 1, j, k);
    tmat[4 + 5 * 2] =
        -dt * tx2 * (-C2 * (u(2, i - 1, j, k) * u(1, i - 1, j, k)) * tmp2) -
        dt * tx1 * (c34 - c1345) * tmp2 * u(2, i - 1, j, k);
    tmat[4 + 5 * 3] =
        -dt * tx2 * (-C2 * (u(3, i - 1, j, k) * u(1, i - 1, j, k)) * tmp2) -
        dt * tx1 * (c34 - c1345) * tmp2 * u(3, i - 1, j, k);
    tmat[4 + 5 * 4] = -dt * tx2 * (C1 * (u(1, i - 1, j, k) * tmp1)) -
                      dt * tx1 * c1345 * tmp1 - dt * tx1 * dx5;
    for (m = 0; m < 5; m++) {
        tv[m] = tv[m] - omega * (tmat[m + 0 * 5] * v(0, i - 1, j, k) +
                                 tmat[m + 5 * 1] * v(1, i - 1, j, k) +
                                 tmat[m + 5 * 2] * v(2, i - 1, j, k) +
                                 tmat[m + 5 * 3] * v(3, i - 1, j, k) +
                                 tmat[m + 5 * 4] * v(4, i - 1, j, k));
    }
    /*
     * ---------------------------------------------------------------------
     * form the block diagonal
     * ---------------------------------------------------------------------
     */
    tmp1 = rho_i(i, j, k);
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;
    tmat[0 + 5 * 0] = 1.0 + dt * 2.0 * (tx1 * dx1 + ty1 * dy1 + tz1 * dz1);
    tmat[0 + 5 * 1] = 0.0;
    tmat[0 + 5 * 2] = 0.0;
    tmat[0 + 5 * 3] = 0.0;
    tmat[0 + 5 * 4] = 0.0;
    tmat[1 + 5 * 0] =
        -dt * 2.0 * (tx1 * r43 + ty1 + tz1) * c34 * tmp2 * u(1, i, j, k);
    tmat[1 + 5 * 1] = 1.0 + dt * 2.0 * c34 * tmp1 * (tx1 * r43 + ty1 + tz1) +
                      dt * 2.0 * (tx1 * dx2 + ty1 * dy2 + tz1 * dz2);
    tmat[1 + 5 * 2] = 0.0;
    tmat[1 + 5 * 3] = 0.0;
    tmat[1 + 5 * 4] = 0.0;
    tmat[2 + 5 * 0] =
        -dt * 2.0 * (tx1 + ty1 * r43 + tz1) * c34 * tmp2 * u(2, i, j, k);
    tmat[2 + 5 * 1] = 0.0;
    tmat[2 + 5 * 2] = 1.0 + dt * 2.0 * c34 * tmp1 * (tx1 + ty1 * r43 + tz1) +
                      dt * 2.0 * (tx1 * dx3 + ty1 * dy3 + tz1 * dz3);
    tmat[2 + 5 * 3] = 0.0;
    tmat[2 + 5 * 4] = 0.0;
    tmat[3 + 5 * 0] =
        -dt * 2.0 * (tx1 + ty1 + tz1 * r43) * c34 * tmp2 * u(3, i, j, k);
    tmat[3 + 5 * 1] = 0.0;
    tmat[3 + 5 * 2] = 0.0;
    tmat[3 + 5 * 3] = 1.0 + dt * 2.0 * c34 * tmp1 * (tx1 + ty1 + tz1 * r43) +
                      dt * 2.0 * (tx1 * dx4 + ty1 * dy4 + tz1 * dz4);
    tmat[3 + 5 * 4] = 0.0;
    tmat[4 + 5 * 0] = -dt * 2.0 *
                      (((tx1 * (r43 * c34 - c1345) + ty1 * (c34 - c1345) +
                         tz1 * (c34 - c1345)) *
                            (u(1, i, j, k) * u(1, i, j, k)) +
                        (tx1 * (c34 - c1345) + ty1 * (r43 * c34 - c1345) +
                         tz1 * (c34 - c1345)) *
                            (u(2, i, j, k) * u(2, i, j, k)) +
                        (tx1 * (c34 - c1345) + ty1 * (c34 - c1345) +
                         tz1 * (r43 * c34 - c1345)) *
                            (u(3, i, j, k) * u(3, i, j, k))) *
                           tmp3 +
                       (tx1 + ty1 + tz1) * c1345 * tmp2 * u(4, i, j, k));
    tmat[4 + 5 * 1] =
        dt * 2.0 * tmp2 * u(1, i, j, k) *
        (tx1 * (r43 * c34 - c1345) + ty1 * (c34 - c1345) + tz1 * (c34 - c1345));
    tmat[4 + 5 * 2] =
        dt * 2.0 * tmp2 * u(2, i, j, k) *
        (tx1 * (c34 - c1345) + ty1 * (r43 * c34 - c1345) + tz1 * (c34 - c1345));
    tmat[4 + 5 * 3] =
        dt * 2.0 * tmp2 * u(3, i, j, k) *
        (tx1 * (c34 - c1345) + ty1 * (c34 - c1345) + tz1 * (r43 * c34 - c1345));
    tmat[4 + 5 * 4] = 1.0 + dt * 2.0 * (tx1 + ty1 + tz1) * c1345 * tmp1 +
                      dt * 2.0 * (tx1 * dx5 + ty1 * dy5 + tz1 * dz5);
    /*
     * ---------------------------------------------------------------------
     * diagonal block inversion
     * ---------------------------------------------------------------------
     * forward elimination
     * ---------------------------------------------------------------------
     */
    tmp1 = 1.0 / tmat[0 + 0 * 5];
    tmp2 = tmp1 * tmat[1 + 0 * 5];
    tmat[1 + 1 * 5] -= tmp2 * tmat[0 + 1 * 5];
    tmat[1 + 2 * 5] -= tmp2 * tmat[0 + 2 * 5];
    tmat[1 + 3 * 5] -= tmp2 * tmat[0 + 3 * 5];
    tmat[1 + 4 * 5] -= tmp2 * tmat[0 + 4 * 5];
    tv[1] -= tmp2 * tv[0];
    tmp2 = tmp1 * tmat[2 + 0 * 5];
    tmat[2 + 1 * 5] -= tmp2 * tmat[0 + 1 * 5];
    tmat[2 + 2 * 5] -= tmp2 * tmat[0 + 2 * 5];
    tmat[2 + 3 * 5] -= tmp2 * tmat[0 + 3 * 5];
    tmat[2 + 4 * 5] -= tmp2 * tmat[0 + 4 * 5];
    tv[2] -= tmp2 * tv[0];
    tmp2 = tmp1 * tmat[3 + 0 * 5];
    tmat[3 + 1 * 5] -= tmp2 * tmat[0 + 1 * 5];
    tmat[3 + 2 * 5] -= tmp2 * tmat[0 + 2 * 5];
    tmat[3 + 3 * 5] -= tmp2 * tmat[0 + 3 * 5];
    tmat[3 + 4 * 5] -= tmp2 * tmat[0 + 4 * 5];
    tv[3] -= tmp2 * tv[0];
    tmp2 = tmp1 * tmat[4 + 0 * 5];
    tmat[4 + 1 * 5] -= tmp2 * tmat[0 + 1 * 5];
    tmat[4 + 2 * 5] -= tmp2 * tmat[0 + 2 * 5];
    tmat[4 + 3 * 5] -= tmp2 * tmat[0 + 3 * 5];
    tmat[4 + 4 * 5] -= tmp2 * tmat[0 + 4 * 5];
    tv[4] -= tmp2 * tv[0];
    tmp1 = 1.0 / tmat[1 + 1 * 5];
    tmp2 = tmp1 * tmat[2 + 1 * 5];
    tmat[2 + 2 * 5] -= tmp2 * tmat[1 + 2 * 5];
    tmat[2 + 3 * 5] -= tmp2 * tmat[1 + 3 * 5];
    tmat[2 + 4 * 5] -= tmp2 * tmat[1 + 4 * 5];
    tv[2] -= tmp2 * tv[1];
    tmp2 = tmp1 * tmat[3 + 1 * 5];
    tmat[3 + 2 * 5] -= tmp2 * tmat[1 + 2 * 5];
    tmat[3 + 3 * 5] -= tmp2 * tmat[1 + 3 * 5];
    tmat[3 + 4 * 5] -= tmp2 * tmat[1 + 4 * 5];
    tv[3] -= tmp2 * tv[1];
    tmp2 = tmp1 * tmat[4 + 1 * 5];
    tmat[4 + 2 * 5] -= tmp2 * tmat[1 + 2 * 5];
    tmat[4 + 3 * 5] -= tmp2 * tmat[1 + 3 * 5];
    tmat[4 + 4 * 5] -= tmp2 * tmat[1 + 4 * 5];
    tv[4] -= tmp2 * tv[1];
    tmp1 = 1.0 / tmat[2 + 2 * 5];
    tmp2 = tmp1 * tmat[3 + 2 * 5];
    tmat[3 + 3 * 5] -= tmp2 * tmat[2 + 3 * 5];
    tmat[3 + 4 * 5] -= tmp2 * tmat[2 + 4 * 5];
    tv[3] -= tmp2 * tv[2];
    tmp2 = tmp1 * tmat[4 + 2 * 5];
    tmat[4 + 3 * 5] -= tmp2 * tmat[2 + 3 * 5];
    tmat[4 + 4 * 5] -= tmp2 * tmat[2 + 4 * 5];
    tv[4] -= tmp2 * tv[2];
    tmp1 = 1.0 / tmat[3 + 3 * 5];
    tmp2 = tmp1 * tmat[4 + 3 * 5];
    tmat[4 + 4 * 5] -= tmp2 * tmat[3 + 4 * 5];
    tv[4] -= tmp2 * tv[3];
    /*
     * ---------------------------------------------------------------------
     * back substitution
     * ---------------------------------------------------------------------
     */
    v(4, i, j, k) = tv[4] / tmat[4 + 4 * 5];
    tv[3] = tv[3] - tmat[3 + 4 * 5] * v(4, i, j, k);
    v(3, i, j, k) = tv[3] / tmat[3 + 3 * 5];
    tv[2] = tv[2] - tmat[2 + 3 * 5] * v(3, i, j, k) -
            tmat[2 + 4 * 5] * v(4, i, j, k);
    v(2, i, j, k) = tv[2] / tmat[2 + 2 * 5];
    tv[1] = tv[1] - tmat[1 + 2 * 5] * v(2, i, j, k) -
            tmat[1 + 3 * 5] * v(3, i, j, k) - tmat[1 + 4 * 5] * v(4, i, j, k);
    v(1, i, j, k) = tv[1] / tmat[1 + 1 * 5];
    tv[0] = tv[0] - tmat[0 + 1 * 5] * v(1, i, j, k) -
            tmat[0 + 2 * 5] * v(2, i, j, k) - tmat[0 + 3 * 5] * v(3, i, j, k) -
            tmat[0 + 4 * 5] * v(4, i, j, k);
    v(0, i, j, k) = tv[0] / tmat[0 + 0 * 5];
}

__global__ void jacu_buts_gpu_kernel(const int plane, const int klower,
                                     const int jlower, const double* u,
                                     const double* rho_i, const double* qs,
                                     double* v, const int nx, const int ny,
                                     const int nz) {
    int i, j, k, m;
    double tmp, tmp1, tmp2, tmp3, tmat[5 * 5], tv[5];
    double r43, c1345, c34;

    k = klower + blockIdx.x + 1;
    j = jlower + threadIdx.x + 1;

    i = plane - j - k + 3;

    if ((i < 1) || (i > (nx - 2)) || (j > (ny - 2))) {
        return;
    }

    using namespace constants_device;
    r43 = 4.0 / 3.0;
    c1345 = C1 * C3 * C4 * C5;
    c34 = C3 * C4;
    /*
     * ---------------------------------------------------------------------
     * form the first block sub-diagonal
     * ---------------------------------------------------------------------
     */
    tmp1 = rho_i(i + 1, j, k);
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;
    tmat[0 + 5 * 0] = -dt * tx1 * dx1;
    tmat[0 + 5 * 1] = dt * tx2;
    tmat[0 + 5 * 2] = 0.0;
    tmat[0 + 5 * 3] = 0.0;
    tmat[0 + 5 * 4] = 0.0;
    tmat[1 + 5 * 0] =
        dt * tx2 *
            (-(u(1, i + 1, j, k) * tmp1) * (u(1, i + 1, j, k) * tmp1) +
             C2 * qs(i + 1, j, k) * tmp1) -
        dt * tx1 * (-r43 * c34 * tmp2 * u(1, i + 1, j, k));
    tmat[1 + 5 * 1] = dt * tx2 * ((2.0 - C2) * (u(1, i + 1, j, k) * tmp1)) -
                      dt * tx1 * (r43 * c34 * tmp1) - dt * tx1 * dx2;
    tmat[1 + 5 * 2] = dt * tx2 * (-C2 * (u(2, i + 1, j, k) * tmp1));
    tmat[1 + 5 * 3] = dt * tx2 * (-C2 * (u(3, i + 1, j, k) * tmp1));
    tmat[1 + 5 * 4] = dt * tx2 * C2;
    tmat[2 + 5 * 0] =
        dt * tx2 * (-(u(1, i + 1, j, k) * u(2, i + 1, j, k)) * tmp2) -
        dt * tx1 * (-c34 * tmp2 * u(2, i + 1, j, k));
    tmat[2 + 5 * 1] = dt * tx2 * (u(2, i + 1, j, k) * tmp1);
    tmat[2 + 5 * 2] = dt * tx2 * (u(1, i + 1, j, k) * tmp1) -
                      dt * tx1 * (c34 * tmp1) - dt * tx1 * dx3;
    tmat[2 + 5 * 3] = 0.0;
    tmat[2 + 5 * 4] = 0.0;
    tmat[3 + 5 * 0] =
        dt * tx2 * (-(u(1, i + 1, j, k) * u(3, i + 1, j, k)) * tmp2) -
        dt * tx1 * (-c34 * tmp2 * u(3, i + 1, j, k));
    tmat[3 + 5 * 1] = dt * tx2 * (u(3, i + 1, j, k) * tmp1);
    tmat[3 + 5 * 2] = 0.0;
    tmat[3 + 5 * 3] = dt * tx2 * (u(1, i + 1, j, k) * tmp1) -
                      dt * tx1 * (c34 * tmp1) - dt * tx1 * dx4;
    tmat[3 + 5 * 4] = 0.0;
    tmat[4 + 5 * 0] =
        dt * tx2 *
            ((C2 * 2.0 * qs(i + 1, j, k) - C1 * u(4, i + 1, j, k)) *
             (u(1, i + 1, j, k) * tmp2)) -
        dt * tx1 *
            (-(r43 * c34 - c1345) * tmp3 *
                 (u(1, i + 1, j, k) * u(1, i + 1, j, k)) -
             (c34 - c1345) * tmp3 * (u(2, i + 1, j, k) * u(2, i + 1, j, k)) -
             (c34 - c1345) * tmp3 * (u(3, i + 1, j, k) * u(3, i + 1, j, k)) -
             c1345 * tmp2 * u(4, i + 1, j, k));
    tmat[4 + 5 * 1] = dt * tx2 *
                          (C1 * (u(4, i + 1, j, k) * tmp1) -
                           C2 * (u(1, i + 1, j, k) * u(1, i + 1, j, k) * tmp2 +
                                 qs(i + 1, j, k) * tmp1)) -
                      dt * tx1 * (r43 * c34 - c1345) * tmp2 * u(1, i + 1, j, k);
    tmat[4 + 5 * 2] =
        dt * tx2 * (-C2 * (u(2, i + 1, j, k) * u(1, i + 1, j, k)) * tmp2) -
        dt * tx1 * (c34 - c1345) * tmp2 * u(2, i + 1, j, k);
    tmat[4 + 5 * 3] =
        dt * tx2 * (-C2 * (u(3, i + 1, j, k) * u(1, i + 1, j, k)) * tmp2) -
        dt * tx1 * (c34 - c1345) * tmp2 * u(3, i + 1, j, k);
    tmat[4 + 5 * 4] = dt * tx2 * (C1 * (u(1, i + 1, j, k) * tmp1)) -
                      dt * tx1 * c1345 * tmp1 - dt * tx1 * dx5;
    for (m = 0; m < 5; m++) {
        tv[m] = omega * (tmat[m + 5 * 0] * v(0, i + 1, j, k) +
                         tmat[m + 5 * 1] * v(1, i + 1, j, k) +
                         tmat[m + 5 * 2] * v(2, i + 1, j, k) +
                         tmat[m + 5 * 3] * v(3, i + 1, j, k) +
                         tmat[m + 5 * 4] * v(4, i + 1, j, k));
    }
    /*
     * ---------------------------------------------------------------------
     * form the second block sub-diagonal
     * ---------------------------------------------------------------------
     */
    tmp1 = rho_i(i, j + 1, k);
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;
    tmat[0 + 5 * 0] = -dt * ty1 * dy1;
    tmat[0 + 5 * 1] = 0.0;
    tmat[0 + 5 * 2] = dt * ty2;
    tmat[0 + 5 * 3] = 0.0;
    tmat[0 + 5 * 4] = 0.0;
    tmat[1 + 5 * 0] =
        dt * ty2 * (-(u(1, i, j + 1, k) * u(2, i, j + 1, k)) * tmp2) -
        dt * ty1 * (-c34 * tmp2 * u(1, i, j + 1, k));
    tmat[1 + 5 * 1] = dt * ty2 * (u(2, i, j + 1, k) * tmp1) -
                      dt * ty1 * (c34 * tmp1) - dt * ty1 * dy2;
    tmat[1 + 5 * 2] = dt * ty2 * (u(1, i, j + 1, k) * tmp1);
    tmat[1 + 5 * 3] = 0.0;
    tmat[1 + 5 * 4] = 0.0;
    tmat[2 + 5 * 0] =
        dt * ty2 *
            (-(u(2, i, j + 1, k) * tmp1) * (u(2, i, j + 1, k) * tmp1) +
             C2 * (qs(i, j + 1, k) * tmp1)) -
        dt * ty1 * (-r43 * c34 * tmp2 * u(2, i, j + 1, k));
    tmat[2 + 5 * 1] = dt * ty2 * (-C2 * (u(1, i, j + 1, k) * tmp1));
    tmat[2 + 5 * 2] = dt * ty2 * ((2.0 - C2) * (u(2, i, j + 1, k) * tmp1)) -
                      dt * ty1 * (r43 * c34 * tmp1) - dt * ty1 * dy3;
    tmat[2 + 5 * 3] = dt * ty2 * (-C2 * (u(3, i, j + 1, k) * tmp1));
    tmat[2 + 5 * 4] = dt * ty2 * C2;
    tmat[3 + 5 * 0] =
        dt * ty2 * (-(u(2, i, j + 1, k) * u(3, i, j + 1, k)) * tmp2) -
        dt * ty1 * (-c34 * tmp2 * u(3, i, j + 1, k));
    tmat[3 + 5 * 1] = 0.0;
    tmat[3 + 5 * 2] = dt * ty2 * (u(3, i, j + 1, k) * tmp1);
    tmat[3 + 5 * 3] = dt * ty2 * (u(2, i, j + 1, k) * tmp1) -
                      dt * ty1 * (c34 * tmp1) - dt * ty1 * dy4;
    tmat[3 + 5 * 4] = 0.0;
    tmat[4 + 5 * 0] =
        dt * ty2 *
            ((C2 * 2.0 * qs(i, j + 1, k) - C1 * u(4, i, j + 1, k)) *
             (u(2, i, j + 1, k) * tmp2)) -
        dt * ty1 *
            (-(c34 - c1345) * tmp3 * (u(1, i, j + 1, k) * u(1, i, j + 1, k)) -
             (r43 * c34 - c1345) * tmp3 *
                 (u(2, i, j + 1, k) * u(2, i, j + 1, k)) -
             (c34 - c1345) * tmp3 * (u(3, i, j + 1, k) * u(3, i, j + 1, k)) -
             c1345 * tmp2 * u(4, i, j + 1, k));
    tmat[4 + 5 * 1] =
        dt * ty2 * (-C2 * (u(1, i, j + 1, k) * u(2, i, j + 1, k)) * tmp2) -
        dt * ty1 * (c34 - c1345) * tmp2 * u(1, i, j + 1, k);
    tmat[4 + 5 * 2] =
        dt * ty2 *
            (C1 * (u(4, i, j + 1, k) * tmp1) -
             C2 * (qs(i, j + 1, k) * tmp1 +
                   u(2, i, j + 1, k) * u(2, i, j + 1, k) * tmp2)) -
        dt * ty1 * (r43 * c34 - c1345) * tmp2 * u(2, i, j + 1, k);
    tmat[4 + 5 * 3] =
        dt * ty2 * (-C2 * (u(2, i, j + 1, k) * u(3, i, j + 1, k)) * tmp2) -
        dt * ty1 * (c34 - c1345) * tmp2 * u(3, i, j + 1, k);
    tmat[4 + 5 * 4] = dt * ty2 * (C1 * (u(2, i, j + 1, k) * tmp1)) -
                      dt * ty1 * c1345 * tmp1 - dt * ty1 * dy5;
    for (m = 0; m < 5; m++) {
        tv[m] = tv[m] + omega * (tmat[m + 5 * 0] * v(0, i, j + 1, k) +
                                 tmat[m + 5 * 1] * v(1, i, j + 1, k) +
                                 tmat[m + 5 * 2] * v(2, i, j + 1, k) +
                                 tmat[m + 5 * 3] * v(3, i, j + 1, k) +
                                 tmat[m + 5 * 4] * v(4, i, j + 1, k));
    }
    /*
     * ---------------------------------------------------------------------
     * form the third block sub-diagonal
     * ---------------------------------------------------------------------
     */
    tmp1 = rho_i(i, j, k + 1);
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;
    tmat[0 + 5 * 0] = -dt * tz1 * dz1;
    tmat[0 + 5 * 1] = 0.0;
    tmat[0 + 5 * 2] = 0.0;
    tmat[0 + 5 * 3] = dt * tz2;
    tmat[0 + 5 * 4] = 0.0;
    tmat[1 + 5 * 0] =
        dt * tz2 * (-(u(1, i, j, k + 1) * u(3, i, j, k + 1)) * tmp2) -
        dt * tz1 * (-c34 * tmp2 * u(1, i, j, k + 1));
    tmat[1 + 5 * 1] = dt * tz2 * (u(3, i, j, k + 1) * tmp1) -
                      dt * tz1 * c34 * tmp1 - dt * tz1 * dz2;
    tmat[1 + 5 * 2] = 0.0;
    tmat[1 + 5 * 3] = dt * tz2 * (u(1, i, j, k + 1) * tmp1);
    tmat[1 + 5 * 4] = 0.0;
    tmat[2 + 5 * 0] =
        dt * tz2 * (-(u(2, i, j, k + 1) * u(3, i, j, k + 1)) * tmp2) -
        dt * tz1 * (-c34 * tmp2 * u(2, i, j, k + 1));
    tmat[2 + 5 * 1] = 0.0;
    tmat[2 + 5 * 2] = dt * tz2 * (u(3, i, j, k + 1) * tmp1) -
                      dt * tz1 * (c34 * tmp1) - dt * tz1 * dz3;
    tmat[2 + 5 * 3] = dt * tz2 * (u(2, i, j, k + 1) * tmp1);
    tmat[2 + 5 * 4] = 0.0;
    tmat[3 + 5 * 0] =
        dt * tz2 *
            (-(u(3, i, j, k + 1) * tmp1) * (u(3, i, j, k + 1) * tmp1) +
             C2 * (qs(i, j, k + 1) * tmp1)) -
        dt * tz1 * (-r43 * c34 * tmp2 * u(3, i, j, k + 1));
    tmat[3 + 5 * 1] = dt * tz2 * (-C2 * (u(1, i, j, k + 1) * tmp1));
    tmat[3 + 5 * 2] = dt * tz2 * (-C2 * (u(2, i, j, k + 1) * tmp1));
    tmat[3 + 5 * 3] = dt * tz2 * (2.0 - C2) * (u(3, i, j, k + 1) * tmp1) -
                      dt * tz1 * (r43 * c34 * tmp1) - dt * tz1 * dz4;
    tmat[3 + 5 * 4] = dt * tz2 * C2;
    tmat[4 + 5 * 0] =
        dt * tz2 *
            ((C2 * 2.0 * qs(i, j, k + 1) - C1 * u(4, i, j, k + 1)) *
             (u(3, i, j, k + 1) * tmp2)) -
        dt * tz1 *
            (-(c34 - c1345) * tmp3 * (u(1, i, j, k + 1) * u(1, i, j, k + 1)) -
             (c34 - c1345) * tmp3 * (u(2, i, j, k + 1) * u(2, i, j, k + 1)) -
             (r43 * c34 - c1345) * tmp3 *
                 (u(3, i, j, k + 1) * u(3, i, j, k + 1)) -
             c1345 * tmp2 * u(4, i, j, k + 1));
    tmat[4 + 5 * 1] =
        dt * tz2 * (-C2 * (u(1, i, j, k + 1) * u(3, i, j, k + 1)) * tmp2) -
        dt * tz1 * (c34 - c1345) * tmp2 * u(1, i, j, k + 1);
    tmat[4 + 5 * 2] =
        dt * tz2 * (-C2 * (u(2, i, j, k + 1) * u(3, i, j, k + 1)) * tmp2) -
        dt * tz1 * (c34 - c1345) * tmp2 * u(2, i, j, k + 1);
    tmat[4 + 5 * 3] =
        dt * tz2 *
            (C1 * (u(4, i, j, k + 1) * tmp1) -
             C2 * (qs(i, j, k + 1) * tmp1 +
                   u(3, i, j, k + 1) * u(3, i, j, k + 1) * tmp2)) -
        dt * tz1 * (r43 * c34 - c1345) * tmp2 * u(3, i, j, k + 1);
    tmat[4 + 5 * 4] = dt * tz2 * (C1 * (u(3, i, j, k + 1) * tmp1)) -
                      dt * tz1 * c1345 * tmp1 - dt * tz1 * dz5;
    for (m = 0; m < 5; m++) {
        tv[m] = tv[m] + omega * (tmat[m + 5 * 0] * v(0, i, j, k + 1) +
                                 tmat[m + 5 * 1] * v(1, i, j, k + 1) +
                                 tmat[m + 5 * 2] * v(2, i, j, k + 1) +
                                 tmat[m + 5 * 3] * v(3, i, j, k + 1) +
                                 tmat[m + 5 * 4] * v(4, i, j, k + 1));
    }
    /*
     * ---------------------------------------------------------------------
     * form the block daigonal
     * ---------------------------------------------------------------------
     */
    tmp1 = rho_i(i, j, k);
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;
    tmat[0 + 5 * 0] = 1.0 + dt * 2.0 * (tx1 * dx1 + ty1 * dy1 + tz1 * dz1);
    tmat[0 + 5 * 1] = 0.0;
    tmat[0 + 5 * 2] = 0.0;
    tmat[0 + 5 * 3] = 0.0;
    tmat[0 + 5 * 4] = 0.0;
    tmat[1 + 5 * 0] =
        dt * 2.0 * (-tx1 * r43 - ty1 - tz1) * (c34 * tmp2 * u(1, i, j, k));
    tmat[1 + 5 * 1] = 1.0 + dt * 2.0 * c34 * tmp1 * (tx1 * r43 + ty1 + tz1) +
                      dt * 2.0 * (tx1 * dx2 + ty1 * dy2 + tz1 * dz2);
    tmat[1 + 5 * 2] = 0.0;
    tmat[1 + 5 * 3] = 0.0;
    tmat[1 + 5 * 4] = 0.0;
    tmat[2 + 5 * 0] =
        dt * 2.0 * (-tx1 - ty1 * r43 - tz1) * (c34 * tmp2 * u(2, i, j, k));
    tmat[2 + 5 * 1] = 0.0;
    tmat[2 + 5 * 2] = 1.0 + dt * 2.0 * c34 * tmp1 * (tx1 + ty1 * r43 + tz1) +
                      dt * 2.0 * (tx1 * dx3 + ty1 * dy3 + tz1 * dz3);
    tmat[2 + 5 * 3] = 0.0;
    tmat[2 + 5 * 4] = 0.0;
    tmat[3 + 5 * 0] =
        dt * 2.0 * (-tx1 - ty1 - tz1 * r43) * (c34 * tmp2 * u(3, i, j, k));
    tmat[3 + 5 * 1] = 0.0;
    tmat[3 + 5 * 2] = 0.0;
    tmat[3 + 5 * 3] = 1.0 + dt * 2.0 * c34 * tmp1 * (tx1 + ty1 + tz1 * r43) +
                      dt * 2.0 * (tx1 * dx4 + ty1 * dy4 + tz1 * dz4);
    tmat[3 + 5 * 4] = 0.0;
    tmat[4 + 5 * 0] = -dt * 2.0 *
                      (((tx1 * (r43 * c34 - c1345) + ty1 * (c34 - c1345) +
                         tz1 * (c34 - c1345)) *
                            (u(1, i, j, k) * u(1, i, j, k)) +
                        (tx1 * (c34 - c1345) + ty1 * (r43 * c34 - c1345) +
                         tz1 * (c34 - c1345)) *
                            (u(2, i, j, k) * u(2, i, j, k)) +
                        (tx1 * (c34 - c1345) + ty1 * (c34 - c1345) +
                         tz1 * (r43 * c34 - c1345)) *
                            (u(3, i, j, k) * u(3, i, j, k))) *
                           tmp3 +
                       (tx1 + ty1 + tz1) * c1345 * tmp2 * u(4, i, j, k));
    tmat[4 + 5 * 1] = dt * 2.0 *
                      (tx1 * (r43 * c34 - c1345) + ty1 * (c34 - c1345) +
                       tz1 * (c34 - c1345)) *
                      tmp2 * u(1, i, j, k);
    tmat[4 + 5 * 2] = dt * 2.0 *
                      (tx1 * (c34 - c1345) + ty1 * (r43 * c34 - c1345) +
                       tz1 * (c34 - c1345)) *
                      tmp2 * u(2, i, j, k);
    tmat[4 + 5 * 3] = dt * 2.0 *
                      (tx1 * (c34 - c1345) + ty1 * (c34 - c1345) +
                       tz1 * (r43 * c34 - c1345)) *
                      tmp2 * u(3, i, j, k);
    tmat[4 + 5 * 4] = 1.0 + dt * 2.0 * (tx1 + ty1 + tz1) * c1345 * tmp1 +
                      dt * 2.0 * (tx1 * dx5 + ty1 * dy5 + tz1 * dz5);
    /*
     * ---------------------------------------------------------------------
     * diagonal block inversion
     * ---------------------------------------------------------------------
     */
    tmp1 = 1.0 / tmat[0 + 0 * 5];
    tmp = tmp1 * tmat[1 + 0 * 5];
    tmat[1 + 1 * 5] -= tmp * tmat[0 + 1 * 5];
    tmat[1 + 2 * 5] -= tmp * tmat[0 + 2 * 5];
    tmat[1 + 3 * 5] -= tmp * tmat[0 + 3 * 5];
    tmat[1 + 4 * 5] -= tmp * tmat[0 + 4 * 5];
    tv[1] -= tmp * tv[0];
    tmp = tmp1 * tmat[2 + 0 * 5];
    tmat[2 + 1 * 5] -= tmp * tmat[0 + 1 * 5];
    tmat[2 + 2 * 5] -= tmp * tmat[0 + 2 * 5];
    tmat[2 + 3 * 5] -= tmp * tmat[0 + 3 * 5];
    tmat[2 + 4 * 5] -= tmp * tmat[0 + 4 * 5];
    tv[2] -= tmp * tv[0];
    tmp = tmp1 * tmat[3 + 0 * 5];
    tmat[3 + 1 * 5] -= tmp * tmat[0 + 1 * 5];
    tmat[3 + 2 * 5] -= tmp * tmat[0 + 2 * 5];
    tmat[3 + 3 * 5] -= tmp * tmat[0 + 3 * 5];
    tmat[3 + 4 * 5] -= tmp * tmat[0 + 4 * 5];
    tv[3] -= tmp * tv[0];
    tmp = tmp1 * tmat[4 + 0 * 5];
    tmat[4 + 1 * 5] -= tmp * tmat[0 + 1 * 5];
    tmat[4 + 2 * 5] -= tmp * tmat[0 + 2 * 5];
    tmat[4 + 3 * 5] -= tmp * tmat[0 + 3 * 5];
    tmat[4 + 4 * 5] -= tmp * tmat[0 + 4 * 5];
    tv[4] -= tmp * tv[0];
    tmp1 = 1.0 / tmat[1 + 1 * 5];
    tmp = tmp1 * tmat[2 + 1 * 5];
    tmat[2 + 2 * 5] -= tmp * tmat[1 + 2 * 5];
    tmat[2 + 3 * 5] -= tmp * tmat[1 + 3 * 5];
    tmat[2 + 4 * 5] -= tmp * tmat[1 + 4 * 5];
    tv[2] -= tmp * tv[1];
    tmp = tmp1 * tmat[3 + 1 * 5];
    tmat[3 + 2 * 5] -= tmp * tmat[1 + 2 * 5];
    tmat[3 + 3 * 5] -= tmp * tmat[1 + 3 * 5];
    tmat[3 + 4 * 5] -= tmp * tmat[1 + 4 * 5];
    tv[3] -= tmp * tv[1];
    tmp = tmp1 * tmat[4 + 1 * 5];
    tmat[4 + 2 * 5] -= tmp * tmat[1 + 2 * 5];
    tmat[4 + 3 * 5] -= tmp * tmat[1 + 3 * 5];
    tmat[4 + 4 * 5] -= tmp * tmat[1 + 4 * 5];
    tv[4] -= tmp * tv[1];
    tmp1 = 1.0 / tmat[2 + 2 * 5];
    tmp = tmp1 * tmat[3 + 2 * 5];
    tmat[3 + 3 * 5] -= tmp * tmat[2 + 3 * 5];
    tmat[3 + 4 * 5] -= tmp * tmat[2 + 4 * 5];
    tv[3] -= tmp * tv[2];
    tmp = tmp1 * tmat[4 + 2 * 5];
    tmat[4 + 3 * 5] -= tmp * tmat[2 + 3 * 5];
    tmat[4 + 4 * 5] -= tmp * tmat[2 + 4 * 5];
    tv[4] -= tmp * tv[2];
    tmp1 = 1.0 / tmat[3 + 3 * 5];
    tmp = tmp1 * tmat[4 + 3 * 5];
    tmat[4 + 4 * 5] -= tmp * tmat[3 + 4 * 5];
    tv[4] -= tmp * tv[3];
    /*
     * ---------------------------------------------------------------------
     * back substitution
     * ---------------------------------------------------------------------
     */
    tv[4] = tv[4] / tmat[4 + 4 * 5];
    tv[3] = tv[3] - tmat[3 + 4 * 5] * tv[4];
    tv[3] = tv[3] / tmat[3 + 3 * 5];
    tv[2] = tv[2] - tmat[2 + 3 * 5] * tv[3] - tmat[2 + 4 * 5] * tv[4];
    tv[2] = tv[2] / tmat[2 + 2 * 5];
    tv[1] = tv[1] - tmat[1 + 2 * 5] * tv[2] - tmat[1 + 3 * 5] * tv[3] -
            tmat[1 + 4 * 5] * tv[4];
    tv[1] = tv[1] / tmat[1 + 1 * 5];
    tv[0] = tv[0] - tmat[0 + 1 * 5] * tv[1] - tmat[0 + 2 * 5] * tv[2] -
            tmat[0 + 3 * 5] * tv[3] - tmat[0 + 4 * 5] * tv[4];
    tv[0] = tv[0] / tmat[0 + 0 * 5];
    v(0, i, j, k) -= tv[0];
    v(1, i, j, k) -= tv[1];
    v(2, i, j, k) -= tv[2];
    v(3, i, j, k) -= tv[3];
    v(4, i, j, k) -= tv[4];
}

__global__ void l2norm_gpu_kernel(const double* v, double* sum, const int nx,
                                  const int ny, const int nz) {
    int i, j, k, m;

    double* sum_loc = (double*)extern_share_data;

    k = blockIdx.x + 1;
    j = blockIdx.y + 1;
    i = threadIdx.x + 1;

    for (m = 0; m < 5; m++) {
        sum_loc[m + 5 * threadIdx.x] = 0.0;
    }
    while (i < (nx - 1)) {
        for (m = 0; m < 5; m++) {
            sum_loc[m + 5 * threadIdx.x] += v(m, i, j, k) * v(m, i, j, k);
        }
        i += blockDim.x;
    }
    i = threadIdx.x;
    int loc_max = blockDim.x;
    int dist = (loc_max + 1) / 2;
    __syncthreads();
    while (loc_max > 1) {
        if ((i < dist) && (i + dist < loc_max)) {
            for (m = 0; m < 5; m++) {
                sum_loc[m + 5 * i] += sum_loc[m + 5 * (i + dist)];
            }
        }
        loc_max = dist;
        dist = (dist + 1) / 2;
        __syncthreads();
    }
    if (i == 0) {
        for (m = 0; m < 5; m++) {
            sum[m + 5 * (blockIdx.y + gridDim.y * blockIdx.x)] = sum_loc[m];
        }
    }
}

__global__ void norm_gpu_kernel(double* rms, const int size) {
    int i, m, loc_max, dist;

    double* buffer = (double*)extern_share_data;

    i = threadIdx.x;

    for (m = 0; m < 5; m++) {
        buffer[m + 5 * i] = 0.0;
    }
    while (i < size) {
        for (m = 0; m < 5; m++) {
            buffer[m + 5 * threadIdx.x] += rms[m + 5 * i];
        }
        i += blockDim.x;
    }
    loc_max = blockDim.x;
    dist = (loc_max + 1) / 2;
    i = threadIdx.x;
    __syncthreads();
    while (loc_max > 1) {
        if ((i < dist) && ((i + dist) < loc_max)) {
            for (m = 0; m < 5; m++) {
                buffer[m + 5 * i] += buffer[m + 5 * (i + dist)];
            }
        }
        loc_max = dist;
        dist = (dist + 1) / 2;
        __syncthreads();
    }
    if (threadIdx.x < 5) {
        rms[threadIdx.x] = buffer[threadIdx.x];
    }
}

__global__ void rhs_gpu_kernel_1(const double* u, double* rsd,
                                 const double* frct, double* qs, double* rho_i,
                                 const int nx, const int ny, const int nz) {
    int i_j_k, i, j, k, m;
    double tmp;

    i_j_k = blockIdx.x * blockDim.x + threadIdx.x;

    i = i_j_k % nx;
    j = (i_j_k / nx) % ny;
    k = i_j_k / (nx * ny);

    if (i_j_k >= (nx * ny * nz)) {
        return;
    }

    for (m = 0; m < 5; m++) {
        rsd(m, i, j, k) = -frct(m, i, j, k);
    }
    rho_i(i, j, k) = tmp = 1.0 / u(0, i, j, k);
    qs(i, j, k) =
        0.5 *
        (u(1, i, j, k) * u(1, i, j, k) + u(2, i, j, k) * u(2, i, j, k) +
         u(3, i, j, k) * u(3, i, j, k)) *
        tmp;
}

__global__ void rhs_gpu_kernel_2(const double* u, double* rsd, const double* qs,
                                 const double* rho_i, const int nx,
                                 const int ny, const int nz) {
    int i, j, k, m, nthreads;
    double q, u21;

    double* flux = (double*)extern_share_data;
    double* utmp = (double*)flux + (blockDim.x * 5);
    double* rtmp = (double*)utmp + (blockDim.x * 5);
    double* rhotmp = (double*)rtmp + (blockDim.x * 5);
    double* u21i = (double*)rhotmp + (blockDim.x);
    double* u31i = (double*)u21i + (blockDim.x);
    double* u41i = (double*)u31i + (blockDim.x);
    double* u51i = (double*)u41i + (blockDim.x);

    k = blockIdx.x + 1;
    j = blockIdx.y + 1;
    i = threadIdx.x;

    using namespace constants_device;
    while (i < nx) {
        nthreads = nx - (i - threadIdx.x);
        if (nthreads > blockDim.x) {
            nthreads = blockDim.x;
        }
        m = threadIdx.x;
        utmp[m] = u(m % 5, (i - threadIdx.x) + m / 5, j, k);
        rtmp[m] = rsd(m % 5, (i - threadIdx.x) + m / 5, j, k);
        m += nthreads;
        utmp[m] = u(m % 5, (i - threadIdx.x) + m / 5, j, k);
        rtmp[m] = rsd(m % 5, (i - threadIdx.x) + m / 5, j, k);
        m += nthreads;
        utmp[m] = u(m % 5, (i - threadIdx.x) + m / 5, j, k);
        rtmp[m] = rsd(m % 5, (i - threadIdx.x) + m / 5, j, k);
        m += nthreads;
        utmp[m] = u(m % 5, (i - threadIdx.x) + m / 5, j, k);
        rtmp[m] = rsd(m % 5, (i - threadIdx.x) + m / 5, j, k);
        m += nthreads;
        utmp[m] = u(m % 5, (i - threadIdx.x) + m / 5, j, k);
        rtmp[m] = rsd(m % 5, (i - threadIdx.x) + m / 5, j, k);
        rhotmp[threadIdx.x] = rho_i(i, j, k);
        __syncthreads();
        /*
         * ---------------------------------------------------------------------
         * xi-direction flux differences
         * ---------------------------------------------------------------------
         */
        flux[threadIdx.x + (0 * blockDim.x)] = utmp[threadIdx.x * 5 + 1];
        u21 = utmp[threadIdx.x * 5 + 1] * rhotmp[threadIdx.x];
        q = qs(i, j, k);
        flux[threadIdx.x + (1 * blockDim.x)] =
            utmp[threadIdx.x * 5 + 1] * u21 +
            C2 * (utmp[threadIdx.x * 5 + 4] - q);
        flux[threadIdx.x + (2 * blockDim.x)] = utmp[threadIdx.x * 5 + 2] * u21;
        flux[threadIdx.x + (3 * blockDim.x)] = utmp[threadIdx.x * 5 + 3] * u21;
        flux[threadIdx.x + (4 * blockDim.x)] =
            (C1 * utmp[threadIdx.x * 5 + 4] - C2 * q) * u21;
        __syncthreads();
        if ((threadIdx.x >= 1) && (threadIdx.x < (blockDim.x - 1)) &&
            (i < (nx - 1))) {
            for (m = 0; m < 5; m++) {
                rtmp[threadIdx.x * 5 + m] =
                    rtmp[threadIdx.x * 5 + m] -
                    tx2 * (flux[(threadIdx.x + 1) + (m * blockDim.x)] -
                           flux[(threadIdx.x - 1) + (m * blockDim.x)]);
            }
        }
        u21i[threadIdx.x] = rhotmp[threadIdx.x] * utmp[threadIdx.x * 5 + 1];
        u31i[threadIdx.x] = rhotmp[threadIdx.x] * utmp[threadIdx.x * 5 + 2];
        u41i[threadIdx.x] = rhotmp[threadIdx.x] * utmp[threadIdx.x * 5 + 3];
        u51i[threadIdx.x] = rhotmp[threadIdx.x] * utmp[threadIdx.x * 5 + 4];
        __syncthreads();
        if (threadIdx.x >= 1) {
            flux[threadIdx.x + (1 * blockDim.x)] =
                (4.0 / 3.0) * tx3 * (u21i[threadIdx.x] - u21i[threadIdx.x - 1]);
            flux[threadIdx.x + (2 * blockDim.x)] =
                tx3 * (u31i[threadIdx.x] - u31i[threadIdx.x - 1]);
            flux[threadIdx.x + (3 * blockDim.x)] =
                tx3 * (u41i[threadIdx.x] - u41i[threadIdx.x - 1]);
            flux[threadIdx.x + (4 * blockDim.x)] =
                0.5 * (1.0 - C1 * C5) * tx3 *
                    ((u21i[threadIdx.x] * u21i[threadIdx.x] +
                      u31i[threadIdx.x] * u31i[threadIdx.x] +
                      u41i[threadIdx.x] * u41i[threadIdx.x]) -
                     (u21i[threadIdx.x - 1] * u21i[threadIdx.x - 1] +
                      u31i[threadIdx.x - 1] * u31i[threadIdx.x - 1] +
                      u41i[threadIdx.x - 1] * u41i[threadIdx.x - 1])) +
                (1.0 / 6.0) * tx3 *
                    (u21i[threadIdx.x] * u21i[threadIdx.x] -
                     u21i[threadIdx.x - 1] * u21i[threadIdx.x - 1]) +
                C1 * C5 * tx3 * (u51i[threadIdx.x] - u51i[threadIdx.x - 1]);
        }
        __syncthreads();
        if ((threadIdx.x >= 1) && (threadIdx.x < (blockDim.x - 1)) &&
            (i < (nx - 1))) {
            rtmp[threadIdx.x * 5 + 0] +=
                dx1 * tx1 *
                (utmp[threadIdx.x * 5 - 5] - 2.0 * utmp[threadIdx.x * 5 + 0] +
                 utmp[threadIdx.x * 5 + 5]);
            rtmp[threadIdx.x * 5 + 1] +=
                tx3 * C3 * C4 *
                    (flux[(threadIdx.x + 1) + (1 * blockDim.x)] -
                     flux[threadIdx.x + (1 * blockDim.x)]) +
                dx2 * tx1 *
                    (utmp[threadIdx.x * 5 - 4] -
                     2.0 * utmp[threadIdx.x * 5 + 1] +
                     utmp[threadIdx.x * 5 + 6]);
            rtmp[threadIdx.x * 5 + 2] +=
                tx3 * C3 * C4 *
                    (flux[(threadIdx.x + 1) + (2 * blockDim.x)] -
                     flux[threadIdx.x + (2 * blockDim.x)]) +
                dx3 * tx1 *
                    (utmp[threadIdx.x * 5 - 3] -
                     2.0 * utmp[threadIdx.x * 5 + 2] +
                     utmp[threadIdx.x * 5 + 7]);
            rtmp[threadIdx.x * 5 + 3] +=
                tx3 * C3 * C4 *
                    (flux[(threadIdx.x + 1) + (3 * blockDim.x)] -
                     flux[threadIdx.x + (3 * blockDim.x)]) +
                dx4 * tx1 *
                    (utmp[threadIdx.x * 5 - 2] -
                     2.0 * utmp[threadIdx.x * 5 + 3] +
                     utmp[threadIdx.x * 5 + 8]);
            rtmp[threadIdx.x * 5 + 4] +=
                tx3 * C3 * C4 *
                    (flux[(threadIdx.x + 1) + (4 * blockDim.x)] -
                     flux[threadIdx.x + (4 * blockDim.x)]) +
                dx5 * tx1 *
                    (utmp[threadIdx.x * 5 - 1] -
                     2.0 * utmp[threadIdx.x * 5 + 4] +
                     utmp[threadIdx.x * 5 + 9]);
            /*
             * ---------------------------------------------------------------------
             * fourth-order dissipation
             * ---------------------------------------------------------------------
             */
            if (i == 1) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp *
                        (5.0 * utmp[threadIdx.x * 5 + m] -
                         4.0 * utmp[threadIdx.x * 5 + m + 5] + u(m, 3, j, k));
                }
            }
            if (i == 2) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp *
                        (-4.0 * utmp[threadIdx.x * 5 + m - 5] +
                         6.0 * utmp[threadIdx.x * 5 + m] -
                         4.0 * utmp[threadIdx.x * 5 + m + 5] + u(m, 4, j, k));
                }
            }
            if ((i >= 3) && (i < (nx - 3))) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp * (u(m, i - 2, j, k) -
                                4.0 * utmp[threadIdx.x * 5 + m - 5] +
                                6.0 * utmp[threadIdx.x * 5 + m] -
                                4.0 * utmp[threadIdx.x * 5 + m + 5] +
                                u(m, i + 2, j, k));
                }
            }
            if (i == (nx - 3)) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp * (u(m, nx - 5, j, k) -
                                4.0 * utmp[threadIdx.x * 5 + m - 5] +
                                6.0 * utmp[threadIdx.x * 5 + m] -
                                4.0 * utmp[threadIdx.x * 5 + m + 5]);
                }
            }
            if (i == (nx - 2)) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp * (u(m, nx - 4, j, k) -
                                4.0 * utmp[threadIdx.x * 5 + m - 5] +
                                5.0 * utmp[threadIdx.x * 5 + m]);
                }
            }
        }
        m = threadIdx.x;
        rsd(m % 5, (i - threadIdx.x) + m / 5, j, k) = rtmp[m];
        m += nthreads;
        rsd(m % 5, (i - threadIdx.x) + m / 5, j, k) = rtmp[m];
        m += nthreads;
        rsd(m % 5, (i - threadIdx.x) + m / 5, j, k) = rtmp[m];
        m += nthreads;
        rsd(m % 5, (i - threadIdx.x) + m / 5, j, k) = rtmp[m];
        m += nthreads;
        rsd(m % 5, (i - threadIdx.x) + m / 5, j, k) = rtmp[m];
        i += blockDim.x - 2;
    }
}

__global__ void rhs_gpu_kernel_3(const double* u, double* rsd, const double* qs,
                                 const double* rho_i, const int nx,
                                 const int ny, const int nz) {
    int i, j, k, m, nthreads;
    double q, u31;

    double* flux = (double*)extern_share_data;
    double* utmp = (double*)flux + (blockDim.x * 5);
    double* rtmp = (double*)utmp + (blockDim.x * 5);
    double* rhotmp = (double*)rtmp + (blockDim.x * 5);
    double* u21j = (double*)rhotmp + (blockDim.x);
    double* u31j = (double*)u21j + (blockDim.x);
    double* u41j = (double*)u31j + (blockDim.x);
    double* u51j = (double*)u41j + (blockDim.x);

    k = blockIdx.x + 1;
    i = blockIdx.y + 1;
    j = threadIdx.x;

    using namespace constants_device;
    while (j < ny) {
        nthreads = ny - (j - threadIdx.x);
        if (nthreads > blockDim.x) {
            nthreads = blockDim.x;
        }
        m = threadIdx.x;
        utmp[m] = u(m % 5, i, (j - threadIdx.x) + m / 5, k);
        rtmp[m] = rsd(m % 5, i, (j - threadIdx.x) + m / 5, k);
        m += nthreads;
        utmp[m] = u(m % 5, i, (j - threadIdx.x) + m / 5, k);
        rtmp[m] = rsd(m % 5, i, (j - threadIdx.x) + m / 5, k);
        m += nthreads;
        utmp[m] = u(m % 5, i, (j - threadIdx.x) + m / 5, k);
        rtmp[m] = rsd(m % 5, i, (j - threadIdx.x) + m / 5, k);
        m += nthreads;
        utmp[m] = u(m % 5, i, (j - threadIdx.x) + m / 5, k);
        rtmp[m] = rsd(m % 5, i, (j - threadIdx.x) + m / 5, k);
        m += nthreads;
        utmp[m] = u(m % 5, i, (j - threadIdx.x) + m / 5, k);
        rtmp[m] = rsd(m % 5, i, (j - threadIdx.x) + m / 5, k);
        rhotmp[threadIdx.x] = rho_i(i, j, k);
        __syncthreads();
        /*
         * ---------------------------------------------------------------------
         * eta-direction flux differences
         * ---------------------------------------------------------------------
         */
        flux[threadIdx.x + (0 * blockDim.x)] = utmp[threadIdx.x * 5 + 2];
        u31 = utmp[threadIdx.x * 5 + 2] * rhotmp[threadIdx.x];
        q = qs(i, j, k);
        flux[threadIdx.x + (1 * blockDim.x)] = utmp[threadIdx.x * 5 + 1] * u31;
        flux[threadIdx.x + (2 * blockDim.x)] =
            utmp[threadIdx.x * 5 + 2] * u31 +
            C2 * (utmp[threadIdx.x * 5 + 4] - q);
        flux[threadIdx.x + (3 * blockDim.x)] = utmp[threadIdx.x * 5 + 3] * u31;
        flux[threadIdx.x + (4 * blockDim.x)] =
            (C1 * utmp[threadIdx.x * 5 + 4] - C2 * q) * u31;
        __syncthreads();
        if ((threadIdx.x >= 1) && (threadIdx.x < (blockDim.x - 1)) &&
            (j < (ny - 1))) {
            for (m = 0; m < 5; m++) {
                rtmp[threadIdx.x * 5 + m] =
                    rtmp[threadIdx.x * 5 + m] -
                    ty2 * (flux[(threadIdx.x + 1) + (m * blockDim.x)] -
                           flux[(threadIdx.x - 1) + (m * blockDim.x)]);
            }
        }
        u21j[threadIdx.x] = rhotmp[threadIdx.x] * utmp[threadIdx.x * 5 + 1];
        u31j[threadIdx.x] = rhotmp[threadIdx.x] * utmp[threadIdx.x * 5 + 2];
        u41j[threadIdx.x] = rhotmp[threadIdx.x] * utmp[threadIdx.x * 5 + 3];
        u51j[threadIdx.x] = rhotmp[threadIdx.x] * utmp[threadIdx.x * 5 + 4];
        __syncthreads();
        if (threadIdx.x >= 1) {
            flux[threadIdx.x + (1 * blockDim.x)] =
                ty3 * (u21j[threadIdx.x] - u21j[threadIdx.x - 1]);
            flux[threadIdx.x + (2 * blockDim.x)] =
                (4.0 / 3.0) * ty3 * (u31j[threadIdx.x] - u31j[threadIdx.x - 1]);
            flux[threadIdx.x + (3 * blockDim.x)] =
                ty3 * (u41j[threadIdx.x] - u41j[threadIdx.x - 1]);
            flux[threadIdx.x + (4 * blockDim.x)] =
                0.5 * (1.0 - C1 * C5) * ty3 *
                    ((u21j[threadIdx.x] * u21j[threadIdx.x] +
                      u31j[threadIdx.x] * u31j[threadIdx.x] +
                      u41j[threadIdx.x] * u41j[threadIdx.x]) -
                     (u21j[threadIdx.x - 1] * u21j[threadIdx.x - 1] +
                      u31j[threadIdx.x - 1] * u31j[threadIdx.x - 1] +
                      u41j[threadIdx.x - 1] * u41j[threadIdx.x - 1])) +
                (1.0 / 6.0) * ty3 *
                    (u31j[threadIdx.x] * u31j[threadIdx.x] -
                     u31j[threadIdx.x - 1] * u31j[threadIdx.x - 1]) +
                C1 * C5 * ty3 * (u51j[threadIdx.x] - u51j[threadIdx.x - 1]);
        }
        __syncthreads();
        if ((threadIdx.x >= 1) &&
            (threadIdx.x < (blockDim.x - 1) && (j < (ny - 1)))) {
            rtmp[threadIdx.x * 5 + 0] +=
                dy1 * ty1 *
                (utmp[5 * (threadIdx.x - 1)] - 2.0 * utmp[threadIdx.x * 5 + 0] +
                 utmp[5 * (threadIdx.x + 1)]);
            rtmp[threadIdx.x * 5 + 1] +=
                ty3 * C3 * C4 *
                    (flux[(threadIdx.x + 1) + (1 * blockDim.x)] -
                     flux[threadIdx.x + (1 * blockDim.x)]) +
                dy2 * ty1 *
                    (utmp[5 * threadIdx.x - 4] -
                     2.0 * utmp[threadIdx.x * 5 + 1] +
                     utmp[5 * threadIdx.x + 6]);
            rtmp[threadIdx.x * 5 + 2] +=
                ty3 * C3 * C4 *
                    (flux[(threadIdx.x + 1) + (2 * blockDim.x)] -
                     flux[threadIdx.x + (2 * blockDim.x)]) +
                dy3 * ty1 *
                    (utmp[5 * threadIdx.x - 3] -
                     2.0 * utmp[threadIdx.x * 5 + 2] +
                     utmp[5 * threadIdx.x + 7]);
            rtmp[threadIdx.x * 5 + 3] +=
                ty3 * C3 * C4 *
                    (flux[(threadIdx.x + 1) + (3 * blockDim.x)] -
                     flux[threadIdx.x + (3 * blockDim.x)]) +
                dy4 * ty1 *
                    (utmp[5 * threadIdx.x - 2] -
                     2.0 * utmp[threadIdx.x * 5 + 3] +
                     utmp[5 * threadIdx.x + 8]);
            rtmp[threadIdx.x * 5 + 4] +=
                ty3 * C3 * C4 *
                    (flux[(threadIdx.x + 1) + (4 * blockDim.x)] -
                     flux[threadIdx.x + (4 * blockDim.x)]) +
                dy5 * ty1 *
                    (utmp[5 * threadIdx.x - 1] -
                     2.0 * utmp[threadIdx.x * 5 + 4] +
                     utmp[5 * threadIdx.x + 9]);
            /*
             * ---------------------------------------------------------------------
             * fourth-order dissipation
             * ---------------------------------------------------------------------
             */
            if (j == 1) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp *
                        (5.0 * utmp[threadIdx.x * 5 + m] -
                         4.0 * utmp[5 * threadIdx.x + m + 5] + u(m, i, 3, k));
                }
            }
            if (j == 2) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp *
                        (-4.0 * utmp[threadIdx.x * 5 + m - 5] +
                         6.0 * utmp[threadIdx.x * 5 + m] -
                         4.0 * utmp[threadIdx.x * 5 + m + 5] + u(m, i, 4, k));
                }
            }
            if ((j >= 3) && (j < (ny - 3))) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp * (u(m, i, j - 2, k) -
                                4.0 * utmp[threadIdx.x * 5 + m - 5] +
                                6.0 * utmp[threadIdx.x * 5 + m] -
                                4.0 * utmp[threadIdx.x * 5 + m + 5] +
                                u(m, i, j + 2, k));
                }
            }
            if (j == (ny - 3)) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp * (u(m, i, ny - 5, k) -
                                4.0 * utmp[threadIdx.x * 5 + m - 5] +
                                6.0 * utmp[threadIdx.x * 5 + m] -
                                4.0 * utmp[threadIdx.x * 5 + m + 5]);
                }
            }
            if (j == (ny - 2)) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp * (u(m, i, ny - 4, k) -
                                4.0 * utmp[threadIdx.x * 5 + m - 5] +
                                5.0 * utmp[threadIdx.x * 5 + m]);
                }
            }
        }
        m = threadIdx.x;
        rsd(m % 5, i, (j - threadIdx.x) + m / 5, k) = rtmp[m];
        m += nthreads;
        rsd(m % 5, i, (j - threadIdx.x) + m / 5, k) = rtmp[m];
        m += nthreads;
        rsd(m % 5, i, (j - threadIdx.x) + m / 5, k) = rtmp[m];
        m += nthreads;
        rsd(m % 5, i, (j - threadIdx.x) + m / 5, k) = rtmp[m];
        m += nthreads;
        rsd(m % 5, i, (j - threadIdx.x) + m / 5, k) = rtmp[m];
        j += blockDim.x - 2;
    }
}

__global__ void rhs_gpu_kernel_4(const double* u, double* rsd, const double* qs,
                                 const double* rho_i, const int nx,
                                 const int ny, const int nz) {
    int i, j, k, m, nthreads;
    double q, u41;

    double* flux = (double*)extern_share_data;
    double* utmp = (double*)flux + (blockDim.x * 5);
    double* rtmp = (double*)utmp + (blockDim.x * 5);
    double* rhotmp = (double*)rtmp + (blockDim.x * 5);
    double* u21k = (double*)rhotmp + (blockDim.x);
    double* u31k = (double*)u21k + (blockDim.x);
    double* u41k = (double*)u31k + (blockDim.x);
    double* u51k = (double*)u41k + (blockDim.x);

    j = blockIdx.x + 1;
    i = blockIdx.y + 1;
    k = threadIdx.x;

    using namespace constants_device;
    while (k < nz) {
        nthreads = (nz - (k - threadIdx.x));
        if (nthreads > blockDim.x) {
            nthreads = blockDim.x;
        }
        m = threadIdx.x;
        utmp[m] = u(m % 5, i, j, (k - threadIdx.x) + m / 5);
        rtmp[m] = rsd(m % 5, i, j, (k - threadIdx.x) + m / 5);
        m += nthreads;
        utmp[m] = u(m % 5, i, j, (k - threadIdx.x) + m / 5);
        rtmp[m] = rsd(m % 5, i, j, (k - threadIdx.x) + m / 5);
        m += nthreads;
        utmp[m] = u(m % 5, i, j, (k - threadIdx.x) + m / 5);
        rtmp[m] = rsd(m % 5, i, j, (k - threadIdx.x) + m / 5);
        m += nthreads;
        utmp[m] = u(m % 5, i, j, (k - threadIdx.x) + m / 5);
        rtmp[m] = rsd(m % 5, i, j, (k - threadIdx.x) + m / 5);
        m += nthreads;
        utmp[m] = u(m % 5, i, j, (k - threadIdx.x) + m / 5);
        rtmp[m] = rsd(m % 5, i, j, (k - threadIdx.x) + m / 5);
        rhotmp[threadIdx.x] = rho_i(i, j, k);
        __syncthreads();
        /*
         * ---------------------------------------------------------------------
         * zeta-direction flux differences
         * ---------------------------------------------------------------------
         */
        flux[threadIdx.x + (0 * blockDim.x)] = utmp[threadIdx.x * 5 + 3];
        u41 = utmp[threadIdx.x * 5 + 3] * rhotmp[threadIdx.x];
        q = qs(i, j, k);
        flux[threadIdx.x + (1 * blockDim.x)] = utmp[threadIdx.x * 5 + 1] * u41;
        flux[threadIdx.x + (2 * blockDim.x)] = utmp[threadIdx.x * 5 + 2] * u41;
        flux[threadIdx.x + (3 * blockDim.x)] =
            utmp[threadIdx.x * 5 + 3] * u41 +
            C2 * (utmp[threadIdx.x * 5 + 4] - q);
        flux[threadIdx.x + (4 * blockDim.x)] =
            (C1 * utmp[threadIdx.x * 5 + 4] - C2 * q) * u41;
        __syncthreads();
        if ((threadIdx.x >= 1) && (threadIdx.x < (blockDim.x - 1)) &&
            (k < (nz - 1))) {
            for (m = 0; m < 5; m++) {
                rtmp[threadIdx.x * 5 + m] =
                    rtmp[threadIdx.x * 5 + m] -
                    tz2 * (flux[(threadIdx.x + 1) + (m * blockDim.x)] -
                           flux[(threadIdx.x - 1) + (m * blockDim.x)]);
            }
        }
        u21k[threadIdx.x] = rhotmp[threadIdx.x] * utmp[threadIdx.x * 5 + 1];
        u31k[threadIdx.x] = rhotmp[threadIdx.x] * utmp[threadIdx.x * 5 + 2];
        u41k[threadIdx.x] = rhotmp[threadIdx.x] * utmp[threadIdx.x * 5 + 3];
        u51k[threadIdx.x] = rhotmp[threadIdx.x] * utmp[threadIdx.x * 5 + 4];
        __syncthreads();
        if (threadIdx.x >= 1) {
            flux[threadIdx.x + (1 * blockDim.x)] =
                tz3 * (u21k[threadIdx.x] - u21k[threadIdx.x - 1]);
            flux[threadIdx.x + (2 * blockDim.x)] =
                tz3 * (u31k[threadIdx.x] - u31k[threadIdx.x - 1]);
            flux[threadIdx.x + (3 * blockDim.x)] =
                (4.0 / 3.0) * tz3 * (u41k[threadIdx.x] - u41k[threadIdx.x - 1]);
            flux[threadIdx.x + (4 * blockDim.x)] =
                0.5 * (1.0 - C1 * C5) * tz3 *
                    ((u21k[threadIdx.x] * u21k[threadIdx.x] +
                      u31k[threadIdx.x] * u31k[threadIdx.x] +
                      u41k[threadIdx.x] * u41k[threadIdx.x]) -
                     (u21k[threadIdx.x - 1] * u21k[threadIdx.x - 1] +
                      u31k[threadIdx.x - 1] * u31k[threadIdx.x - 1] +
                      u41k[threadIdx.x - 1] * u41k[threadIdx.x - 1])) +
                (1.0 / 6.0) * tz3 *
                    (u41k[threadIdx.x] * u41k[threadIdx.x] -
                     u41k[threadIdx.x - 1] * u41k[threadIdx.x - 1]) +
                C1 * C5 * tz3 * (u51k[threadIdx.x] - u51k[threadIdx.x - 1]);
        }
        __syncthreads();
        if ((threadIdx.x >= 1) && (threadIdx.x < (blockDim.x - 1)) &&
            (k < (nz - 1))) {
            rtmp[threadIdx.x * 5 + 0] +=
                dz1 * tz1 *
                (utmp[threadIdx.x * 5 - 5] - 2.0 * utmp[threadIdx.x * 5 + 0] +
                 utmp[threadIdx.x * 5 + 5]);
            rtmp[threadIdx.x * 5 + 1] +=
                tz3 * C3 * C4 *
                    (flux[(threadIdx.x + 1) + (1 * blockDim.x)] -
                     flux[threadIdx.x + (1 * blockDim.x)]) +
                dz2 * tz1 *
                    (utmp[5 * threadIdx.x - 4] -
                     2.0 * utmp[threadIdx.x * 5 + 1] +
                     utmp[threadIdx.x * 5 + 6]);
            rtmp[threadIdx.x * 5 + 2] +=
                tz3 * C3 * C4 *
                    (flux[(threadIdx.x + 1) + (2 * blockDim.x)] -
                     flux[threadIdx.x + (2 * blockDim.x)]) +
                dz3 * tz1 *
                    (utmp[5 * threadIdx.x - 3] -
                     2.0 * utmp[threadIdx.x * 5 + 2] +
                     utmp[threadIdx.x * 5 + 7]);
            rtmp[threadIdx.x * 5 + 3] +=
                tz3 * C3 * C4 *
                    (flux[(threadIdx.x + 1) + (3 * blockDim.x)] -
                     flux[threadIdx.x + (3 * blockDim.x)]) +
                dz4 * tz1 *
                    (utmp[5 * threadIdx.x - 2] -
                     2.0 * utmp[threadIdx.x * 5 + 3] +
                     utmp[threadIdx.x * 5 + 8]);
            rtmp[threadIdx.x * 5 + 4] +=
                tz3 * C3 * C4 *
                    (flux[(threadIdx.x + 1) + (4 * blockDim.x)] -
                     flux[threadIdx.x + (4 * blockDim.x)]) +
                dz5 * tz1 *
                    (utmp[5 * threadIdx.x - 1] -
                     2.0 * utmp[threadIdx.x * 5 + 4] +
                     utmp[threadIdx.x * 5 + 9]);
            /*
             * ---------------------------------------------------------------------
             * fourth-order dissipation
             * ---------------------------------------------------------------------
             */
            if (k == 1) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp *
                        (5.0 * utmp[threadIdx.x * 5 + m] -
                         4.0 * utmp[threadIdx.x * 5 + m + 5] + u(m, i, j, 3));
                }
            }
            if (k == 2) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp *
                        (-4.0 * utmp[threadIdx.x * 5 + m - 5] +
                         6.0 * utmp[threadIdx.x * 5 + m] -
                         4.0 * utmp[threadIdx.x * 5 + m + 5] + u(m, i, j, 4));
                }
            }
            if ((k >= 3) && (k < (nz - 3))) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp * (u(m, i, j, k - 2) -
                                4.0 * utmp[threadIdx.x * 5 + m - 5] +
                                6.0 * utmp[threadIdx.x * 5 + m] -
                                4.0 * utmp[threadIdx.x * 5 + m + 5] +
                                u(m, i, j, k + 2));
                }
            }
            if (k == (nz - 3)) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp * (u(m, i, j, nz - 5) -
                                4.0 * utmp[threadIdx.x * 5 + m - 5] +
                                6.0 * utmp[threadIdx.x * 5 + m] -
                                4.0 * utmp[threadIdx.x * 5 + m + 5]);
                }
            }
            if (k == (nz - 2)) {
                for (m = 0; m < 5; m++) {
                    rtmp[threadIdx.x * 5 + m] -=
                        dssp * (u(m, i, j, nz - 4) -
                                4.0 * utmp[threadIdx.x * 5 + m - 5] +
                                5.0 * utmp[threadIdx.x * 5 + m]);
                }
            }
        }
        m = threadIdx.x;
        rsd(m % 5, i, j, (k - threadIdx.x) + m / 5) = rtmp[m];
        m += nthreads;
        rsd(m % 5, i, j, (k - threadIdx.x) + m / 5) = rtmp[m];
        m += nthreads;
        rsd(m % 5, i, j, (k - threadIdx.x) + m / 5) = rtmp[m];
        m += nthreads;
        rsd(m % 5, i, j, (k - threadIdx.x) + m / 5) = rtmp[m];
        m += nthreads;
        rsd(m % 5, i, j, (k - threadIdx.x) + m / 5) = rtmp[m];
        k += blockDim.x - 2;
    }
}

__global__ void ssor_gpu_kernel_1(double* rsd, const int nx, const int ny,
                                  const int nz) {
    int i, j, k;

    if (threadIdx.x >= (nx - 2)) return;

    i = threadIdx.x + 1;
    j = blockIdx.y + 1;
    k = blockIdx.x + 1;

    using namespace constants_device;
    rsd(0, i, j, k) *= dt;
    rsd(1, i, j, k) *= dt;
    rsd(2, i, j, k) *= dt;
    rsd(3, i, j, k) *= dt;
    rsd(4, i, j, k) *= dt;
}

__global__ void ssor_gpu_kernel_2(double* u, double* rsd, const double tmp,
                                  const int nx, const int ny, const int nz) {
    int i, j, k;

    if (threadIdx.x >= (nx - 2)) return;

    i = threadIdx.x + 1;
    j = blockIdx.y + 1;
    k = blockIdx.x + 1;

    u(0, i, j, k) += tmp * rsd(0, i, j, k);
    u(1, i, j, k) += tmp * rsd(1, i, j, k);
    u(2, i, j, k) += tmp * rsd(2, i, j, k);
    u(3, i, j, k) += tmp * rsd(3, i, j, k);
    u(4, i, j, k) += tmp * rsd(4, i, j, k);
}
