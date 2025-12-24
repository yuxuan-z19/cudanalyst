#include <cuda.h>

#include "ep.cuh"

__global__ void gpu_kernel(double* q_global, double* sx_global,
                           double* sy_global, double an) {
    double x_local[2 * RECOMPUTATION];
    double q_local[NQ];
    double sx_local, sy_local;
    double t1, t2, t3, t4, x1, x2, seed;
    int i, ii, ik, kk, l;

    q_local[0] = 0.0;
    q_local[1] = 0.0;
    q_local[2] = 0.0;
    q_local[3] = 0.0;
    q_local[4] = 0.0;
    q_local[5] = 0.0;
    q_local[6] = 0.0;
    q_local[7] = 0.0;
    q_local[8] = 0.0;
    q_local[9] = 0.0;
    sx_local = 0.0;
    sy_local = 0.0;

    kk = blockIdx.x * blockDim.x + threadIdx.x;
    if (kk >= NN) return;

    t1 = S;
    t2 = an;

    /* find starting seed t1 for this kk */
    for (i = 1; i <= 100; i++) {
        ik = kk / 2;
        if ((2 * ik) != kk) t3 = randlc_device(&t1, t2);
        if (ik == 0) break;
        t3 = randlc_device(&t2, t2);
        kk = ik;
    }

    seed = t1;
    for (ii = 0; ii < NK; ii = ii + RECOMPUTATION) {
        /* compute uniform pseudorandom numbers */
        vranlc_device(2 * RECOMPUTATION, &seed, A, x_local);

        /*
         * compute gaussian deviates by acceptance-rejection method and
         * tally counts in concentric square annuli. this loop is not
         * vectorizable.
         */
        for (i = 0; i < RECOMPUTATION; i++) {
            x1 = 2.0 * x_local[2 * i] - 1.0;
            x2 = 2.0 * x_local[2 * i + 1] - 1.0;
            t1 = x1 * x1 + x2 * x2;
            if (t1 <= 1.0) {
                t2 = sqrt(-2.0 * log(t1) / t1);
                t3 = (x1 * t2);
                t4 = (x2 * t2);
                l = max(fabs(t3), fabs(t4));
                q_local[l] += 1.0;
                sx_local += t3;
                sy_local += t4;
            }
        }
    }

    atomicAdd(q_global + blockIdx.x * NQ + 0, q_local[0]);
    atomicAdd(q_global + blockIdx.x * NQ + 1, q_local[1]);
    atomicAdd(q_global + blockIdx.x * NQ + 2, q_local[2]);
    atomicAdd(q_global + blockIdx.x * NQ + 3, q_local[3]);
    atomicAdd(q_global + blockIdx.x * NQ + 4, q_local[4]);
    atomicAdd(q_global + blockIdx.x * NQ + 5, q_local[5]);
    atomicAdd(q_global + blockIdx.x * NQ + 6, q_local[6]);
    atomicAdd(q_global + blockIdx.x * NQ + 7, q_local[7]);
    atomicAdd(q_global + blockIdx.x * NQ + 8, q_local[8]);
    atomicAdd(q_global + blockIdx.x * NQ + 9, q_local[9]);
    atomicAdd(sx_global + blockIdx.x, sx_local);
    atomicAdd(sy_global + blockIdx.x, sy_local);
}
