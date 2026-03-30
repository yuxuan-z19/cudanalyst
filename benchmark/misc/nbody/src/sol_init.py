import numpy as np
from numba import jit

eps_2 = np.float32(1e-6)
zero = np.float32(0.0)
one = np.float32(1.0)


@jit
def run_custom(positions: np.ndarray, weights: np.ndarray):
    accelerations = np.zeros_like(positions)
    n = weights.shape[0]
    for i in range(n):
        ax = zero
        ay = zero
        for j in range(n):
            rx = positions[j, 0] - positions[i, 0]
            ry = positions[j, 1] - positions[i, 1]
            sqr_dist = rx * rx + ry * ry + eps_2
            sixth_dist = sqr_dist * sqr_dist * sqr_dist
            inv_dist_cube = one / np.sqrt(sixth_dist)
            s = weights[j] * inv_dist_cube
            ax += s * rx
            ay += s * ry
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay
    return accelerations
