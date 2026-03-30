from collections.abc import Callable

import numpy as np

eps_2 = np.float32(1e-6)
zero = np.float32(0.0)
one = np.float32(1.0)


def run_golden(positions: np.ndarray, weights: np.ndarray):
    accelerations = np.zeros_like(positions)
    n = weights.size
    for j in range(n):
        r: np.ndarray = positions[j] - positions
        rx = r[:, 0]
        ry = r[:, 1]
        sqr_dist = rx * rx + ry * ry + eps_2
        sixth_dist = sqr_dist * sqr_dist * sqr_dist
        inv_dist_cube = one / np.sqrt(sixth_dist)
        s = weights[j] * inv_dist_cube
        accelerations += (r.transpose() * s).transpose()
    return accelerations


def make_nbody_samples(n_bodies: int):
    positions = np.random.RandomState(0).uniform(-1.0, 1.0, (n_bodies, 2))
    weights = np.random.RandomState(0).uniform(1.0, 2.0, n_bodies)
    return positions.astype(np.float32), weights.astype(np.float32)


def sanity(run_exec: Callable):
    try:
        p, w = make_nbody_samples(10)
        golden_res = run_golden(p, w)
        custom_res = run_exec(p, w)
        if not np.allclose(golden_res, custom_res, 1e-4):
            return False, f"Value error"
        return True, ""
    except Exception as e:
        return False, f"Exception during execution: {str(e)}"
