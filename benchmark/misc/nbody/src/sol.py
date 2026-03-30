import numpy as np
from numba import jit

eps_2 = np.float32(1e-6)
zero = np.float32(0.0)
one = np.float32(1.0)


@jit
def run_custom(positions: np.ndarray, weights: np.ndarray):
    # TODO:
    pass
