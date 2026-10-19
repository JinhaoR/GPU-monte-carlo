import numpy as np

def binder_cumulant(m2_samples, gaussian_factor):
    """
    m2_samples: M² for every sample and replica.
                Shape (S, R).

    gaussian_factor:
        3 for scalar Ising magnetization
        2 for two-component XY magnetization
    """
    m2 = np.asarray(m2_samples, dtype=np.float64)

    mean_m2 = np.mean(m2, axis=0)
    mean_m4 = np.mean(m2**2, axis=0)

    return 1 - mean_m4 / (gaussian_factor * mean_m2**2)
