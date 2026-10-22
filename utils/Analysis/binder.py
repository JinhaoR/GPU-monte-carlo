"""Binder cumulant from samples of squared magnetization."""

import numpy as np


def binder_cumulant(m2_samples, gaussian_factor):
    return 1 - np.mean(m2_samples**2, axis=0) / (gaussian_factor * np.mean(m2_samples, axis=0)**2)