"""Crossings of piecewise-linear curves."""

import numpy as np

def linear_crossings(T, y1, y2=0.0, *, temperature_range=None):
    T = np.asarray(T)
    d = np.broadcast_to(np.asarray(y1) - y2, T.shape)
    if not np.all(np.isfinite(d)):
        return [np.nan]
    if np.any((d[:-1] == 0) & (d[1:] == 0)):
        return [np.nan]  # A whole interval of roots is not a unique crossing.

    roots = list(T[d == 0])
    indices = np.where(d[:-1] * d[1:] < 0)[0]
    for i in indices:
        roots.append(T[i] - d[i] * (T[i + 1] - T[i]) / (d[i + 1] - d[i]))

    if temperature_range is not None:
        low, high = temperature_range
        roots = [t for t in roots if low <= t <= high]
    return sorted(float(t) for t in roots)


def bootstrap_crossings(T, left, right=0.0, *, temperature_range=None):
    left, right = np.broadcast_arrays(left, right)
    estimates = np.full(len(left), np.nan)
    for b, (a, c) in enumerate(zip(left, right)):
        roots = linear_crossings(T, a, c, temperature_range=temperature_range)
        if len(roots) == 1:
            estimates[b] = roots[0]
    return estimates
