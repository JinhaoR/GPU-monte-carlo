"""Weighted Weber–Minnhagen fit using SciPy least squares."""

import numpy as np
from scipy.optimize import least_squares


def fit_weber_minnhagen(T, L, helicity, errors, *, profile=True):
    """Fit Tc and the leading logarithmic correction; arrays have shape (size, T)."""
    T, L = np.asarray(T), np.asarray(L)
    rho, sigma = np.asarray(helicity), np.asarray(errors)
    log_sizes = 2 * np.log(L / L[0])

    def residual(params):
        Tc, a = params                    # a = 1 / (2 log(L_min) + C)
        measured = np.array([np.interp(Tc, T, row) for row in rho])
        uncertainty = np.array([np.interp(Tc, T, row) for row in sigma])
        predicted = 2 * Tc / np.pi * (1 + a / (1 + a * log_sizes))
        return (measured - predicted) / uncertainty

    # A few starts avoid depending on one initial temperature guess.
    fits = [least_squares(residual, [t, 0.1],
                          bounds=([T[0], 0], [T[-1], np.inf]))
            for t in np.linspace(T[0], T[-1], 5)]
    best = min(fits, key=lambda fit: np.dot(fit.fun, fit.fun))
    Tc, a = best.x
    chi2 = float(np.dot(best.fun, best.fun))
    at_boundary = (abs(Tc - T[0]) < 1e-6 * (T[-1] - T[0])
                   or abs(Tc - T[-1]) < 1e-6 * (T[-1] - T[0]))
    result = {
        "temperature": float(Tc) if not at_boundary and a > 1e-8 else None,
        "C": float(1 / a - 2 * np.log(L[0])) if a > 1e-8 else None,
        "chi2": chi2,
        "min_size": int(L[0]),
        "degrees_of_freedom": len(L) - 2,
        "reason": "boundary optimum" if at_boundary else "vanishing correction" if a <= 1e-8 else None,
    }

    if profile:
        # For the fit-quality plot, optimize only a at each fixed temperature.
        scores = []
        for t in T:
            fit = least_squares(lambda p: residual((t, p[0])), [max(a, 1e-4)],
                                bounds=(0, np.inf))
            scores.append(float(np.dot(fit.fun, fit.fun)))
        result["profile_temperatures"] = T
        result["profile_chi2"] = np.asarray(scores)

    return result
