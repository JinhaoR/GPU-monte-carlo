"""Weighted Weber–Minnhagen fit using SciPy least squares."""

import numpy as np
from scipy.optimize import least_squares


def fit_weber_minnhagen(T, L, helicity, errors, *, profile=True, min_size=None, temperature_range=None):
    """Fit Tc and the leading logarithmic correction; arrays have shape (size, T)."""
    T, L = np.asarray(T, float), np.asarray(L, float)
    rho, sigma = np.asarray(helicity, float), np.asarray(errors, float)
    def invalid(reason):
        return dict(temperature=None, C=None, chi2=None, degrees_of_freedom=len(L)-2,
                    min_size=int(min(L)) if len(L) else None, reason=reason)
    if T.ndim != 1 or L.ndim != 1 or rho.shape != (len(L),len(T)) or sigma.shape != rho.shape:
        return invalid('invalid shapes')
    sizes = np.argsort(L)
    if min_size is not None:
        sizes = sizes[L[sizes] >= min_size]
    times = np.argsort(T)
    if temperature_range is not None:
        times = times[(T[times]>=temperature_range[0]) & (T[times]<=temperature_range[1])]
    T,L,rho,sigma = T[times],L[sizes],rho[np.ix_(sizes,times)],sigma[np.ix_(sizes,times)]
    if len(L) < 3:
        return invalid('at least three sizes required')
    if len(T) < 3:
        return invalid('at least three temperatures required')
    if (not np.all(np.isfinite(T) & (T>0)) or not np.all(np.isfinite(L) & (L>0))
            or not np.all(np.diff(T)>0) or not np.all(np.diff(L)>0)):
        return invalid('distinct positive finite sizes/temperatures required')
    if not np.all(np.isfinite(rho) & np.isfinite(sigma) & (sigma>0)):
        return invalid('positive finite errors and finite helicity required')
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
    scores_at_optima = np.array([np.dot(fit.fun,fit.fun) for fit in fits])
    flat = (np.ptp([fit.x[0] for fit in fits]) > .1*np.ptp(T)
            and np.ptp(scores_at_optima) < 1e-9*max(1.,chi2))
    at_boundary = (abs(Tc - T[0]) < 1e-6 * (T[-1] - T[0])
                   or abs(Tc - T[-1]) < 1e-6 * (T[-1] - T[0]))
    reason = ('flat or indistinguishable temperature minima' if flat else
              'boundary optimum' if at_boundary else
              'infinite C (vanishing correction)' if a <= 1e-6 else None)
    result = {
        "temperature": float(Tc) if reason is None else None,
        "C": float(1 / a - 2 * np.log(L[0])) if a > 1e-6 else None,
        "chi2": chi2,
        "min_size": int(L[0]),
        "degrees_of_freedom": len(L) - 2,
        "reason": reason,
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


def fit_pseudocritical_bkt(sizes, temperatures, errors, *, L0=None):
    """Conditional T_L=T_BKT+A/log(L/L0)^2 fit, not a BKT classifier.

    Supply a fixed L0 to study its systematic sensitivity. A free L0 needs at
    least four sizes (three sizes would fit three parameters with zero dof).
    Statistical errors must come from refitting jointly resampled T_L values.
    """
    L, T, error = map(lambda a: np.asarray(a, float), (sizes, temperatures, errors))
    parameters = 3 if L0 is None else 2
    if (L.ndim != 1 or T.shape != L.shape or error.shape != L.shape
            or len(L) <= parameters or len(np.unique(L)) != len(L)
            or not np.all(np.isfinite(L) & np.isfinite(T) & np.isfinite(error) & (L > 0) & (error > 0))):
        raise ValueError('Need distinct sizes, positive errors and positive fit degrees of freedom')
    if L0 is not None and not 0 < L0 < min(L):
        raise ValueError('Fixed L0 must lie between zero and the smallest size')
    def prediction(p):
        scale = np.exp(p[2]) if L0 is None else L0
        return p[0]+p[1]/np.log(L/scale)**2
    def residual(p):
        return (prediction(p)-T)/error
    lower, upper = ([-np.inf, 0.], [np.inf, np.inf])
    starts = [[min(T)-.001, .01]]
    if L0 is None:
        lower += [np.log(.01)]
        upper += [np.log(.95*min(L))]
        starts = [starts[0]+[np.log(scale)] for scale in (.1, 1., .2*min(L))]
    fits = [least_squares(residual, start, bounds=(lower, upper), max_nfev=3000, x_scale='jac') for start in starts]
    best = min(fits, key=lambda fit: np.dot(fit.fun, fit.fun))
    boundary = bool(np.any(best.active_mask))
    condition = float(np.linalg.cond(best.jac))
    valid = bool(best.success and not boundary and condition < 1e10)
    return dict(temperature=float(best.x[0]) if valid else None, formal_temperature=float(best.x[0]),
                A=float(best.x[1]), L0=float(np.exp(best.x[2])) if L0 is None else float(L0),
                chi2=float(np.dot(best.fun, best.fun)), degrees_of_freedom=len(L)-parameters,
                condition=condition, boundary=boundary, fitted=prediction(best.x),
                reason=None if valid else 'boundary, nonconvergence or ill-conditioned fit')
