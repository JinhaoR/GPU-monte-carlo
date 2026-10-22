
import numpy as np


def make_temperature_ladder(
    T_min,
    T_max,
    n_T,
    Tc=None,
    tc_window=0.08,
    tc_fraction=0.60,
):
    """
    Temperature ladder constructed in inverse temperature.

    Tc=None: uniform beta spacing.

    Otherwise:
        tc_window: full temperature width centered on Tc.
        tc_fraction: target fraction of ladder density
                     inside that window (0 to 1).
    """
    if not (0 < T_min < T_max):
        raise ValueError("Require 0 < T_min < T_max")

    if n_T < 2:
        raise ValueError("Need at least two temperatures")

    # Always work in inverse temperature
    beta_min = 1.0 / T_max
    beta_max = 1.0 / T_min

    # Ordinary uniform beta ladder
    if Tc is None:
        betas = np.linspace(beta_min, beta_max, n_T)
        return (1.0 / betas)[::-1]

    # Focus window in temperature
    T_lo = Tc - tc_window / 2
    T_hi = Tc + tc_window / 2

    if not T_min < T_lo < T_hi < T_max:
        raise ValueError("Focus window must lie inside range")

    if not 0 < tc_fraction < 1:
        raise ValueError("tc_fraction must be between 0 and 1")

    # Convert focus window to beta coordinates
    b_lo = 1.0 / T_hi
    b_hi = 1.0 / T_lo

    beta = np.linspace(beta_min, beta_max, 4097)

    # Smooth density enhancement near Tc
    shoulder = (b_hi - b_lo) / 8

    profile = 0.5 * (
        np.tanh((beta - b_lo) / shoulder)
        - np.tanh((beta - b_hi) / shoulder)
    )

    # Calculate strength required for requested focus fraction
    span = beta_max - beta_min
    width = b_hi - b_lo

    total = np.trapezoid(profile, beta)

    focus_grid = np.r_[
        b_lo,
        beta[(beta > b_lo) & (beta < b_hi)],
        b_hi,
    ]
    inside = np.trapezoid(
        np.interp(focus_grid, beta, profile),
        focus_grid,
    )

    numerator = tc_fraction * span - width
    denominator = inside - tc_fraction * total

    if numerator < -1e-12 or denominator <= 0:
        raise ValueError(
            "Requested focus fraction is not achievable"
        )

    amplitude = max(0.0, numerator / denominator)
    density = 1.0 + amplitude * profile

    # Cumulative density
    cdf = np.r_[
        0.0,
        np.cumsum(
            0.5 * (density[1:] + density[:-1])
            * np.diff(beta)
        ),
    ]
    cdf /= cdf[-1]

    # Equally spaced quantiles of the density
    betas = np.interp(
        np.linspace(0, 1, n_T),
        cdf,
        beta,
    )
    return (1.0 / betas)[::-1]
