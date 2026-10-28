"""Float32 conditional Gaussian proposals with full-Hamiltonian MH correction.

This retains GaussianUpdater as the double-precision experimental reference.
The reference precision is A=[[8K+mass,2K],[2K,8K+mass]]. Its fixed-neighbor
mean is approximated in float32, then a correlated Gaussian autoregression
with coefficient rho proposes every Cartesian field component. Unlike a
quartic-only correction, the final acceptance explicitly includes the full
original Hamiltonian change and the reverse/forward Gaussian density ratio.
Consequently an approximate neighbor mean does not change the target in
exact arithmetic. Four-color conditional decisions commute.

Both Box-Muller normals are used. All proposal, transcendental and MH
arithmetic is float32, matching the baseline's numerical precision. Finite
rounding is not a bitwise stationarity proof on floating-point field states.
"""

import math

import numpy as np
from numba import complex64, cuda, float32
from numba.cuda.random import xoroshiro128p_uniform_float32

from .update import gradient_difference, norm2, potential


@cuda.jit(device=True)
def neighbor_field_float32(state, r, i, j, field):
    L = state.shape[1]
    ip, im = (i+1)%L, (i-1)%L
    jp, jm = (j+1)%L, (j-1)%L
    three = float32(3)
    for mu in range(3):
        rx, ry = state[r, ip, j, mu, 0], state[r, ip, j, mu, 1]
        ux, uy = state[r, i, jp, mu, 0], state[r, i, jp, mu, 1]
        lx, ly = state[r, im, j, mu, 0], state[r, im, j, mu, 1]
        dx, dy = state[r, i, jm, mu, 0], state[r, i, jm, mu, 1]
        lux, luy = state[r, im, jp, mu, 0], state[r, im, jp, mu, 1]
        drx, dry = state[r, ip, jm, mu, 0], state[r, ip, jm, mu, 1]
        field[mu, 0] = (-three*(rx+lx)-ux-uy-ry-ly-dx-dy+luy+dry)
        field[mu, 1] = (-rx-ry-ux-lx-ly-dx-three*(uy+dy)+lux+drx)

@cuda.jit(device=True)
def _normal_pair(rng, index):
    u = xoroshiro128p_uniform_float32(rng, index)
    # Direct U avoids 1-U becoming zero when a float32 uniform rounds to 1.
    if u <= float32(0):
        u = float32(1.1102230246251565e-16)
    angle = float32(6.283185307179586) * xoroshiro128p_uniform_float32(rng, index)
    radius = math.sqrt(-float32(2) * math.log(u))
    return complex64(complex(radius * math.cos(angle), radius * math.sin(angle)))


@cuda.jit(device=True)
def gaussian_proposal_float32(psi, field, beta, K, rho, mass, rng, index):
    """Overwrite psi with the proposal and return log q(reverse)/q(forward).

    The direct Cartesian AR keeps coefficient rho despite the rounded
    plus/minus transform. Density coefficients use the actual sigma and
    transform constants used to generate the noise. The original psi is
    retained in scalar temporaries until its density contribution is known.
    """
    one, two = float32(1), float32(2)
    root = float32(0.7071067811865475)
    plus, minus = float32(10) * K + mass, float32(6) * K + mass
    variance_factor = one - rho * rho
    sigma_plus = math.sqrt(variance_factor / (two * beta * plus))
    sigma_minus = math.sqrt(variance_factor / (two * beta * minus))
    # u=x+y and v=x-y have noise variances (2*root*sigma)^2 per real
    # component. For a Gaussian AR, its reversible precision is
    # (1-rho^2)/(2*noise_variance) in the log density.
    q_plus = variance_factor / (float32(8) * root * root * sigma_plus * sigma_plus)
    q_minus = variance_factor / (float32(8) * root * root * sigma_minus * sigma_minus)
    log_ratio = float32(0)
    for mu in range(3):
        x, y = psi[mu, 0], psi[mu, 1]
        hp = (field[mu, 0] + field[mu, 1]) * root
        hm = (field[mu, 0] - field[mu, 1]) * root
        mp, mm = -K * hp / plus, -K * hm / minus
        mx, my = (mp + mm) * root, (mp - mm) * root
        zp, zm = sigma_plus * _normal_pair(rng, index), sigma_minus * _normal_pair(rng, index)
        psi[mu, 0] = mx + rho * (x - mx) + root * (zp + zm)
        psi[mu, 1] = my + rho * (y - my) + root * (zp - zm)
        old_plus, old_minus = (x - mx) + (y - my), (x - mx) - (y - my)
        new_plus = (psi[mu, 0] - mx) + (psi[mu, 1] - my)
        new_minus = (psi[mu, 0] - mx) - (psi[mu, 1] - my)
        log_ratio += q_plus * (norm2(new_plus) - norm2(old_plus))
        log_ratio += q_minus * (norm2(new_minus) - norm2(old_minus))
    return log_ratio


@cuda.jit
def fast_gaussian_color(state, rng, inverse_temperatures, color, coefficients, K, rho, mass, counts, g_d):
    col, row, r = cuda.grid(3)
    L = state.shape[1]
    i, j = 2 * row + color // 2, 2 * col + color % 2
    if r >= state.shape[0] or i >= L or j >= L:
        return
    index = r * L * L + i * L + j
    psi = cuda.local.array((3, 2), dtype=complex64)
    field = cuda.local.array((3, 2), dtype=complex64)
    neighbor_field_float32(state, r, i, j, field)
    for mu in range(3):
        for a in range(2):
            psi[mu, a] = state[r, i, j, mu, a]
    old_potential = potential(psi, coefficients, g_d)
    log_proposal_ratio = gaussian_proposal_float32(
        psi, field, inverse_temperatures[r], K, rho, mass, rng, index)
    delta_h = (potential(psi, coefficients, g_d) - old_potential
               + K * gradient_difference(state, r, i, j, psi))
    log_ratio = -inverse_temperatures[r] * delta_h + log_proposal_ratio
    u = xoroshiro128p_uniform_float32(rng, index)
    counts[0, r, i, j] += 1
    if not math.isfinite(log_ratio):
        counts[2, r, i, j] += 1
    elif log_ratio >= float32(0) or u < math.exp(log_ratio):
        counts[1, r, i, j] += 1
        for mu in range(3):
            for a in range(2):
                state[r, i, j, mu, a] = psi[mu, a]


class FastGaussianUpdater:
    """A faster, baseline-precision alternative; step() replaces one sweep."""

    def __init__(self, runtime, rho=0.0, mass=0.0):
        if len(runtime.state.shape) != 5 or runtime.state.shape[-2:] != (3, 2):
            raise ValueError("FastGaussianUpdater requires full complex 3x2 He3 fields")
        rho, mass = np.float32(rho), np.float32(mass)
        eigenvalues = np.asarray([6, 10], dtype=np.float32) * runtime.model.K + mass
        if not np.isfinite(mass) or not np.all(np.isfinite(eigenvalues) & (eigenvalues > 0)):
            raise ValueError("Conditional quadratic Gaussian requires positive finite float32 orbital eigenvalues")
        if not np.isfinite(rho) or not -1 < rho < 1:
            raise ValueError("Require -1 < float32(rho) < 1 for nondegenerate proposals")
        self.runtime, self.rho, self.mass = runtime, rho, mass
        self.counts = cuda.to_device(np.zeros((3, runtime.R, runtime.L, runtime.L), dtype=np.int64))
        self.sweeps = 0

    def step(self):
        rt = self.runtime
        rt.invalidate_energy_cache()
        for color in range(4):
            fast_gaussian_color[rt.blocks_per_grid, rt.threads_per_block](
                rt.state, rt.rng_states, rt.betas, color, rt.model.betas,
                rt.model.K, self.rho, self.mass, self.counts, rt.model.g_d)
        self.sweeps += 1

    def diagnostics(self):
        counts = self.counts.copy_to_host().sum(axis=(2, 3))
        return {
            "algorithm": "float32_conditional_gaussian_full_hamiltonian_mh",
            "sweeps_including_compilation": self.sweeps,
            "rho": float(self.rho), "proposal_mass": float(self.mass),
            "attempts": counts[0].tolist(), "accepted": counts[1].tolist(),
            "nonfinite_rejections": counts[2].tolist(),
            "acceptance_rates": np.divide(counts[1], counts[0], out=np.zeros(self.runtime.R), where=counts[0] > 0).tolist(),
        }
