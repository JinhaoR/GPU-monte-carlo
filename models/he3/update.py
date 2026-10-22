"""Cartesian Metropolis proposals for the six unconstrained complex fields."""

import math

from numba import cuda, complex64, float32
from numba.cuda.random import xoroshiro128p_uniform_float32


@cuda.jit(device=True)
def norm2(z):
    return z.real * z.real + z.imag * z.imag


@cuda.jit(device=True)
def potential(psi, betas):
    """-2 n + beta_1 I_1 + ... + beta_5 I_5, using orbital contractions."""
    nx, ny = norm2(psi[0, 0]), norm2(psi[0, 1])
    xx, yy = psi[0, 0] * psi[0, 0], psi[0, 1] * psi[0, 1]
    xy = psi[0, 0] * psi[0, 1]
    overlap = psi[0, 0].conjugate() * psi[0, 1]
    for mu in range(1, 3):
        x, y = psi[mu, 0], psi[mu, 1]
        nx += norm2(x)
        ny += norm2(y)
        xx += x * x
        yy += y * y
        xy += x * y
        overlap += x.conjugate() * y
    n, two = nx + ny, float32(2)
    i1 = norm2(xx + yy)
    i2 = n * n
    i3 = nx * nx + ny * ny + two * (overlap * overlap).real
    i4 = nx * nx + ny * ny + two * norm2(overlap)
    i5 = norm2(xx) + norm2(yy) + two * norm2(xy)
    return (-two * n + betas[0] * i1 + betas[1] * i2
            + betas[2] * i3 + betas[3] * i4 + betas[4] * i5)


@cuda.jit(device=True)
def gradient(dx_x, dx_y, dy_x, dy_y):
    """The three forward-difference gradient contractions for one spin row."""
    return (float32(3) * (norm2(dx_x) + norm2(dy_y))
            + norm2(dx_y) + norm2(dy_x)
            + float32(2) * (dx_x.conjugate() * dy_y
                            + dx_y.conjugate() * dy_x).real)


@cuda.jit(device=True)
def local_energy(state, r, i, j, psi, betas, K):
    """Onsite energy and the three gradient cells affected by this site."""
    L = state.shape[1]
    ip, im, jp, jm = (i + 1) % L, (i - 1) % L, (j + 1) % L, (j - 1) % L
    g = float32(0)
    for mu in range(3):
        x, y = psi[mu, 0], psi[mu, 1]
        left_x, left_y = state[r, im, j, mu, 0], state[r, im, j, mu, 1]
        down_x, down_y = state[r, i, jm, mu, 0], state[r, i, jm, mu, 1]
        g += gradient(state[r, ip, j, mu, 0] - x, state[r, ip, j, mu, 1] - y,
                      state[r, i, jp, mu, 0] - x, state[r, i, jp, mu, 1] - y)
        g += gradient(x - left_x, y - left_y,
                      state[r, im, jp, mu, 0] - left_x,
                      state[r, im, jp, mu, 1] - left_y)
        g += gradient(state[r, ip, jm, mu, 0] - down_x,
                      state[r, ip, jm, mu, 1] - down_y,
                      x - down_x, y - down_y)
    return potential(psi, betas) + K * g


@cuda.jit
def metropolis_color(state, rng_states, inverse_temperatures, color, betas, K, delta):
    col, row, r = cuda.grid(3)
    L = state.shape[1]
    i, j = 2 * row + color // 2, 2 * col + color % 2
    if r < state.shape[0] and i < L and j < L:
        index = r * L * L + i * L + j
        psi = cuda.local.array((3, 2), dtype=complex64)
        for mu in range(3):
            for a in range(2):
                psi[mu, a] = state[r, i, j, mu, a]
        old_energy = local_energy(state, r, i, j, psi, betas, K)
        # A symmetric box proposal in all twelve Cartesian coordinates.
        for mu in range(3):
            for a in range(2):
                re = delta * (float32(2) * xoroshiro128p_uniform_float32(rng_states, index) - float32(1))
                im = delta * (float32(2) * xoroshiro128p_uniform_float32(rng_states, index) - float32(1))
                psi[mu, a] = complex64(complex(psi[mu, a].real + re, psi[mu, a].imag + im))
        difference = local_energy(state, r, i, j, psi, betas, K) - old_energy
        u = xoroshiro128p_uniform_float32(rng_states, index)
        if difference <= 0 or u < math.exp(-inverse_temperatures[r] * difference):
            for mu in range(3):
                for a in range(2):
                    state[r, i, j, mu, a] = psi[mu, a]
