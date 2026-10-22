import math

from numba import cuda, float32
from numba.cuda.random import xoroshiro128p_uniform_float32


@cuda.jit(device=True)
def local_energy(state, r, i, j, theta, tau, dx, dy, dz, J):
    """Four bonds touching this site, with a proposed local state."""
    L = state.shape[1]
    energy = float32(0.0)
    for ni, nj in (((i - 1) % L, j), ((i + 1) % L, j),
                   (i, (j - 1) % L), (i, (j + 1) % L)):
        dot = (dx * state[r, ni, nj, 2] + dy * state[r, ni, nj, 3]
               + dz * state[r, ni, nj, 4])
        energy -= J * (float32(1.0) + tau * state[r, ni, nj, 1]) * dot * math.cos(
            theta - state[r, ni, nj, 0])
    return energy


@cuda.jit
def metropolis_color(state, rng_states, betas, color, J, delta):
    col, row, r = cuda.grid(3)
    L = state.shape[1]
    i, j = 2 * row + color // 2, 2 * col + color % 2
    if r < state.shape[0] and i < L and j < L:
        one, two = float32(1.0), float32(2.0)
        pi, two_pi = float32(math.pi), float32(2 * math.pi)
        index = r * L * L + i * L + j
        theta, tau = state[r, i, j, 0], state[r, i, j, 1]
        dx, dy, dz = state[r, i, j, 2], state[r, i, j, 3], state[r, i, j, 4]
        energy = local_energy(state, r, i, j, theta, tau, dx, dy, dz, J)

        # Each proposal is symmetric, so ordinary Metropolis applies.
        proposed_theta = theta + delta * (two * xoroshiro128p_uniform_float32(rng_states, index) - one)
        if proposed_theta >= pi:
            proposed_theta -= two_pi
        elif proposed_theta < -pi:
            proposed_theta += two_pi
        proposed_energy = local_energy(state, r, i, j, proposed_theta, tau, dx, dy, dz, J)
        dE = proposed_energy - energy
        u = xoroshiro128p_uniform_float32(rng_states, index)
        if dE <= 0 or u < math.exp(-betas[r] * dE):
            theta, energy = proposed_theta, proposed_energy

        proposed_tau = one if xoroshiro128p_uniform_float32(rng_states, index) < float32(0.5) else -one
        proposed_energy = local_energy(state, r, i, j, theta, proposed_tau, dx, dy, dz, J)
        dE = proposed_energy - energy
        u = xoroshiro128p_uniform_float32(rng_states, index)
        if dE <= 0 or u < math.exp(-betas[r] * dE):
            tau, energy = proposed_tau, proposed_energy

        # Uniform sphere proposal: z uniform in [-1,1], azimuth uniform in [0,2pi).
        z = two * xoroshiro128p_uniform_float32(rng_states, index) - one
        phi = two_pi * xoroshiro128p_uniform_float32(rng_states, index)
        radius = math.sqrt(one - z * z)
        x, y = radius * math.cos(phi), radius * math.sin(phi)
        proposed_energy = local_energy(state, r, i, j, theta, tau, x, y, z, J)
        dE = proposed_energy - energy
        u = xoroshiro128p_uniform_float32(rng_states, index)
        if dE <= 0 or u < math.exp(-betas[r] * dE):
            dx, dy, dz = x, y, z

        state[r, i, j, 0], state[r, i, j, 1] = theta, tau
        state[r, i, j, 2], state[r, i, j, 3], state[r, i, j, 4] = dx, dy, dz
