import math

from numba import cuda, float32
from numba.cuda.random import xoroshiro128p_uniform_float32

@cuda.jit
def metropolis_color(state, rng_states, betas, color, J, delta):
    col, row, r = cuda.grid(3)

    one = float32(1.0)
    two = float32(2.0)
    pi = float32(math.pi)
    two_pi = float32(2.0 * math.pi)
    zero = float32(0.0)

    L = state.shape[1]
    i = 2 * row + color // 2
    j = 2 * col + color % 2

    if r < state.shape[0] and i < L and j < L:
        theta = state[r, i, j]
        rng_index = r * L * L + i * L + j

        u = xoroshiro128p_uniform_float32(rng_states, rng_index)
        proposed = theta + delta * (two * u - one)

        # Delta <= pi and theta is in [-pi, pi), so one wrap suffices.
        if proposed >= pi:
            proposed -= two_pi
        elif proposed < -pi:
            proposed += two_pi

        up = state[r, (i - 1) % L, j]
        down = state[r, (i + 1) % L, j]
        left = state[r, i, (j - 1) % L]
        right = state[r, i, (j + 1) % L]

        e_old = -J * (
            math.cos(theta - up)
            + math.cos(theta - down)
            + math.cos(theta - left)
            + math.cos(theta - right)
        )
        e_new = -J * (
            math.cos(proposed - up)
            + math.cos(proposed - down)
            + math.cos(proposed - left)
            + math.cos(proposed - right)
        )
        dE = e_new - e_old

        v = xoroshiro128p_uniform_float32(rng_states, rng_index)
        if dE <= zero or v < math.exp(-betas[r] * dE):
            state[r, i, j] = proposed

