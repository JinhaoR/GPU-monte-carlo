import math

from numba import cuda, float32
from numba.cuda.random import xoroshiro128p_uniform_float32


@cuda.jit
def metropolis_color(spins, rng_states, betas, color, J):
    col, row, r = cuda.grid(3)

    L = spins.shape[1]
    i = 2 * row + color // 2
    j = 2 * col + color % 2

    if r < spins.shape[0] and i < L and j < L:
        s = spins[r, i, j]

        up = spins[r, (i - 1) % L, j]
        down = spins[r, (i + 1) % L, j]
        left = spins[r, i, (j - 1) % L]
        right = spins[r, i, (j + 1) % L]

        # Spins stay int8; the acceptance calculation uses float32.
        dE = float32(2 * s * (up + down + left + right)) * J
        rng_index = r * L * L + i * L + j
        random_value = xoroshiro128p_uniform_float32(rng_states, rng_index)

        if dE <= float32(0.0) or random_value < math.exp(-betas[r] * dE):
            spins[r, i, j] = -s
