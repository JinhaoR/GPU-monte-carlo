
import math

from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


import numpy as np
from core.model import Model


@cuda.jit
def metropolis_color(spins, rng_states, betas, color, J):
    row, col, r = cuda.grid(3)

    L = spins.shape[1]

    i = 2 * row + color // 2
    j = 2 * col + color % 2

    if r < spins.shape[0] and i < L and j < L:
        s = spins[r, i, j]

        up = spins[r, (i - 1) % L, j]
        down = spins[r, (i + 1) % L, j]
        left = spins[r, i, (j - 1) % L]
        right = spins[r, i, (j + 1) % L]

        dE = 2 * J * s * (up + down + left + right)

        rng_index = r * L * L + i * L + j

        random_value = xoroshiro128p_uniform_float32(
            rng_states, rng_index
        )

        if dE <= 0 or random_value < math.exp(-betas[r] * dE):
            spins[r, i, j] = -s



class IsingModel(Model):

    def __init__(self, J=1.0):
        self.J = J

    def initialize_state(self, L, replicas, seed):
        return np.ones(
            (replicas, L, L), dtype=np.int8
        )

    def update_color(
        self, state, rng_states, betas,
        color, blocks, threads
    ):
        metropolis_color[blocks, threads](
            state,
            rng_states,
            betas,
            color,
            self.J
        )

    def measure(self, state):
        spins = state.astype(np.int32)

        M = np.sum(spins, axis=(1, 2))

        E = -self.J * np.sum(
            spins * (
                np.roll(spins, -1, axis=1)
                + np.roll(spins, -1, axis=2)
            ),
            axis=(1, 2)
        )

        return {
            "energy": E,
            "magnetization": M
        }

    def parameters(self):
        return {"J": self.J}