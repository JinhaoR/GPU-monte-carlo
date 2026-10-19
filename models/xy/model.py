
import math
import numpy as np

from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

from core.model import Model


@cuda.jit
def metropolis_color(state, rng_states, betas, color, J, delta):
    row, col, r = cuda.grid(3)

    L = state.shape[1]

    i = 2 * row + color // 2
    j = 2 * col + color % 2

    if r < state.shape[0] and i < L and j < L:

        theta = state[r, i, j]

        # Propose a new angle
        rng_index = r * L * L + i * L + j

        u = xoroshiro128p_uniform_float32(
            rng_states, rng_index
        )

        proposed = theta + delta * (2.0 * u - 1.0)

        # Keep angle within [-pi, pi)
        if proposed >= math.pi:
            proposed -= 2.0 * math.pi
        elif proposed < -math.pi:
            proposed += 2.0 * math.pi

        # Nearest neighbors
        up = state[r, (i - 1) % L, j]
        down = state[r, (i + 1) % L, j]
        left = state[r, i, (j - 1) % L]
        right = state[r, i, (j + 1) % L]

        # Energy before proposal
        E_old = -J * (
            math.cos(theta - up)
            + math.cos(theta - down)
            + math.cos(theta - left)
            + math.cos(theta - right)
        )

        # Energy after proposal
        E_new = -J * (
            math.cos(proposed - up)
            + math.cos(proposed - down)
            + math.cos(proposed - left)
            + math.cos(proposed - right)
        )

        dE = E_new - E_old

        # Metropolis acceptance
        v = xoroshiro128p_uniform_float32(
            rng_states, rng_index
        )

        if dE <= 0 or v < math.exp(-betas[r] * dE):
            state[r, i, j] = proposed


class XYModel(Model):

    def __init__(self, J=1.0, delta=1.5):
        if J <= 0:
            raise ValueError("J must be positive")
        if not (0 < delta <= math.pi):
            raise ValueError("delta must be in (0, pi]")

        self.J = J
        self.delta = delta

    def initialize_state(self, L, replicas, seed):
        # Ordered initial configuration: theta = 0
        return np.zeros(
            (replicas, L, L), dtype=np.float32
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
            self.J,
            self.delta
        )

    def measure(self, state):
        # Use float32 trigonometry with float64 accumulation.
        theta = np.asarray(state, dtype=np.float32)

        # Angular differences along positive lattice directions
        dx = theta - np.roll(theta, -1, axis=1)
        dy = theta - np.roll(theta, -1, axis=2)

        cos_x = np.cos(dx)
        cos_y = np.cos(dy)

        # Total energy
        E = -self.J * np.sum(
            cos_x + cos_y, axis=(1, 2), dtype=np.float64
        )

        # Magnetization components
        Mx = np.sum(
            np.cos(theta), axis=(1, 2), dtype=np.float64
        )
        My = np.sum(
            np.sin(theta), axis=(1, 2), dtype=np.float64
        )

        # Quantities required for x-direction helicity modulus
        twist_cos_x = np.sum(
            cos_x, axis=(1, 2), dtype=np.float64
        )
        twist_sin_x = np.sum(
            np.sin(dx), axis=(1, 2), dtype=np.float64
        )

        return {
            "energy": E,
            "mx": Mx,
            "my": My,
            "twist_cos_x": twist_cos_x,
            "twist_sin_x": twist_sin_x,
        }

    def parameters(self):
        return {
            "J": self.J,
            "delta": self.delta,
        }
