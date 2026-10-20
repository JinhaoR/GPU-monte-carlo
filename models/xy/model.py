
import math

import numpy as np
from numba import cuda
from core.model import Model
from .update import metropolis_color
from .measure import MEASUREMENT_NAMES, MEASURE_THREADS, xy_measure_kernel


class XYModel(Model):
    def __init__(self, J=1.0, delta=1.5):
        if J <= 0:
            raise ValueError("J must be positive")
        if not (0 < delta <= math.pi):
            raise ValueError("delta must be in (0, pi]")
        self.J = np.float32(J)
        self.delta = np.float32(delta)

    def initialize_state(self, L, replicas, seed):
        # Ordered starting state, as in our original XY implementation.
        return np.zeros((replicas, L, L), dtype=np.float32)

    def update_color(self, state, rng_states, betas, color, blocks, threads):
        metropolis_color[blocks, threads](
            state, rng_states, betas, color, self.J, self.delta
        )

    def measure(self, state):
        """Reduce the GPU state to five CPU measurement arrays."""
        R = state.shape[0]
        output = cuda.device_array(
            (len(MEASUREMENT_NAMES), R), dtype=np.float64
        )
        xy_measure_kernel[R, MEASURE_THREADS](state, output, self.J)
        results = output.copy_to_host()

        return {
            name: results[channel]
            for channel, name in enumerate(MEASUREMENT_NAMES)
        }

    def parameters(self):
        return {"J": self.J, "delta": self.delta}
