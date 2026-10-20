import numpy as np
from numba import cuda
from core.model import Model
from .update import metropolis_color
from .measure import MEASUREMENT_NAMES, MEASURE_THREADS, ising_measure_kernel


class IsingModel(Model):
    def __init__(self, J=1.0):
        self.J = np.float32(J)

    def initialize_state(self, L, replicas, seed):
        return np.ones((replicas, L, L), dtype=np.int8)

    def update_color(self, state, rng_states, betas, color, blocks, threads):
        metropolis_color[blocks, threads](
            state, rng_states, betas, color, self.J
        )

    def measure(self, state):
        """Reduce the GPU state to two CPU measurement arrays."""
        R = state.shape[0]
        output = cuda.device_array(
            (len(MEASUREMENT_NAMES), R), dtype=np.float64
        )
        ising_measure_kernel[R, MEASURE_THREADS](state, output, self.J)
        results = output.copy_to_host()

        return {
            name: results[channel]
            for channel, name in enumerate(MEASUREMENT_NAMES)
        }

    def parameters(self):
        return {"J": self.J}
