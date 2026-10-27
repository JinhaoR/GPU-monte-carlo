
import math

import numpy as np
from numba import cuda
from core.model import Model
from .update import metropolis_color
from .measure import (
    MEASUREMENT_NAMES, MEASURE_THREADS, xy_measure_kernel,
    xy_ordered_measure_kernel, xy_energy_kernel, xy_cached_measure_kernel,
    xy_ordered_cached_measure_kernel,
)


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

    def measure(self, state, temperature_indices=None, cached_energies=None):
        """Reduce the GPU state to five CPU measurement arrays."""
        R = state.shape[0]
        output = cuda.device_array(
            (len(MEASUREMENT_NAMES), R), dtype=np.float64
        )
        if cached_energies is None:
            if temperature_indices is None:
                xy_measure_kernel[R, MEASURE_THREADS](state, output, self.J)
            else:
                xy_ordered_measure_kernel[R, MEASURE_THREADS](
                    state, output, self.J, temperature_indices
                )
        else:
            if temperature_indices is None:
                xy_cached_measure_kernel[R, MEASURE_THREADS](
                    state, output, self.J, cached_energies
                )
            else:
                xy_ordered_cached_measure_kernel[R, MEASURE_THREADS](
                    state, output, self.J, temperature_indices, cached_energies
                )
        results = output.copy_to_host()

        return {
            name: results[channel]
            for channel, name in enumerate(MEASUREMENT_NAMES)
        }

    def measure_with_energy(self, state, energies, temperature_indices=None):
        return self.measure(
            state,
            temperature_indices=temperature_indices,
            cached_energies=energies,
        )

    def parameters(self):
        return {"J": self.J, "delta": self.delta}

    def measure_energy(self, state, output):
        """Leave total energies on the GPU for replica exchange."""
        xy_energy_kernel[state.shape[0], MEASURE_THREADS](state, output, self.J)
