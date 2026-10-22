"""Kawamura, PRL 82, 964 (1999), Eq. (3).

H = -J sum_<ij> (1 + tau_i*tau_j) cos(theta_i-theta_j) dot(d_i, d_j).
State components are theta, tau, dx, dy, dz; d is a unit vector.
"""

import numpy as np
from numba import cuda

from core.model import Model
from .update import metropolis_color
from .measure import (
    MEASUREMENT_NAMES, MEASURE_THREADS, kawamura_measure_kernel,
    kawamura_ordered_measure_kernel, kawamura_energy_kernel,
)


class KawamuraModel(Model):
    def __init__(self, J=1.0, delta=1.5):
        self.J = np.float32(J)
        self.delta = np.float32(delta)  # Phase proposal half-width, <= pi.

    def initialize_state(self, L, replicas, seed):
        state = np.zeros((replicas, L, L, 5), dtype=np.float32)
        state[..., 1] = 1  # Ordered chirality, theta=0, d=(0,0,1).
        state[..., 4] = 1
        return state

    def update_color(self, state, rng_states, betas, color, blocks, threads):
        metropolis_color[blocks, threads](
            state, rng_states, betas, color, self.J, self.delta)

    def measure(self, state, temperature_indices=None):
        R = state.shape[0]
        output = cuda.device_array((len(MEASUREMENT_NAMES), R), dtype=np.float64)
        if temperature_indices is None:
            kawamura_measure_kernel[R, MEASURE_THREADS](state, output, self.J)
        else:
            kawamura_ordered_measure_kernel[R, MEASURE_THREADS](
                state, output, self.J, temperature_indices)
        values = output.copy_to_host()
        return dict(zip(MEASUREMENT_NAMES, values))

    def measure_energy(self, state, output):
        kawamura_energy_kernel[state.shape[0], MEASURE_THREADS](state, output, self.J)

    def parameters(self):
        return {"J": self.J, "delta": self.delta}
