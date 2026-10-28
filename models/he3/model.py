"""Full complex 3x2 thin-film Ginzburg-Landau model, thesis Chapter 9."""

import numpy as np
from numba import cuda

from core.model import Model
from .update import metropolis_color
from .measure import (
    MEASUREMENT_NAMES, MEASURE_THREADS, he3_measure_kernel,
    he3_ordered_measure_kernel, he3_energy_kernel, he3_cached_measure_kernel,
    he3_ordered_cached_measure_kernel,
)


# Regan, Wiman & Sauls, PRB 101, 024517 (2020), Appendix Table I.
# Rows are powers 0..5 of p/34, columns are delta_beta_i / |beta_1^wc|.
# https://arxiv.org/pdf/1908.04190 ; calibrated for 0 <= pressure <= 34 bar.
_STRONG_COUPLING = np.array([
    [-0.009849, -0.04193, -0.01322, -0.004747, -0.08987],
    [-0.05043, -0.1177,  -0.05428, -0.3788,   -0.6925],
    [ 0.02205, -0.04322,  0.09559, -0.1774,    0.8761],
    [-0.02557,  0.08793, -0.06419,  0.1735,   -0.5929],
    [ 0.05023, -0.08598, -0.009310, 0.1878,    0.02904],
    [-0.02769,  0.03639,  0.01862, -0.1522,    0.08870],
])


class He3Model(Model):
    def __init__(self, K=1.0, betas=(-0.5, 1, 1, 1, -1), delta=0.15,
                 pressure=None, reduced_temperature=1.0, g_d=0):
        """Add pressure corrections to betas at a fixed physical T/Tc.

        pressure=None keeps the supplied betas; pressure=0 includes the
        zero-bar strong-coupling correction. Pressure is in bar (0..34).
        reduced_temperature is independent of the MC temperature ladder.
        """
        self.K = np.float32(K)
        self.pressure = None if pressure is None else float(pressure)
        self.reduced_temperature = float(reduced_temperature)
        coefficients = np.asarray(betas, dtype=np.float64)
        if self.pressure is not None:
            correction = np.polynomial.polynomial.polyval(self.pressure / 34, _STRONG_COUPLING)
            # The thesis's lattice coefficients are beta_i / (2 |beta_1^wc|).
            coefficients = coefficients + 0.5 * self.reduced_temperature * correction
        self.betas = tuple(np.float32(b) for b in coefficients)
        self.delta = np.float32(delta)

        if not np.isfinite(g_d) or g_d < 0:
            raise ValueError("g_d must be finite and nonnegative")
        self.g_d = np.float32(g_d)

    def initialize_state(self, L, replicas, seed):
        state = np.zeros((replicas, L, L, 3, 2), dtype=np.complex64)
        # A homogeneous positive-chirality minimum: d=(0,0,1), theta=0.
        beta_chiral = self.betas[1] + self.betas[3] + self.betas[4]
        amplitude = np.float32(1 / np.sqrt(2 * beta_chiral))
        state[..., 2, 0] = amplitude
        state[..., 2, 1] = 1j * amplitude
        return state

    def update_color(self, state, rng_states, betas, color, blocks, threads):
        metropolis_color[blocks, threads](
            state, rng_states, betas, color, self.betas, self.K, self.delta, self.g_d)

    def measure(self, state, temperature_indices=None, cached_energies=None):
        R = state.shape[0]
        output = cuda.device_array(
            (len(MEASUREMENT_NAMES), R),
            dtype=np.float64,
        )

        if cached_energies is None:
            if temperature_indices is None:
                he3_measure_kernel[R, MEASURE_THREADS](
                    state, output, self.betas, self.K, self.g_d,
                )
            else:
                he3_ordered_measure_kernel[R, MEASURE_THREADS](
                    state, output, self.betas, self.K,
                    temperature_indices, self.g_d,
                )
        else:
            if temperature_indices is None:
                he3_cached_measure_kernel[R, MEASURE_THREADS](
                    state, output, self.betas, self.K,
                    cached_energies, self.g_d,
                )
            else:
                he3_ordered_cached_measure_kernel[R, MEASURE_THREADS](
                    state, output, self.betas, self.K,
                    temperature_indices, cached_energies, self.g_d,
                )

        return dict(zip(MEASUREMENT_NAMES, output.copy_to_host()))

    def measure_with_energy(self, state, energies, temperature_indices=None):
        return self.measure(
            state,
            temperature_indices=temperature_indices,
            cached_energies=energies,
        )


    def measure_energy(self, state, output):
        he3_energy_kernel[state.shape[0], MEASURE_THREADS](
            state, output, self.betas, self.K, self.g_d)

    def parameters(self):
        return {"K": self.K, "betas": self.betas, "delta": self.delta,
                "pressure": self.pressure, "reduced_temperature": self.reduced_temperature,
                "g_d": float(self.g_d), "dipole_length_lattice": (
                    float(np.sqrt(2.0 * self.K / self.g_d))
                    if self.g_d > 0 else None
                )}
