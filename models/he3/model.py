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

        # Optional exploration: every Nth scalar measurement also records spatial data.
        self.spatial_every = 0
        self.spatial_samples = []
        self.realspace_every = 0
        self.realspace_indices = []
        self.realspace_samples = []
        self.charged_profiles = False
        self._measurement_count = 0
        self._spatial_meter = None

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

        result = dict(zip(MEASUREMENT_NAMES, output.copy_to_host()))
        if self.spatial_every or self.realspace_every:
            self._measurement_count += 1
            if self.spatial_every and self._measurement_count % self.spatial_every == 0:
                from .measure_spatial import SpatialMeter
                if self._spatial_meter is None:
                    self._spatial_meter = SpatialMeter(state.shape[1], R)
                amplitudes = self._spatial_meter.measure(state, temperature_indices)
                structure = self._spatial_meter.structure(state, self, temperature_indices)
                walkers = (np.arange(R, dtype=np.int32) if temperature_indices is None
                           else np.argsort(temperature_indices.copy_to_host()).astype(np.int32))
                self.spatial_samples.append(
                    (self._measurement_count - 1, amplitudes, structure, walkers)
                )
            if self.realspace_every and self._measurement_count % self.realspace_every == 0:
                from .spatial import measurement_profiles
                walkers = (np.arange(R, dtype=np.int32) if temperature_indices is None
                           else np.argsort(temperature_indices.copy_to_host()).astype(np.int32))
                indices = np.asarray(self.realspace_indices, dtype=np.int32)
                profiles, means = [], []
                for walker in walkers[indices]:
                    c, mean = measurement_profiles(state[int(walker)].copy_to_host(), self.charged_profiles)
                    profiles.append(c)
                    means.append(mean)
                self.realspace_samples.append((self._measurement_count - 1,
                    np.stack(profiles), np.stack(means), walkers[indices], indices.copy()))
        return result

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

    def measurement_settings(self):
        return dict(spatial_every=self.spatial_every, realspace_every=self.realspace_every,
                    realspace_indices=self.realspace_indices, charged_profiles=self.charged_profiles)

    def measurement_state(self):
        return dict(settings=self.measurement_settings(), count=self._measurement_count,
                    spatial=[np.stack(column) for column in zip(*self.spatial_samples)],
                    realspace=[np.stack(column) for column in zip(*self.realspace_samples)])

    def validate_measurement_state(self, saved, samples, scalar_count, R, L):
        from utils.checkpoint import same_settings
        if not same_settings(saved['settings'], self.measurement_settings()):
            raise ValueError('Measurement cadence/fields mismatch')
        expected = scalar_count if self.spatial_every or self.realspace_every else 0
        if saved['count'] != expected:
            raise ValueError('Measurement counter mismatch')
        if scalar_count and (set(samples) != set(MEASUREMENT_NAMES)
                or any(v.dtype != np.float64 or v.shape != (scalar_count, R) for v in samples.values())):
            raise ValueError('He3 scalar measurement channels/dtypes mismatch')
        for key, cadence, columns in [('spatial', self.spatial_every, 4), ('realspace', self.realspace_every, 5)]:
            data = saved[key]
            events = expected//cadence if cadence else 0
            if (events == 0 and data) or (events and (len(data) != columns or any(len(c) != events for c in data))):
                raise ValueError('Measurement record length mismatch')
            if events and not np.array_equal(data[0], np.arange(cadence-1, expected, cadence)):
                raise ValueError('Measurement sample indices mismatch')
            if events:
                if key == 'spatial':
                    shapes = [(events,), (events,R,16,5), (events,R,68), (events,R)]
                    types = [np.int64, np.complex128, np.float64, np.int32]
                else:
                    probes, fields = len(self.realspace_indices), 11 if self.charged_profiles else 7
                    shapes = [(events,), (events,probes,fields,2,L//2+1),
                              (events,probes,fields), (events,probes), (events,probes)]
                    types = [np.int64, np.complex128, np.complex128, np.int32, np.int32]
                for array, shape, dtype in zip(data, shapes, types):
                    if array.shape != shape or array.dtype != dtype:
                        raise ValueError('Measurement shape/dtype mismatch')

    def load_measurement_state(self, saved):
        self._measurement_count = saved['count']
        self.spatial_samples = list(zip(*saved['spatial']))
        self.realspace_samples = list(zip(*saved['realspace']))
        self._spatial_meter = None  # scratch space, no physical or RNG state
