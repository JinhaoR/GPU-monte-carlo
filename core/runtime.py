
import numpy as np

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states


class Runtime:

    def __init__(
        self, model, L, temperatures,
        threads=16, seed=42
    ):
        if L < 4 or L % 2 != 0:
            raise ValueError("L must be even and >= 4")

        if threads < 1 or threads**2 > 1024:
            raise ValueError("Invalid thread configuration")

        temperatures = np.asarray(
            temperatures, dtype=np.float64
        )

        if temperatures.ndim != 1 or len(temperatures) == 0:
            raise ValueError("Temperatures must be a nonempty 1D array")

        if not np.all(
            np.isfinite(temperatures) & (temperatures > 0)
        ):
            raise ValueError("Temperatures must be positive and finite")

        self.model = model
        self.L = L
        self.R = len(temperatures)
        self.temperatures = temperatures.copy()
        self.seed = seed

        # Model defines the initial physical state
        initial_state = np.asarray(
            model.initialize_state(L, self.R, seed)
        )

        if initial_state.shape[:3] != (self.R, L, L):
            raise ValueError("Invalid model state shape")

        # Runtime owns GPU memory
        self.state = cuda.to_device(
            np.ascontiguousarray(initial_state)
        )

        self.rng_states = create_xoroshiro128p_states(
            self.R * L * L, seed=seed
        )

        self.betas = cuda.to_device(
            (1.0 / temperatures).astype(np.float32)
        )

        # Shared four-color launch configuration
        self.threads_per_block = (threads, threads, 1)

        half_L = L // 2

        self.blocks_per_grid = (
            (half_L + threads - 1) // threads,
            (half_L + threads - 1) // threads,
            self.R
        )

        # GPU energy cache, associated with the current physical state.
        self._configuration_version = 0
        self._energy_version = -1
        self._cached_energies = None

    def invalidate_energy_cache(self):
        """Invalidate energies before changing fields or Hamiltonian parameters.

        sweep() does this automatically. Call explicitly before restoring or
        directly modifying state, or changing model parameters. Temperature
        exchanges do not change the physical energy and need no invalidation.
        """
        self._configuration_version += 1
        self._energy_version = -1
        self._cached_energies = None


    def sweep(self):
        # Invalidate before launching potentially asynchronous GPU updates.
        self.invalidate_energy_cache()

        for color in range(4):
            self.model.update_color(
                self.state,
                self.rng_states,
                self.betas,
                color,
                self.blocks_per_grid,
                self.threads_per_block
            )

    def measure(self, temperature_indices=None):
        """Return CPU observables, optionally ordered by temperature."""

        cache_valid = (
            self._cached_energies is not None
            and self._energy_version == self._configuration_version
        )

        if cache_valid:
            return self.model.measure_with_energy(
                self.state,
                self._cached_energies,
                temperature_indices=temperature_indices,
            )

        return self.model.measure(
            self.state,
            temperature_indices=temperature_indices,
        )

    def measure_energy(self, output):
        """Calculate total energies in physical walker order."""
        # A failed producer may partially overwrite the previously cached buffer.
        self._cached_energies = None
        self._energy_version = -1
        result = self.model.measure_energy(self.state, output)

        # Retain the caller-owned GPU buffer, not a CPU copy.
        self._cached_energies = output
        self._energy_version = self._configuration_version

        return result

    def snapshot(self):
        return self.state.copy_to_host()
