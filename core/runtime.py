
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

    def sweep(self):
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
        """Return CPU measurements, optionally ordered by GPU temperature indices."""
        if temperature_indices is None:
            return self.model.measure(self.state)
        return self.model.measure(
            self.state, temperature_indices=temperature_indices
        )

    def measure_energy(self, output):
        """Write current total energies to a float64 GPU array in walker order."""
        return self.model.measure_energy(self.state, output)

    def snapshot(self):
        return self.state.copy_to_host()
