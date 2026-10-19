
import numpy as np

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from IsingModel import metropolis_color


class IsingRuntime:

    def __init__(self, model, L, temperatures, threads=16, seed=42):
        if L < 2 or L % 2 != 0:
            raise ValueError("L must be an even integer >= 2")

        if threads < 1 or threads**2 > 1024:
            raise ValueError("Invalid threads per block")

        temperatures = np.asarray(temperatures, dtype=np.float32)

        if temperatures.ndim != 1 or len(temperatures) == 0:
            raise ValueError("Expected a nonempty 1D temperature array")

        if not np.all(np.isfinite(temperatures) & (temperatures > 0)):
            raise ValueError("Temperatures must be finite and positive")

        self.model = model
        self.L = L
        self.R = len(temperatures)
        self.threads = threads

        self.spins = cuda.to_device(
            np.ones((self.R, L, L), dtype=np.int8)
        )

        self.rng_states = create_xoroshiro128p_states(
            self.R * L * L, seed=seed
        )

        self.betas = cuda.to_device(1.0 / temperatures)

        half_L = L // 2

        self.threads_per_block = (threads, threads, 1)

        self.blocks_per_grid = (
            (half_L + threads - 1) // threads,
            (half_L + threads - 1) // threads,
            self.R
        )

    def sweep(self):
        for color in range(4):
            metropolis_color[
                self.blocks_per_grid,
                self.threads_per_block
            ](
                self.spins,
                self.rng_states,
                self.betas,
                color,
                self.model.J
            )
