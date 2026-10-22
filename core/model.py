
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):

    @abstractmethod
    def initialize_state(self, L, replicas, seed) -> np.ndarray:
        """
        Return initial CPU state with shape (replicas, L, L, ...).
        """
        pass

    @abstractmethod
    def update_color(
        self, state, rng_states, betas,
        color, blocks, threads
    ):
        """
        Launch the CUDA kernel for one checkerboard color.
        """
        pass

    @abstractmethod
    def measure(self, state, temperature_indices=None) -> dict:
        """
        Measure the GPU state, using CPU or GPU calculations.

        Return named CPU arrays, each with leading replica axis. Optional
        temperature_indices is a GPU array giving the destination temperature
        index for each physical walker; scatter outputs on the GPU when given.
        """
        pass

    def measure_energy(self, state, output):
        """Write total energies to a caller-owned float64 GPU array (R,).

        Energies follow physical walker order and use the same Hamiltonian
        for every temperature. This optional path must not copy to the CPU
        or allocate temporary GPU arrays.
        """
        raise NotImplementedError(
            "This model does not provide GPU total-energy measurements"
        )

    def parameters(self) -> dict:
        """Physical parameters to include in run metadata."""
        return {}
