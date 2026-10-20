
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
    def measure(self, state) -> dict:
        """
        Measure the GPU state, using CPU or GPU calculations.

        Return named CPU arrays, each with leading replica axis.
        """
        pass

    def parameters(self) -> dict:
        """Physical parameters to include in run metadata."""
        return {}
