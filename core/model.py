
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
    def measure(self, state: np.ndarray) -> dict:
        """
        Measure a CPU configuration.

        Return named arrays, each with leading replica axis.
        """
        pass

    def parameters(self) -> dict:
        """Physical parameters to include in run metadata."""
        return {}
