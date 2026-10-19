
import numpy as np


class IsingSimulation:

    def __init__(
        self,
        runtime,
        thermalization_sweeps=1000,
        measurement_sweeps=4000,
        sample_interval=10,
    ):
        self.runtime = runtime
        self.thermalization_sweeps = thermalization_sweeps
        self.measurement_sweeps = measurement_sweeps
        self.sample_interval = sample_interval

        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")

        if measurement_sweeps < sample_interval:
            raise ValueError("No measurements would be collected")

    def measure(self):
        spins = self.runtime.spins.copy_to_host().astype(np.int32)

        M = np.sum(spins, axis=(1, 2))

        E = -self.runtime.model.J * np.sum(
            spins * (
                np.roll(spins, -1, axis=1)
                + np.roll(spins, -1, axis=2)
            ),
            axis=(1, 2)
        )

        return E, M

    def run(self):
        energies = []
        magnetizations = []

        # Thermalization
        for _ in range(self.thermalization_sweeps):
            self.runtime.sweep()

        # Sampling
        for sweep in range(self.measurement_sweeps):
            self.runtime.sweep()

            if (sweep + 1) % self.sample_interval == 0:
                E, M = self.measure()

                energies.append(E)
                magnetizations.append(M)

        E = np.asarray(energies, dtype=np.float64)
        M = np.asarray(magnetizations, dtype=np.float64)

        return E, M
