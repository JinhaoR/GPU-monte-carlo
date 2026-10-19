
import numpy as np

class Simulation:

    def __init__(
        self,
        runtime,
        thermalization_sweeps=1000,
        measurement_sweeps=4000,
        sample_interval=10
    ):
        if thermalization_sweeps < 0:
            raise ValueError("Thermalization sweeps cannot be negative")

        if measurement_sweeps < 1:
            raise ValueError("Measurement sweeps must be positive")

        if sample_interval < 1 or sample_interval > measurement_sweeps:
            raise ValueError("Invalid sampling interval")

        self.runtime = runtime
        self.thermalization_sweeps = thermalization_sweeps
        self.measurement_sweeps = measurement_sweeps
        self.sample_interval = sample_interval

    def run(self):
        rt = self.runtime

        # Named measurement histories
        samples = {}
        sample_sweeps = []

        # Thermalization
        for _ in range(self.thermalization_sweeps):
            rt.sweep()

        # Sampling
        for step in range(1, self.measurement_sweeps + 1):
            rt.sweep()

            if step % self.sample_interval != 0:
                continue

            snapshot = rt.snapshot()
            measurement = rt.model.measure(snapshot)

            if not measurement:
                raise ValueError("Model returned no measurements")

            # Initialize storage using measurement names
            if not samples:
                samples = {
                    name: [] for name in measurement
                }

            if measurement.keys() != samples.keys():
                raise ValueError("Measurement channels changed")

            for name, value in measurement.items():
                value = np.asarray(value)

                if value.ndim < 1 or value.shape[0] != rt.R:
                    raise ValueError(
                        f"Invalid replica dimension for {name}"
                    )

                samples[name].append(value.copy())

            sample_sweeps.append(
                self.thermalization_sweeps + step
            )

        # Stack along the new sampling axis
        samples = {
            name: np.stack(values, axis=0)
            for name, values in samples.items()
        }

        metadata = {
            "L": rt.L,
            "temperatures": rt.temperatures.copy(),
            "seed": rt.seed,
            "model_parameters": rt.model.parameters(),
            "thermalization_sweeps": self.thermalization_sweeps,
            "measurement_sweeps": self.measurement_sweeps,
            "sample_interval": self.sample_interval,
            "sample_sweeps": np.asarray(sample_sweeps),
        }

        return {
            "samples": samples,
            "metadata": metadata
        }
