import numpy as np
import matplotlib.pyplot as plt

from utils.Analysis.binder import binder_cumulant

class IsingAnalysis:

    def __init__(self, runs):
        """runs maps each lattice size L to a Simulation.run() result."""
        if len(runs) < 2:
            raise ValueError("At least two lattice sizes required")

        self.runs = runs
        self.sizes = sorted(runs)

        first = runs[self.sizes[0]]["metadata"]

        self.temperatures = np.asarray(
            first["temperatures"], dtype=np.float64
        )
        self.J = first["model_parameters"]["J"]

        for L, run in runs.items():
            metadata = run["metadata"]

            if metadata["L"] != L:
                raise ValueError(f"L mismatch for {L}")

            if not np.array_equal(
                metadata["temperatures"], self.temperatures
            ):
                raise ValueError("All sizes must use the same temperatures")

            if metadata["model_parameters"]["J"] != self.J:
                raise ValueError("All sizes must use the same J")

        self.results = self.compute()

    def compute(self):
        results = {}
        T = self.temperatures

        for L in self.sizes:
            samples = self.runs[L]["samples"]
            N = L**2

            E = np.asarray(samples["energy"], dtype=np.float64)
            M = np.asarray(samples["magnetization"], dtype=np.float64)

            mean_E = np.mean(E, axis=0)
            mean_E2 = np.mean(E**2, axis=0)

            mean_M = np.mean(M, axis=0)
            mean_abs_M = np.mean(np.abs(M), axis=0)
            mean_M2 = np.mean(M**2, axis=0)

            results[L] = {
                "energy": mean_E / N,
                "magnetization": mean_abs_M / N,
                "heat_capacity": (
                    (mean_E2 - mean_E**2) / (N * T**2)
                ),
                "susceptibility": (
                    (mean_M2 - mean_M**2) / (N * T)
                ),
                "binder": binder_cumulant(
                    M**2, gaussian_factor=3.0
                ),
            }

        return results

    def plot(self):
        Tc = 2 * self.J / np.log(1 + np.sqrt(2))

        labels = {
            "energy": "Energy per spin",
            "magnetization": "Absolute magnetization per spin",
            "heat_capacity": "Heat capacity per spin",
            "susceptibility": "Susceptibility per spin",
            "binder": "Binder cumulant",
        }

        fig, axes = plt.subplots(2, 3, figsize=(13, 8))

        for ax, (name, label) in zip(
            axes.flat, labels.items()
        ):
            for L in self.sizes:
                ax.plot(
                    self.temperatures, self.results[L][name],
                    "o-", label=f"L={L}"
                )

            ax.axvline(Tc, linestyle="--", color="gray")

            ax.set_xlabel("Temperature")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            ax.legend()

        axes.flat[-1].axis("off")

        fig.suptitle("2D Ising model")
        fig.tight_layout()

        plt.show()
