import numpy as np
import matplotlib.pyplot as plt


class IsingAnalysis:

    def __init__(self, L, temperatures, E, M):
        self.L = L
        self.N = L * L

        self.temperatures = np.asarray(
            temperatures, dtype=np.float64
        )

        self.E = np.asarray(E, dtype=np.float64)
        self.M = np.asarray(M, dtype=np.float64)

    def compute(self):
        E = self.E
        M = self.M
        T = self.temperatures
        N = self.N

        mean_E = np.mean(E, axis=0)
        mean_E2 = np.mean(E**2, axis=0)

        mean_M = np.mean(M, axis=0)
        mean_abs_M = np.mean(np.abs(M), axis=0)
        mean_M2 = np.mean(M**2, axis=0)
        mean_M4 = np.mean(M**4, axis=0)

        self.results = {
            "energy": mean_E / N,
            "magnetization": mean_abs_M / N,
            "heat_capacity": (
                (mean_E2 - mean_E**2) / (N * T**2)
            ),
            "susceptibility": (
                (mean_M2 - mean_M**2) / (N * T)
            ),
            "binder": (
                1 - mean_M4 / (3 * mean_M2**2)
            ),
        }

        return self.results

    def plot(self):
        results = self.compute()

        Tc = 2 / np.log(1 + np.sqrt(2))

        labels = {
            "energy": "Energy per spin",
            "magnetization": "Absolute magnetization per spin",
            "heat_capacity": "Heat capacity per spin",
            "susceptibility": "Susceptibility per spin",
            "binder": "Binder cumulant",
        }

        fig, axes = plt.subplots(2, 3, figsize=(13, 8))

        for ax, (name, values) in zip(
            axes.flat, results.items()
        ):
            ax.plot(self.temperatures, values, "o-")
            ax.axvline(Tc, linestyle="--", color="gray")

            ax.set_xlabel("Temperature")
            ax.set_ylabel(labels[name])
            ax.grid(alpha=0.3)

        axes.flat[-1].axis("off")

        fig.suptitle(f"2D Ising model, L = {self.L}")
        fig.tight_layout()

        plt.show()
