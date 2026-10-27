"""2D Ising observables with block-jackknife errors."""

import numpy as np
import matplotlib.pyplot as plt

from utils.Analysis.statistics import block_means, jackknife_from_blocks
from utils.data_io import load_runs
from utils.Analysis.joint_histogram import joint_histogram


class IsingAnalysis:

    @classmethod
    def from_file(cls, path, block_size=200):
        return cls(load_runs(path), block_size=block_size)

    def __init__(self, runs, block_size=200):
        self.sizes = sorted(runs)
        self.runs = runs
        metadata = runs[self.sizes[0]]["metadata"]
        self.temperatures = np.asarray(metadata["temperatures"], dtype=float)
        self.J = float(metadata["model_parameters"]["J"])

        self.results = {}

        for L in self.sizes:
            samples = runs[L]["samples"]
            N = L**2
            T = self.temperatures

            E = np.asarray(samples["energy"], dtype=float)
            M = np.asarray(samples["magnetization"], dtype=float)

            # Moments for thermodynamic observables
            moments = {
                "E": E,
                "E2": E**2,
                "M": M,
                "absM": np.abs(M),
                "M2": M**2,
                "M4": M**4,
            }

            # Time blocking
            if len(E) < 2 * block_size:
                raise ValueError(f"L={L}: need at least two complete time blocks")

            blocks = {
                name: block_means(x, block_size)
                for name, x in moments.items()
            }

            # Calculate observables from averaged moments
            def estimate(mean):
                energy = mean["E"] / N

                magnetization = mean["absM"] / N

                heat_capacity = (
                    mean["E2"] - mean["E"]**2
                ) / (N * T**2)

                susceptibility = (
                    mean["M2"] - mean["M"]**2
                ) / (N * T)

                binder = 1 - mean["M4"] / (3 * mean["M2"]**2)

                return np.stack((
                    energy,
                    magnetization,
                    heat_capacity,
                    susceptibility,
                    binder,
                ))

            # Block jackknife
            values, errors = jackknife_from_blocks(blocks, estimate)

            names = (
                "energy",
                "magnetization",
                "heat_capacity",
                "susceptibility",
                "binder",
            )

            self.results[L] = {}

            for name, value, error in zip(names, values, errors):
                self.results[L][name] = value
                self.results[L][name + "_err"] = error

    def joint_distribution(self, L, temperature, bins=40):
        run = self.runs[L]

        T = np.asarray(run["metadata"]["temperatures"])
        index = np.argmin(np.abs(T - temperature))

        samples = run["samples"]

        E = samples["energy"][:, index]
        M = samples["magnetization"][:, index]

        N = L * L

        e = E / N
        m = M / N

        hist = joint_histogram(e, m, bins=bins)

        return float(T[index]), hist

    def plot(self, show=True):
        T = self.temperatures

        # Exact Tc for the zero-field nearest-neighbor square-lattice Ising model
        Tc = 2 * self.J / np.log(1 + np.sqrt(2))

        labels = {
            "energy": "Energy per spin",
            "magnetization": "Absolute magnetization per spin",
            "heat_capacity": "Heat capacity per spin",
            "susceptibility": "Susceptibility per spin",
            "binder": "Binder cumulant",
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        for ax, (name, label) in zip(axes.flat, labels.items()):
            for L in self.sizes:
                result = self.results[L]

                ax.errorbar(
                    T,
                    result[name],
                    yerr=result[name + "_err"],
                    fmt=".-",
                    capsize=2,
                    label=f"L={L}",
                )

            ax.axvline(
                Tc,
                color="gray",
                linestyle="--",
                label=rf"$T_c={Tc:.4f}$",
            )

            ax.set_xlabel("Temperature")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(alpha=0.3)
            ax.legend()

        axes.flat[-1].axis("off")

        fig.suptitle("2D Ising model")
        fig.tight_layout()

        if show:
            plt.show()

        return fig