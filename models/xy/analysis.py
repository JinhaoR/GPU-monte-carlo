
import numpy as np
import matplotlib.pyplot as plt

from utils.binder import binder_cumulant

class XYAnalysis:

    def __init__(self, runs):
        """
        runs: dictionary mapping L -> Simulation.run() result.

        Example:
            {
                16: {"samples": ..., "metadata": ...},
                32: {"samples": ..., "metadata": ...},
            }
        """
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

            E = samples["energy"]
            Mx = samples["mx"]
            My = samples["my"]
            C = samples["twist_cos_x"]
            Q = samples["twist_sin_x"]

            M2 = Mx**2 + My**2

            energy = np.mean(E, axis=0) / N

            magnetization = np.mean(
                np.sqrt(M2), axis=0
            ) / N

            helicity = (
                self.J * np.mean(C, axis=0)
                - self.J**2 / T * (
                    np.mean(Q**2, axis=0)
                    - np.mean(Q, axis=0)**2
                )
            ) / N

            binder = binder_cumulant(
                M2, gaussian_factor=2.0
            )

            results[L] = {
                "energy": energy,
                "magnetization": magnetization,
                "helicity": helicity,
                "binder": binder,
            }

        return results

    def plot(self):
        T = self.temperatures

        fig, (ax_b, ax_h) = plt.subplots(
            1, 2, figsize=(12, 5)
        )

        for L in self.sizes:
            values = self.results[L]

            ax_b.plot(
                T, values["binder"],
                "o-", label=f"L={L}"
            )

            ax_h.plot(
                T, values["helicity"],
                "o-", label=f"L={L}"
            )

        ax_h.plot(
            T, 2 * T / np.pi,
            "k--", label=r"$2T/\pi$"
        )

        for ax in (ax_b, ax_h):
            ax.set_xlabel("Temperature")
            ax.grid(alpha=0.3)
            ax.legend()

        ax_b.set_title("XY Binder cumulant")
        ax_b.set_ylabel("Binder cumulant")

        ax_h.set_title("Helicity modulus")
        ax_h.set_ylabel("Helicity modulus")

        fig.tight_layout()
        plt.show()
