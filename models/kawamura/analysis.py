"""Kawamura's chirality, doubled-phase and spin-tensor Binder ratios, Eqs. (4)-(6)."""

import matplotlib.pyplot as plt
import numpy as np

from utils.Analysis.statistics import blocked_jackknife
from utils.data_io import load_runs


OBSERVABLES = (
    "energy", "chirality", "phase", "spin",
    "binder_chirality", "binder_phase", "binder_spin", "helicity",
)


class KawamuraAnalysis:
    @classmethod
    def from_file(cls, path, block_size=100):
        return cls(load_runs(path), block_size=block_size)

    def __init__(self, runs, block_size=100):
        self.runs, self.sizes = runs, sorted(runs)
        metadata = runs[self.sizes[0]]["metadata"]
        self.temperatures = np.asarray(metadata["temperatures"])
        self.J = float(metadata["model_parameters"]["J"])
        self.block_size = block_size
        self.results = self.compute()

    def _estimate(self, L, m):
        helicity = (self.J * m["cx"] - self.J**2 / self.temperatures
                    * (m["qx2"] - m["qx"]**2)) / L**2
        return np.stack((
            m["energy"], np.sqrt(m["tau2"]), np.sqrt(m["phase2"]), np.sqrt(m["spin2"]),
            0.5 * (3 - m["tau4"] / m["tau2"]**2),
            2 - m["phase4"] / m["phase2"]**2,
            0.5 * (7 - 5 * m["spin4"] / m["spin2"]**2), helicity,
        ))

    def compute(self):
        results = {}
        for L in self.sizes:
            s = self.runs[L]["samples"]
            tau2 = s["chirality"]**2 / L**4
            phase2 = (s["px"]**2 + s["py"]**2) / L**4
            spin2 = s["spin2"] / L**4
            moments = {
                "energy": s["energy"] / L**2,
                "tau2": tau2, "tau4": tau2**2,
                "phase2": phase2, "phase4": phase2**2,
                "spin2": spin2, "spin4": spin2**2,
                "cx": s["twist_cos_x"], "qx": s["twist_sin_x"], "qx2": s["twist_sin_x"]**2,
            }
            values, errors = blocked_jackknife(
                moments, lambda m: self._estimate(L, m), self.block_size)
            results[L] = {key: array for name, value, error in zip(OBSERVABLES, values, errors)
                          for key, array in ((name, value), (name + "_err", error))}
        return results

    def plot(self, show=True):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        labels = ("binder_chirality", "binder_phase", "binder_spin", "helicity")
        for ax, name in zip(axes.flat, labels):
            for L in self.sizes:
                r = self.results[L]
                ax.errorbar(self.temperatures, r[name], yerr=r[name + "_err"], fmt=".-", label=f"L={L}")
            ax.set(xlabel="Temperature", ylabel=name.replace("_", " "))
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("Kawamura film model")
        fig.tight_layout()
        if show:
            plt.show()
        return fig
