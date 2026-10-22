"""Chirality, composite U(1) phase and helicity of the full thin-film model."""

import matplotlib.pyplot as plt
import numpy as np

from utils.Analysis.statistics import blocked_jackknife
from utils.data_io import load_runs


OBSERVABLES = (
    "energy", "density", "chirality", "phase", "binder_chirality", "binder_phase",
    "helicity_x", "helicity_y", "helicity",
)


class He3Analysis:
    @classmethod
    def from_file(cls, path, block_size=100):
        return cls(load_runs(path), block_size=block_size)

    def __init__(self, runs, block_size=100):
        self.runs, self.sizes = runs, sorted(runs)
        self.temperatures = np.asarray(runs[self.sizes[0]]["metadata"]["temperatures"])
        self.block_size = block_size
        self.results = self.compute()

    def _estimate(self, L, m):
        hx = (m["kx"] - (m["ix2"] - m["ix"]**2) / self.temperatures) / L**2
        hy = (m["ky"] - (m["iy2"] - m["iy"]**2) / self.temperatures) / L**2
        return np.stack((
            m["energy"], m["density"], m["chirality"], np.sqrt(m["phase2"]),
            1 - m["chi4"] / (3 * m["chi2"]**2),
            2 - m["phase4"] / m["phase2"]**2,
            hx, hy, 0.5 * (hx + hy),
        ))

    def compute(self):
        results = {}
        for L in self.sizes:
            s = self.runs[L]["samples"]
            chirality = s["chirality"] / s["density"]
            phase2 = (s["phase_real"]**2 + s["phase_imag"]**2) / L**4
            moments = {
                "energy": s["energy"] / L**2, "density": s["density"] / L**2,
                "chirality": np.abs(chirality),
                "chi2": chirality**2, "chi4": chirality**4,
                "phase2": phase2, "phase4": phase2**2,
                "ix": s["current_x"], "ix2": s["current_x"]**2,
                "kx": s["curvature_x"],
                "iy": s["current_y"], "iy2": s["current_y"]**2,
                "ky": s["curvature_y"],
            }
            values, errors = blocked_jackknife(
                moments, lambda m: self._estimate(L, m), self.block_size)
            results[L] = {key: array for name, value, error in zip(OBSERVABLES, values, errors)
                          for key, array in ((name, value), (name + "_err", error))}
        return results

    def plot(self, show=True):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        labels = ("binder_chirality", "binder_phase", "helicity", "density")
        for ax, name in zip(axes.flat, labels):
            for L in self.sizes:
                r = self.results[L]
                ax.errorbar(self.temperatures, r[name], yerr=r[name + "_err"], fmt=".-", label=f"L={L}")
            ax.set(xlabel="Temperature", ylabel=name.replace("_", " "))
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("Full thin-film helium-3 model")
        fig.tight_layout()
        if show:
            plt.show()
        return fig
