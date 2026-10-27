"""XY energy, Binder cumulant and helicity modulus, with block-jackknife errors."""

import matplotlib.pyplot as plt
import numpy as np

from utils.Analysis.statistics import block_means, jackknife_from_blocks
from utils.data_io import load_runs

from utils.Analysis.bkt import fit_weber_minnhagen

class XYAnalysis:
    @classmethod
    def from_file(cls, path, block_size=200):
        return cls(load_runs(path), block_size=block_size)

    def __init__(self, runs, block_size=200):
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.sizes = sorted(runs)
        metadata = runs[self.sizes[0]]["metadata"]
        self.temperatures = np.asarray(metadata["temperatures"], dtype=float)
        self.J = float(metadata["model_parameters"]["J"])
        self.results = {}

        for L in self.sizes:
            samples = runs[L]["samples"]
            mx = np.asarray(samples["mx"], dtype=float)
            my = np.asarray(samples["my"], dtype=float)
            qx = np.asarray(samples["twist_sin_x"], dtype=float)
            m2 = mx**2 + my**2

            moments = {
                "energy": np.asarray(samples["energy"], dtype=float) / L**2,
                "m2": m2,
                "m4": m2**2,
                "cx": np.asarray(samples["twist_cos_x"], dtype=float),
                "qx": qx,
                "qx2": qx**2,
            }

            if len(mx) < 2 * block_size:
                raise ValueError(f"L={L}: need at least two complete time blocks")

            blocks = {name: block_means(x, block_size)
                      for name, x in moments.items()}

            def estimate(mean):
                energy = mean["energy"]
                binder = 1 - mean["m4"] / (2 * mean["m2"]**2)
                helicity = (
                    self.J * mean["cx"]
                    - self.J**2 / self.temperatures * (mean["qx2"] - mean["qx"]**2)
                ) / L**2
                return np.stack((energy, binder, helicity))

            values, errors = jackknife_from_blocks(blocks, estimate)
            names = ("energy", "binder", "helicity")
            self.results[L] = {}
            for name, value, error in zip(names, values, errors):
                self.results[L][name] = value
                self.results[L][name + "_err"] = error

    def fit_wm(self, temperature_range=None):
        T = self.temperatures
        L = np.asarray(self.sizes)

        helicity = np.stack([
            self.results[l]["helicity"] for l in L
        ])
        errors = np.stack([
            self.results[l]["helicity_err"] for l in L
        ])

        valid = np.all(
            np.isfinite(helicity) & np.isfinite(errors) & (errors > 0),
            axis=0
        )

        if temperature_range is not None:
            low, high = temperature_range
            valid &= (T >= low) & (T <= high)

        if len(L) < 3 or np.count_nonzero(valid) < 2:
            raise ValueError("Insufficient data for Weber–Minnhagen fit")

        fit = fit_weber_minnhagen(
            T[valid],
            L,
            helicity[:, valid],
            errors[:, valid],
            profile=False
        )

        if fit["temperature"] is None or fit["C"] is None:
            raise ValueError(f"WM fit failed: {fit['reason']}")

        return fit

    def plot(self, C=None, show=True):
        """If C is given, divide helicity by its WM finite-size factor."""
        T = self.temperatures
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

        for L in self.sizes:
            result = self.results[L]
            factor = 1.0
            if C is not None:
                denominator = 2 * np.log(L) + C
                if not np.isfinite(denominator) or denominator <= 0:
                    raise ValueError(f"Invalid WM correction for L={L}, C={C}")
                factor = 1 + 1 / denominator

            for ax, name in zip(axes, ("energy", "binder", "helicity")):
                ax.errorbar(
                    T,
                    result[name] / (factor if name == "helicity" else 1),
                    yerr=result[name + "_err"] / (factor if name == "helicity" else 1),
                    fmt=".-", capsize=2, label=f"L={L}",
                )

        axes[2].plot(T, 2 * T / np.pi, "k--", label=r"$2T/\pi$")
        axes[0].set(title="Energy per site", ylabel=r"$E/L^2$")
        axes[1].set(title="Binder cumulant", ylabel=r"$U_L$")
        axes[2].set(
            title="WM-corrected helicity" if C is not None else "Helicity modulus",
            ylabel=r"$\widetilde{\Upsilon}_L$" if C is not None else r"$\Upsilon_L$",
        )

        for ax in axes:
            ax.set_xlabel("Temperature")
            ax.grid(alpha=0.3)
        axes[2].legend()
        fig.tight_layout()
        if show:
            plt.show()
        return fig
