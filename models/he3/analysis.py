"""Chirality, composite U(1) phase and helicity of the full thin-film model."""

import matplotlib.pyplot as plt
import numpy as np

from utils.Analysis.statistics import blocked_jackknife
from utils.data_io import load_runs

from utils.Analysis.bkt import fit_weber_minnhagen


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


    def fit_wm(self, temperature_range=None, q=0.5):
        """Weber–Minnhagen fit for vortex charge q."""

        if not np.isfinite(q) or q <= 0:
            raise ValueError("Vortex charge q must be positive and finite")

        T = self.temperatures
        L = np.asarray(self.sizes)

        helicity = np.stack([
            self.results[l]["helicity"] for l in L
        ])
        errors = np.stack([
            self.results[l]["helicity_err"] for l in L
        ])

        valid = np.all(
            np.isfinite(helicity)
            & np.isfinite(errors)
            & (errors > 0),
            axis=0
        )
        valid &= np.isfinite(T) & (T > 0)

        if temperature_range is not None:
            low, high = temperature_range
            valid &= (T >= low) & (T <= high)

        if len(L) < 3 or np.count_nonzero(valid) < 2:
            raise ValueError("Insufficient data for Weber–Minnhagen fit")

        # Transform the charge-q jump to the ordinary 2T/pi form.
        factor = q**2

        fit = fit_weber_minnhagen(
            T[valid],
            L,
            factor * helicity[:, valid],
            factor * errors[:, valid],
            profile=False
        )

        if fit["temperature"] is None or fit["C"] is None:
            raise ValueError(f"WM fit failed: {fit['reason']}")

        fit["q"] = q

        return fit


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


    def plot(self, C=None, q=0.5, show=True):
        T = self.temperatures

        observables = {
            "binder_chirality": (
                "Chirality Binder cumulant",
                r"$U_\chi$",
            ),
            "binder_phase": (
                "Composite phase Binder cumulant",
                r"$U_{\Phi_2}$",
            ),
            "helicity": (
                "WM-corrected helicity" if C is not None else "Helicity modulus",
                r"$\widetilde{\Upsilon}_L$" if C is not None else r"$\Upsilon_L$",
            ),
            "density": (
                "Density",
                r"$n$",
            ),
        }

        colors = [
            "#4C78A8",
            "#F58518",
            "#54A24B",
            "#E45756",
            "#B279A2",
        ]

        figures = {}
        errorevery = max(1, len(T) // 20)

        for name, (title, ylabel) in observables.items():
            fig, ax = plt.subplots(figsize=(7, 5))

            for index, L in enumerate(self.sizes):
                result = self.results[L]
                color = colors[index % len(colors)]

                y = result[name]
                yerr = result[name + "_err"]

                if name == "helicity" and C is not None:
                    denominator = 2 * np.log(L) + C
                    if denominator <= 0 or not np.isfinite(denominator):
                        raise ValueError(
                            f"Invalid WM correction: L={L}, C={C}"
                        )

                    factor = 1 + 1 / denominator
                    y = y / factor
                    yerr = yerr / factor

                # Continuous line without markers
                ax.plot(
                    T, y,
                    color=color,
                    linewidth=1.8,
                    label=f"L={L}",
                )

                ax.errorbar(
                    T, y,
                    yerr=yerr,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.4,
                    capsize=0,
                    alpha=0.9,
                    errorevery=1,
                )

            if name == "helicity":
                ax.plot(
                    T,
                    2 * T / (np.pi * q**2),
                    "k--",
                    linewidth=1.3,
                    label=rf"$2T/(\pi q^2),\ q={q:g}$",
                )

            ax.set(
                title=title,
                xlabel="Temperature",
                ylabel=ylabel,
            )
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
            fig.tight_layout()

            figures[name] = fig

        if show:
            plt.show()

        return figures


