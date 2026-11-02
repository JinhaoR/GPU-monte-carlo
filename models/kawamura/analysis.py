"""Order parameters, Binder ratios, and helicity for Kawamura's film model.

Chirality, doubled-phase, and spin-nematic Binder ratios follow
Kawamura, Phys. Rev. Lett. 82, 964 (1999), Eqs. (4)-(6).

Only the x-direction helicity is measured by the current GPU kernels.
"""

import matplotlib.pyplot as plt
import numpy as np

from utils.Analysis.bkt import fit_weber_minnhagen
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
        # cx = sum of cosine-weighted x-bonds; qx = sum of sine-weighted x-bonds.
        # The covariance term uses <qx**2> - <qx>**2, not a sum of bond variances.
        helicity = (
            self.J * m["cx"]
            - self.J**2 / self.temperatures * (m["qx2"] - m["qx"]**2)
        ) / L**2

        return np.stack((
            m["energy"],
            np.sqrt(m["tau2"]),
            np.sqrt(m["phase2"]),
            np.sqrt(m["spin2"]),
            0.5 * (3 - m["tau4"] / m["tau2"]**2),
            2 - m["phase4"] / m["phase2"]**2,
            0.5 * (7 - 5 * m["spin4"] / m["spin2"]**2),
            helicity,
        ))

    def compute(self):
        results = {}
        for L in self.sizes:
            s = self.runs[L]["samples"]

            # Per-configuration squared order parameters (N=L**2).
            tau2 = s["chirality"]**2 / L**4
            phase2 = (s["px"]**2 + s["py"]**2) / L**4
            spin2 = s["spin2"] / L**4

            # Pass raw moments, not pre-averaged Binder ratios, to jackknife.
            moments = {
                "energy": s["energy"] / L**2,
                "tau2": tau2, "tau4": tau2**2,
                "phase2": phase2, "phase4": phase2**2,
                "spin2": spin2, "spin4": spin2**2,
                "cx": s["twist_cos_x"],
                "qx": s["twist_sin_x"],
                "qx2": s["twist_sin_x"]**2,
            }

            values, errors = blocked_jackknife(
                moments, lambda m: self._estimate(L, m), self.block_size
            )
            results[L] = {
                key: array
                for name, value, error in zip(OBSERVABLES, values, errors)
                for key, array in ((name, value), (name + "_err", error))
            }

        return results

    def fit_wm(self, temperature_range=None, q=1.0):
        """Fit x-helicity to Weber--Minnhagen scaling for assumed vortex charge q.

        Assumes the conventional BKT finite-size scaling form applies; a fit
        alone does not establish the transition class or the vortex charge.
        """
        if not np.isfinite(q) or q <= 0:
            raise ValueError("Vortex charge q must be positive and finite")

        T = self.temperatures
        L = np.asarray(self.sizes)
        helicity = np.stack([self.results[l]["helicity"] for l in L])
        errors = np.stack([self.results[l]["helicity_err"] for l in L])

        valid = np.all(
            np.isfinite(helicity) & np.isfinite(errors) & (errors > 0),
            axis=0,
        )
        valid &= np.isfinite(T) & (T > 0)

        if temperature_range is not None:
            low, high = temperature_range
            valid &= (T >= low) & (T <= high)

        if len(L) < 3 or np.count_nonzero(valid) < 2:
            raise ValueError("Insufficient data for Weber--Minnhagen fit")

        # q**2 * Upsilon has the ordinary-vortex 2T/pi jump normalization.
        factor = q**2
        fit = fit_weber_minnhagen(
            T[valid],
            L,
            factor * helicity[:, valid],
            factor * errors[:, valid],
            profile=False,
        )

        if fit["temperature"] is None or fit["C"] is None:
            raise ValueError(f"WM fit failed: {fit['reason']}")

        fit["q"] = q
        return fit

    def plot(self, C=None, q=1.0, show=True, names=None):
        """Separate figures, optionally with a WM correction on x-helicity.

        Returns a dictionary mapping observable names to Matplotlib figures.
        Use names=(...) to plot only a selected subset.
        """
        if not np.isfinite(q) or q <= 0:
            raise ValueError("Vortex charge q must be positive and finite")

        T = self.temperatures
        observables = {
            "binder_chirality": ("Chirality Binder cumulant", r"$U_\tau$"),
            "binder_phase": ("Doubled-phase Binder cumulant", r"$U_{P_2}$"),
            "binder_spin": ("Spin-nematic Binder cumulant", r"$U_D$"),
            "helicity": (
                "WM-corrected x-helicity" if C is not None else "x-helicity modulus",
                r"$\widetilde{\Upsilon}_{x,L}$" if C is not None else r"$\Upsilon_{x,L}$",
            ),
            "chirality": ("Chirality order parameter", r"$\sqrt{\langle m_\tau^2\rangle}$"),
            "phase": ("Doubled-phase order parameter", r"$\sqrt{\langle |P_2|^2\rangle}$"),
            "spin": ("Spin-nematic order parameter", r"$\sqrt{\langle D^2\rangle}$"),
            "energy": ("Energy per site", r"$\langle H\rangle/L^2$"),
        }

        if names is not None:
            unknown = set(names) - set(observables)
            if unknown:
                raise ValueError(f"Unknown plot observable(s): {sorted(unknown)}")
            observables = {name: observables[name] for name in names}

        colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
        figures = {}

        for name, (title, ylabel) in observables.items():
            fig, ax = plt.subplots(figsize=(7, 5))
            for index, L in enumerate(self.sizes):
                result = self.results[L]
                color = colors[index % len(colors)]
                y = result[name]
                yerr = result[name + "_err"]

                if name == "helicity" and C is not None:
                    denominator = 2 * np.log(L) + C
                    if not np.isfinite(denominator) or denominator <= 0:
                        raise ValueError(
                            f"Invalid WM correction: L={L}, C={C}"
                        )
                    factor = 1 + 1 / denominator
                    y = y / factor
                    yerr = yerr / factor

                ax.plot(T, y, color=color, linewidth=1.8, label=f"L={L}")
                ax.errorbar(
                    T, y, yerr=yerr, fmt="none", ecolor=color,
                    elinewidth=1.4, capsize=0, alpha=0.9,
                )

            if name == "helicity":
                ax.plot(
                    T, 2 * T / (np.pi * q**2), "k--", linewidth=1.3,
                    label=rf"$2T/(\pi q^2),\ q={q:g}$",
                )

            ax.set(title=title, xlabel="Temperature", ylabel=ylabel)
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
            fig.tight_layout()
            figures[name] = fig

        if show:
            plt.show()
        return figures
