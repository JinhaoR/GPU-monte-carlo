"""XY thermodynamic observables, statistical errors, and BKT estimates."""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from utils.Analysis.statistics import block_means, bootstrap_block_means, bootstrap_error, jackknife_from_blocks
from utils.Analysis.crossings import linear_crossings, bootstrap_crossings
from utils.Analysis.bkt import fit_weber_minnhagen
from utils.data_io import load_runs


OBSERVABLES = ("energy", "magnetization", "helicity", "binder")


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items() if k != "bootstrap_temperatures"}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class XYAnalysis:
    @classmethod
    def from_file(cls, path, block_size=200, **kwargs):
        return cls(load_runs(path), block_size=block_size, **kwargs)

    def __init__(self, runs, block_size=200, n_resamples=400, seed=42,
                 temperature_range=None, min_size=None):
        self.runs = runs
        self.sizes = sorted(runs)
        metadata = runs[self.sizes[0]]["metadata"]
        self.temperatures = np.asarray(metadata["temperatures"], dtype=float)
        self.J = float(metadata["model_parameters"]["J"])
        self.block_size = block_size
        self.n_resamples = n_resamples
        self.seed = seed
        self.temperature_range = temperature_range
        self.min_size = min_size
        self.blocks = {}
        self._bootstrap = None
        self._transitions = None
        self.results = self.compute()

    def _estimate(self, L, moments):
        binder = 1 - moments["m4"] / (2 * moments["m2"]**2)
        helicity = (self.J * moments["cx"] - self.J**2 / self.temperatures
                    * (moments["qx2"] - moments["qx"]**2)) / L**2
        return np.stack((moments["energy"], moments["magnetization"], helicity, binder))

    def compute(self):
        results = {}
        for L in self.sizes:
            samples = self.runs[L]["samples"]
            mx, my = np.asarray(samples["mx"], float), np.asarray(samples["my"], float)
            qx = np.asarray(samples["twist_sin_x"], float)
            m2 = mx**2 + my**2
            moments = {
                "energy": np.asarray(samples["energy"], float) / L**2,
                "magnetization": np.sqrt(m2) / L**2,
                "m2": m2, "m4": m2**2,
                "cx": np.asarray(samples["twist_cos_x"], float),
                "qx": qx, "qx2": qx**2,
            }
            self.blocks[L] = {name: block_means(x, self.block_size)
                              for name, x in moments.items()}
            values, errors = jackknife_from_blocks(
                self.blocks[L], lambda means: self._estimate(L, means))
            results[L] = {}
            for name, value, error in zip(OBSERVABLES, values, errors):
                results[L][name] = value
                results[L][name + "_err"] = error
        return results

    def _bootstrap_curves(self):
        if self._bootstrap is None:
            self._bootstrap = {}
            for L, seed in zip(self.sizes, np.random.SeedSequence(self.seed).spawn(len(self.sizes))):
                means = bootstrap_block_means(
                    self.blocks[L], self.n_resamples, np.random.default_rng(seed))
                self._bootstrap[L] = dict(zip(OBSERVABLES, self._estimate(L, means)))
        return self._bootstrap

    def _crossings(self, left, right, left_draws, right_draws=None):
        T = self.temperatures
        if right_draws is None:
            right_draws = right
        roots = linear_crossings(T, left, right, temperature_range=self.temperature_range)
        draws = bootstrap_crossings(T, left_draws, right_draws,
                                    temperature_range=self.temperature_range)
        error = bootstrap_error(draws) if len(roots) == 1 and np.isfinite(roots[0]) else None
        return {"crossings": roots, "error": error}

    def _fit_wm(self, cutoff, helicity, errors, *, profile=False):
        sizes = np.asarray(self.sizes)
        use_L = sizes >= cutoff
        T = self.temperatures
        use_T = np.ones(len(T), bool)
        if self.temperature_range is not None:
            low, high = self.temperature_range
            use_T &= (T >= low) & (T <= high)
        use_T &= np.all(errors[use_L] > 0, axis=0)
        return fit_weber_minnhagen(
            T[use_T], sizes[use_L], helicity[use_L][:, use_T],
            errors[use_L][:, use_T], profile=profile)

    def estimate_transitions(self):
        if self._transitions is not None:
            return self._transitions
        T = self.temperatures
        boot = self._bootstrap_curves()
        binder = [{"sizes": (a, b), **self._crossings(
            self.results[a]["binder"], self.results[b]["binder"],
            boot[a]["binder"], boot[b]["binder"])}
            for a, b in combinations(self.sizes, 2)]
        jumps = [{"L": L, **self._crossings(
            self.results[L]["helicity"], 2 * T / np.pi, boot[L]["helicity"])}
            for L in self.sizes]

        helicity = np.stack([self.results[L]["helicity"] for L in self.sizes])
        errors = np.stack([self.results[L]["helicity_err"] for L in self.sizes])
        sizes = [L for L in self.sizes if self.min_size is None or L >= self.min_size]
        wm = []
        for cutoff in sizes[:-2]:
            fit = self._fit_wm(cutoff, helicity, errors, profile=True)
            fit["error"] = None
            if fit["temperature"] is not None:
                draws = np.full(self.n_resamples, np.nan)
                for b in range(self.n_resamples):
                    sampled = np.stack([boot[L]["helicity"][b] for L in self.sizes])
                    trial = self._fit_wm(cutoff, sampled, errors)
                    if trial["temperature"] is not None:
                        draws[b] = trial["temperature"]
                fit["error"] = bootstrap_error(draws)
            wm.append(fit)
        self._transitions = {
            "binder_crossings": binder,
            "universal_jump_crossings": jumps,
            "weber_minnhagen": wm,
        }
        return self._transitions

    def crossings(self):
        transitions = self.estimate_transitions()
        return {
            "binder": {tuple(x["sizes"]): x["crossings"] for x in transitions["binder_crossings"]},
            "helicity": {x["L"]: x["crossings"] for x in transitions["universal_jump_crossings"]},
        }

    def print_crossings(self):
        for kind, groups in self.crossings().items():
            print(f"\n{kind} crossings:")
            for label, roots in groups.items():
                print(f"  {label}: {np.round(roots, 5)}")

    def format_summary(self):
        lines = ["XY: blocked jackknife; joint block-bootstrap transition errors."]
        for fit in self.estimate_transitions()["weber_minnhagen"]:
            if fit["temperature"] is None:
                lines.append(f"WM L >= {fit['min_size']}: {fit['reason']}")
                continue
            error = fit["error"]
            suffix = f" ± {error:.5f}" if error is not None else " (error unresolved)"
            lines.append(f"WM L >= {fit['min_size']}: T={fit['temperature']:.6f}{suffix}, "
                         f"C={fit['C']:.4g}, chi2/dof={fit['chi2']/fit['degrees_of_freedom']:.3g}")
        return "\n".join(lines)

    def summary(self):
        return _json_ready({
            "settings": {"sizes": self.sizes, "temperatures": self.temperatures,
                         "J": self.J, "block_size": self.block_size,
                         "resamples": self.n_resamples, "seed": self.seed,
                         "temperature_range": self.temperature_range, "min_size": self.min_size},
            "observables": self.results,
            "transitions": self.estimate_transitions(),
        })

    def plot(self, show=True):
        T = self.temperatures
        fig, (ax_b, ax_h) = plt.subplots(1, 2, figsize=(12, 5))
        for L in self.sizes:
            result = self.results[L]
            ax_b.errorbar(T, result["binder"], yerr=result["binder_err"], fmt=".-", label=f"L={L}")
            ax_h.errorbar(T, result["helicity"], yerr=result["helicity_err"], fmt=".-", label=f"L={L}")
        ax_h.plot(T, 2 * T / np.pi, "k--", label=r"$2T/\pi$")
        ax_b.set(title="XY Binder cumulant", ylabel=r"$U_L$")
        ax_h.set(title="XY helicity modulus", ylabel=r"$\Upsilon_L$")
        for ax in (ax_b, ax_h):
            ax.set_xlabel("Temperature")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_transitions(self, show=True):
        transitions = self.estimate_transitions()
        fig, (ax_roots, ax_fit) = plt.subplots(1, 2, figsize=(12, 5))
        rows = ([(f"Binder {x['sizes']}", x) for x in transitions["binder_crossings"]]
                + [(f"Jump L={x['L']}", x) for x in transitions["universal_jump_crossings"]])
        for row, (label, result) in enumerate(rows):
            for root in result["crossings"]:
                if np.isfinite(root):
                    ax_roots.errorbar(root, row, xerr=result["error"], fmt="o", color="tab:blue")
        ax_roots.set_yticks(range(len(rows)), [label for label, _ in rows])
        ax_roots.invert_yaxis()
        ax_roots.set(xlabel="Crossing temperature", title="Finite-size crossings")
        for fit in transitions["weber_minnhagen"]:
            ax_fit.plot(fit["profile_temperatures"],
                        fit["profile_chi2"] / (fit["degrees_of_freedom"] + 1),
                        label=f"L >= {fit['min_size']}")
            if fit["temperature"] is not None:
                ax_fit.axvline(fit["temperature"], linestyle="--", alpha=0.5)
        ax_fit.set(xlabel="Trial temperature", ylabel=r"Profile $\chi^2/(n_L-1)$",
                   title="Weber–Minnhagen fit")
        if ax_fit.lines:
            ax_fit.legend()
        for ax in (ax_roots, ax_fit):
            ax.grid(alpha=0.3)
        fig.tight_layout()
        if show:
            plt.show()
        return fig
