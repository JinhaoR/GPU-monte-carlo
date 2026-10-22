"""CPU-only integration tests; fixtures are synthetic observables, not MC data."""

import copy
import json
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.xy.analysis import XYAnalysis
from utils.Analysis.statistics import blocked_jackknife, blocked_mean_error


def synthetic_runs(sizes=(16, 24, 32, 48), samples=800, seed=71, J=1.0):
    """Known Binder crossing and leading-WM curves, with shared T/channel noise."""
    rng = np.random.default_rng(seed)
    T = np.linspace(0.84, 0.99, 21) * J
    Tc = 0.9137 * J
    runs = {}
    for L in sizes:
        N = L**2
        # Balanced full-sample moments give an exact known central crossing;
        # shuffled blocks still carry nonzero sampling uncertainty.
        z = np.resize(np.array([-1.0, 1.0]), samples).reshape(samples, 1)
        rng.shuffle(z, axis=0)
        binder = 0.4 + (0.25 + L / 80) * (Tc - T) / J
        amplitude = np.sqrt(1 - 2 * binder)
        m2 = (0.5 * N)**2 * (1 + amplitude * z)
        phase = rng.uniform(-np.pi, np.pi, (samples, 1))
        critical = 2 * Tc / np.pi * (1 + 1 / (2 * np.log(L) + 1.8))
        rho = critical + (-0.3 - 0.15 * np.log(L)) * (T - Tc)
        q = 0.04 * np.sqrt(N * T) / J * rng.normal(size=(samples, len(T)))
        noise = 0.002 * J * (0.5 * z + rng.normal(size=(samples, 1)))
        c = N / J * (rho + 0.0016 + noise) + 0 * q
        runs[L] = {
            "metadata": {"L": L, "temperatures": T, "sample_interval": 50,
                         "model_parameters": {"J": J}, "seed": seed + L},
            "samples": {
                "energy": N * (-1.4 * J + noise + 0 * q),
                "mx": np.sqrt(m2) * np.cos(phase),
                "my": np.sqrt(m2) * np.sin(phase),
                "twist_cos_x": c, "twist_sin_x": q,
            },
        }
    return runs


class XYAnalysisTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def analysis(self, runs=None, **kwargs):
        return XYAnalysis(synthetic_runs() if runs is None else runs,
                          block_size=20, n_resamples=20, **kwargs)

    def test_existing_observables_and_corrected_errors_are_preserved(self):
        runs = synthetic_runs(J=1.7)
        analysis = self.analysis(runs)
        for L, run in runs.items():
            s, T, N, J = run["samples"], analysis.temperatures, L**2, analysis.J
            m2 = s["mx"]**2 + s["my"]**2
            references = {
                "energy": blocked_mean_error(s["energy"] / N, 20),
                "magnetization": blocked_mean_error(np.sqrt(m2) / N, 20),
                "binder": blocked_jackknife(
                    {"m2": m2, "m4": m2**2},
                    lambda m: 1 - m["m4"] / (2 * m["m2"]**2), 20),
                "helicity": blocked_jackknife(
                    {"c": s["twist_cos_x"], "q": s["twist_sin_x"],
                     "q2": s["twist_sin_x"]**2},
                    lambda m: (J * m["c"] - J**2 / T * (m["q2"] - m["q"]**2)) / N, 20),
            }
            self.assertEqual(set(analysis.results[L]), {
                name + suffix for name in references for suffix in ("", "_err")})
            for name, (mean, error) in references.items():
                np.testing.assert_allclose(analysis.results[L][name], mean, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(analysis.results[L][name + "_err"], error,
                                           rtol=1e-11, atol=1e-14)

    def test_all_size_pairs_and_minimum_size_fits(self):
        analysis = self.analysis()
        result = analysis.estimate_transitions()
        self.assertIs(result, analysis.estimate_transitions())
        self.assertEqual(len(result["binder_crossings"]), 6)
        self.assertEqual(len(result["universal_jump_crossings"]), 4)
        self.assertEqual([fit["min_size"] for fit in result["weber_minnhagen"]], [16, 24])
        for fit in result["weber_minnhagen"]:
            self.assertAlmostEqual(fit["temperature"], 0.9137, delta=0.004)
            draws = fit["bootstrap_temperatures"]
            if np.all(np.isfinite(draws)):
                self.assertAlmostEqual(fit["error"], draws.std(ddof=1))
            else:
                self.assertIsNone(fit["error"])
        for pair in result["binder_crossings"]:
            self.assertAlmostEqual(pair["crossings"][0], 0.9137, delta=0.025)

    def test_joint_crossing_errors_and_ambiguous_curves(self):
        analysis = self.analysis(synthetic_runs(sizes=(16, 24)))
        T = analysis.temperatures
        offsets = np.linspace(-0.01, 0.01, 20)[:, None]
        record = analysis._crossings(T - 0.9, 0, T - 0.9 + offsets)
        self.assertAlmostEqual(record["crossings"][0], 0.9)
        self.assertAlmostEqual(record["error"], offsets.std(ddof=1))
        ambiguous = analysis._crossings(np.cos(np.linspace(0, 4 * np.pi, len(T))), 0,
                                       np.tile(T - 0.9, (20, 1)))
        self.assertGreater(len(ambiguous["crossings"]), 1)
        self.assertIsNone(ambiguous["error"])
        missing = analysis._crossings(T - 0.9, 0, np.ones((20, len(T))))
        self.assertIsNone(missing["error"])

    def test_reproducibility_json_and_input_immutability(self):
        runs = synthetic_runs()
        before = copy.deepcopy(runs)
        first, second = self.analysis(runs), self.analysis(runs)
        report = first.summary()
        json.dumps(report, allow_nan=False)
        self.assertEqual(report, second.summary())
        self.assertNotIn("bootstrap_temperatures", json.dumps(report))
        for L in runs:
            for name in runs[L]["samples"]:
                np.testing.assert_array_equal(runs[L]["samples"][name], before[L]["samples"][name])
        self.assertEqual(first.blocking[16]["blocks"], 40)

    def test_existing_raw_plot_keeps_observables_and_error_bars(self):
        analysis = self.analysis(synthetic_runs(sizes=(16, 24)))
        with patch.object(plt, "show") as show:
            figure = analysis.plot()
            show.assert_called_once()
        for ax, name in zip(figure.axes, ("binder", "helicity")):
            self.assertEqual(len(ax.containers), 2)
            for bars, L in zip(ax.containers, analysis.sizes):
                values, errors = analysis.results[L][name], analysis.results[L][name + "_err"]
                np.testing.assert_array_equal(bars.lines[0].get_ydata(), values)
                segments = np.array(bars.lines[2][0].get_segments())
                np.testing.assert_allclose(segments[:, 0, 1], values - errors)
                np.testing.assert_allclose(segments[:, 1, 1], values + errors)
        self.assertEqual(analysis.estimate_transitions()["weber_minnhagen"], [])
        self.assertEqual(len(analysis.plot_transitions(show=False).axes), 2)

    def test_size_cut_and_temperature_window(self):
        analysis = self.analysis(min_size=24, temperature_range=(0.88, 0.95))
        fits = analysis.estimate_transitions()["weber_minnhagen"]
        self.assertEqual(len(fits), 1)
        self.assertEqual(fits[0]["min_size"], 24)
        self.assertAlmostEqual(fits[0]["temperature"], 0.9137, delta=0.004)

    def test_wm_plot_rescales_values_and_errors_with_one_shared_constant(self):
        analysis = self.analysis(min_size=24)
        T, C = analysis.temperatures, 1.8
        Tc = T[10]
        for L in analysis.sizes:
            factor = 1 + 1 / (2 * np.log(L) + C)
            analysis.results[L]["helicity"] = (
                2 * Tc / np.pi * factor - (0.3 + 0.15 * np.log(L)) * (T - Tc))
            analysis.results[L]["helicity_err"][:] = 0.001
        before = copy.deepcopy(analysis.results)
        figure = analysis.plot(show=False)
        self.assertIsNone(analysis._bootstrap)
        self.assertEqual(len(figure.axes[0].containers), 4)
        bars = figure.axes[1].containers
        self.assertEqual(len(bars), 3)
        for L, curve in zip((24, 32, 48), bars):
            factor = 1 + 1 / (2 * np.log(L) + C)
            values, errors = before[L]["helicity"] / factor, before[L]["helicity_err"] / factor
            np.testing.assert_allclose(curve.lines[0].get_ydata(), values, atol=1e-7)
            segments = np.array(curve.lines[2][0].get_segments())
            np.testing.assert_allclose(segments[:, 0, 1], values - errors, atol=1e-7)
            np.testing.assert_allclose(segments[:, 1, 1], values + errors, atol=1e-7)
            self.assertAlmostEqual(np.interp(Tc, T, values), 2 * Tc / np.pi, places=7)
        for L in analysis.sizes:
            for name in before[L]:
                np.testing.assert_array_equal(analysis.results[L][name], before[L][name])

    def test_zero_uncertainty_cannot_be_used_as_a_weight(self):
        analysis = self.analysis()
        rho = np.stack([analysis.results[L]["helicity"] for L in analysis.sizes])
        errors = np.stack([analysis.results[L]["helicity_err"] for L in analysis.sizes])
        errors[:, 5] = 0
        fit = analysis._fit_wm(16, rho, errors)
        self.assertIsNone(fit["temperature"])
        self.assertIn("positive finite errors", fit["reason"])

    def test_singular_bootstrap_moments_remain_explicitly_undefined(self):
        runs = synthetic_runs(sizes=(16, 24), samples=400)
        for run in runs.values():
            for name in ("mx", "my"):
                run["samples"][name][40:] = 0
        analysis = XYAnalysis(runs, block_size=20, n_resamples=100)
        with self.assertWarns(RuntimeWarning):
            report = analysis.summary()
        json.dumps(report, allow_nan=False)
        self.assertTrue(any(np.any(~np.isfinite(c["binder"])) for c in analysis._bootstrap.values()))


if __name__ == "__main__":
    unittest.main()
