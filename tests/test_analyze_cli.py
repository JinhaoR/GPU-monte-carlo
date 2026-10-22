"""Saved-file CLI integration tests: NumPy/Matplotlib only, no GPU work."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze
from utils.data_io import save_runs


def xy_runs():
    rng = np.random.default_rng(51)
    temperatures = np.linspace(0.8, 1.0, 5)
    runs = {}
    for L in (8, 16):
        N = L**2
        shape = (60, len(temperatures))
        mx = N * (0.4 + 0.1 * rng.normal(size=shape))
        runs[L] = {
            "metadata": {"L": L, "temperatures": temperatures,
                         "model_parameters": {"J": 1.0}, "sample_interval": 10},
            "samples": {"energy": -N + rng.normal(size=shape), "mx": mx,
                        "my": N * 0.1 * rng.normal(size=shape),
                        "twist_cos_x": N * (1.1 - temperatures) + rng.normal(size=shape),
                        "twist_sin_x": rng.normal(size=shape)},
        }
    return runs


def foreign_pt_run():
    return {12: {"samples": {"other_model_order_parameter": np.ones((2, 2))},
                 "metadata": {"L": 12, "temperatures": [1.0, 2.0],
                              "parallel_tempering": {
                                  "attempts": [10], "accepted": [5],
                                  "acceptance_rates": [0.5],
                                  "walker_to_temperature": [1, 0],
                                  "temperature_to_walker": [1, 0],
                                  "walker_round_trips": [0, 0],
                                  "walker_round_trip_total_sweeps": [0, 0],
                                  "mean_round_trip_sweeps": None,
                                  "diagnostics_scope": "since_wrapper_creation_including_thermalization",
                              }}}}


class AnalyzeCLITests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_legacy_xy_export_reads_once_preserves_input_and_closes_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xy.npz"
            output = Path(directory) / "analysis"
            save_runs(path, xy_runs())
            before = path.read_bytes()
            with patch("analyze.load_runs", wraps=analyze.load_runs) as load:
                with contextlib.redirect_stdout(io.StringIO()):
                    report = analyze.main([str(path), "--block-size", "2", "--resamples", "20",
                                           "--no-show", "--output-dir", str(output)])
            load.assert_called_once_with(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertIn("xy", report)
            self.assertEqual(report["parallel_tempering"], {})
            self.assertEqual(sorted(p.name for p in output.iterdir()),
                             ["analysis.json", "xy_observables.pdf", "xy_observables.png",
                              "xy_transitions.pdf", "xy_transitions.png"])
            with (output / "analysis.json").open(encoding="utf-8") as saved:
                restored = json.load(saved)
            self.assertIn("transitions", restored["xy"])
            self.assertEqual(plt.get_fignums(), [])

    def test_pt_only_accepts_single_size_and_foreign_model_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.npz"
            save_runs(path, foreign_pt_run())
            with patch("analyze.XYAnalysis") as xy, patch("analyze.plt.show") as show:
                with contextlib.redirect_stdout(io.StringIO()):
                    report = analyze.main([str(path), "--pt-only", "--no-show"])
            xy.assert_not_called()
            show.assert_not_called()
            self.assertNotIn("xy", report)
            self.assertEqual(report["parallel_tempering"][12]["aggregate_acceptance"], 0.5)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["other.npz"])

    def test_cli_passes_settings_and_resolves_one_sided_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xy.npz"
            save_runs(path, xy_runs())
            with patch("analyze.XYAnalysis") as xy, patch("analyze.plt.show") as show:
                xy.return_value.format_summary.return_value = "XY report"
                xy.return_value.summary.return_value = {"ok": True}
                xy.return_value.plot.return_value = plt.figure()
                xy.return_value.plot_transitions.return_value = plt.figure()
                with contextlib.redirect_stdout(io.StringIO()):
                    analyze.main([str(path), "--block-size", "3", "--resamples", "21",
                                  "--seed", "13", "--t-min", "0.85", "--min-size", "16"])
                kwargs = xy.call_args.kwargs
                self.assertEqual(kwargs, {"block_size": 3, "n_resamples": 21, "seed": 13,
                                          "temperature_range": (0.85, 1.0), "min_size": 16})
                xy.return_value.plot.assert_called_once_with(show=False)
                xy.return_value.plot_transitions.assert_called_once_with(show=False)
                show.assert_called_once_with()

    def test_default_input_and_block_size_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xy.npz"
            save_runs(path, xy_runs())
            # Mock is_file/load to examine defaults without changing the cwd.
            with patch("pathlib.Path.is_file", return_value=True), \
                 patch("analyze.load_runs", return_value=xy_runs()) as load, \
                 patch("analyze.XYAnalysis") as xy:
                xy.return_value.format_summary.return_value = "XY report"
                xy.return_value.summary.return_value = {}
                xy.return_value.plot.return_value = plt.figure()
                xy.return_value.plot_transitions.return_value = plt.figure()
                with contextlib.redirect_stdout(io.StringIO()):
                    analyze.main(["--no-show"])
            load.assert_called_once_with(Path("data/xy_run.npz"))
            self.assertEqual(xy.call_args.kwargs["block_size"], 200)
            self.assertEqual(xy.call_args.kwargs["n_resamples"], 400)

    def test_missing_input_is_helpful_and_help_does_not_load(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not-yet-saved.npz"
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as exit:
                analyze.main([str(missing), "--no-show"])
            self.assertEqual(exit.exception.code, 2)
            self.assertIn("may not have saved its data yet", errors.getvalue())
        with patch("analyze.load_runs") as load, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exit:
                analyze.main(["--help"])
            self.assertEqual(exit.exception.code, 0)
            load.assert_not_called()

    def test_invalid_settings_fail_before_reading(self):
        for args in (["--block-size", "0"], ["--resamples", "2"], ["--seed", "-1"],
                     ["--t-min", "nan"], ["--t-min", "1", "--t-max", "0.8"],
                     ["--min-size", "0"]):
            with self.subTest(args=args), patch("analyze.load_runs") as load:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    analyze.main(args)
                load.assert_not_called()

    def test_truncated_saved_file_has_readable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unfinished.npz"
            save_runs(path, xy_runs())
            path.write_bytes(path.read_bytes()[:50])
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
                analyze.main([str(path), "--no-show"])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("Could not analyze", errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
