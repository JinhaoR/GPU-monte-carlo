"""Workflow checks use temporary records; GPU checks are tiny parity runs only."""

import contextlib
import io
from itertools import product
from pathlib import Path
import re
import runpy
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

from core.parallel_tempering import ParallelTempering
from core.runtime import Runtime
from core.simulation import Simulation
from main import he3 as simulation_script
from analysis import he3 as analysis_script
from models.he3.analysis import He3Analysis
from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from utils.data_io import _json_ready, load_runs, save_runs
from utils.temperature_ladder import make_temperature_ladder


def record(L=4, pressure=20.0, g_d=0.001):
    rng = np.random.default_rng(100 + L)
    shape, N = (12, 3), L**2
    density = N * rng.uniform(1, 1.2, shape)
    return {
        "metadata": {
            "L": L, "temperatures": [0.3, 0.4, 0.5], "seed": 42 + L,
            "model_parameters": He3Model(pressure=pressure, g_d=g_d).parameters(),
            "thermalization_sweeps": 2, "measurement_sweeps": 24, "sample_interval": 2,
            "sample_sweeps": np.arange(4, 27, 2),
            "timing": {"simulation_seconds": 3.0},
            "parallel_tempering": {
                "exchange_interval": 2, "exchange_seed": 43 + L,
                "diagnostics_scope": "since_wrapper_creation_including_thermalization",
                "sweeps": 26, "exchange_rounds": 13,
                "attempts": [7, 6], "accepted": [4, 3],
                "walker_to_temperature": [1, 0, 2], "temperature_to_walker": [1, 0, 2],
                "walker_round_trips": [0, 0, 0],
                "walker_round_trip_total_sweeps": [0, 0, 0],
                "walker_history": {
                    "walker_ids": [0, 1, 2], "sweeps": [0, 26],
                    "temperature_indices": [[0, 1, 2], [1, 0, 2]],
                },
            },
        },
        "samples": {
            "energy": -N * rng.uniform(1, 2, shape), "density": density,
            "chirality": density * rng.uniform(-1, 1, shape),
            "phase_real": N * rng.uniform(0.2, 0.4, shape),
            "phase_imag": N * rng.uniform(-0.1, 0.1, shape),
            "current_x": rng.normal(size=shape), "current_y": rng.normal(size=shape),
            "curvature_x": N * rng.uniform(2, 3, shape),
            "curvature_y": N * rng.uniform(2, 3, shape),
            "spin_alignment": rng.uniform(0.1, 0.8, shape),
        },
    }


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data" / "he3"
        self.output_root = self.root / "analysis_results" / "he3"
        self.stdout = io.StringIO()
        redirect = contextlib.redirect_stdout(self.stdout)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.addCleanup(plt.close, "all")
        for target, settings in (
            (simulation_script, dict(pressures=[20.0], g_d_values=[.001], sizes=[4, 6],
                temperature_ladder=dict(T_min=.3, T_max=.5, n_T=3, Tc=None),
                campaign="scan", data_root=self.data_root)),
            (analysis_script, dict(output_root=self.output_root, run_pt=False,
                run_joint=False, run_thermo=False, joint_L=6, joint_temperature=.401,
                joint_bins=4, block_size=2, figure_dpi=25)),
        ):
            setting_patch = patch.multiple(target, **settings)
            setting_patch.start()
            self.addCleanup(setting_patch.stop)
        self.runtime = self.enterContext(patch.object(simulation_script, "He3GaussianRuntime"))
        self.pt = self.enterContext(patch.object(simulation_script, "ParallelTempering"))
        self.simulation = self.enterContext(patch.object(simulation_script, "Simulation"))
        self.runtime.return_value.sampling_diagnostics.return_value = {}
        self.simulation.return_value.run.side_effect = lambda: record()

    def write_dataset(self, sizes=(4, 6, 8)):
        runs = {L: record(L) for L in sizes}
        for L, run in runs.items():
            run["metadata"].update(campaign={"id": "scan", "sizes": list(sizes)},
                sampler="gaussian", sampler_parameters={"rho": -.9, "proposal_mass": 0.0})
            path = simulation_script._output_path(20, .001, L)
            simulation_script._save_size(path, L, run)
        return path.parent, runs

    def test_paths_are_stable_distinct_and_sampler_and_campaign_specific(self):
        path = simulation_script._output_path(20.0, .001, 64)
        self.assertEqual(path, self.data_root / "scan" / "P=20_gd=0.001_gaussian" / "L=64.npz")
        self.assertEqual(path, simulation_script._output_path(20, 1e-3, 64))
        self.assertNotEqual(path, simulation_script._output_path(20, .0010000000000000002, 64))
        with patch.object(simulation_script, "sampler", "cartesian"):
            self.assertNotEqual(path, simulation_script._output_path(20, .001, 64))
        with patch.object(simulation_script, "campaign", "another"):
            self.assertNotEqual(path, simulation_script._output_path(20, .001, 64))
        for campaign in ("", ".", "..", "../scan", "scan\\other", "scan."):
            with self.subTest(campaign=campaign), patch.object(simulation_script, "campaign", campaign):
                with self.assertRaises(ValueError):
                    simulation_script._output_path(20, .001, 64)
        for value in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                simulation_script._output_path(value, .001, 64)

    def test_sequential_combinations_save_before_next_case_and_keep_metadata(self):
        expected = list(product([10, 20], [0, .001], [4, 6]))
        calls = []
        def simulate():
            for pressure, g_d, L in calls:
                path = simulation_script._output_path(pressure, g_d, L)
                self.assertEqual(list(load_runs(path)), [L])
            pressure, g_d, L = expected[len(calls)]
            options = self.runtime.call_args.kwargs
            self.assertEqual(options["L"], L)
            self.assertEqual(options["model"].pressure, pressure)
            self.assertAlmostEqual(float(options["model"].g_d), g_d)
            self.assertEqual(options["seed"], 42 + L)
            self.assertEqual(self.pt.call_args.kwargs["seed"], 43 + L)
            self.assertEqual(list(self.pt.call_args.kwargs["tracked_walkers"]), [0, 1, 2])
            calls.append((pressure, g_d, L))
            return record(L, pressure, g_d)
        self.simulation.return_value.run.side_effect = simulate
        with patch.multiple(simulation_script, pressures=[10, 20], g_d_values=[0, .001]):
            paths = simulation_script.main()
        self.assertEqual(calls, expected)
        self.assertEqual(len(paths), 8)
        for path in paths:
            metadata = next(iter(load_runs(path).values()))["metadata"]
            self.assertEqual(metadata["campaign"], {"id": "scan", "sizes": [4, 6]})
            self.assertEqual(metadata["sampler"], "gaussian")
            self.assertEqual(metadata["sampler_parameters"], {"rho": -.9, "proposal_mass": 0.0})
            self.assertEqual(metadata["threads"], 16)
            self.assertGreaterEqual(metadata["timing"]["simulation_seconds"], 0)
        self.assertIn("He3 Simulation - scan", self.stdout.getvalue())
        self.assertIn("Saved to:", self.stdout.getvalue())

    def test_preflight_refuses_existing_or_duplicate_outputs_before_simulating(self):
        path = simulation_script._output_path(20, .001, 6)
        simulation_script._save_size(path, 6, record(6))
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            simulation_script.main()
        self.simulation.assert_not_called()
        self.assertEqual(path.read_bytes(), original)
        with patch.multiple(simulation_script, campaign="duplicate", sizes=[4, 4]):
            with self.assertRaises(ValueError):
                simulation_script.main()

    def test_save_is_exclusive_atomic_and_does_not_leave_partial_files(self):
        path = self.root / "L=4.npz"
        with patch.object(simulation_script, "save_runs", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                simulation_script._save_size(path, 4, record())
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.glob(".he3-save-*")), [])
        link = simulation_script.os.link
        def competing_save(source, destination):
            Path(destination).write_bytes(b"other result")
            link(source, destination)
        with patch.object(simulation_script.os, "link", side_effect=competing_save):
            with self.assertRaises(FileExistsError):
                simulation_script._save_size(path, 4, record())
        self.assertEqual(path.read_bytes(), b"other result")

    def test_completed_size_survives_failure_of_the_next_size(self):
        self.simulation.return_value.run.side_effect = [record(), RuntimeError("stopped")]
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            simulation_script.main()
        path = simulation_script._output_path(20, .001, 4)
        self.assertEqual(list(load_runs(path)), [4])
        self.assertFalse(simulation_script._output_path(20, .001, 6).exists())
        with self.assertRaisesRegex(ValueError, "missing sizes \\[6\\]"):
            analysis_script._load_dataset(path.parent)

    def test_directory_and_legacy_loading_preserve_analysis_inputs(self):
        directory, expected = self.write_dataset()
        loaded = analysis_script._load_dataset(directory)
        self.assertEqual(list(loaded), [4, 6, 8])
        for L in loaded:
            for name, values in expected[L]["samples"].items():
                np.testing.assert_array_equal(loaded[L]["samples"][name], values)
        direct = He3Analysis(expected, block_size=2)
        reconstructed = He3Analysis(loaded, block_size=2)
        for L in expected:
            for name in direct.results[L]:
                np.testing.assert_array_equal(direct.results[L][name], reconstructed.results[L][name])
        for run in expected.values():
            for key in ("campaign", "sampler", "sampler_parameters"):
                run["metadata"].pop(key)
        legacy = self.root / "legacy.npz"
        save_runs(legacy, expected)
        self.assertEqual(list(analysis_script._load_dataset(legacy)), [4, 6, 8])

    def test_incompatible_parameters_temperatures_campaigns_and_samples_are_rejected(self):
        changes = (
            lambda run: run["metadata"]["model_parameters"].update(g_d=.1),
            lambda run: run["metadata"].update(temperatures=[.3, .41, .5]),
            lambda run: run["metadata"].update(sampler="cartesian"),
            lambda run: run["metadata"]["campaign"].update(id="other"),
            lambda run: run["metadata"]["sampler_parameters"].update(rho=0),
            lambda run: run["samples"].update(energy=np.zeros((3, 2))),
            lambda run: run["metadata"].update(L=10),
        )
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                directory = self.root / f"incompatible-{index}"
                for L in (4, 6):
                    run = record(L)
                    run["metadata"].update(campaign={"id": "scan", "sizes": [4, 6]},
                                           sampler="gaussian", sampler_parameters={"rho": -.9})
                    if L == 6:
                        change(run)
                    simulation_script._save_size(directory / f"L={L}.npz", L, run)
                with self.assertRaises(ValueError):
                    analysis_script._load_dataset(directory)

    def test_duplicate_empty_misnamed_and_multi_size_directory_files_are_rejected(self):
        directory, _ = self.write_dataset()
        save_runs(directory / "copy.npz", {4: record()})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analysis_script._load_dataset(directory)
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            analysis_script._load_dataset(empty)
        wrong = self.root / "wrong"
        simulation_script._save_size(wrong / "L=6.npz", 4, record())
        with self.assertRaisesRegex(ValueError, "Filename"):
            analysis_script._load_dataset(wrong)
        multi = self.root / "multi"
        save_runs(multi / "old.npz", {4: record(), 6: record(6)})
        with self.assertRaisesRegex(ValueError, "one lattice size"):
            analysis_script._load_dataset(multi)

    def test_timing_and_runtime_options_are_preserved(self):
        self.runtime.return_value.sampling_diagnostics.return_value = {"rho": -.8, "proposal_mass": .2}
        with patch.multiple(simulation_script, sizes=[4], gaussian_rho=-.8,
                gaussian_proposal_mass=.2, threads=2, seed_base=100, pt_seed_base=200,
                tracked_walkers=[0, 2], history_stride=3, exchange_interval=4,
                thermalization_sweeps=7, measurement_sweeps=18, sample_interval=2), \
             patch.object(simulation_script, "perf_counter", side_effect=[10, 12.5, 20, 20.25]):
            paths = simulation_script.main()
        self.assertEqual(self.runtime.call_args.kwargs["seed"], 104)
        self.assertEqual(self.runtime.call_args.kwargs["rho"], -.8)
        self.assertEqual(self.runtime.call_args.kwargs["proposal_mass"], .2)
        self.assertEqual(self.pt.call_args.kwargs["seed"], 204)
        self.assertEqual(self.pt.call_args.kwargs["tracked_walkers"], (0, 2))
        self.assertEqual(self.pt.call_args.kwargs["history_stride"], 3)
        self.assertEqual(self.pt.call_args.kwargs["exchange_interval"], 4)
        self.assertEqual(self.simulation.call_args.kwargs, dict(thermalization_sweeps=7,
                         measurement_sweeps=18, sample_interval=2))
        metadata = load_runs(paths[0])[4]["metadata"]
        self.assertEqual(metadata["timing"], {"simulation_seconds": 2.5})
        self.assertEqual(metadata["sampling"], {"rho": -.8, "proposal_mass": .2})

    def test_three_analysis_switches_are_independent(self):
        directory, _ = self.write_dataset()
        for switches in product((False, True), repeat=3):
            with self.subTest(switches=switches), \
                 patch.multiple(analysis_script, input_dir=directory,
                    run_pt=switches[0], run_joint=switches[1], run_thermo=switches[2]), \
                 patch.object(analysis_script, "PTAnalysis") as pt, \
                 patch.object(analysis_script, "plot_joint_histogram", return_value={}) as joint, \
                 patch.object(analysis_script, "He3Analysis") as thermo, \
                 patch.object(plt, "show"):
                pt.return_value.format_summary.return_value = "PT summary"
                pt.return_value.plot.return_value = None
                pt.return_value.plot_walkers.return_value = None
                thermo.return_value.fit_wm.return_value = dict(
                    temperature=.39, C=1.0, degrees_of_freedom=3, chi2=2)
                report = analysis_script.main()
                self.assertTrue(report.exists())
                self.assertEqual(pt.call_count, int(switches[0]))
                self.assertEqual(pt.return_value.plot.call_count, int(switches[0]))
                self.assertEqual(pt.return_value.plot_walkers.call_count, int(switches[0]))
                self.assertEqual(joint.call_count, 2 * int(switches[1]))
                self.assertEqual(thermo.call_count, int(switches[2]))
                self.assertEqual(thermo.return_value.fit_wm.call_count, int(switches[2]))
                self.assertEqual(thermo.return_value.plot.call_count, int(switches[2]))

    def test_report_figures_output_paths_and_disabled_sections(self):
        directory, _ = self.write_dataset()
        fit = dict(temperature=.39, C=1.0, degrees_of_freedom=3, chi2=2)
        with patch.multiple(analysis_script, input_dir=directory, run_pt=True,
                run_joint=True, run_thermo=True, wm_temperature_range=(.3, .5)), \
             patch.object(He3Analysis, "fit_wm", return_value=fit) as fitting, \
             patch.object(plt, "show") as show:
            report = analysis_script.main()
            fitting.assert_called_once_with(temperature_range=(.3, .5), q=.5)
            show.assert_called_once()
        self.assertEqual(report.parent, self.output_root / "scan" / directory.name)
        text = report.read_text(encoding="utf-8")
        self.assertIn(str(directory), text)
        self.assertIn("**System sizes:** [4, 6, 8]", text)
        self.assertIn("- System size: L = 6", text)
        images = re.findall(r"\]\(\./([^)]*)\)", text)
        self.assertEqual(len(images), 11)
        for name in images:
            self.assertTrue((report.parent / name).is_file(), name)
        self.assertIn("joint_L6_energy_chirality_joint_heatmap.png", images)
        self.assertIn("spin_alignment.png", images)
        plt.close("all")
        with patch.object(analysis_script, "input_dir", directory):
            analysis_script.main()
        self.assertNotIn("](./", report.read_text(encoding="utf-8"))
        self.assertNotIn("Weber-Minnhagen Fit", report.read_text(encoding="utf-8"))
        with patch.object(analysis_script, "output_root", Path("analysis_results/he3")):
            self.assertEqual(analysis_script._output_directory("data/he3/scan/P=20_gd=0.001_gaussian"),
                             Path("analysis_results/he3/scan/P=20_gd=0.001_gaussian"))

    def test_joint_normalizations_and_valid_sample_mask_are_unchanged(self):
        run = record(4)
        run["samples"]["energy"][0, 1] = np.nan
        run["samples"]["density"][1, 1] = 0
        histogram = analysis_script.joint_histogram
        with patch.object(analysis_script, "_load_dataset", return_value={4: run}), \
             patch.multiple(analysis_script, run_joint=True, joint_L=4), \
             patch.object(analysis_script, "joint_histogram", wraps=histogram) as calculate, \
             patch.object(analysis_script, "plot_joint_histogram", return_value={}):
            analysis_script.main()
        samples = run["samples"]
        x, chi = calculate.call_args_list[0].args
        np.testing.assert_array_equal(x, samples["energy"][2:, 1] / 16)
        np.testing.assert_array_equal(chi, samples["chirality"][2:, 1] / samples["density"][2:, 1])
        np.testing.assert_array_equal(calculate.call_args_list[1].args[1],
            np.hypot(samples["phase_real"][2:, 1], samples["phase_imag"][2:, 1]) / 16)

    def test_entry_scripts_are_import_safe_and_use_their_top_level_settings(self):
        with patch("core.simulation.Simulation") as simulate, \
             patch("utils.data_io.load_runs") as load:
            sim_settings = runpy.run_path("main/he3.py", run_name="experiment_settings")
            ana_settings = runpy.run_path("analysis/he3.py", run_name="experiment_settings")
            simulate.assert_not_called()
            load.assert_not_called()
        self.assertEqual(sim_settings["pressures"], [20])
        self.assertEqual(sim_settings["g_d_values"], [.001])
        self.assertEqual(sim_settings["betas"], (-.5, 1, 1, 1, -1))
        self.assertEqual(sim_settings["seed_base"], 42)
        self.assertEqual(sim_settings["history_stride"], 10)
        self.assertEqual([ana_settings[key] for key in ("run_pt", "run_joint", "run_thermo")],
                         [False, True, False])
        # Exercise the same __main__ entry block used by an IDE; stop at the
        # mocked simulation, before any GPU work or output-file creation.
        with patch("models.he3.sampling.He3GaussianRuntime"), \
             patch("core.parallel_tempering.ParallelTempering"), \
             patch("core.simulation.Simulation") as direct:
            direct.return_value.run.side_effect = RuntimeError("IDE entry reached")
            with self.assertRaisesRegex(RuntimeError, "IDE entry reached"):
                runpy.run_path("main/he3.py", run_name="__main__")
            direct.return_value.run.assert_called_once()
        directory, _ = self.write_dataset()
        with patch.dict(ana_settings["main"].__globals__, input_dir=directory,
                output_root=self.output_root, run_pt=False, run_joint=False, run_thermo=False,
                load_runs=load_runs):
            self.assertTrue(ana_settings["main"]().is_file())


@unittest.skipUnless(cuda.is_available(), "A CUDA GPU is required")
class TinyNumericalParityTests(unittest.TestCase):
    def test_script_preserves_direct_execution_samples_and_pt_for_both_samplers(self):
        ladder = dict(T_min=.3, T_max=.5, n_T=3, Tc=None)
        for sampler, runtime_class in (("cartesian", Runtime), ("gaussian", He3GaussianRuntime)):
            with self.subTest(sampler=sampler), tempfile.TemporaryDirectory() as directory, \
                 contextlib.redirect_stdout(io.StringIO()), \
                 patch.multiple(simulation_script, pressures=[20], g_d_values=[.001], sizes=[4],
                    temperature_ladder=ladder, campaign="parity", data_root=Path(directory),
                    sampler=sampler, threads=2, thermalization_sweeps=2,
                    measurement_sweeps=8, sample_interval=2, exchange_interval=2, history_stride=1):
                paths = simulation_script.main()
                actual = analysis_script._load_dataset(paths[0])[4]
                runtime = runtime_class(He3Model(pressure=20, g_d=.001), 4,
                                        make_temperature_ladder(**ladder), threads=2, seed=46)
                pt = ParallelTempering(runtime, exchange_interval=2, seed=47,
                                       tracked_walkers=range(3), history_stride=1)
                expected = Simulation(pt, thermalization_sweeps=2, measurement_sweeps=8,
                                      sample_interval=2).run()
                for name in expected["samples"]:
                    np.testing.assert_array_equal(actual["samples"][name], expected["samples"][name])
                self.assertEqual(actual["metadata"]["parallel_tempering"],
                                 _json_ready(expected["metadata"]["parallel_tempering"]))
                self.assertEqual(actual["metadata"]["seed"], 46)
                self.assertGreaterEqual(actual["metadata"]["timing"]["simulation_seconds"], 0)
                if sampler == "gaussian":
                    self.assertEqual(actual["metadata"]["sampling"], runtime.sampling_diagnostics())


if __name__ == "__main__":
    unittest.main()
