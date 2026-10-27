"""CPU-only tests of saved PT endpoint diagnostics."""

import copy
import json
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.Analysis.parallel_tempering import PTAnalysis


def saved_run():
    return {
        "samples": {},
        "metadata": {
            "L": 16, "temperatures": [0.8, 0.9, 1.0],
            "parallel_tempering": {
                "exchange_interval": 10, "exchange_seed": 9,
                "sweeps": 1000, "exchange_rounds": 100, "next_parity": 0,
                "diagnostics_scope": "since_wrapper_creation_including_thermalization",
                "attempts": [50, 50], "accepted": [40, 10],
                "acceptance_rates": [0.8, 0.2],
                "walker_to_temperature": [2, 0, 1],
                "temperature_to_walker": [1, 2, 0],
                "walker_round_trips": [2, 1, 0],
                "walker_round_trip_total_sweeps": [600, 400, 0],
                "mean_round_trip_sweeps": 1000 / 3,
            },
        },
    }


class PTAnalysisTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_weighted_rates_and_completed_trip_times(self):
        run = saved_run()
        run["metadata"]["parallel_tempering"].update(attempts=[50, 25], accepted=[40, 10])
        result = PTAnalysis({16: run}).summary()[16]
        self.assertEqual(result["aggregate_acceptance"], 50 / 75)
        self.assertEqual(result["acceptance_rates"], [0.8, 0.4])
        self.assertEqual(result["walker_round_trips"], [2, 1, 0])
        self.assertEqual(result["walker_mean_round_trip_sweeps"], [300, 400, None])
        self.assertAlmostEqual(result["mean_round_trip_sweeps"], 1000 / 3)
        self.assertEqual(result["diagnostics_scope"], "since_wrapper_creation_including_thermalization")
        json.dumps(result, allow_nan=False)

    def test_unattempted_edges_and_unfinished_trips_have_no_mean(self):
        run = saved_run()
        run["metadata"]["parallel_tempering"].update(attempts=[0, 0], accepted=[0, 0],
            walker_round_trips=[0, 0, 0], walker_round_trip_total_sweeps=[0, 0, 0])
        result = PTAnalysis({16: run}).summary()[16]
        self.assertEqual(result["acceptance_rates"], [None, None])
        self.assertIsNone(result["aggregate_acceptance"])
        self.assertIsNone(result["mean_round_trip_sweeps"])

    def test_ordinary_runs_are_skipped_and_no_physics_channels_are_needed(self):
        analysis = PTAnalysis({16: saved_run(), 32: {"metadata": {}}})
        self.assertEqual(list(analysis.summary()), [16])
        self.assertIsNotNone(analysis.plot(show=False))
        ordinary = PTAnalysis({32: {"metadata": {}}})
        self.assertEqual(ordinary.summary(), {})
        self.assertIsNone(ordinary.plot(show=False))

    def test_plot_and_report_preserve_inputs_and_counter_scope(self):
        runs = {16: saved_run()}
        before = copy.deepcopy(runs)
        analysis = PTAnalysis(runs)
        figure = analysis.plot(show=False)
        self.assertEqual(runs, before)
        self.assertEqual(len(figure.axes), 2)
        self.assertIn("including thermalization", analysis.format_summary())
        self.assertIn("exclude unfinished trips", analysis.format_summary())

    def test_missing_history_produces_no_additional_plot(self):
        analysis = PTAnalysis({16: saved_run()})
        self.assertIsNone(analysis.plot_walkers(show=False))
        self.assertFalse(analysis.summary()[16]["history_available"])

    def test_heatmap_preserves_walker_ids_temperatures_and_irregular_times(self):
        run = saved_run()
        run["metadata"]["thermalization_sweeps"] = 10
        run["metadata"]["parallel_tempering"]["walker_history"] = {
            "walker_ids": [2, 0], "sweeps": [0, 10, 11],
            "temperature_indices": [[2, 0], [1, 1], [0, 2]],
            "stride": 1, "stride_units": "exchange_rounds",
        }
        before = copy.deepcopy(run)
        analysis = PTAnalysis({16: run})
        ax = analysis.plot_walkers(show=False).axes[0]
        mesh = ax.collections[0]
        np.testing.assert_array_equal(mesh.get_array(), [[.8, .9, 1.0], [1.0, .9, .8]])
        np.testing.assert_array_equal(mesh.get_coordinates()[0, :, 0], [0, 5, 10.5, 11])
        self.assertEqual([tick.get_text() for tick in ax.get_yticklabels()], ["0", "2"])
        self.assertEqual(run, before)
        self.assertTrue(analysis.summary()[16]["history_available"])

    def test_heatmap_includes_all_sixty_walkers(self):
        run = saved_run()
        temperatures = np.linspace(.36, .42, 60)
        run["metadata"]["temperatures"] = temperatures
        positions = np.stack([np.arange(60), np.roll(np.arange(60), 1)])
        run["metadata"]["parallel_tempering"]["walker_history"] = {
            "walker_ids": np.arange(60), "sweeps": [0, 100],
            "temperature_indices": positions,
        }
        figure = PTAnalysis({16: run}).plot_walkers(show=False)
        ax = figure.axes[0]
        self.assertIn("all walkers", ax.get_title())
        np.testing.assert_array_equal(ax.collections[0].get_array(), temperatures[positions].T)
        self.assertEqual(ax.collections[0].get_array().shape, (60, 2))

    def test_repeated_sweeps_keep_last_position_and_single_times_render(self):
        run = saved_run()
        run["metadata"]["parallel_tempering"]["walker_history"] = {
            "walker_ids": [0], "sweeps": [0, 0],
            "temperature_indices": [[0], [2]],
        }
        ax = PTAnalysis({16: run}).plot_walkers(show=False).axes[0]
        np.testing.assert_array_equal(ax.collections[0].get_array(), [[1.0]])
        np.testing.assert_array_equal(ax.collections[0].get_coordinates()[0, :, 0], [-.5, .5])

    def test_invalid_history_is_rejected(self):
        run = saved_run()
        for positions in ([[0], [3]], [[0.0], [1.0]], [[0, 1]]):
            with self.subTest(positions=positions):
                run["metadata"]["parallel_tempering"]["walker_history"] = {
                    "walker_ids": [0], "sweeps": [0, 10],
                    "temperature_indices": positions,
                }
                with self.assertRaises(ValueError):
                    PTAnalysis({16: run})


if __name__ == "__main__":
    unittest.main()
