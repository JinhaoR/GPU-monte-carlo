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


if __name__ == "__main__":
    unittest.main()
