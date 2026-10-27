"""GPU trajectory capture agrees with the actual exchange mappings."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numba import cuda

from core.parallel_tempering import ParallelTempering
from core.runtime import Runtime
from models.xy.model import XYModel
from utils.data_io import load_runs, save_runs


@unittest.skipUnless(cuda.is_available(), "A CUDA GPU is required")
class WalkerHistoryTests(unittest.TestCase):
    def make_pt(self, R=4, **kwargs):
        runtime = Runtime(XYModel(), L=4, temperatures=np.linspace(.8, 1.2, R),
                          threads=2, seed=162)
        return runtime, ParallelTempering(runtime, exchange_interval=1, seed=163,
                                         **kwargs)

    def test_recorded_positions_match_live_mappings_and_chunk_growth(self):
        runtime, pt = self.make_pt(tracked_walkers=range(4), history_stride=1)
        # Exercise chunk rollover without needing a long physical simulation.
        pt._history_chunk_rows = 4
        expected = [np.arange(4)]
        with patch.object(runtime, "sweep"):
            for _ in range(10):
                pt.sweep()
                expected.append(pt.walker_to_temperature.copy_to_host())
        history = pt.diagnostics()["walker_history"]
        self.assertEqual(len(pt._history_chunks), 3)
        np.testing.assert_array_equal(history["temperature_indices"], expected)
        np.testing.assert_array_equal(history["sweeps"], np.arange(11))
        np.testing.assert_array_equal(history["exchange_rounds"], np.arange(11))
        np.testing.assert_array_equal(history["walker_ids"], np.arange(4))

    def test_stride_final_snapshot_and_serialization(self):
        runtime, pt = self.make_pt(tracked_walkers=[3, 0], history_stride=3)
        with patch.object(runtime, "sweep"):
            for _ in range(7):
                pt.sweep()
        first = pt.diagnostics()
        history = first["walker_history"]
        np.testing.assert_array_equal(history["sweeps"], [0, 3, 6, 7])
        np.testing.assert_array_equal(history["temperature_indices"][-1],
                                      pt.walker_to_temperature.copy_to_host()[[3, 0]])
        self.assertEqual(len(pt._history_sweeps), 3)
        second = pt.diagnostics()["walker_history"]
        for key in ("sweeps", "exchange_rounds", "temperature_indices"):
            np.testing.assert_array_equal(history[key], second[key])
        with tempfile.TemporaryDirectory(prefix="pt_history_test_") as directory:
            self.assertEqual(Path(directory).resolve().parent,
                             Path(tempfile.gettempdir()).resolve())
            target = Path(directory) / "run.npz"
            save_runs(target, {4: {"samples": {}, "metadata": {
                "temperatures": runtime.temperatures, "parallel_tempering": first}}})
            saved = load_runs(target)[4]["metadata"]["parallel_tempering"]["walker_history"]
            np.testing.assert_array_equal(saved["temperature_indices"],
                                          history["temperature_indices"])

    def test_manual_exchanges_can_share_sweep_coordinate(self):
        _, pt = self.make_pt(R=2, tracked_walkers=[0, 1], history_stride=1)
        for _ in range(4):
            pt.attempt_exchanges()
        diagnostic = pt.diagnostics()
        history = diagnostic["walker_history"]
        np.testing.assert_array_equal(history["sweeps"], 0)
        np.testing.assert_array_equal(history["temperature_indices"],
                                      [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
        np.testing.assert_array_equal(diagnostic["walker_round_trips"], [2, 2])

    def test_disabled_history_has_no_buffers_or_schema_change(self):
        _, pt = self.make_pt()
        self.assertEqual(pt._history_chunks, [])
        self.assertNotIn("walker_history", pt.diagnostics())

    def test_invalid_recording_settings(self):
        runtime = Runtime(XYModel(), L=4, temperatures=[.8, 1.0, 1.2], threads=2)
        for ids in ([], [0, 0], [-1], [3], [1.5], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                ParallelTempering(runtime, tracked_walkers=ids)
        for stride in (0, -1, 1.5, True):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                ParallelTempering(runtime, tracked_walkers=[0], history_stride=stride)


if __name__ == "__main__":
    unittest.main()
