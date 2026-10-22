import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numba import cuda

from core.parallel_tempering import ParallelTempering
from core.runtime import Runtime
from core.simulation import Simulation
from models.ising.model import IsingModel
from models.xy.model import XYModel
from utils.data_io import load_runs, save_runs


@unittest.skipUnless(cuda.is_available(), "A CUDA GPU is required")
class ParallelTemperingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", message=".*Grid size.*")

    @staticmethod
    def constant_xy(angles, exchange_interval=1):
        runtime = Runtime(
            XYModel(J=1.37), L=4,
            temperatures=np.linspace(0.6, 1.4, len(angles)), seed=31,
        )
        state = np.broadcast_to(
            np.asarray(angles, dtype=np.float32)[:, None, None],
            (len(angles), 4, 4),
        ).copy()
        runtime.state.copy_to_device(state)
        pt = ParallelTempering(
            runtime, exchange_interval=exchange_interval, seed=97
        )
        return runtime, pt, state

    def assert_mappings(self, pt, temperature_to_walker):
        expected = np.asarray(temperature_to_walker, dtype=np.int32)
        actual = pt.temperature_to_walker.copy_to_host()
        reverse = pt.walker_to_temperature.copy_to_host()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(reverse[actual], np.arange(pt.R))
        np.testing.assert_array_equal(actual[reverse], np.arange(pt.R))
        np.testing.assert_array_equal(np.sort(actual), np.arange(pt.R))
        ladder_betas = (1.0 / pt.temperatures).astype(np.float32)
        np.testing.assert_array_equal(
            pt.runtime.betas.copy_to_host(), ladder_betas[reverse]
        )

    @staticmethod
    def xy_reference(state, J):
        spins = np.exp(1j * state.astype(np.float64))
        bonds_x = spins * np.conj(np.roll(spins, -1, axis=1))
        bonds_y = spins * np.conj(np.roll(spins, -1, axis=2))
        return {
            "energy": -float(J) * np.sum(
                bonds_x.real + bonds_y.real, axis=(1, 2)
            ),
            "mx": np.sum(spins.real, axis=(1, 2)),
            "my": np.sum(spins.imag, axis=(1, 2)),
            "twist_cos_x": np.sum(bonds_x.real, axis=(1, 2)),
            "twist_sin_x": np.sum(bonds_x.imag, axis=(1, 2)),
        }

    def test_accepted_exchanges_preserve_walkers_and_order_every_channel(self):
        runtime, pt, state = self.constant_xy([0.0, 0.5, 1.0, -0.5])
        site_rng = runtime.rng_states.copy_to_host()
        reference = self.xy_reference(state, runtime.model.J)

        # All configurations have identical bond energies, so both proposals
        # are accepted independently of the exchange random numbers.
        pt.attempt_exchanges()
        order = np.array([1, 0, 3, 2])
        self.assert_mappings(pt, order)
        np.testing.assert_array_equal(runtime.snapshot(), state)
        np.testing.assert_array_equal(
            runtime.rng_states.copy_to_host(), site_rng
        )
        np.testing.assert_array_equal(pt.snapshot(), state[order])

        measured = pt.measure()
        self.assertEqual(measured.keys(), reference.keys())
        for name, expected in reference.items():
            np.testing.assert_allclose(
                measured[name], expected[order], rtol=1e-12, atol=1e-10
            )

        # The odd phase now exchanges temperatures 1 and 2; their walkers
        # are no longer the walkers with physical indices 1 and 2.
        pt.attempt_exchanges()
        self.assert_mappings(pt, [1, 3, 0, 2])
        diagnostics = pt.diagnostics()
        np.testing.assert_array_equal(diagnostics["attempts"], [1, 1, 1])
        np.testing.assert_array_equal(diagnostics["accepted"], [1, 1, 1])
        np.testing.assert_array_equal(
            diagnostics["acceptance_rates"], [1.0, 1.0, 1.0]
        )

    def test_odd_replica_count_alternates_disjoint_neighbor_pairs(self):
        _, pt, _ = self.constant_xy([0.0, 0.2, 0.4, 0.6, 0.8])
        order = np.arange(5)
        attempts = np.zeros(4, dtype=np.int64)
        for round_index in range(6):
            for temperature in range(round_index % 2, 4, 2):
                order[temperature:temperature + 2] = (
                    order[temperature:temperature + 2][::-1]
                )
                attempts[temperature] += 1
            pt.attempt_exchanges()
            self.assert_mappings(pt, order)
            diagnostics = pt.diagnostics()
            np.testing.assert_array_equal(diagnostics["attempts"], attempts)
            np.testing.assert_array_equal(diagnostics["accepted"], attempts)
            self.assertEqual(diagnostics["exchange_rounds"], round_index + 1)

    def test_exchange_block_handles_more_pairs_than_threads(self):
        # The exchange kernel has 128 threads. There are 129 even pairs here,
        # so at least one thread must process a second pair before the barrier.
        R = 259
        _, pt, _ = self.constant_xy(np.linspace(-1.0, 1.0, R))
        order = np.arange(R)
        attempts = np.zeros(R - 1, dtype=np.int64)
        for parity in (0, 1):
            for temperature in range(parity, R - 1, 2):
                order[temperature:temperature + 2] = (
                    order[temperature:temperature + 2][::-1]
                )
                attempts[temperature] += 1
            pt.attempt_exchanges()
            self.assert_mappings(pt, order)
            diagnostics = pt.diagnostics()
            np.testing.assert_array_equal(diagnostics["attempts"], attempts)
            np.testing.assert_array_equal(diagnostics["accepted"], attempts)

    def test_nonfinite_energy_prevents_swap_and_diagnostics_raise(self):
        runtime, pt, state = self.constant_xy([0.0, 1.0])
        state[0, 0, 0] = np.nan
        runtime.state.copy_to_device(state)
        pt.attempt_exchanges()
        self.assertTrue(np.isnan(pt.energies.copy_to_host()[0]))
        self.assert_mappings(pt, [0, 1])
        with self.assertRaises(FloatingPointError):
            pt.diagnostics()

    def test_acceptance_sign_and_current_total_ising_energies(self):
        runtime = Runtime(
            IsingModel(J=1.37), L=4, temperatures=[1e-6, 1e6], seed=32
        )
        pt = ParallelTempering(runtime, exchange_interval=1, seed=98)
        ordered = np.ones((4, 4), dtype=np.int8)
        checkerboard = (
            1 - 2 * (np.indices((4, 4)).sum(axis=0) % 2)
        ).astype(np.int8)
        magnitude = 2 * float(runtime.model.J) * 4**2

        # A low-energy cold configuration and high-energy hot configuration
        # make the proposed swap overwhelmingly unfavorable.
        state = np.stack([ordered, checkerboard])
        runtime.state.copy_to_device(state)
        pt.attempt_exchanges()
        np.testing.assert_array_equal(
            pt.energies.copy_to_host(), [-magnitude, magnitude]
        )
        self.assert_mappings(pt, [0, 1])
        np.testing.assert_array_equal(pt.diagnostics()["accepted"], [0])

        # Change the actual GPU states between rounds. Reusing stale energy
        # would reject this proposal; recomputing total energies accepts it.
        state = np.stack([checkerboard, ordered])
        runtime.state.copy_to_device(state)
        pt.attempt_exchanges()
        np.testing.assert_array_equal(
            pt.energies.copy_to_host(), [magnitude, -magnitude]
        )
        self.assert_mappings(pt, [1, 0])
        np.testing.assert_array_equal(pt.diagnostics()["attempts"], [2])
        np.testing.assert_array_equal(pt.diagnostics()["accepted"], [1])
        measured = pt.measure()
        np.testing.assert_array_equal(
            measured["energy"], [-magnitude, magnitude]
        )
        np.testing.assert_array_equal(measured["magnetization"], [16, 0])
        np.testing.assert_array_equal(runtime.snapshot(), state)

    def test_xy_exchange_energies_are_fresh_and_match_cpu_bonds(self):
        runtime = Runtime(
            XYModel(J=1.37), L=6, temperatures=[0.7, 1.0, 1.3], seed=33
        )
        pt = ParallelTempering(runtime, seed=99)
        rng = np.random.default_rng(34)
        for _ in range(2):
            state = rng.uniform(-np.pi, np.pi, (3, 6, 6)).astype(np.float32)
            runtime.state.copy_to_device(state)
            expected = self.xy_reference(state, runtime.model.J)
            pt.attempt_exchanges()
            np.testing.assert_allclose(
                pt.energies.copy_to_host(), expected["energy"],
                rtol=1e-12, atol=1e-10,
            )
            order = pt.temperature_to_walker.copy_to_host()
            self.assert_mappings(pt, order)
            measured = pt.measure()
            for name, values in expected.items():
                np.testing.assert_allclose(
                    measured[name], values[order], rtol=1e-12, atol=1e-10
                )

    def test_exchange_interval_and_two_walker_round_trip(self):
        runtime, pt, _ = self.constant_xy([0.0, 1.0], exchange_interval=2)
        initial = pt.diagnostics()
        self.assertIsNone(initial["mean_round_trip_sweeps"])
        np.testing.assert_array_equal(initial["walker_round_trips"], [0, 0])

        # Keep the equal-energy configurations fixed while exercising the
        # actual wrapper's sweep counter, exchange schedule, and diagnostics.
        with patch.object(runtime, "sweep") as sweep:
            for completed_sweeps in range(1, 7):
                pt.sweep()
                self.assertEqual(pt.sweeps, completed_sweeps)
                self.assertEqual(
                    pt.diagnostics()["exchange_rounds"], completed_sweeps // 2
                )
            self.assertEqual(sweep.call_count, 6)

        diagnostics = pt.diagnostics()
        np.testing.assert_array_equal(diagnostics["attempts"], [3])
        np.testing.assert_array_equal(diagnostics["accepted"], [3])
        np.testing.assert_array_equal(diagnostics["walker_round_trips"], [1, 1])
        np.testing.assert_array_equal(
            diagnostics["walker_round_trip_total_sweeps"], [4, 4]
        )
        self.assertEqual(diagnostics["mean_round_trip_sweeps"], 4.0)

    def test_interior_walker_round_trip_starts_at_first_endpoint(self):
        runtime, pt, _ = self.constant_xy([0.0, 0.5, 1.0])
        with patch.object(runtime, "sweep"):
            for _ in range(6):
                pt.sweep()
            # Walker 1 first reached the cold endpoint at sweep 1, then the
            # hot endpoint at sweep 4. That partial journey is not a trip.
            diagnostics = pt.diagnostics()
            np.testing.assert_array_equal(
                diagnostics["walker_round_trips"], [1, 0, 1]
            )
            pt.sweep()
        diagnostics = pt.diagnostics()
        np.testing.assert_array_equal(diagnostics["walker_round_trips"], [1, 1, 1])
        np.testing.assert_array_equal(
            diagnostics["walker_round_trip_total_sweeps"], [5, 6, 6]
        )

    def test_simulation_preserves_temperature_order_and_saves_diagnostics(self):
        runtime, pt, state = self.constant_xy([0.0, 1.0], exchange_interval=3)
        simulation = Simulation(
            pt, thermalization_sweeps=4,
            measurement_sweeps=7, sample_interval=2,
        )
        with patch.object(runtime, "sweep") as sweep:
            run = simulation.run()
        self.assertEqual(sweep.call_count, 11)
        np.testing.assert_array_equal(run["metadata"]["sample_sweeps"], [6, 8, 10])
        mx = self.xy_reference(state, runtime.model.J)["mx"]
        np.testing.assert_allclose(
            run["samples"]["mx"], [mx, mx, mx[::-1]], atol=1e-12
        )
        diagnostics = run["metadata"]["parallel_tempering"]
        self.assertEqual(diagnostics["exchange_interval"], 3)
        self.assertEqual(diagnostics["exchange_seed"], 97)
        self.assertEqual(diagnostics["sweeps"], 11)
        self.assertEqual(diagnostics["exchange_rounds"], 3)
        np.testing.assert_array_equal(diagnostics["attempts"], [3])
        np.testing.assert_array_equal(diagnostics["walker_round_trips"], [1, 1])
        self.assertEqual(diagnostics["mean_round_trip_sweeps"], 6.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pt_run.npz"
            save_runs(path, {4: run})
            loaded = load_runs(path)[4]
        for name, values in run["samples"].items():
            np.testing.assert_array_equal(loaded["samples"][name], values)
        saved_diagnostics = loaded["metadata"]["parallel_tempering"]
        for name, value in diagnostics.items():
            np.testing.assert_equal(saved_diagnostics[name], value)

    def test_invalid_ladders_and_exchange_intervals(self):
        for temperatures in ([1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0, 1.5]):
            with self.subTest(temperatures=temperatures):
                runtime = Runtime(IsingModel(), L=4, temperatures=temperatures)
                with self.assertRaises(ValueError):
                    ParallelTempering(runtime)

        # Distinct nominal temperatures must remain distinct in the float32
        # inverse temperatures actually consumed by the update kernels.
        runtime = Runtime(IsingModel(), L=4, temperatures=[1.0, 1.0 + 1e-10])
        with self.assertRaises(ValueError):
            ParallelTempering(runtime)

        runtime = Runtime(IsingModel(), L=4, temperatures=[1.0, 2.0])
        for interval in (0, -1, 1.5, True):
            with self.subTest(exchange_interval=interval):
                with self.assertRaises(ValueError):
                    ParallelTempering(runtime, exchange_interval=interval)


if __name__ == "__main__":
    unittest.main()
