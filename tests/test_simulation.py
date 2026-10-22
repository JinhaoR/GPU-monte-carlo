import unittest
import warnings
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

from core.runtime import Runtime
from core.simulation import Simulation
from models.ising.analysis import IsingAnalysis
from models.ising.model import IsingModel
from models.xy.analysis import XYAnalysis
from models.xy.model import XYModel


@unittest.skipUnless(cuda.is_available(), "A CUDA GPU is required")
class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", message=".*Grid size.*")

    def test_ordered_xy_measurements_include_coupling(self):
        model = XYModel(J=1.37)
        runtime = Runtime(model, L=4, temperatures=[0.8, 1.0])
        measured = runtime.measure()

        expected = {
            "energy": -2 * model.J * 16,
            "mx": 16,
            "my": 0,
            "twist_cos_x": 16,
            "twist_sin_x": 0,
        }
        self.assertEqual(measured.keys(), expected.keys())
        for name, value in expected.items():
            self.assertEqual(measured[name].shape, (2,))
            self.assertEqual(measured[name].dtype, np.float64)
            np.testing.assert_allclose(measured[name], value, atol=1e-12)

    def test_xy_measurements_match_independent_cpu_reference(self):
        model = XYModel(J=1.37)
        rng = np.random.default_rng(17)

        for L in (4, 6, 32):
            with self.subTest(L=L):
                angles = rng.uniform(-np.pi, np.pi, (3, L, L))
                angles = angles.astype(np.float32)
                measured = model.measure(cuda.to_device(angles))

                # Complex spins give an independent reference for the bonds.
                spins = np.exp(1j * angles.astype(np.float64))
                bonds_x = spins * np.conj(np.roll(spins, -1, axis=1))
                bonds_y = spins * np.conj(np.roll(spins, -1, axis=2))
                expected = {
                    "energy": -model.J * np.sum(
                        bonds_x.real + bonds_y.real, axis=(1, 2)
                    ),
                    "mx": np.sum(spins.real, axis=(1, 2)),
                    "my": np.sum(spins.imag, axis=(1, 2)),
                    "twist_cos_x": np.sum(bonds_x.real, axis=(1, 2)),
                    "twist_sin_x": np.sum(bonds_x.imag, axis=(1, 2)),
                }
                for name in expected:
                    np.testing.assert_allclose(
                        measured[name], expected[name],
                        rtol=1e-12, atol=1e-10
                    )

    def test_ordered_and_alternating_ising_measurements(self):
        model = IsingModel(J=1.37)
        L = 6
        checkerboard = 1 - 2 * (np.indices((L, L)).sum(axis=0) % 2)
        spins = np.stack((np.ones((L, L)), -np.ones((L, L)), checkerboard))
        measured = model.measure(cuda.to_device(spins.astype(np.int8)))

        expected = {
            "energy": float(model.J) * L**2 * np.array([-2.0, -2.0, 2.0]),
            "magnetization": np.array([L**2, -L**2, 0]),
        }
        self.assertEqual(measured.keys(), expected.keys())
        for name, values in expected.items():
            self.assertEqual(measured[name].dtype, np.float64)
            np.testing.assert_array_equal(measured[name], values)

    def test_ising_measurements_match_cpu_bond_count(self):
        model = IsingModel(J=1.37)
        rng = np.random.default_rng(18)
        for L in (4, 6, 32, 64):
            with self.subTest(L=L):
                spins = rng.choice(np.array([-1, 1], dtype=np.int8), (3, L, L))
                measured = model.measure(cuda.to_device(spins))

                # Equal neighbors contribute -J, unequal neighbors +J.
                equal_bonds = np.sum(
                    spins == np.roll(spins, -1, axis=1), axis=(1, 2)
                ) + np.sum(
                    spins == np.roll(spins, -1, axis=2), axis=(1, 2)
                )
                energy = float(model.J) * (2 * L**2 - 2 * equal_bonds)
                np.testing.assert_array_equal(measured["energy"], energy)
                np.testing.assert_array_equal(
                    measured["magnetization"], np.sum(spins, axis=(1, 2))
                )

    def test_ising_updates_in_temperature_limits(self):
        L = 6
        runtime = Runtime(IsingModel(J=1.37), L, [1.0, 2.0], threads=2)
        rng = np.random.default_rng(19)
        expected = rng.choice(np.array([-1, 1], dtype=np.int8), (2, L, L))
        runtime.state.copy_to_device(expected)
        # Positive energy moves have probability zero at this large beta.
        # At beta=0 every proposal is accepted.
        runtime.betas.copy_to_device(np.array([1e8, 0.0], dtype=np.float32))

        for color in range(4):
            for r in range(2):
                for i in range(color // 2, L, 2):
                    for j in range(color % 2, L, 2):
                        neighbors = (
                            int(expected[r, (i - 1) % L, j])
                            + int(expected[r, (i + 1) % L, j])
                            + int(expected[r, i, (j - 1) % L])
                            + int(expected[r, i, (j + 1) % L])
                        )
                        if r == 1 or expected[r, i, j] * neighbors <= 0:
                            expected[r, i, j] *= -1

            runtime.model.update_color(
                runtime.state, runtime.rng_states, runtime.betas, color,
                runtime.blocks_per_grid, runtime.threads_per_block
            )
            np.testing.assert_array_equal(runtime.snapshot(), expected)

    def test_both_models_run_analyze_and_plot_multiple_sizes(self):
        cases = (
            (XYModel, XYAnalysis, [0.8, 1.0]),
            (IsingModel, IsingAnalysis, [1.5, 2.0]),
        )
        for model_type, analysis_type, temperatures in cases:
            with self.subTest(model=model_type.__name__):
                runs = {}
                for L in (4, 8):
                    runtime = Runtime(
                        model_type(), L, temperatures, seed=42 + L
                    )
                    simulation = Simulation(
                        runtime, thermalization_sweeps=4,
                        measurement_sweeps=102, sample_interval=5
                    )
                    with patch.object(
                        runtime, "sweep", wraps=runtime.sweep
                    ) as sweep:
                        run = simulation.run()
                    self.assertEqual(sweep.call_count, 106)
                    np.testing.assert_array_equal(
                        run["metadata"]["sample_sweeps"], np.arange(9, 105, 5)
                    )
                    for values in run["samples"].values():
                        self.assertIsInstance(values, np.ndarray)
                        self.assertEqual(values.shape, (20, 2))
                        self.assertTrue(np.all(np.isfinite(values)))
                    runs[L] = run

                analysis = (
                    analysis_type(runs, block_size=1)
                    if model_type is XYModel else analysis_type(runs)
                )
                for values in analysis.results.values():
                    for observable in values.values():
                        self.assertTrue(np.all(np.isfinite(observable)))
                with patch.object(plt, "show") as show:
                    analysis.plot()
                    show.assert_called_once()
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
