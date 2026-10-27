"""Regression checks for reusable walker energies and cache invalidation."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
from numba import cuda

from core.parallel_tempering import ParallelTempering
from core.runtime import Runtime
from models.he3.model import He3Model
from models.ising.model import IsingModel
from models.kawamura.model import KawamuraModel
from models.xy.model import XYModel


@unittest.skipUnless(cuda.is_available(), "A CUDA GPU is required")
class EnergyCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", message=".*Grid size.*")

    @staticmethod
    def he3_runtime(L=6, replicas=3):
        runtime = Runtime(
            He3Model(pressure=20), L=L,
            temperatures=np.linspace(0.2, 0.5, replicas),
            threads=2, seed=211,
        )
        rng = np.random.default_rng(212)
        state = (0.25 * (rng.normal(size=(replicas, L, L, 3, 2))
                         + 1j * rng.normal(size=(replicas, L, L, 3, 2))))
        runtime.state.copy_to_device(state.astype(np.complex64))
        runtime.invalidate_energy_cache()
        return runtime

    @staticmethod
    def cache_energy(runtime):
        energies = cuda.device_array(runtime.R, dtype=np.float64)
        runtime.measure_energy(energies)
        return energies

    def assert_measurements_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for name in actual:
            with self.subTest(observable=name):
                np.testing.assert_allclose(
                    actual[name], expected[name], rtol=2e-6, atol=1e-5,
                )

    def test_he3_cached_measurements_match_full_measurements_and_keep_energy_readonly(self):
        runtime = self.he3_runtime()
        expected = runtime.model.measure(runtime.state)
        energies = self.cache_energy(runtime)
        original_energies = energies.copy_to_host()

        with patch.object(runtime.model, "measure_with_energy",
                          wraps=runtime.model.measure_with_energy) as cached:
            actual = runtime.measure()
        cached.assert_called_once()
        self.assert_measurements_equal(actual, expected)
        np.testing.assert_array_equal(actual["energy"], original_energies)
        np.testing.assert_array_equal(energies.copy_to_host(), original_energies)

    def test_he3_postexchange_measurement_uses_walker_energy_and_new_temperature_order(self):
        runtime = Runtime(He3Model(pressure=20), L=4,
                          temperatures=[0.2, 0.5], threads=2, seed=213)
        state = np.zeros((2, 4, 4, 3, 2), dtype=np.complex64)
        amplitudes = np.array([1.5, 0.5], dtype=np.float32)
        state[..., 2, 0] = amplitudes[:, None, None]
        state[..., 2, 1] = 1j * amplitudes[:, None, None]
        runtime.state.copy_to_device(state)
        runtime.invalidate_energy_cache()
        expected = runtime.model.measure(runtime.state)
        # The initially cold walker has higher energy, guaranteeing acceptance.
        self.assertGreater(expected["energy"][0], expected["energy"][1])
        pt = ParallelTempering(runtime, exchange_interval=1, seed=214)
        walker_energies = pt.energies.copy_to_host()
        pt.attempt_exchanges()
        np.testing.assert_array_equal(pt.temperature_to_walker.copy_to_host(), [1, 0])

        with patch.object(runtime.model, "measure_with_energy",
                          wraps=runtime.model.measure_with_energy) as cached:
            actual = pt.measure()
        cached.assert_called_once()
        self.assert_measurements_equal(
            actual, {name: values[::-1] for name, values in expected.items()},
        )
        np.testing.assert_array_equal(actual["energy"], walker_energies[::-1])
        np.testing.assert_array_equal(pt.energies.copy_to_host(), walker_energies)
        np.testing.assert_array_equal(runtime.snapshot(), state)

    def test_sweep_invalidates_cached_energy_before_state_updates(self):
        runtime = self.he3_runtime(L=4, replicas=2)
        old_energies = self.cache_energy(runtime).copy_to_host()
        # Accept every finite proposal so this exercises a changed state.
        runtime.betas.copy_to_device(np.zeros(runtime.R, dtype=np.float32))
        runtime.sweep()
        with patch.object(runtime.model, "measure_with_energy",
                          wraps=runtime.model.measure_with_energy) as cached:
            actual = runtime.measure()
        cached.assert_not_called()
        self.assert_measurements_equal(actual, runtime.model.measure(runtime.state))
        self.assertFalse(np.allclose(actual["energy"], old_energies))

    def test_explicit_invalidation_handles_external_state_and_hamiltonian_changes(self):
        runtime = self.he3_runtime(L=4, replicas=2)
        for change in ("state", "Hamiltonian"):
            with self.subTest(change=change):
                before = self.cache_energy(runtime).copy_to_host()
                if change == "state":
                    runtime.state.copy_to_device(
                        (1.3 * runtime.snapshot()).astype(np.complex64),
                    )
                else:
                    runtime.model.K = np.float32(1.7)
                runtime.invalidate_energy_cache()
                with patch.object(runtime.model, "measure_with_energy",
                                  wraps=runtime.model.measure_with_energy) as cached:
                    actual = runtime.measure()
                cached.assert_not_called()
                self.assert_measurements_equal(actual, runtime.model.measure(runtime.state))
                self.assertFalse(np.allclose(actual["energy"], before))

    def test_failed_sweep_cannot_reuse_energy_from_before_partial_updates(self):
        runtime = self.he3_runtime(L=4, replicas=2)
        self.cache_energy(runtime)
        changed = (1.2 * runtime.snapshot()).astype(np.complex64)

        def fail_after_mutation(*args, **kwargs):
            runtime.state.copy_to_device(changed)
            raise RuntimeError("interrupted update")

        with patch.object(runtime.model, "update_color", side_effect=fail_after_mutation):
            with self.assertRaisesRegex(RuntimeError, "interrupted update"):
                runtime.sweep()
        with patch.object(runtime.model, "measure_with_energy",
                          wraps=runtime.model.measure_with_energy) as cached:
            actual = runtime.measure()
        cached.assert_not_called()
        self.assert_measurements_equal(actual, runtime.model.measure(runtime.state))

    def test_failed_energy_refresh_cannot_publish_a_partially_overwritten_cache(self):
        runtime = self.he3_runtime(L=4, replicas=2)
        energies = self.cache_energy(runtime)
        expected = runtime.model.measure(runtime.state)

        def fail_after_overwrite(state, output):
            output.copy_to_device(np.full(runtime.R, 12345.0, dtype=np.float64))
            raise RuntimeError("interrupted energy calculation")

        with patch.object(runtime.model, "measure_energy", side_effect=fail_after_overwrite):
            with self.assertRaisesRegex(RuntimeError, "interrupted energy calculation"):
                runtime.measure_energy(energies)
        with patch.object(runtime.model, "measure_with_energy",
                          wraps=runtime.model.measure_with_energy) as cached:
            actual = runtime.measure()
        cached.assert_not_called()
        self.assert_measurements_equal(actual, expected)

    def test_ising_xy_kawamura_reuse_energy_and_preserve_measurements_and_ordering(self):
        rng = np.random.default_rng(215)
        for model in (XYModel(J=1.3), IsingModel(J=0.9), KawamuraModel(J=1.1)):
            for L in (4, 6, 20):
                with self.subTest(model=type(model).__name__, L=L):
                    runtime = Runtime(model, L=L, temperatures=[0.4, 0.6, 0.8],
                                      threads=2, seed=216)
                    if isinstance(model, XYModel):
                        state = rng.uniform(-np.pi, np.pi, (3, L, L)).astype(np.float32)
                    elif isinstance(model, IsingModel):
                        state = rng.choice([-1, 1], size=(3, L, L)).astype(np.int8)
                    else:
                        state = np.empty((3, L, L, 5), dtype=np.float32)
                        state[..., 0] = rng.uniform(-np.pi, np.pi, (3, L, L))
                        state[..., 1] = rng.choice([-1, 1], size=(3, L, L))
                        spin = rng.normal(size=(3, L, L, 3))
                        state[..., 2:] = spin / np.linalg.norm(spin, axis=-1, keepdims=True)
                    runtime.state.copy_to_device(state)
                    runtime.invalidate_energy_cache()
                    expected = model.measure(runtime.state)
                    energies = self.cache_energy(runtime)
                    original_energies = energies.copy_to_host()
                    with patch.object(model, "measure_with_energy",
                                      wraps=model.measure_with_energy) as cached:
                        actual = runtime.measure()
                    cached.assert_called_once()
                    for name in expected:
                        np.testing.assert_allclose(actual[name], expected[name],
                                                   rtol=1e-12, atol=1e-10)
                    np.testing.assert_array_equal(actual["energy"], original_energies)

                    order = np.array([2, 0, 1], dtype=np.int32)
                    mapping = cuda.to_device(order)
                    ordered = runtime.measure(temperature_indices=mapping)
                    for name in actual:
                        np.testing.assert_allclose(ordered[name][order], actual[name],
                                                   rtol=1e-12, atol=1e-10)
                    np.testing.assert_array_equal(energies.copy_to_host(), original_energies)

                    # Distinct supplied values prove that the cached path uses
                    # its input, rather than silently recomputing the energy.
                    sentinel = np.array([12345., -6789., .125])
                    energies.copy_to_device(sentinel)
                    reused = runtime.measure(temperature_indices=mapping)
                    np.testing.assert_array_equal(reused["energy"][order], sentinel)
                    for name in expected.keys() - {"energy"}:
                        np.testing.assert_allclose(reused[name][order], expected[name],
                                                   rtol=1e-12, atol=1e-10)
                    np.testing.assert_array_equal(energies.copy_to_host(), sentinel)

                    runtime.sweep()
                    with patch.object(model, "measure_with_energy",
                                      wraps=model.measure_with_energy) as cached:
                        fresh = runtime.measure()
                    cached.assert_not_called()
                    reference = model.measure(runtime.state)
                    for name in reference:
                        np.testing.assert_array_equal(fresh[name], reference[name])


if __name__ == "__main__":
    unittest.main()
