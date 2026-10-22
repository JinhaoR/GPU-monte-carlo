"""Small GPU checks against independent NumPy expressions for Kawamura's model."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

from core.parallel_tempering import ParallelTempering
from core.runtime import Runtime
from core.simulation import Simulation
from models.kawamura.analysis import KawamuraAnalysis
from models.kawamura.model import KawamuraModel
from models.kawamura.update import local_energy
from utils.data_io import load_runs, save_runs


def reference(state, J):
    theta, tau, d = state[..., 0], state[..., 1], state[..., 2:]
    phase = np.exp(1j * theta)
    bonds = []
    for axis in (1, 2):
        weight = (1 + tau * np.roll(tau, -1, axis)) * np.sum(d * np.roll(d, -1, axis), axis=-1)
        bonds.append(weight * phase * np.conj(np.roll(phase, -1, axis)))
    tensor = np.sum(d[..., :, None] * d[..., None, :] - np.eye(3) / 3, axis=(1, 2))
    return {
        "energy": -J * np.sum(bonds[0].real + bonds[1].real, axis=(1, 2)),
        "chirality": tau.sum(axis=(1, 2)),
        "px": (phase**2).real.sum(axis=(1, 2)),
        "py": (phase**2).imag.sum(axis=(1, 2)),
        "spin2": np.sum(tensor**2, axis=(1, 2)) / 3,
        "twist_cos_x": bonds[0].real.sum(axis=(1, 2)),
        "twist_sin_x": bonds[0].imag.sum(axis=(1, 2)),
    }


@cuda.jit
def local_energies(state, proposed, output, J):
    r = cuda.grid(1)
    if r < state.shape[0]:
        output[0, r] = local_energy(state, r, 1, 2, state[r, 1, 2, 0], state[r, 1, 2, 1],
            state[r, 1, 2, 2], state[r, 1, 2, 3], state[r, 1, 2, 4], J)
        output[1, r] = local_energy(state, r, 1, 2, proposed[r, 0], proposed[r, 1],
            proposed[r, 2], proposed[r, 3], proposed[r, 4], J)


@unittest.skipUnless(cuda.is_available(), "A CUDA GPU is required")
class KawamuraTests(unittest.TestCase):
    def random_state(self, L=6, replicas=3):
        rng = np.random.default_rng(63)
        state = np.empty((replicas, L, L, 5), dtype=np.float32)
        state[..., 0] = rng.uniform(-np.pi, np.pi, state.shape[:3])
        state[..., 1] = rng.choice([-1, 1], state.shape[:3])
        d = rng.normal(size=(*state.shape[:3], 3))
        state[..., 2:] = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return state

    def test_ordered_energy_and_observables(self):
        model = KawamuraModel(J=1.37)
        rt = Runtime(model, 4, [0.6, 0.9], threads=2)
        r = rt.measure()
        expected = {"energy": -4 * float(model.J) * 16, "chirality": 16,
                    "px": 16, "py": 0, "spin2": 2 * 16**2 / 9,
                    "twist_cos_x": 32, "twist_sin_x": 0}
        for name, value in expected.items():
            np.testing.assert_allclose(r[name], value, atol=1e-12)
        self.assertEqual(rt.snapshot().dtype, np.float32)
        run = {"metadata": {"temperatures": [0.6, 0.9], "model_parameters": model.parameters()},
               "samples": {name: np.tile(values, (20, 1)) for name, values in r.items()}}
        result = KawamuraAnalysis({4: run}, block_size=2).results[4]
        for name in ("binder_chirality", "binder_phase", "binder_spin"):
            np.testing.assert_allclose(result[name], 1, atol=1e-12)
        np.testing.assert_allclose(result["helicity"], 2 * float(model.J), atol=1e-12)

    def test_measurements_energy_only_and_temperature_order(self):
        model = KawamuraModel(J=1.37)
        for L in (4, 6, 20):
            state = self.random_state(L)
            device = cuda.to_device(state)
            expected = reference(state.astype(float), float(model.J))
            measured = model.measure(device)
            for name in expected:
                np.testing.assert_allclose(measured[name], expected[name], atol=1e-10, rtol=1e-12)
            energy = cuda.device_array(3, dtype=np.float64)
            model.measure_energy(device, energy)
            np.testing.assert_allclose(energy.copy_to_host(), expected["energy"], atol=1e-10)
            order = np.array([2, 0, 1], dtype=np.int32)
            ordered = model.measure(device, cuda.to_device(order))
            for name in expected:
                np.testing.assert_allclose(ordered[name][order], expected[name], atol=1e-10)

    def test_local_energy_changes_match_the_total_hamiltonian(self):
        model = KawamuraModel(J=1.37)
        state = self.random_state()
        device = cuda.to_device(state)
        old_energy = reference(state.astype(float), float(model.J))["energy"]
        for field in ("phase", "chirality", "spin"):
            proposed = state[:, 1, 2].copy()
            if field == "phase":
                proposed[:, 0] += 0.7
            elif field == "chirality":
                proposed[:, 1] *= -1
            else:
                proposed[:, 2:] = np.array([0.6, 0, 0.8])
            output = cuda.device_array((2, 3), dtype=np.float32)
            local_energies[1, 32](device, cuda.to_device(proposed), output, model.J)
            energies = output.copy_to_host()
            changed = state.copy()
            changed[:, 1, 2] = proposed
            difference = reference(changed.astype(float), float(model.J))["energy"] - old_energy
            np.testing.assert_allclose(energies[1] - energies[0], difference, atol=3e-6, rtol=1e-5)

    def test_local_gauge_symmetry(self):
        model = KawamuraModel()
        state = self.random_state()
        original = model.measure(cuda.to_device(state))
        # A simultaneous pi phase shift and d reversal at arbitrary sites is a symmetry.
        state[:, ::2, ::2, 0] += np.float32(np.pi)
        state[:, ::2, ::2, 2:] *= -1
        transformed = model.measure(cuda.to_device(state))
        for name in original:
            np.testing.assert_allclose(transformed[name], original[name], atol=1e-5)

    def test_updates_keep_constraints_and_accept_the_hot_limit(self):
        rt = Runtime(KawamuraModel(), 6, [0.6, 1.0], threads=2)
        rt.betas.copy_to_device(np.zeros(2, dtype=np.float32))
        for _ in range(20):
            rt.sweep()
        state = rt.snapshot()
        self.assertTrue(np.all(np.abs(state[..., 0]) <= np.pi + 1e-6))
        self.assertEqual(set(np.unique(state[..., 1])), {-1.0, 1.0})
        np.testing.assert_allclose(np.linalg.norm(state[..., 2:], axis=-1), 1, atol=2e-6)
        self.assertGreater(np.std(state[..., 0]), 0.1)
        self.assertGreater(np.std(state[..., 2:]), 0.1)
        cold = Runtime(KawamuraModel(), 4, [0.6, 1.0], threads=2)
        cold.betas.copy_to_device(np.full(2, 1e8, dtype=np.float32))
        for _ in range(4):
            cold.sweep()
        np.testing.assert_allclose(cold.measure()["energy"], -64, atol=1e-5)

    def test_simulation_pt_save_and_analysis(self):
        runs = {}
        for L in (4, 6):
            pt = ParallelTempering(Runtime(KawamuraModel(), L, [0.65, 0.75, 0.9], threads=2),
                                   exchange_interval=2, seed=17)
            runs[L] = Simulation(pt, thermalization_sweeps=4, measurement_sweeps=40,
                                 sample_interval=2).run()
            self.assertGreater(sum(runs[L]["metadata"]["parallel_tempering"]["attempts"]), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kawamura.npz"
            save_runs(path, runs)
            restored = load_runs(path)
            analysis = KawamuraAnalysis.from_file(path, block_size=2)
            for L in runs:
                for name in runs[L]["samples"]:
                    np.testing.assert_array_equal(restored[L]["samples"][name], runs[L]["samples"][name])
                for values in analysis.results[L].values():
                    self.assertTrue(np.isfinite(values).all())
            self.assertEqual(len(analysis.plot(show=False).axes), 4)
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
