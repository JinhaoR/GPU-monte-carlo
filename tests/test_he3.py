"""Independent NumPy checks for the thin-film He3 Hamiltonian and GPU kernels."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import complex64, cuda

from core.parallel_tempering import ParallelTempering
from core.runtime import Runtime
from core.simulation import Simulation
from models.he3.analysis import He3Analysis
from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from models.he3.update import local_energy
from utils.data_io import load_runs, save_runs


def dipole_invariant_numpy(psi):
    """Full 3x3 contraction with the absent orbital column set to zero."""
    psi = np.asarray(psi, dtype=np.complex128)
    matrix = np.zeros(psi.shape[:-2] + (3, 3), dtype=np.complex128)
    matrix[..., :, :2] = psi
    return (np.abs(np.trace(matrix, axis1=-2, axis2=-1))**2
            + np.einsum("...ij,...ji->...", matrix, matrix.conj()).real
            - (2 / 3) * np.sum(np.abs(matrix)**2, axis=(-2, -1)))


def total_energy(state, K, betas, twist=(0.0, 0.0), g_d=0.0):
    """Evaluate the original contractions, without the GPU's local expansion."""
    psi = np.asarray(state, dtype=np.complex128)
    density = np.sum(np.abs(psi)**2, axis=(-2, -1))
    Q = np.einsum("...ma,...na->...mn", psi, psi)
    G = np.einsum("...ma,...na->...mn", psi, psi.conj())
    invariants = (
        np.abs(np.trace(Q, axis1=-2, axis2=-1))**2,
        density**2,
        np.sum(np.abs(Q)**2, axis=(-2, -1)),
        np.sum(np.abs(G)**2, axis=(-2, -1)),
        np.sum((G**2).real, axis=(-2, -1)),
    )
    potential = -2 * density + sum(b * invariant for b, invariant in zip(betas, invariants))
    potential += g_d * dipole_invariant_numpy(psi)
    dx = np.exp(-1j * twist[0]) * np.roll(psi, -1, axis=1) - psi
    dy = np.exp(-1j * twist[1]) * np.roll(psi, -1, axis=2) - psi
    G1 = np.sum(np.abs(dx)**2 + np.abs(dy)**2, axis=(-2, -1))
    G2 = np.sum(np.abs(dx[..., 0] + dy[..., 1])**2, axis=-1)
    G3 = np.sum(np.abs(dx[..., 0])**2 + np.abs(dy[..., 1])**2
                + 2 * (dx[..., 1].conj() * dy[..., 0]).real, axis=-1)
    return np.sum(potential + K * (G1 + G2 + G3), axis=(1, 2))


def reference(state, K, betas, g_d=0.0):
    psi = np.asarray(state, dtype=np.complex128)
    phase = np.sum(psi[..., 0]**2 - psi[..., 1]**2, axis=-1)
    return {
        "energy": total_energy(psi, K, betas, g_d=g_d),
        "density": np.sum(np.abs(psi)**2, axis=(1, 2, 3, 4)),
        "chirality": 2 * np.sum((psi[..., 0].conj() * psi[..., 1]).imag, axis=(1, 2, 3)),
        "phase_real": phase.real.sum(axis=(1, 2)),
        "phase_imag": phase.imag.sum(axis=(1, 2)),
        "spin_alignment": 0.5 * (
            3 * np.sum(np.abs(psi[..., 2, :])**2, axis=(1, 2, 3))
            / np.sum(np.abs(psi)**2, axis=(1, 2, 3, 4)) - 1),
    }


@cuda.jit
def local_energies(state, positions, proposed, output, coefficients, K, g_d):
    r = cuda.grid(1)
    if r < state.shape[0]:
        i, j = positions[r, 0], positions[r, 1]
        old = cuda.local.array((3, 2), dtype=complex64)
        new = cuda.local.array((3, 2), dtype=complex64)
        for mu in range(3):
            for alpha in range(2):
                old[mu, alpha] = state[r, i, j, mu, alpha]
                new[mu, alpha] = proposed[r, mu, alpha]
        output[0, r] = local_energy(state, r, i, j, old, coefficients, K, g_d)
        output[1, r] = local_energy(state, r, i, j, new, coefficients, K, g_d)


class CoefficientTests(unittest.TestCase):
    def test_pressure_corrections_match_the_published_table(self):
        # Regan, Wiman & Sauls, arXiv:1908.04190, Appendix Table II.
        # These tabulated values are independent of the polynomial implementation.
        pressures = (0, 4, 10, 14, 20, 24, 30, 34)
        corrections = np.array([
            [-0.0098, -0.0419, -0.0132, -0.0047, -0.0899],
            [-0.0155, -0.0562, -0.0184, -0.0514, -0.1602],
            [-0.0231, -0.0786, -0.0226, -0.1260, -0.2324],
            [-0.0275, -0.0936, -0.0239, -0.1751, -0.2660],
            [-0.0330, -0.1160, -0.0249, -0.2419, -0.3051],
            [-0.0358, -0.1306, -0.0255, -0.2795, -0.3280],
            [-0.0391, -0.1517, -0.0265, -0.3255, -0.3611],
            [-0.0413, -0.1645, -0.0268, -0.3518, -0.3815],
        ])
        weak = np.array([-0.5, 1, 1, 1, -1])
        for pressure, correction in zip(pressures, corrections):
            np.testing.assert_allclose(He3Model(pressure=pressure).betas,
                                       weak + 0.5 * correction, atol=6e-5, rtol=0)

    def test_pressure_opt_in_temperature_scaling_and_custom_baseline(self):
        weak = np.array([-0.5, 1, 1, 1, -1], dtype=np.float32)
        np.testing.assert_array_equal(He3Model().betas, weak)
        np.testing.assert_array_equal(He3Model(pressure=20, reduced_temperature=0).betas, weak)
        self.assertFalse(np.array_equal(He3Model(pressure=0).betas, weak))
        full = np.asarray(He3Model(pressure=20).betas)
        half = np.asarray(He3Model(pressure=20, reduced_temperature=0.5).betas)
        np.testing.assert_allclose(half - weak, 0.5 * (full - weak), atol=1e-7, rtol=0)
        baseline = np.array([-0.4, 1.1, 0.9, 1.2, -0.8], dtype=np.float32)
        custom = He3Model(betas=baseline, pressure=20).betas
        np.testing.assert_allclose(np.asarray(custom) - baseline, full - weak, atol=1e-7, rtol=0)


@unittest.skipUnless(cuda.is_available(), "A CUDA GPU is required")
class He3Tests(unittest.TestCase):
    def random_state(self, L=6, replicas=3):
        rng = np.random.default_rng(253)
        shape = (replicas, L, L, 3, 2)
        return (0.3 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))).astype(np.complex64)

    def test_ordered_state_and_binder_normalizations(self):
        K, betas = 1.3, (-0.5, 1.0, 1.0, 1.0, -1.0)
        model = He3Model(K=K, betas=betas)
        rt = Runtime(model, 4, [0.6, 0.9], threads=2)
        state = rt.snapshot()
        self.assertEqual(state.dtype, np.complex64)
        B = sum(betas[i] for i in (1, 3, 4))
        n = float(np.sum(np.abs(state[0, 0, 0].astype(np.complex128))**2))
        np.testing.assert_allclose(n, 1 / B, atol=1e-7)
        measured = rt.measure()
        expected = {"energy": -2 * n * 16 + B * n**2 * 16,
                    "density": n * 16, "chirality": n * 16,
                    "phase_real": n * 16, "phase_imag": 0,
                    "current_x": 0, "current_y": 0,
                    "curvature_x": 4 * float(np.float32(K)) * n * 16,
                    "curvature_y": 4 * float(np.float32(K)) * n * 16,
                    "spin_alignment": 1}
        for name, value in expected.items():
            np.testing.assert_allclose(measured[name], value, atol=1e-6)
        run = {"metadata": {"temperatures": [0.6, 0.9], "model_parameters": model.parameters()},
               "samples": {name: np.tile(values, (20, 1)) for name, values in measured.items()}}
        result = He3Analysis({4: run}, block_size=2).results[4]
        np.testing.assert_allclose(result["binder_chirality"], 2 / 3, atol=1e-12)
        np.testing.assert_allclose(result["binder_phase"], 1, atol=1e-12)
        np.testing.assert_allclose(result["spin_alignment"], 1, atol=1e-12)
        for name in ("helicity_x", "helicity_y", "helicity"):
            np.testing.assert_allclose(result[name], 4 * float(np.float32(K)) * n, atol=1e-6)

    def test_measurements_total_energy_and_temperature_order(self):
        K = np.float32(0.73)
        betas = np.asarray((-0.4, 1.1, 0.8, 1.3, -0.7), dtype=np.float32)
        model = He3Model(K=K, betas=betas)
        for L in (4, 6, 20, 80):
            state = self.random_state(L)
            device = cuda.to_device(state)
            expected = reference(state, float(K), betas.astype(float))
            measured = model.measure(device)
            for name in expected:
                # Local algebra uses float32, while lattice sums use float64.
                # For these bounded fields, check an absolute per-site budget
                # against the independent double-precision Hamiltonian.
                budget = (8 if name == "energy" else 2) * np.finfo(np.float32).eps
                np.testing.assert_allclose(measured[name] / L**2, expected[name] / L**2,
                                           atol=budget, rtol=0, err_msg=name)
            energy = cuda.device_array(3, dtype=np.float64)
            model.measure_energy(device, energy)
            np.testing.assert_array_equal(energy.copy_to_host(), measured["energy"])
            cached = model.measure_with_energy(device, energy)
            for name in measured:
                np.testing.assert_array_equal(cached[name], measured[name])
            order = np.array([2, 0, 1], dtype=np.int32)
            ordered = model.measure(device, cuda.to_device(order))
            for name in measured:
                np.testing.assert_array_equal(ordered[name][order], measured[name])

    def test_helicity_derivatives_of_the_twisted_hamiltonian(self):
        K = np.float32(0.73)
        betas = np.asarray((-0.4, 1.1, 0.8, 1.3, -0.7), dtype=np.float32)
        state = self.random_state(L=4)
        measured = He3Model(K=K, betas=betas).measure(cuda.to_device(state))
        h = 2e-4
        energy = total_energy(state, float(K), betas.astype(float))
        for axis, direction in enumerate(("x", "y")):
            twist = np.zeros(2)
            twist[axis] = h
            plus = total_energy(state, float(K), betas.astype(float), twist)
            minus = total_energy(state, float(K), betas.astype(float), -twist)
            np.testing.assert_allclose(measured[f"current_{direction}"],
                                       (plus - minus) / (2 * h), atol=2e-6, rtol=1e-7)
            np.testing.assert_allclose(measured[f"curvature_{direction}"],
                                       (plus - 2 * energy + minus) / h**2, atol=2e-5, rtol=1e-6)

    def test_dipole_energy_local_changes_and_cached_temperature_order(self):
        state = self.random_state(L=6)
        device = cuda.to_device(state)
        positions = np.array(((0, 0), (0, 5), (5, 0)), dtype=np.int32)
        proposed = self.random_state(L=4)[:, 1, 2].copy()
        changed = state.copy()
        for r, (i, j) in enumerate(positions):
            changed[r, i, j] = proposed[r]
        order = np.array([2, 0, 1], dtype=np.int32)
        mapping = cuda.to_device(order)
        for g_d in (0.0, 0.001, 0.1):
            with self.subTest(g_d=g_d):
                model = He3Model(K=0.73, pressure=20, g_d=g_d)
                expected = reference(state, float(model.K), model.betas, float(model.g_d))
                measured = model.measure(device)
                for name in expected:
                    np.testing.assert_allclose(measured[name], expected[name],
                                               rtol=2e-6, atol=1e-5, err_msg=name)
                energies = cuda.device_array(3, dtype=np.float64)
                model.measure_energy(device, energies)
                np.testing.assert_allclose(energies.copy_to_host(), expected["energy"],
                                           rtol=2e-6, atol=1e-5)
                for actual, indices in (
                    (model.measure_with_energy(device, energies), np.arange(3)),
                    (model.measure(device, mapping), order),
                    (model.measure_with_energy(device, energies, mapping), order),
                ):
                    for name in measured:
                        np.testing.assert_array_equal(actual[name][indices], measured[name])
                local = cuda.device_array((2, 3), dtype=np.float64)
                local_energies[1, 32](device, cuda.to_device(positions),
                                     cuda.to_device(proposed), local,
                                     model.betas, model.K, model.g_d)
                local = local.copy_to_host()
                difference = (total_energy(changed, float(model.K), model.betas,
                                           g_d=float(model.g_d)) - expected["energy"])
                np.testing.assert_allclose(local[1] - local[0], difference,
                                           rtol=2e-5, atol=3e-5)

    def test_spin_alignment_normalization_and_density_weighting(self):
        state = np.zeros((5, 4, 4, 3, 2), dtype=np.complex64)
        orbital = np.array([1, 1j], dtype=np.complex64) / np.sqrt(2)
        state[0, ..., 2, :] = orbital
        state[1, ..., 0, :] = orbital
        state[2] = np.ones((3, 1)) * orbital / np.sqrt(3)
        # Half the sites have z-directed spin with three times the density;
        # the other half have x-directed spin. Average local Qzz would be 1/4.
        state[3, :2, :, 2, :] = np.sqrt(3) * orbital
        state[3, 2:, :, 0, :] = orbital
        # The fifth replica has zero density and uses Qzz=0 by convention.
        measured = He3Model(g_d=0.001).measure(cuda.to_device(state))
        np.testing.assert_allclose(measured["spin_alignment"], [1, -0.5, 0, 0.625, 0],
                                   atol=1e-7, rtol=0)

    def test_both_samplers_dipole_pt_save_and_spin_analysis(self):
        for runtime_class in (Runtime, He3GaussianRuntime):
            for g_d in (0.0, 0.001):
                with self.subTest(sampler=runtime_class.__name__, g_d=g_d):
                    model = He3Model(pressure=20, g_d=g_d)
                    runtime = runtime_class(model, 4, [0.3, 0.4, 0.6],
                                            threads=2, seed=227)
                    initial = runtime.snapshot()
                    pt = ParallelTempering(runtime, exchange_interval=2, seed=228)
                    run = Simulation(pt, thermalization_sweeps=2,
                                     measurement_sweeps=16, sample_interval=2).run()
                    self.assertTrue(np.isfinite(runtime.snapshot()).all())
                    self.assertGreater(np.max(np.abs(runtime.snapshot() - initial)), 0.01)
                    self.assertEqual(run["samples"]["spin_alignment"].shape, (8, 3))
                    for sample in run["samples"].values():
                        self.assertTrue(np.isfinite(sample).all())
                    np.testing.assert_allclose(pt.measure()["energy"],
                                               runtime.model.measure(runtime.state,
                                                   pt.walker_to_temperature)["energy"],
                                               rtol=2e-6, atol=1e-5)
                    if runtime_class is He3GaussianRuntime:
                        self.assertEqual(sum(runtime.sampling_diagnostics()[
                            "nonfinite_rejections"]), 0)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "dipole.npz"
                        save_runs(path, {4: run})
                        restored = load_runs(path)[4]
                        self.assertAlmostEqual(restored["metadata"]["model_parameters"][
                            "g_d"], g_d, places=8)
                        np.testing.assert_array_equal(restored["samples"]["spin_alignment"],
                                                      run["samples"]["spin_alignment"])
                        analysis = He3Analysis.from_file(path, block_size=2)
                        result = analysis.results[4]
                        np.testing.assert_allclose(result["spin_alignment"],
                                                   run["samples"]["spin_alignment"].mean(axis=0))
                        blocks = run["samples"]["spin_alignment"].reshape(4, 2, 3).mean(axis=1)
                        np.testing.assert_allclose(result["spin_alignment_err"],
                                                   blocks.std(axis=0, ddof=1) / 2)
                        figures = analysis.plot(show=False)
                        self.assertIn("spin_alignment", figures)
                        self.assertEqual(figures["spin_alignment"].axes[0].get_ylabel(),
                                         r"$Q_{zz}$")
                        plt.close("all")

    def test_legacy_analysis_omits_unrecorded_spin_alignment(self):
        measured = He3Model().measure(cuda.to_device(self.random_state(L=4)))
        samples = {name: np.tile(value, (8, 1)) for name, value in measured.items()
                   if name != "spin_alignment"}
        run = {"metadata": {"temperatures": [0.3, 0.4, 0.6]}, "samples": samples}
        analysis = He3Analysis({4: run}, block_size=2)
        self.assertNotIn("spin_alignment", analysis.results[4])
        figures = analysis.plot(show=False)
        self.assertNotIn("spin_alignment", figures)
        self.assertEqual(len(figures), 4)
        plt.close("all")

    def test_local_changes_include_all_gradient_cells(self):
        K = np.float32(0.73)
        betas = np.asarray((-0.4, 1.1, 0.8, 1.3, -0.7), dtype=np.float32)
        state = self.random_state()
        positions = np.asarray(((0, 0), (0, 5), (3, 2)), dtype=np.int32)
        proposed = self.random_state(L=4)[:, 1, 2].copy()
        output = cuda.device_array((2, 3), dtype=np.float64)
        local_energies[1, 32](cuda.to_device(state), cuda.to_device(positions),
                             cuda.to_device(proposed), output, cuda.to_device(betas),
                             K, np.float32(0))
        local = output.copy_to_host()
        changed = state.copy()
        for r, (i, j) in enumerate(positions):
            changed[r, i, j] = proposed[r]
        expected = (total_energy(changed, float(K), betas.astype(float))
                    - total_energy(state, float(K), betas.astype(float)))
        np.testing.assert_allclose(local[1] - local[0], expected, atol=3e-5, rtol=2e-5)

    def test_global_gauge_spin_symmetry_and_phase_candidate(self):
        model = He3Model()
        state = self.random_state()
        original = model.measure(cuda.to_device(state))
        rotation, _ = np.linalg.qr(np.random.default_rng(14).normal(size=(3, 3)))
        transformed = np.einsum("mn,...na->...ma", rotation, state)
        angle = 0.47
        transformed = (np.exp(1j * angle) * transformed).astype(np.complex64)
        measured = model.measure(cuda.to_device(transformed))
        for name in ("energy", "density", "chirality", "current_x", "current_y",
                     "curvature_x", "curvature_y"):
            np.testing.assert_allclose(measured[name], original[name], atol=2e-5, rtol=1e-6)
        old_phase = original["phase_real"] + 1j * original["phase_imag"]
        new_phase = measured["phase_real"] + 1j * measured["phase_imag"]
        np.testing.assert_allclose(new_phase, np.exp(2j * angle) * old_phase, atol=2e-6)
        # Both chiralities give the same doubled-phase order, with opposite chirality.
        chiral = np.zeros((2, 4, 4, 3, 2), dtype=np.complex64)
        chiral[..., 2, 0] = 1 / np.sqrt(2)
        chiral[0, ..., 2, 1] = 1j / np.sqrt(2)
        chiral[1, ..., 2, 1] = -1j / np.sqrt(2)
        values = model.measure(cuda.to_device(chiral))
        np.testing.assert_allclose(values["chirality"], [16, -16], atol=2e-6)
        np.testing.assert_allclose(values["phase_real"], 16, atol=2e-6)
        np.testing.assert_allclose(values["phase_imag"], 0, atol=1e-12)

    def test_updates_move_hot_fields_and_preserve_a_cold_minimum(self):
        rt = Runtime(He3Model(delta=0.15), 6, [0.6, 0.9], threads=2)
        initial = rt.snapshot()
        rt.betas.copy_to_device(np.zeros(2, dtype=np.float32))
        for _ in range(10):
            rt.sweep()
        state = rt.snapshot()
        self.assertTrue(np.isfinite(state).all())
        self.assertGreater(np.std(state - initial), 0.03)
        cold = Runtime(He3Model(delta=0.15), 4, [0.6, 0.9], threads=2)
        initial_energy = cold.measure()["energy"]
        cold.betas.copy_to_device(np.full(2, 1e8, dtype=np.float32))
        for _ in range(4):
            cold.sweep()
        np.testing.assert_allclose(cold.measure()["energy"], initial_energy, atol=1e-5)

    def test_simulation_pt_save_and_analysis(self):
        runs = {}
        for L in (4, 6):
            model = He3Model(pressure=20, reduced_temperature=0.8)
            pt = ParallelTempering(Runtime(model, L, [0.6, 0.8, 1.0], threads=2),
                                   exchange_interval=2, seed=19)
            runs[L] = Simulation(pt, thermalization_sweeps=4, measurement_sweeps=40,
                                 sample_interval=2).run()
            self.assertGreater(sum(runs[L]["metadata"]["parallel_tempering"]["attempts"]), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "he3.npz"
            save_runs(path, runs)
            restored = load_runs(path)
            analysis = He3Analysis.from_file(path, block_size=2)
            for L in runs:
                parameters = restored[L]["metadata"]["model_parameters"]
                self.assertEqual(parameters["pressure"], 20)
                self.assertAlmostEqual(parameters["reduced_temperature"], 0.8)
                np.testing.assert_array_equal(parameters["betas"],
                                              runs[L]["metadata"]["model_parameters"]["betas"])
                for name in runs[L]["samples"]:
                    np.testing.assert_array_equal(restored[L]["samples"][name], runs[L]["samples"][name])
                for values in analysis.results[L].values():
                    self.assertTrue(np.isfinite(values).all())
            self.assertEqual(len(analysis.plot(show=False)), 5)
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
