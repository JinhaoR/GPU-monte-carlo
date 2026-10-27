"""Check the direct Metropolis energy difference against the full Hamiltonian.

The reference uses the original spin-matrix invariants and all forward cells.
It deliberately does not call the retained local_energy implementation.
"""

import unittest

import numpy as np
from numba import complex64, cuda

from models.he3.model import He3Model
from models.he3.update import gradient_difference, potential


def reference_potential(psi, coefficients):
    """Spin-matrix potential and the magnitude of its component terms."""
    psi = np.asarray(psi, dtype=np.complex128)
    n = np.sum(np.abs(psi)**2, axis=(-2, -1))
    symmetric = np.einsum("...ma,...na->...mn", psi, psi)
    hermitian = np.einsum("...ma,...na->...mn", psi, psi.conj())
    invariants = (
        np.abs(np.trace(symmetric, axis1=-2, axis2=-1))**2,
        n**2,
        np.sum(np.abs(symmetric)**2, axis=(-2, -1)),
        np.sum(np.abs(hermitian)**2, axis=(-2, -1)),
        np.sum((hermitian**2).real, axis=(-2, -1)),
    )
    terms = [-2 * n] + [float(b) * invariant
                        for b, invariant in zip(coefficients, invariants)]
    return sum(terms), sum(np.abs(term) for term in terms)


def reference_energy_parts(state, coefficients):
    """Independent float64 potential and gradient totals, in walker order."""
    psi = np.asarray(state, dtype=np.complex128)
    onsite, _ = reference_potential(psi, coefficients)
    dx = np.roll(psi, -1, axis=1) - psi
    dy = np.roll(psi, -1, axis=2) - psi
    g1 = np.sum(np.abs(dx)**2 + np.abs(dy)**2, axis=(-2, -1))
    g2 = np.sum(np.abs(dx[..., 0] + dy[..., 1])**2, axis=-1)
    g3 = np.sum(np.abs(dx[..., 0])**2 + np.abs(dy[..., 1])**2
                + 2 * (dx[..., 1].conj() * dy[..., 0]).real, axis=-1)
    return onsite.sum(axis=(1, 2)), (g1 + g2 + g3).sum(axis=(1, 2))


@cuda.jit
def direct_differences(state, positions, proposed, coefficients, K, output):
    r = cuda.grid(1)
    if r < state.shape[0]:
        i, j = positions[r, 0], positions[r, 1]
        old = cuda.local.array((3, 2), dtype=complex64)
        new = cuda.local.array((3, 2), dtype=complex64)
        for mu in range(3):
            for a in range(2):
                old[mu, a] = state[r, i, j, mu, a]
                new[mu, a] = proposed[r, mu, a]
        dg = gradient_difference(state, r, i, j, new)
        dv = potential(new, coefficients) - potential(old, coefficients)
        output[0, r] = dg
        output[1, r] = dv
        output[2, r] = dv + K * dg


@cuda.jit
def color_differences(state, proposed, color, coefficients, K, output):
    col, row, r = cuda.grid(3)
    i, j = 2 * row + color // 2, 2 * col + color % 2
    if r < state.shape[0] and i < state.shape[1] and j < state.shape[2]:
        old = cuda.local.array((3, 2), dtype=complex64)
        new = cuda.local.array((3, 2), dtype=complex64)
        for mu in range(3):
            for a in range(2):
                old[mu, a] = state[r, i, j, mu, a]
                new[mu, a] = proposed[r, i, j, mu, a]
        output[r, i, j] = (potential(new, coefficients)
                           - potential(old, coefficients)
                           + K * gradient_difference(state, r, i, j, new))


@unittest.skipUnless(cuda.is_available(), "A CUDA GPU is required")
class DirectEnergyDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261027)

    def random_fields(self, shape, scale=0.3):
        # All six complex fields fluctuate independently, including nonunitary
        # configurations that cannot be represented by a chiral ansatz.
        return (scale * (self.rng.normal(size=shape)
                         + 1j * self.rng.normal(size=shape))).astype(np.complex64)

    def rounded_proposal(self, old, delta=0.15):
        re = self.rng.uniform(-delta, delta, size=old.shape).astype(np.float32)
        im = self.rng.uniform(-delta, delta, size=old.shape).astype(np.float32)
        new = np.empty_like(old)
        new.real = old.real + re
        new.imag = old.imag + im
        return new

    def evaluate(self, state, positions, proposed, model):
        output = cuda.device_array((3, state.shape[0]), dtype=np.float64)
        direct_differences[(state.shape[0] + 31) // 32, 32](
            cuda.to_device(state), cuda.to_device(positions),
            cuda.to_device(proposed), model.betas, model.K, output)
        return output.copy_to_host()

    def assert_matches_full_change(self, state, positions, proposed, model):
        measured = self.evaluate(state, positions, proposed, model)
        changed = state.copy()
        for r, (i, j) in enumerate(positions):
            changed[r, i, j] = proposed[r]
        old_v, old_g = reference_energy_parts(state, model.betas)
        new_v, new_g = reference_energy_parts(changed, model.betas)
        K = float(model.K)
        # Bounds cover float32 stencil/proposal arithmetic; the reference is
        # float64 and compares the actual stored complex64 endpoints.
        np.testing.assert_allclose(measured[0], new_g - old_g,
                                   rtol=3e-6, atol=6e-6)
        local_old = np.stack([state[r, i, j] for r, (i, j) in enumerate(positions)])
        _, old_scale = reference_potential(local_old, model.betas)
        _, new_scale = reference_potential(proposed, model.betas)
        eps = np.finfo(np.float32).eps
        # The implementation still subtracts two float32 onsite potentials.
        # Its absolute roundoff scales with those terms, not with a tiny delta.
        potential_bound = 8 * eps * np.maximum(1, old_scale + new_scale)
        np.testing.assert_array_less(np.abs(measured[1] - (new_v - old_v)),
                                     potential_bound)
        gradient_bound = 6e-6 + 3e-6 * np.abs(new_g - old_g)
        total_bound = (potential_bound + K * gradient_bound
                       + 2 * eps * (np.abs(new_v - old_v)
                                    + np.abs(K * (new_g - old_g))))
        np.testing.assert_array_less(
            np.abs(measured[2] - (new_v - old_v + K * (new_g - old_g))),
            total_bound)
        return measured, changed

    def test_random_nonunitary_fields_pressures_and_periodic_boundaries(self):
        for L in (4, 6, 20):
            positions = np.array([
                (0, 0), (0, L - 1), (L - 1, 0), (L - 1, L - 1),
                (0, L // 2), (L // 2, 0),
                (L - 1, L // 2), (L // 2, L - 1),
                (1, 1), (1, 2), (2, 1), (2, 2),
            ], dtype=np.int32)
            for pressure in (None, 0.0, 20.0, 34.0):
                with self.subTest(L=L, pressure=pressure):
                    model = He3Model(K=0.73, pressure=pressure)
                    state = self.random_fields((len(positions), L, L, 3, 2))
                    old = np.stack([state[r, i, j]
                                    for r, (i, j) in enumerate(positions)])
                    proposed = self.rounded_proposal(old)
                    measured, _ = self.assert_matches_full_change(
                        state, positions, proposed, model)
                    self.assertTrue(np.isfinite(measured).all())

    def test_homogeneous_fields_and_small_actual_rounded_increments(self):
        model = He3Model(K=1.3, pressure=20.0)
        for L in (4, 6, 20):
            with self.subTest(L=L):
                old = self.random_fields((4, 3, 2), scale=0.7)
                state = np.broadcast_to(old[:, None, None], (4, L, L, 3, 2)).copy()
                positions = np.array([(0, 0), (L - 1, L - 1),
                                      (0, L - 1), (L // 2, L // 2)], dtype=np.int32)
                proposed = old.copy()
                for r, delta in enumerate((0.0, 2**-18, 0.15, 0.4)):
                    proposed[r] = self.rounded_proposal(old[r], delta=delta)
                measured, _ = self.assert_matches_full_change(
                    state, positions, proposed, model)
                step = proposed.astype(np.complex128) - old.astype(np.complex128)
                expected = (8 * np.sum(np.abs(step)**2, axis=(1, 2))
                            + 4 * np.sum((step[..., 0].conj()
                                          * step[..., 1]).real, axis=1))
                # A tight absolute bound ensures the tiny proposal is tested,
                # rather than being hidden by a typical-energy tolerance.
                np.testing.assert_allclose(measured[0], expected,
                                           rtol=3e-6, atol=1e-13)
                np.testing.assert_array_equal(measured[:, 0], 0.0)

    def test_forward_and_reverse_differences(self):
        model = He3Model(K=1.17, pressure=20.0, reduced_temperature=0.8)
        L = 6
        positions = np.array([(0, 0), (0, 5), (5, 0), (5, 5),
                              (2, 3), (3, 2)], dtype=np.int32)
        state = self.random_fields((len(positions), L, L, 3, 2))
        old = np.stack([state[r, i, j] for r, (i, j) in enumerate(positions)])
        proposed = self.rounded_proposal(old)
        forward, changed = self.assert_matches_full_change(
            state, positions, proposed, model)
        reverse, restored = self.assert_matches_full_change(
            changed, positions, old, model)
        np.testing.assert_array_equal(restored, state)
        np.testing.assert_allclose(forward, -reverse, rtol=1e-5, atol=1e-5)

    def test_four_color_changes_with_padded_gpu_launches(self):
        # L20 has ten sites per color axis, so an 8x8 thread tile pads both
        # axes. Summed local differences must equal simultaneous color moves.
        for L in (4, 6, 20):
            model = He3Model(K=0.91, pressure=20.0)
            state = self.random_fields((2, L, L, 3, 2))
            proposed = self.rounded_proposal(state)
            device_state, device_proposed = cuda.to_device(state), cuda.to_device(proposed)
            half = L // 2
            blocks = ((half + 7) // 8, (half + 7) // 8, state.shape[0])
            old_v, old_g = reference_energy_parts(state, model.betas)
            K = float(model.K)
            for color in range(4):
                with self.subTest(L=L, color=color):
                    output = cuda.to_device(np.zeros((2, L, L), dtype=np.float64))
                    color_differences[blocks, (8, 8, 1)](
                        device_state, device_proposed, color, model.betas, model.K, output)
                    changed = state.copy()
                    rows, cols = slice(color // 2, None, 2), slice(color % 2, None, 2)
                    changed[:, rows, cols] = proposed[:, rows, cols]
                    new_v, new_g = reference_energy_parts(changed, model.betas)
                    measured = output.copy_to_host().sum(axis=(1, 2))
                    np.testing.assert_allclose(
                        measured, new_v - old_v + K * (new_g - old_g),
                        rtol=1e-5, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
