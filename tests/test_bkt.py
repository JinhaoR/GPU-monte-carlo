"""CPU-only checks of the leading Weber--Minnhagen estimator."""

import unittest

import numpy as np

from utils.Analysis.bkt import fit_weber_minnhagen


def critical_curves(temperatures, sizes, critical_t=0.9137, constant=1.8):
    sizes = np.asarray(sizes, dtype=float)
    critical = 2.0 * critical_t / np.pi * (1.0 + 1.0 / (2.0 * np.log(sizes) + constant))
    slopes = -0.3 - 0.15 * np.log(sizes)
    return critical[:, None] + slopes[:, None] * (temperatures - critical_t)


class WeberMinnhagenTests(unittest.TestCase):
    def setUp(self):
        self.temperatures = np.linspace(0.85, 0.98, 27)
        self.sizes = np.array([16, 24, 32, 48, 64, 96])
        self.helicity = critical_curves(self.temperatures, self.sizes)
        self.errors = np.full_like(self.helicity, 0.002)

    def fit(self, **kwargs):
        return fit_weber_minnhagen(self.temperatures, self.sizes,
                                  self.helicity, self.errors, **kwargs)

    def test_known_critical_curve_with_sub_grid_temperature(self):
        fit = self.fit()
        self.assertIsNone(fit["reason"])
        self.assertAlmostEqual(fit["temperature"], 0.9137, delta=1e-7)
        self.assertAlmostEqual(fit["C"], 1.8, delta=0.01)
        self.assertEqual(fit["degrees_of_freedom"], 4)
        self.assertEqual(fit["profile_chi2"].shape, self.temperatures.shape)

    def test_absolute_energy_units_are_not_assumed_to_be_unit_J(self):
        scale = 2.7
        fit = fit_weber_minnhagen(self.temperatures * scale, self.sizes,
                                  self.helicity * scale, self.errors * scale)
        self.assertAlmostEqual(fit["temperature"], 0.9137 * scale, delta=8e-5)
        self.assertAlmostEqual(fit["C"], 1.8, delta=0.01)

    def test_uncertainty_weighting_downweights_a_bad_size(self):
        self.helicity[-1] += 0.03
        unweighted = self.fit()
        self.errors[-1] = 0.3
        weighted = self.fit()
        self.assertIsNone(weighted["reason"])
        self.assertLess(abs(weighted["temperature"] - 0.9137), 5e-5)
        self.assertGreater(abs(unweighted["temperature"] - 0.9137),
                           abs(weighted["temperature"] - 0.9137) * 10)

    def test_refinement_does_not_shrink_away_from_a_better_minimum(self):
        # Three neighboring saved-data points: no dataset/file dependency.
        # A first parabola stays near .8923; shrinking there prematurely
        # excludes the true interpolated minimum near .89195.
        T = [0.8910545464, 0.8923169116, 0.8935828933]
        rho = [[0.6259484324, 0.6242144713, 0.6227658390],
               [0.6250978708, 0.6213922803, 0.6203715587],
               [0.6228850989, 0.6192656886, 0.6169999586]]
        errors = [[0.0012772290, 0.0012925704, 0.0014301342],
                  [0.0011339394, 0.0015397859, 0.0012475128],
                  [0.0011883435, 0.0013095494, 0.0012671403]]
        fit = fit_weber_minnhagen(T, [64, 80, 96], rho, errors)
        self.assertIsNone(fit["reason"])
        self.assertAlmostEqual(fit["temperature"], 0.8919522554, delta=1e-7)
        self.assertLess(fit["chi2"], 0.027)

    def test_size_cut_and_unsorted_inputs(self):
        order = [5, 0, 3, 2, 1, 4]
        fit = fit_weber_minnhagen(self.temperatures, self.sizes[order],
                                  self.helicity[order], self.errors[order],
                                  min_size=32, temperature_range=(0.88, 0.95))
        self.assertIsNone(fit["reason"])
        self.assertEqual(fit["degrees_of_freedom"], 2)
        self.assertEqual(fit["min_size"], 32)
        self.assertAlmostEqual(fit["temperature"], 0.9137, delta=3e-5)

    def test_insufficient_size_or_temperature_data(self):
        fit = self.fit(min_size=64)
        self.assertIn("three sizes", fit["reason"])
        self.assertIsNone(fit["temperature"])
        fit = self.fit(temperature_range=(0.90, 0.906))
        self.assertIn("three temperatures", fit["reason"])
        self.assertIsNone(fit["temperature"])
        self.assertIsNone(self.fit(min_size=1000)["temperature"])

    def test_boundary_is_not_a_critical_temperature_estimate(self):
        self.helicity = critical_curves(self.temperatures, self.sizes, critical_t=1.02)
        fit = self.fit()
        self.assertIn("boundary", fit["reason"])
        self.assertIsNone(fit["temperature"])

    def test_vanishing_correction_is_not_a_finite_C(self):
        self.temperatures = np.linspace(0.85, 0.95, 21)
        self.helicity = (2.0 * self.temperatures[None, :] / np.pi
                         + np.linspace(0.1, 0.3, len(self.sizes))[:, None]
                         * (self.temperatures - 0.9))
        self.errors = np.full_like(self.helicity, 0.002)
        fit = self.fit()
        self.assertIn("infinite", fit["reason"])
        self.assertIsNone(fit["temperature"])
        self.assertIsNone(fit["C"])

    def test_flat_temperature_profile_is_explicit(self):
        self.helicity = (2.0 * self.temperatures[None, :] / np.pi
                         * (1.0 + 1.0 / (2.0 * np.log(self.sizes[:, None]) + 1.8)))
        self.assertIn("flat", self.fit()["reason"])

    def test_zero_error_is_not_a_valid_weight(self):
        self.errors[:, 5] = 0
        fit = self.fit()
        self.assertIsNone(fit["temperature"])
        self.assertIn("positive finite errors", fit["reason"])
        # Selecting a contiguous window avoids the undefined weight.
        self.assertIsNone(self.fit(temperature_range=(0.9, 0.96))["reason"])


if __name__ == "__main__":
    unittest.main()
