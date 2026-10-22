import unittest
import numpy as np

from utils.Analysis.crossings import linear_crossings, bootstrap_crossings
from utils.Analysis.statistics import bootstrap_error


class CrossingTests(unittest.TestCase):
    def test_nonuniform_grid_and_two_curves(self):
        self.assertEqual(linear_crossings([0, 1, 4], [0, 2, 8], [3, 3, 3]), [1.5])

    def test_exact_endpoints_and_multiple_roots(self):
        self.assertEqual(linear_crossings([0, 1, 2, 3], [-1, 0, 1, 0]), [1, 3])
        self.assertEqual(linear_crossings([0, 1, 2, 3], [-1, 1, -1, 1]), [0.5, 1.5, 2.5])
        self.assertEqual(linear_crossings([0, 1], [1, 2]), [])

    def test_search_window_does_not_extrapolate(self):
        self.assertEqual(linear_crossings([0, 1], [-1, 1], temperature_range=(2, 3)), [])
        self.assertEqual(linear_crossings([0, 1, 2, 3], [-1, 1, -1, 1],
                                         temperature_range=(1, 2)), [1.5])

    def test_flat_interval_and_undefined_curve_have_no_unique_root(self):
        for curve in ([-1, 0, 0, 1], [-1, np.nan, 1, 1]):
            roots = linear_crossings([0, 1, 2, 3], curve)
            self.assertEqual(len(roots), 1)
            self.assertTrue(np.isnan(roots[0]))
        self.assertEqual(linear_crossings([0, 1, 2], [0, 0, 1],
                                         temperature_range=(1.5, 2)), [])

    def test_bootstrap_uncertainty_for_a_unique_crossing(self):
        curves = np.array([[-1., 1.], [-2., 1.], [-1., 2.]])
        draws = bootstrap_crossings([0, 1], curves)
        np.testing.assert_allclose(draws, [0.5, 2 / 3, 1 / 3])
        self.assertAlmostEqual(bootstrap_error(draws), np.std([0.5, 2 / 3, 1 / 3], ddof=1))

    def test_paired_temperature_and_channel_covariance(self):
        offsets = np.linspace(-0.2, 0.2, 20)[:, None]
        left = [-1., 1.] + offsets
        right = np.zeros((20, 2)) + offsets
        paired = bootstrap_crossings([0, 1], left, right)
        independent = bootstrap_crossings([0, 1], left, right[::-1])
        self.assertAlmostEqual(bootstrap_error(paired), 0, places=14)
        self.assertGreater(bootstrap_error(independent), 0.1)

    def test_ambiguous_or_missing_draws_do_not_get_a_selected_branch(self):
        curves = [[-1, -1, 1, 1], [1, 1, 1, 1], [-1, 1, -1, 1], [-1, 0, 0, 1]]
        draws = bootstrap_crossings([0, 1, 2, 3], curves)
        np.testing.assert_array_equal(np.isnan(draws), [False, True, True, True])
        self.assertIsNone(bootstrap_error(draws))


if __name__ == "__main__":
    unittest.main()
