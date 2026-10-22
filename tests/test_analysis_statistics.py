import unittest

import numpy as np

from utils.Analysis.statistics import (
    block_means,
    blocked_jackknife,
    blocked_mean_error,
    bootstrap_block_means,
    jackknife_from_blocks,
    bootstrap_error,
)


class JackknifeTests(unittest.TestCase):
    def test_linear_jackknife_matches_blocked_standard_error(self):
        data = np.random.default_rng(12).normal(size=(240, 4))
        expected_mean, expected_error = blocked_mean_error(data, block_size=6)
        actual_mean, actual_error = blocked_jackknife(
            {"x": data}, lambda means: means["x"], block_size=6,
        )
        np.testing.assert_allclose(actual_mean, expected_mean, atol=1e-15)
        np.testing.assert_allclose(actual_error, expected_error, rtol=1e-13)

    def test_nonlinear_jackknife_matches_deleting_contiguous_raw_blocks(self):
        samples = np.random.default_rng(91).normal(loc=0.7, size=(139, 3))
        data = {"m2": samples**2, "m4": samples**4}
        block_size = 4

        def binder(means):
            return 1 - means["m4"] / (2 * means["m2"]**2)

        actual, error = blocked_jackknife(data, binder, block_size)
        n_blocks = len(samples) // block_size
        complete = {name: values[:n_blocks * block_size]
                    for name, values in data.items()}
        expected = binder({name: values.mean(axis=0) for name, values in complete.items()})
        deleted = []
        for block in range(n_blocks):
            keep = np.ones(n_blocks * block_size, dtype=bool)
            keep[block * block_size:(block + 1) * block_size] = False
            deleted.append(binder({name: values[keep].mean(axis=0)
                                   for name, values in complete.items()}))
        deleted = np.asarray(deleted)
        expected_error = np.sqrt((n_blocks - 1) / n_blocks
                                 * np.sum((deleted - deleted.mean(axis=0))**2, axis=0))
        np.testing.assert_allclose(actual, expected, rtol=1e-13)
        np.testing.assert_allclose(error, expected_error, rtol=1e-13)

    def test_preblocked_api_matches_legacy_api(self):
        rng = np.random.default_rng(35)
        data = {"a": rng.uniform(1, 2, (163, 4)),
                "b": rng.uniform(-1, 1, (163, 4))}
        estimator = lambda means: means["b"] / means["a"]**2
        expected = blocked_jackknife(data, estimator, block_size=5)
        actual = jackknife_from_blocks(
            {name: block_means(values, 5) for name, values in data.items()}, estimator,
        )
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)


class BootstrapBlockMeanTests(unittest.TestCase):
    def test_fixed_seed_is_reproducible_and_supports_generators(self):
        blocks = {"a": np.arange(60.).reshape(20, 3),
                  "b": np.arange(20.)**2}
        first = bootstrap_block_means(blocks, n_resamples=37, rng=123)
        second = bootstrap_block_means(blocks, n_resamples=37, rng=123)
        generator = bootstrap_block_means(blocks, n_resamples=37,
                                          rng=np.random.default_rng(123))
        for name in blocks:
            np.testing.assert_array_equal(first[name], second[name])
            np.testing.assert_array_equal(first[name], generator[name])
            self.assertEqual(first[name].shape, (37, *blocks[name].shape[1:]))

    def test_full_channel_temperature_covariance_is_preserved(self):
        # Eight blocks give exactly representable count/B weights. Thus these
        # deterministic relations must hold exactly, not merely on average.
        x = np.arange(8.)
        temperatures = np.stack((x, 2 * x + 3, -x), axis=1)
        blocks = {"x": x, "other_channel": 4 * x + 2,
                  "temperatures": temperatures,
                  "tensor": np.stack((temperatures, 2 * temperatures), axis=2)}
        result = bootstrap_block_means(blocks, n_resamples=100, rng=74)
        np.testing.assert_array_equal(result["other_channel"], 4 * result["x"] + 2)
        np.testing.assert_array_equal(result["temperatures"][:, 0], result["x"])
        np.testing.assert_array_equal(result["temperatures"][:, 1], 2 * result["x"] + 3)
        np.testing.assert_array_equal(result["temperatures"][:, 2], -result["x"])
        np.testing.assert_array_equal(result["tensor"][:, :, 0], result["temperatures"])
        np.testing.assert_array_equal(result["tensor"][:, :, 1], 2 * result["temperatures"])

    def test_resample_means_match_explicit_multinomial_counts(self):
        blocks = {"a": np.arange(12.).reshape(4, 3), "b": np.array([1., 4., 2., 7.])}
        counts = np.random.default_rng(18).multinomial(4, np.full(4, 0.25), size=9)
        actual = bootstrap_block_means(blocks, n_resamples=9, rng=18)
        for name, values in blocks.items():
            expected = np.stack([np.repeat(values, row, axis=0).mean(axis=0)
                                 for row in counts])
            np.testing.assert_array_equal(actual[name], expected)


class ErrorEstimateTests(unittest.TestCase):
    def test_two_blocks_are_the_minimum_for_an_error(self):
        np.testing.assert_array_equal(block_means(np.arange(5.), 2), [0.5, 2.5])
        with self.assertRaises(ValueError):
            block_means(np.arange(3.), 2)

    def test_bootstrap_error_requires_defined_draws(self):
        draws = np.array([0.4, 0.5, 0.7])
        self.assertEqual(bootstrap_error(draws), draws.std(ddof=1))
        self.assertIsNone(bootstrap_error([0.4, np.nan, 0.7]))
        self.assertIsNone(bootstrap_error([0.4]))


if __name__ == "__main__":
    unittest.main()
