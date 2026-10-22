"""CPU-only Ising caller regression; no simulation or GPU imports."""
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.ising.analysis import IsingAnalysis
from utils.Analysis.binder import binder_cumulant


class IsingAnalysisTests(unittest.TestCase):
    def test_binder_normalization_and_existing_ising_observables(self):
        rng = np.random.default_rng(13)
        m2 = rng.uniform(0.2, 2, (60, 3))
        for g in (2, 3):
            np.testing.assert_allclose(binder_cumulant(m2, g),
                1 - (m2**2).mean(axis=0) / (g * m2.mean(axis=0)**2))
        T, J = np.array([1.8, 2.2, 2.6]), 1.7
        runs = {L: {"metadata": {"L": L, "temperatures": T, "model_parameters": {"J": J}},
                    "samples": {"energy": rng.normal(-L**2, 2, (60, 3)),
                                "magnetization": rng.normal(0, L, (60, 3))}}
                for L in (8, 16)}
        analysis = IsingAnalysis(runs)
        for L, run in runs.items():
            E, M = run["samples"]["energy"], run["samples"]["magnetization"]
            expected = {"energy": E.mean(axis=0) / L**2,
                        "magnetization": abs(M).mean(axis=0) / L**2,
                        "heat_capacity": ((E**2).mean(axis=0) - E.mean(axis=0)**2) / (L**2 * T**2),
                        "susceptibility": ((M**2).mean(axis=0) - M.mean(axis=0)**2) / (L**2 * T),
                        "binder": 1 - (M**4).mean(axis=0) / (3 * (M**2).mean(axis=0)**2)}
            for name, values in expected.items():
                np.testing.assert_allclose(analysis.results[L][name], values)
        with patch.object(plt, "show") as show:
            analysis.plot()
            show.assert_called_once()
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
