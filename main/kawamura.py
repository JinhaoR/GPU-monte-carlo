"""Run the Kawamura film model; edit sizes, temperatures and sweeps below."""

from pathlib import Path

import numpy as np

from core.parallel_tempering import ParallelTempering
from core.runtime import Runtime
from core.simulation import Simulation
from models.kawamura.model import KawamuraModel
from utils.data_io import save_runs


def main():
    sizes = [20, 30, 40, 60]
    temperatures = np.linspace(0.60, 0.85, 32)
    runs = {}
    for L in sizes:
        print(f"Running Kawamura model: L = {L}")
        runtime = Runtime(KawamuraModel(J=1.0, delta=1.5), L,
                          temperatures, threads=16, seed=42 + L)
        pt = ParallelTempering(runtime, exchange_interval=10, seed=43)
        runs[L] = Simulation(
            pt, thermalization_sweeps=1000, measurement_sweeps=4000,
            sample_interval=10,
        ).run()
    path = Path("data/kawamura_run.npz")
    save_runs(path, runs)
    print(f"Saved simulation to {path}")
    return path


if __name__ == "__main__":
    main()
