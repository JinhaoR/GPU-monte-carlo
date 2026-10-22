"""Run the full thin-film model; edit coefficients, sizes and sweeps below."""

from pathlib import Path

import numpy as np

from core.parallel_tempering import ParallelTempering
from core.runtime import Runtime
from core.simulation import Simulation
from models.he3.model import He3Model
from utils.data_io import save_runs


def main():
    sizes = [16, 24, 32]
    temperatures = np.linspace(0.15, 0.45, 24)
    pressure = 20.0  # bar; Wiman strong-coupling polynomial, valid from 0 to 34 bar.
    reduced_temperature = 1.0  # Physical T/Tc for coefficients, fixed across MC replicas.
    runs = {}
    for L in sizes:
        print(f"Running full thin-film helium-3 model: L = {L}, p = {pressure} bar")
        model = He3Model(K=1.0, delta=0.15, pressure=pressure,
                         reduced_temperature=reduced_temperature)
        runtime = Runtime(model, L,
                          temperatures, threads=16, seed=42 + L)
        pt = ParallelTempering(runtime, exchange_interval=10, seed=43)
        runs[L] = Simulation(
            pt, thermalization_sweeps=1000, measurement_sweeps=4000,
            sample_interval=10,
        ).run()
    path = Path("data/he3_run.npz")
    save_runs(path, runs)
    print(f"Saved simulation to {path}")
    return path


if __name__ == "__main__":
    main()
