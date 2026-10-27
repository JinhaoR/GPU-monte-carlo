
import numpy as np

from pathlib import Path

from core.runtime import Runtime
from core.simulation import Simulation
from core.parallel_tempering import ParallelTempering

from models.ising.model import IsingModel

from utils.data_io import save_runs
from utils.temperature_ladder import make_temperature_ladder


def main():
    sizes = [48, 64, 80]

    J = 1.0
    Tc = 2 * J / np.log(1 + np.sqrt(2))

    temperatures = make_temperature_ladder(
        T_min=2.0,
        T_max=2.5,
        n_T=100,
        Tc=Tc,
        tc_window=0.15,
        tc_fraction=0.60,
    )

    runs = {}

    for L in sizes:
        print(f"Running Ising model: L = {L}")

        model = IsingModel(J=J)

        runtime = Runtime(
            model=model,
            L=L,
            temperatures=temperatures,
            threads=16,
            seed=42 + L
        )

        pt = ParallelTempering(
            runtime,
            exchange_interval=10,
            seed=43
        )

        simulation = Simulation(
            runtime=pt,
            thermalization_sweeps=100000,
            measurement_sweeps=500000,
            sample_interval=10
        )

        runs[L] = simulation.run()

    data_path = Path("data/ising_run.npz")

    save_runs(data_path, runs)

    print(f"Saved simulation to {data_path}")

    return data_path


if __name__ == "__main__":
    main()
