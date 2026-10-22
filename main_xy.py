import numpy as np

from core.runtime import Runtime
from core.simulation import Simulation

from models.xy.model import XYModel
from models.xy.analysis import XYAnalysis

from pathlib import Path
from utils.data_io import save_runs
from utils.temperature_ladder import make_temperature_ladder

from core.parallel_tempering import ParallelTempering


def main():
    sizes = [48, 64, 80, 96]
    temperatures = make_temperature_ladder(
        T_min=0.78,
        T_max=1.02,
        n_T=100,
        Tc=0.893,
        tc_window=0.08,
        tc_fraction=0.60,
    )

    runs = {}

    for L in sizes:
        print(f"Running XY model: L = {L}")

        model = XYModel(J=1.0, delta=1.5)

        runtime = Runtime(
            model=model,
            L=L,
            temperatures=temperatures,
            threads=16,
            seed=42 + L
        )

        pt = ParallelTempering(runtime, exchange_interval=10, seed=43)

        simulation = Simulation(
            runtime=pt,
            thermalization_sweeps= 500000,
            measurement_sweeps= 1000000,
            sample_interval=50
        )

        runs[L] = simulation.run()

    data_path = Path("data/xy_run.npz")

    save_runs(data_path, runs)

    print(f"Saved simulation to {data_path}")

    return data_path

if __name__ == "__main__":
    main()
