from pathlib import Path

from core.runtime import Runtime
from core.parallel_tempering import ParallelTempering
from core.simulation import Simulation

from models.he3.model import He3Model

from utils.data_io import save_runs
from utils.temperature_ladder import make_temperature_ladder


def main():
    # System sizes
    sizes = [32, 48, 64, 80]

    # Physical GL parameters
    pressure = 20.0
    reduced_temperature = 1.0

    # Monte Carlo temperature ladder
    temperatures = make_temperature_ladder(
        T_min=0.36,
        T_max=0.42,
        n_T=60,
        Tc=0.385,
        tc_window=0.02,
        tc_fraction=0.40,
    )

    runs = {}

    for L in sizes:
        print(f"Running He3 model: L = {L}, P = {pressure} bar")

        model = He3Model(
            K=1.0,
            delta=0.15,
            pressure=pressure,
            reduced_temperature=reduced_temperature,
        )

        runtime = Runtime(
            model=model,
            L=L,
            temperatures=temperatures,
            threads=16,
            seed=42 + L,
        )

        pt = ParallelTempering(
            runtime,
            exchange_interval=10,
            seed=43 + L,
        )

        simulation = Simulation(
            runtime=pt,
            thermalization_sweeps=100000,
            measurement_sweeps=500000,
            sample_interval=10,
        )

        runs[L] = simulation.run()

    data_path = Path("data/he3_run.npz")

    save_runs(data_path, runs)

    print(f"Saved simulation to {data_path}")

    return data_path


if __name__ == "__main__":
    main()