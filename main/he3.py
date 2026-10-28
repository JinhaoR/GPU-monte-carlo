from pathlib import Path

from core.runtime import Runtime
from core.parallel_tempering import ParallelTempering
from core.simulation import Simulation

from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime

from utils.data_io import save_runs
from utils.temperature_ladder import make_temperature_ladder


def main():
    # System sizes
    sizes = [64, 80, 96]

    # Physical GL parameters
    pressure = 20.0
    reduced_temperature = 1.0

    # Monte Carlo temperature ladder
    temperatures = make_temperature_ladder(
        T_min=0.365,
        T_max=0.415,
        n_T=100,
        Tc=0.385,
        tc_window=0.01,
        tc_fraction=0.35,
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

        # Choose sampling method
        sampler = "gaussian"  # "cartesian" or "gaussian"

        # Inside your existing loop over L
        runtime_class = (
            He3GaussianRuntime
            if sampler == "gaussian"
            else Runtime
        )

        runtime = runtime_class(
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
            tracked_walkers=range(len(temperatures)),
            history_stride=1,
        )

        simulation = Simulation(
            runtime=pt,
            thermalization_sweeps=100000,
            measurement_sweeps=500000,
            sample_interval=10,
        )

        run = simulation.run()

        if sampler == "gaussian":
            run["metadata"]["sampling"] = (
                runtime.sampling_diagnostics()
            )

        runs[L] = run

    data_path = Path(f"data/he3_{sampler}_run.npz")

    save_runs(data_path, runs)

    print(f"Saved simulation to {data_path}")

    return data_path


if __name__ == "__main__":
    main()
