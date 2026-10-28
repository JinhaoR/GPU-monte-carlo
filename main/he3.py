from pathlib import Path
from time import perf_counter

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
    g_d = 0.001

    # Sampling algorithm: "cartesian" or "gaussian"
    sampler = "gaussian"

    # Monte Carlo settings
    thermalization_sweeps = 500_000
    measurement_sweeps = 1_000_000
    sample_interval = 10
    exchange_interval = 10

    # Monte Carlo temperature ladder
    temperatures = make_temperature_ladder(
        T_min=0.365,
        T_max=0.415,
        n_T=100,
        Tc=0.385,
        tc_window=0.01,
        tc_fraction=0.35,
    )

    # Each lattice size has its own output file.
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    data_paths = {
        L: data_dir / f"he3_gd={float(g_d):.6g}_P={pressure:g}_L={L}_run.npz"
        for L in sizes
    }

    # Do not waste simulation time or overwrite an existing dataset.
    existing = [path for path in data_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Output file(s) already exist:\n" + "\n".join(map(str, existing))
        )

    if sampler == "gaussian":
        runtime_class = He3GaussianRuntime
    elif sampler == "cartesian":
        runtime_class = Runtime
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    saved_paths = []

    for L in sizes:
        data_path = data_paths[L]
        print(f"\nRunning He3: L={L}, P={pressure:g} bar, g_d={g_d:g}, sampler={sampler}")

        model = He3Model(
            K=1.0,
            delta=0.15,
            pressure=pressure,
            reduced_temperature=reduced_temperature,
            g_d=g_d,
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
            exchange_interval=exchange_interval,
            seed=43 + L,
            tracked_walkers=range(len(temperatures)),
            history_stride=10,
        )

        simulation = Simulation(
            runtime=pt,
            thermalization_sweeps=thermalization_sweeps,
            measurement_sweeps=measurement_sweeps,
            sample_interval=sample_interval,
        )

        # Time this lattice size's simulation separately from file saving.
        start = perf_counter()
        run = simulation.run()
        simulation_seconds = perf_counter() - start

        if sampler == "gaussian":
            run["metadata"]["sampling"] = runtime.sampling_diagnostics()

        run["metadata"]["timing"] = {
            "simulation_seconds": simulation_seconds,
        }

        # Save immediately, before starting the next lattice size.
        save_start = perf_counter()
        save_runs(data_path, {L: run})
        save_seconds = perf_counter() - save_start
        saved_paths.append(data_path)

        print(f"L={L} complete")
        print(f"  Simulation: {simulation_seconds:.1f} s ({simulation_seconds / 60:.2f} min)")
        print(f"  Saving:     {save_seconds:.1f} s")
        print(f"  Total:      {simulation_seconds + save_seconds:.1f} s")
        print(f"  Saved to:   {data_path}")

    return saved_paths


if __name__ == "__main__":
    main()
