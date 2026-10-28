
import os
import tempfile
from pathlib import Path
from time import perf_counter

from core.runtime import Runtime
from core.parallel_tempering import ParallelTempering
from core.simulation import Simulation

from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime

from utils.data_io import save_runs
from utils.temperature_ladder import make_temperature_ladder


# ============================================================
# SETTINGS
# ============================================================

# Physical parameters
pressures = [20.0]
g_d_values = [0.001]
sizes = [64, 80, 96]

K = 1.0
betas = (-0.5, 1, 1, 1, -1)
delta = 0.15
reduced_temperature = 1.0

# Sampling
sampler = "gaussian"  # "gaussian" or "cartesian"
gaussian_rho = -0.9
gaussian_proposal_mass = 0.0

thermalization_sweeps = 500_000
measurement_sweeps = 1_000_000
sample_interval = 10
exchange_interval = 10

# GPU and parallel tempering
threads = 16
seed_base = 42
pt_seed_base = 43
tracked_walkers = "all"  # "all", None, or list of IDs
history_stride = 10

# Temperature ladder
temperature_ladder = {
    "T_min": 0.365,
    "T_max": 0.415,
    "n_T": 100,
    "Tc": 0.385,
    "tc_window": 0.01,
    "tc_fraction": 0.35,
}

# Output
campaign = "soc_scan_01"
data_root = Path("data/he3")


# ============================================================
# SIMULATION
# ============================================================

def main():

    temperatures = make_temperature_ladder(**temperature_ladder)

    if sampler == "gaussian":
        runtime_class = He3GaussianRuntime
        runtime_options = {
            "rho": gaussian_rho,
            "proposal_mass": gaussian_proposal_mass,
        }
    elif sampler == "cartesian":
        runtime_class = Runtime
        runtime_options = {}
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    walker_ids = (
        range(len(temperatures))
        if tracked_walkers == "all"
        else tracked_walkers
    )

    # Check output files before starting expensive simulations.
    def output_path(pressure, g_d, L):
        folder = f"P={pressure:g}_gd={g_d:g}_{sampler}"
        return data_root / campaign / folder / f"L={L}.npz"

    paths = [
        output_path(pressure, g_d, L)
        for pressure in pressures
        for g_d in g_d_values
        for L in sizes
    ]

    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate simulation output paths")

    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Output files already exist:\n"
            + "\n".join(map(str, existing))
        )

    saved_paths = []

    print(f"\nHe3 Simulation: {campaign}")

    for pressure in pressures:
        for g_d in g_d_values:

            print(
                f"\nP={pressure:g} bar, "
                f"g_d={g_d:g}, sampler={sampler}"
            )

            for L in sizes:

                print(f"\nL={L}")

                # Model
                model = He3Model(
                    K=K,
                    betas=betas,
                    delta=delta,
                    pressure=pressure,
                    reduced_temperature=reduced_temperature,
                    g_d=g_d,
                )

                # Runtime
                runtime = runtime_class(
                    model=model,
                    L=L,
                    temperatures=temperatures,
                    threads=threads,
                    seed=seed_base + L,
                    **runtime_options,
                )

                # Parallel tempering
                pt = ParallelTempering(
                    runtime,
                    exchange_interval=exchange_interval,
                    seed=pt_seed_base + L,
                    tracked_walkers=walker_ids,
                    history_stride=history_stride,
                )

                # Simulation
                simulation = Simulation(
                    runtime=pt,
                    thermalization_sweeps=thermalization_sweeps,
                    measurement_sweeps=measurement_sweeps,
                    sample_interval=sample_interval,
                )

                # Run
                start = perf_counter()
                run = simulation.run()
                simulation_seconds = perf_counter() - start

                # Metadata
                metadata = run["metadata"]

                metadata["timing"] = {
                    "simulation_seconds": simulation_seconds,
                }
                metadata["campaign"] = {
                    "id": campaign,
                    "sizes": list(sizes),
                }
                metadata["sampler"] = sampler
                metadata["threads"] = threads
                metadata["temperature_ladder"] = dict(temperature_ladder)
                metadata["sampler_parameters"] = (
                    dict(runtime_options)
                    if sampler == "gaussian"
                    else {"delta": delta}
                )

                if sampler == "gaussian":
                    metadata["sampling"] = runtime.sampling_diagnostics()

                # Save this size before starting the next one.
                path = output_path(pressure, g_d, L)
                path.parent.mkdir(parents=True, exist_ok=True)

                start = perf_counter()

                with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
                    staging = Path(tmp) / "run.npz"
                    save_runs(staging, {L: run})
                    os.link(staging, path)

                saving_seconds = perf_counter() - start
                saved_paths.append(path)

                print(f"  Simulation: {simulation_seconds / 60:.2f} min")
                print(f"  Saving:     {saving_seconds:.1f} s")
                print(f"  Saved:      {path}")

    return saved_paths


if __name__ == "__main__":
    main()
