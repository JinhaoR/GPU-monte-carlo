
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
g_d_values = [0.0, 0.001, 0.01, 0.1]
sizes = [64, 96, 128, 160]

K = 1.0
betas = (-0.5, 1, 1, 1, -1)
delta = 0.15
reduced_temperature = 1.0

# Sampling
sampler = "gaussian"  # "gaussian" or "cartesian"
gaussian_rho = -0.9
gaussian_proposal_mass = 0.0

thermalization_sweeps = 2_000_000
measurement_sweeps = 3_000_000
sample_interval = 50
exchange_interval = 5

# GPU and parallel tempering
threads = 16
seed_base = 42
pt_seed_base = 43
tracked_walkers = "all"
history_stride = 100

# Temperature ladder
temperature_ladder = {
    "T_min": 0.33,
    "T_max": 0.47,
    "n_T": 160,
    "Tc": 0.39,
    "tc_window": 0.06,
    "tc_fraction": 0.55,
}

# Output
campaign = "soc_scan_02"
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

    for L in sizes:
        for pressure in pressures:
            for g_d in g_d_values:

                print(
                    f"\nHe3: L={L}, P={pressure:g}, "
                    f"g_d={g_d:g}, sampler={sampler}",
                    flush=True,
                )

                model = He3Model(
                    K=K,
                    betas=betas,
                    delta=delta,
                    pressure=pressure,
                    reduced_temperature=reduced_temperature,
                    g_d=g_d,
                )

                runtime = runtime_class(
                    model=model,
                    L=L,
                    temperatures=temperatures,
                    threads=threads,
                    seed=seed_base + L,
                    **runtime_options,
                )

                pt = ParallelTempering(
                    runtime,
                    exchange_interval=exchange_interval,
                    seed=pt_seed_base + L,
                    tracked_walkers=walker_ids,
                    history_stride=history_stride,
                )

                simulation = Simulation(
                    runtime=pt,
                    thermalization_sweeps=thermalization_sweeps,
                    measurement_sweeps=measurement_sweeps,
                    sample_interval=sample_interval,
                )

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

                # Save each lattice size separately
                folder = f"P={pressure:g}_gd={g_d:g}_{sampler}"
                output_dir = data_root / campaign / folder
                output_dir.mkdir(parents=True, exist_ok=True)

                output_file = output_dir / f"L={L}.npz"

                start = perf_counter()
                save_runs(output_file, {L: run})
                save_seconds = perf_counter() - start

                print(
                    f"  Simulation: {simulation_seconds / 3600:.2f} h"
                )
                print(f"  Saving:     {save_seconds:.1f} s")
                print(f"  Saved:      {output_file}", flush=True)


if __name__ == "__main__":
    main()
