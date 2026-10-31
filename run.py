
from pathlib import Path
from time import perf_counter
import numpy as np

from core.runtime import Runtime
from core.parallel_tempering import ParallelTempering
from core.simulation import Simulation

from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from models.he3.spatial import add_spatial_samples

from utils.data_io import save_runs
from utils.temperature_ladder import make_temperature_ladder


# ============================================================
# SETTINGS
# ============================================================

# Physical parameters
pressures = [20.0]
g_d_values = [0.0, 0.001, 0.01]
sizes = [64, 96, 128, 256]

K = 1.0
betas = (-0.5, 1, 1, 1, -1)
delta = 0.15
reduced_temperature = 1.0

# Sampling
sampler = "gaussian"  # "gaussian" or "cartesian"
gaussian_rho = -0.9
gaussian_proposal_mass = 0.0

thermalization_sweeps = 2_000_000
measurement_sweeps = 4_000_000
sample_interval = 50
exchange_interval = 10
spatial_interval = 1000  # sweeps; the original scalar channels stay at interval 10
realspace_interval = 10_000  # sweeps; 0 disables occasional kappa/P/Q profiles
realspace_temperatures = {
    0.0: [0.375, 0.3815, 0.3825, 0.384, 0.415],
    0.001: [0.375, 0.382, 0.3825, 0.384, 0.415],
    0.01: [0.375, 0.3852, 0.3856, 0.387, 0.415],
}

# GPU and parallel tempering
threads = 16
seed_base = 42
pt_seed_base = 43
tracked_walkers = "all"
history_stride = 10  # exchange rounds = 100 sweeps

# One common ladder, designed from all three L160 energy curves for L<=256.
# Tc centers the protected .380-.388 region; it is not a fitted transition.
temperature_ladder = {
    "T_min": 0.365, "T_max": 0.425, "n_T": 502,
    "Tc": 0.384, "tc_window": 0.012, "tc_fraction": 0.43,
}

# Output
campaign = "weak_soc_spatial_simple_01"
data_root = Path("data/he3")


# ============================================================
# SIMULATION
# ============================================================

def main():

    temperatures = make_temperature_ladder(**temperature_ladder)

    if spatial_interval < sample_interval or spatial_interval % sample_interval:
        raise ValueError("spatial_interval must be a positive multiple of sample_interval")
    if realspace_interval < 0 or realspace_interval % sample_interval:
        raise ValueError("realspace_interval must be zero or a positive multiple of sample_interval")

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

    walker_ids = range(len(temperatures)) if tracked_walkers == "all" else tracked_walkers

    for L in sizes:
        for pressure in pressures:
            for g_d in g_d_values:
                folder = f"P={pressure:g}_gd={g_d:g}_{sampler}"
                output_dir = data_root / campaign / folder
                output_file = output_dir / f"L={L}.npz"
                if output_file.exists():
                    raise FileExistsError(output_file)

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
                model.spatial_every = spatial_interval // sample_interval
                model.realspace_every = realspace_interval // sample_interval
                model.realspace_indices = np.unique([
                    np.argmin(abs(temperatures - t)) for t in realspace_temperatures[g_d]
                ])

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
                add_spatial_samples(run, model.spatial_samples, model.realspace_samples)

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
                metadata["spatial_interval"] = spatial_interval
                metadata["realspace_interval"] = realspace_interval
                metadata["initialization"] = "ordered: positive chirality, d=(0,0,1), phase=0"
                metadata["sampler_parameters"] = (
                    dict(runtime_options)
                    if sampler == "gaussian"
                    else {"delta": delta}
                )

                if sampler == "gaussian":
                    metadata["sampling"] = runtime.sampling_diagnostics()

                # Save each lattice size separately
                output_dir.mkdir(parents=True, exist_ok=True)

                start = perf_counter()
                save_runs(output_file, {L: run})
                save_seconds = perf_counter() - start

                print(
                    f"  Simulation: {simulation_seconds / 3600:.2f} h"
                )
                print(f"  Saving:     {save_seconds:.1f} s")
                print(f"  Saved:      {output_file}", flush=True)
                del run, simulation, pt, runtime, model


if __name__ == "__main__":
    main()
