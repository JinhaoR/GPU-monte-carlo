"""Longer L96 A/B/C comparison; run inside an allocated GPU job.

From the repository root:
    python -u test_run.py

Runs 12 cases sequentially. Edit the lists below to select a smaller subset.
Frozen ladders are read from exploration/he3_pt_ladder_validation/.
Results go to analysis/pt_ladder_test/. Existing output files are never
overwritten. This is a fresh-run script, not a checkpoint resume.
"""

import gc
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import cuda

from core.parallel_tempering import ParallelTempering
from core.simulation import Simulation
from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from models.he3.spatial import add_spatial_samples
from utils.data_io import save_runs


# ============================================================
# SETTINGS
# ============================================================

pressures = [20.0]
g_d_values = [0.0, 0.01]
sizes = [96]
initializations = ["ordered", "random"]
ladder_names = ["A", "B", "C"]
# A: 200 replicas, 0.370-0.420; B: 200, 0.375-0.405; C: 160, 0.375-0.405.
# Use the frozen ladders from the pilot, without regenerating or adapting them.

K = 1.0
betas = (-0.5, 1, 1, 1, -1)
delta = 0.15
reduced_temperature = 1.0

gaussian_rho = -0.9
gaussian_proposal_mass = 0.0

thermalization_sweeps = 500_000  # diagnostic burn-in, retained in the output
measurement_sweeps = 1_000_000  # comparison window after diagnostic burn-in
sample_interval = 50
exchange_interval = 5
spatial_interval = 1_000
realspace_interval = 0

threads = 16
history_stride = 1  # every exchange (5 sweeps); sparse histories miss trips

# Fresh seeds for this longer test; distinct between the two initializations,
# matched across ladders/couplings. Different replica counts change RNG assignment.
seeds = {"ordered": 20261112, "random": 20261115}
random_initialization_seed = 20261118
# The exchange RNG uses seeds[initialization] + 1.

campaign = "pt_ladder_L96_long_01"
directory = Path(__file__).resolve().parent
ladder_dir = directory / "exploration" / "he3_pt_ladder_validation"
output_dir = directory / "analysis" / "pt_ladder_test"


# ============================================================
# SIMULATION
# ============================================================

def main():
    ladders = {
        name: np.loadtxt(ladder_dir / f"ladder_{name}.txt")
        for name in ladder_names
    }
    for name, temperatures in ladders.items():
        if (temperatures.ndim != 1 or not np.all(np.isfinite(temperatures))
                or np.any(temperatures <= 0) or np.any(np.diff(temperatures) <= 0)
                or np.any(np.diff((1 / temperatures).astype(np.float32)) >= 0)):
            raise ValueError(f"Invalid temperature ladder: {name}")
    if spatial_interval < sample_interval or spatial_interval % sample_interval:
        raise ValueError("spatial_interval must be a positive multiple of sample_interval")
    if realspace_interval != 0:
        raise ValueError("Real-space profiles are disabled for this PT comparison")

    runtime_options = {
        "rho": gaussian_rho,
        "proposal_mass": gaussian_proposal_mass,
    }

    for L in sizes:
        for pressure in pressures:
            for g_d in g_d_values:
                for initialization in initializations:
                    for ladder in ladder_names:
                        temperatures = ladders[ladder]
                        seed = seeds[initialization]
                        output_file = output_dir / (
                            f"L={L}_P={pressure:g}_gd={g_d:g}_{initialization}_{ladder}.npz"
                        )
                        if output_file.exists():
                            raise FileExistsError(output_file)

                        print(
                            f"\nHe3 PT test: L={L}, P={pressure:g}, g_d={g_d:g}, "
                            f"{initialization}, ladder={ladder}, R={len(temperatures)}\n"
                            f"  Diagnostic burn: {thermalization_sweeps:,}; "
                            f"comparison: {measurement_sweeps:,} sweeps",
                            flush=True,
                        )

                        model = He3Model(
                            K=K, betas=betas, delta=delta, pressure=pressure,
                            reduced_temperature=reduced_temperature, g_d=g_d,
                        )
                        model.spatial_every = spatial_interval // sample_interval
                        model.realspace_every = 0

                        runtime = He3GaussianRuntime(
                            model=model, L=L, temperatures=temperatures,
                            threads=threads, seed=seed, **runtime_options,
                        )
                        # The default field is the positive-chirality ordered state.
                        # Random starts explore all six complex components, with
                        # the same mean density as that homogeneous ordered state.
                        if initialization == "random":
                            rng = np.random.default_rng(random_initialization_seed)
                            n0 = 1 / float(model.betas[1] + model.betas[3] + model.betas[4])
                            field = (
                                np.sqrt(n0 / 12) * (
                                    rng.normal(size=runtime.state.shape)
                                    + 1j * rng.normal(size=runtime.state.shape)
                                )
                            ).astype(np.complex64)
                            runtime.invalidate_energy_cache()
                            runtime.state.copy_to_device(field)
                            del field

                        pt = ParallelTempering(
                            runtime, exchange_interval=exchange_interval,
                            seed=seed + 1, tracked_walkers=range(len(temperatures)),
                            history_stride=history_stride,
                        )

                        # Save the transient as in the short pilots. The producer
                        # records every sweep interval; analysis excludes samples
                        # with sample_sweeps <= diagnostic_burn_sweeps.
                        simulation = Simulation(
                            runtime=pt, thermalization_sweeps=0,
                            measurement_sweeps=thermalization_sweeps + measurement_sweeps,
                            sample_interval=sample_interval,
                        )
                        cuda.synchronize()
                        start = perf_counter()
                        run = simulation.run()
                        cuda.synchronize()
                        simulation_seconds = perf_counter() - start
                        add_spatial_samples(run, model.spatial_samples, model.realspace_samples)

                        metadata = run["metadata"]
                        metadata["campaign"] = campaign
                        metadata["ladder"] = ladder
                        metadata["initialization"] = initialization
                        metadata["host_initialization_seed"] = (
                            random_initialization_seed if initialization == "random" else None
                        )
                        metadata["diagnostic_burn_sweeps"] = thermalization_sweeps
                        metadata["diagnostic_production_sweeps"] = measurement_sweeps
                        metadata["sampler"] = "gaussian"
                        metadata["sampler_parameters"] = dict(runtime_options)
                        metadata["sampling"] = runtime.sampling_diagnostics()
                        metadata["threads"] = threads
                        metadata["gpu"] = str(cuda.get_current_device().name)
                        metadata["timing"] = {"simulation_seconds": simulation_seconds}
                        metadata["spatial_interval"] = spatial_interval
                        metadata["realspace_interval"] = realspace_interval
                        metadata["note"] = (
                            "Transient deliberately saved. Comparison samples satisfy "
                            "sample_sweeps > diagnostic_burn_sweeps. First-case timing "
                            "includes any kernel compilation. No checkpoint/resume."
                        )

                        output_dir.mkdir(parents=True, exist_ok=True)
                        start = perf_counter()
                        save_runs(output_file, {L: run})
                        print(f"  Simulation: {simulation_seconds / 3600:.2f} h")
                        print(f"  Saving:     {perf_counter() - start:.1f} s")
                        print(f"  Saved:      {output_file}", flush=True)
                        del metadata, run, simulation, pt, runtime, model
                        gc.collect()


if __name__ == "__main__":
    main()
