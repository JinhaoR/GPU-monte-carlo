"""Thin-film He3 simulation with the frozen C160 ladder and checkpointing."""
from pathlib import Path
from time import perf_counter, monotonic
import signal
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

# Physical parameters
pressures = [20.0]
g_d_values = [0.0]                 # Zero SOC first
sizes = [320]                     # Add sizes when ready

K = 1.0
betas = (-0.5, 1, 1, 1, -1)
delta = 0.15
reduced_temperature = 1.0

# Sampling
rho = -0.9
proposal_mass = 0.0
thermalization_sweeps = 3_000_000   # L320 diagnostic, not proven equilibration
measurement_sweeps = 5_000_000
sample_interval = 50
exchange_interval = 5
spatial_interval = 1_000
realspace_interval = 10_000
realspace_temperatures = {
    0.0: [0.3815, 0.38182, 0.3830],
    0.01: [0.3852, 0.3856, 0.3870, 0.3896],
}

# GPU and parallel tempering
threads = 16
seed_base = 42
pt_seed_base = 43
history_stride = 10              # Exchange rounds = 50 sweeps
traffic_interval = 50_000

# Checkpointing (for a 12-hour Slurm allocation)
checkpoint_seconds = 3_600
max_wall_hours = 11.5

# Temperature ladder and output
root = Path(__file__).resolve().parent
ladder_file = root / "exploration/he3_pt_ladder_validation/ladder_C.txt"
campaign = "he3_large_C160"
data_root = root / "data/he3"


# ============================================================
# SIMULATION
# ============================================================

def main():
    temperatures = np.loadtxt(ladder_file)  # Exact frozen C160 values
    if len(temperatures) != 160:
        raise ValueError("Expected the frozen 160-temperature ladder")

    deadline = monotonic() + max_wall_hours * 3600
    stopping = [False]

    def request_stop(signum, frame):
        stopping[0] = True

    for name in ("SIGINT", "SIGTERM", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)

    for L in sizes:
        for pressure in pressures:
            for g_d in g_d_values:
                if stopping[0] or monotonic() >= deadline:
                    return

                folder = f"P={pressure:g}_gd={g_d:g}_gaussian"
                output_dir = data_root / campaign / folder
                output_file = output_dir / f"L={L}.npz"
                partial_file = output_dir / f"L={L}.partial.npz"
                checkpoint_file = output_dir / f"L={L}.checkpoint.npz"

                if output_file.exists():
                    print(f"Skipping completed: {output_file}", flush=True)
                    continue
                if partial_file.exists() and not checkpoint_file.exists():
                    raise RuntimeError(f"Partial output without checkpoint: {partial_file}")
                output_dir.mkdir(parents=True, exist_ok=True)

                print(f"\nHe3: L={L}, P={pressure:g}, g_d={g_d:g}", flush=True)

                model = He3Model(
                    K=K, betas=betas, delta=delta, pressure=pressure,
                    reduced_temperature=reduced_temperature, g_d=g_d,
                )
                model.spatial_every = spatial_interval // sample_interval
                model.realspace_every = realspace_interval // sample_interval
                model.charged_profiles = True
                model.realspace_indices = np.unique([
                    np.argmin(abs(temperatures - t)) for t in realspace_temperatures[g_d]
                ])

                runtime = He3GaussianRuntime(
                    model=model, L=L, temperatures=temperatures, threads=threads,
                    seed=seed_base + L, rho=rho, proposal_mass=proposal_mass,
                )
                pt = ParallelTempering(
                    runtime, exchange_interval=exchange_interval, seed=pt_seed_base + L,
                    tracked_walkers=range(len(temperatures)), history_stride=history_stride,
                )
                simulation = Simulation(
                    runtime=pt, thermalization_sweeps=thermalization_sweeps,
                    measurement_sweeps=measurement_sweeps,
                    sample_interval=sample_interval, traffic_interval=traffic_interval,
                )
                if checkpoint_file.exists():
                    simulation.load_checkpoint(checkpoint_file)

                start = perf_counter()
                run = simulation.run(
                    checkpoint_path=checkpoint_file,
                    checkpoint_seconds=checkpoint_seconds,
                    stop_requested=lambda: stopping[0] or monotonic() >= deadline,
                )
                simulation_seconds = perf_counter() - start
                add_spatial_samples(
                    run, model.spatial_samples, model.realspace_samples,
                    charged_profiles=True, clear=False,
                )

                # Metadata
                metadata = run["metadata"]
                metadata["timing"] = {"this_invocation_seconds": simulation_seconds}
                metadata["campaign"] = {"id": campaign, "sizes": list(sizes)}
                metadata["sampler"] = "gaussian"
                metadata["threads"] = threads
                metadata["temperature_ladder"] = {"file": str(ladder_file.relative_to(root))}
                metadata["spatial_interval"] = spatial_interval
                metadata["realspace_interval"] = realspace_interval
                metadata["initialization"] = "ordered: positive chirality, d=(0,0,1), phase=0"
                metadata["sampler_parameters"] = {"rho": rho, "proposal_mass": proposal_mass}
                metadata["sampling"] = runtime.sampling_diagnostics()
                metadata["gpu"] = str(cuda.get_current_device().name)

                # Save complete or partial run; checkpoint is the restart state
                destination = output_file if simulation.complete else partial_file
                start = perf_counter()
                save_runs(destination, {L: run}, binary_metadata=True)
                print(f"  Simulation: {simulation_seconds / 3600:.2f} h")
                print(f"  Saving:     {perf_counter() - start:.1f} s")
                print(f"  Saved:      {destination}", flush=True)

                if not simulation.complete:
                    return
                del run, simulation, pt, runtime, model


if __name__ == "__main__":
    main()
