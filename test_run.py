"""Run the He3 C160 campaign. Edit SETTINGS; rerun to resume checkpoints."""

from pathlib import Path
from time import monotonic, perf_counter
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

pressures = [20.0]
g_d_values = [0.0]
sizes = [256]
initializations = ["ordered"]  # or ["ordered", "random"]

# (thermalization sweeps, production sweeps) for each size
allocations = {
    160: (1_000_000, 3_000_000),
    256: (3_000_000, 6_000_000),
    320: (3_000_000, 5_000_000),
    384: (4_000_000, 8_000_000),
}

K = 1.0
betas = (-0.5, 1, 1, 1, -1)
delta = 0.15
reduced_temperature = 1.0
rho = -0.9
proposal_mass = 0.0

sample_interval = 100
exchange_interval = 5
spatial_interval = 1_000
realspace_interval = 20_000
realspace_temperatures = {
    0.0: [0.3815, 0.38182, 0.3830],
    0.01: [0.3852, 0.3856, 0.3870],
}

threads = 16
history_stride = 10  # 10 exchanges = 50 sweeps
traffic_interval = 50_000
checkpoint_seconds = 3_600
max_wall_hours = 11.5
resume = True

seeds = {
    (0.0, "ordered"): 81_000, (0.0, "random"): 82_000,
    (0.01, "ordered"): 83_000, (0.01, "random"): 84_000,
}

root = Path(__file__).resolve().parent
ladder_file = root / "exploration/he3_pt_ladder_validation/ladder_C.txt"
data_root = root / "data/he3_large_C160"


# ============================================================
# SIMULATION
# ============================================================

def main():
    temperatures = np.loadtxt(ladder_file)
    if len(temperatures) != 160 or not np.all(np.diff((1 / temperatures).astype(np.float32)) < 0):
        raise ValueError("Invalid C160 temperature ladder")
    for interval in (spatial_interval, realspace_interval):
        if interval < sample_interval or interval % sample_interval:
            raise ValueError("Spatial cadences must be multiples of sample_interval")

    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    for name in ("SIGINT", "SIGTERM", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)
    deadline = monotonic() + 3600 * max_wall_hours

    for L in sizes:
        for pressure in pressures:
            for g_d in g_d_values:
                for initialization in initializations:
                    if stopping or monotonic() >= deadline:
                        return

                    folder = data_root / f"P={pressure:g}_gd={g_d:g}"
                    output = folder / f"L={L}_{initialization}.npz"
                    checkpoint = folder / f"L={L}_{initialization}.checkpoint.npz"
                    partial = folder / f"L={L}_{initialization}.partial.npz"

                    if output.exists():
                        print(f"Already complete: {output}", flush=True)
                        continue
                    if partial.exists() and not checkpoint.exists():
                        raise RuntimeError(f"Partial output without checkpoint: {partial}")
                    if checkpoint.exists() and not resume:
                        raise FileExistsError(checkpoint)
                    if initialization not in ("ordered", "random"):
                        raise ValueError(initialization)

                    folder.mkdir(parents=True, exist_ok=True)
                    seed = seeds[g_d, initialization] + L
                    burn, production = allocations[L]
                    model = He3Model(K=K, betas=betas, delta=delta, pressure=pressure,
                                     reduced_temperature=reduced_temperature, g_d=g_d)
                    model.spatial_every = spatial_interval // sample_interval
                    model.realspace_every = realspace_interval // sample_interval
                    model.charged_profiles = True
                    model.realspace_indices = np.unique([
                        np.argmin(abs(temperatures - t)) for t in realspace_temperatures[g_d]
                    ])

                    runtime = He3GaussianRuntime(model, L, temperatures, threads=threads,
                                                 seed=seed, rho=rho, proposal_mass=proposal_mass)
                    if initialization == "random" and not checkpoint.exists():
                        rng = np.random.default_rng(seed + 2)
                        n0 = 1 / float(model.betas[1] + model.betas[3] + model.betas[4])
                        for walker in range(runtime.R):
                            field = (np.sqrt(n0 / 12) * (
                                rng.normal(size=(L, L, 3, 2)) + 1j * rng.normal(size=(L, L, 3, 2))
                            )).astype(np.complex64)
                            runtime.state[walker].copy_to_device(field)
                        runtime.invalidate_energy_cache()

                    pt = ParallelTempering(runtime, exchange_interval=exchange_interval,
                                           seed=seed + 1, tracked_walkers=range(len(temperatures)),
                                           history_stride=history_stride)
                    simulation = Simulation(pt, burn, production, sample_interval,
                                            traffic_interval=traffic_interval)
                    if checkpoint.exists():
                        simulation.load_checkpoint(checkpoint)

                    print(f"L={L}, g_D={g_d}, {initialization}: sweep {simulation.completed_sweeps}", flush=True)
                    start = perf_counter()
                    run = simulation.run(
                        checkpoint_path=checkpoint,
                        checkpoint_seconds=checkpoint_seconds,
                        stop_requested=lambda: stopping or monotonic() >= deadline,
                    )

                    # ========================================================
                    # SAVE
                    # ========================================================
                    add_spatial_samples(run, model.spatial_samples, model.realspace_samples,
                                        charged_profiles=True, clear=False)
                    run["metadata"].update(
                        campaign="large_C160", initialization=initialization,
                        host_initialization_seed=seed + 2 if initialization == "random" else None,
                        sampler="gaussian", sampler_parameters={"rho": rho, "proposal_mass": proposal_mass},
                        sampling=runtime.sampling_diagnostics(),
                        gpu=str(cuda.get_current_device().name), threads=threads,
                        ladder_file=str(ladder_file.relative_to(root)),
                        spatial_interval=spatial_interval, realspace_interval=realspace_interval,
                        timing={"this_invocation_seconds": perf_counter() - start},
                    )
                    destination = output if simulation.complete else partial
                    save_runs(destination, {L: run}, binary_metadata=True)
                    print(f"Saved: {destination}", flush=True)

                    if simulation.complete:
                        partial.unlink(missing_ok=True)
                    else:
                        return  # Resume this case on the next invocation.


if __name__ == "__main__":
    main()
