"""Run Kawamura (PRL 82, 964, 1999) simulations with parallel tempering.

Run from the GPU-monte-carlo project root. Each lattice size is saved
independently as data/kawamura/<campaign>/L=<size>.npz.
"""

import os
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from core.runtime import Runtime
from core.parallel_tempering import ParallelTempering
from core.simulation import Simulation

from models.kawamura.model import KawamuraModel

from utils.data_io import save_runs
from utils.temperature_ladder import make_temperature_ladder


# ============================================================
# SETTINGS
# ============================================================

# Physical/model parameters
J = 1.0
delta = 1.5                # Phase proposal half-width, in radians (0 < delta <= pi)
sizes = [64, 80, 96]     # Even L >= 4 for the four-color update

# Sampling
thermalization_sweeps = 500_000
measurement_sweeps = 1_000_000
sample_interval = 10
exchange_interval = 10

# GPU and parallel tempering
threads = 16
seed_base = 42
pt_seed_base = 43
tracked_walkers = "all"  # "all", None, or list of walker IDs
history_stride = 10       # PT exchange rounds, not Metropolis sweeps

# Temperature ladder
# Both candidate transitions are inside this range:
# lower transition T/J ~ 0.692, chiral transition T/J ~ 0.753.
# Tc=None gives a ladder uniform in inverse temperature (beta).
temperature_ladder = {
    "T_min": 0.65 * J,
    "T_max": 0.80 * J,
    "n_T": 100,
    "Tc": None,
}

# Output
campaign = "kawamura_reproduction_01"
data_root = Path("data/kawamura")


# ============================================================
# SIMULATION
# ============================================================

def main():
    if not np.isfinite(J) or J <= 0:
        raise ValueError("J must be positive and finite")
    if not np.isfinite(delta) or not 0 < delta <= np.pi:
        raise ValueError("Require 0 < delta <= pi")
    if not sizes or any(L < 4 or L % 2 for L in sizes):
        raise ValueError("All lattice sizes must be even and >= 4")

    temperatures = make_temperature_ladder(**temperature_ladder)
    walker_ids = (
        range(len(temperatures))
        if tracked_walkers == "all"
        else tracked_walkers
    )

    # Save one file per size, in the same structure as the He3 runner.
    def output_path(L):
        return data_root / campaign / f"L={L}.npz"

    paths = [output_path(L) for L in sizes]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate simulation output paths (check sizes)")

    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Output files already exist:\n" + "\n".join(map(str, existing))
        )

    print(f"\nKawamura simulation: {campaign}")
    print(f"J={J:g}, delta={delta:g}")
    print(f"Lattice sizes: {sizes}")
    print(
        f"Temperature ladder: {len(temperatures)} replicas, "
        f"T/J in [{temperatures[0] / J:.5f}, "
        f"{temperatures[-1] / J:.5f}]"
    )
    for label, target_over_J in (("lower", 0.692), ("chiral", 0.753)):
        idx = int(np.argmin(np.abs(temperatures - J * target_over_J)))
        print(
            f"  Nearest to {label} transition T/J≈{target_over_J:g}: "
            f"T/J={temperatures[idx] / J:.6f} (index {idx})"
        )

    saved_paths = []

    for L in sizes:
        print(f"\nL={L}")

        model = KawamuraModel(J=J, delta=delta)
        runtime = Runtime(
            model=model,
            L=L,
            temperatures=temperatures,
            threads=threads,
            seed=seed_base + L,
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

        metadata = run["metadata"]
        metadata["timing"] = {
            "simulation_seconds": simulation_seconds,
        }
        metadata["campaign"] = {
            "id": campaign,
            "sizes": list(sizes),
        }
        metadata["sampler"] = "metropolis"
        metadata["threads"] = threads
        metadata["temperature_ladder"] = dict(temperature_ladder)
        metadata["sampler_parameters"] = {"delta": delta}

        # Stage in the destination filesystem, then create the final name
        # without overwriting an existing file.
        path = output_path(L)
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
