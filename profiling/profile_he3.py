"""Bounded, warmed He3 benchmarks and NVIDIA profiler capture targets.

Run from the repository root. This uses the production model and kernels
without modifying their implementation or writing simulation datasets.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from numba import cuda

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.parallel_tempering import ParallelTempering
from core.runtime import Runtime
from models.he3.model import He3Model
from models.he3.measure import (
    he3_energy_kernel, he3_ordered_measure_kernel,
    he3_ordered_cached_measure_kernel,
)
from models.he3.update import metropolis_color
from utils.temperature_ladder import make_temperature_ladder


def kernel_resources(kernel):
    result = {}
    for label, method in (
        ("registers_per_thread", "get_regs_per_thread"),
        ("local_bytes_per_thread", "get_local_mem_per_thread"),
        ("shared_bytes_per_block", "get_shared_mem_per_block"),
        ("max_threads_per_block", "get_max_threads_per_block"),
    ):
        try:
            result[label] = {str(k): int(v) for k, v in getattr(kernel, method)().items()}
        except (AttributeError, RuntimeError) as error:
            result[label] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--replicas", type=int, default=100)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--T-min", type=float, default=.15)
    parser.add_argument("--T-max", type=float, default=.45)
    parser.add_argument("--Tc", type=float, default=.257)
    parser.add_argument("--tc-window", type=float, default=.06)
    parser.add_argument("--tc-fraction", type=float, default=.60)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--sweeps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["updates", "pt", "production"],
                        choices=["updates", "pt", "production", "energy", "measure", "probe"])
    parser.add_argument("--cuda-profile-api", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump-ptx", type=Path)
    args = parser.parse_args()
    if args.replicas < 2 or min(args.sweeps, args.repeats) < 1 or args.warmup < 10:
        parser.error("Need at least two replicas, positive sweeps/repeats, and warmup >= 10")

    device = cuda.get_current_device()
    name = device.name.decode() if isinstance(device.name, bytes) else device.name
    report = {"device": name, "arguments": vars(args).copy(), "cases": []}
    report["arguments"] = {k: str(v) if isinstance(v, Path) else v
                           for k, v in report["arguments"].items()}
    temperatures = make_temperature_ladder(
        T_min=args.T_min, T_max=args.T_max, n_T=args.replicas,
        Tc=args.Tc, tc_window=args.tc_window, tc_fraction=args.tc_fraction,
    )
    for L in args.sizes:
        setup_start = time.perf_counter()
        runtime = Runtime(He3Model(K=1, delta=.15, pressure=20, reduced_temperature=1),
                          L=L, temperatures=temperatures, threads=args.threads, seed=42+L)
        pt = ParallelTempering(runtime, exchange_interval=10, seed=43+L)
        for _ in range(args.warmup):
            pt.sweep()
        pt.measure()  # Compile every path before collecting any timing or trace.
        cuda.synchronize()
        setup_seconds = time.perf_counter() - setup_start

        def workload(mode):
            if mode == "probe":
                runtime.sweep()
                pt.attempt_exchanges()
                pt.measure()
                return
            for step in range(1, args.sweeps + 1):
                if mode == "updates":
                    runtime.sweep()
                elif mode in ("pt", "production"):
                    pt.sweep()
                    if mode == "production" and step % 10 == 0:
                        # Same per-sample host copies as Simulation.run().
                        measurement = pt.measure()
                        for value in measurement.values():
                            value.copy()
                elif mode == "energy":
                    runtime.measure_energy(pt.energies)
                elif mode == "measure":
                    pt.measure()

        for mode in args.modes:
            elapsed = []
            for _ in range(args.repeats):
                cuda.synchronize()
                if args.cuda_profile_api:
                    cuda.profile_start()
                started = time.perf_counter()
                workload(mode)
                cuda.synchronize()
                elapsed.append(time.perf_counter() - started)
                if args.cuda_profile_api:
                    cuda.profile_stop()
            iterations = 1 if mode == "probe" else args.sweeps
            case = {"L": L, "replicas": args.replicas, "mode": mode,
                    "threads": args.threads, "iterations": iterations,
                    "seconds": elapsed, "median_seconds": statistics.median(elapsed),
                    "median_ms_per_iteration": 1000*statistics.median(elapsed)/iterations,
                    "setup_and_warmup_seconds": setup_seconds}
            if mode in ("updates", "pt", "production"):
                case["site_proposals_per_second"] = L*L*args.replicas*iterations/statistics.median(elapsed)
                case["sweeps_per_second"] = iterations/statistics.median(elapsed)
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
        del pt, runtime
        gc.collect()
        cuda.current_context().deallocations.clear()

    kernels = {"metropolis_color": metropolis_color, "he3_energy_kernel": he3_energy_kernel,
               "he3_ordered_measure_kernel": he3_ordered_measure_kernel,
               "he3_ordered_cached_measure_kernel": he3_ordered_cached_measure_kernel}
    report["kernel_resources"] = {name: kernel_resources(kernel) for name, kernel in kernels.items()}
    if args.dump_ptx:
        args.dump_ptx.mkdir(parents=True, exist_ok=True)
        for name, kernel in kernels.items():
            for index, ptx in enumerate(kernel.inspect_asm().values()):
                (args.dump_ptx / f"{name}_{index}.ptx").write_text(ptx, encoding="utf-8")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved benchmark: {args.output}", flush=True)


if __name__ == "__main__":
    main()
