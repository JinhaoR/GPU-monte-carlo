"""Bounded device/measurement/checkpoint benchmark; no equilibration claims.

python exploration/he3_large_size_preparation/benchmark.py --L 256
One size per process gives an interpretable host peak. At most 120 local sweeps.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import numpy as np
from numba import cuda

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.parallel_tempering import ParallelTempering
from core.simulation import Simulation
from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from models.he3.spatial import add_spatial_samples
from utils.data_io import save_runs


def host_memory():
    if sys.platform == 'win32':
        import ctypes as c
        class Counters(c.Structure):
            _fields_ = [('cb', c.c_ulong), ('PageFaultCount', c.c_ulong)] + [
                (name, c.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        value = Counters()
        value.cb = c.sizeof(value)
        c.windll.kernel32.GetCurrentProcess.restype = c.c_void_p
        if not c.windll.psapi.GetProcessMemoryInfo(c.c_void_p(c.windll.kernel32.GetCurrentProcess()), c.byref(value), value.cb):
            raise OSError('GetProcessMemoryInfo failed')
        return value.PeakWorkingSetSize, value.WorkingSetSize
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, None


def timed(call, repetitions):
    cuda.synchronize()
    start = perf_counter()
    for _ in range(repetitions):
        call()
    cuda.synchronize()
    return (perf_counter()-start)/repetitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--L', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    args = parser.parse_args()
    T = np.loadtxt(ROOT/'exploration/he3_pt_ladder_validation/ladder_C.txt')
    start = perf_counter()
    model = He3Model(pressure=20, g_d=.01)
    runtime = He3GaussianRuntime(model, args.L, T, seed=931, rho=-.9, proposal_mass=0.)
    pt = ParallelTempering(runtime, exchange_interval=5, seed=932, tracked_walkers=range(len(T)), history_stride=10)
    initialization_seconds = perf_counter()-start
    timed(pt.sweep, 20)  # compile/warm up, NOT timed as equilibrium dynamics
    sweep_seconds = timed(pt.sweep, 100)
    scalar_seconds = timed(pt.measure, 1)  # compile scalar measurement
    scalar_seconds = timed(pt.measure, 5)
    model.spatial_every = 1
    timed(pt.measure, 1)
    spatial_seconds = timed(pt.measure, 3)
    model.realspace_every, model.charged_profiles = 1, True
    model.realspace_indices = np.unique([np.argmin(abs(T-t)) for t in (.3852,.3856,.3870)])
    timed(pt.measure, 1)
    profile_seconds = timed(pt.measure, 3)
    gpu_bytes = sum(a.nbytes for a in (runtime.state, runtime.rng_states, runtime.betas,
                    runtime.gaussian.counts, pt.energies, pt._rng_states, pt._swap_counts,
                    pt._trips, pt.walker_to_temperature, pt.temperature_to_walker))
    gpu_bytes += model._spatial_meter.allocated_bytes + sum(c.nbytes for c in pt._history_chunks+pt._endpoint_history_chunks)
    # The measurements above are timing probes, not one Simulation's sample axis.
    model.spatial_samples.clear(); model.realspace_samples.clear(); model._measurement_count = 0
    simulation = Simulation(pt, 0, 1, 1, traffic_interval=1)
    run = simulation.run()
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary)/'checkpoint.npz'
        start = perf_counter(); simulation.save_checkpoint(checkpoint)
        checkpoint_seconds = perf_counter()-start
        checkpoint_bytes = checkpoint.stat().st_size
        start = perf_counter(); simulation.load_checkpoint(checkpoint)
        restore_seconds = perf_counter()-start
        add_spatial_samples(run, model.spatial_samples, model.realspace_samples, charged_profiles=True, clear=False)
        output = Path(temporary)/'measurements.npz'
        start = perf_counter(); save_runs(output, {args.L:run}, binary_metadata=True)
        save_seconds = perf_counter()-start
        output_bytes = output.stat().st_size
    host_peak, host_rss = host_memory()
    result = dict(L=args.L, R=len(T), gpu=str(cuda.get_current_device().name),
                  initialization_seconds=initialization_seconds, sweeps_timed=100,
                  sweep_seconds=sweep_seconds, scalar_seconds=scalar_seconds,
                  scalar_plus_spatial_seconds=spatial_seconds,
                  scalar_spatial_profiles_seconds=profile_seconds,
                  estimated_cadence_overhead=(scalar_seconds/50+(spatial_seconds-scalar_seconds)/1000
                                             +(profile_seconds-spatial_seconds)/20000)/sweep_seconds,
                  allocated_device_bytes=gpu_bytes, host_peak_bytes=host_peak,
                  host_rss_bytes=host_rss, checkpoint_bytes=checkpoint_bytes,
                  checkpoint_seconds=checkpoint_seconds, restore_seconds=restore_seconds,
                  one_event_output_bytes=output_bytes, save_seconds=save_seconds,
                  scalar_bytes_per_event=sum(v[0].nbytes for k,v in run['samples'].items() if not k.startswith(('spatial_', 'realspace_'))),
                  spatial_bytes_per_event=sum(v[0].nbytes for k,v in run['samples'].items() if k.startswith('spatial_')),
                  profile_bytes_per_event=sum(v[0].nbytes for k,v in run['samples'].items() if k.startswith('realspace_')))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/f'benchmark_L{args.L}.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
