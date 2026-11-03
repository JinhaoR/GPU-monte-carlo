"""C160 reference runner. Edit SETTINGS, then run inside a GPU allocation.

Matching completed cases are skipped; unfinished cases resume their checkpoints.
The default remains one L256, gd=0, ordered reference. No jobs are submitted.
"""
from pathlib import Path
from time import monotonic, perf_counter
import signal
import numpy as np
from numba import cuda

from core.parallel_tempering import ParallelTempering
from core.simulation import Simulation
from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from models.he3.measure import MEASUREMENT_NAMES
from models.he3.spatial import add_spatial_samples, FIELDS, CHARGED_PROFILES
from utils.checkpoint import same_settings
from utils.data_io import load_runs, save_runs


# ============================================================
# SETTINGS
# ============================================================

# Cases -- enable larger sizes only after reviewing the L256 reference.
pressures = [20.0]
g_d_values = [0.0]                  # campaign: [0.0, 0.01]
sizes = [256]                      # planned: 160, 256, 320; 384 conditional
initializations = ['ordered']      # independent partner: ['random']

# Sweeps by size: (thermalization, measurement).
allocations = {
    160: (1_000_000, 3_000_000),
    256: (3_000_000, 6_000_000),
    320: (3_000_000, 5_000_000),
    384: (4_000_000, 8_000_000),
}

# Physical parameters and Gaussian sampler.
K = 1.0
betas = (-0.5, 1, 1, 1, -1)
delta = 0.15
reduced_temperature = 1.0
gaussian_rho = -0.9
gaussian_proposal_mass = 0.0

# Measurements and parallel tempering (intervals are in sweeps).
sample_interval = 100
exchange_interval = 5
spatial_interval = 1_000
realspace_interval = 20_000
profile_temperatures = {0.0: [.3815, .38182, .3830], 0.01: [.3852, .3856, .3870]}
threads = 16
history_stride = 10                # exchange rounds: all walkers every 50 sweeps
traffic_interval = 50_000          # also recorded at burn boundary and completion

# Separate seeds by coupling/preparation; add L, as in existing checkpoints.
seeds = {(0.0, 'ordered'): 81000, (0.0, 'random'): 82000,
         (0.01, 'ordered'): 83000, (0.01, 'random'): 84000}

# Continuation and output.
resume = True
checkpoint_seconds = 3_600
max_wall_hours = 11.5              # leaves 30 minutes in a 12-hour allocation
directory = Path(__file__).resolve().parent
ladder_file = directory / 'exploration/he3_pt_ladder_validation/ladder_C.txt'
output_dir = directory / 'data/he3_large_C160'


# ============================================================
# SIMULATION
# ============================================================

def main():
    temperatures = np.loadtxt(ladder_file)  # exact frozen values; never regenerate
    if len(temperatures) != 160 or not np.all(np.diff((1/temperatures).astype(np.float32)) < 0):
        raise ValueError('Invalid frozen C160 ladder')
    for cadence in (spatial_interval, realspace_interval):
        if cadence < sample_interval or cadence % sample_interval:
            raise ValueError('Measurement cadences must be multiples of sample_interval')
    if any(preparation not in ('ordered', 'random') for preparation in initializations):
        raise ValueError('Unknown preparation')
    case_seeds = [seeds[gd, preparation]+L for L in sizes for _ in pressures
                  for gd in g_d_values for preparation in initializations]
    if len(case_seeds) != len(set(case_seeds)):
        raise ValueError('Selected cases must have independent seeds; check the case lists and seeds')

    stopping = False
    def stop_signal(signum, frame):
        nonlocal stopping
        stopping = True  # no file or CUDA operations inside signal handlers
    signals = [getattr(signal, name) for name in ('SIGINT', 'SIGTERM', 'SIGUSR1') if hasattr(signal, name)]
    previous_handlers = {sig: signal.signal(sig, stop_signal) for sig in signals}
    deadline = monotonic()+3600*max_wall_hours
    def stop_requested():
        return stopping or monotonic() >= deadline
    try:
        for L in sizes:
            for pressure in pressures:
                for g_d in g_d_values:
                    for initialization in initializations:
                        if stop_requested():
                            return
                        completed = run_case(L, pressure, g_d, initialization, temperatures, stop_requested)
                        if not completed:
                            return
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def run_case(L, pressure, g_d, initialization, temperatures, stop_requested):
    folder = output_dir / f'P={pressure:g}_gd={g_d:g}'
    output = folder / f'L={L}_{initialization}.npz'
    checkpoint = folder / f'L={L}_{initialization}.checkpoint.npz'
    partial = folder / f'L={L}_{initialization}.partial.npz'
    seed = seeds[g_d, initialization]+L
    thermalization_sweeps, measurement_sweeps = allocations[L]

    model = He3Model(K=K, betas=betas, delta=delta, pressure=pressure,
                     reduced_temperature=reduced_temperature, g_d=g_d)
    model.spatial_every = spatial_interval//sample_interval
    model.realspace_every = realspace_interval//sample_interval
    model.charged_profiles = True
    model.realspace_indices = np.unique([np.argmin(abs(temperatures-t)) for t in profile_temperatures[g_d]])
    runtime = He3GaussianRuntime(model, L, temperatures, threads=threads, seed=seed,
                                rho=gaussian_rho, proposal_mass=gaussian_proposal_mass)
    if initialization == 'random' and not checkpoint.exists() and not output.exists():
        initialize_random(runtime)
    pt = ParallelTempering(runtime, exchange_interval=exchange_interval, seed=seed+1,
                           tracked_walkers=range(len(temperatures)), history_stride=history_stride)
    simulation = Simulation(pt, thermalization_sweeps, measurement_sweeps,
                            sample_interval, traffic_interval=traffic_interval)
    metadata = case_metadata(simulation, initialization)
    if resume_or_skip(simulation, output, checkpoint, partial, metadata):
        return True

    print(f'L={L}, gd={g_d}, {initialization}; starting at sweep {simulation.completed_sweeps}', flush=True)
    start = perf_counter()
    run = simulation.run(checkpoint_path=checkpoint, checkpoint_seconds=checkpoint_seconds,
                         stop_requested=stop_requested)
    save_case(run, model, runtime, metadata, output if simulation.complete else partial, start)
    print(f'Checkpoint: {checkpoint}', flush=True)
    return simulation.complete


def initialize_random(runtime):
    """Same draws and walker order as the original runner; bounded host memory."""
    model, L = runtime.model, runtime.L
    rng = np.random.default_rng(runtime.seed+2)
    n0 = 1/float(model.betas[1]+model.betas[3]+model.betas[4])
    for walker in range(runtime.R):
        field = (np.sqrt(n0/12)*(rng.normal(size=(L,L,3,2))
                  + 1j*rng.normal(size=(L,L,3,2)))).astype(np.complex64)
        runtime.state[walker].copy_to_device(field)
    runtime.invalidate_energy_cache()


def resume_or_skip(simulation, output, checkpoint, partial, metadata):
    """Accept only matching files. A partial archive is never a restart state."""
    if partial.exists() and not checkpoint.exists() and not output.exists():
        raise ValueError(f'Partial output has no checkpoint: {partial}')
    reference = None
    if checkpoint.exists():
        if not resume and not output.exists():
            raise FileExistsError(f'Resume is disabled: {checkpoint}')
        simulation.load_checkpoint(checkpoint)  # existing exact compatibility checks
        if output.exists() and not simulation.complete:
            raise ValueError(f'Final output conflicts with an unfinished checkpoint: {output}')
        if output.exists() or partial.exists():
            reference = simulation.result()
            model = simulation.runtime.model
            add_spatial_samples(reference, model.spatial_samples, model.realspace_samples,
                                charged_profiles=True, clear=False)

    for path in (output, partial):
        if not path.exists():
            continue
        saved = load_runs(path)
        if set(saved) != {simulation.runtime.L}:
            raise ValueError(f'Wrong lattice sizes in {path}')
        run = saved[simulation.runtime.L]
        meta, samples = run['metadata'], run['samples']
        for key, expected in metadata.items():
            if key not in meta or not same_settings(meta[key], expected):
                raise ValueError(f'Incompatible {key} in {path}')
        pt = meta['parallel_tempering']
        if (pt['exchange_interval'] != exchange_interval or pt['exchange_seed'] != metadata['seed']+1
                or pt['walker_history']['stride'] != history_stride
                or not np.array_equal(pt['walker_history']['walker_ids'], np.arange(simulation.runtime.R))):
            raise ValueError(f'Incompatible PT settings in {path}')
        count = meta['completed_sweeps']
        total = simulation.thermalization_sweeps+simulation.measurement_sweeps
        if (not 0 <= count <= total or meta['complete'] != (count == total)
                or (path == output and not meta['complete']) or (path == partial and meta['complete'])):
            raise ValueError(f'Ambiguous completion state in {path}')
        indices = np.arange(simulation.thermalization_sweeps+sample_interval, count+1, sample_interval)
        if (not np.array_equal(meta['sample_sweeps'], indices)
                or any(name not in samples or samples[name].shape != (len(indices), simulation.runtime.R)
                       for name in MEASUREMENT_NAMES if len(indices))):
            raise ValueError(f'Incomplete measurement history in {path}')
        for prefix, cadence, channels in (
                ('spatial', spatial_interval, ('amplitudes', 'structure', 'walkers', 'energy')),
                ('realspace', realspace_interval, ('profiles', 'means', 'walkers', 'energy'))):
            events = np.arange(cadence//sample_interval-1, len(indices), cadence//sample_interval)
            if len(events) and (not np.array_equal(meta.get(prefix+'_sample_indices'), events)
                    or any(prefix+'_'+name not in samples or len(samples[prefix+'_'+name]) != len(events)
                           for name in channels)):
                raise ValueError(f'Incomplete {prefix} history in {path}')
        if 'realspace_profiles' in samples and (not np.array_equal(
                meta['realspace_temperature_indices'], simulation.runtime.model.realspace_indices)
                or not same_settings(meta['realspace_fields'], FIELDS[:7]+CHARGED_PROFILES)):
            raise ValueError(f'Incompatible profile temperatures or fields in {path}')
        if reference is not None:
            if count > reference['metadata']['completed_sweeps']:
                raise ValueError(f'Output is ahead of its checkpoint: {path}')
            for name, values in samples.items():
                expected = reference['samples'].get(name)
                if expected is None or not np.array_equal(values, expected[:len(values)]):
                    raise ValueError(f'Conflicting {name} history in {path}')
        elif path == output:
            reference = run  # a retained partial must be a prefix of this completed run
    if output.exists():
        print(f'Skipping completed case: {output}', flush=True)
        return True
    return False


# ============================================================
# SAVE
# ============================================================

def case_metadata(simulation, initialization):
    """Fixed case identity, shared by output validation and saving."""
    pt = simulation.runtime
    return dict(L=pt.L, temperatures=pt.temperatures, seed=pt.seed,
        model_parameters=pt.model.parameters(), **simulation.settings(),
        campaign='large_C160', initialization=initialization,
        host_initialization_seed=pt.seed+2 if initialization == 'random' else None,
        sampler='gaussian', sampler_parameters=dict(rho=gaussian_rho, proposal_mass=gaussian_proposal_mass),
        threads=threads, ladder_file=str(ladder_file.relative_to(directory)),
        spatial_interval=spatial_interval, realspace_interval=realspace_interval)


def save_case(run, model, runtime, metadata, destination, start):
    add_spatial_samples(run, model.spatial_samples, model.realspace_samples,
                        charged_profiles=True, clear=False)
    run['metadata'].update(metadata, sampling=runtime.sampling_diagnostics(),
                           gpu=str(cuda.get_current_device().name),
                           timing=dict(this_invocation_seconds=perf_counter()-start))
    if run['metadata']['complete'] and destination.exists():
        raise FileExistsError(f'Final output appeared while running: {destination}')
    save_runs(destination, {runtime.L: run}, binary_metadata=True)
    print(f'Saved {destination}', flush=True)


if __name__ == '__main__':
    main()
