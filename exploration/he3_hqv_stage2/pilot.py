"""Bounded exploratory snapshots using unchanged Gaussian/PT components.

Run: python -B exploration/he3_hqv_stage2/pilot.py
Default: gD=0,L64 only, 30k thermal +60k measurement, max300s after setup.
Optional --weak-soc explicitly requests the same small gD=.001 pilot.
Existing output directories are refused. No original dataset is overwritten.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from numba import cuda
from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from core.parallel_tempering import ParallelTempering
from utils.data_io import _json_ready
from energy import energy

OUT=Path(__file__).resolve().parent/'results'
TEMPERATURES=np.unique(np.round(np.r_[np.linspace(.33,.37,9),np.linspace(.372,.396,17),np.linspace(.40,.46,7),.35,.375,.3815,.389,.41],7))
TARGETS=np.array([.35,.375,.3815,.389,.41])


def capture(pt,selected,path,extra):
    """Freeze host stepping, copy exact maps and only selected physical slices."""
    cuda.synchronize()
    sweep=pt.sweeps
    tw=pt.temperature_to_walker.copy_to_host()
    wt=pt.walker_to_temperature.copy_to_host()
    np.testing.assert_array_equal(wt[tw],np.arange(pt.R))
    walkers=tw[selected]
    # Contiguous one-walker slices copy only selected fields, not all replicas.
    fields=np.stack([pt.runtime.state[int(w):int(w)+1].copy_to_host()[0] for w in walkers])
    scalars=pt.measure()
    cuda.synchronize()
    np.testing.assert_array_equal(tw,pt.temperature_to_walker.copy_to_host())
    assert sweep==pt.sweeps
    density=np.sum(abs(fields.astype(np.complex128))**2,axis=(1,2,3,4))
    np.testing.assert_allclose(density,np.asarray(scalars['density'])[selected],rtol=2e-6,atol=1e-4)
    meta=dict(schema_version=1,axes=['saved_temperature','x','y','spin','orbital'],
        spin_components=['x','y','z'],orbital_components=['x','y'],boundary='periodic',
        observation_event='after_complete_sweep_and_scheduled_exchange',sweep=sweep,
        exchange_round=pt.exchange_rounds,temperature_indices=selected,temperatures=pt.temperatures[selected],
        physical_walkers=walkers,model_parameters=pt.model.parameters(),L=pt.L,
        seed=pt.seed,exchange_seed=pt.exchange_seed,**extra)
    np.savez_compressed(path,fields=fields,temperature_to_walker=tw,walker_to_temperature=wt,
        metadata=np.array(json.dumps(_json_ready(meta))),
        **{'scalar_'+name:np.asarray(v)[selected] for name,v in scalars.items()})
    return dict(sweep=sweep,path=str(path.relative_to(ROOT)),temperatures=pt.temperatures[selected].tolist(),
        walkers=walkers.tolist(),density_max_relative_difference=float(np.max(abs(density/np.asarray(scalars['density'])[selected]-1))))


def main():
    started=perf_counter();gd=.001 if '--weak-soc' in sys.argv else 0.
    run_dir=OUT/f'pilot_gd={gd:g}_L=64'
    if run_dir.exists():raise FileExistsError(f'Refusing to overwrite {run_dir}')
    run_dir.mkdir(parents=True)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        ROOT/'models/he3/update.py',ROOT/'models/he3/measure.py',ROOT/'models/he3/gaussian_fast.py',
        ROOT/'core/runtime.py',ROOT/'core/parallel_tempering.py',Path(__file__)]}
    source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    # Same-process smoke compiles the existing kernels and verifies exact-order
    # selective capture before the larger pilot. These are actual GPU sweeps.
    model=He3Model(K=1,pressure=20,reduced_temperature=1,g_d=gd)
    smoke=He3GaussianRuntime(model,8,[.35,.38,.40,.44],seed=20261029)
    sp=ParallelTempering(smoke,exchange_interval=5,seed=20261030)
    for _ in range(100):sp.sweep()
    smoke_record=capture(sp,np.array([0,2]),run_dir/'smoke_L8.npz',dict(smoke_test=True))
    with np.load(run_dir/'smoke_L8.npz',allow_pickle=False) as z:
        cpu=[energy(a,betas=model.betas,K=float(model.K),gd=float(model.g_d))['total'] for a in z['fields']]
        gpu=z['scalar_energy']
    np.testing.assert_allclose(cpu,gpu,rtol=3e-6,atol=1e-3)
    smoke_validation=dict(record=smoke_record,cpu_energies=cpu,gpu_energies=gpu.tolist(),
        max_absolute_energy_error=float(np.max(abs(np.array(cpu)-gpu))))
    del sp,smoke
    seed=20261031 if gd==0 else 20262031
    rt=He3GaussianRuntime(model,64,TEMPERATURES,seed=seed,rho=-.9,proposal_mass=0)
    pt=ParallelTempering(rt,exchange_interval=5,seed=seed+1,tracked_walkers=range(rt.R),history_stride=20)
    selected=np.array([np.argmin(abs(TEMPERATURES-t)) for t in TARGETS])
    thermal=30000;measure=60000;scalar_stride=250;snapshot_stride=5000
    extra=dict(smoke_test=False,source_revision=source_revision,source_sha256=hashes,
        initialization='production homogeneous positive-chirality d_z initializer',
        thermalization_sweeps_requested=thermal,measurement_sweeps_requested=measure,
        sampler='float32_conditional_gaussian_full_hamiltonian_mh',rho=-.9,proposal_mass=0.,
        exchange_interval=5,nominal_lattice_spacing=1.)
    sample_sweeps=[];samples={};records=[]
    loop_start=perf_counter();stop_reason='requested_sweeps_completed'
    print(f'Pilot gd={gd:g}, L64, {rt.R} PT temperatures; setup {loop_start-started:.2f}s',flush=True)
    for step in range(1,thermal+measure+1):
        pt.sweep()
        if step%scalar_stride==0:
            values=pt.measure()
            for name,value in values.items():samples.setdefault(name,[]).append(np.asarray(value))
            sample_sweeps.append(step)
        if step>=thermal and (step-thermal)%snapshot_stride==0:
            records.append(capture(pt,selected,run_dir/f'snapshot_{step:06d}.npz',
                dict(extra,production_relative_sweep=step-thermal)))
            print(f'Captured sweep {step}; loop elapsed {perf_counter()-loop_start:.2f}s',flush=True)
        if step%1000==0 and perf_counter()-loop_start>300:
            stop_reason='300_second_loop_cap';break
    cuda.synchronize()
    np.savez_compressed(run_dir/'scalar_history.npz',sweeps=sample_sweeps,temperatures=TEMPERATURES,
        **{k:np.asarray(v) for k,v in samples.items()})
    summary=dict(**extra,g_d=gd,L=64,temperatures=TEMPERATURES,selected_temperatures=TARGETS,
        actual_sweeps=pt.sweeps,stop_reason=stop_reason,snapshot_records=records,
        setup_seconds=loop_start-started,sampling_and_capture_seconds=perf_counter()-loop_start,
        elapsed_seconds=perf_counter()-started,pt=pt.diagnostics(),sampling=rt.sampling_diagnostics(),
        smoke_validation=smoke_validation)
    (run_dir/'run.json').write_text(json.dumps(_json_ready(summary),indent=2),encoding='utf-8')
    print('Saved',run_dir,'total seconds',summary['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
