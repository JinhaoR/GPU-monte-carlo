"""Two bounded L64,gD=0 GPU histories, 100k burn +100k production each.

Requires successful validate.py and a preflight benchmark. Refuses overwrite.
No production file is edited. No cluster submission or CPU substitute exists.
"""
import os
os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS']='0';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from time import perf_counter
import hashlib,json,subprocess,sys,platform
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from numba import cuda
import numba
from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from models.he3.measure import MEASUREMENT_NAMES
from core.parallel_tempering import ParallelTempering
from utils.data_io import _json_ready
from gpu_measure import SpatialMeter
from observables import numpy_amplitudes,FIELDS,MOMENTA
from ladder import make_ladder,OUT


def snapshot(pt,indices,amp,scalars,path):
    tw=pt.temperature_to_walker.copy_to_host();wt=pt.walker_to_temperature.copy_to_host()
    np.testing.assert_array_equal(wt[tw],np.arange(pt.R))
    fields=np.stack([pt.runtime.state[int(w):int(w)+1].copy_to_host()[0] for w in tw[indices]])
    ref=numpy_amplitudes(fields)
    np.testing.assert_allclose(ref,amp[indices],rtol=2e-11,atol=2e-10)
    np.savez_compressed(path,fields=fields,temperature_indices=indices,temperatures=pt.temperatures[indices],
        physical_walkers=tw[indices],sweep=pt.sweeps,amplitudes=amp[indices],global_values=scalars[indices])
    return dict(sweep=pt.sweeps,file=path.name,max_cpu_gpu_amplitude_error=float(np.max(abs(ref-amp[indices]))))


def run(label,seed,T,ladder,cap=120):
    folder=OUT/label
    if folder.exists():raise FileExistsError(f'Refusing to overwrite {folder}')
    folder.mkdir()
    start=perf_counter();model=He3Model(pressure=20,g_d=0);L=64
    rt=He3GaussianRuntime(model,L,T,seed=seed,rho=-.9,proposal_mass=0)
    if label=='hot':
        rng=np.random.default_rng(seed+10000)
        n0=1/float(model.betas[1]+model.betas[3]+model.betas[4]);sigma=np.sqrt(n0/12)
        field=(sigma*(rng.normal(size=rt.state.shape)+1j*rng.normal(size=rt.state.shape))).astype(np.complex64)
        rt.state.copy_to_device(field);rt.invalidate_energy_cache()
    pt=ParallelTempering(rt,exchange_interval=5,seed=seed+1,tracked_walkers=range(rt.R),history_stride=20)
    meter=SpatialMeter(L,rt.R)
    # Warm the read-only spatial/global paths; no update or RNG call is hidden.
    meter.measure(rt.state,pt.walker_to_temperature);pt.measure();cuda.synchronize()
    setup=perf_counter()-start
    gs=[];gv=[];ss=[];amps=[];synced=[];maps=[];snapshots=[];progress=[];burn_diag=None
    burn,production=100000,100000;selected=np.array([np.argmin(abs(T-t)) for t in [.37,.3815,.385,.415]])
    loop=perf_counter();stop='requested_sweeps_completed'
    print(f'{label}: L64, {rt.R} temperatures, setup {setup:.2f}s; bounded loop starts',flush=True)
    for step in range(1,burn+production+1):
        pt.sweep()
        if step%50==0:
            values=pt.measure();arr=np.stack([values[k] for k in MEASUREMENT_NAMES],axis=-1)
            if not np.all(np.isfinite(arr)):raise FloatingPointError('nonfinite global measurement')
            gs.append(step);gv.append(arr)
        if step%250==0:
            amp=meter.measure(rt.state,pt.walker_to_temperature)
            if not np.all(np.isfinite(amp)):raise FloatingPointError('nonfinite Fourier amplitude')
            # This arr is from precisely the same completed sweep/exchange.
            np.testing.assert_allclose(amp[:,0,0].real,arr[:,2],rtol=4e-6,atol=1e-3)
            np.testing.assert_allclose(amp[:,1,0],arr[:,3]+1j*arr[:,4],rtol=4e-6,atol=1e-3)
            ss.append(step);amps.append(amp);synced.append(arr);maps.append(pt.temperature_to_walker.copy_to_host())
            if step in [burn,burn+production//2,burn+production]:
                snapshots.append(snapshot(pt,selected,amp,arr,folder/f'snapshot_{step:06d}.npz'))
        if step==burn:
            burn_diag=pt.diagnostics();burn_diag.pop('walker_history',None)
        if step%10000==0:
            cuda.synchronize();elapsed=perf_counter()-loop
            progress.append(dict(sweep=step,loop_seconds=elapsed))
            print(f'{label}: sweep {step}, {elapsed:.2f}s',flush=True)
        if step%1000==0 and perf_counter()-loop>cap:
            stop='120_second_loop_cap';break
    cuda.synchronize();loop_seconds=perf_counter()-loop
    diag=pt.diagnostics();hist=diag.pop('walker_history')
    np.savez_compressed(folder/'records.npz',temperatures=T,global_sweeps=gs,global_values=np.asarray(gv),
        global_names=np.array(MEASUREMENT_NAMES),spatial_sweeps=ss,amplitudes=np.asarray(amps),
        spatial_global_values=np.asarray(synced),temperature_to_walker=np.asarray(maps),
        field_names=np.array(FIELDS),momenta=np.array(MOMENTA),production_start_sweep=burn,
        walker_history_sweeps=hist['sweeps'],walker_history_indices=hist['temperature_indices'])
    sources=['models/he3/update.py','models/he3/gaussian_fast.py','models/he3/measure.py',
        'models/he3/sampling.py','core/runtime.py','core/parallel_tempering.py',
        'exploration/he3_spatial_pilot/gpu_measure.py','exploration/he3_spatial_pilot/observables.py',
        'exploration/he3_spatial_pilot/pilot.py','exploration/he3_spatial_pilot/ladder.py']
    record=dict(schema_version=1,label=label,L=L,pressure=20,g_D=0,seed=seed,exchange_seed=seed+1,
        initialization='production homogeneous positive chirality, d=z' if label=='ordered' else 'iid zero-mean full complex Gaussian, E[n]=n0, independent host seed',
        model_parameters=model.parameters(),temperatures=T,ladder_design=ladder,
        thermalization_sweeps_requested=burn,production_sweeps_requested=production,
        actual_sweeps=pt.sweeps,actual_production_sweeps=max(0,pt.sweeps-burn),stop_reason=stop,
        global_interval=50,spatial_interval=250,history_interval=100,
        observation_event='after complete local sweep and scheduled PT exchange',
        setup_seconds=setup,loop_seconds=loop_seconds,total_seconds=perf_counter()-start,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},
        python=platform.python_version(),numpy=np.__version__,numba=numba.__version__,gpu=str(cuda.get_current_device().name),
        meter_allocated_bytes=meter.allocated_bytes,snapshots=snapshots,progress=progress,
        pt_at_burn=burn_diag,pt_final=diag,sampler=rt.sampling_diagnostics())
    (folder/'run.json').write_text(json.dumps(_json_ready(record),indent=2),encoding='utf-8')
    print(f'{label}: saved {len(ss)} Fourier events; loop {loop_seconds:.2f}s, stop {stop}',flush=True)


def main():
    validation=json.loads((OUT/'validation.json').read_text())
    if not validation['passed']:raise RuntimeError('validation has not passed')
    for n,h in validation['source_sha256'].items():
        if hashlib.sha256((HERE/n).read_bytes()).hexdigest()!=h:raise RuntimeError(f'{n} changed since validation')
    bench=json.loads((OUT/'benchmark.json').read_text());pred=next(r for r in bench['rows'] if r['L']==64)['predicted_200k_loop_seconds']
    if pred>90:raise RuntimeError(f'Preflight predicts {pred:.1f}s/history; prepare and review runtime before proceeding')
    T,ladder=make_ladder();(OUT/'ladder.json').write_text(json.dumps(ladder,indent=2),encoding='utf-8')
    labels=['ordered','hot'] if len(sys.argv)==1 else sys.argv[1:]
    if any(x not in ['ordered','hot'] for x in labels):raise ValueError('Only ordered/hot pilot labels are supported')
    for label in labels:run(label,2026102951 if label=='ordered' else 2026102953,T,ladder)


if __name__=='__main__':main()
