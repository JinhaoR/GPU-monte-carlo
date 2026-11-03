"""Small local experiments using the unchanged Gaussian and PT implementations.

Usage: python -B .../pilot.py label L gd preparation seed sweeps hot_endpoint
Use --benchmark first. No production defaults, data formats or modules change.
"""
import os
os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS']='0'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from time import perf_counter
import json,sys
import numpy as np
from numba import cuda

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from models.he3.measure import MEASUREMENT_NAMES
from models.he3.spatial import numpy_amplitudes
from core.parallel_tempering import ParallelTempering
from exploration.he3_hqv_stage2.energy import energy
from utils.data_io import _json_ready


def initial(model,L,R,kind,seed):
    b=np.asarray(model.betas,dtype=float);g=float(model.g_d);c=float(np.float32(2/3))
    n=(2+g*c)/(2*(b[1]+b[3]+b[4]))
    a=np.zeros((R,L,L,3,2),np.complex64)
    if kind=='hot':
        rng=np.random.default_rng(seed+100000)
        a[:]=(rng.normal(size=a.shape)+1j*rng.normal(size=a.shape))*np.sqrt(n/12)
    elif kind=='chiral':a[...,2,0]=np.sqrt(n/2);a[...,2,1]=1j*np.sqrt(n/2)
    elif kind=='planar':
        n=(2+g*(1+c))/(2*(b[0]+b[1]+(b[2]+b[3]+b[4])/2))
        a[...,0,1]=np.sqrt(n/2);a[...,1,0]=-np.sqrt(n/2)
    elif kind=='plus':a[...,2,0]=np.sqrt(n/2);a[...,2,1]=np.sqrt(n/2)
    else:raise ValueError(kind)
    return a


def extra(a):
    """Kxy and local antisymmetric-planar weight, in physical-walker order."""
    a=a.astype(complex);x=np.roll(a,-1,axis=1);y=np.roll(a,-1,axis=2)
    kxy=2*np.real(x[...,0].conj()*y[...,1]+x[...,1].conj()*y[...,0]).sum((1,2,3))
    planar=(abs(a[...,0,1]-a[...,1,0])**2/2).sum((1,2))
    return np.stack((kxy,planar),axis=-1)


def run(label,L,gd,kind,seed,total,hot,benchmark=False):
    if L not in (8,16,24,32) or total>100000:raise ValueError('Pilot bounds')
    out=HERE/'pilots';out.mkdir(exist_ok=True)
    path=out/(label+'.npz');meta_path=out/(label+'.json')
    if path.exists() or meta_path.exists():raise FileExistsError(label)
    T=np.linspace(.28,hot,24);burn=total//4
    # State+RNG+counts use 88 bytes/site; include generous buffers/records margin.
    estimated_bytes=24*L*L*88+total//20*24*11*8+total//100*24*(16*5*16+68*8)+8_000_000
    if not benchmark:
        bench=json.loads((HERE/'benchmark.json').read_text())
        seconds=next(r['seconds_per_sweep'] for r in bench if r['L']==L)*total
        print(f'PREFLIGHT {label}: predicted loop {seconds:.1f}s; conservative data/workspace {estimated_bytes/2**20:.1f} MiB',flush=True)
        if seconds>110:raise RuntimeError('Pilot exceeds reviewed local runtime bound')
    model=He3Model(pressure=20,g_d=gd);model.spatial_every=5
    rt=He3GaussianRuntime(model,L,T,seed=seed,rho=-.9,proposal_mass=0)
    a=initial(model,L,len(T),kind,seed);rt.state.copy_to_device(a);rt.invalidate_energy_cache()
    pt=ParallelTempering(rt,exchange_interval=5,seed=seed+1,tracked_walkers=range(len(T)),history_stride=4)
    first=pt.measure();model._measurement_count=0;model.spatial_samples=[]
    expected=np.array([energy(x,betas=model.betas,gd=float(model.g_d))['total'] for x in a])
    initial_error=float(np.max(abs(first['energy']-expected))/L**2)
    assert initial_error<2e-5
    # Warm read-only paths before timing; no RNG changes and no hidden updates.
    from models.he3.measure_spatial import SpatialMeter
    model._spatial_meter=SpatialMeter(L,len(T))
    amp=model._spatial_meter.measure(rt.state,pt.walker_to_temperature)
    model._spatial_meter.structure(rt.state,model,pt.walker_to_temperature)
    assert np.max(abs(amp-numpy_amplitudes(a)))<1e-10
    # Compile the update with one explicitly recorded sweep before loop timing.
    pt.sweep();cuda.synchronize();start=perf_counter()
    values=[];maps=[];extras=[];sweeps=[];sparse=[];snapshots=[]
    for step in range(2,total+1):
        pt.sweep()
        if step%20==0:
            v=pt.measure();values.append(np.stack([v[k] for k in MEASUREMENT_NAMES],axis=-1))
            mapping=pt.temperature_to_walker.copy_to_host();maps.append(mapping);sweeps.append(step)
            if step%100==0:
                state=rt.snapshot();extras.append(extra(state)[mapping]);sparse.append(values[-1])
                if step in (total//2,total):snapshots.append(state[mapping])
        if step%1000==0 and perf_counter()-start>120:raise RuntimeError('Pilot runtime cap reached')
    cuda.synchronize();elapsed=perf_counter()-start
    diag=pt.diagnostics();hist=diag.pop('walker_history')
    records=model.spatial_samples
    assert len(records)==total//100
    amplitudes=np.array([r[1] for r in records]);structure=np.array([r[2] for r in records])
    wm=np.array(maps);gs=np.array(values);sv=np.array(sparse)
    np.testing.assert_array_equal(wm[4::5],np.array([r[3] for r in records]))
    assert np.max(abs(amplitudes[:,:,0,0].real-sv[:,:,2]))/L**2<5e-7
    assert np.max(abs(amplitudes[:,:,1,0]-sv[:,:,3]-1j*sv[:,:,4]))/L**2<5e-7
    if benchmark:return dict(L=L,seconds_per_sweep=elapsed/(total-1),loop_seconds=elapsed,estimated_bytes=estimated_bytes,initial_energy_max_error_per_site=initial_error)
    np.savez_compressed(path,T=T,sweeps=sweeps,values=gs,names=np.array(MEASUREMENT_NAMES),
        temperature_to_walker=wm,amplitudes=amplitudes,structure=structure,sparse_values=sv,
        extra=extras,extra_names=['Kxy','antisymmetric_planar_weight'],snapshot_fields=snapshots,
        walker_history_sweeps=hist['sweeps'],walker_history_indices=hist['temperature_indices'])
    record=dict(label=label,L=L,g_d=gd,preparation=kind,seed=seed,exchange_seed=seed+1,
        total_sweeps=total,analysis_burn=burn,temperatures=T,scalar_interval=20,spatial_interval=100,
        observation='after full local sweep and scheduled exchange',model=model.parameters(),
        initialization_energy_per_site=first['energy']/L**2,initialization_density=first['density']/L**2,
        initial_energy_max_error_per_site=initial_error,loop_seconds=elapsed,estimated_bytes=estimated_bytes,
        pt=diag,sampling=rt.sampling_diagnostics(),snapshot_sweeps=[total//2,total])
    meta_path.write_text(json.dumps(_json_ready(record),indent=2),encoding='utf-8')
    print(f'SAVED {label}: {elapsed:.2f}s, loops={sum(diag["walker_round_trips"])}',flush=True)
    return record


if __name__=='__main__':
    if sys.argv[1:] == ['--benchmark']:
        rows=[run('benchmark',L,.1,'chiral',9021+L,400,.65,True) for L in (8,16)]
        (HERE/'benchmark.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(rows)
    else:
        label,L,gd,kind,seed,total,hot=sys.argv[1:]
        run(label,int(L),float(gd),kind,int(seed),int(total),float(hot))
