"""Bounded local GPU measurements; only L64/L128 get short update tests."""
import os
os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS']='0';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,sys,csv
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from numba import cuda
from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from models.he3.measure import he3_ordered_cached_measure_kernel,he3_energy_kernel
from core.parallel_tempering import ParallelTempering
from gpu_measure import SpatialMeter
from ladder import make_ladder,OUT


def gpu_us(fn,reps=80):
    for _ in range(3):fn()
    cuda.synchronize();a=cuda.event(timing=True);b=cuda.event(timing=True)
    a.record()
    for _ in range(reps):fn()
    b.record();b.synchronize();return a.elapsed_time(b)*1000/reps


def wall_us(fn,reps=40):
    fn();cuda.synchronize();t=perf_counter()
    for _ in range(reps):fn()
    cuda.synchronize();return (perf_counter()-t)*1e6/reps


def main():
    OUT.mkdir(parents=True,exist_ok=True);T,ladder=make_ladder();R=len(T);rows=[];cycles=[]
    model=He3Model(pressure=20,g_d=0)
    for L in [64,128,160,256]:
        rt=None;pt=None
        if L in (64,128):
            rt=He3GaussianRuntime(model,L,T,seed=2026102910+L)
            pt=ParallelTempering(rt,exchange_interval=5,seed=2026102911+L)
            for _ in range(100):pt.sweep()
            state=rt.state;mapping=pt.walker_to_temperature;E=pt.energies
        else:
            # Measurement scaling only. No L160/L256 Monte Carlo is run.
            state=cuda.to_device(model.initialize_state(L,R,0));mapping=cuda.to_device(np.arange(R,dtype=np.int32))
            E=cuda.device_array(R,np.float64);he3_energy_kernel[R,256](state,E,model.betas,model.K,model.g_d)
        meter=SpatialMeter(L,R);glob=cuda.device_array((10,R),np.float64)
        def global_launch():he3_ordered_cached_measure_kernel[R,256](state,glob,model.betas,model.K,mapping,E,model.g_d)
        def spatial_launch():meter.launch(state,mapping)
        spatial=[];globaltimes=[];host=[]
        for rep in range(3):
            globaltimes.append(gpu_us(global_launch));spatial.append(gpu_us(spatial_launch))
            host.append(wall_us(lambda:meter.measure(state,mapping)))
        row=dict(L=L,R=R,state_kind='100-sweep GPU state' if rt is not None else 'uniform measurement-only field',
            global_cached_gpu_us=float(np.median(globaltimes)),spatial_gpu_us=float(np.median(spatial)),
            spatial_gpu_us_min=min(spatial),spatial_gpu_us_max=max(spatial),
            spatial_host_read_us=float(np.median(host)),meter_allocated_bytes=meter.allocated_bytes,
            raw_transfer_bytes=R*8*3*2*8)
        if pt is not None:
            # Repeated alternating order controls gross timing/clock effects.
            n=1000 if L==64 else 300
            for rep in range(3):
                for with_spatial in ([False,True] if rep%2==0 else [True,False]):
                    cuda.synchronize();start=perf_counter()
                    for step in range(1,n+1):
                        pt.sweep()
                        if step%50==0:pt.measure()
                        if with_spatial and step%250==0:meter.measure(rt.state,pt.walker_to_temperature)
                    cuda.synchronize();cycles.append(dict(L=L,rep=rep,with_spatial=with_spatial,sweeps=n,seconds=perf_counter()-start))
            base=np.median([x['seconds']/x['sweeps'] for x in cycles if x['L']==L and not x['with_spatial']])
            added=np.median([x['seconds']/x['sweeps'] for x in cycles if x['L']==L and x['with_spatial']])
            row.update(baseline_us_per_sweep=base*1e6,with_spatial_us_per_sweep=added*1e6,
                observed_cycle_overhead_fraction=added/base-1,
                cadence_estimated_overhead_fraction=row['spatial_host_read_us']/250/(base*1e6))
            if L==64:row['predicted_200k_loop_seconds']=added*200000
        else:
            row.update(baseline_us_per_sweep=None,with_spatial_us_per_sweep=None,
                observed_cycle_overhead_fraction=None,cadence_estimated_overhead_fraction=None)
        rows.append(row);print(row,flush=True)
        del meter,state,glob,E,mapping,pt,rt
    result=dict(gpu=str(cuda.get_current_device().name),timing='CUDA event batches and synchronized host wall time; JIT warmed',
        ladder=ladder,rows=rows,cycle_repeats=cycles)
    (OUT/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'benchmark.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


if __name__=='__main__':main()
