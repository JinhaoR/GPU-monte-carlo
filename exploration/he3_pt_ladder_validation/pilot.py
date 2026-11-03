"""Bounded L96 pilots using the unchanged Runtime, PT and Simulation classes."""
import os,sys,json,argparse,gc
from pathlib import Path
from time import perf_counter
sys.dont_write_bytecode=True
os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS']='0'
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path.insert(0,str(ROOT))
from numba import cuda
from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from models.he3.spatial import add_spatial_samples
from core.parallel_tempering import ParallelTempering
from core.simulation import Simulation
from utils.data_io import save_runs,_json_ready

def construct(ladder,gd,initialization,L=96):
    T=np.loadtxt(HERE/f'ladder_{ladder}.txt')
    seed=20261102 if initialization=='ordered' else 20261105
    model=He3Model(pressure=20,g_d=gd)
    model.spatial_every=20  # existing measurements, every 1000 sweeps
    rt=He3GaussianRuntime(model,L,T,seed=seed,rho=-.9,proposal_mass=0)
    if initialization=='random':
        rng=np.random.default_rng(20261108)
        n0=1/float(model.betas[1]+model.betas[3]+model.betas[4])
        a=(np.sqrt(n0/12)*(rng.normal(size=rt.state.shape)+1j*rng.normal(size=rt.state.shape))).astype(np.complex64)
        rt.invalidate_energy_cache();rt.state.copy_to_device(a)
    pt=ParallelTempering(rt,exchange_interval=5,seed=seed+1,tracked_walkers=range(rt.R),history_stride=1)
    return model,rt,pt

def benchmark():
    rows=[]
    for name in ('A','C'):
        start=perf_counter();model,rt,pt=construct(name,.01,'ordered')
        # Compile every enabled path, then measure complete simulation batches.
        Simulation(pt,0,1000,50).run();cuda.synchronize()
        setup=perf_counter()-start;times=[]
        for _ in range(3):
            start=perf_counter();Simulation(pt,0,2000,50).run();cuda.synchronize();times.append((perf_counter()-start)/2000)
        mem=cuda.current_context().get_memory_info()
        row=dict(ladder=name,L=96,R=rt.R,setup_seconds=setup,seconds_per_sweep=times,median_seconds_per_sweep=float(np.median(times)),
                 core_GPU_bytes=88*rt.R*96**2,device_total=mem.total,device_free=mem.free)
        rows.append(row);print(row,flush=True)
        del pt,rt,model;gc.collect();cuda.current_context().deallocations.clear()
    record=dict(gpu=str(cuda.get_current_device().name),rows=rows,scalar_interval=50,spatial_interval=1000,history_interval=5,exchange_interval=5)
    (OUT/'benchmark.json').write_text(json.dumps(record,indent=2))

def main(total_sweeps):
    bench=json.loads((OUT/'benchmark.json').read_text());rate={r['ladder']:r['median_seconds_per_sweep'] for r in bench['rows']}
    predicted=4*(2*rate['A']+rate['C'])*total_sweeps
    if total_sweeps<30000 or total_sweeps%3000 or predicted>1400:
        raise ValueError(f'Use a multiple of 3000 with predicted combined loop time <=1400s; prediction={predicted:.1f}s')
    # One readable case loop. No scheduler, run wrapper or alternative engine.
    start_all=perf_counter();burn=total_sweeps//3;production=total_sweeps-burn
    model,rt,pt=construct('A',.01,'ordered')
    Simulation(pt,0,1000,50).run();cuda.synchronize()
    warm_seconds=perf_counter()-start_all
    del pt,rt,model;gc.collect();cuda.current_context().deallocations.clear()
    for initialization in ('ordered','random'):
        for gd in (0,.01):
            for ladder in ('A','B','C'):
                path=OUT/f'L96_gd={gd:g}_{initialization}_{ladder}.npz'
                if path.exists():raise FileExistsError(path)
                if perf_counter()-start_all+total_sweeps*rate['A']>1750:
                    print('Bounded local time budget reached; remaining cases not launched.',flush=True);return
                setup=perf_counter();model,rt,pt=construct(ladder,gd,initialization)
                cuda.synchronize();setup=perf_counter()-setup
                # Retain early transients for diagnosis. All sweeps are recorded;
                # analysis excludes the explicitly recorded diagnostic burn-in.
                simulation=Simulation(pt,thermalization_sweeps=0,measurement_sweeps=total_sweeps,sample_interval=50)
                print(f'START {path.stem}: diagnostic burn={burn}, production={production}',flush=True)
                start=perf_counter();run=simulation.run();cuda.synchronize();elapsed=perf_counter()-start
                add_spatial_samples(run,model.spatial_samples,model.realspace_samples)
                metadata=run['metadata'];metadata.update(pilot=True,ladder=ladder,initialization=initialization,
                    diagnostic_burn_sweeps=burn,diagnostic_production_sweeps=production,
                    host_initialization_seed=20261108 if initialization=='random' else None,
                    timing=dict(simulation_seconds=elapsed,setup_seconds=setup),gpu=str(cuda.get_current_device().name),
                    sampling=rt.sampling_diagnostics(),spatial_interval=1000,realspace_interval=0,
                    note='Early transients deliberately saved; production mask is sample_sweeps > diagnostic_burn_sweeps')
                if not all(np.all(np.isfinite(x)) for x in run['samples'].values()):raise FloatingPointError('nonfinite saved measurement')
                save_runs(path,{96:run})
                print(f'SAVED {path.name}: {elapsed:.2f}s, total wall {perf_counter()-start_all:.1f}s',flush=True)
                del run,simulation,pt,rt,model;gc.collect();cuda.current_context().deallocations.clear()
    (OUT/'pilot_completion.json').write_text(json.dumps(dict(total_wall_seconds=perf_counter()-start_all,warmup_seconds=warm_seconds,total_sweeps_per_run=total_sweeps,cases=12),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true');parser.add_argument('--sweeps',type=int)
    args=parser.parse_args()
    if args.benchmark:benchmark()
    elif args.sweeps:main(args.sweeps)
    else:parser.error('Select --benchmark or --sweeps; no simulation runs by default')
