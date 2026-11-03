"""Reproduce preparation plots and storage estimates; reads bounded benchmark results."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from large_run import allocations


def read(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    for extension in ('png','pdf'):
        fig.savefig(HERE/'figures'/f'{name}.{extension}',dpi=170)
    plt.close(fig)


def main():
    (HERE/'figures').mkdir(exist_ok=True)
    inventory = read(ROOT/'exploration/he3_pt_equilibration/results/inventory.csv')
    old = {int(r['L']):float(r['gpu_seconds'])/6e6 for r in inventory if r['campaign']=='new' and float(r['g_d'])==0 and int(r['L']) in (160,256)}
    exponent = np.log(old[256]/old[160])/np.log(256/160)
    resources = []
    for L,(burn,production) in allocations.items():
        b = json.loads((HERE/f'results/benchmark_L{L}.json').read_text())
        scalar = production/50*b['scalar_bytes_per_event']
        spatial = production/1000*b['spatial_bytes_per_event']
        profiles = production/20000*b['profile_bytes_per_event']
        history = ((burn+production)//50+1)*160*5  # int32 temperature + int8 exact endpoint label
        traffic = ((burn+production)//50000+3)*(5*160+3*159)*8
        coordinates = ((burn+production)//50+1)*16+production/50*8
        data = scalar+spatial+profiles+history+traffic+coordinates
        state = 160*L*L*88
        base = old[L] if L in old else old[256]*(L/256)**2
        high = base if L in old else old[256]*(L/256)**exponent
        row = dict(L=L,burn=burn,production=production,
                   local_ms_per_sweep=1000*b['sweep_seconds'],
                   measurement_overhead_percent=100*b['estimated_cadence_overhead'],
                   device_peak_known_GiB=(b['allocated_device_bytes']+history)/2**30,
                   raw_output_GiB=data/2**30, full_checkpoint_GiB=(state+data)/2**30,
                   host_budget_GiB=(2*data+2*state)/2**30+2,
                   historical_hours_low=base*.8*(burn+production)/3600,
                   historical_hours_high=high*.8*(burn+production)/3600,
                   local_hours_projection=b['sweep_seconds']*(burn+production)*(1+b['estimated_cadence_overhead']*production/(burn+production))/3600)
        resources.append(row)
    with (HERE/'results/resources.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(resources[0])); w.writeheader(); w.writerows(resources)
    fig,ax=plt.subplots(1,3,figsize=(12,3.5))
    L=np.array([r['L'] for r in resources])
    ax[0].plot(L,[r['local_ms_per_sweep'] for r in resources],'o-')
    ax[0].set(ylabel='ms per PT sweep',title='Measured: RTX 4070 SUPER\n100 timed sweeps, R=160')
    for key,label in [('device_peak_known_GiB','Device arrays + full history'),('raw_output_GiB','Raw final archive'),('host_budget_GiB','Conservative host budget')]:
        ax[1].plot(L,[r[key] for r in resources],'o-',label=label)
    ax[1].set(ylabel='GiB',title='Full-allocation projections');ax[1].legend(fontsize=7)
    ax[2].plot(L,[r['historical_hours_low'] for r in resources],'o-',label='Historical area scaling')
    ax[2].fill_between(L,[r['historical_hours_low'] for r in resources],[r['historical_hours_high'] for r in resources],alpha=.25)
    ax[2].set(ylabel='GPU hours per case',title='Conditional historical timing\nNot a new RTX 5090 benchmark')
    for a in ax: a.set_xlabel('L');a.grid(alpha=.2)
    save(fig,'resources')

    ladder=np.loadtxt(ROOT/'exploration/he3_pt_ladder_validation/ladder_C.txt')
    fig,ax=plt.subplots(1,2,figsize=(10,3.4))
    ax[0].plot((ladder[:-1]+ladder[1:])/2,np.diff(ladder)*1e6,'.-')
    ax[0].axvspan(.380,.388,color='grey',alpha=.15)
    ax[0].set(xlabel='T midpoint',ylabel='Spacing (millionths)',title='Exact frozen C160: .375 to .405')
    transport=read(ROOT/'exploration/he3_pt_equilibration/results/transport_summary.csv')
    for gd in (0.,.01):
        r=sorted([r for r in transport if r['campaign']=='new' and float(r['g_d'])==gd and int(r['L'])>=96],key=lambda r:int(r['L']))
        ax[1].plot([int(v['L']) for v in r],[float(v['passage_flux_time'])/1e6 for v in r],'o-',label=f'gd={gd:g}, old R200')
    ax[1].set(xlabel='L',ylabel='Sweeps per walker round-trip equivalent (M)',title='Observed traffic scale, old runs\nC160 gains at L256 remain unmeasured');ax[1].legend(fontsize=8)
    save(fig,'ladder_and_prior_transport')

    curves=read(ROOT/'exploration/he3_pt_equilibration/results/scalar_curves.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for j,gd in enumerate((0.,.01)):
        for size in (160,256):
            rows=[r for r in curves if r['campaign']=='new' and int(r['L'])==size and float(r['g_d'])==gd]
            t=np.array([float(r['T']) for r in rows]);mask=(t>.380)&(t<(.385 if gd==0 else .391))
            for ax,key in [(axes[0,j],'Binder'),(axes[1,j],'Y')]:
                ax.errorbar(t[mask],np.array([float(r[key]) for r in rows])[mask],
                            yerr=np.array([float(r[key+'_se']) for r in rows])[mask],fmt='.-',label=f'L={size}',capsize=1)
        t=np.linspace(.380,.385 if gd==0 else .391,100)
        axes[1,j].plot(t,8*t/np.pi,'--',label='q=1/2 jump')
        axes[1,j].plot(t,2*t/np.pi,':',label='q=1 jump')
        axes[0,j].set(title=f'Prior gd={gd:g}: chirality',ylabel='Binder U')
        axes[1,j].set(title='Prior diagonal-average helicity',ylabel='Y',xlabel='T')
        for ax in axes[:,j]: ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'prior_transition_diagnostics')
    physics_validation()
    print(json.dumps(resources,indent=2))


def physics_validation():
    """Finite differences of independent full contractions; no local updates."""
    from numba import cuda
    from models.he3.model import He3Model
    from models.he3.spatial import measurement_profiles
    from tests.test_he3 import total_energy
    rng=np.random.default_rng(91)
    field=(.15*(rng.normal(size=(2,8,8,3,2))+1j*rng.normal(size=(2,8,8,3,2)))).astype(np.complex64)
    field[1]+=.3+.2j
    rows=[]
    for gd in (0.,.01):
        model=He3Model(pressure=20,g_d=gd,K=.83)
        gpu=model.measure(cuda.to_device(field))['curvature_xy']
        def energy(x,y): return total_energy(field,model.K,model.betas,(x,y),gd)
        for h in (.04,.02,.01,.005,.0025):
            levels=[]
            for step in (h,h/2):
                levels.append((energy(step,step)-energy(step,-step)-energy(-step,step)+energy(-step,-step))/(4*step**2))
            difference=(4*levels[1]-levels[0])/3
            for i in range(2):
                rows.append(dict(g_d=gd,field=i,h=h,GPU=float(gpu[i]),finite_difference=float(difference[i]),absolute_error=float(abs(gpu[i]-difference[i]))))
    with (HERE/'results/mixed_twist.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    for gd in (0.,.01):
        selected=[r for r in rows if r['g_d']==gd and r['field']==1]
        ax[0].loglog([r['h'] for r in selected],[r['absolute_error'] for r in selected],'o-',label=f'gd={gd:g}')
    ax[0].set(xlabel='Finite twist step h',ylabel='|GPU Kxy - independent finite difference|',title='Full GL + SOC, forward stencil')
    from matplotlib.ticker import NullFormatter
    ax[0].set_xticks([.0025,.01,.04],labels=['0.0025','0.01','0.04'])
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    ax[0].legend()
    L=64;x=np.arange(L)[:,None];a=np.zeros((L,L,3,2),complex)
    a[...,2,0]=np.exp(2j*np.pi*x/L)/np.sqrt(2);a[...,2,1]=a[...,2,0]
    profiles,_=measurement_profiles(a,charged=True)
    r=np.arange(L//2+1)
    for index,charge,name in [(7,1,'A_plus_z'),(9,2,'B_plus_plus')]:
        ax[1].plot(r/L,profiles[index,0].real,'o',ms=3,label=f'{name} measured')
        ax[1].plot(r/L,np.cos(charge*2*np.pi*r/L),'-',label=f'analytic charge {charge}')
    ax[1].set(xlabel='r/L',ylabel='Re G(r)',title='Synthetic phase wave; not thermal data');ax[1].legend(fontsize=8)
    save(fig,'physics_validation')


if __name__=='__main__':
    main()
