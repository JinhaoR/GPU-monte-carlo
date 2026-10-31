"""Read-only, one-file/one-channel-at-a-time reduction of the SOC campaign.

Run: python exploration/he3_soc/summarize.py
The cache contains 100-measurement block means, never original configurations.
All errors are conditional on the sampled history; they cannot certify equilibrium.
"""
from pathlib import Path
import csv
import gc
import json
import sys
import zipfile

import numpy as np
from scipy.fft import rfft, irfft, next_fast_len

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
CACHE = OUT / "blocks"
BASE = 100
COUPLINGS = (0, .001, .01, .1)
SIZES = (64, 96, 128, 160)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def autocorrelation(x):
    """FFT biased ACF and initial-monotone-sequence tau, in saved samples.

    Pair rho(0)+rho(1), rho(2)+rho(3), ...; stop at first nonpositive pair.
    Retained pairs are made nonincreasing; tau=.5+sum(rho_lag), ESS=n/(2*tau).
    ESS is capped at n. Alternating PT is not reversible: this is a descriptive
    finite-history diagnostic, not a guaranteed variance estimator.
    """
    n, nt = x.shape
    tau = np.empty(nt)
    for start in range(0, nt, 8):
        a = x[:, start:start+8].copy()
        a -= a.mean(axis=0)
        f = rfft(a, n=next_fast_len(2*n), axis=0)
        cov = irfft(f.conj()*f, n=next_fast_len(2*n), axis=0)[:n]
        for k in range(a.shape[1]):
            if cov[0, k] <= 0:
                tau[start+k] = np.nan
                continue
            ac = cov[:, k] / cov[0, k]
            pairs = ac[:2*(n//2)].reshape(-1, 2).sum(axis=1)
            stop = np.flatnonzero(pairs <= 0)
            end = int(stop[0]) if len(stop) else len(pairs)
            tau[start+k] = max(.5, -.5 + np.minimum.accumulate(pairs[:end]).sum())
    return tau


def evaluate(m, T, L):
    hx = (m['kx']-(m['ix2']-m['ix']**2)/T)/L**2
    hy = (m['ky']-(m['iy2']-m['iy']**2)/T)/L**2
    return dict(energy=m['e'], density=m['n'], chirality=m['chi_abs'],
                signed_chirality=m['chi'], phase=np.sqrt(np.maximum(m['p2'], 0)),
                binder_chirality=1-m['chi4']/(3*m['chi2']**2),
                binder_phase=2-m['p4']/m['p2']**2,
                helicity_x=hx, helicity_y=hy, helicity=.5*(hx+hy),
                helicity_uncentered=.5*(m['kx']+m['ky']-(m['ix2']+m['iy2'])/T)/L**2,
                spin_alignment=m['qzz'], specific_heat=L**2*(m['e2']-m['e']**2)/T**2,
                chi_response=L**2*(m['chi2']-m['chi_abs']**2)/T,
                phase_response=L**2*(m['p2']-m['p_abs']**2)/T,
                chi2=m['chi2'], chi4=m['chi4'])


def estimate(blocks, T, L, block=100, portion=None):
    factor = block//BASE
    selected = {k:v[portion] if portion is not None else v for k,v in blocks.items()}
    count = len(next(iter(selected.values())))//factor
    b = {k:v[:count*factor].reshape(count, factor, -1).mean(axis=1) for k,v in selected.items()}
    means = {k:v.mean(axis=0) for k,v in b.items()}
    values = evaluate(means, T, L)
    deleted = evaluate({k:(count*means[k]-v)/(count-1) for k,v in b.items()}, T, L)
    errors = {k:np.sqrt((count-1)*np.var(v, axis=0)) for k,v in deleted.items()}
    return values, errors


def reduce_file(path, gd, L):
    with np.load(path, allow_pickle=False) as z:
        info = json.loads(z['manifest'].item())['runs'][str(L)]
        meta = info['metadata']
        T = np.asarray(meta['temperatures'])
        n = len(meta['sample_sweeps'])
        assert n % BASE == 0
        blocks = {}
        diagnostic = []
        def add(name, a):
            assert a.shape == (n, len(T)), (name, a.shape)
            if not np.all(np.isfinite(a)):
                raise ValueError(f'Nonfinite samples: {path}: {name}')
            blocks[name] = a.reshape(n//BASE, BASE, len(T)).mean(axis=1)
        def diagnose(name, a):
            tau = autocorrelation(a)
            for j,t in enumerate(T):
                se = []
                for b in (1, 100, 500, 2000, 5000):
                    v = a[:(n//b)*b, j].reshape(-1,b).mean(axis=1)
                    se.append(v.std(ddof=1)/np.sqrt(len(v)))
                first, last = a[:n//2,j], a[n//2:,j]
                f = first.reshape(-1, 2000).mean(axis=1)
                s = last.reshape(-1, 2000).mean(axis=1)
                half_se = np.sqrt(f.var(ddof=1)/len(f)+s.var(ddof=1)/len(s))
                diagnostic.append(dict(g_d=gd,L=L,temperature=t,observable=name,
                    tau_samples=tau[j],tau_sweeps=tau[j]*meta['sample_interval'],
                    effective_samples=n/(2*tau[j]),mean=a[:,j].mean(),
                    first_half=first.mean(),second_half=last.mean(),
                    half_difference=last.mean()-first.mean(),half_difference_se=half_se,
                    half_z=(last.mean()-first.mean())/half_se if half_se else np.nan,
                    se_iid=se[0],se_100=se[1],se_500=se[2],se_2000=se[3],se_5000=se[4]))
        def load(name):
            return z[f'L{L}_{name}']
        a = load('energy')/L**2
        add('e',a); add('e2',a*a); diagnose('energy',a)
        del a
        density = load('density')
        add('n',density/L**2)
        diagnose('density',density/L**2)
        a = load('chirality')/density
        assert np.max(np.abs(a)) <= 1+1e-5
        add('chi',a); add('chi_abs',np.abs(a)); add('chi2',a*a); add('chi4',a**4)
        diagnose('signed_chirality',a); diagnose('chirality',np.abs(a))
        del density,a
        a,b = load('phase_real')/L**2,load('phase_imag')/L**2
        p2 = a*a+b*b
        del a,b
        add('p2',p2); add('p4',p2*p2); add('p_abs',np.sqrt(p2))
        diagnose('phase_magnitude',np.sqrt(p2))
        del p2
        for direction in ('x','y'):
            a=load('current_'+direction)
            add('i'+direction,a); add('i'+direction+'2',a*a)
            diagnose('current_'+direction,a)
            del a
            a=load('curvature_'+direction)
            add('k'+direction,a)
            del a
        a=load('spin_alignment')
        assert a.min() >= -.50001 and a.max() <= 1.00001
        add('qzz',a); add('qzz2',a*a); diagnose('spin_alignment',a)
        del a
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE/f'gd={gd:g}_L={L}.npz',T=T,**blocks)
        therm = []
        for block in (100,500,2000,5000):
            v,e = estimate(blocks,T,L,block)
            for j,t in enumerate(T):
                row=dict(g_d=gd,L=L,temperature=t,block_size=block,subset='full')
                for key in v:
                    row[key]=v[key][j]; row[key+'_err']=e[key][j]
                therm.append(row)
        for subset,portion in [('first_half',slice(0,n//BASE//2)),('second_half',slice(n//BASE//2,None))]:
            v,e=estimate(blocks,T,L,2000,portion)
            for j,t in enumerate(T):
                row=dict(g_d=gd,L=L,temperature=t,block_size=2000,subset=subset)
                for key in v:
                    row[key]=v[key][j]; row[key+'_err']=e[key][j]
                therm.append(row)
        with zipfile.ZipFile(path) as archive:
            crc=';'.join(f'{f.filename}:{f.CRC:08x}' for f in archive.infolist())
        params=meta['model_parameters']
        inventory=dict(path=str(path.relative_to(ROOT)),g_d=gd,L=L,bytes=path.stat().st_size,
            measurements=n,n_temperatures=len(T),T_min=T.min(),T_max=T.max(),
            thermalization_sweeps=meta['thermalization_sweeps'],measurement_sweeps=meta['measurement_sweeps'],
            sample_interval=meta['sample_interval'],exchange_interval=meta['parallel_tempering']['exchange_interval'],
            seed=meta['seed'],exchange_seed=meta['parallel_tempering']['exchange_seed'],
            campaign=meta['campaign']['id'],sampler=meta['sampler'],K=params['K'],
            betas=json.dumps(params['betas']),bare_dipole_length=params.get('dipole_length_lattice'),
            nonfinite_rejections=sum(meta['sampling']['nonfinite_rejections']),zip_crc32=crc)
        return therm,diagnostic,inventory


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inventory=[]
    for gd in COUPLINGS:
        for L in SIZES:
            path=ROOT/f'data/he3/P=20_gd={gd:g}_gaussian/L={L}.npz'
            thermo_file=OUT/f'thermo_gd={gd:g}_L={L}.csv'
            diag_file=OUT/f'statistics_gd={gd:g}_L={L}.csv'
            inv_file=OUT/f'inventory_gd={gd:g}_L={L}.json'
            cache_file=CACHE/f'gd={gd:g}_L={L}.npz'
            if thermo_file.exists() and diag_file.exists() and inv_file.exists() and cache_file.exists() and '--force' not in sys.argv:
                previous=json.loads(inv_file.read_text())
                with zipfile.ZipFile(path) as archive:
                    crc=';'.join(f'{f.filename}:{f.CRC:08x}' for f in archive.infolist())
                if previous['bytes']==path.stat().st_size and previous['zip_crc32']==crc:
                    inventory.append(previous)
                    print('Cached',gd,L,flush=True)
                    continue
            print('Reducing',gd,L,flush=True)
            therm,diag,inv=reduce_file(path,gd,L)
            write_csv(thermo_file,therm);write_csv(diag_file,diag)
            inv_file.write_text(json.dumps(inv,indent=2),encoding='utf-8')
            inventory.append(inv)
            gc.collect()
    write_csv(OUT/'inventory.csv',inventory)
    print('Finished',len(inventory),'datasets',flush=True)


if __name__=='__main__':
    main()
