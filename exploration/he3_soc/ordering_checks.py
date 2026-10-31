"""Bootstrap finite-size crossings and grid-resolved response maxima.

Run: python exploration/he3_soc/ordering_checks.py
Crossing selection follows the first ordered-side crossing in a stated narrow
window, retaining failure counts. No thermodynamic extrapolation is attempted.
"""
import numpy as np
from compare import load, roots
from summarize import OUT, COUPLINGS, SIZES, evaluate, write_csv


def main():
    rng=np.random.default_rng(73192)
    rows=[];peaks=[]
    for gd in COUPLINGS[:3]:
        t,bb=load(gd,64)
        blocks={L:{k:v.reshape(30,20,-1).mean(axis=1) for k,v in load(gd,L)[1].items()} for L in SIZES}
        central={L:evaluate({k:v.mean(axis=0) for k,v in b.items()},t,L) for L,b in blocks.items()}
        center=.3855 if gd==.01 else .382
        mask=(t>center-.0015)&(t<center+.0015)
        specs=[]
        for pair in [(64,96),(96,128),(128,160),(64,160)]:
            for name in ['binder_chirality','binder_phase']:
                rr=roots(t[mask],(central[pair[0]][name]-central[pair[1]][name])[mask])
                if rr:specs.append((pair,name,rr[0]))
        boot={s:[] for s in specs}
        peakT={name:[] for name in ['specific_heat','chi_response','phase_response']}
        for rep in range(400):
            values={}
            for L,b in blocks.items():
                ids=rng.integers(0,30,30)
                values[L]=evaluate({k:v[ids].mean(axis=0) for k,v in b.items()},t,L)
            for s in specs:
                pair,name,ref=s
                rr=roots(t[mask],(values[pair[0]][name]-values[pair[1]][name])[mask])
                rr=[x for x in rr if abs(x-ref)<.0008]
                if rr:boot[s].append(min(rr,key=lambda x:abs(x-ref)))
            pmask=(t>center-.003)&(t<center+.003)
            for name in peakT:
                peakT[name].append(t[pmask][np.argmax(values[160][name][pmask])])
        for (pair,name,ref),xs in boot.items():
            rows.append(dict(g_d=gd,sizes=':'.join(map(str,pair)),observable=name,temperature=ref,
                window_low=center-.0015,window_high=center+.0015,root_selection='first ordered-side; bootstrap nearest within 0.0008',
                resamples=400,valid=len(xs),temperature_sd=np.std(xs,ddof=1),p025=np.quantile(xs,.025),p975=np.quantile(xs,.975)))
        for name,xs in peakT.items():
            j=np.argmax(central[160][name][pmask]);tc=t[pmask][j]
            peaks.append(dict(g_d=gd,L=160,observable=name,grid_peak=tc,
                bootstrap_p025=np.quantile(xs,.025),bootstrap_p975=np.quantile(xs,.975),
                fraction_same_grid_point=np.mean(np.asarray(xs)==tc),resamples=400))
    write_csv(OUT/'binder_crossing_bootstrap.csv',rows)
    write_csv(OUT/'peak_bootstrap.csv',peaks)
    print('Saved crossing and peak checks')


if __name__=='__main__':main()
