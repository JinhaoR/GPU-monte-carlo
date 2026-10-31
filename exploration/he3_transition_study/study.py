"""Local WHAM, withheld-temperature tests and joint PT block jackknives.

Run after prepare.py. All calculations use existing data, not Monte Carlo.
30 deletes of 2000 measurements (100,000 sweeps), simultaneous at all PT
temperatures, preserve observable and inter-temperature history correlations.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import csv
import json
import numpy as np
from reweight import load,solve,evaluate,observables,direct
from prepare import OUT,COUPLINGS,SIZES,NAMES

KEYS=('binder_chirality','binder_chirality_raw','binder_phase','helicity',
      'specific_heat','chi_response','chi_raw_response','phase_response',
      'chi2','chirality_raw2','phase2','energy','chirality','spin_alignment')
VALID=('energy','binder_chirality','binder_phase','helicity','specific_heat','spin_alignment')


def csvwrite(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def error(x):
    x=np.asarray(x);return np.sqrt((len(x)-1)*np.mean((x-x.mean(axis=0))**2,axis=0))


def calc(d,L,T,H=None,S=None,guess=None):
    f=solve(d['H'].sum(0) if H is None else H,d['S'].sum(1) if S is None else S,
            d['beta'],d['ebins'],guess)
    v=evaluate(f,T,L,float(d['eref']),True)
    return f,v


def main():
    validation=[];sensitivity=[];overlap=[];central=[]
    for gd in COUPLINGS:
        for L in SIZES:
            print('Reweight',gd,L,flush=True)
            d=load(gd,L);T=np.linspace(d['T'][1],d['T'][-2],601)
            H=d['H'].sum(0);S=d['S'].sum(1);nb=len(d['H'])
            f,v=calc(d,L,T);jk=[]
            for b in range(nb):
                _,a=calc(d,L,T,H-d['H'][b],S-d['S'][:,b],f['f'])
                jk.append([a[k] for k in KEYS])
            jk=np.asarray(jk);point=np.array([v[k] for k in KEYS])
            # Bin width, local source window, and halves are analysis choices,
            # not independent estimates to average together.
            variants=[]
            for bins in (400,1600):
                db=load(gd,L,bins);_,a=calc(db,L,T);variants.append((f'bins{bins}',a))
            for width in (7,11):
                ix=np.arange(7-width//2,8+width//2);h=H.copy();h[np.setdiff1d(np.arange(15),ix)]=0
                # Source-specific full sums allow arbitrary full-history windows.
                _,a=calc(d,L,T,h,d['Ssrc'][:,ix].sum(1));variants.append((f'sources{width}',a))
            for label,sl in [('first_half',slice(0,15)),('second_half',slice(15,30))]:
                _,a=calc(d,L,T,d['H'][sl].sum(0),d['S'][:,sl].sum(1));variants.append((label,a))
            # Independent coarse block grouping; 15 or 10 remaining deletes.
            coarse={}
            for group in (2,3):
                j=[]
                for start in range(0,nb,group):
                    _,a=calc(d,L,T,H-d['H'][start:start+group].sum(0),
                             S-d['S'][:,start:start+group].sum(1),f['f'])
                    j.append([a[k] for k in KEYS])
                coarse[f'jk{group*2000}']=np.array(j)
            np.savez_compressed(OUT/f'curves_gd={gd:g}_L={L}.npz',T=T,keys=np.array(KEYS),
                point=point,jk=jk,variant_names=np.array([label for label,_ in variants]),
                variants=np.array([[a[k] for k in KEYS] for _,a in variants]),**coarse)
            anchor=float(d['T'][7]);i=np.argmin(abs(T-anchor))
            for label,a in [('full',v)]+variants:
                for k in VALID:
                    sensitivity.append(dict(g_d=gd,L=L,variant=label,temperature=T[i],
                        observable=k,value=a[k][i],change_from_full=a[k][i]-v[k][i]))
            for k in VALID:
                central.append(dict(g_d=gd,L=L,temperature=T[i],observable=k,value=v[k][i],
                    se2000=error(jk[:,KEYS.index(k),i]),se4000=error(coarse['jk4000'][:,KEYS.index(k),i]),
                    se6000=error(coarse['jk6000'][:,KEYS.index(k),i])))
            hn=H/H.sum(1)[:,None]
            for k in range(14):
                overlap.append(dict(g_d=gd,L=L,Tlow=d['T'][k],Thigh=d['T'][k+1],
                    histogram_overlap=np.minimum(hn[k],hn[k+1]).sum()))
            for hh,k in enumerate(d['hold']):
                target=np.array([1/d['beta'][k]])
                h=H.copy();h[k]=0;s=S-d['Shold'][hh].sum(1)
                fh,pred=calc(d,L,target,h,s)
                actual=direct(d,k,L);pd=[];ac=[]
                for b in range(nb):
                    hb=d['H'][b].copy();hb[k]=0
                    sb=d['S'][:,b]-d['Shold'][hh,:,b]
                    _,vb=calc(d,L,target,h-hb,s-sb,fh['f'])
                    src=(d['Shold'][hh].sum(1)-d['Shold'][hh,:,b]).sum(1)/(H[k].sum()-d['H'][b,k].sum())
                    va=observables({name:np.array([src[j]]) for j,name in enumerate(NAMES)},target,L,float(d['eref']))
                    pd.append([vb[name][0] for name in VALID]);ac.append([va[name][0] for name in VALID])
                pd=np.array(pd);ac=np.array(ac);differr=error(pd-ac)
                # Independent raw-weight ESS is deliberately not called time ESS.
                for j,name in enumerate(VALID):
                    diff=pred[name][0]-actual[name]
                    validation.append(dict(g_d=gd,L=L,withheld_T=target[0],observable=name,
                        actual=actual[name],prediction=pred[name][0],difference=diff,
                        actual_se=error(ac)[j],prediction_se=error(pd)[j],difference_se=differr[j],
                        paired_z=diff/differr[j],raw_weight_ess=pred['weight_ess'][0],
                        max_source_fraction=pred['max_source_fraction'][0]))
    csvwrite('withheld_validation.csv',validation);csvwrite('moment_sensitivity.csv',sensitivity)
    csvwrite('energy_overlap.csv',overlap);csvwrite('block_errors.csv',central)
    summary=dict(withheld_tests=len(validation),max_abs_paired_z=max(abs(r['paired_z']) for r in validation),
        number_abs_z_above_3=int(sum(abs(r['paired_z'])>3 for r in validation)))
    (OUT/'reweight_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(summary,flush=True)


if __name__=='__main__':main()
