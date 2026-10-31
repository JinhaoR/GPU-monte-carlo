"""Charge hypotheses, size/window sensitivity, correlated-block bootstrap.

Run after summarize.py: python exploration/he3_soc/bkt_checks.py
Bootstrap blocks are sampled jointly across channels and temperatures for each L.
Distinct L runs are resampled separately. Errors describe existing histories only.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from exploration.he3_soc.summarize import OUT, COUPLINGS, SIZES, estimate, evaluate, write_csv
from exploration.he3_soc.compare import load
from utils.Analysis.bkt import fit_weber_minnhagen


def fit_arrays(T,L,rho,err,q,window=None):
    mask=np.ones(len(T),bool) if window is None else (T>=window[0])&(T<=window[1])
    tt=T[mask];rr=q*q*rho[:,mask];ss=q*q*err[:,mask]
    logs=2*np.log(np.asarray(L)/L[0])
    scores,starts=profile(tt,L,rho[:,mask],err[:,mask],q,tt)
    # The production helper's five uniformly spaced starts can miss a narrow
    # optimum on a dense ladder. Profile every saved temperature before refining.
    minima=np.flatnonzero(np.r_[True,scores[1:]<=scores[:-1]] & np.r_[scores[:-1]<=scores[1:],True])
    candidates=sorted(set(minima.tolist()+np.argsort(scores)[:3].tolist()),key=lambda j:scores[j])[:12]
    def residual(par):
        t,a=par
        y=np.array([np.interp(t,tt,r) for r in rr])
        s=np.array([np.interp(t,tt,r) for r in ss])
        return (y-2*t/np.pi*(1+a/(1+a*logs)))/s
    fits=[least_squares(residual,[np.clip(tt[j],tt[0]+1e-12,tt[-1]-1e-12),max(starts[j],1e-5)],
        bounds=([tt[0],0],[tt[-1],np.inf])) for j in candidates]
    f=min(fits,key=lambda x:sum(x.fun**2));t,a=f.x
    boundary=t<=tt[0]+1e-6*(tt[-1]-tt[0]) or t>=tt[-1]-1e-6*(tt[-1]-tt[0])
    reason='boundary optimum' if boundary else 'vanishing correction' if a<1e-8 else None
    return dict(temperature=float(t) if reason is None else None,C=float(1/a-2*np.log(L[0])) if a>1e-8 else None,
        chi2=float(sum(f.fun**2)),degrees_of_freedom=len(L)-2,reason=reason)


def profile(T,L,rho,err,q,grid):
    log=2*np.log(np.asarray(L)/L[0])
    scores=[];aa=[]
    for t in grid:
        y=np.array([np.interp(t,T,r) for r in rho])*q*q
        s=np.array([np.interp(t,T,r) for r in err])*q*q
        def objective(a):
            return np.sum(((y-2*t/np.pi*(1+a/(1+a*log)))/s)**2)
        f=minimize_scalar(objective,bounds=(0,50),method='bounded')
        opts=[(f.fun,f.x),(objective(0),0),(objective(50),50)]
        score,a=min(opts)
        scores.append(score);aa.append(a)
    return np.array(scores),np.array(aa)


def optimizer_audit():
    """Expose sensitivity of the original five-start helper, without changing it."""
    import csv
    with (OUT/'wm_sensitivity.csv').open() as f:
        refined=[r for r in csv.DictReader(f) if r['block_size']=='2000' and r['subset']=='full' and r['window']=='all' and r['sizes']=='64:96:128:160']
    rows=[]
    for gd in COUPLINGS:
        T=load(gd,64)[0]
        values=[estimate(load(gd,L)[1],T,L,2000) for L in SIZES]
        rho=np.array([v['helicity'] for v,e in values]);err=np.array([e['helicity'] for v,e in values])
        for q in [.5,1.]:
            original=fit_weber_minnhagen(T,np.array(SIZES),q*q*rho,q*q*err,profile=False)
            f=next(r for r in refined if float(r['g_d'])==gd and float(r['charge'])==q)
            rows.append(dict(g_d=gd,charge=q,block_size=2000,original_temperature=original['temperature'],
                original_C=original['C'],original_chi2=original['chi2'],original_reason=original['reason'],
                refined_temperature=f['temperature'],refined_C=f['C'],refined_chi2=f['chi2']))
    write_csv(OUT/'wm_optimizer_audit.csv',rows)


def main():
    rows=[];bootrows=[];profrows=[]
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    rng=np.random.default_rng(20261029)
    for col,gd in enumerate(COUPLINGS):
        blocks={L:load(gd,L)[1] for L in SIZES}
        T=load(gd,64)[0]
        central={}
        for q,color in [(.5,'#4477aa'),(1.,'#cc6677')]:
            for bs in [100,500,2000,5000]:
                v=[];e=[]
                for L in SIZES:
                    vv,ee=estimate(blocks[L],T,L,bs)
                    v.append(vv['helicity']);e.append(ee['helicity'])
                rho,err=np.array(v),np.array(e)
                for subset in [SIZES,(96,128,160),(64,128,160),(64,96,160),(64,96,128)]:
                    if bs!=2000 and subset!=SIZES:continue
                    ii=[SIZES.index(L) for L in subset]
                    windows=[None]
                    if bs==2000 and subset==SIZES:
                        windows += [(.375,.39),(.38,.395),(.39,.41)] if gd<.1 else [(.35,.39),(.42,.46),(.435,.45)]
                    for win in windows:
                        f=fit_arrays(T,subset,rho[ii],err[ii],q,win)
                        rows.append(dict(g_d=gd,charge=q,block_size=bs,subset='full',sizes=':'.join(map(str,subset)),
                            window='all' if win is None else ':'.join(map(str,win)),temperature=f['temperature'],C=f['C'],
                            chi2=f['chi2'],dof=f['degrees_of_freedom'],reduced_chi2=f['chi2']/f['degrees_of_freedom'],reason=f['reason']))
                        if bs==2000 and subset==SIZES and win is None:central[q]=(f,rho,err)
            for label,sl in [('first_half',slice(0,300)),('second_half',slice(300,None))]:
                vv=[];ee=[]
                for L in SIZES:
                    v,e=estimate(blocks[L],T,L,2000,sl);vv.append(v['helicity']);ee.append(e['helicity'])
                f=fit_arrays(T,SIZES,np.array(vv),np.array(ee),q)
                rows.append(dict(g_d=gd,charge=q,block_size=2000,subset=label,sizes=':'.join(map(str,SIZES)),window='all',
                    temperature=f['temperature'],C=f['C'],chi2=f['chi2'],dof=2,reduced_chi2=f['chi2']/2,reason=f['reason']))

            f,rho,err=central[q]
            grid=np.linspace(.372,.405,600) if gd<.1 else np.linspace(.34,.465,1000)
            score,a=profile(T,SIZES,rho,err,q,grid)
            axes[0,col].plot(grid,score,color=color,label=f'q={q:g}')
            for t,s,aa in zip(grid,score,a):
                profrows.append(dict(g_d=gd,charge=q,temperature=t,chi2=s,a=aa))
            # q=.5 curves for context; both universal jump lines displayed.
            if q==.5:
                for L,r,s in zip(SIZES,rho,err):
                    axes[1,col].plot(T,r,label=f'L={L}')
                    axes[1,col].fill_between(T,r-s,r+s,alpha=.12)
            axes[1,col].plot(T,2*T/(np.pi*q*q),'--',color=color,label=f'jump q={q:g}')

            if f['temperature'] is None:
                continue
            # Local fit bootstrap with central error weights; does not fold in
            # unknown equilibration bias or independently seeded run variation.
            tc=f['temperature'];a0=1/(2*np.log(64)+f['C'])
            mask=(T>=tc-.006)&(T<=tc+.006)
            tb=T[mask];logs=2*np.log(np.asarray(SIZES)/64)
            samples=[];cs=[];chi=[];bad=0
            for rep in range(240):
                rr=[]
                for L in SIZES:
                    bb={k:v.reshape(30,20,-1).mean(axis=1) for k,v in blocks[L].items() if k in ['ix','ix2','iy','iy2','kx','ky']}
                    indices=rng.integers(0,30,30)
                    m={k:v[indices].mean(axis=0) for k,v in bb.items()}
                    h=.5*(m['kx']+m['ky']-(m['ix2']-m['ix']**2+m['iy2']-m['iy']**2)/T)/L**2
                    rr.append(h[mask]*q*q)
                rr=np.array(rr);ss=err[:,mask]*q*q
                def residual(par):
                    t,a=par
                    measured=np.array([np.interp(t,tb,r) for r in rr])
                    se=np.array([np.interp(t,tb,s) for s in ss])
                    return (measured-2*t/np.pi*(1+a/(1+a*logs)))/se
                fits=[least_squares(residual,[np.clip(start,tb[0]+1e-10,tb[-1]-1e-10),max(a0,1e-5)],
                    bounds=([tb[0],0],[tb[-1],100]),max_nfev=150) for start in [tc,tc-.001,tc+.001]]
                best=min(fits,key=lambda ff:sum(ff.fun**2))
                t,a=best.x
                if t<tb[0]+1e-6 or t>tb[-1]-1e-6 or a<1e-7 or a>99:
                    bad+=1;continue
                samples.append(t);cs.append(1/a-2*np.log(64));chi.append(sum(best.fun**2))
            samples=np.array(samples);cs=np.array(cs)
            bootrows.append(dict(g_d=gd,charge=q,block_size=2000,resamples=240,valid=len(samples),boundary_or_vanishing=bad,
                central_temperature=tc,temperature_sd=np.std(samples,ddof=1) if len(samples)>1 else np.nan,
                temperature_p025=np.quantile(samples,.025) if len(samples) else np.nan,
                temperature_p975=np.quantile(samples,.975) if len(samples) else np.nan,
                C_central=f['C'],C_sd=np.std(cs,ddof=1) if len(cs)>1 else np.nan,
                C_p025=np.quantile(cs,.025) if len(cs) else np.nan,C_p975=np.quantile(cs,.975) if len(cs) else np.nan))
            print('Bootstrap',gd,q,'Tc',tc,'valid',len(samples),'sd',np.std(samples),flush=True)
        axes[0,col].set(title=f'$g_D={gd:g}$',yscale='log',ylim=(.01,1e7),xlabel='Trial temperature')
        axes[1,col].set(xlabel='Temperature',xlim=(.375,.40) if gd<.1 else (.33,.47),ylim=(-.2,3.5))
    axes[0,0].set_ylabel('Profile chi-square (2 d.o.f. at free optimum)');axes[0,0].legend()
    axes[1,0].set_ylabel('Raw common-phase helicity');axes[1,0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'bkt_charge_comparison.png',dpi=160);plt.close(fig)
    write_csv(OUT/'wm_sensitivity.csv',rows);write_csv(OUT/'wm_bootstrap.csv',bootrows);write_csv(OUT/'wm_profiles.csv',profrows)
    optimizer_audit()


if __name__=='__main__':main()
