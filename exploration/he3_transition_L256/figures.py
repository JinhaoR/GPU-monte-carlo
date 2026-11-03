"""Diagnostic figures and correlated finite-size comparisons; no raw re-reading."""
from pathlib import Path
import csv,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import OUT,ROOT,SIZES,PAIRS,interval,write,crossing
COLORS=['tab:blue','tab:orange','tab:green','tab:red']

def read(name):return list(csv.DictReader((OUT/name).open()))

def main():
    with np.load(OUT/'main.npz') as f:z=dict(f)
    fits=read('main_wm.csv');binder=read('main_binder.csv');direct=read('main_direct_jump.csv');T=z['T'];raw=z['raw'];br=z['bootraw'];se=br.std(1,ddof=1)
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for j,L in enumerate(SIZES):
        axs[0,0].plot(z['critical_T'],z['critical'][j,0],c=COLORS[j],label=f'L={L}')
        axs[0,0].errorbar(T,raw[j,0],se[j,0],fmt='.',ms=3,c=COLORS[j])
        axs[0,1].errorbar(T,raw[j,0],se[j,0],fmt='.-',ms=3,c=COLORS[j],label=f'L={L}')
    axs[0,0].set(xlim=(.3815,.38205),ylim=(.44,.67),xlabel='T',ylabel='U chirality',title='Ordering-region Binder crossing')
    axs[0,1].set(xlim=(.3822,.386),ylim=(-.24,.03),xlabel='T',ylabel='U chirality',title='Negative dip becomes shallower')
    for col,(q,name) in enumerate(((.5,'critical'),(1.,'integer'))):
        ax=axs[1,col];fit=next(f for f in fits if f['sizes']=='96_128_160_256' and float(f['q'])==q);tc=float(fit['T']);C=float(fit['C'])
        grid=z[name+'_T'];right=.3822 if q==.5 else .3862;tt=np.linspace(grid[0],right,200)
        for j,L in enumerate(SIZES):
            ax.errorbar(T,raw[j,1],se[j,1],fmt='.',ms=2,c=COLORS[j],alpha=.6)
            ax.plot(grid,z[name][j,1],c=COLORS[j],label=f'L={L}')
            ax.plot(grid,2*grid/(np.pi*q*q)*(1+1/(2*np.log(L)+C)),c=COLORS[j],ls='--',lw=.7)
            d=next(d for d in direct if int(d['L'])==L and float(d['q'])==q);dt=float(d['T'])
            ax.errorbar(dt,2*dt/(np.pi*q*q),xerr=float(d['se']),fmt='v',ms=4,c=COLORS[j])
        ax.plot(tt,2*tt/(np.pi*q*q),'k:',label='universal jump');ax.axvline(tc,c='k',lw=.7)
        ax.set(xlim=(.3812,right) if q==.5 else (.38285,right),ylim=(.75,1.32) if q==.5 else (.1,.65),xlabel='T; triangles: direct crossings',ylabel='Y',title=f'q={q:g}: WM chi2/dof={float(fit["chi2"]):.2f}/2 (poor fit)')
    for ax in axs.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/'transitions.png',dpi=190);plt.close(fig)
    # Explicit size-scaling test, including the independent new-size prediction.
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    oldrows=list(csv.DictReader((ROOT/'exploration/he3_transition_L96_160/results/main_wm.csv').open()))
    for ax,q,oldfit in [(axs[0],.5,True),(axs[1],.5,False),(axs[2],1.,False)]:
        f=next(f for f in (oldrows if oldfit else fits) if float(f['q'])==q and (float(f['g_d'])==0 if oldfit else f['sizes']=='96_128_160_256'))
        t=float(f['T']);C=float(f['C']);name='critical' if q==.5 else 'integer';grid=z[name+'_T'];yy=z[name][:,1];sig=z[name+'_boot'][:,:,1].std(1,ddof=1)
        l=np.linspace(90,270,250);ax.plot(l,2*t/(np.pi*q*q)*(1+1/(2*np.log(l)+C)),'k--',label='WM at fitted T')
        for j,L in enumerate(SIZES):ax.errorbar(L,np.interp(t,grid,yy[j]),np.interp(t,grid,sig[j]),fmt='o',c=COLORS[j],label=f'L={L}')
        ax.set(xlabel='L',ylabel='Y(L, fitted T)',title=f'q={q:g}; '+('old 3-size prediction' if oldfit else '4-size fit')+f'\nT={t:.7f}');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(OUT/'WM_size_tests.png',dpi=190);plt.close(fig)
    # Existing spatial observables, identical definitions at all four sizes.
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True);spatialpoints=[]
    for j,L in enumerate(SIZES):
        p=OUT/'spatial_summary.npz' if L==256 else ROOT/f'exploration/he3_weak_soc_spatial_01/results/gd=0_L={L}_summary.npz'
        with np.load(p) as f:s=dict(f)
        mask=(T>.3811)&(T<.3827)
        for c,name in enumerate(('kappa','P','Q')):
            axs.flat[c].errorbar(T[mask],s['full_xi_xy'][mask,c],s['full_xi_xy_se'][mask,c],fmt='.-',ms=3,c=COLORS[j],label=f'L={L}')
            axs.flat[c].set(xlabel='T',ylabel=f'xi_{name}/L',title=f'{name}: full correlations')
        axs[1,1].errorbar(T[mask],s['epsilon'][mask],s['epsilon_se'][mask],fmt='.-',ms=3,c=COLORS[j],label=f'L={L}')
        for target in (.38148556,.38179623,.38195089,.38256789):
            k=np.argmin(abs(T-target))
            for c,name in enumerate(('kappa','P','Q')):spatialpoints.append(dict(L=int(L),T=T[k],sector=name,xi_over_L=s['full_xi_xy'][k,c],se=s['full_xi_xy_se'][k,c]))
    axs[1,1].set(xlabel='T',ylabel='density-weighted epsilon',title='Local departure from the chiral manifold')
    for ax in axs.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/'spatial_correlations.png',dpi=190);plt.close(fig);write('spatial_comparison.csv',spatialpoints)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    mask=(T>.381)&(T<.383)
    with np.load(OUT/'spatial_summary.npz') as s:
        for k in ('D','S','helicity'):axs[0].errorbar(T[mask],s[k][mask],s[k+'_se'][mask],fmt='.-',ms=3,label=k)
    for L in SIZES:
        p=OUT/'spatial_summary.npz' if L==256 else ROOT/f'exploration/he3_weak_soc_spatial_01/results/gd=0_L={L}_summary.npz'
        with np.load(p) as s:axs[1].errorbar(T[mask],s['specific_heat'][mask],s['specific_heat_se'][mask],fmt='.-',ms=3,label=f'L={L}')
    axs[0].set(xlabel='T',ylabel='per-site helicity contributions',title='L256: Y = D - S')
    axs[1].set(xlabel='T',ylabel='specific heat / site',title='Energy fluctuation peak')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'helicity_components.png',dpi=190);plt.close(fig)
    with np.load(OUT/'spatial/gd=0_L=256.npz') as f:d=dict(f)
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    axs[0,0].plot((T[:-1]+T[1:])/2,d['pt_acceptance']);axs[0,0].set(xlabel='T midpoint',ylabel='swap acceptance',title='L256: entire run, including burn-in')
    axs[0,1].plot(T,d['pt_flow']);axs[0,1].set(xlabel='T',ylabel='cold-labelled fraction',title='L256: production sampled replica flow')
    for target in (.381641,.381796,.381951):
        k=np.argmin(abs(T-target));x=np.arange(1,61)*.05
        axs[1,0].plot(x,d['chi2'][:,k],label=f'T={T[k]:.6f}')
        axs[1,1].plot(x,d['e'][:,k],label=f'T={T[k]:.6f}')
    axs[1,0].set(xlabel='production sweeps (millions)',ylabel='mean m^2 in 50k block',title='Critical-region chirality history')
    axs[1,1].set(xlabel='production sweeps (millions)',ylabel='energy / site in 50k block',title='Critical-region energy history')
    for ax in axs.flat:ax.grid(alpha=.2)
    axs[1,0].legend(fontsize=7);axs[1,1].legend(fontsize=7)
    fig.savefig(OUT/'PT_and_stationarity.png',dpi=190);plt.close(fig)
    # Size drift and extrema differences use correlated bootstrap realizations.
    drift=z['crossing_boot'][-1]-z['crossing_boot'][1]
    write('binder_size_drift.csv',[dict(difference=float(binder[-1]['T'])-float(binder[1]['T']),**interval(drift))])
    roots=[]
    for a,b in ((0,3),(1,3),(2,3)):
        diff=raw[a,0]-raw[b,0]
        for k in np.where(diff[:-1]*diff[1:]<0)[0]:
            t=T[k]-diff[k]*(T[k+1]-T[k])/(diff[k+1]-diff[k]);roots.append(dict(L1=int(SIZES[a]),L2=256,T=t,U=np.interp(t,T,raw[a,0])))
    write('all_raw_crossings_with_L256.csv',roots)
    print('Figures and comparison tables saved',flush=True)

if __name__=='__main__':main()
