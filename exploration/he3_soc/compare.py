"""Scientific comparisons from summarize.py's compact block cache.

Run: python exploration/he3_soc/compare.py
Fits are explicitly finite-size diagnostics, not transition identifications.
"""
from pathlib import Path
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from exploration.he3_soc.summarize import OUT, CACHE, COUPLINGS, SIZES, estimate, evaluate, write_csv
from utils.Analysis.bkt import fit_weber_minnhagen

COLORS=['#4477aa','#228833','#ee7733','#aa3377']
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})


def load(gd,L):
    with np.load(CACHE/f'gd={gd:g}_L={L}.npz') as z:
        return z['T'],{k:z[k] for k in z.files if k!='T'}


def roots(T,y):
    idx=np.flatnonzero(y[:-1]*y[1:]<0)
    return [float(T[i]-y[i]*(T[i+1]-T[i])/(y[i+1]-y[i])) for i in idx]


def main():
    data={}
    allrows=[]
    peaks=[]
    crossings=[]
    for gd in COUPLINGS:
        for L in SIZES:
            T,b=load(gd,L)
            v,e=estimate(b,T,L,2000)
            data[gd,L]=(T,v,e)
            for name in ['specific_heat','chi_response','phase_response']:
                mask=(T>.35)&(T<.42) if gd<.1 else np.ones(len(T),bool)
                ids=np.flatnonzero(mask);j=ids[np.argmax(v[name][mask])]
                peaks.append(dict(g_d=gd,L=L,observable=name,temperature=T[j],
                    bracket_low=T[max(0,j-1)],bracket_high=T[min(len(T)-1,j+1)],
                    value=v[name][j],within_history_err=e[name][j]))
            for q in [.5,1.]:
                r=roots(T,v['helicity']-2*T/(np.pi*q*q))
                for t in r:
                    crossings.append(dict(g_d=gd,L=L,kind='jump',charge=q,temperature=t))
        for a,bb in [(64,96),(96,128),(128,160),(64,160)]:
            t,v,e=data[gd,a];vv=data[gd,bb][1]
            for name in ['binder_chirality','binder_phase']:
                mask=(t>.372)&(t<.40) if gd<.1 else (t>.35)&(t<.46)
                for r in roots(t[mask],(v[name]-vv[name])[mask]):
                    crossings.append(dict(g_d=gd,L=f'{a}:{bb}',kind=name,charge='',temperature=r))
    write_csv(OUT/'response_peaks.csv',peaks)
    write_csv(OUT/'crossings.csv',crossings)

    # One overview per comparison direction, with strong SOC visually separated.
    names=['helicity','chirality','phase','spin_alignment','density','energy']
    labels=[r'$\Upsilon$',r'$\langle|m_\chi|\rangle$',r'$\sqrt{\langle|\Phi_2/N|^2\rangle}$',r'$Q_{zz}$',r'$n$',r'$e$']
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,name,label in zip(axes.flat,names,labels):
        for color,gd in zip(COLORS,COUPLINGS):
            T,v,e=data[gd,160]
            ax.plot(T,v[name],color=color,label=f'$g_D={gd:g}$')
            ax.fill_between(T,v[name]-e[name],v[name]+e[name],color=color,alpha=.15)
        ax.set(xlabel='Temperature',ylabel=label)
    axes[0,0].legend();fig.suptitle('SOC comparison at L=160; 2,000-measurement block errors')
    fig.tight_layout();fig.savefig(OUT/'soc_overview.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(16,10),sharex='col')
    for col,gd in enumerate(COUPLINGS):
        for color,L in zip(COLORS,SIZES):
            T,v,e=data[gd,L]
            for row,name in enumerate(['helicity','binder_chirality','spin_alignment']):
                ax=axes[row,col];ax.plot(T,v[name],color=color,label=f'L={L}')
                ax.fill_between(T,v[name]-e[name],v[name]+e[name],color=color,alpha=.12)
        for row in range(3):
            axes[row,col].set_xlim((.37,.4) if gd<.1 else (.33,.47))
        axes[0,col].set_title(f'$g_D={gd:g}$')
        axes[2,col].set_xlabel('Temperature')
        if gd<.1:axes[1,col].set_ylim(-.5,.7)
        else:axes[1,col].set_yscale('symlog',linthresh=1)
    axes[0,0].set_ylabel(r'$\Upsilon$');axes[1,0].set_ylabel(r'$U_\chi$');axes[2,0].set_ylabel(r'$Q_{zz}$')
    axes[0,0].legend();fig.tight_layout();fig.savefig(OUT/'finite_size.png',dpi=160);plt.close(fig)

    # Response peaks and composite ordering show whether characteristic scales coincide.
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for col,gd in enumerate(COUPLINGS[:3]):
        for color,L in zip(COLORS,SIZES):
            T,v,e=data[gd,L]
            axes[0,col].plot(T,v['binder_phase'],color=color,label=f'L={L}')
            axes[1,col].plot(T,v['specific_heat'],color=color)
        axes[0,col].set(title=f'$g_D={gd:g}$',ylim=(-.4,1.05),xlim=(.375,.398))
        axes[1,col].set(xlabel='Temperature',xlim=(.375,.398))
    axes[0,0].set_ylabel('Composite Binder');axes[1,0].set_ylabel(r'$c_V=\mathrm{Var}(E)/(NT^2)$')
    axes[0,0].legend();fig.tight_layout();fig.savefig(OUT/'weak_ordering.png',dpi=160);plt.close(fig)

    # Representative reliability summaries, retaining exact temperature/observable rows.
    stats=[]
    for gd in COUPLINGS:
        for L in SIZES:
            with (OUT/f'statistics_gd={gd:g}_L={L}.csv').open() as f:
                a=list(csv.DictReader(f))
            for name in ['energy','density','chirality','signed_chirality','phase_magnitude','spin_alignment','current_x','current_y']:
                rr=[r for r in a if r['observable']==name]
                ids=[int(np.argmin([abs(float(r['temperature'])-target) for r in rr])) for target in ([.33,.38,.385,.40,.45] if gd<.1 else [.33,.36,.37,.40,.443,.46])]
                stats.extend([rr[j] for j in sorted(set(ids))])
    write_csv(OUT/'statistics_selected.csv',stats)
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
    for col,name in enumerate(['energy','chirality','spin_alignment']):
        for color,gd in zip(COLORS,COUPLINGS):
            rr=[r for r in stats if float(r['g_d'])==gd and int(r['L'])==160 and r['observable']==name]
            axes[0,col].plot([float(r['temperature']) for r in rr],[float(r['effective_samples']) for r in rr],'.-',color=color,label=f'$g_D={gd:g}$')
            axes[1,col].plot([float(r['temperature']) for r in rr],[float(r['se_5000'])/float(r['se_100']) for r in rr],'.-',color=color)
        axes[0,col].set(title=name,yscale='log');axes[1,col].set(xlabel='Temperature')
    axes[0,0].set_ylabel('Estimated ESS (conditional on history)');axes[1,0].set_ylabel('SE(5,000 block) / SE(100 block)')
    axes[0,0].legend();fig.tight_layout();fig.savefig(OUT/'statistical_reliability.png',dpi=160);plt.close(fig)

    # Fixed-low-temperature SOC response and paired shifts use conservative independent errors.
    low=[]
    for gd in COUPLINGS:
        for L in SIZES:
            T,v,e=data[gd,L]
            row=dict(g_d=gd,L=L,temperature=T[0])
            for name in ['spin_alignment','density','energy','chirality','helicity']:
                row[name]=v[name][0];row[name+'_err']=e[name][0]
            low.append(row)
    write_csv(OUT/'low_temperature.csv',low)
    print('Saved thermodynamic comparisons',flush=True)


if __name__=='__main__':main()
