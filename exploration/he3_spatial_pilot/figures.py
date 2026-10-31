"""Three compact figures from saved pilot data; no GPU calculations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from observables import SECTORS
from ladder import OUT


def read(name):
    with (OUT/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'ordered':'#4477aa','hot':'#cc6677'};sp=read('spatial_summary.csv')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for col,sector in enumerate(SECTORS):
        for label,color in colors.items():
            r=[x for x in sp if x['history']==label and x['sector']==sector];t=np.array([float(x['temperature']) for x in r])
            s=np.array([float(x['S0'])/64**2 for x in r]);se=np.array([float(x['S0_se'])/64**2 for x in r]);xi=np.array([float(x['xi_over_L']) for x in r]);ee=np.array([float(x['xi_over_L_se']) for x in r])
            axes[0,col].errorbar(t,s,yerr=se,fmt='.-',color=color,ms=3,lw=1,label=label)
            axes[1,col].errorbar(t,xi,yerr=ee,fmt='.-',color=color,ms=3,lw=1,label=label)
        axes[0,col].set(title=sector,yscale='log',ylabel=r'$S(0)/L^2$')
        axes[1,col].set(xlabel='Temperature',ylabel=r'$\xi_2/L$')
        for ax in axes[:,col]:ax.axvline(.3815,color='gray',ls=':',lw=.8)
    axes[0,0].legend();fig.suptitle('L=64, gD=0: independent ordered and hot starts; conditional block errors')
    fig.tight_layout();fig.savefig(OUT/'spatial_vs_temperature.png',dpi=170);plt.close(fig)

    times=read('time_history.csv');edge=read('pt_edges.csv');blocks=read('blocking.csv')
    fig,ax=plt.subplots(2,3,figsize=(13,7))
    for label,color in colors.items():
        for i,(key,t,title) in enumerate([('energy',.3815,'Energy near ordering'),('signed_chirality',.37,'Signed chirality at cold control'),('spin_alignment',.37,'Spin orientation at cold control')]):
            r=[x for x in times if x['history']==label and float(x['temperature'])==t]
            ax[0,i].plot([float(x['sweep'])/1000 for x in r],[float(x[key]) for x in r],color=color,label=label,lw=.9)
            ax[0,i].axvline(100,color='gray',ls=':');ax[0,i].set(title=title,xlabel='Sweep / 1000',ylabel=key)
        r=[x for x in edge if x['history']==label]
        ax[1,0].plot([(float(x['Tlow'])+float(x['Thigh']))/2 for x in r],[float(x['production_acceptance']) for x in r],'.-',color=color,label=label)
        ax[1,0].set(xlabel='Temperature-edge midpoint',ylabel='Production swap acceptance',ylim=(0,1),title='No severe pilot swap bottleneck')
        for sector,marker in [('chirality','o'),('composite','s'),('nematic','^')]:
            r=[x for x in blocks if x['history']==label and x['sector']==sector and float(x['temperature'])==.3815]
            ax[1,1].plot([float(x['block_sweeps']) for x in r],[float(x['xi_over_L_se']) for x in r],marker=marker,color=color,ls='-' if label=='ordered' else '--',ms=4,label=label+' '+sector)
        r=[x for x in sp if x['history']==label and x['sector']=='chirality' and float(x['temperature'])>=.395]
        ax[1,2].errorbar([float(x['temperature']) for x in r],[float(x['ratio'])-1 for x in r],yerr=[float(x['ratio_se']) for x in r],fmt='.-',color=color,label=label)
    ax[0,0].legend();ax[1,1].set(xscale='log',xlabel='Jackknife block / sweeps',ylabel=r'SE of $\xi_2/L$',title='Critical block sensitivity (5–80 events)');ax[1,1].legend(fontsize=7,ncol=2)
    ax[1,1].set_xticks([1250,2500,5000,10000,20000],['1.25k','2.5k','5k','10k','20k'])
    ax[1,1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[1,2].axhline(0,color='gray',ls=':');ax[1,2].set(xlabel='Temperature',ylabel=r'$S(0)/S(k_{min})-1$',title='Hot ratios can be negative: retain and flag')
    fig.tight_layout();fig.savefig(OUT/'sampling_checks.png',dpi=170);plt.close(fig)

    corr=read('sparse_realspace_correlations.csv');gauss=read('gaussian_reference.csv')
    fig,ax=plt.subplots(2,2,figsize=(11,8));cc={.37:'#4477aa',.3815:'#228833',.385:'#cc6677',.415:'#aa3377'}
    for panel,sector in zip(ax.flat[:3],SECTORS):
        for t,color in cc.items():
            x=[];y=[]
            for r in range(1,33):
                rr=[a for a in corr if a['sector']==sector and float(a['temperature'])==t and int(a['r'])==r]
                # Ensemble/site-average correlations first, normalization second.
                if rr:
                    x.append(r);y.append(np.mean([(float(a['Gx_real'])+float(a['Gy_real']))/2 for a in rr])/np.mean([float(a['C0']) for a in rr]))
            panel.plot(x,y,'.-',ms=3,color=color,label=f'T={t:g}')
        panel.axhline(0,color='gray',lw=.6);panel.set(xlabel='Separation / sites',ylabel=r'$\mathrm{Re}\,G(r)/G(0)$',title=sector+': six sparse fields, illustrative')
    ax[0,0].legend(fontsize=8)
    for label,color in colors.items():
        r=[x for x in gauss if x['history']==label and .37<=float(x['temperature'])<=.3862]
        ax[1,1].errorbar([float(x['temperature']) for x in r],[float(x['xiP_over_L_measured']) for x in r],yerr=[float(x['xiP_over_L_se']) for x in r],fmt='.-',color=color,ms=3,label=label+' measured')
        ax[1,1].plot([float(x['temperature']) for x in r],[float(x['xiP_over_L_gaussian']) for x in r],'--',color=color,label=label+' harmonic reference')
    ax[1,1].set(xlabel='Temperature',ylabel=r'Composite $\xi_2/L$',title='Phase-only zero-winding reference is incomplete');ax[1,1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'spatial_interpretation.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
