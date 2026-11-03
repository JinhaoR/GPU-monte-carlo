"""Focused figures from preparation-resolved pilot tables; no new sampling."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'pilot_results'


def main():
    rows=list(csv.DictReader((OUT/'curves.csv').open()))
    colors=dict(chiral='tab:blue',hot='tab:orange',planar='tab:green')
    groups={}
    for r in rows:
        if r['subset']=='full' and r['L']=='16':groups.setdefault((float(r['g_d']),r['preparation']),[]).append(r)
    fig,axs=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for j,g in enumerate((.04,.06,.08,.1)):
        for kind in colors:
            z=groups[g,kind];T=np.array([float(r['T']) for r in z])
            for key,ls,label in [('Azplus_S0','-',kind+' A+,z'),('Aplanar_S0','--',kind+' A antisym.')]:
                v=np.array([float(r[key]) for r in z])/256
                axs[0,j].plot(T,v,color=colors[kind],ls=ls,label=label)
            for key,ls in [('Bpp_S0','-'),('Bmm_S0','--')]:
                axs[1,j].plot(T,[float(r[key])/256 for r in z],color=colors[kind],ls=ls,label=kind+' '+key[:3])
        axs[0,j].set(title=f'g_D={g:g}',ylabel='S_A(0) / N',xlim=(.28,.5))
        axs[1,j].set(xlabel='MC temperature',ylabel='S_B(0) / N',xlim=(.28,.5))
        for ax in axs[:,j]:ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7);axs[1,0].legend(fontsize=7)
    fig.suptitle('L=16 orbital channels: every preparation retained separately')
    fig.savefig(OUT/'orbital_channels.png',dpi=180);plt.close(fig)

    joint=list(csv.DictReader((OUT/'joint_samples.csv').open()))
    fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for row,t in enumerate((.28,.3765217391304348)):
        for kind in colors:
            z=[r for r in joint if r['L']=='16' and float(r['g_d'])==.08 and r['preparation']==kind and abs(float(r['T'])-t)<1e-8]
            q=[float(r['q']) for r in z]
            for col,key in enumerate(('e','epsilon','planar_weight')):
                axs[row,col].scatter(q,[float(r[key]) for r in z],s=6,alpha=.35,color=colors[kind],label=kind)
                axs[row,col].set(xlabel='spin alignment',ylabel=key,title=f'T={t:.6f}')
                axs[row,col].grid(alpha=.2)
    axs[0,0].legend();fig.suptitle('g_D=.08, L=16: cold trapping versus a common planar-like regime')
    fig.savefig(OUT/'joint_structure.png',dpi=180);plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for kind in colors:
        z=groups[.04,kind];T=np.array([float(r['T']) for r in z])
        for ax,key in zip(axs,('Yxy','lambda_min','sqrt_det')):
            ax.errorbar(T,[float(r[key]) for r in z],[float(r[key+'_se']) for r in z],color=colors[kind],label=kind,lw=1)
            ax.set(xlim=(.28,.44),xlabel='MC temperature',ylabel=key);ax.grid(alpha=.2)
    axs[0].axhline(0,c='k',lw=.7);axs[0].legend();fig.suptitle('g_D=.04, L=16: full stiffness tensor, joint conditional errors')
    fig.savefig(OUT/'stiffness_tensor.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
