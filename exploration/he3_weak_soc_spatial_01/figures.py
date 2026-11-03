"""Publication-style static figures from the reduced measurements."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from findings import load,OUT,ROOT

FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
SIZES=(64,96,128,160);COLORS=dict(zip(SIZES,plt.cm.viridis(np.linspace(.1,.85,4))))
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})


def save(fig,name):fig.savefig(FIG/name,dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    runs={(g,L):load(g,L) for g in (0,.01) for L in SIZES}
    orb={(g,L):load(g,L,'orbital') for g in (0,.01) for L in SIZES}
    fig,axs=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for col,g in enumerate((0,.01)):
        for L in SIZES:
            z=runs[g,L];T=z['T']
            for row,key in enumerate(('helicity','chirality','specific_heat')):axs[row,col].plot(T,z[key],color=COLORS[L],label=f'L={L}')
        old=ROOT/f'exploration/he3_transition_study/results/curves_gd={g:g}_L=160.npz'
        with np.load(old) as z:
            for row,key in enumerate(('helicity','chirality','specific_heat')):axs[row,col].plot(z['T'],z['point'][list(z['keys']).index(key)],'k--',lw=1,label='old L=160')
        T=runs[g,160]['T'];axs[0,col].plot(T,8*T/np.pi,':',color='firebrick',label='8T/pi');axs[0,col].plot(T,2*T/np.pi,':',color='gray',label='2T/pi')
        for row in range(3):axs[row,col].set(xlim=((.378,.39) if g==0 else (.382,.395)),xlabel='T',ylabel=('helicity Y','normalized |chirality|','heat capacity / site')[row])
        axs[0,col].set_title(f'g_D={g:g}');axs[0,col].legend(fontsize=8,ncol=2)
    save(fig,'global_comparison.png')
    fig,axs=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for col,g in enumerate((0,.01)):
        for L in SIZES:
            z=runs[g,L];T=z['T'];o=orb[g,L]
            axs[0,col].plot(T,z['full_xi_xy'][:,1],color=COLORS[L],label=f'P, L={L}')
            axs[0,col].plot(T,o['B_plus_xi'],'--',color=COLORS[L],label=f'B++, L={L}')
            axs[1,col].plot(T,o['A_plus_z_xi'] if g else o['A_plus_xi'],color=COLORS[L],label=f'plus, L={L}')
            axs[1,col].plot(T,o['A_minus_z_xi'] if g else o['A_minus_xi'],'--',color=COLORS[L],label=f'minus, L={L}')
            axs[2,col].plot(T,z['full_xi_xy'][:,2],color=COLORS[L],label=f'full, L={L}')
            axs[2,col].plot(T,z['connected_xi_xy'][:,6],'--',color=COLORS[L],label=f'connected tilts, L={L}')
        for row in range(3):axs[row,col].set(xlim=((.378,.388) if g==0 else (.382,.394)),ylim=(0,1.8 if row else 1.1),xlabel='T',ylabel=('charge-two xi/L','charge-one xi/L','Q xi/L')[row])
        axs[0,col].set_title(f'g_D={g:g}');axs[0,col].legend(fontsize=7,ncol=2)
        axs[1,col].set_title('All spin components' if not g else 'z-spin component')
        axs[2,col].legend(fontsize=7,ncol=2)
    save(fig,'sector_flows.png')
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,(g,t) in enumerate(((0,.38395826969),(.01,.38695009696))):
        for name,label,marker in [('P','P','o'),('B_plus','B++','s'),('A_plus_z' if g else 'A_plus','A+ z' if g else 'A+','^'),('A_minus','A-','v')]:
            x=[];s=[];err=[]
            for L in SIZES:
                z=runs[g,L];o=orb[g,L];j=np.argmin(abs(z['T']-t))
                if name=='P':x.append(z['full_xi_xy'][j,1]);s.append(z['full_spectra'][j,1,0]);err.append(z['full_xi_xy_se'][j,1])
                else:x.append(o[name+'_xi'][j]);s.append(o[name+'_S'][j,0]);err.append(o[name+'_xi_se'][j])
            axs[0,col].errorbar(SIZES,x,yerr=err,fmt=marker+'-',label=label,capsize=2)
            axs[1,col].loglog(SIZES,s,marker+'-',label=label)
        axs[0,col].set(title=f'g_D={g:g}, T={t:.5f}',xlabel='L',ylabel='xi / L');axs[0,col].legend()
        axs[1,col].set(xlabel='L',ylabel='S(0)');axs[1,col].legend()
    save(fig,'orbital_evidence.png')
    fig,axs=plt.subplots(2,3,figsize=(14,7.5),constrained_layout=True)
    for row,g in enumerate((0,.01)):
        for col,p in enumerate((1,2,3)):
            ax=axs[row,col]
            for L in SIZES:
                z=runs[g,L];r=np.arange(L//2+1);c=z['profile_full'][p,1].real.mean(0)
                ax.semilogy(r,np.where(c>0,c,np.nan),color=COLORS[L],label=f'L={L}')
            ax.set(title=f'g_D={g:g}, T={z["profile_T"][p]:.5f}',xlabel='r',ylabel='C_P(r)',xlim=(1,80),ylim=(1e-5,.4));ax.legend(fontsize=8)
    save(fig,'P_profiles.png')
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for col,g in enumerate((0,.01)):
        z=runs[g,160];T=z['T']
        for i,label in [(2,'full Q'),(5,'m=0'),(6,'m=1 tilts'),(7,'m=2')]:
            key='full_xi_xy' if i==2 else 'connected_xi_xy'
            axs[0,col].plot(T,z[key][:,i]*160,label=label)
        axs[0,col].set(title=f'g_D={g:g}, L=160',xlabel='T',ylabel='second-moment length',xlim=(.374,.399),ylim=(0,280));axs[0,col].legend()
        for p in (0,1,3):
            vals=[runs[g,L]['connected_xi_xy'][np.argmin(abs(runs[g,L]['T']-runs[g,L]['profile_T'][p])),6]*L for L in SIZES]
            axs[1,col].plot(SIZES,vals,'o-',label=f'T={runs[g,160]["profile_T"][p]:.5f}')
        axs[1,col].set(xlabel='L',ylabel='connected tilt-channel length');axs[1,col].legend()
    save(fig,'spin_alignment_and_fluctuations.png')
    fig,axs=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for col,g in enumerate((0,.01)):
        z=runs[g,160];T=z['T']
        for key in ('D','S','helicity'):axs[0,col].plot(T,z[key],label=key)
        for key in ('epsilon','orbital_loss','spin_loss'):axs[1,col].plot(T,z[key],label=key)
        for key in ('energy_bulk','energy_gradient','energy_soc'):axs[2,col].plot(T,z[key],label=key)
        for row in range(3):axs[row,col].set(xlim=((.375,.39) if not g else (.378,.395)),xlabel='T');axs[row,col].legend()
        axs[0,col].set_title(f'g_D={g:g}, L=160');axs[1,col].set_ylabel('density-weighted residual');axs[2,col].set_ylabel('energy per site')
    save(fig,'curvature_structure_energy.png')
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,g in enumerate((0,.01)):
        for L in SIZES:
            with np.load(OUT/f'gd={g:g}_L={L}.npz') as z:
                T=z['T'];axs[0,col].plot((T[1:]+T[:-1])/2,z['pt_acceptance'],color=COLORS[L],label=f'L={L}')
                axs[1,col].plot(T,z['pt_flow'],color=COLORS[L],label=f'L={L}')
        axs[0,col].set(title=f'g_D={g:g}',ylabel='neighbor swap acceptance',xlabel='T',ylim=(0,1));axs[0,col].legend()
        axs[1,col].set(xlabel='T',ylabel='fraction last visiting cold endpoint',ylim=(0,1))
    save(fig,'PT_transport.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    gauss=list(csv.DictReader((OUT/'gaussian_checks.csv').open()))
    for ax,g in zip(axs,(0,.01)):
        rows=sorted([r for r in gauss if abs(float(r['g_d'])-g)<1e-6 and r['L']=='160' and .38<float(r['T'])<.39],key=lambda r:float(r['T']))
        ts=np.array([float(r['T']) for r in rows]);ax.plot(ts,[float(r['xi_gaussian']) for r in rows],'k--',label='scalar compact reference')
        ax.plot(ts,[float(r['xi_P_over_L']) for r in rows],'o-',label='P')
        o=orb[g,160];ax.plot(ts,np.interp(ts,o['T'],o['B_plus_xi']),'s-',label='B++')
        ax.set(title=f'g_D={g:g}, L=160',xlabel='T',ylabel='xi / L');ax.legend()
    save(fig,'operator_and_gaussian.png')


if __name__=='__main__':main()
