"""Three figures and small descriptive tables, all from existing histories."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import brentq
from reweight import load,solve,direct
from scaling import read,line,crossing
from study import OUT,SIZES,KEYS,COUPLINGS,csvwrite,error


def table(name):
    with (OUT/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=plt.cm.viridis(np.linspace(.05,.85,4));data={L:read(0,L) for L in SIZES}
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for L,color in zip(SIZES,colors):
        a=data[L];d=load(0,L);dv=[direct(d,k,L) for k in range(15)]
        for ax,key in zip(axes[0],['binder_chirality','binder_phase']):
            i=KEYS.index(key);se=error(a['jk'][:,i]);ax.plot(a['T'],a['point'][i],color=color,label=f'L={L}')
            ax.fill_between(a['T'],a['point'][i]-se,a['point'][i]+se,color=color,alpha=.13)
            ax.plot(1/d['beta'],[v[key] for v in dv],':o',ms=3,color=color,alpha=.6)
        ax=axes[1,0];i=KEYS.index('helicity');ax.plot(a['T'],a['point'][i],color=color,label=f'L={L}')
    for ax in axes.flat:ax.set_xlabel('Temperature')
    axes[0,0].set(xlim=(.3812,.3825),ylim=(.1,.68),ylabel='Chirality Binder',title='Zero SOC: curves versus linear grid joins')
    axes[0,1].set(xlim=(.3812,.3825),ylim=(-.3,1),ylabel='Composite Binder',title='Same configurations; strongly correlated crossings')
    axes[0,0].legend(fontsize=8)
    T=data[64]['T'];axes[1,0].plot(T,8*T/np.pi,'k--',label=r'$8T/\pi$')
    axes[1,0].axvline(.38151891,color='k',ls=':',label='q=1/2 WM diagnostic')
    axes[1,0].set(xlim=(.3808,.3825),ylim=(.3,1.65),ylabel='Common-phase helicity',title='Leading WM form is conditional');axes[1,0].legend(fontsize=8)
    for (L1,L2),color in zip([(64,96),(96,128),(128,160)],colors):
        eta=-np.log(line(data[L2],'phase2',T)/line(data[L1],'phase2',T))/np.log(L2/L1)
        axes[1,1].plot(T,eta,color=color,label=f'{L1},{L2}')
    axes[1,1].plot(T,2*T/(np.pi*line(data[160],'helicity',T)),'k--',label=r'$2T/(\pi\Upsilon_{160})$')
    axes[1,1].axhline(.25,color='gray',ls=':');axes[1,1].axvline(.38151891,color='k',ls=':')
    axes[1,1].set(xlim=(.3804,.3821),ylim=(.1,1.3),ylabel=r'Effective charge-two exponent $\eta_P$',title='Independent test of a simple phase-only interpretation')
    axes[1,1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'zero_soc_diagnostics.png',dpi=170);plt.close(fig)

    vals=table('withheld_validation.csv');ovs=table('energy_overlap.csv')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for L,color in zip(SIZES,colors):
        r=[x for x in ovs if float(x['g_d'])==0 and int(x['L'])==L]
        ax[0].plot([(float(x['Tlow'])+float(x['Thigh']))/2 for x in r],[float(x['histogram_overlap']) for x in r],'-o',ms=3,color=color,label=f'L={L}')
    ax[0].set(xlabel='Neighboring-temperature midpoint',ylabel=r'$\sum_b\min(p_i(b),p_{i+1}(b))$',title='Energy overlap at zero SOC');ax[0].legend(fontsize=8)
    ks=['energy','binder_chirality','binder_phase','helicity','specific_heat','spin_alignment']
    for g,col,offset in [(0,'#4477aa',-.16),(.001,'#228833',0),(.01,'#cc6677',.16)]:
        for j,k in enumerate(ks):
            r=[float(x['paired_z']) for x in vals if float(x['g_d'])==g and x['observable']==k]
            ax[1].scatter(j+offset+np.linspace(-.03,.03,len(r)),r,c=col,s=13,label=f'gD={g:g}' if j==0 else None)
    ax[1].axhline(0,color='gray',lw=.7);ax[1].axhline(3,color='gray',ls=':');ax[1].axhline(-3,color='gray',ls=':')
    ax[1].set(xticks=range(6),xticklabels=['E','Uchi','UP','helicity','Cv','Qzz'],ylabel='Prediction minus measurement / paired error',title='Omitted-temperature validation, 216 correlated tests');ax[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'reweight_validation.png',dpi=170);plt.close(fig)

    peaks=table('response_peaks.csv');cross=table('crossings.csv');wm=table('wm_fits.csv');histrows=[];modes=[];fitrows=[]
    fig,ax=plt.subplots(2,2,figsize=(11,8))
    for key,marker in [('specific_heat','o'),('chi_response','s'),('phase_response','^')]:
        r=[x for x in peaks if float(x['g_d'])==0 and x['observable']==key];y=np.array([float(x['height']) for x in r]);s=np.array([float(x['height_se']) for x in r]);L=np.array(SIZES)
        # Descriptive interval exponents, not asymptotic exponent fits.
        fitrows.append(dict(observable=key,description='endpoint_effective_power',estimate=np.log(y[-1]/y[0])/np.log(L[-1]/L[0]),chi2_dof=np.nan))
        if key=='specific_heat':
            for label,x in [('a+b_logL',np.log(L)),('a+b_L2',L**2)]:
                X=np.column_stack([np.ones(4),x]);coef=np.linalg.lstsq(X/s[:,None],y/s,rcond=None)[0]
                fitrows.append(dict(observable=key,description=label,estimate=coef[1],chi2_dof=np.sum(((X@coef-y)/s)**2)/2))
        ax[0,0].errorbar(L,y/y[0],yerr=s/y[0],marker=marker,label=key)
    ax[0,0].plot(SIZES,(np.array(SIZES)/64)**2,'k--',label=r'$L^2$ reference')
    ax[0,0].set(xlabel='L',ylabel='Peak / L=64 peak',title='Peak growth is slower than asymptotic coexistence');ax[0,0].legend(fontsize=8)
    for L,color in zip(SIZES,colors):
        r=next(x for x in peaks if float(x['g_d'])==0 and int(x['L'])==L and x['observable']=='specific_heat');t=float(r['temperature'])
        d=load(0,L);f=solve(d['H'].sum(0),d['S'].sum(1),d['beta'],d['ebins'])
        lw=-(1/t-f['beta_ref'])*f['e']-f['ld']+np.log(f['h']);p=np.exp(lw-np.logaddexp.reduce(lw));mean=p@f['e'];sigma=np.sqrt(p@(f['e']-mean)**2)
        x=(f['e']-mean)/L**2;dens=p/(d['edges'][1]-d['edges'][0])*L**2
        ax[0,1].plot(x,gaussian_filter1d(dens,2),color=color,label=f'L={L}')
        for xx,pp in zip(x,dens):histrows.append(dict(L=L,temperature=t,centered_energy_density=xx,probability_density=pp))
        for smooth in (1,2,4):
            yy=gaussian_filter1d(p,smooth);ii,_=find_peaks(yy,prominence=.08*yy.max(),distance=max(1,int(.5*sigma/(d['edges'][1]-d['edges'][0]))))
            modes.append(dict(L=L,temperature=t,smoothing_bins=smooth,prominent_modes=len(ii)))
    ax[0,1].set(xlabel=r'$E/L^2-\langle E/L^2\rangle$',ylabel='Probability density',title='Energy distributions at reweighted Cv maxima');ax[0,1].legend(fontsize=8)
    for g,color in zip(COUPLINGS,['#4477aa','#228833','#cc6677']):
        base={0:0,.001:.00045,.01:.0036}[g]
        for k,marker,offset in [('binder_chirality','o',-1.5),('binder_phase','s',1.5)]:
            r=[x for x in cross if float(x['g_d'])==g and x['observable']==k and (int(x['L1']),int(x['L2'])) in [(64,96),(96,128),(128,160)]]
            ax[1,0].errorbar([int(x['L2'])+offset for x in r],[float(x['temperature'])-base for x in r],yerr=[float(x['joint_pt_se']) for x in r],marker=marker,linestyle='-',color=color,label=f'gD={g:g}, '+('chi' if k=='binder_chirality' else 'P'))
    ax[1,0].set(xlabel='Larger L in adjacent-size pair',ylabel='Binder crossing T (weak-SOC offsets subtracted)',title='No stable four-size extrapolation');ax[1,0].legend(fontsize=7,ncol=2)
    joint=np.load(OUT/'joint_order_distribution.npz');h=joint['hist'];ax[1,1].pcolormesh(joint['xedges'],joint['yedges'],h.T,cmap='magma',shading='auto')
    ax[1,1].set(xlabel='Absolute normalized chirality',ylabel='Composite magnitude',title='Global magnitudes share a strong fluctuation mode')
    fig.tight_layout();fig.savefig(OUT/'finite_size_and_distributions.png',dpi=170);plt.close(fig)
    csvwrite('energy_distributions.csv',histrows);csvwrite('energy_modes.csv',modes);csvwrite('descriptive_scaling.csv',fitrows)
    # At the conditional WM diagnostic, compare the phase-only identity and
    # measured two-size exponents, with independent-size jackknife errors.
    eta=[]
    for g in COUPLINGS:
        dd={L:read(g,L) for L in SIZES};t=float(next(x['temperature'] for x in wm if float(x['g_d'])==g and float(x['q'])==.5 and x['sizes']=='64:96:128:160'))
        for L1,L2 in [(64,96),(96,128),(128,160)]:
            def ee(va=None,vb=None):return -np.log(line(dd[L2],'phase2',t,vb)/line(dd[L1],'phase2',t,va))/np.log(L2/L1)
            se=np.sqrt(error([ee(v,None) for v in dd[L1]['jk']])**2+error([ee(None,v) for v in dd[L2]['jk']])**2)
            eta.append(dict(g_d=g,L1=L1,L2=L2,T_WM=t,eta_P_effective=ee(),se=se,gaussian_from_helicity=2*t/(np.pi*line(dd[L2],'helicity',t))))
    csvwrite('phase_consistency.csv',eta)
    # Summaries used in the report; no fit is hidden in prose-only calculations.
    unbin=table('unbinned_reference.csv');sens=table('temperature_sensitivity.csv')
    summary=dict(withheld_tests=len(vals),withheld_max_abs_z=max(abs(float(x['paired_z'])) for x in vals),
        withheld_abs_z_above_3=sum(abs(float(x['paired_z']))>3 for x in vals),
        max_binder_withheld_z=max(abs(float(x['paired_z'])) for x in vals if x['observable'].startswith('binder')),
        max_abs_800bin_unbinned_difference={k:max(abs(float(x['difference'])) for x in unbin if x['observable']==k and int(x['bins'])==800) for k in ['energy','binder_chirality','binder_phase','helicity','specific_heat']},
        maximum_Binder_T_shift_bins_or_sources=max(abs(float(x['shift'])) for x in sens if x['estimator'].startswith('binder') and (x['variant'].startswith('bins') or x['variant'].startswith('sources'))))
    (OUT/'reweight_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(summary,flush=True)


if __name__=='__main__':main()
