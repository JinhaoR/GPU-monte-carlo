"""Blocking, PT transport and spatial diagnostics of the two bounded pilots.

Read-only analysis of saved GPU data; no updates or new simulations.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exploration.he3_soc.summarize import autocorrelation
from observables import ensemble,powers,local_fields,correlation_ratio,SECTORS
from ladder import OUT


def write(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def jk_error(j):
    j=np.asarray(j);out=np.full(j.shape[1:],np.nan);valid=np.isfinite(j).all(axis=0)
    x=j[:,valid];out[valid]=np.sqrt((len(j)-1)*np.mean((x-x.mean(axis=0))**2,axis=0))
    return out


def global_estimate(g,T,L):
    N=L*L;e=g[...,0];d=g[...,1];m=g[...,2]/d;p2=(g[...,3]**2+g[...,4]**2)/N**2
    hx=(g[...,6].mean(0)-g[...,5].var(0)/T)/N
    hy=(g[...,8].mean(0)-g[...,7].var(0)/T)/N
    return dict(energy=e.mean(0)/N,density=d.mean(0)/N,chirality=np.mean(abs(m),0),signed_chirality=m.mean(0),
        phase2=p2.mean(0),phase=np.sqrt(p2.mean(0)),
        binder_chirality=1-np.mean(m**4,0)/(3*np.mean(m*m,0)**2),binder_phase=2-np.mean(p2*p2,0)/p2.mean(0)**2,
        helicity=(hx+hy)/2,helicity_x=hx,helicity_y=hy,specific_heat=e.var(0)/(N*T*T),spin_alignment=g[...,9].mean(0))


def spatial_estimate(a,L,block=20):
    if len(a)%block:raise ValueError('use a block length dividing the selected history')
    point=ensemble(a,L);keys=[k for k in point if not k.startswith('status')]
    reps={k:[] for k in keys};statuses=[]
    for start in range(0,len(a),block):
        keep=np.r_[np.arange(start),np.arange(start+block,len(a))];v=ensemble(a[keep],L)
        for k in keys:reps[k].append(v[k])
        statuses.append(v['status_xy'])
    return point,{k:jk_error(v) for k,v in reps.items()},np.array(statuses)


def globals_with_error(g,T,L,block=100):
    point=global_estimate(g,T,L);reps={k:[] for k in point}
    for start in range(0,len(g),block):
        keep=np.r_[np.arange(start),np.arange(start+block,len(g))];v=global_estimate(g[keep],T,L)
        for k in point:reps[k].append(v[k])
    return point,{k:jk_error(v) for k,v in reps.items()}


def gaussian_reference(T,rho,L):
    """Deterministic finite-torus harmonic covariance, charge two, zero winding.
    This is a conditional effective-theory reference, not a full-GL prediction.
    """
    if not np.isfinite(rho) or rho<=0:return np.nan,np.nan
    k=2*np.pi*np.fft.fftfreq(L);lam=4*np.sin(k[:,None]/2)**2+4*np.sin(k[None,:]/2)**2
    lam[0,0]=np.inf;green=np.fft.ifft2(1/lam).real
    corr=np.exp(-4*T/rho*(green[0,0]-green));s=np.fft.fft2(corr).real
    _,xi,_=correlation_ratio(s[0,0],(s[1,0]+s[0,1])/2,L)
    return float(xi),2*T/(np.pi*rho)


def main():
    spatial=[];globalrows=[];blockrows=[];halves=[];directions=[];spin=[];auto=[];ptrows=[];edges=[];gauss=[];gaussgap=[];fullcorr=[];history=[];allresults={};prior=[];phase_sign=[]
    for label in ['ordered','hot']:
        folder=OUT/label;meta=json.loads((folder/'run.json').read_text());L=meta['L']
        with np.load(folder/'records.npz') as z:d={k:z[k] for k in z.files}
        T=d['temperatures'];burn=int(d['production_start_sweep']);a=d['amplitudes'][d['spatial_sweeps']>burn];g=d['global_values'][d['global_sweeps']>burn]
        p,se,statuses=spatial_estimate(a,L);v,ge=globals_with_error(g,T,L)
        allresults[label]=(p,se,v,ge)
        for i,t in enumerate(T):
            for j,sector in enumerate(SECTORS):
                spatial.append(dict(history=label,temperature=t,sector=sector,S0=p['spectra'][i,j,0],
                    S0_se=se['spectra'][i,j,0],Sk=p['spectra'][i,j,1:].mean(),
                    ratio=p['ratio_xy'][i,j],ratio_se=se['ratio_xy'][i,j],xi_over_L=p['xi_over_L_xy'][i,j],
                    xi_over_L_se=se['xi_over_L_xy'][i,j],status=int(p['status_xy'][i,j]),
                    valid_jackknife_deletes=int(np.sum(statuses[:,i,j]==0)),
                    xi_x=p['xi_over_L_x'][i,j],xi_x_se=se['xi_over_L_x'][i,j],
                    xi_y=p['xi_over_L_y'][i,j],xi_y_se=se['xi_over_L_y'][i,j],
                    sample_disconnected_fraction=p['zero_disconnected_estimate'][i,j]/p['spectra'][i,j,0]))
            for key in v:globalrows.append(dict(history=label,temperature=t,observable=key,value=v[key][i],se=ge[key][i]))
            gx,eta=gaussian_reference(t,v['helicity'][i],L)
            gauss.append(dict(history=label,temperature=t,helicity=v['helicity'][i],helicity_se=ge['helicity'][i],
                eta_phase_only=eta,xiP_over_L_gaussian=gx,xiP_over_L_measured=p['xi_over_L_xy'][i,1],
                xiP_over_L_se=se['xi_over_L_xy'][i,1]))
        # 20 Fourier events and 100 global events cover the same 5000 sweeps.
        # Delete them together for the helicity/spatial comparison covariance.
        selected=[int(np.argmin(abs(T-t))) for t in [.37,.38,.3815,.3825,.385]]
        delta=[]
        for b in range(20):
            ka=np.r_[np.arange(b*20),np.arange((b+1)*20,len(a))]
            kg=np.r_[np.arange(b*100),np.arange((b+1)*100,len(g))]
            ap=ensemble(a[ka],L);gg=global_estimate(g[kg],T,L)
            delta.append([ap['xi_over_L_xy'][i,1]-gaussian_reference(T[i],gg['helicity'][i],L)[0] for i in selected])
        es=jk_error(delta)
        for j,i in enumerate(selected):
            reference=gaussian_reference(T[i],v['helicity'][i],L)[0];difference=p['xi_over_L_xy'][i,1]-reference
            gaussgap.append(dict(history=label,temperature=T[i],measured_minus_gaussian=difference,paired_se=es[j],conditional_z=difference/es[j]))
        for bs in [5,10,20,40,80]:
            pp,ee,st=spatial_estimate(a,L,bs)
            for t in [.37,.3815,.385,.415]:
                i=np.argmin(abs(T-t))
                for j,sector in enumerate(SECTORS):blockrows.append(dict(history=label,temperature=T[i],sector=sector,
                    block_spatial_events=bs,block_sweeps=bs*250,blocks=len(a)//bs,
                    xi_over_L_se=ee['xi_over_L_xy'][i,j],ratio_se=ee['ratio_xy'][i,j]))
        # Paired directional differences are formed on the same deletes.
        dif=[];asym=[]
        for start in range(0,len(a),20):
            keep=np.r_[np.arange(start),np.arange(start+20,len(a))];ev=ensemble(a[keep],L)
            dif.append(ev['xi_over_L_x']-ev['xi_over_L_y'])
            sp=powers(a[keep],L).mean(0)
            asym.append((sp[:,1,1]-sp[:,1,2])/(sp[:,1,1]+sp[:,1,2]))
        err=jk_error(dif);sp=powers(a,L).mean(0);asy=(sp[:,1,1]-sp[:,1,2])/(sp[:,1,1]+sp[:,1,2]);ae=jk_error(asym)
        for i,t in enumerate(T):
            for j,sector in enumerate(SECTORS):
                df=p['xi_over_L_x'][i,j]-p['xi_over_L_y'][i,j]
                directions.append(dict(history=label,temperature=t,sector=sector,xi_x_minus_y=df,se=err[i,j],paired_z=df/err[i,j]))
            phase_sign.append(dict(history=label,temperature=t,composite_Splus_minus_x_asymmetry=asy[i],se=ae[i]))
        # Covariance of independent histories is not invented: their errors add.
        for half,sl in [('first',slice(0,len(a)//2)),('second',slice(len(a)//2,None))]:
            hp,he,_=spatial_estimate(a[sl],L,20)
            for i,t in enumerate(T):
                for j,sector in enumerate(SECTORS):halves.append(dict(history=label,half=half,temperature=t,sector=sector,
                    xi_over_L=hp['xi_over_L_xy'][i,j],se=he['xi_over_L_xy'][i,j],ratio=hp['ratio_xy'][i,j],ratio_se=he['ratio_xy'][i,j]))
        # Component anisotropy and mean-vector bias are sampling diagnostics.
        cp=(abs(a[:,:,2:,:])**2/L**2).mean(0);fractions=cp[:,:,0]/cp[:,:,0].sum(1)[:,None]
        for i,t in enumerate(T):
            for k in range(5):spin.append(dict(history=label,temperature=t,component=k+1,zero_power_fraction=fractions[i,k],
                kmin_power_fraction=cp[i,k,1:].mean()/cp[i,:,1:].mean(axis=1).sum(),
                mean_zero_mode_real=float(a[:,:,k+2,0].mean(0)[i].real/L**2)))
        series={'energy':g[:,:,0]/L**2,'absolute_chirality':abs(g[:,:,2]/g[:,:,1]),'spin_alignment':g[:,:,9]}
        for j,sector in enumerate(SECTORS):series[sector+'_S0']=powers(a,L)[:,:,j,0]
        for name,x in series.items():
            tau=autocorrelation(x);cadence=250 if name.endswith('_S0') else 50
            for i,t in enumerate(T):auto.append(dict(history=label,temperature=t,observable=name,records=len(x),
                tau_int_records=tau[i],tau_int_sweeps=tau[i]*cadence,finite_history_ess=min(len(x),len(x)/(2*tau[i]))))
        # Average short intervals for drift figures, preserving thermal history.
        for start in range(0,len(d['global_values']),40):
            gg=d['global_values'][start:start+40];mid=d['global_sweeps'][start:start+40].mean()
            for t in [.37,.3815,.415]:
                i=np.argmin(abs(T-t));history.append(dict(history=label,sweep=mid,temperature=T[i],
                    energy=gg[:,i,0].mean()/L**2,abs_chirality=np.mean(abs(gg[:,i,2]/gg[:,i,1])),
                    signed_chirality=np.mean(gg[:,i,2]/gg[:,i,1]),spin_alignment=gg[:,i,9].mean()))
        df=meta['pt_final'];db=meta['pt_at_burn'];attempt=np.array(df['attempts'])-db['attempts'];accepted=np.array(df['accepted'])-db['accepted']
        trips=np.array(df['walker_round_trips']);newtrips=trips-np.array(db['walker_round_trips'])
        wh=d['walker_history_indices'];ws=d['walker_history_sweeps'];prod=wh[ws>=burn]
        both=(prod.min(0)==0)&(prod.max(0)==len(T)-1)
        ptrows.append(dict(history=label,completed_round_trips_full=trips.sum(),completed_round_trips_during_production=newtrips.sum(),
            walkers_with_trip_full=np.sum(trips>0),walkers_with_trip_during_production=np.sum(newtrips>0),
            sampled_walkers_visiting_both_endpoints_during_production=both.sum(),walkers=len(T),
            mean_completed_trip_sweeps=df['mean_round_trip_sweeps'],min_production_acceptance=np.min(accepted/attempt),
            mean_production_acceptance=np.mean(accepted/attempt),loop_seconds=meta['loop_seconds']))
        for i in range(len(T)-1):
            e1,e2=g[:,i,0],g[:,i+1,0];bins=np.linspace(min(e1.min(),e2.min())-1e-7,max(e1.max(),e2.max())+1e-7,81)
            h1=np.histogram(e1,bins)[0]/len(e1);h2=np.histogram(e2,bins)[0]/len(e2)
            edges.append(dict(history=label,Tlow=T[i],Thigh=T[i+1],production_acceptance=accepted[i]/attempt[i],
                measured_histogram_overlap=np.minimum(h1,h2).sum()))
        # Sparse fields: supplementary real-space shapes, not precise fits.
        cr={}
        for path in sorted(folder.glob('snapshot_*.npz')):
            with np.load(path) as z:
                fields=z['fields'];temp=z['temperatures']
            k,pair,tensor,_=local_fields(fields)
            fft=[np.fft.fft2(k,axes=(1,2)),np.fft.fft2(pair,axes=(1,2)),np.fft.fft2(tensor,axes=(1,2))]
            corr=[np.fft.ifft2(abs(fft[0])**2,axes=(1,2))/L**2,
                np.fft.ifft2(abs(fft[1])**2,axes=(1,2))/L**2,
                np.fft.ifft2(np.sum(abs(fft[2])**2,axis=-1),axes=(1,2))/L**2]
            for i,t in enumerate(temp):
                for j,sector in enumerate(SECTORS):cr.setdefault((t,sector),[]).append(corr[j][i])
        for (t,sector),arr in cr.items():
            c=np.mean(arr,axis=0)
            for r in range(L//2+1):fullcorr.append(dict(history=label,temperature=t,sector=sector,r=r,snapshots=len(arr),
                C0=c[0,0].real,Gx_real=c[r,0].real,Gx_imag=c[r,0].imag,Gy_real=c[0,r].real,Gy_imag=c[0,r].imag))
        # Independent previous long history, only within the verified RW window.
        with np.load(ROOT/'exploration/he3_transition_study/results/curves_gd=0_L=64.npz') as z:
            old={k:z[k] for k in z.files}
        for key in ['energy','chirality','helicity','phase2']:
            j=list(old['keys']).index(key);oldse=jk_error(old['jk'][:,j])
            for t in [.38,.3815,.3825,.385]:
                i=np.argmin(abs(T-t));ref=np.interp(t,old['T'],old['point'][j]);rse=np.interp(t,old['T'],oldse)
                prior.append(dict(history=label,temperature=T[i],observable=key,pilot=v[key][i],pilot_se=ge[key][i],
                    long_history=ref,long_history_se=rse,difference_over_combined_error=(v[key][i]-ref)/np.hypot(ge[key][i],rse)))
        print(label,'ratios at .3815',p['xi_over_L_xy'][np.argmin(abs(T-.3815))],flush=True)
    comparison=[]
    p0,e0,v0,g0=allresults['ordered'];p1,e1,v1,g1=allresults['hot']
    for i,t in enumerate(T):
        for j,sector in enumerate(SECTORS):
            err=np.hypot(e0['ratio_xy'][i,j],e1['ratio_xy'][i,j]);diff=p1['ratio_xy'][i,j]-p0['ratio_xy'][i,j]
            comparison.append(dict(temperature=t,observable=sector+'_ratio',hot_minus_ordered=diff,combined_se=err,z=diff/err))
        for key in v0:
            err=np.hypot(g0[key][i],g1[key][i]);diff=v1[key][i]-v0[key][i]
            comparison.append(dict(temperature=t,observable=key,hot_minus_ordered=diff,combined_se=err,z=diff/err))
    for name,rows in [('spatial_summary.csv',spatial),('global_summary.csv',globalrows),('blocking.csv',blockrows),
        ('half_history.csv',halves),('directional_checks.csv',directions),('spin_components.csv',spin),
        ('autocorrelation.csv',auto),('pt_summary.csv',ptrows),('pt_edges.csv',edges),('gaussian_reference.csv',gauss),('gaussian_comparison_gap.csv',gaussgap),
        ('sparse_realspace_correlations.csv',fullcorr),('time_history.csv',history),('long_history_comparison.csv',prior),
        ('composite_momentum_asymmetry.csv',phase_sign),('initialization_comparison.csv',comparison)]:write(name,rows)
    summary=dict(pilot_only=True,histories=2,L=64,spatial_production_events_per_history=400,
        main_block_events=20,main_block_sweeps=5000,
        invalid_central_xi_estimates=[dict(history=r['history'],T=r['temperature'],sector=r['sector'],ratio=r['ratio'],status=r['status']) for r in spatial if r['status']!=0],
        pt=ptrows,max_abs_initialization_z=float(max(abs(x['z']) for x in comparison if np.isfinite(x['z']))))
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()),encoding='utf-8');print(summary,flush=True)


if __name__=='__main__':main()
