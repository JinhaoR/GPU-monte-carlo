"""Joint time-block statistics and physical-walker diagnostics for saved pilots."""
import sys,os,json,csv
from pathlib import Path
sys.dont_write_bytecode=True
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from ladder_design import HERE,ROOT,OLD,OUT,write
from exploration.he3_pt_equilibration.transport import events,km,acceptance_chain
from exploration.he3_pt_equilibration.physical import evaluate
from exploration.he3_soc.summarize import autocorrelation

def moments(a,L):
    c=a['chirality']/a['density'];pr=a['phase_real']/L**2;pi=a['phase_imag']/L**2
    b=dict(e=a['energy']/L**2,m=c,abs_m=abs(c),m2=c*c,m4=c**4,p2=pr*pr+pi*pi,qzz=a['spin_alignment'])
    for axis in ('x','y'):
        b['i'+axis]=a['current_'+axis];b['i'+axis+'2']=a['current_'+axis]**2;b['k'+axis]=a['curvature_'+axis]
    return b

def interpolate(a,T,target):
    return np.array([np.interp(target,T,row) for row in a])

def blocked(b,T,L,block):
    n=len(b['e'])//block
    means={k:v[:n*block].reshape(n,block,-1).mean(1) for k,v in b.items()}
    avg={k:v.mean(0) for k,v in means.items()}
    value=evaluate(avg,T,L)
    jk=evaluate({k:(n*avg[k]-v)/(n-1) for k,v in means.items()},T,L)
    error={k:np.sqrt((n-1)*np.var(v,axis=0)) for k,v in jk.items()}
    return value,error,means

def mean_curve_error(blocks,T,L):
    n=len(blocks['e']);avg={k:v.mean(0) for k,v in blocks.items()}
    value=evaluate(avg,T,L);jk=evaluate({k:(n*avg[k]-v)/(n-1) for k,v in blocks.items()},T,L)
    return value,{k:np.sqrt((n-1)*np.var(v,axis=0)) for k,v in jk.items()}

def main():
    summaries=[];curves=[];stats=[];native=[];flows=[];survival=[];memory=[];conditioned=[];series=[];residences=[];excursions=[];hot_energy=[]
    for path in sorted(OUT.glob('L96_*.npz')):
        with np.load(path) as z:
            m=json.loads(z['manifest'].item())['runs']['96']['metadata'];a={k:z['L96_'+k] for k in ('energy','density','chirality','phase_real','phase_imag','spin_alignment','current_x','current_y','curvature_x','curvature_y')}
            sp_s=np.array(m['spatial_sample_sweeps']);sp_a=z['L96_spatial_amplitudes'];sp_w=z['L96_spatial_walkers']
        T=np.array(m['temperatures']);R=len(T);L=96;gd=m['model_parameters']['g_d'];gd=0. if gd==0 else .01
        base=dict(case=path.stem,ladder=m['ladder'],g_d=gd,initialization=m['initialization'],L=L,R=R)
        ts=np.array(m['sample_sweeps']);burn=m['diagnostic_burn_sweeps'];prod=m['diagnostic_production_sweeps'];keep=ts>burn
        pt=m['parallel_tempering'];history=pt['walker_history'];hs=np.array(history['sweeps']);pos=np.array(history['temperature_indices'],dtype=np.int16)
        assert np.all(np.sort(pos,axis=1)==np.arange(R))
        hi=np.searchsorted(hs,ts);assert np.array_equal(hs[hi],ts)
        sp_i=np.searchsorted(hs,sp_s);np.testing.assert_array_equal(np.argsort(pos[sp_i],axis=1),sp_w)
        tc=.38210530380170976 if gd==0 else .3856721419201371
        probes=np.array([.375,tc-.0003,tc,tc+.0003,.402,.405]);b=moments(a,L);bp={k:v[keep] for k,v in b.items()}
        v,se,_=blocked(bp,T,L,100) # 5000-sweep blocks
        j=np.argmin(abs(T-tc));avg={k:x[:,j].mean() for k,x in bp.items()}
        raw_influence=dict(energy=bp['e'][:,j],m2=bp['m2'][:,j],P2=bp['p2'][:,j],spin=bp['qzz'][:,j],
            Binder=-bp['m4'][:,j]/(3*avg['m2']**2)+2*avg['m4']*bp['m2'][:,j]/(3*avg['m2']**3),
            Y=(bp['kx'][:,j]+bp['ky'][:,j]-(bp['ix2'][:,j]+bp['iy2'][:,j]-2*avg['ix']*bp['ix'][:,j]-2*avg['iy']*bp['iy'][:,j])/T[j])/(2*L*L))
        for key,x in raw_influence.items():
            tau=autocorrelation(x[:,None])[0]*50;ess=x.var(ddof=1)/se[key][j]**2
            native.append(dict(**base,T=T[j],target_T=tc,observable=key,value=v[key][j],se=se[key][j],tau_sweeps=tau,acf_ESS=prod/(2*tau),block_ESS=ess,
                block_ESS_per_total_hour=ess*3600/m['timing']['simulation_seconds'],variance_times_total_seconds=se[key][j]**2*m['timing']['simulation_seconds']))
        for j,t in enumerate(T):
            row=dict(**base,T=t)
            for k in v:row[k]=v[k][j];row[k+'_se']=se[k][j]
            curves.append(row)
        for block_sweeps in (1000,5000,10000):
            q={k:interpolate(v,T,probes) for k,v in bp.items()};values,errors,mb=blocked(q,probes,L,block_sweeps//50)
            first,fe,_=blocked({k:v[:len(v)//2] for k,v in q.items()},probes,L,100)
            second,ee,_=blocked({k:v[len(v)//2:] for k,v in q.items()},probes,L,100)
            means={k:x.mean(0) for k,x in q.items()}
            influences=dict(energy=q['e'],m2=q['m2'],Binder=-q['m4']/(3*means['m2']**2)+2*means['m4']*q['m2']/(3*means['m2']**3),
                Y=(q['kx']+q['ky']-(q['ix2']+q['iy2']-2*means['ix']*q['ix']-2*means['iy']*q['iy'])/probes)/(2*L*L),P2=q['p2'],spin=q['qzz'],abs_chirality=q['abs_m'])
            for k,x in influences.items():
                tau=autocorrelation(x)*50;ess=prod/(2*tau);block_ess=x.var(0,ddof=1)/errors[k]**2
                for j,t in enumerate(probes):
                    stats.append(dict(**base,T=t,observable=k,block_sweeps=block_sweeps,value=values[k][j],se=errors[k][j],tau_sweeps=tau[j],acf_ESS=ess[j],block_ESS=block_ess[j],
                        block_ESS_per_hour=block_ess[j]*3600/(m['timing']['simulation_seconds']*prod/(burn+prod)),
                        block_ESS_per_total_hour=block_ess[j]*3600/m['timing']['simulation_seconds'],total_sweeps_per_block_ESS=(burn+prod)/block_ess[j],
                        first_half=first[k][j],second_half=second[k][j],half_shift=second[k][j]-first[k][j],half_combined_se=np.hypot(fe[k][j],ee[k][j])))
        # Retain early relaxation, including the interval excluded as burn-in.
        time_targets=np.array([.375,tc,.402])
        means={k:interpolate(v.reshape(-1,20,R).mean(1),T,time_targets) for k,v in b.items()};ev=evaluate(means,time_targets,L)
        for i,t in enumerate(ts.reshape(-1,20)[:,-1]):
            for j,target in enumerate(time_targets):
                row=dict(**base,sweep=t,T=target,target_T=target)
                row.update({k:v[i,j] for k,v in ev.items()});series.append(row)
        full_loops,full_passages,_,_,labels=events(pos,hs)
        hp=hs>=burn;loops,passages,first,walkers,_=events(pos[hp],hs[hp])
        carry=[r[5] for r in full_passages if r[1] and r[5]>burn];counts=np.histogram(carry,np.linspace(burn,burn+prod,9))[0]
        # Compare common interior surfaces as well as each ladder's endpoints.
        cold=.379 if gd==0 else .383;temp=T[pos];surfaces=np.where(temp<=cold,0,np.where(temp>=.4,R-1,1))
        common_loops,common_passages,_,_,_=events(surfaces,hs);common_carry=[r[5] for r in common_passages if r[1] and r[5]>burn]
        last=np.take_along_axis(labels[hp],np.argsort(pos[hp],axis=1),axis=1);h0=(last==0).sum(0);h1=(last==1).sum(0)
        flow=np.divide(h0,h0+h1,out=np.full(R,np.nan),where=h0+h1>0)
        for j,t in enumerate(T):flows.append(dict(**base,T=t,flow=flow[j],labelled_fraction=(h0[j]+h1[j])/len(last)))
        for kind,recs in [('first_loop',first),('passage',passages)]:
            ss,median=km(recs)
            for t,s,risk,ne in ss:survival.append(dict(**base,kind=kind,time=t,survival=s,risk=risk,events=ne))
        # A one-step index model has no structural memory; skip if an edge never
        # accepted during the transient (an infinite null would be uninformative).
        accept=np.array(pt['acceptance_rates']);elapsed=m['timing']['simulation_seconds'];npass=len(carry)
        summaries.append(dict(**base,burn=burn,production=prod,simulation_seconds=elapsed,min_acceptance=accept.min(),mean_acceptance=accept.mean(),local_acceptance=np.mean(m['sampling']['acceptance_rates']),
            exact_full_loops=sum(pt['walker_round_trips']),sampled_full_loops=sum(r[1] for r in full_loops),production_loops=sum(r[1] for r in loops),walkers_loop=sum(w['loops']>0 for w in walkers),walkers_both=sum(w['both'] for w in walkers),walkers_no_endpoint=sum(not w['started'] for w in walkers),
            first_loop_censored=sum(not r[1] for r in first),first_loop_KM_median=km(first)[1],passage_KM_median=km(passages)[1],production_passages=npass,
            flux_time=2*R*prod/npass if npass else np.inf,passages_per_walker_M=npass/R/prod*1e6,passage_rate_block_se=np.std(counts,ddof=1)*np.sqrt(8)/R/prod*1e6,
            GPU_seconds_per_passage=elapsed*prod/(burn+prod)/npass if npass else np.inf,common_surface_passages=len(common_carry),common_passages_per_walker_M=len(common_carry)/R/prod*1e6,
            common_seconds_per_passage=elapsed*prod/(burn+prod)/len(common_carry) if common_carry else np.inf,
            total_seconds_per_production_passage=elapsed/npass if npass else np.inf,total_seconds_per_common_passage=elapsed/len(common_carry) if common_carry else np.inf,
            min_labelled_fraction=np.min((h0+h1)/len(last))))
        # Same-configuration memory, using scalar-time permutations, not slot ACF.
        p=pos[hi][keep];tprod=ts[keep];temp=T[p]
        w={k:np.take_along_axis(bp[name],p,axis=1) for k,name in [('energy','e'),('chirality','m'),('P2','p2'),('spin','qzz')]}
        chi_all=np.take_along_axis(b['m'],pos[np.searchsorted(hs,ts)],axis=1)
        temp_scalar=T[pos[np.searchsorted(hs,ts)]]
        scalar_surfaces=np.where(temp_scalar<=cold,0,np.where(temp_scalar>=.4,R-1,1))
        scalar_loops=events(scalar_surfaces,ts)[0]
        first_excursions={}
        for duration,event,wi,origin,start,end in scalar_loops:
            if not event or origin!=0 or start<burn:continue
            i,j=np.searchsorted(ts,[start,end]);x,y=chi_all[i,wi],chi_all[j,wi]
            if min(abs(x),abs(y))<=.1:continue
            first_excursions.setdefault(wi,x*y<0)
            excursions.append(dict(**base,walker=wi,start=start,end=end,chirality_before=x,chirality_after=y,opposite=x*y<0))
        summaries[-1].update(cold_hot_cold_walkers=len(first_excursions),cold_hot_cold_opposite=sum(first_excursions.values()))
        residual={k:np.take_along_axis((bp[name]-bp[name].mean(0))/bp[name].std(0),p,axis=1) for k,name in [('energy','e'),('abs_chirality','abs_m'),('P2','p2'),('spin','qzz')]}
        label_atT=np.take_along_axis(labels[hi][keep],np.argsort(p,axis=1),axis=1)
        for name,channel in [('energy','e'),('m2','m2'),('P2','p2'),('spin','qzz')]:
            a=bp[channel];j=np.argmin(abs(T-tc));n=[];s=[]
            for endpoint in (0,1):
                mask=label_atT[:,j]==endpoint;n.append(mask.reshape(8,-1).sum(1));s.append((a[:,j]*mask).reshape(8,-1).sum(1))
            n=np.array(n);s=np.array(s);total=n.sum(1);sums=s.sum(1)
            if total.min()>20 and np.min(total[:,None]-n)>0:
                means=sums/total;diff=(sums[:,None]-s)/(total[:,None]-n);diff=diff[0]-diff[1]
                conditioned.append(dict(**base,T=T[j],observable=name,cold_n=total[0],hot_n=total[1],cold_mean=means[0],hot_mean=means[1],difference=means[0]-means[1],se=np.sqrt(7*np.var(diff))))
        indices_scalar=np.searchsorted(hs,tprod)
        for band,inside in [('hot',T[pos]>=.4),('critical',abs(T[pos]-tc)<.0005)]:
            bad=np.vstack((np.zeros((1,R),int),np.cumsum(~inside,axis=0)))
            for lag in (10,20,100):
                # Require continuous residence at every exchange, including
                # visits between the more sparsely saved scalar measurements.
                valid=(bad[indices_scalar[lag:]+1]-bad[indices_scalar[:-lag]])==0
                for k in ('energy','abs_chirality','P2','spin'):
                    a=residual[k];x=a[:-lag][valid];y=a[lag:][valid]
                    memory.append(dict(**base,band=band,observable=k,lag_sweeps=lag*50,pairs=len(x),correlation=np.corrcoef(x,y)[0,1] if len(x)>20 else np.nan))
                    if band=='hot' and lag==10 and k=='energy':
                        contribution=np.where(valid,(a[:-lag]-x.mean())*(a[lag:]-y.mean()),0).sum(0)
                        top=np.argsort(contribution)[-3:][::-1]
                        hot_energy.append(dict(case=path.stem,correlation=memory[-1]['correlation'],pairs=len(x),
                            largest_walkers=top.tolist(),top3_covariance_fraction=float(contribution[top].sum()/contribution.sum()),
                            max_abs_energy_residual=float(abs(a).max())))
        # Full five-component spin tensor, using the existing spatial channel.
        use=sp_s>burn;indices_sp=sp_i[use];Q=sp_a[use,:,2:7,0].real/L**2
        Q-=Q.mean(0);Q=np.take_along_axis(Q,pos[indices_sp,:,None],axis=1)
        for band,inside in [('hot',T[pos]>=.4),('critical',abs(T[pos]-tc)<.0005)]:
            bad=np.vstack((np.zeros((1,R),int),np.cumsum(~inside,axis=0)))
            for lag in (1,5):
                mask=(bad[indices_sp[lag:]+1]-bad[indices_sp[:-lag]])==0
                x=Q[:-lag][mask];y=Q[lag:][mask]
                corr=np.nan
                if len(x)>20:
                    x=x-x.mean(0);y=y-y.mean(0);corr=np.sum(x*y)/np.sqrt(np.sum(x*x)*np.sum(y*y))
                memory.append(dict(**base,band=band,observable='Q_tensor',lag_sweeps=lag*1000,pairs=len(x),correlation=corr))
        # Sampled region episodes, including left/right boundary censoring.
        region=np.where(temp<tc-.0005,0,np.where(temp>tc+.0005,2,1))
        back=through=0
        for wi in range(R):
            changes=np.r_[0,np.flatnonzero(region[1:,wi]!=region[:-1,wi])+1,len(tprod)]
            sequence=region[changes[:-1],wi]
            for a0,b0,c0 in zip(sequence[:-2],sequence[1:-1],sequence[2:]):
                if b0==1 and a0!=1 and c0!=1:
                    back+=int(a0==c0);through+=int(a0!=c0)
            for lo,hi0 in zip(changes[:-1],changes[1:]):residences.append(dict(**base,walker=wi,region=int(region[lo,wi]),duration=(hi0-lo)*50,left_censored=lo==0,right_censored=hi0==len(tprod)))
        delta=np.diff(p.astype(float),axis=0);both=(delta[:-1]!=0)&(delta[1:]!=0)&(abs(temp[1:-1]-tc)<.0005)
        reverse=(delta[:-1]*delta[1:])<0
        M=(acceptance_chain(accept)**10).toarray()[:R,:R]
        incoming_below=np.triu(M,1).sum(0);incoming_above=np.tril(M,-1).sum(0)
        outgoing_below=np.tril(M,-1).sum(1);outgoing_above=np.triu(M,1).sum(1)
        band=abs(T-tc)<.0005
        null_num=(incoming_below*outgoing_below+incoming_above*outgoing_above)[band].sum()
        null_den=((incoming_below+incoming_above)*(outgoing_below+outgoing_above))[band].sum()
        summaries[-1].update(critical_returns=back,critical_through=through,coarse_reversal_fraction=float(reverse[both].mean()) if both.any() else np.nan,acceptance_null_reversal_fraction=null_num/null_den)
        print(path.stem,'analyzed',npass,'endpoint passages',flush=True)
    for name,rows in [('summary',summaries),('curves',curves),('statistics',stats),('native_statistics',native),('flow',flows),('survival',survival),('memory',memory),('endpoint_conditioned',conditioned),('time_blocks',series),('residence',residences),('cold_hot_cold',excursions)]:
        if rows:write(name+'.csv',rows)
    (OUT/'hot_energy_check.json').write_text(json.dumps(hot_energy,indent=2))
    # Long saved L96 histories provide a reference distribution, not extra pilots.
    refs=[]
    for g in (0,.01):
        tc=.38210530380170976 if g==0 else .3856721419201371;probes=np.array([.375,tc-.0003,tc,tc+.0003,.402,.405])
        with np.load(OLD/f'results/new_gd={g:g}_L=96.npz') as z:
            T=z['T'];b={k[6:]:interpolate(z[k].reshape(30,2,-1).mean(1),T,probes) for k in z.files if k.startswith('block_')}
        v,se=mean_curve_error(b,probes,96)
        for j,t in enumerate(probes):
            for k in v:refs.append(dict(g_d=g,T=t,observable=k,value=v[k][j],se=se[k][j]))
    write('long_reference.csv',refs)
    contrasts=[]
    for s in stats:
        if s['block_sweeps']!=5000:continue
        ref=next(r for r in refs if r['g_d']==s['g_d'] and r['T']==s['T'] and r['observable']==s['observable'])
        contrasts.append(dict(case=s['case'],ladder=s['ladder'],g_d=s['g_d'],initialization=s['initialization'],T=s['T'],observable=s['observable'],difference=s['value']-ref['value'],combined_se=np.hypot(s['se'],ref['se'])))
    write('reference_contrasts.csv',contrasts)
    pairs=[]
    for a in stats:
        if a['initialization']!='ordered' or a['block_sweeps']!=5000:continue
        b=next((r for r in stats if r['initialization']=='random' and r['block_sweeps']==5000 and r['ladder']==a['ladder'] and r['g_d']==a['g_d'] and r['T']==a['T'] and r['observable']==a['observable']),None)
        if b is not None:pairs.append(dict(ladder=a['ladder'],g_d=a['g_d'],T=a['T'],observable=a['observable'],ordered=a['value'],random=b['value'],difference=a['value']-b['value'],combined_se=np.hypot(a['se'],b['se'])))
    if pairs:write('initialization_contrasts.csv',pairs)

if __name__=='__main__':main()
