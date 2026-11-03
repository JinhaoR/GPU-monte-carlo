"""Read-only strong-SOC physical-walker audit and fixed-population sorting test.

No Monte Carlo. Outputs only beside this script. The sorting model conditions
on the observed number of high-Q configurations; it is NOT a phase equilibrium.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def fixed_count_prob(logw,M):
    """Exact one-site marginals of delta(sum b,M)*prod(exp(logw_i*b_i))."""
    R=len(logw);F=np.full((R+1,M+1),-np.inf);B=F.copy()
    F[0,0]=B[R,0]=0.
    for i in range(R):
        F[i+1,0]=0
        F[i+1,1:]=np.logaddexp(F[i,1:],F[i,:-1]+logw[i])
    for i in range(R-1,-1,-1):
        B[i,0]=0
        B[i,1:]=np.logaddexp(B[i+1,1:],B[i+1,:-1]+logw[i])
    p=np.array([np.exp(logw[i]+logsumexp(F[i,:M]+B[i+1,M-1::-1])-F[R,M]) for i in range(R)])
    assert abs(p.sum()-M)<1e-7
    return p


def selected_helicity(arr,T,N,mask=None):
    def avg(x):
        if mask is None:return x.mean(0)
        return np.divide((x*mask).sum(0),mask.sum(0),out=np.full(x.shape[1],np.nan),where=mask.sum(0)>0)
    return .5*sum((avg(arr['curvature_'+a])-(avg(arr['current_'+a]**2)-avg(arr['current_'+a])**2)/T)/N for a in ('x','y'))


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    summary=[];walkers=[];events=[];profiles=[];segments=[];endpoint_rows=[];validation=[];thresholds=[]
    fig,axs=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    fig2,bxs=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    old=list(csv.DictReader((ROOT/'exploration/he3_soc/results/strong_temperature.csv').open()))
    # Check exact constrained marginals against explicit enumeration on a toy.
    from itertools import combinations
    lw=np.array([-.9,.2,.7,1.1,-.4]);weights=[];labels=[]
    for ids in combinations(range(5),2):
        b=np.zeros(5);b[list(ids)]=1;labels.append(b);weights.append(np.exp(b@lw))
    expected=np.average(labels,axis=0,weights=weights)
    assert np.max(abs(fixed_count_prob(lw,2)-expected))<1e-13
    for col,L in enumerate((64,96,128,160)):
        print('Reading strong SOC',L,flush=True)
        N=L*L;path=ROOT/f'data/he3/P=20_gd=0.1_gaussian/L={L}.npz'
        with np.load(path) as z:
            meta=json.loads(z['manifest'].item())['runs'][str(L)]['metadata']
            T=np.asarray(meta['temperatures']);sw=np.asarray(meta['sample_sweeps']);burn=meta['thermalization_sweeps']
            pt=meta['parallel_tempering'];h=pt['walker_history'];hs=np.asarray(h['sweeps']);pos=np.asarray(h['temperature_indices'],dtype=int)
            assert np.array_equal(h['walker_ids'],np.arange(len(T)))
            assert np.all(np.sort(pos,axis=1)==np.arange(len(T)))
            q=z[f'L{L}_spin_alignment'];density=z[f'L{L}_density'];chi=z[f'L{L}_chirality']/density
            energy=z[f'L{L}_energy']/N
            high=q>.3;p=high.mean(0);inventory=high.sum(1)
            for cutoff in (.2,.3,.4):
                count=(q>cutoff).sum(1)
                thresholds.append(dict(L=L,cutoff=cutoff,initial=int(count[0]),final=int(count[-1]),
                    increases=int(np.sum(np.diff(count)>0)),decreases=int(np.sum(np.diff(count)<0))))
            prod=hs>burn;times=hs[prod];pidx=pos[prod];ix=np.searchsorted(sw,times)
            assert np.array_equal(sw[ix],times)
            wq=np.take_along_axis(q[ix],pidx,axis=1);wb=wq>.3
            same=np.all(wb==wb[0],axis=0);always_high=wb.all(0);always_low=(~wb).all(0)
            changed_t,changed_w=np.where(wb[1:]!=wb[:-1])
            inv_events=np.flatnonzero(np.diff(inventory)!=0)
            for ti,wi in zip(changed_t,changed_w):
                fine=inv_events[(sw[inv_events]>=times[ti])&(sw[inv_events+1]<=times[ti+1])]
                row=dict(L=L,walker=wi,history_before=int(times[ti]),history_after=int(times[ti+1]),
                    T_before=T[pidx[ti,wi]],T_after=T[pidx[ti+1,wi]],Q_before=wq[ti,wi],Q_after=wq[ti+1,wi],
                    direction='high_to_low' if wb[ti,wi] else 'low_to_high',
                    inventory_before=int(sw[fine[0]]) if len(fine)==1 else None,
                    inventory_after=int(sw[fine[0]+1]) if len(fine)==1 else None)
                events.append(row)
            # Each exact full-run endpoint counter is independent of sampled histories.
            passage_counts=[0,0]
            for w in range(len(T)):
                ep=np.flatnonzero((pos[:,w]==0)|(pos[:,w]==len(T)-1));last=None
                for i in ep:
                    side=int(pos[i,w]>0)
                    if last is not None and side!=last:
                        passage_counts[0 if last==0 else 1]+=1
                        endpoint_rows.append(dict(L=L,walker=w,sweep=int(hs[i]),direction='cold_to_hot' if last==0 else 'hot_to_cold'))
                    last=side
                acceptance=meta['sampling']['acceptance_rates'][w]
                walkers.append(dict(L=L,walker=w,cohort='always_high' if always_high[w] else 'always_low' if always_low[w] else 'converted',
                    high_fraction=float(wb[:,w].mean()),T_min=float(T[pidx[:,w].min()]),T_max=float(T[pidx[:,w].max()]),
                    index_span=int(np.ptp(pidx[:,w])),cold_visits=int((pidx[:,w]==0).sum()),hot_visits=int((pidx[:,w]==len(T)-1).sum()),
                    local_acceptance_including_burn=acceptance))
            front=int(np.argmin(abs(p-.5)));gap=N*(energy[high[:,front],front].mean()-energy[~high[:,front],front].mean())
            # A temperature-independent additive constant in log(Z_H/Z_L)
            # cancels when M is fixed. A constant physical energy offset does not.
            logw=-gap*(1/T-1/T[front]);pred=np.zeros(len(T))
            values,counts=np.unique(inventory,return_counts=True)
            for M,count in zip(values,counts):pred+=count/len(sw)*fixed_count_prob(logw,int(M))
            step=sum(count/len(sw)*(np.arange(len(T))<M) for M,count in zip(values,counts))
            half=len(sw)//2;hfirst=high[:half,front];e_first=energy[:half,front]
            gap_first=N*(e_first[hfirst].mean()-e_first[~hfirst].mean())
            pred_second=np.zeros(len(T));v2,c2=np.unique(inventory[half:],return_counts=True)
            for M,count in zip(v2,c2):
                pred_second+=count/(len(sw)-half)*fixed_count_prob(-gap_first*(1/T-1/T[front]),int(M))
            width=lambda curve:float(np.interp(.1,curve[::-1],T[::-1])-np.interp(.9,curve[::-1],T[::-1]))
            cuts=np.r_[0,inv_events+1,len(sw)]
            for left,right in zip(cuts[:-1],cuts[1:]):
                pp=high[left:right].mean(0);M=int(inventory[left]);pj=fixed_count_prob(logw,M)
                segments.append(dict(L=L,first_sweep=int(sw[left]),last_sweep=int(sw[right-1]),samples=int(right-left),
                    inventory=M,observed_front_T=float(np.interp(.5,pp[::-1],T[::-1])),
                    predicted_front_T=float(np.interp(.5,pj[::-1],T[::-1])),mean_abs_probability_error=float(abs(pp-pj).mean()),
                    max_probability_error=float(abs(pp-pj).max())))
            # Raw recheck of the earlier branch table, including helicity.
            raw={k:z[f'L{L}_{k}'] for k in ('current_x','current_y','curvature_x','curvature_y')}
            y=selected_helicity(raw,T,N);yl=selected_helicity(raw,T,N,~high);yh=selected_helicity(raw,T,N,high)
            hm=np.stack([raw['current_x'][:,-1],raw['current_y'][:,-1],raw['current_x'][:,-1]**2,
                raw['current_y'][:,-1]**2,raw['curvature_x'][:,-1],raw['curvature_y'][:,-1]],axis=-1).reshape(30,2000,6).mean(1)
            hj=(30*hm.mean(0)-hm)/29
            hy=(hj[:,4]+hj[:,5]-(hj[:,2]+hj[:,3]-hj[:,0]**2-hj[:,1]**2)/T[-1])/(2*N)
            hot_se=np.sqrt(29*hy.var())
            prev=sorted([r for r in old if int(r['L'])==L],key=lambda x:float(x['T']))
            assert np.array_equal(T,np.array([float(x['T']) for x in prev]))
            errs={name:float(np.max(abs(np.array([float(x[name]) for x in prev])-v))) for name,v in
                  [('high_fraction',p),('Qzz',q.mean(0)),('abs_chi',abs(chi).mean(0)),('helicity',y),('energy',energy.mean(0))]}
            assert max(errs.values())<1e-8
            validation.append(dict(L=L,**errs))
            rates=np.asarray(pt['acceptance_rates']);chi_match=abs(chi)>.1
            evar=energy.var(0);cv=N*evar/T**2;jheat=int(cv.argmax())
            fixed_changes=int(np.count_nonzero(high[ix][1:]!=high[ix][:-1]))
            row=dict(L=L,Tmin=T[0],Tmax=T[-1],replicas=len(T),burn=burn,production=meta['measurement_sweeps'],
                exact_loops=int(sum(pt['walker_round_trips'])),sampled_cold_to_hot=passage_counts[0],sampled_hot_to_cold=passage_counts[1],
                high_initial=int(inventory[0]),high_final=int(inventory[-1]),always_high=int(always_high.sum()),always_low=int(always_low.sum()),
                physical_high_to_low=int(np.sum(wb[changed_t,changed_w])),physical_low_to_high=int(np.sum(~wb[changed_t,changed_w])),
                fixed_temperature_label_changes_at_history_cadence=fixed_changes,
                inventory_up_events=int(np.sum(np.diff(inventory)>0)),inventory_down_events=int(np.sum(np.diff(inventory)<0)),
                chirality_classifier_disagreements=int(np.count_nonzero(high!=chi_match)),
                hot_high_records=int(high[:,-1].sum()),cold_low_records=int((~high[:,0]).sum()),
                hot_Y=y[-1],hot_Y_se_100k=hot_se,hot_Q=q[:,-1].mean(),hot_abs_chi=abs(chi[:,-1]).mean(),hot_Cv=cv[-1],heat_peak_T=T[jheat],heat_peak_Cv=cv[jheat],
                mean_swap=float(rates.mean()),min_swap=float(rates.min()),minimum_swap_T=float((T[:-1]+T[1:])[rates.argmin()]/2),
                front_T=T[front],front_gap_total=gap,front_high_Y=yh[front],front_low_Y=yl[front],
                sorting_mean_probability_error=float(abs(p-pred).mean()),sorting_max_probability_error=float(abs(p-pred).max()),
                sorting_front_T=float(np.interp(.5,pred[::-1],T[::-1])),
                sharp_step_max_probability_error=float(abs(p-step).max()),
                observed_10_90_width=width(p),sorting_10_90_width=width(pred),
                first_half_gap=gap_first,held_out_second_half_max_error=float(abs(high[half:].mean(0)-pred_second).max()))
            summary.append(row)
            for j,t in enumerate(T):
                profiles.append(dict(L=L,T=t,high_fraction=p[j],sorting_prediction=pred[j],sharp_step_prediction=step[j],
                    second_half_high_fraction=high[half:,j].mean(),held_out_second_half_prediction=pred_second[j],
                    Q=q[:,j].mean(),abs_chi=abs(chi[:,j]).mean(),
                    Y=y[j],Y_high=yh[j],Y_low=yl[j],Cv=cv[j],energy=energy[:,j].mean()))
            for cohort,mask in [('high',always_high),('low',always_low)]:
                counts=np.bincount(pidx[:,mask].ravel(),minlength=len(T))
                axs[0,col].plot(T,counts/counts.sum(),label=cohort)
            axs[0,col].set(title=f'L={L}: unchanged walkers',xlabel='T',ylabel='normalized occupation');axs[0,col].legend()
            axs[1,col].plot(T,p,label='observed');axs[1,col].plot(T,pred,'--',label='fixed-inventory prediction')
            axs[1,col].set(xlim=(T[max(front-8,0)],T[min(front+8,len(T)-1)]),xlabel='T',ylabel='high-Q fraction');axs[1,col].legend(fontsize=8)
            bxs[0,col].step((sw[::10]-burn)/1e6,inventory[::10],where='post');bxs[0,col].set(title=f'L={L}',xlabel='production / M sweeps',ylabel='high-Q inventory')
            for w in np.unique(changed_w):
                bxs[1,col].plot((times-burn)/1e6,T[pidx[:,w]],linewidth=.5,label=f'walker {w}')
                jj=np.flatnonzero(np.diff(wb[:,w].astype(int))!=0)+1
                bxs[1,col].scatter((times[jj]-burn)/1e6,T[pidx[jj,w]],s=25,c='black')
            bxs[1,col].set(xlabel='production / M sweeps',ylabel='physical walker T');bxs[1,col].legend(fontsize=7)
            np.savez_compressed(OUT/f'traces_L={L}.npz',T=T,sweeps=times,walker_temperature=pidx,walker_high=wb,
                inventory_sweeps=sw[::10],inventory=inventory[::10],acceptance=rates)
        print(row,flush=True)
    write('summary.csv',summary);write('walkers.csv',walkers);write('events.csv',events);write('temperature_profiles.csv',profiles)
    write('inventory_segments.csv',segments);write('endpoint_passages.csv',endpoint_rows);write('raw_validation.csv',validation)
    write('threshold_checks.csv',thresholds)
    fig.savefig(OUT/'sorting_and_occupation.png',dpi=180);plt.close(fig)
    fig2.savefig(OUT/'physical_conversions.png',dpi=180);plt.close(fig2)


if __name__=='__main__':main()
