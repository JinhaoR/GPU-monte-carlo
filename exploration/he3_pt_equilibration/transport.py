"""Endpoint censoring, measured flow/diffusion, and acceptance-only null models."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv
import numpy as np
from scipy.sparse import coo_matrix,eye
from scipy.sparse.linalg import spsolve
from extract import OUT,write

def acceptance_chain(a):
    """State=(next exchange parity, temperature index); independent acceptances."""
    R=len(a)+1;rows=[];cols=[];vals=[]
    for parity in (0,1):
        for i in range(R):
            j=i+1 if i%2==parity else i-1;s=parity*R+i
            if 0<=j<R:
                p=a[min(i,j)];rows.extend((s,s));cols.extend(((1-parity)*R+j,(1-parity)*R+i));vals.extend((p,1-p))
            else:rows.append(s);cols.append((1-parity)*R+i);vals.append(1.)
    return coo_matrix((vals,(rows,cols)),shape=(2*R,2*R)).tocsr()

def hitting(P,targets):
    keep=np.ones(P.shape[0],bool);keep[targets]=False;out=np.zeros(P.shape[0]);Q=P[keep][:,keep]
    out[keep]=spsolve(eye(int(keep.sum()),format='csr')-Q,np.ones(keep.sum()))
    return out

def null_transport(a,exchange):
    R=len(a)+1;P=acceptance_chain(a)
    cold_to_hot=hitting(P,[R-1,2*R-1])[R]
    hot_to_cold=hitting(P,[0,R])[((R-1)%2)*R+R-1]
    symmetric=2*exchange*R*np.sum(1/np.asarray(a))
    return P,(cold_to_hot+hot_to_cold)*exchange,symmetric

def km(records):
    """Descriptive KM; repeated walkers/finite starts preclude iid confidence bars."""
    if not records:return [],np.nan
    d=np.array([r[0] for r in records]);event=np.array([r[1] for r in records]);S=1.;rows=[(0.,1.,len(d),0)]
    for t in np.unique(d):
        risk=np.sum(d>=t);n=np.sum((d==t)&event);S*=1-n/risk;rows.append((float(t),S,int(risk),int(n)))
    median=next((t for t,s,_,_ in rows if s<=.5),np.nan)
    return rows,median

def events(pos,t):
    R=pos.shape[1];loops=[];passages=[];first=[];walkers=[];labels=np.full(pos.shape,-1,dtype=np.int8)
    for w in range(R):
        p=pos[:,w];hits=np.flatnonzero((p==0)|(p==R-1));both=[]
        for end in (0,R-1):
            z=np.flatnonzero(p==end);both.append(int(z[0]) if len(z) else None)
        both_time=float(t[max(both)]-t[0]) if all(v is not None for v in both) else float(t[-1]-t[0])
        count=0
        if len(hits):
            past=np.searchsorted(hits,np.arange(len(p)),side='right')-1;valid=past>=0
            labels[valid,w]=(p[hits[past[valid]]]>0).astype(np.int8)
            start=hits[0];origin=p[start];other=False;pass_start=start;pass_origin=origin;first_done=False
            for k in hits[1:]:
                if p[k]!=pass_origin:
                    passages.append((float(t[k]-t[pass_start]),True,w,int(pass_origin>0),float(t[pass_start]),float(t[k])))
                    pass_start=k;pass_origin=p[k]
                if p[k]!=origin:other=True
                elif other:
                    rec=(float(t[k]-t[start]),True,w,int(origin>0),float(t[start]),float(t[k]));loops.append(rec)
                    if not first_done:first.append(rec);first_done=True
                    count+=1;start=k;other=False
            tail=(float(t[-1]-t[start]),False,w,int(origin>0),float(t[start]),float(t[-1]))
            loops.append(tail)
            if not first_done:first.append(tail)
            passages.append((float(t[-1]-t[pass_start]),False,w,int(pass_origin>0),float(t[pass_start]),float(t[-1])))
        walkers.append(dict(walker=w,started=bool(len(hits)),both=all(v is not None for v in both),both_time=both_time,loops=count,minimum=int(p.min()),maximum=int(p.max())))
    return loops,passages,first,walkers,labels

def main():
    inventory=list(csv.DictReader((OUT/'inventory.csv').open()));summary=[];flowrows=[];diffusion=[];survival=[];evrows=[];wr=[];residence=[]
    for m in inventory:
        tag=m['tag'];L=int(m['L']);gd=float(m['g_d']);campaign=m['campaign'];base=dict(campaign=campaign,g_d=gd,L=L)
        with np.load(OUT/(tag+'.npz')) as f:d=dict(f)
        T=d['T'];R=len(T);burn=int(m['burn']);sel=d['sweeps']>=burn;pos=d['positions'][sel];t=d['sweeps'][sel];dt=int(t[1]-t[0]);P,null,symmetric=null_transport(d['acceptance'],int(m['exchange_interval']))
        loops,passages,first,walkers,labels=events(pos,t)
        full_loops,full_passages,_,_,full_labels=events(d['positions'],d['sweeps']);full_labels=full_labels[sel]
        completed_carry=[r[5] for r in full_loops if r[1] and r[5]>burn]
        passage_carry=[r[5] for r in full_passages if r[1] and r[5]>burn]
        passage_counts=np.histogram(passage_carry,np.linspace(burn,t[-1],7))[0]
        duration=t[-1]-burn;flux_time=2*R*duration/len(passage_carry)
        flux_jk=2*R*(duration*5/6)/(len(passage_carry)-passage_counts)
        atT=np.take_along_axis(full_labels,np.argsort(pos,axis=1),axis=1);h0=(atT==0).sum(0);h1=(atT==1).sum(0)
        flow=np.divide(h0,h0+h1,out=np.full(R,np.nan),where=h0+h1>0)
        reset=np.take_along_axis(labels,np.argsort(pos,axis=1),axis=1);a=(reset==0).sum(0);b=(reset==1).sum(0);resetflow=np.divide(a,a+b,out=np.full(R,np.nan),where=a+b>0)
        drops=flow[:-1]-flow[1:];j=int(np.nanargmax(drops));center=(T[j]+T[j+1])/2;critical=abs((T[:-1]+T[1:])/2-float(m['Cv_peak_T']))<.001
        firstsurv,median=km(first);passsurv,pmedian=km(passages)
        for label,recs in [('first_loop',firstsurv),('passage',passsurv)]:
            for tt,s,risk,nev in recs:survival.append(dict(**base,kind=label,time=tt,survival=s,risk=risk,events=nev))
        completed=[r[0] for r in loops if r[1]]
        for kind,recs in [('loop',loops),('passage',passages)]:
            for dur,event,w,origin,start,end in recs:evrows.append(dict(**base,kind=kind,walker=w,duration=dur,completed=event,origin=origin,start=start,end=end))
        for row in walkers:wr.append(dict(**base,**row))
        # Compare common 500-sweep recording to distinguish sampling cadence effects.
        stride=500//dt;_,_,_,common,_=events(pos[::stride],t[::stride])
        xp=pos[::stride];counts=np.bincount((R*xp[:-1].astype(np.int64)+xp[1:]).ravel(),minlength=R*R).reshape(R,R);Pc=counts/counts.sum(1,keepdims=True)
        from scipy.sparse import csr_matrix
        coarse=(hitting(csr_matrix(Pc),[R-1])[0]+hitting(csr_matrix(Pc),[0])[-1])*500
        for lag_sweeps in (500,2000,10000,100000):
            lag=lag_sweeps//dt;idx=np.arange(0,len(pos)-lag,max(1,500//dt));p=pos[idx];delta=pos[idx+lag].astype(float)-p
            avg=np.bincount(p.ravel(),weights=delta.ravel(),minlength=R)/len(idx);v=np.bincount(p.ravel(),weights=(delta*delta).ravel(),minlength=R)/len(idx)-avg*avg
            a=np.tile(np.arange(R,dtype=float),2);b=a*a
            for _ in range(lag_sweeps//int(m['exchange_interval'])):a=P@a;b=P@b
            vn=b[:R]-a[:R]**2
            for k in range(R):diffusion.append(dict(**base,T=T[k],lag=lag_sweeps,D_index=v[k]/(2*lag_sweeps),D_null=vn[k]/(2*lag_sweeps),drift=avg[k]/lag_sweeps))
        for k in range(R):flowrows.append(dict(**base,T=T[k],flow=flow[k],production_reset_flow=resetflow[k],labelled_fraction=(h0[k]+h1[k])/len(t)))
        # Continuous sampled occupancy episodes in broad regions; boundaries censored.
        tc=float(m['Cv_peak_T']);region=np.where(T[pos]<tc-.0003,0,np.where(T[pos]>tc+.0003,2,1))
        for w in range(R):
            changes=np.r_[0,np.where(region[1:,w]!=region[:-1,w])[0]+1,len(t)]
            for lo,hi in zip(changes[:-1],changes[1:]):
                residence.append(dict(**base,walker=w,region=int(region[lo,w]),duration=(hi-lo)*dt,left_censored=lo==0,right_censored=hi==len(t)))
        row=dict(**base,R=R,history_dt=dt,completed_production_loops=len(completed),walkers_with_loop=sum(w['loops']>0 for w in walkers),walkers_both=sum(w['both'] for w in walkers),walkers_never_endpoint=sum(not w['started'] for w in walkers),
            completed_production_loop_mean=float(np.mean(completed)) if completed else np.nan,completed_loop_median=float(np.median(completed)) if completed else np.nan,
            first_loop_KM_median=median,passage_KM_median=pmedian,first_loop_censored=sum(not r[1] for r in first),both_restricted_mean_3M=np.mean([w['both_time'] for w in walkers]),
            loops_at_common_500=sum(w['loops'] for w in common),both_at_common_500=sum(w['both'] for w in common),null_alternating_RT=null,null_symmetric_RT=symmetric,coarse500_markov_RT=coarse,
            empirical_completed_to_null=float(m['completed_trip_mean'])/null,flow_max_drop=float(drops[j]),flow_drop_T=center,flow_drop_in_002=float(np.nansum(drops[critical])),labelled_fraction_min=float(np.nanmin((h0+h1)/len(t))),
            sampled_full_loops=sum(r[1] for r in full_loops),carry_production_loops=len(completed_carry),carry_production_passages=len(passage_carry),passage_flux_time=flux_time,passage_flux_time_block_se=np.sqrt(5*np.var(flux_jk)),flux_to_null=flux_time/null)
        summary.append(row);print(tag,'null',round(null),'actual completed',m['completed_trip_mean'],'prod loops',len(completed),'both',row['walkers_both'],flush=True)
        c0=(atT[1:]==0).reshape(30,-1,R).sum(1);c1=(atT[1:]==1).reshape(30,-1,R).sum(1)
        np.savez_compressed(OUT/(tag+'_transport.npz'),T=T,flow=flow,resetflow=resetflow,coarse_transition=Pc,cold_block_counts=c0,hot_block_counts=c1)
    for name,rows in [('transport_summary',summary),('flow',flowrows),('diffusion',diffusion),('survival',survival),('endpoint_events',evrows),('walkers',wr),('residence',residence)]:write(name+'.csv',rows)
    # Exact elementary checks of transition schedule and censoring.
    for R in (4,8,20):
        P,rt,_=null_transport(np.ones(R-1),5);assert abs(rt-2*R*5)<1e-8
        np.testing.assert_allclose(np.asarray(P.sum(0)),1);np.testing.assert_allclose(np.asarray(P.sum(1)),1)
    (OUT/'transport_validation.json').write_text(json.dumps(dict(all_accept_ballistic_limit=True,row_and_column_stochastic=True),indent=2))

if __name__=='__main__':main()
