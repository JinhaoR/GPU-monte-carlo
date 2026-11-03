"""Fixed-temperature errors versus evolution of the same physical walkers."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,numpy as np
from extract import OUT,write

def evaluate(b,T,L):
    return dict(energy=b['e'],abs_chirality=b['abs_m'],m2=b['m2'],P2=b['p2'],spin=b['qzz'],
        Binder=1-b['m4']/(3*b['m2']**2),Y=(b['kx']+b['ky']-((b['ix2']-b['ix']**2)+(b['iy2']-b['iy']**2))/T)/(2*L*L))

def block_estimate(blocks,T,L,factor=2,portion=slice(None)):
    b={k:v[portion].reshape((-1,factor)+v.shape[1:]).mean(1) for k,v in blocks.items()};n=len(b['e']);mean={k:v.mean(0) for k,v in b.items()}
    v=evaluate(mean,T,L);jk=evaluate({k:(n*mean[k]-v)/(n-1) for k,v in b.items()},T,L)
    return v,{k:np.sqrt((n-1)*np.var(v,axis=0)) for k,v in jk.items()}

def main():
    inv=list(csv.DictReader((OUT/'inventory.csv').open()));errors=[];memory=[];excursions=[];hot=[];curves=[]
    for m in inv:
        tag=m['tag'];L=int(m['L']);gd=float(m['g_d']);base=dict(campaign=m['campaign'],g_d=gd,L=L)
        with np.load(OUT/(tag+'.npz')) as z:d=dict(z)
        T=d['T'];R=len(T);blocks={k[6:]:v for k,v in d.items() if k.startswith('block_')}
        tc=float(m['Cv_peak_T']);chosen=np.unique(np.r_[0,R-1,[np.argmin(abs(T-t)) for t in (tc-.0003,tc,tc+.0003,.40,.41)]])
        first,fe=block_estimate(blocks,T,L,2,slice(0,30));second,se=block_estimate(blocks,T,L,2,slice(30,None))
        for factor in (2,5,10):
            v,e=block_estimate(blocks,T,L,factor)
            for j in chosen:
                for key in v:errors.append(dict(**base,T=T[j],observable=key,block_sweeps=factor*50000,value=v[key][j],se=e[key][j],first_half=first[key][j],second_half=second[key][j],half_shift=second[key][j]-first[key][j],half_combined_se=np.hypot(fe[key][j],se[key][j])))
            if factor==2:
                for j,t in enumerate(T):
                    row=dict(**base,T=t)
                    for key in v:row[key]=v[key][j];row[key+'_se']=e[key][j]
                    curves.append(row)
        # Pooled physical-walker correlations: subtract only a common mean,
        # retaining differences between walker means rather than hiding freezing.
        keys=('energy_residual','abs_m_residual','P2_residual','qzz_residual')
        for key in keys:
            a=d['walker_'+key].astype(float);a-=a.mean();var=np.mean(a*a)
            for lag_sweeps in (500,1000,5000,10000,50000,100000,500000,1000000):
                lag=lag_sweeps//500;corr=np.mean(a[:-lag]*a[lag:])/var
                memory.append(dict(**base,observable=key,lag=lag_sweeps,pooled_correlation=corr,walker_mean_variance_fraction=np.var(a.mean(0))/var))
        # Conditional memory during continuously sampled hot-band visits.
        p=d['physical_positions'];ps=d['physical_sweeps'];temp=T[p]
        for threshold in (.40,.41):
            bad=np.vstack((np.zeros((1,R),int),np.cumsum(temp<threshold,axis=0)))
            for lag_sweeps in (500,1000,5000):
                lag=lag_sweeps//500;keep=(bad[lag+1:]-bad[:-lag-1])==0
                for key in ('chirality','qzz_residual','P2_residual'):
                    a=d['walker_'+key].astype(float);x=a[:-lag][keep];y=a[lag:][keep]
                    corr=np.corrcoef(x,y)[0,1] if len(x)>10 else np.nan
                    hot.append(dict(**base,hot_threshold=threshold,observable=key,lag=lag_sweeps,pairs=len(x),correlation=corr))
        # Cold -> hot -> cold excursions on the SAME walker, at common 1000 cadence.
        use=(ps%1000)==0;ps=ps[use];p=p[use];temp=T[p]
        chi=d['walker_chirality'][use];q=d['walker_qzz_residual'][use]
        Q=d.get('Q_walkers');cold=.379 if gd==0 else .383
        for hot_threshold in (.40,.41):
            for w in range(R):
                lastcold=None;before=None;reached=False;hot_steps=0;firsthot=None
                for k in range(len(ps)):
                    if temp[k,w]<=cold:
                        if reached and before is not None:
                            if min(abs(chi[before,w]),abs(chi[k,w]))>.1:
                                qdot=np.nan
                                if Q is not None:
                                    x,y=Q[before,w],Q[k,w];den=np.linalg.norm(x)*np.linalg.norm(y)
                                    if den>0:qdot=float(np.dot(x,y)/den)
                                excursions.append(dict(**base,walker=w,hot_threshold=hot_threshold,start=int(ps[before]),first_hot=int(ps[firsthot]),end=int(ps[k]),sampled_hot_sweeps=hot_steps*1000,
                                    sign_before=int(chi[before,w]>0),sign_after=int(chi[k,w]>0),opposite=bool(chi[before,w]*chi[k,w]<0),q_before=float(q[before,w]),q_after=float(q[k,w]),Q_normalized_dot=qdot))
                            reached=False;hot_steps=0;firsthot=None;before=None
                        lastcold=k
                    if temp[k,w]>=hot_threshold:
                        if not reached and lastcold is not None:before=lastcold;reached=True;firsthot=k
                        if reached:hot_steps+=1
        print(tag,'physical diagnostics',flush=True)
    for name,rows in [('blocking_and_drift',errors),('walker_memory',memory),('hot_memory',hot),('cold_hot_cold',excursions),('scalar_curves',curves)]:write(name+'.csv',rows)
    # Summarize whole excursions and the first eligible one per physical walker.
    rows=[]
    for m in inv:
        for threshold in (.40,.41):
            r=[x for x in excursions if x['campaign']==m['campaign'] and x['g_d']==float(m['g_d']) and x['L']==int(m['L']) and x['hot_threshold']==threshold]
            for onlyfirst in (False,True):
                if onlyfirst:
                    seen=set();rr=[]
                    for x in r:
                        if x['walker'] not in seen:rr.append(x);seen.add(x['walker'])
                else:rr=r
                a=np.array([x['sign_before'] for x in rr]);b=np.array([x['sign_after'] for x in rr]);qa=np.array([x['q_before'] for x in rr]);qb=np.array([x['q_after'] for x in rr]);qd=np.array([x['Q_normalized_dot'] for x in rr])
                rows.append(dict(campaign=m['campaign'],g_d=float(m['g_d']),L=int(m['L']),hot_threshold=threshold,first_per_walker=onlyfirst,events=len(rr),walkers=len(set(x['walker'] for x in rr)),
                    opposite_fraction=np.mean(a!=b) if len(a) else np.nan,independent_marginal_opposite=a.mean()*(1-b.mean())+(1-a.mean())*b.mean() if len(a) else np.nan,
                    spin_residual_correlation=np.corrcoef(qa,qb)[0,1] if len(a)>3 else np.nan,Q_dot_mean=np.nanmean(qd) if np.isfinite(qd).any() else np.nan,median_hot_exposure=np.median([x['sampled_hot_sweeps'] for x in rr]) if len(rr) else np.nan))
    write('excursion_summary.csv',rows)

if __name__=='__main__':main()
