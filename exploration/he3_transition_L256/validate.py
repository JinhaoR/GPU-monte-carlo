"""Bounded numerical controls and empirical goodness-of-fit diagnostics."""
import json,numpy as np
from analyze import OUT,SIZES,PAIRS,KEYS,load_blocks,raw_observables,wham_curve,wm,crossing,write

def main():
    with np.load(OUT/'main.npz') as f:z=dict(f)
    grid=z['critical_T'];sig=z['critical_boot'][:,:,1].std(1,ddof=1)
    variants={v:[] for v in ('800','400','trimmed')};held=[];overlap=[]
    for i,L in enumerate(SIZES):
        d=load_blocks(L);H=d['critical_H'].sum(0);S=d['critical_S'].sum(0);src=d['critical_src'];e=d['critical_energy'];beta=d['critical_beta'];temps=d['critical_T']
        for variant in variants:
            if variant=='400':hh=H.reshape(len(beta),400,2).sum(-1);ss=S.reshape(8,400,2).sum(-1);ee=e.reshape(400,2).mean(-1);bb=beta
            elif variant=='trimmed':hh=H[2:-2];ss=src[2:-2].sum(0);ee=e;bb=beta[2:-2]
            else:hh,ss,ee,bb=H,S,e,beta
            v,f=wham_curve(hh,ss,bb,ee,grid,L);variants[variant].append(v)
        pdf=H/H.sum(1,keepdims=True)
        for j in range(len(beta)-1):overlap.append(dict(L=int(L),T1=temps[j],T2=temps[j+1],histogram_overlap=np.minimum(pdf[j],pdf[j+1]).sum()))
        for j in (len(beta)//2-2,len(beta)//2-1,len(beta)//2,len(beta)//2+1):
            keep=np.arange(len(beta))!=j;v,_=wham_curve(H[keep],S-src[j],beta[keep],e,np.array([temps[j]]),L)
            k=np.argmin(abs(d['T']-temps[j]));mm=np.stack([d[key][:,k] for key in KEYS],axis=1);direct=raw_observables(mm.mean(0)[:,None],np.array([temps[j]]),L)[:,0]
            b=mm.reshape(30,2,8).mean(1);jk=raw_observables(((30*b.mean(0)-b)/29)[...,None],np.array([temps[j]]),L)[...,0];err=np.sqrt(29*np.var(jk,axis=0))
            held.append(dict(L=int(L),T=temps[j],U_direct=direct[0],U_heldout=v[0,0],U_direct_se=err[0],Y_direct=direct[1],Y_heldout=v[1,0],Y_direct_se=err[1]))
    controls=[]
    for variant,v in variants.items():
        v=np.array(v)
        for a,b in PAIRS:
            tc,nr=crossing(grid,v[a,0]-v[b,0],.38118,.38248,.38183)
            controls.append(dict(variant=variant,quantity=f'Binder_{SIZES[a]}_{SIZES[b]}',T=tc,chi2=np.nan))
        for ids in (np.arange(4),np.array([1,2,3])):
            fit=wm(grid,v[ids,1],sig[ids],.5,SIZES[ids])
            controls.append(dict(variant=variant,quantity='WM_'+str(tuple(SIZES[ids])),T=fit[0],chi2=fit[2]))
    write('numerical_controls.csv',controls);write('withheld_temperatures.csv',held);write('energy_overlap.csv',overlap)
    # Null-centered joint bootstrap: shift each curve to obey fitted WM at Tc,
    # preserve its sampled temperature dependence, then refit all parameters.
    goodness=[]
    for ids in (np.arange(4),np.array([1,2,3])):
        yy=z['critical'][ids,1];by=z['critical_boot'][ids,:,1];ss=sig[ids];sizes=SIZES[ids]
        fit=wm(grid,yy,ss,.5,sizes);t,C,observed,_=fit
        mismatch=np.array([np.interp(t,grid,v) for v in yy])-2*t/(np.pi*.25)*(1+1/(2*np.log(sizes)+C))
        scores=[wm(grid,by[:,b]-mismatch[:,None],ss,.5,sizes,fit)[2] for b in range(by.shape[1])]
        goodness.append(dict(sizes='_'.join(map(str,sizes)),observed_chi2=observed,null_bootstrap_p=(1+sum(np.array(scores)>=observed))/(1+len(scores)),replicates=len(scores)))
    write('empirical_fit_quality.csv',goodness)
    # Separate changes in the interpolated means from the information gain of
    # reweighting when comparing goodness of fit with direct-grid estimates.
    rawy=np.array([np.interp(grid,z['T'],v) for v in z['raw'][:,1]])
    rawse=np.array([np.interp(grid,z['T'],v) for v in z['bootraw'][:,:,1].std(1,ddof=1)])
    interpolation=[]
    for yn,yy in [('WHAM',z['critical'][:,1]),('raw_linear',rawy)]:
        for sn,ss in [('WHAM',sig),('raw_linear',rawse)]:
            fit=wm(grid,yy,ss,.5,SIZES)
            interpolation.append(dict(mean_curve=yn,error_curve=sn,T=fit[0],C=fit[1],chi2=fit[2],dof=2))
    write('interpolation_error_controls.csv',interpolation)
    # Confirm imported estimator and subset-size normalization on known curves.
    T=np.linspace(.37,.4,101);m=np.zeros((8,len(T)));m[:2]=1;m[4]=m[7]=4*256**2
    np.testing.assert_allclose(raw_observables(m,T,256),np.array([np.full(len(T),2/3),np.full(len(T),4)]))
    for sizes in (SIZES,SIZES[1:]):
        for q in (.5,1.):
            y=np.array([2*T/(np.pi*q*q)*(1+1/(2*np.log(L)+1))-20*(T-.383) for L in sizes])
            fit=wm(T,y,np.full_like(y,.001),q,sizes);assert abs(fit[0]-.383)<1e-8
    (OUT/'validation.json').write_text(json.dumps(dict(known_Binder_helicity=True,synthetic_WM_both_charges_and_size_sets=True),indent=2))
    print('Numerical controls and empirical fit checks passed',flush=True)

if __name__=='__main__':main()
