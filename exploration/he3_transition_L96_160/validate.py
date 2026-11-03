"""Estimator, histogram binning, source-window and withheld-temperature checks."""
import json
import numpy as np
from analyze import OUT,SIZES,KEYS,raw_observables,wham_curve,wm,crossing,write


def main():
    T=np.linspace(.37,.4,101);m=np.zeros((8,len(T)));m[0:2]=1;m[4]=m[7]=4*96**2
    np.testing.assert_allclose(raw_observables(m,T,96),np.array([np.full(len(T),2/3),np.full(len(T),4)]))
    batch=raw_observables(np.stack([m,m]),T,96);assert batch.shape==(2,2,len(T))
    for q in (.5,1.):
        y=np.array([2*T/(np.pi*q*q)*(1+1/(2*np.log(L)+1))-20*(T-.383) for L in SIZES])
        fit=wm(T,y,np.full_like(y,.001),q);assert abs(fit[0]-.383)<1e-8
    comparisons=[];held=[]
    for g in (0,.01):
        with np.load(OUT/f'main_gd={g:g}.npz') as z:reference=dict(z)
        for name in ('critical','integer'):
            grids=reference[name+'_T'];se=reference[name+'_boot'][:,:,1].std(1,ddof=1)
            by_variant={v:[] for v in ('800','400','trimmed')}
            ranges=[]
            for i,L in enumerate(SIZES):
                with np.load(OUT/f'gd={g:g}_L={L}.npz') as z:d=dict(z)
                H=d[name+'_H'].sum(0);S=d[name+'_S'].sum(0);src=d[name+'_src'];e=d[name+'_energy'];beta=d[name+'_beta'];temps=d[name+'_T']
                keep=np.arange(2,len(beta)-2);ranges.append((temps[2],temps[-3]))
                for variant in by_variant:
                    if variant=='400':hh=H.reshape(len(beta),400,2).sum(-1);ss=S.reshape(8,400,2).sum(-1);ee=e.reshape(400,2).mean(-1);bb=beta
                    elif variant=='trimmed':hh=H[keep];ss=src[keep].sum(0);ee=e;bb=beta[keep]
                    else:hh,ss,ee,bb=H,S,e,beta
                    v,_=wham_curve(hh,ss,bb,ee,grids,L);by_variant[variant].append(v)
                for j in (len(beta)//2-1,len(beta)//2,len(beta)//2+1):
                    keepj=np.arange(len(beta))!=j
                    v,_=wham_curve(H[keepj],S-src[j],beta[keepj],e,np.array([temps[j]]),L)
                    k=np.argmin(abs(d['T']-temps[j]));mm=np.stack([d[key][:,k] for key in KEYS],axis=1)
                    direct=raw_observables(mm.mean(0)[:,None],np.array([temps[j]]),L)[:,0]
                    blocks=mm.reshape(30,2,8).mean(1);jk=raw_observables(((30*blocks.mean(0)-blocks)/29)[...,None],np.array([temps[j]]),L)[...,0]
                    err=np.sqrt(29*np.var(jk,axis=0))
                    held.append(dict(g_d=g,L=int(L),window=name,T=temps[j],U_direct=direct[0],U_heldout=v[0,0],
                        U_difference=v[0,0]-direct[0],U_direct_se=err[0],Y_direct=direct[1],Y_heldout=v[1,0],Y_difference=v[1,0]-direct[1],Y_direct_se=err[1]))
            mask=(grids>=max(x[0] for x in ranges))&(grids<=min(x[1] for x in ranges))
            for variant,curves in by_variant.items():
                curves=np.array(curves);fit=wm(grids[mask],curves[:,1,mask],se[:,mask],.5 if name=='critical' else 1.)
                comparisons.append(dict(g_d=g,window=name,variant=variant,quantity='WM',T=fit[0],chi2=fit[2]))
                if name=='critical':
                    center=.38183 if g==0 else .38558
                    for a,b in ((0,1),(1,2),(0,2)):
                        t,_=crossing(grids,curves[a,0]-curves[b,0],center-.00065,center+.00065,center)
                        comparisons.append(dict(g_d=g,window=name,variant=variant,quantity=f'Binder_{SIZES[a]}_{SIZES[b]}',T=t,chi2=np.nan))
    write('systematics.csv',comparisons);write('withheld_temperatures.csv',held)
    (OUT/'estimator_validation.json').write_text(json.dumps(dict(batched_shapes_pass=True,known_Binder_pass=True,known_helicity_pass=True,synthetic_q_half_and_integer_WM_pass=True),indent=2))


if __name__=='__main__':main()
