"""Independent raw-record checks: unbinned MBAR, single-histogram prediction,
and the measured joint distribution of global order-parameter magnitudes.
Uses five existing temperature columns at gD=0,L=160. No new sampling.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import root
from prepare import ROOT,OUT,NAMES,COUPLINGS,SIZES
from reweight import load,solve,evaluate,observables,direct
from study import csvwrite,error,VALID


def main():
    checks=[]
    # Independent earlier block cache: no energy histogram has been used in it.
    mapping={'dE':'e','density':'n','m':'chi','absm':'chi_abs','m2':'chi2','m4':'chi4',
        'p':'p_abs','p2':'p2','p4':'p4','ix':'ix','ix2':'ix2','kx':'kx',
        'iy':'iy','iy2':'iy2','ky':'ky','qzz':'qzz'}
    for gd in COUPLINGS:
        for L in SIZES:
            d=load(gd,L);meta=json.loads((OUT/'cache'/f'gd={gd:g}_L={L}.json').read_text())
            with np.load(ROOT/'exploration/he3_soc/results/blocks'/f'gd={gd:g}_L={L}.npz') as old:
                for n,k in mapping.items():
                    a=d['Ssrc'][NAMES.index(n)].sum(1)/60000
                    b=old[k].mean(0)[meta['selected_temperature_indices']]
                    if n=='dE':b=b*L**2-float(d['eref'])
                    checks.append(dict(g_d=gd,L=L,moment=n,max_abs_difference=float(np.max(abs(a-b)))))
    csvwrite('independent_moment_checks.csv',checks)
    d=load(0,160);L=160;N=L**2;eref=float(d['eref'])
    meta=json.loads((OUT/'cache'/'gd=0_L=160.json').read_text());local=np.arange(5,10)
    ids=np.array(meta['selected_temperature_indices'])[local]
    with np.load(ROOT/meta['source'],allow_pickle=False) as z:
        def get(name):return z[f'L160_{name}'][:,ids].copy()
        E=get('energy');n=get('density');C=get('chirality');m=C/n;c=C/N
        pr=get('phase_real')/N;pi=get('phase_imag')/N;p2=pr*pr+pi*pi;p=np.sqrt(p2)
        values={'dE':E-eref,'dE2':(E-eref)**2,'density':n/N,'m':m,'absm':abs(m),'m2':m*m,'m4':m**4,
            'c':c,'absc':abs(c),'c2':c*c,'c4':c**4,'p':p,'p2':p2,'p4':p2*p2}
        for ax in ['x','y']:
            I=get('current_'+ax);values['i'+ax]=I;values['i'+ax+'2']=I*I;values['k'+ax]=get('curvature_'+ax)
        values['qzz']=get('spin_alignment')
    beta=d['beta'][local];train=np.array([0,1,3,4]);held=2;beta_train=beta[train]
    e=(E[:,train]-eref).ravel();u=(beta_train-beta_train[0])[:,None]*e
    logn=np.log(np.full(4,60000.))
    def equations(x):
        f=np.r_[0.,x];ld=logsumexp(logn[:,None]+f[:,None]-u,axis=0)
        zz=logsumexp(-u-ld,axis=1);ps=np.exp(-u-ld-zz[:,None])
        post=np.exp(logn[:,None]+f[:,None]-u-ld)
        jac=np.eye(4)-ps@post.T
        return (f+zz)[1:],jac[1:,1:]
    r=root(equations,np.zeros(3),jac=True,tol=1e-10)
    assert max(abs(equations(r.x)[0]))<1e-8
    f=np.r_[0.,r.x];ld=logsumexp(logn[:,None]+f[:,None]-u,axis=0)
    target=np.array([.38151891,.38189801,1/beta[held]])
    logw=-(1/target[:,None]-beta_train[0])*e-ld
    w=np.exp(logw-logsumexp(logw,axis=1)[:,None])
    means={k:v[:,train].ravel()@w.T for k,v in values.items()}
    unb=observables(means,target,L,eref)
    rows=[]
    for bins in (400,800,1600):
        db=load(0,L,bins);h=db['H'].sum(0);h[np.setdiff1d(np.arange(15),local[train])]=0
        s=db['Ssrc'][:,local[train]].sum(1);ff=solve(h,s,db['beta'],db['ebins']);a=evaluate(ff,target,L,eref)
        for j,t in enumerate(target):
            for key in VALID:
                rows.append(dict(bins=bins,temperature=t,observable=key,unbinned=unb[key][j],
                    binned=a[key][j],difference=a[key][j]-unb[key][j]))
    csvwrite('unbinned_reference.csv',rows)
    # Single-histogram left/right predictions of the central omitted state.
    sh=[]
    for k in (1,3):
        lw=-(beta[held]-beta[k])*(E[:,k]-eref);ww=np.exp(lw-lw.max());ww/=ww.sum()
        av=observables({name:np.array([v[:,k]@ww]) for name,v in values.items()},np.array([1/beta[held]]),L,eref)
        diffs=[];predicts=[]
        for b in range(30):
            keep=np.ones(60000,dtype=bool);keep[b*2000:(b+1)*2000]=False;wb=ww[keep]/ww[keep].sum()
            ap=observables({name:np.array([v[keep,k]@wb]) for name,v in values.items()},np.array([1/beta[held]]),L,eref)
            ac=observables({name:np.array([v[keep,held].mean()]) for name,v in values.items()},np.array([1/beta[held]]),L,eref)
            predicts.append([ap[name][0] for name in VALID]);diffs.append([ap[name][0]-ac[name][0] for name in VALID])
        actual=observables({name:np.array([v[:,held].mean()]) for name,v in values.items()},np.array([1/beta[held]]),L,eref)
        for j,key in enumerate(VALID):
            sh.append(dict(source_T=1/beta[k],target_T=1/beta[held],observable=key,prediction=av[key][0],
                actual=actual[key][0],prediction_se=error(predicts)[j],difference_se=error(diffs)[j],
                paired_z=(av[key][0]-actual[key][0])/error(diffs)[j],raw_weight_ess=1/(ww@ww)))
    csvwrite('single_histogram.csv',sh)
    # Use all five raw columns for distribution reconstruction, with a separate
    # binned normalizer; no order-parameter bins enter the free-energy solve.
    H=d['H'].sum(0);H[np.setdiff1d(np.arange(15),local)]=0
    ff=solve(H,d['Ssrc'][:,local].sum(1),d['beta'],d['ebins'])
    eraw=(E-eref).ravel();ub=(d['beta'][local]-ff['beta_ref'])[:,None]*eraw
    ld=logsumexp(np.log(np.full(5,60000.))[:,None]+ff['f'][:,None]-ub,axis=0)
    distr=[];cov=[];joint=None
    for t in [.38151891,.38189801,.38200483,.38230483]:
        lw=-(1/t-ff['beta_ref'])*eraw-ld;ww=np.exp(lw-logsumexp(lw))
        ma=abs(m).ravel();pa=p.ravel();ea=E.ravel()/N
        mc=ma-ww@ma;pc=pa-ww@pa
        cov.append(dict(temperature=t,correlation_abs_chirality_phase=(ww@(mc*pc))/np.sqrt((ww@mc**2)*(ww@pc**2)),
            raw_weight_ess=1/(ww@ww)))
        for name,arr,edges in [('abs_chirality',ma,np.linspace(0,.55,101)),('phase_magnitude',pa,np.linspace(0,.35,101)),('energy_density',ea,np.linspace(ea.min(),ea.max(),151))]:
            h,_=np.histogram(arr,bins=edges,weights=ww);dens=h/np.diff(edges)
            for x,y in zip((edges[:-1]+edges[1:])/2,dens):distr.append(dict(temperature=t,observable=name,x=x,probability_density=y))
        if abs(t-.38189801)<1e-9:
            joint,xe,ye=np.histogram2d(ma,pa,bins=(np.linspace(0,.5,81),np.linspace(0,.3,81)),weights=ww)
    csvwrite('order_distributions.csv',distr);csvwrite('order_covariance.csv',cov)
    np.savez_compressed(OUT/'joint_order_distribution.npz',hist=joint,xedges=xe,yedges=ye)
    result=dict(unbinned_residual=float(max(abs(equations(r.x)[0]))),
        max_direct_moment_difference_except_energy=float(max(x['max_abs_difference'] for x in checks if x['moment']!='dE')),
        energy_shift_check_max=float(max(x['max_abs_difference'] for x in checks if x['moment']=='dE')))
    (OUT/'reference_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result,flush=True)


if __name__=='__main__':main()
