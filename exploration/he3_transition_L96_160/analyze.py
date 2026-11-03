"""Joint PT-block bootstrap, local WHAM crossings, and three-size WM fits.

Only L=96,128,160 are loaded or fitted. No simulations or production writes.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,csv
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import least_squares,brentq,minimize_scalar
from scipy.interpolate import PchipInterpolator
from scipy.stats import chi2
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'exploration/he3_transition_study'))
from reweight import solve
from utils.Analysis.bkt import fit_weber_minnhagen
SIZES=np.array([96,128,160]);KEYS=('chi2','chi4','ix','ix2','kx','iy','iy2','ky')


def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def raw_observables(m,T,L):
    u=1-m[...,1,:]/(3*m[...,0,:]**2)
    y=(m[...,4,:]+m[...,7,:]-((m[...,3,:]-m[...,2,:]**2)+(m[...,6,:]-m[...,5,:]**2))/T)/(2*L**2)
    return np.stack((u,y),axis=-2)


def wham_curve(H,S,beta,energy,T,L,guess=None):
    f=solve(H,S,beta,energy,guess)
    lw=-(1/T[:,None]-f['beta_ref'])*f['e'][None,:]-f['ld'][None,:]
    weights=np.exp(lw-logsumexp(lw+np.log(f['h'])[None,:],axis=1)[:,None])
    mu=f['S']@weights.T
    return raw_observables(mu,T,L),f


def crossing(T,diff,low,high,target=None,pchip=False):
    ids=np.flatnonzero((diff[:-1]*diff[1:]<0)&(T[:-1]>=low)&(T[1:]<=high))
    if not len(ids):return np.nan,0
    rr=[float(T[j]-diff[j]*(T[j+1]-T[j])/(diff[j+1]-diff[j])) for j in ids]
    j=0 if target is None else int(np.argmin(abs(np.array(rr)-target)))
    if pchip:
        p=PchipInterpolator(T,diff);i=ids[j];rr[j]=brentq(p,T[i],T[i+1],xtol=1e-13)
    return rr[j],len(rr)


def wm(T,y,se,q,start=None):
    logs=2*np.log(SIZES/SIZES[0])
    def res(p):
        t,a=p;pred=2*t/(np.pi*q*q)*(1+a/(1+a*logs))
        return (np.array([np.interp(t,T,z) for z in y])-pred)/np.array([np.interp(t,T,z) for z in se])
    if start is None:
        initial=fit_weber_minnhagen(T,SIZES,q*q*y,q*q*se,profile=False)
        starts=[]
        if initial['temperature'] is not None:starts.append((initial['temperature'],1/(2*np.log(SIZES[0])+initial['C'])))
        # The original five starts can stick at raw-grid interpolation knots.
        # Profile every recorded temperature before refining the broad-grid fit.
        if T[-1]-T[0]>.01:
            scan=[]
            for t in T:
                f=minimize_scalar(lambda a:np.sum(res((t,a))**2),bounds=(0,10),method='bounded')
                scan.append((f.fun,(t,f.x)))
            starts.extend(x[1] for x in sorted(scan,key=lambda x:x[0])[:4])
        else:starts.extend((t,.1) for t in np.linspace(T[0],T[-1],5))
    else:
        a0=1/(2*np.log(SIZES[0])+start[1]);starts=[(t,max(a0,1e-4)) for t in (start[0],(T[0]+T[-1])/2)]
    fits=[least_squares(res,x,bounds=([T[0],0],[T[-1],np.inf]),xtol=1e-10,ftol=1e-10,gtol=1e-10) for x in starts]
    f=min(fits,key=lambda x:sum(x.fun*x.fun));t,a=f.x
    edge=(t-T[0]<1e-8 or T[-1]-t<1e-8 or a<1e-8)
    return t,1/a-2*np.log(SIZES[0]) if a>0 else np.nan,float(sum(f.fun*f.fun)),edge


def interval(a):
    v=np.asarray(a);v=v[np.isfinite(v)]
    if len(v)<2:return dict(se=np.nan,lo=np.nan,hi=np.nan,valid=len(v),mean=np.nan)
    return dict(se=float(v.std(ddof=1)),lo=float(np.quantile(v,.025)),hi=float(np.quantile(v,.975)),valid=len(v),mean=float(v.mean()))


def run(scenario):
    factor,portion,B={'main':(2,slice(None),600),'250k':(5,slice(None),200),'500k':(10,slice(None),200),
        'first':(2,slice(0,30),200),'second':(2,slice(30,None),200)}[scenario]
    rng=np.random.default_rng(20261101+list(('main','250k','500k','first','second')).index(scenario))
    binders=[];fits=[];dips=[];direct=[];contrasts=[];validation=[]
    for gd in (0,.01):
        runs=[];weights=[]
        for L in SIZES:
            with np.load(OUT/f'gd={gd:g}_L={L}.npz') as z:runs.append(dict(z))
            n=len(runs[-1]['chi2'][portion])//factor;weights.append(rng.multinomial(n,np.full(n,1/n),size=B))
        T=runs[0]['T'];center=.38183 if gd==0 else .38558
        raw=[];bootraw=[]
        def group(x):
            x=x[portion];return x.reshape((len(x)//factor,factor)+x.shape[1:]).mean(1)
        for i,(L,d) in enumerate(zip(SIZES,runs)):
            mm=np.stack([group(d[k]) for k in KEYS],axis=1);raw.append(raw_observables(mm.mean(0),T,L))
            bm=np.einsum('bn,nkt->bkt',weights[i]/n,mm);bb=raw_observables(bm,T,L)
            assert bb.shape==(B,2,len(T));bootraw.append(bb)
            j=np.argmin(raw[-1][0]);mins=bb[:,0,:].min(-1);point=interval(bb[:,0,j]);mi=interval(mins)
            dips.append(dict(g_d=gd,L=int(L),T_grid_min=T[j],U_min=raw[-1][0,j],pointwise_se=point['se'],
                selected_min_se=mi['se'],selected_min_lo=mi['lo'],selected_min_hi=mi['hi'],bootstrap_min_mean=mi['mean'],
                T_min_lo=np.quantile(T[np.argmin(bb[:,0,:],axis=1)],.025),T_min_hi=np.quantile(T[np.argmin(bb[:,0,:],axis=1)],.975)))
            for q in (.5,1.):
                jump=2*T/(np.pi*q*q);tc,nr=crossing(T,raw[-1][1]-jump,T[0],T[-1])
                vals=[crossing(T,v[1]-jump,T[0],T[-1],tc)[0] for v in bb]
                direct.append(dict(g_d=gd,L=int(L),q=q,T=tc,roots=nr,**interval(vals)))
        raw=np.array(raw);bootraw=np.array(bootraw)
        curves={};boots={};grids={};centers={}
        for name in ('critical','integer'):
            grid=np.linspace(max(d[name+'_T'][0] for d in runs),min(d[name+'_T'][-1] for d in runs),241)
            grids[name]=grid;values=[];bvalues=[]
            for i,(L,d) in enumerate(zip(SIZES,runs)):
                H=factor*group(d[name+'_H']);S=factor*group(d[name+'_S']);energy=d[name+'_energy'];beta=d[name+'_beta']
                value,f=wham_curve(H.sum(0),S.sum(0),beta,energy,grid,L);values.append(value)
                bv=[]
                for b,w in enumerate(weights[i]):
                    v,_=wham_curve(np.einsum('b,bjk->jk',w,H),np.einsum('b,bjk->jk',w,S),beta,energy,grid,L,f['f']);bv.append(v)
                bvalues.append(bv)
                if scenario=='main':
                    # Central histogram-resolution control (800 versus 400 bins).
                    hc=H.sum(0).reshape(len(beta),400,2).sum(-1);sc=S.sum(0).reshape(8,400,2).sum(-1)
                    vc,_=wham_curve(hc,sc,beta,energy.reshape(400,2).mean(-1),grid,L)
                    validation.append(dict(g_d=gd,L=int(L),window=name,wham_residual=f['residual'],
                        max_bin_U_shift=float(np.max(abs(vc[0]-value[0]))),max_bin_Y_shift=float(np.max(abs(vc[1]-value[1])))))
                print(scenario,gd,name,L,'resampled',B,flush=True)
            curves[name]=np.array(values);boots[name]=np.array(bvalues)
        grid=grids['critical'];u=curves['critical'][:,0];bu=boots['critical'][:,:,0]
        crossing_boot={}
        for a,b in ((0,1),(1,2),(0,2)):
            tc,nr=crossing(grid,u[a]-u[b],center-.00065,center+.00065,center)
            vals=np.array([crossing(grid,x-y,center-.00065,center+.00065,tc)[0] for x,y in zip(bu[a],bu[b])])
            crossing_boot[a,b]=vals
            linear,_=crossing(T,raw[a,0]-raw[b,0],center-.00065,center+.00065,center)
            cubic,_=crossing(T,raw[a,0]-raw[b,0],center-.00065,center+.00065,center,True)
            binders.append(dict(g_d=gd,L1=int(SIZES[a]),L2=int(SIZES[b]),T=tc,roots=nr,linear_grid_T=linear,pchip_grid_T=cubic,**interval(vals)))
        for q,name in ((.5,'critical'),(1.,'integer')):
            grid=grids[name];yy=curves[name][:,1];by=boots[name][:,:,1];se=by.std(1,ddof=1)
            fit=wm(grid,yy,se,q);sample=np.array([wm(grid,by[:,b],se,q,fit) for b in range(B)])
            good=sample[:,3]==0
            rr=wm(T,raw[:,1],bootraw[:,:,1].std(1,ddof=1),q)
            fits.append(dict(g_d=gd,q=q,T=fit[0],C=fit[1],chi2=fit[2],dof=1,p_value=chi2.sf(fit[2],1),boundary=fit[3],
                correction_L96=1/(2*np.log(96)+fit[1]),correction_L160=1/(2*np.log(160)+fit[1]),
                raw_linear_T=rr[0],raw_linear_chi2=rr[2],**interval(np.where(good,sample[:,0],np.nan))))
            for pair,v in crossing_boot.items():
                delta=v-sample[:,0];delta[~good]=np.nan
                nominal=next(x['T'] for x in binders if x['g_d']==gd and x['L1']==SIZES[pair[0]] and x['L2']==SIZES[pair[1]])-fit[0]
                contrasts.append(dict(g_d=gd,q=q,L1=int(SIZES[pair[0]]),L2=int(SIZES[pair[1]]),difference=nominal,**interval(delta)))
            centers[q]=sample
        dipboot=bootraw[:,:,0].min(-1)
        for a,b in ((0,1),(1,2),(0,2)):
            d=dipboot[b]-dipboot[a]
            contrasts.append(dict(g_d=gd,q='dip',L1=int(SIZES[a]),L2=int(SIZES[b]),
                difference=float(raw[b,0].min()-raw[a,0].min()),**interval(d)))
        np.savez_compressed(OUT/f'{scenario}_gd={gd:g}.npz',T=T,raw=raw,bootraw=bootraw,
            critical_T=grids['critical'],critical=curves['critical'],critical_boot=boots['critical'],
            integer_T=grids['integer'],integer=curves['integer'],integer_boot=boots['integer'],
            crossing_boot=np.array(list(crossing_boot.values())),wm_half=centers[.5],wm_integer=centers[1.])
    for name,rows in [('binder',binders),('wm',fits),('dip',dips),('direct_jump',direct),('contrasts',contrasts),('validation',validation)]:
        if rows:write(scenario+'_'+name+'.csv',rows)


if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'main')
