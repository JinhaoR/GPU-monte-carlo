"""Extend the previous joint PT bootstrap with L256; no L64 input."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,csv,json
import numpy as np
from scipy.stats import chi2
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';PREVIOUS=ROOT/'exploration/he3_transition_L96_160/results'
sys.path.insert(0,str(ROOT))
from exploration.he3_transition_L96_160 import analyze as old
raw_observables=old.raw_observables;wham_curve=old.wham_curve;crossing=old.crossing;interval=old.interval
SIZES=np.array([96,128,160,256]);KEYS=old.KEYS
PAIRS=[(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]

def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def wm(T,y,se,q,sizes,start=None):
    # Reuse the previous fitter with the selected size set, only in this process.
    old.SIZES=np.asarray(sizes)
    return old.wm(T,y,se,q,start)

def load_blocks(L):
    p=OUT if L==256 else PREVIOUS
    with np.load(p/f'gd=0_L={L}.npz') as z:return dict(z)

def run(scenario):
    factor,portion,B={'main':(2,slice(None),600),'250k':(5,slice(None),200),'500k':(10,slice(None),200),'first':(2,slice(0,30),200),'second':(2,slice(30,None),200)}[scenario]
    with np.load(PREVIOUS/f'{scenario}_gd=0.npz') as f:prior=dict(f)
    d=load_blocks(256);T=d['T'];np.testing.assert_array_equal(T,prior['T'])
    def group(x):
        x=x[portion];return x.reshape((len(x)//factor,factor)+x.shape[1:]).mean(1)
    n=len(d['chi2'][portion])//factor
    rng=np.random.default_rng(20261102+list(('main','250k','500k','first','second')).index(scenario))
    weights=rng.multinomial(n,np.full(n,1/n),size=B)
    mm=np.stack([group(d[k]) for k in KEYS],axis=1)
    rv=raw_observables(mm.mean(0),T,256);br=raw_observables(np.einsum('bn,nkt->bkt',weights/n,mm),T,256)
    raw=np.concatenate((prior['raw'],rv[None]));bootraw=np.concatenate((prior['bootraw'],br[None]))
    data=dict(T=T,raw=raw,bootraw=bootraw)
    for name in ('critical','integer'):
        grid=prior[name+'_T'];H=factor*group(d[name+'_H']);S=factor*group(d[name+'_S'])
        val,f=wham_curve(H.sum(0),S.sum(0),d[name+'_beta'],d[name+'_energy'],grid,256)
        bv=[]
        for w in weights:
            v,_=wham_curve(np.einsum('b,bjk->jk',w,H),np.einsum('b,bjk->jk',w,S),d[name+'_beta'],d[name+'_energy'],grid,256,f['f']);bv.append(v)
        data[name+'_T']=grid;data[name]=np.concatenate((prior[name],val[None]));data[name+'_boot']=np.concatenate((prior[name+'_boot'],np.array(bv)[None]))
        print(scenario,name,'L256 resampled',B,'residual',f['residual'],flush=True)
    bind=[];cb=[];direct=[];dips=[];fits=[];contrasts=[];holdout=[];fitboots={}
    grid=data['critical_T'];u=data['critical'][:,0];bu=data['critical_boot'][:,:,0]
    for a,b in PAIRS:
        tc,nr=crossing(grid,u[a]-u[b],.38118,.38248,.38183)
        vals=np.array([crossing(grid,x-y,.38118,.38248,tc)[0] for x,y in zip(bu[a],bu[b])]);cb.append(vals)
        lin,_=crossing(T,raw[a,0]-raw[b,0],.38118,.38248,tc)
        pc,_=crossing(T,raw[a,0]-raw[b,0],.38118,.38248,tc,True)
        bind.append(dict(L1=int(SIZES[a]),L2=int(SIZES[b]),T=tc,roots=nr,linear_grid_T=lin,pchip_grid_T=pc,**interval(vals)))
    for j,L in enumerate(SIZES):
        for q in (.5,1.):
            line=2*T/(np.pi*q*q);tc,nr=crossing(T,raw[j,1]-line,T[0],T[-1])
            vals=[crossing(T,v[1]-line,T[0],T[-1],tc)[0] for v in bootraw[j]]
            direct.append(dict(L=int(L),q=q,T=tc,roots=nr,**interval(vals)))
        k=np.argmin(raw[j,0]);mins=bootraw[j,:,0].min(-1);iv=interval(mins)
        dips.append(dict(L=int(L),T_grid_min=T[k],U_min=raw[j,0,k],pointwise_se=bootraw[j,:,0,k].std(ddof=1),**iv))
    for q,name in ((.5,'critical'),(1.,'integer')):
        grid=data[name+'_T'];yy=data[name][:,1];by=data[name+'_boot'][:,:,1];se=by.std(1,ddof=1)
        for ids in (np.arange(4),np.array([1,2,3])):
            sizes=SIZES[ids];tag='_'.join(str(x) for x in sizes)
            fit=wm(grid,yy[ids],se[ids],q,sizes)
            samples=np.array([wm(grid,by[ids,b],se[ids],q,sizes,fit) for b in range(B)])
            good=samples[:,3]==0;fitboots[f'q{q:g}_{tag}']=samples
            rr=wm(T,raw[ids,1],bootraw[ids,:,1].std(1,ddof=1),q,sizes)
            fits.append(dict(q=q,sizes=tag,T=fit[0],C=fit[1],chi2=fit[2],dof=len(sizes)-2,p_value=chi2.sf(fit[2],len(sizes)-2),boundary=fit[3],correction_Lmin=1/(2*np.log(sizes[0])+fit[1]),correction_L256=1/(2*np.log(256)+fit[1]),raw_linear_T=rr[0],raw_linear_chi2=rr[2],**interval(np.where(good,samples[:,0],np.nan))))
            vals=np.asarray(cb[-1])-samples[:,0];vals[~good]=np.nan
            contrasts.append(dict(quantity='Binder160_256_minus_WM',q=q,sizes=tag,difference=bind[-1]['T']-fit[0],**interval(vals)))
        # Independent L256 tests the old three-size fit without refitting it.
        oldsample=prior['wm_half' if q==.5 else 'wm_integer']
        oldfit=next(r for r in csv.DictReader((PREVIOUS/f'{scenario}_wm.csv').open()) if float(r['g_d'])==0 and float(r['q'])==q)
        t=float(oldfit['T']);C=float(oldfit['C']);pred=2*t/(np.pi*q*q)*(1+1/(2*np.log(256)+C));obs=np.interp(t,grid,yy[3])
        vals=[]
        for b,(bt,bc,_,_) in enumerate(oldsample):
            vals.append(np.interp(bt,grid,by[3,b])-2*bt/(np.pi*q*q)*(1+1/(2*np.log(256)+bc)))
        holdout.append(dict(q=q,old_T=t,observed_Y256=obs,predicted_Y256=pred,difference=obs-pred,**interval(vals)))
    mins=bootraw[:,:,0].min(-1)
    for a,b in PAIRS:
        contrasts.append(dict(quantity='dip_difference',q='',sizes=f'{SIZES[a]}_{SIZES[b]}',difference=raw[b,0].min()-raw[a,0].min(),**interval(mins[b]-mins[a])))
    for q in (.5,1.):
        dj=next(r for r in direct if r['L']==256 and r['q']==q);line=2*T/(np.pi*q*q)
        vals=np.array([crossing(T,v[1]-line,T[0],T[-1],dj['T'])[0] for v in bootraw[3]])
        contrasts.append(dict(quantity='Binder160_256_minus_direct256',q=q,sizes='160_256',difference=bind[-1]['T']-dj['T'],**interval(cb[-1]-vals)))
    data['crossing_boot']=np.array(cb);data.update(fitboots)
    np.savez_compressed(OUT/(scenario+'.npz'),**data)
    for name,rows in [('binder',bind),('wm',fits),('direct_jump',direct),('dip',dips),('contrasts',contrasts),('holdout',holdout)]:write(scenario+'_'+name+'.csv',rows)
    print(scenario,'complete',flush=True)

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'main')
