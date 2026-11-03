"""Plots and paired contrasts from saved bootstrap realizations; no refitting data."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from analyze import OUT,SIZES,crossing,interval,write


def main():
    fits=list(csv.DictReader((OUT/'main_wm.csv').open()))
    bind=list(csv.DictReader((OUT/'main_binder.csv').open()))
    direct=list(csv.DictReader((OUT/'main_direct_jump.csv').open()))
    figb,ab=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    figh,ah=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    figp,ap=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    figd,ad=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    paired=[];drifts=[];diptrends=[];profiles=[];allroots=[]
    colors=['tab:blue','tab:orange','tab:green']
    for row,g in enumerate((0,.01)):
        with np.load(OUT/f'main_gd={g:g}.npz') as f:z=dict(f)
        T=z['T'];raw=z['raw'];br=z['bootraw'];se=br.std(1,ddof=1);ct=z['critical_T']
        for a,b in ((0,1),(1,2),(0,2)):
            d=raw[a,0]-raw[b,0]
            for k in np.flatnonzero(d[:-1]*d[1:]<0):
                t=T[k]-d[k]*(T[k+1]-T[k])/(d[k+1]-d[k]);u=np.interp(t,T,raw[a,0])
                vals=[];counts=[]
                for x,y in zip(br[a,:,0],br[b,:,0]):
                    v,count=crossing(T,x-y,max(T[0],t-.0005),min(T[-1],t+.0005),t)
                    vals.append(v);counts.append(count)
                allroots.append(dict(g_d=g,L1=int(SIZES[a]),L2=int(SIZES[b]),T=t,U=u,
                    unique_root_fraction=float(np.mean(np.array(counts)==1)),**interval(vals)))
        for j,L in enumerate(SIZES):
            ab[row,0].plot(ct,z['critical'][j,0],c=colors[j],label=f'L={L}')
            ab[row,0].errorbar(T,raw[j,0],se[j,0],c=colors[j],fmt='.',ms=3,alpha=.7)
            ab[row,1].errorbar(T,raw[j,0],se[j,0],c=colors[j],fmt='.-',ms=3,label=f'L={L}')
        ab[row,0].set(xlim=(.3814,.3822) if g==0 else (.3851,.38605),ylim=(-.05,.67),title=f'g_D={g:g}: Binder crossing',xlabel='T',ylabel='U_chi')
        ab[row,1].set(xlim=(.3825,.387) if g==0 else (.387,.394),ylim=(-.24,.04),title=f'g_D={g:g}: negative dip',xlabel='T',ylabel='U_chi')
        for ax in ab[row]:ax.axhline(0,c='k',lw=.6);ax.legend();ax.grid(alpha=.2)
        for col,(q,name) in enumerate(((1.,'integer'),(.5,'critical'))):
            f=next(x for x in fits if float(x['g_d'])==g and float(x['q'])==q)
            tc=float(f['T']);C=float(f['C']);grid=z[name+'_T'];yy=z[name][:,1];sig=z[name+'_boot'][:,:,1].std(1,ddof=1)
            for j,L in enumerate(SIZES):
                ah[row,col].errorbar(T,raw[j,1],se[j,1],c=colors[j],fmt='.-',lw=.6,ms=3,alpha=.8)
                ah[row,col].plot(grid,yy[j],c=colors[j],label=f'L={L}, reweighted')
                ah[row,col].plot(grid,2*grid/(np.pi*q*q)*(1+1/(2*np.log(L)+C)),c=colors[j],ls='--',lw=.8)
            jump=2*grid/(np.pi*q*q)
            dd=[r for r in direct if float(r['g_d'])==g and float(r['q'])==q]
            right=max(grid[-1],max(float(r['T'])+3*float(r['se']) for r in dd))
            lineT=np.linspace(grid[0],right,100);ah[row,col].plot(lineT,2*lineT/(np.pi*q*q),'k:',label='direct jump line')
            for j,L in enumerate(SIZES):
                d=next(r for r in dd if int(r['L'])==L);dt=float(d['T'])
                ah[row,col].errorbar(dt,2*dt/(np.pi*q*q),xerr=float(d['se']),fmt='v',c=colors[j],ms=5)
            ah[row,col].axvline(tc,c='k',lw=.8,alpha=.6)
            ah[row,col].set(xlim=(grid[0],right),ylim=(min(jump)*.85,max(yy.max(),jump.max())*1.06),xlabel='T (triangles: direct crossings)',ylabel='Y',title=f'g_D={g:g}, q={q:g}; WM chi2/df={float(f["chi2"]):.2f}')
            ah[row,col].legend(fontsize=7);ah[row,col].grid(alpha=.2)
            scores=[]
            for k,t in enumerate(grid):
                ff=minimize_scalar(lambda a:np.sum(((yy[:,k]-2*t/(np.pi*q*q)*(1+a/(1+2*a*np.log(SIZES/96))))/sig[:,k])**2),bounds=(0,10),method='bounded')
                scores.append(ff.fun);profiles.append(dict(g_d=g,q=q,T=t,chi2=ff.fun))
            scores=np.array(scores);ap[row,col].plot(grid,scores-float(f['chi2']))
            ap[row,col].axhline(1,c='grey',ls='--');ap[row,col].axvspan(float(f['lo']),float(f['hi']),alpha=.2,label='95% block-bootstrap T range')
            ap[row,col].set(xlim=(tc-6*float(f['se']),tc+6*float(f['se'])),ylim=(0,20),xlabel='T',ylabel='profile chi2 - minimum',title=f'g_D={g:g}, q={q:g}');ap[row,col].legend(fontsize=7)
            if q==.5:diff=z['crossing_boot'][1]-z['wm_half'][:,0]
            line=2*T/(np.pi*q*q)
            dj=next(x for x in direct if float(x['g_d'])==g and int(x['L'])==160 and float(x['q'])==q)
            vals=np.array([crossing(T,v[1]-line,T[0],T[-1],float(dj['T']))[0] for v in br[2]])
            bc=next(x for x in bind if float(x['g_d'])==g and int(x['L1'])==128)
            paired.append(dict(g_d=g,q=q,difference=float(bc['T'])-float(dj['T']),**interval(z['crossing_boot'][1]-vals)))
        drift=z['crossing_boot'][1]-z['crossing_boot'][0]
        bb=[x for x in bind if float(x['g_d'])==g]
        drifts.append(dict(g_d=g,difference=float(bb[1]['T'])-float(bb[0]['T']),**interval(drift)))
        mins=br[:,:,0].min(-1);nom=raw[:,0].min(-1)
        ad[row].errorbar(SIZES,nom,[se[j,0,np.argmin(raw[j,0])] for j in range(3)],fmt='o-',label='observed grid min; pointwise SE')
        ad[row].set(title=f'g_D={g:g}',xlabel='L',ylabel='minimum U_chi');ad[row].grid(alpha=.2);ad[row].legend(fontsize=7)
        diptrends.append(dict(g_d=g,monotone_resample_fraction=float(np.mean((mins[1]>mins[0])&(mins[2]>mins[1]))),
            overall_shallower_fraction=float(np.mean(mins[2]>mins[0]))))
    for fig,name in [(figb,'binder_crossings_and_dips'),(figh,'helicity_crossings_and_WM'),(figp,'WM_profiles'),(figd,'dip_size_trend')]:
        fig.savefig(OUT/(name+'.png'),dpi=180);plt.close(fig)
    write('direct_vs_binder.csv',paired);write('binder_drift.csv',drifts);write('dip_trends.csv',diptrends);write('WM_profiles.csv',profiles);write('all_raw_binder_roots.csv',allroots)


if __name__=='__main__':main()
