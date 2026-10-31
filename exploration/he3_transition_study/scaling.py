"""Finite-size diagnostics; statistical errors stay conditional on histories.

Separate L runs are independent. Within each run all observables/temperatures
are deleted together. In particular, crossing differences use paired errors.
No four-size thermodynamic extrapolation is imposed.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import least_squares,minimize_scalar
from scipy.interpolate import PchipInterpolator
from study import OUT,COUPLINGS,SIZES,KEYS,csvwrite,error
from reweight import load,direct


def read(gd,L):
    with np.load(OUT/f'curves_gd={gd:g}_L={L}.npz') as z:return {k:z[k] for k in z.files}


def line(a,key,x,values=None):
    y=a['point'][KEYS.index(key)] if values is None else values[KEYS.index(key)]
    return np.interp(x,a['T'],y)


def crossing(a,b,key,va=None,vb=None,center=.3817):
    T=a['T'];ya=line(a,key,T,va);yb=line(b,key,T,vb);diff=ya-yb
    ids=np.flatnonzero(diff[:-1]*diff[1:]<0)
    roots=T[ids]-diff[ids]*np.diff(T)[ids]/(diff[ids+1]-diff[ids])
    # Root selection is stated explicitly and repeated for every delete.
    roots=[t for t in roots if abs(t-center)<.0015 and .1<line(a,key,t,va)<.95]
    return float(min(roots,key=lambda t:abs(t-center))) if roots else np.nan


def wm(data,q,subset=SIZES,replacements=None,start=None,t_bounds=None):
    replacements={} if replacements is None else replacements
    T=data[64]['T'];logs=2*np.log(np.array(subset)/subset[0]);lo,hi=(T[0],T[-1]) if t_bounds is None else t_bounds
    yy=np.array([data[L]['point'][KEYS.index('helicity')] if L not in replacements else replacements[L][KEYS.index('helicity')] for L in subset])
    ss=np.array([error(data[L]['jk'][:,KEYS.index('helicity')]) for L in subset])
    def residual(p):
        t,a=p;y=np.array([np.interp(t,T,r) for r in yy])*q*q
        s=np.array([np.interp(t,T,r) for r in ss])*q*q
        return (y-2*t/np.pi*(1+a/(1+a*logs)))/s
    if start is None:
        starts=[]
        for t in np.linspace(lo+1e-9,hi-1e-9,111):
            f=minimize_scalar(lambda a:sum(residual([t,a])**2),bounds=(0,30),method='bounded')
            starts.append((f.fun,[t,f.x]))
        starts=[x for score,x in sorted(starts)[:4]]
    else:starts=[start]
    fits=[least_squares(residual,x,bounds=([lo,1e-7],[hi,100]),xtol=1e-12,ftol=1e-11,gtol=1e-9) for x in starts]
    f=min(fits,key=lambda a:sum(a.fun**2));t,a=f.x
    return np.array([t,a]),float(sum(f.fun**2)),float(1/a-2*np.log(subset[0]))


def main():
    crosses=[];wms=[];gaps=[];sens=[];peaks=[];exponents=[];slopes=[]
    for gd in COUPLINGS:
        data={L:read(gd,L) for L in SIZES};center={0:.3817,.001:.3823,.01:.3855}[gd]
        # Add direct-data linear interpolation as a deliberately inferior baseline.
        for L,a in data.items():
            d=load(gd,L);v=[direct(d,k,L) for k in range(15)]
            linear=np.array([np.interp(a['T'],1/d['beta'],[x[key] for x in v]) for key in KEYS])
            a['variant_names']=np.concatenate([a['variant_names'],['linear_direct']])
            a['variants']=np.concatenate([a['variants'],linear[None]],axis=0)
        for L1,L2 in [(64,96),(96,128),(128,160),(64,128),(96,160)]:
            a,b=data[L1],data[L2];keys=['binder_chirality','binder_chirality_raw','binder_phase']
            roots=np.array([crossing(a,b,k,center=center) for k in keys]);deletes=[];vars=[]
            for L in (L1,L2):
                jk=[]
                for v in data[L]['jk']:
                    jk.append([crossing(a,b,k,v if L==L1 else None,v if L==L2 else None,center) for k in keys])
                jk=np.array(jk);deletes.append(jk);vars.append(error(jk)**2)
            se=np.sqrt(sum(vars))
            for j,k in enumerate(keys):
                crosses.append(dict(g_d=gd,L1=L1,L2=L2,observable=k,temperature=roots[j],
                    joint_pt_se=se[j],failed_deletes=int(sum(np.isnan(x[:,j]).sum() for x in deletes))))
            diffs=[x[:,0]-x[:,2] for x in deletes];gap_se=np.sqrt(sum(error(x)**2 for x in diffs))
            gaps.append(dict(g_d=gd,L1=L1,L2=L2,definition='chi_minus_phase_Binder',
                temperature_difference=roots[0]-roots[2],paired_se=gap_se,
                unpaired_se=np.sqrt(se[0]**2+se[2]**2)))
            if (L1,L2)==(128,160):
                largest=(roots,deletes)
                for idx,label in enumerate(a['variant_names']):
                    rv=[crossing(a,b,k,a['variants'][idx],b['variants'][idx],center) for k in keys]
                    for j,k in enumerate(keys):sens.append(dict(g_d=gd,estimator=k,variant=str(label),temperature=rv[j],shift=rv[j]-roots[j]))
                for group in (4000,6000):
                    vals=[]
                    for L in (L1,L2):
                        vals.append(np.array([[crossing(a,b,k,v if L==L1 else None,v if L==L2 else None,center) for k in keys] for v in data[L][f'jk{group}']]))
                    for j,k in enumerate(keys):sens.append(dict(g_d=gd,estimator=k,variant=f'block{group}_se',temperature=np.sqrt(sum(error(x[:,j])**2 for x in vals)),shift=np.nan))
                    gaps.append(dict(g_d=gd,L1=L1,L2=L2,definition=f'chi_minus_phase_Binder_block{group}',
                        temperature_difference=roots[0]-roots[2],paired_se=np.sqrt(sum(error(x[:,0]-x[:,2])**2 for x in vals)),unpaired_se=np.nan))
        for q in (.5,1.):
            opt,chi,C=wm(data,q);jks={};variance=0.
            boundary=min(abs(opt[0]-data[64]['T'][0]),abs(opt[0]-data[64]['T'][-1]))<1e-7
            if boundary:
                wms.append(dict(g_d=gd,q=q,sizes='64:96:128:160',temperature=np.nan,joint_pt_se=np.nan,C=C,chi2_dof=chi/2))
                print('WM boundary: no accepted estimate',gd,q,flush=True)
                continue
            for L in SIZES:
                j=np.array([wm(data,q,replacements={L:v},start=opt)[0][0] for v in data[L]['jk']]);jks[L]=j;variance+=error(j)**2
            wms.append(dict(g_d=gd,q=q,sizes='64:96:128:160',temperature=opt[0],joint_pt_se=np.sqrt(variance),C=C,chi2_dof=chi/2))
            for subset in [(96,128,160),(64,128,160),(64,96,160),(64,96,128)]:
                o,ch,c=wm(data,q,subset);wms.append(dict(g_d=gd,q=q,sizes=':'.join(map(str,subset)),temperature=o[0],joint_pt_se=np.nan,C=c,chi2_dof=ch))
            for group in (4000,6000):
                coarse={L:{**data[L],'jk':data[L][f'jk{group}']} for L in SIZES}
                o,ch,c=wm(coarse,q);var=0.
                for L in SIZES:
                    j=[wm(coarse,q,replacements={L:v},start=o)[0][0] for v in coarse[L]['jk']];var+=error(j)**2
                sens.append(dict(g_d=gd,estimator=f'WM_q{q:g}',variant=f'block{group}_fit',temperature=o[0],shift=o[0]-opt[0]))
                sens.append(dict(g_d=gd,estimator=f'WM_q{q:g}',variant=f'block{group}_se',temperature=np.sqrt(var),shift=np.nan))
            for idx,label in enumerate(data[64]['variant_names']):
                bounds=None
                if str(label).startswith('sources'):
                    width=int(str(label)[7:]);dd=load(gd,64);bounds=(dd['T'][7-width//2],dd['T'][7+width//2])
                o,ch,c=wm(data,q,replacements={L:data[L]['variants'][idx] for L in SIZES},t_bounds=bounds)
                boundary=bounds is not None and min(abs(o[0]-bounds[0]),abs(o[0]-bounds[1]))<1e-7
                sens.append(dict(g_d=gd,estimator=f'WM_q{q:g}',variant=str(label)+('_boundary' if boundary else ''),temperature=o[0] if not boundary else np.nan,shift=o[0]-opt[0] if not boundary else np.nan))
            roots,deleted=largest;var_gap=0.
            for L in SIZES:
                z=deleted[(128,160).index(L)][:,0] if L in (128,160) else np.full(len(jks[L]),roots[0])
                var_gap+=error(z-jks[L])**2
            gaps.append(dict(g_d=gd,L1=128,L2=160,definition=f'chi_Binder_minus_WM_q{q:g}',temperature_difference=roots[0]-opt[0],paired_se=np.sqrt(var_gap),unpaired_se=np.nan))
        for L,a in data.items():
            for key in ('specific_heat','chi_response','chi_raw_response','phase_response'):
                j=KEYS.index(key);i=int(np.argmax(a['point'][j]));T=a['T']
                # Quadratic refinement avoids a zero delete variance when an
                # extremum and all deletes fall in the same fine-grid cell.
                def peak(y):
                    ii=int(np.argmax(y));ii=max(1,min(len(T)-2,ii));x=T[ii-1:ii+2]-T[ii]
                    p=np.polyfit(x,y[ii-1:ii+2],2);delta=np.clip(-p[1]/(2*p[0]),x[0],x[-1])
                    return T[ii]+delta,np.polyval(p,delta)
                temp,height=peak(a['point'][j]);ts=[];hs=[]
                for v in a['jk']:
                    tt,hh=peak(v[j]);ts.append(tt);hs.append(hh)
                peaks.append(dict(g_d=gd,L=L,observable=key,temperature=temp,temperature_se=error(ts),
                    height=height,height_se=error(hs),grid_step=T[1]-T[0]))
            t=largest[0][0]
            for key in ('binder_chirality','binder_chirality_raw','binder_phase'):
                y=a['point'][KEYS.index(key)];slope=np.interp(t,a['T'],np.gradient(y,a['T']))
                slopes.append(dict(g_d=gd,L=L,observable=key,temperature=t,slope=slope))
        for L1,L2 in [(64,96),(96,128),(128,160)]:
            a,b=data[L1],data[L2]
            for t in (largest[0][0],largest[0][0]-.0003,largest[0][0]+.0003):
                for key in ('phase2','chirality_raw2','chi2'):
                    eta=-np.log(line(b,key,t)/line(a,key,t))/np.log(L2/L1)
                    vs=[]
                    for L in (L1,L2):
                        ee=[-np.log(line(b,key,t,v if L==L2 else None)/line(a,key,t,v if L==L1 else None))/np.log(L2/L1) for v in data[L]['jk']]
                        vs.append(error(ee)**2)
                    exponents.append(dict(g_d=gd,L1=L1,L2=L2,temperature=t,observable=key,eta_effective=eta,se=np.sqrt(sum(vs))))
        print('Scaling',gd,'WM',[(x['q'],x['temperature'],x['chi2_dof']) for x in wms if x['g_d']==gd and x['sizes']=='64:96:128:160'],flush=True)
    for name,rows in [('crossings.csv',crosses),('wm_fits.csv',wms),('paired_gaps.csv',gaps),('temperature_sensitivity.csv',sens),('response_peaks.csv',peaks),('effective_exponents.csv',exponents),('binder_slopes.csv',slopes)]:csvwrite(name,rows)


if __name__=='__main__':main()
