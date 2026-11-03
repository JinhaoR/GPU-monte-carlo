"""Size flows, prior-result comparisons, and focused tests of the orbital finding."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from analyze import OUT, ROOT, gaussian, estimate, NAMES, PROFILE_NAMES
from reduce import csv_write


def load(gd,L,suffix='summary'):
    with np.load(OUT/f'gd={gd:g}_L={L}_{suffix}.npz') as z:return dict(z)


def roots(T,y,low,high):
    ix=np.flatnonzero((y[:-1]*y[1:]<0)&(T[:-1]>=low)&(T[1:]<=high))
    return [float(T[i]-y[i]*(T[i+1]-T[i])/(y[i+1]-y[i])) for i in ix]


WM_STARTS={}


def wm_fit(runs,q,sizes,curves=None):
    ss=np.asarray(sizes);target=np.array([.3815,-4.] if q==.5 else [.389,-6.])
    gd=runs[sizes[0]]['g_d'];center=.382 if gd==0 else .386
    def residual(p):
        t,C=p;theory=2*t/(np.pi*q*q)*(1+1/(2*np.log(ss)+C))
        measured=np.array([np.interp(t,runs[L]['T'],runs[L]['helicity'] if curves is None else curves[L]) for L in sizes])
        error=np.array([np.interp(t,runs[L]['T'],runs[L]['helicity_se']) for L in sizes])
        return (measured-theory)/error
    key=(gd,q,tuple(sizes));lower=-2*np.log(min(sizes))+.05
    if key not in WM_STARTS:
        scan=[]
        for t in np.linspace(center-.004,center+.012,401):
            one=minimize_scalar(lambda C:np.sum(residual((t,C))**2),bounds=(lower,50),method='bounded')
            scan.append((one.fun,np.array([t,one.x])))
        # Distinct temperature starts, including the best broad-grid solutions.
        WM_STARTS[key]=[x[1] for x in sorted(scan,key=lambda x:x[0])[:4]]
    fits=[]
    for start in WM_STARTS[key]:
        fits.append(least_squares(residual,start,bounds=([center-.004,-2*np.log(min(sizes))+.05],[center+.012,50]),xtol=1e-12,ftol=1e-12,gtol=1e-12))
    fit=min(fits,key=lambda x:np.sum(x.fun*x.fun))
    return fit.x,float(np.sum(fit.fun**2)/(len(sizes)-2))


def main():
    markers=[];flows=[];crossings=[];wm=[];account=[];bgaps=[];decays=[];checks=[]
    sizes=(64,96,128,160)
    for gd in (0,.01):
        runs={L:load(gd,L) for L in sizes};orb={L:load(gd,L,'orbital') for L in sizes}
        for L in sizes:runs[L]['g_d']=gd
        center=.382 if gd==0 else .386
        for L,z in runs.items():
            T=z['T']
            for key in ('specific_heat','chi_response','phase_response'):
                j=np.argmax(z[key]);markers.append(dict(g_d=gd,L=L,observable=key,T_grid_peak=T[j],value=z[key][j],se=z[key+'_se'][j],grid_spacing=np.median(np.diff(T[max(0,j-1):j+2]))))
            with np.load(OUT/f'gd={gd:g}_L={L}.npz') as raw:
                blocks={k:raw[k] for k in raw.files if k not in ('T','profile_T','profile_indices') and not k.startswith('pt_')}
            vv,ee,jk=estimate(blocks,T,L)
            lo,hi=((.38030047,.38413969) if gd==0 else (.38349459,.38739762))
            def allocation(v):
                delta=lambda k:np.interp(hi,T,v[k])-np.interp(lo,T,v[k])
                loss=-delta('helicity');d=-delta('D');screen=delta('S')
                n0,n1=np.interp([lo,hi],T,v['density']);r0,r1=np.interp([lo,hi],T,v['R'])
                return np.array([loss,d/loss,screen/loss, -4*(r0+r1)/2*(n1-n0)/d,
                    -delta('spatial_D'),-4*(np.interp(hi,T,blocks['sp_n'].mean(0))-np.interp(lo,T,blocks['sp_n'].mean(0))),
                    delta('energy_gradient')/2,-delta('W_per_site')])
            # Use each delete's own synchronized density in the exact allocation.
            vals=allocation(vv);delvals=[];spn=blocks['sp_n'].reshape(30,2,-1).mean(1)
            for b in range(30):
                x=allocation({k:a[b] for k,a in jk.items()})
                nn=(30*spn.mean(0)-spn[b])/29
                x[5]=-4*(np.interp(hi,T,nn)-np.interp(lo,T,nn));delvals.append(x)
            se=np.sqrt(29*np.var(delvals,axis=0))
            names=['Y_loss','curvature_share','screening_share','product_density_share_of_D','spatial_D_loss','exact_density_loss','exact_gradient_loss','exact_W_loss']
            row=dict(g_d=gd,L=L,T_low=lo,T_high=hi)
            for k,a,b in zip(names,vals,se):row[k]=a;row[k+'_se']=b
            row['exact_identity_residual']=vals[4]-vals[5:].sum();account.append(row)
            # B++/B-- are fixed diagonal-orbital tensor projections.
            for t in z['profile_T'][1:4]:
                j=np.argmin(abs(T-t));c,_,_=gaussian(L,T[j],z['helicity'][j],np.pi)
                if c is None:continue
                s=np.fft.fft2(c).real;sk=s[[1,-1,0,0],[0,0,1,-1]].mean()
                xg=np.sqrt(s[0,0]/sk-1)/(2*L*np.sin(np.pi/L))
                for sector in ('B_plus','B_minus'):
                    gaps=[]
                    for b in range(30):
                        cb,_,_=gaussian(L,T[j],z['jk_Y'][b,j],np.pi)
                        sb=np.fft.fft2(cb).real;skb=sb[[1,-1,0,0],[0,0,1,-1]].mean()
                        gx=np.sqrt(sb[0,0]/skb-1)/(2*L*np.sin(np.pi/L))
                        gaps.append(orb[L]['jk_'+sector+'_xi'][b,j]-gx)
                    bgaps.append(dict(g_d=gd,L=L,T=T[j],sector=sector,xi=orb[L][sector+'_xi'][j],reference=xg,
                        gap=orb[L][sector+'_xi'][j]-xg,gap_se=np.sqrt(29*np.var(gaps))))
            # Profile slope diagnostics use simultaneous delete-one profiles.
            for p,t in enumerate(z['profile_T']):
                for sector,si,label in [('P',1,'full'),('kappa',0,'full'),('Q',2,'connected'),('Q_m1',4,'connected')]:
                    cc=vv['profile_'+label][p,si].real.mean(0);cj=jk['profile_'+label][:,p,si].real.mean(1)
                    for r in (4,8,16,32):
                        if 2*r>L//2 or cc[r]<=0 or cc[2*r]<=0:continue
                        valid=(cj[:,r]>0)&(cj[:,2*r]>0)
                        eta=-np.log(cc[2*r]/cc[r])/np.log(2)
                        ej=-np.log(cj[valid,2*r]/cj[valid,r])/np.log(2)
                        decays.append(dict(g_d=gd,L=L,T=t,sector=sector,r=r,eta_eff=eta,
                            eta_eff_se=np.sqrt(29*np.var(ej)) if valid.all() else np.nan,
                            inverse_length_eff=eta*np.log(2)/r,valid_deletes=int(valid.sum())))
        # Effective size exponents: S(0) ~ L^(2-eta), not a fitted asymptotic exponent.
        T=runs[64]['T']
        for l1,l2 in zip(sizes[:-1],sizes[1:]):
            for name,index in [(name,i) for i,name in enumerate(NAMES)]+[(x,None) for x in ('A_plus','A_minus','A_plus_z','A_minus_z','A_plus_transverse','B_plus','B_minus')]:
                def powers(L):
                    if index is None:return orb[L][name+'_S'][:,0],orb[L]['jk_'+name+'_S'][:,:,0]
                    return runs[L]['full_spectra'][:,index,0],runs[L]['jk_S'][:,:,index,0]
                s1,j1=powers(l1);s2,j2=powers(l2);lr=np.log(l2/l1)
                eta=2-np.log(s2/s1)/lr
                # Separate-run jackknife contributions, no fabricated time pairing.
                ee=np.sqrt(29*(np.var(np.log(j1),axis=0)+np.var(np.log(j2),axis=0)))/lr
                for j,t in enumerate(T):flows.append(dict(g_d=gd,L1=l1,L2=l2,T=t,sector=name,eta_eff=eta[j],se=ee[j]))
            for key in ('binder_chirality','binder_phase'):
                rr=roots(T,runs[l2][key]-runs[l1][key],center-.0015,center+.0015)
                for r in rr:crossings.append(dict(g_d=gd,L1=l1,L2=l2,observable=key,T=r,roots_in_window=len(rr)))
        for q in (.5,1.):
            for use in (sizes,sizes[1:]):
                fit,chi=wm_fit(runs,q,use)
                row=dict(g_d=gd,q=q,sizes=':'.join(map(str,use)),T=fit[0],C=fit[1],chi2_dof=chi)
                # Delta-method independent-run jackknife: delete blocks from one run at a time.
                variance=0
                if len(use)==4:
                    for changed in use:
                        fs=[]
                        for b in range(30):
                            cv={L:(runs[L]['jk_Y'][b] if L==changed else runs[L]['helicity']) for L in use}
                            fs.append(wm_fit(runs,q,use,cv)[0][0])
                        variance+=29*np.var(fs)
                    row['conditional_T_se']=np.sqrt(variance)
                else:row['conditional_T_se']=np.nan
                wm.append(row)
    for name,rows in [('markers',markers),('size_flows',flows),('binder_crossings',crossings),('wm_fits',wm),
                     ('curvature_accounting',account),('B_gaussian_checks',bgaps),('profile_slopes',decays)]:csv_write(OUT/(name+'.csv'),rows)
    # The previous reweighting window stopped below the new finite-SOC q=1 candidate.
    # Repeat the same broad search on the OLD raw-grid moment caches as a control.
    from exploration.he3_soc.summarize import estimate as old_estimate
    oldfits=[]
    for gd in (0,.01):
        oldruns={}
        for L in sizes:
            with np.load(ROOT/f'exploration/he3_soc/results/blocks/gd={gd:g}_L={L}.npz') as z:
                ot=z['T'];bl={k:z[k] for k in z.files if k!='T'}
            val,err=old_estimate(bl,ot,L,2000)
            oldruns[L]=dict(T=ot,g_d=gd,helicity=val['helicity'],helicity_se=err['helicity'])
        for q in (.5,1.):
            WM_STARTS.pop((gd,q,tuple(sizes)),None)
            fit,ch=wm_fit(oldruns,q,sizes)
            oldfits.append(dict(g_d=gd,q=q,T=fit[0],C=fit[1],chi2_dof=ch,method='linear interpolation of old raw grid; broad search'))
    csv_write(OUT/'old_extended_wm.csv',oldfits)
    checks.append(dict(max_curvature_identity_residual=max(abs(x['exact_identity_residual']) for x in account)))
    (OUT/'findings_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')


if __name__=='__main__':main()
