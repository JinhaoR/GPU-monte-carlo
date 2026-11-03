"""Joint blocked estimates for every recorded scalar, tensor and profile channel.

Uses compact moment blocks written by reduce.py. No simulation or original writes.
"""
from pathlib import Path
import csv
import json
import sys
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
sys.path.insert(0,str(ROOT))
from exploration.he3_soc.summarize import evaluate as old_evaluate
from models.he3.spatial import contracted_components, correlation_ratio, SECTORS
from reduce import csv_write
NAMES=list(SECTORS)+['B_trace','B_difference','B_xy']
PROFILE_NAMES=['kappa','P','Q','Q_m0','Q_m1','Q_m2']


def evaluate(m,T,L):
    N=L*L
    v=old_evaluate(m,T,L)
    v['D']=(m['kx']+m['ky'])/(2*N)
    v['S']=((m['ix2']-m['ix']**2)+(m['iy2']-m['iy']**2))/(2*T*N)
    v['mean_current_correction']=(m['ix']**2+m['iy']**2)/(2*T*N)
    v['Yx_minus_Yy']=v['helicity_x']-v['helicity_y']
    v['R']=v['D']/(4*m['n'])
    v['current_cross_covariance_per_TN']=(m['ixiy']-m['ix']*m['iy'])/(T*N)
    v['current_correlation']=(m['ixiy']-m['ix']*m['iy'])/np.sqrt((m['ix2']-m['ix']**2)*(m['iy2']-m['iy']**2))
    v['density_susceptibility']=N*(m['n2']-m['n']**2)/T
    dd=0;ds=0
    for a in ('x','y'):
        i,i2,k=m['i'+a],m['i'+a+'2'],m['k'+a]
        dd+=(m['k'+a+'e']-k*m['e'])/T**2/2
        ds+=((m['i'+a+'2e']-i2*m['e'])-2*i*(m['i'+a+'e']-i*m['e']))/(2*T**3)-(i2-i*i)/(2*N*T**2)
    v['dD_dT']=dd;v['dS_dT']=ds;v['dY_dT']=dd-ds
    s=m['structure'];den=s[...,0]
    for name,index in [('epsilon',6),('orbital_loss',7),('spin_loss',8),('n_plus_fraction',9),('n_minus_fraction',10),
                       ('abs_s_plus_per_density',11),('abs_s_minus_per_density',12),('local_abs_kappa_per_density',19)]:
        v[name]=s[...,index]/den
    for name,index in [('energy_bulk',13),('energy_gradient',14),('dipole_bare',15),('energy_soc',16)]:v[name]=s[...,index]/N
    v['spatial_D']=(m['sp_kx']+m['sp_ky'])/(2*N)
    v['spatial_Y']=v['spatial_D']-((m['sp_ix2']-m['sp_ix']**2)+(m['sp_iy2']-m['sp_iy']**2))/(2*T*N)
    v['spatial_minus_full_Y']=v['spatial_Y']-v['helicity']
    v['W_per_site']=v['spatial_D']-4*m['sp_n']+v['energy_gradient']/2
    v['epsilon_below_01']=s[...,21]/s[...,3]
    v['epsilon_above_05']=s[...,26:32].sum(-1)/s[...,3]
    v['epsilon_hist_count']=s[...,20:32]/s[...,3,None]
    v['epsilon_hist_density']=s[...,32:44]/s[...,1,None]
    v['orbital_hist_count']=s[...,44:56]/s[...,3,None]
    v['spin_hist_count']=s[...,56:68]/s[...,3,None]
    a=m['amplitude'];p=m['power'];b=m['b_power']
    bm=np.stack(((a[...,7,:]+a[...,8,:])/np.sqrt(2),(a[...,7,:]-a[...,8,:])/np.sqrt(2),np.sqrt(2)*a[...,9,:]),axis=-2)
    for connected,label in [(False,'full'),(True,'connected')]:
        pc=p-abs(a)**2 if connected else p
        bc=b-abs(bm)**2 if connected else b
        spec=np.concatenate((contracted_components(pc),bc),axis=-2)/N
        v[label+'_spectra']=spec
        for d,sk in [('xy',spec[...,1:].mean(-1)),('x',spec[...,1:3].mean(-1)),('y',spec[...,3:5].mean(-1))]:
            ratio,xi,status=correlation_ratio(spec[...,0],sk,L)
            v[label+'_ratio_'+d]=ratio;v[label+'_xi_'+d]=xi
        v[label+'_Q_components']=pc[...,2:7,:]/N
        v[label+'_A_components']=pc[...,10:,:]/N
    v['Q_mean']=a[...,2:7,0].real/N
    v['Q_disconnected_fraction']=np.sum(abs(a[...,2:7,0])**2,axis=-1)/np.sum(p[...,2:7,0],axis=-1)
    v['A_z_power_fraction']=p[...,14:16,0].sum(-1)/p[...,10:16,0].sum(-1)
    v['B_trace_to_P']=b[...,0,0]/p[...,1,0]
    v['B_xy_to_difference']=b[...,2,0]/b[...,1,0]
    v['B_cross_normalized']=m['b_cross']/np.sqrt(np.stack([b[...,i,:]*b[...,j,:] for i,j in [(0,1),(0,2),(1,2)]],axis=-2))
    profiles=m['profiles'];mean=m['profile_means']
    for connected,label in [(False,'full'),(True,'connected')]:
        c=profiles-abs(mean)[...,None,None]**2 if connected else profiles
        v['profile_'+label]=np.stack((c[...,0,:,:],c[...,1,:,:],c[...,2:7,:,:].sum(-3),c[...,3,:,:],
                                      c[...,5:7,:,:].sum(-3),c[...,[2,4],:,:].sum(-3)),axis=-3)
    return v


def aggregate(blocks,factor=2,portion=None):
    out={}
    for k,x in blocks.items():
        if portion is not None:x=x[portion]
        count=len(x)//factor
        out[k]=x[:count*factor].reshape((count,factor)+x.shape[1:]).mean(1)
    return out


def estimate(blocks,T,L,factor=2,portion=None):
    b=aggregate(blocks,factor,portion);B=len(b['e']);means={k:x.mean(0) for k,x in b.items()}
    point=evaluate(means,T,L)
    deletes=evaluate({k:(B*means[k]-x)/(B-1) for k,x in b.items()},T,L)
    errors={}
    for k,x in deletes.items():
        if np.iscomplexobj(x):errors[k]=np.sqrt((B-1)*np.var(x.real,axis=0))+1j*np.sqrt((B-1)*np.var(x.imag,axis=0))
        else:errors[k]=np.sqrt((B-1)*np.var(x,axis=0))
    return point,errors,deletes


def gaussian(L,T,rho,period=None):
    if rho<=0:return None,np.nan,np.nan
    k=2*np.pi*np.fft.fftfreq(L);m=np.arange(-30,31)
    def winding(r):
        weights=np.exp(-r*period**2*m*m/(2*T));weights/=weights.sum()
        y=r-r*r*period**2*np.dot(m*m,weights)/T
        f=np.cos(2*period*np.arange(L)[:,None]*m[None,:]/L)@weights
        return y,f,1-weights[30]
    prob=0
    if period is not None:
        rho=brentq(lambda r:winding(r)[0]-rho,.02*T,rho+10*T,xtol=1e-12)
        _,w,prob=winding(rho)
    lam=4*np.sin(k[:,None]/2)**2+4*np.sin(k[None,:]/2)**2;lam[0,0]=np.inf
    green=np.fft.ifft2(1/lam).real
    c=np.exp(-4*T/rho*(green[0,0]-green))
    if period is not None:c=c*w[:,None]*w[None,:]
    return c,rho,prob


def gaussian_comparison(point,deletes,T,probes,L,gd,profile_T):
    rows=[]
    for p,j in enumerate(probes):
        y=point['helicity'][j];c,rho,prob=gaussian(L,T[j],y,np.pi)
        if c is None:continue
        s=np.fft.fft2(c).real;sk=s[[1,-1,0,0],[0,0,1,-1]].mean()
        _,xg,_=correlation_ratio(s[0,0],sk,L)
        obs=point['full_spectra'][j,1];o0=obs[0];ok=obs[1:].mean()
        C0=point['profile_full'][p,1,:,0].real.mean()
        amp=(o0-ok)/(s[0,0]-sk);white=ok-amp*sk
        gaps=[];white_c0=[]
        for b in range(len(deletes['helicity'])):
            cb,_,_=gaussian(L,T[j],deletes['helicity'][b,j],np.pi)
            if cb is None:gaps.append(np.nan);white_c0.append(np.nan);continue
            sb=np.fft.fft2(cb).real;skb=sb[[1,-1,0,0],[0,0,1,-1]].mean()
            _,xgb,_=correlation_ratio(sb[0,0],skb,L)
            gaps.append(deletes['full_xi_xy'][b,j,1]-xgb)
            ob=deletes['full_spectra'][b,j,1];ab=(ob[0]-ob[1:].mean())/(sb[0,0]-skb)
            wb=ob[1:].mean()-ab*skb
            white_c0.append(ab+wb-deletes['profile_full'][b,p,1,:,0].real.mean())
        B=len(gaps)
        rows.append(dict(g_d=gd,L=L,T=T[j],Y=y,rho=rho,half_winding_probability=prob,
            xi_P_over_L=point['full_xi_xy'][j,1],xi_gaussian=float(xg),gap=point['full_xi_xy'][j,1]-xg,
            gap_se=np.sqrt((B-1)*np.var(gaps)),measured_C0=C0,signal_amplitude=amp,white_background=white,
            required_C0=amp+white,C0_excess=amp+white-C0,C0_excess_se=np.sqrt((B-1)*np.var(white_c0)),
            required_to_measured_C0=(amp+white)/C0,illustrative_background_area=white/(C0-amp)))
    return rows


def main():
    curves=[];spatial=[];stability=[];hist=[];profiles=[];gauss=[];components=[];comparisons=[]
    for path in sorted(OUT.glob('gd=*_L=*.json')):
        meta=json.loads(path.read_text());L=meta['L'];gd=meta['model_parameters']['g_d'];tag=path.stem
        with np.load(path.with_suffix('.npz')) as z:
            T=z['T'];probes=z['profile_indices'];pt=z['profile_T'];blocks={k:z[k] for k in z.files if k not in ('T','profile_T','profile_indices') and not k.startswith('pt_')}
        print('Analyzing',tag,flush=True)
        v,e,jk=estimate(blocks,T,L)
        np.savez_compressed(OUT/(tag+'_summary.npz'),T=T,profile_T=pt,sectors=np.asarray(NAMES),
            profile_sectors=np.asarray(PROFILE_NAMES),**v,**{k+'_se':x for k,x in e.items()},
            jk_Y=jk['helicity'],jk_xi=jk['full_xi_xy'],jk_S=jk['full_spectra'],
            jk_binder_chirality=jk['binder_chirality'],jk_binder_phase=jk['binder_phase'])
        scalar_keys=[k for k,x in v.items() if x.shape==T.shape]
        for j,t in enumerate(T):
            row=dict(g_d=gd,L=L,T=t)
            for k in scalar_keys:row[k]=v[k][j];row[k+'_se']=e[k][j]
            curves.append(row)
            for c,name in enumerate(NAMES):
                row=dict(g_d=gd,L=L,T=t,sector=name)
                for label in ('full','connected'):
                    for key in ('xi_xy','xi_x','xi_y','ratio_xy'):
                        row[label+'_'+key]=v[label+'_'+key][j,c];row[label+'_'+key+'_se']=e[label+'_'+key][j,c]
                    row[label+'_S0']=v[label+'_spectra'][j,c,0];row[label+'_S0_se']=e[label+'_spectra'][j,c,0]
                    row[label+'_Sk']=v[label+'_spectra'][j,c,1:].mean()
                spatial.append(row)
            for c in range(5):components.append(dict(g_d=gd,L=L,T=t,q=c+1,mean=v['Q_mean'][j,c],mean_se=e['Q_mean'][j,c],
                S0_full=v['full_Q_components'][j,c,0],S0_connected=v['connected_Q_components'][j,c,0]))
            for name in ('epsilon_hist_count','epsilon_hist_density','orbital_hist_count','spin_hist_count'):
                for b in range(12):hist.append(dict(g_d=gd,L=L,T=t,histogram=name,bin=b,fraction=v[name][j,b],se=e[name][j,b]))
        select=np.unique(np.r_[probes,[np.argmin(abs(T-t)) for t in (.3815,.3825,.384,.385,.387)]])
        metrics=['helicity','D','S','energy','signed_chirality','chirality','spin_alignment','epsilon','W_per_site']
        for factor in (1,2,4,6,10):
            vv,ee,_=estimate(blocks,T,L,factor)
            for j in select:
                for k in metrics:stability.append(dict(g_d=gd,L=L,T=T[j],metric=k,subset='full',block_sweeps=factor*50_000,mean=vv[k][j],se=ee[k][j]))
        halves=[]
        for subset,portion in [('first',slice(0,30)),('second',slice(30,None))]:
            vv,ee,_=estimate(blocks,T,L,2,portion);halves.append((vv,ee))
            for j in select:
                for k in metrics:stability.append(dict(g_d=gd,L=L,T=T[j],metric=k,subset=subset,block_sweeps=100_000,mean=vv[k][j],se=ee[k][j]))
        for p,t in enumerate(pt):
            for c,name in enumerate(PROFILE_NAMES):
                for d,direction in enumerate(('x','y')):
                    for r in range(L//2+1):
                        row=dict(g_d=gd,L=L,T=t,sector=name,direction=direction,r=r)
                        for label in ('full','connected'):
                            row[label]=v['profile_'+label][p,c,d,r].real;row[label+'_se']=e['profile_'+label][p,c,d,r].real
                            row[label+'_imag']=v['profile_'+label][p,c,d,r].imag;row[label+'_imag_se']=e['profile_'+label][p,c,d,r].imag
                        profiles.append(row)
        gauss.extend(gaussian_comparison(v,jk,T,probes,L,gd,pt))
        oldpath=ROOT/f'exploration/he3_transition_study/results/curves_gd={gd:g}_L={L}.npz'
        with np.load(oldpath) as old:
            ot=old['T'];ok=list(old['keys']);op=old['point'];oj=old['jk'];oe=np.sqrt((len(oj)-1)*np.var(oj,axis=0))
            for t in pt[1:4]:
                j=np.argmin(abs(T-t))
                for key in ('helicity','energy','chirality','specific_heat','spin_alignment','binder_chirality','binder_phase'):
                    k=ok.index(key)
                    if not ot[0]<=t<=ot[-1]:continue
                    oldvalue=np.interp(t,ot,op[k]);oldse=np.interp(t,ot,oe[k]);se=np.hypot(e[key][j],oldse)
                    comparisons.append(dict(g_d=gd,L=L,T=t,observable=key,new=v[key][j],new_se=e[key][j],old=oldvalue,old_se=oldse,
                        difference=v[key][j]-oldvalue,combined_se=se,z=(v[key][j]-oldvalue)/se))
    for name,rows in [('curves',curves),('spatial',spatial),('stability',stability),('histograms',hist),('profiles',profiles),
                      ('gaussian_checks',gauss),('Q_components',components),('old_comparison',comparisons)]:csv_write(OUT/(name+'.csv'),rows)


if __name__=='__main__':main()
