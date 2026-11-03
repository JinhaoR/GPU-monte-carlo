"""Offline ladder comparisons and conditional cost scenarios; no GL sampling."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json
import numpy as np
from scipy.special import erfc,logsumexp
from scipy.ndimage import maximum_filter1d,gaussian_filter1d
from extract import ROOT,OUT,write
from transport import null_transport
from exploration.he3_transition_L96_160.analyze import solve
from utils.temperature_ladder import make_temperature_ladder

def density_grid(T,density,R,lo,hi):
    keep=(T>=lo)&(T<=hi);t=T[keep];v=density[keep];s=np.r_[0,np.cumsum((v[1:]+v[:-1])*np.diff(t)/2)];s/=s[-1]
    return np.interp(np.linspace(0,1,R),s,t)

def predicted_acceptance(tt,grid,cv,L):
    probes=tt[:-1,None]+np.diff(tt)[:,None]*np.linspace(0,1,17)
    sig=np.max(L*probes*np.sqrt(np.interp(probes,grid,cv)),axis=1)
    return erfc(abs(np.diff(1/tt))*sig/2)

def fine_cv(gd,d):
    T=d['T'];k=np.argmax(d['Cv']);ids=np.arange(k-6,k+7)
    with np.load(ROOT/f'data/weak_soc_spatial_simple_01/P=20_gd={gd:g}_gaussian/L=256.npz') as z:E=z['L256_energy'][:,ids]
    ref=E.mean();E-=ref;edges=np.linspace(E.min()-1e-6,E.max()+1e-6,801);centers=(edges[:-1]+edges[1:])/2
    H=np.array([np.histogram(E[:,j],edges)[0] for j in range(len(ids))]);h=H.sum(0);S=np.array([h*centers,h*centers**2])
    f=solve(H,S,(1/T[ids]).astype(np.float32).astype(float),centers);tt=np.linspace(T[ids[0]],T[ids[-1]],501)
    lw=-(1/tt[:,None]-f['beta_ref'])*f['e'][None,:]-f['ld'][None,:];w=np.exp(lw-logsumexp(lw+np.log(f['h'])[None,:],axis=1)[:,None]);mom=f['S']@w.T
    return tt,(mom[1]-mom[0]**2)/(256**2*tt*tt)

def main():
    inv=list(csv.DictReader((OUT/'inventory.csv').open()));transport=list(csv.DictReader((OUT/'transport_summary.csv').open()))
    grid=np.linspace(.37,.42,20001);sources={};peaks=[]
    for gd in (0,.01):
        with np.load(OUT/f'new_gd={gd:g}_L=256.npz') as z:d=dict(z)
        with np.load(OUT/f'new_gd={gd:g}_L=256_transport.npz') as z:tr=dict(z)
        T=d['T'];fineT,fineC=fine_cv(gd,d);raw=np.interp(grid,T,d['Cv']);mask=(grid>=fineT[0])&(grid<=fineT[-1]);raw[mask]=np.interp(grid[mask],fineT,fineC)
        e=d['block_e'].reshape(30,2,-1).mean(1);e2=d['block_e2'].reshape(30,2,-1).mean(1)
        jk1=(30*e.mean(0)-e)/29;jk2=(30*e2.mean(0)-e2)/29;cvjk=256**2*(jk2-jk1**2)/T**2;cvse=np.sqrt(29*np.var(cvjk,axis=0))
        tc=fineT[np.argmax(fineC)];near=abs(grid-tc)<.0015
        envelope=raw+2*np.interp(grid,T,cvse);envelope[near]=maximum_filter1d(envelope,size=241,mode='nearest')[near]
        flow=gaussian_filter1d(tr['flow'],1.0);drop=np.maximum(0,-np.diff(flow));drop/=drop.sum();dt=np.diff(T);mid=(T[1:]+T[:-1])/2
        eta=np.sqrt(drop)/dt;eta/=np.sum(eta*dt);fb=np.interp(grid,mid,eta)
        m=next(m for m in inv if m['campaign']=='new' and float(m['g_d'])==gd and int(m['L'])==256)
        ts=next(t for t in transport if t['campaign']=='new' and float(t['g_d'])==gd and int(t['L'])==256)
        growth=np.exp(-((grid-tc)/.0015)**4)
        sources[gd]=dict(T=T,cv=raw,envelope=envelope,growth=growth,feedback=fb,drop=drop,flow=tr['flow'],mid=mid,dt=dt,meta=m,transport=ts)
        peaks.append(dict(g_d=gd,Cv_grid_peak=float(d['Cv'].max()),Cv_reweighted_peak=float(fineC.max()),reweighted_peak_T=float(tc),max_Cv_SE=float(cvse.max())))
    write('refined_energy_peaks.csv',peaks)
    rows=[];candidate_arrays={}
    for L in (256,320,384,512):
        for scenario in ('fixed_Cv','area_Cv_critical'):
            cv={g:s['envelope'].copy() for g,s in sources.items()}
            if scenario=='area_Cv_critical':
                # Smoothly taper the stress case into the noncritical background;
                # an abrupt cutoff would invent an energy-overlap bottleneck.
                for g,s in sources.items():cv[g]*=1+((L/256)**2-1)*s['growth']
            energy=np.maximum.reduce([np.sqrt(c)/grid for c in cv.values()]);energy/=np.trapezoid(energy,grid)
            feedback=np.maximum.reduce([s['feedback'] for s in sources.values()]);feedback/=np.trapezoid(feedback,grid)
            # Fifteen alternatives using the unmodified existing ladder function.
            # Center and widths follow the two measured peak positions. Select on
            # worst-coupling acceptance-only RT, not on measured physical mixing.
            center=np.mean([p['reweighted_peak_T'] for p in peaks]);span=abs(peaks[1]['reweighted_peak_T']-peaks[0]['reweighted_peak_T']);simple=[]
            for width in span*np.array([1.2,1.6,2.]):
                for fraction in (.25,.35,.45,.55,.65):
                    t0=make_temperature_ladder(.37,.42,200,center,width,fraction)
                    score=max(null_transport(np.maximum(predicted_acceptance(t0,grid,c,L),1e-10),5)[1] for c in cv.values())
                    simple.append((score,width,fraction,t0))
            _,simple_width,simple_fraction,simple_t=min(simple,key=lambda x:x[0])
            designs=[('current',200,0.,.37,.42),('simple_best',200,0.,.37,.42),('energy',160,0.,.37,.42),('energy',200,0.,.37,.42),('separate_energy',200,0.,.37,.42),('blend25',200,.25,.37,.42),('blend50',200,.5,.37,.42),('blend50',224,.5,.37,.42),('blend50',256,.5,.37,.42),('blend50',300,.5,.37,.42),('feedback',200,1.,.37,.42),('narrow_energy',160,0.,.375,.405),('narrow_blend50',200,.5,.375,.405)]
            for name,R,alpha,lo,hi in designs:
                density=(1-alpha)*energy+alpha*feedback
                tt=sources[0]['T'] if name=='current' else density_grid(grid,density,R,lo,hi)
                if name=='simple_best':tt=simple_t
                if L==320 and scenario=='area_Cv_critical' and name in ('current','simple_best','energy','blend25','blend50','narrow_energy','narrow_blend50'):
                    candidate_arrays[f'{name}_R{R}']=tt
                for g,s in sources.items():
                    if name=='separate_energy':tt=density_grid(grid,np.sqrt(cv[g])/grid,R,lo,hi)
                    a=predicted_acceptance(tt,grid,cv[g],L)
                    _,null,_=null_transport(np.maximum(a,1e-10),5)
                    # Fixed-D feedback gain is only a geometric surrogate, not a
                    # prediction that physical memory survives ladder changes.
                    eta_old=1/((len(s['T'])-1)*s['dt']);idx=np.clip(np.searchsorted(tt,s['mid'],side='right')-1,0,R-2);eta_new=1/((R-1)*np.diff(tt)[idx]);keep=(s['mid']>=lo)&(s['mid']<=hi)
                    proxy=np.sum(s['drop'][keep]*eta_old[keep]/eta_new[keep])
                    actual=float(s['transport']['passage_flux_time']);baseline=float(s['transport']['null_alternating_RT'])
                    estimate=actual*null/baseline
                    rows.append(dict(L=L,scenario=scenario,design=name,R=R,g_d=g,T_min=lo,T_max=hi,min_spacing=np.diff(tt).min(),min_predicted_acceptance=a.min(),mean_predicted_acceptance=a.mean(),null_RT=null,
                        R_times_null_RT=R*null,flow_fixed_D_ratio=proxy,calibrated_flux_fixed_memory=estimate,calibrated_flux_growing_memory=estimate*(L/256),
                        simple_Tc=center if name=='simple_best' else np.nan,simple_window=simple_width if name=='simple_best' else np.nan,simple_fraction=simple_fraction if name=='simple_best' else np.nan))
    write('ladder_predictions.csv',rows)
    for name,t in candidate_arrays.items():np.savetxt(OUT/(name+'_candidate.txt'),t,fmt='%.12f')
    np.savez_compressed(OUT/'design_profiles.npz',T=grid,**{f'cv_{g:g}':s['cv'] for g,s in sources.items()},**{f'envelope_{g:g}':s['envelope'] for g,s in sources.items()},**{f'feedback_{g:g}':s['feedback'] for g,s in sources.items()})
    # Measured timing trend is steeper than area scaling: bracket with both.
    old160=next(m for m in inv if m['campaign']=='new' and float(m['g_d'])==0 and int(m['L'])==160)
    ref=sources[0]['meta'];sec256=float(ref['gpu_seconds'])/6e6;sec160=float(old160['gpu_seconds'])/6e6;ztime=np.log(sec256/sec160)/np.log(256/160)
    costs=[]
    for L,R in ((256,200),(320,160),(320,200),(320,224),(384,160),(384,200),(384,224),(384,256),(512,160),(512,256),(512,300)):
        low=sec256*(L/256)**2*(R/200);high=sec256*(L/256)**ztime*(R/200)
        for total,production in ((6_000_000,3_000_000),(16_000_000,8_000_000),(24_000_000,12_000_000),(32_000_000,16_000_000)):
            # Existing cadences: scalar50, spatial1000, profile10000, history100.
            scalar=10*8*R*production/50;spatial=(16*5*16+68*8+4+8)*R*production/1000
            profile=(7*2*(L//2+1)*16+7*16+4+8)*5*production/10000
            history=4*R*total/100;payload=scalar+spatial+profile;gpu_core=88*R*L*L
            costs.append(dict(L=L,R=R,total_sweeps=total,production_sweeps=production,seconds_per_sweep_area=low,seconds_per_sweep_recent=high,hours_area=total*low/3600,hours_recent=total*high/3600,
                max_sweeps_72h_recent=72*3600/high,GPU_core_GiB=gpu_core/2**30,GPU_history_GiB=history/2**30,raw_measurement_GiB=payload/2**30,
                conservative_host_array_peak_GiB=(2*payload+history*4)/2**30,compressed_output_reference_GiB=float(ref['file_bytes'])/2**30*(production/3e6)*(R/200),timing_local_exponent=ztime))
    write('resource_scenarios.csv',costs)
    # Gaussian autoregression toy: exact fixed-neighbour Gaussian, no MH rejection.
    toy=[]
    for rho in (-.95,-.9,-.7,0.):toy.append(dict(rho=rho,noise_variance_fraction=1-rho*rho,linear_tau=(1+rho)/(2*(1-rho)),quadratic_tau=(1+rho*rho)/(2*(1-rho*rho))))
    write('Gaussian_AR_reference.csv',toy)
    print('Ladder and resource scenarios complete; timing exponent',ztime,flush=True)

if __name__=='__main__':main()
