"""Read-only finite-torus and operator checks on the two existing L64 pilots.
No simulation, fit to an arbitrary decay exponent, or production-code changes.
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from models.he3.spatial import local_fields, chiral_distance
HERE = Path(__file__).resolve().parent
OUT = HERE/'correlations'
PILOT = ROOT/'exploration/he3_spatial_pilot/results'
L=64; N=L*L
KGRID=2*np.pi*np.fft.fftfreq(L)
LX=4*np.sin(KGRID[:,None]/2)**2
LY=4*np.sin(KGRID[None,:]/2)**2


def xi_from_spectrum(s):
    sk=(s[1,0]+s[-1,0]+s[0,1]+s[0,-1]).real/4
    ratio=s[0,0].real/sk
    return np.sqrt(ratio-1)/(2*L*np.sin(np.pi/L)) if ratio>=1 and sk>0 else np.nan


def harmonic(T, rx, ry=None, rxy=0., stencil='forward'):
    if ry is None:ry=rx
    if rx<=0 or ry<=0 or rx*ry<=rxy*rxy:return None
    if stencil=='forward':
        px=np.exp(1j*KGRID[:,None])-1;py=np.exp(1j*KGRID[None,:])-1
        cross=(px.conj()*py).real
    else:cross=np.sin(KGRID[:,None])*np.sin(KGRID[None,:])
    lam=rx*LX+ry*LY+2*rxy*cross;lam[0,0]=np.inf
    green=np.fft.ifft2(1/lam).real
    return np.exp(-4*T*(green[0,0]-green))


def winding(rho,T,period):
    m=np.arange(-30,31)
    weights=np.exp(-rho*period**2*m*m/(2*T));weights/=weights.sum()
    variance=np.sum(m*m*weights)
    y=rho-rho*rho*period**2*variance/T
    factor=np.cos(2*period*m[:,None]*np.arange(L)[None,:]/L).T@weights
    return y,factor,float(1-weights[m==0][0])


def compact_reference(T, observed_y, period):
    if observed_y<=0:return None,np.nan,np.nan
    # The compact scalar theory predicts BOTH torus response and pair correlation.
    # Solve its response identity instead of treating the measured Y as bare rho.
    rho=brentq(lambda r:winding(r,T,period)[0]-observed_y,.02*T,observed_y+10*T,xtol=1e-13)
    y,w,prob=winding(rho,T,period)
    assert abs(y-observed_y)<1e-10
    return harmonic(T,rho)*w[:,None]*w[None,:],rho,prob


def measured(g,a,T):
    yx=(g[:,6].mean()-g[:,5].var()/T)/N
    yy=(g[:,8].mean()-g[:,7].var()/T)/N
    yxy_cov=np.mean((g[:,5]-g[:,5].mean())*(g[:,7]-g[:,7].mean()))/(T*N)
    power=np.mean(abs(a[:,1])**2,axis=0)/N
    ratio=power[0]/power[1:].mean()
    xi=np.sqrt(ratio-1)/(2*L*np.sin(np.pi/L)) if ratio>=1 else np.nan
    return yx,yy,yxy_cov,power,xi


def write_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def snapshot_checks():
    rows=[]; spectra={}; cross=[]
    for history in ('ordered','hot'):
        for path in sorted((PILOT/history).glob('snapshot_*.npz')):
            with np.load(path) as z:
                if int(z['sweep'])<=100000:continue  # exclude burn-boundary fields
                fields=z['fields'];temps=z['temperatures'];old=z['global_values'];sweep=int(z['sweep'])
            for j,T in enumerate(temps):
                a=fields[j].astype(complex);f=local_fields(a);P=f[...,1]
                n=np.sum(abs(a)**2,axis=(-2,-1));d=chiral_distance(a)
                xp,yp=np.roll(a,-1,axis=0),np.roll(a,-1,axis=1)
                dx,dy=xp-a,yp-a
                g1=np.sum(abs(dx)**2+abs(dy)**2,axis=(-2,-1))
                g2=np.sum(abs(dx[...,0]+dy[...,1])**2,axis=-1)
                g3=np.sum(abs(dx[...,0])**2+abs(dy[...,1])**2+2*(dx[...,1].conj()*dy[...,0]).real,axis=-1)
                grad=np.sum(g1+g2+g3)
                W=np.sum(((xp[...,0]-yp[...,0]).conj()*(xp[...,1]-yp[...,1])).real)
                kxy=2*np.sum((xp[...,0].conj()*yp[...,1]+xp[...,1].conj()*yp[...,0]).real)
                D=(old[j,6]+old[j,8])/(2*N)
                residual=D-(4*n.mean()-grad/(2*N)+W/N)
                assert abs(residual)<1e-6
                good=abs(P)>1e-12*n
                phase=np.divide(P,abs(P),out=np.zeros_like(P),where=good)
                b0=(f[...,7]+f[...,8])/np.sqrt(2);b2=P/np.sqrt(2);bxy=np.sqrt(2)*f[...,9]
                operators={'P':P[...,None],'P_over_n':(P/n)[...,None], 'P_unit_phase':phase[...,None],
                           'B_trace':b0[...,None],'B_difference':b2[...,None],'B_xy':bxy[...,None],
                           'B_full':np.stack((b0,b2,bxy),-1),'A':f[...,10:],'Q':f[...,2:7]}
                for name,o in operators.items():
                    ft=np.fft.fftn(o,axes=(0,1));sp=np.sum(abs(ft)**2,axis=-1)/N
                    spectra.setdefault((history,float(T),name),[]).append(sp)
                    spectra.setdefault(('pooled',float(T),name),[]).append(sp)
                rows.append(dict(history=history,sweep=sweep,T=float(T),C0=float(np.mean(abs(P)**2)),
                    density=float(n.mean()),epsilon=float(np.sum(n*d['epsilon'])/n.sum()),
                    orbital_loss=float(np.sum(n*d['orbital_loss'])/n.sum()),
                    spin_loss=float(np.sum(n*d['spin_loss'])/n.sum()),
                    curvature=D,gradient_per_site=grad/N,Wdiag_per_site=W/N,Kxy_per_site=kxy/N,
                    curvature_identity_residual=residual,excluded_phase_fraction=float(1-good.mean())))
    powers=[]
    for (history,T,name),values in spectra.items():
        s=np.mean(values,axis=0)
        powers.append(dict(history=history,T=T,operator=name,snapshots=len(values),S0=s[0,0],
            Sk=(s[1,0]+s[-1,0]+s[0,1]+s[0,-1])/4,xi_over_L=xi_from_spectrum(s),C0=float(s.sum()/N)))
    write_csv('snapshot_structure.csv',rows);write_csv('snapshot_operator_spectra.csv',powers)
    return rows,spectra


def main():
    OUT.mkdir(exist_ok=True)
    snapshots,spectra=snapshot_checks();rows=[];backgrounds=[];profiles=[]
    legacy=list(csv.DictReader((PILOT/'gaussian_reference.csv').open()))
    selected=(.37,.38,.3815,.3825,.385)
    for history in ('ordered','hot'):
        with np.load(PILOT/history/'records.npz') as z:
            ts=z['temperatures'];g=z['global_values'][z['global_sweeps']>100000];a=z['amplitudes'][z['spatial_sweeps']>100000]
        assert len(g)==2000 and len(a)==400
        for requested in selected:
            j=int(np.argmin(abs(ts-requested)));T=float(ts[j]);yx,yy,cov,observed,xi=measured(g[:,j],a[:,j],T);Y=(yx+yy)/2
            c0=harmonic(T,Y);refs={'zero_winding':c0,'diagonal_anisotropy':harmonic(T,yx,yy)}
            ri, rhoi, pi=compact_reference(T,Y,2*np.pi);rp,rhop,pp=compact_reference(T,Y,np.pi)
            refs.update(integer_windings=ri,paired_windings=rp)
            match=next(r for r in legacy if r['history']==history and abs(float(r['temperature'])-T)<1e-12)
            np.testing.assert_allclose(xi,float(match['xiP_over_L_measured']),atol=2e-13)
            np.testing.assert_allclose(xi_from_spectrum(np.fft.fft2(c0)),float(match['xiP_over_L_gaussian']),atol=2e-13)
            snap=[s for s in snapshots if s['history']==history and abs(s['T']-T)<1e-12]
            rxy=np.nan
            if snap:
                rxy=float(np.mean([s['Kxy_per_site'] for s in snap])-cov)
                refs['tensor_forward']=harmonic(T,yx,yy,rxy,'forward')
                refs['tensor_centered']=harmonic(T,yx,yy,rxy,'centered')
            row=dict(history=history,T=T,Y=Y,Yx=yx,Yy=yy,Yxy_sparse=rxy,xi_measured=xi,
                     rho_integer=rhoi,rho_paired=rhop,p_integer_winding=pi,p_paired_winding=pp)
            for name,c in refs.items():row['xi_'+name]=xi_from_spectrum(np.fft.fft2(c)) if c is not None else np.nan
            # Joint delete-one 5000-sweep intervals, the same as the old pilot.
            deleted=[];deleted_xi=[]
            for b in range(20):
                gg=np.concatenate((g[:100*b,j],g[100*(b+1):,j]));aa=np.concatenate((a[:20*b,j],a[20*(b+1):,j]))
                xx,yy2,_,_,xm=measured(gg,aa,T);yr=(xx+yy2)/2
                cc,_,_=compact_reference(T,yr,np.pi)
                deleted.append(xm-xi_from_spectrum(np.fft.fft2(cc)))
                deleted_xi.append(xm)
            row['paired_reference_gap']=xi-row['xi_paired_windings']
            row['paired_gap_jackknife_se']=float(np.sqrt(19*np.var(deleted)))
            row['xi_measured_se']=float(np.sqrt(19*np.var(deleted_xi)))
            rows.append(row)
            if snap:
                c0_actual=float(np.mean([s['C0'] for s in snap]));sk=observed[1:].mean()
                for name in ('zero_winding','paired_windings','tensor_forward'):
                    sr=np.fft.fft2(refs[name]).real;srk=(sr[1,0]+sr[-1,0]+sr[0,1]+sr[0,-1])/4
                    amplitude=(observed[0]-sk)/(sr[0,0]-srk);white=sk-amplitude*srk
                    backgrounds.append(dict(history=history,T=T,reference=name,signal_amplitude=amplitude,
                        required_white_variance=white,required_C0=amplitude+white,C0_snapshots=c0_actual,
                        C0_min=min(s['C0'] for s in snap),C0_max=max(s['C0'] for s in snap),
                        required_to_observed_C0=(amplitude+white)/c0_actual,
                        illustrative_background_area=white/(c0_actual-amplitude)))
                for distance in range(L//2+1):
                    record=dict(history=history,T=T,r=distance)
                    for name,c in refs.items():record['reference_'+name]=float((c[distance,0]+c[0,distance])/2)
                    for op in ('P','P_over_n','P_unit_phase','B_trace','B_xy','B_full','Q','A'):
                        s=np.mean(spectra[(history,T,op)],axis=0);c=np.fft.ifft2(s).real
                        record[op]=float((c[distance,0]+c[0,distance])/(2*c[0,0]))
                    profiles.append(record)
    # Optional tensor columns absent at temperatures without snapshots.
    fields=list(dict.fromkeys(k for r in rows for k in r))
    rows=[{k:r.get(k,np.nan) for k in fields} for r in rows]
    write_csv('finite_volume_references.csv',rows);write_csv('white_background_test.csv',backgrounds);write_csv('snapshot_profiles.csv',profiles)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for history,ax in zip(('ordered','hot'),axes):
        r=[r for r in rows if r['history']==history];t=[r['T'] for r in r]
        for key,label in [('xi_measured','measured P'),('xi_zero_winding','Gaussian'),('xi_paired_windings','compact paired'),('xi_tensor_forward','tensor (sparse Kxy)')]:
            if key=='xi_measured':
                ax.errorbar(t,[r[key] for r in r],yerr=[r['xi_measured_se'] for r in r],fmt='o-',label=label,ms=4,capsize=2)
            else:ax.plot(t,[r[key] for r in r],'o-',label=label,ms=4)
        ax.set(title=history,xlabel='T',ylabel='xi_P / L');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/'finite_volume_references.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for T,ax in zip((.37,.3815,.385),axes):
        for op in ('P','P_over_n','P_unit_phase','B_xy','B_full','Q'):
            # Pool raw powers before normalization, as for all second moments.
            spectrum=np.mean(spectra[('pooled',T,op)],axis=0)
            corr=np.fft.ifft2(spectrum).real
            vals=np.array([(corr[r,0]+corr[0,r])/(2*corr[0,0]) for r in range(L//2+1)])
            ax.plot(np.arange(1,L//2+1),vals[1:],label=op)
        ax.set(title=f'T={T}',xlabel='r',ylabel='normalized axial correlation');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Four saved production fields per T: descriptive operator comparison')
    fig.savefig(OUT/'snapshot_operator_profiles.png',dpi=170);plt.close(fig)
    result=dict(simulations_run=0,production_snapshots=16,fields_per_temperature=4,records_per_history=2000,
        spatial_events_per_history=400,legacy_reference_reproduced=True,
        snapshot_max_curvature_identity_error=max(abs(r['curvature_identity_residual']) for r in snapshots),
        half_windings_at_03815=[r for r in rows if r['T']==.3815])
    (OUT/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
