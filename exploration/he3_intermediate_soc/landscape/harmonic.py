"""Conditional Gaussian baseline about the two homogeneous onsite minima.

This integrates the exact lattice quadratic fluctuation operator, not Monte
Carlo or a finite-temperature effective potential. Both global phase zero
modes are omitted in the same orthonormal Cartesian-coordinate convention.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from onsite import HERE, state, unpack, pack, evaluate, hessian, energy_density, csv_write


def kinetic(L):
    k=2*np.pi*np.fft.fftfreq(L)
    px,py=np.meshgrid(np.exp(1j*k)-1,np.exp(1j*k)-1,indexing='ij')
    m=np.empty((L,L,2,2))
    m[...,0,0]=3*abs(px)**2+abs(py)**2;m[...,1,1]=abs(px)**2+3*abs(py)**2
    m[...,0,1]=m[...,1,0]=2*(px.conj()*py).real
    # Real coordinates contain six pairs: three spin rows, real then imaginary.
    return 2*np.einsum('ab,xyij->xyaibj',np.eye(6),m).reshape(L,L,12,12)


def check_kernel():
    rng=np.random.default_rng(8106);errors=[]
    for g in (.01,.1):
        for kind in ('chiral_z','planar_antisymmetric'):
            x=state(kind,g);L=6;v=rng.normal(0,.2,(L,L,12))
            a=unpack(v);base=unpack(x)
            ft=np.fft.fftn(v,axes=(0,1));hh=kinetic(L)+hessian(x,g)
            quadratic=.5*np.einsum('xyi,xyij,xyj->',ft.conj(),hh,ft).real/L**2
            values=[]
            zero=np.sum(energy_density(np.broadcast_to(base,(L,L,3,2)),gd=g)['total'])
            for eps in (.01,.005):
                eplus=np.sum(energy_density(base+eps*a,gd=g)['total'])
                eminus=np.sum(energy_density(base-eps*a,gd=g)['total'])
                values.append((eplus+eminus-2*zero)/(2*eps**2))
            errors.append(abs((4*values[1]-values[0])/3-quadratic))
    assert max(errors)<1e-7
    return dict(full_lattice_quadratic_energy_max_error=max(errors))


def main():
    checks=check_kernel();rows=[];spectra=[]
    for L in (32,64,128):
        kk=kinetic(L);N=L*L
        for nominal in (.01,.02,.04,.06,.08,.1):
            g=float(np.float32(nominal));values={}
            for kind in ('chiral_z','planar_antisymmetric'):
                x=state(kind,g);n=x@x
                eigen,vectors=np.linalg.eigh(kk+hessian(x,g))
                assert abs(eigen[0,0,0])<1e-10 and np.all(eigen.reshape(-1)[1:]>0)
                eigen[0,0,0]=np.inf # omit exactly one common global gauge mode
                inv=1/eigen;phase=pack(1j*unpack(x))/np.sqrt(n);radial=x/np.sqrt(n)
                phase_weights=np.einsum('...ij,i->...j',vectors,phase)**2
                radial_weights=np.einsum('...ij,i->...j',vectors,radial)**2
                logdet=float(np.log(eigen[np.isfinite(eigen)]).sum()/N)
                trace=float(inv.sum()/N);phase_trace=float((inv*phase_weights).sum()/N)
                radial_trace=float((inv*radial_weights).sum()/N)
                values[kind]=dict(energy=float(evaluate(x,g)[0]),logdet_per_site=logdet,
                    relative_variance_per_T=trace/n,
                    phase_variance_per_T=phase_trace/n,
                    nonphase_relative_variance_per_T=(trace-phase_trace)/n,
                    relative_density_variance_per_T=4*radial_trace/n)
                spectra.append(dict(g_d=nominal,L=L,state=kind,**values[kind]))
            a=values['chiral_z'];p=values['planar_antisymmetric']
            gap=p['energy']-a['energy'];slope=(p['logdet_per_site']-a['logdet_per_site'])/2
            rows.append(dict(g_d=nominal,L=L,zero_temperature_gap=gap,delta_F_per_T=slope,
                harmonic_equal_free_energy_T=-gap/slope if slope<0 else np.nan,
                delta_F_at_T028=gap+.28*slope,delta_F_at_T038=gap+.38*slope,
                delta_F_at_T040=gap+.4*slope))
            print('harmonic',nominal,L,'deltaF/T',slope,'formal crossing',-gap/slope,flush=True)
    csv_write('harmonic_free_energy.csv',rows);csv_write('harmonic_variances.csv',spectra)
    (HERE/'harmonic_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for L in (32,64,128):
        z=[r for r in rows if r['L']==L]
        axs[0].plot([r['g_d'] for r in z],[r['harmonic_equal_free_energy_T'] for r in z],'-o',label=f'L={L}')
    axs[0].set(xlabel='g_D',ylabel='formal harmonic crossing T',title='Low-temperature Gaussian extrapolation');axs[0].legend()
    for kind in ('chiral_z','planar_antisymmetric'):
        z=[r for r in spectra if r['L']==128 and r['state']==kind]
        axs[1].plot([r['g_d'] for r in z],[.38*r['nonphase_relative_variance_per_T'] for r in z],'-o',label=kind)
    axs[1].set(xlabel='g_D',ylabel='non-phase fluctuation norm / n',title='Expansion-size check at T=.38, L=128');axs[1].legend()
    fig.savefig(HERE/'harmonic_baseline.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
