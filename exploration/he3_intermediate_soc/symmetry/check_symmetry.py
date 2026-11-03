"""Independent finite-twist and symmetry controls; no MC or production writes."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from exploration.he3_hqv_stage2.energy import BETAS, energy
from models.he3.spatial import local_fields


def source_functions():
    path=ROOT/'models/he3/update.py';source=path.read_text(encoding='utf-8-sig')
    tree=ast.parse(source);wanted={'norm2','dipole_invariant','potential','gradient'}
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    for n in funcs:n.decorator_list=[]
    ns={'float32':np.float32}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path),'exec'),ns)
    return ns,hashlib.sha256(path.read_bytes()).hexdigest()


def source_energy(a,qx,qy,gd,ns):
    # Differentiate the actual production scalar polynomial, not a guessed tensor.
    dx=np.exp(-1j*qx)*np.roll(a,-1,axis=0)-a
    dy=np.exp(-1j*qy)*np.roll(a,-1,axis=1)-a
    potential=sum(ns['potential'](x,BETAS,gd) for x in a.reshape(-1,3,2))
    gradient=ns['gradient'](dx[...,0],dx[...,1],dy[...,0],dy[...,1]).sum()
    return float(potential+gradient)


def twist_observables(a):
    xp,yp=np.roll(a,-1,axis=0),np.roll(a,-1,axis=1)
    dx,dy=xp-a,yp-a
    zx=(3*dx[...,0].conj()+dy[...,1].conj())*xp[...,0]+(dx[...,1].conj()+dy[...,0].conj())*xp[...,1]
    zy=(dy[...,0].conj()+dx[...,1].conj())*yp[...,0]+(3*dy[...,1].conj()+dx[...,0].conj())*yp[...,1]
    return np.array([2*zx.imag.sum(),2*zy.imag.sum(),
        2*(3*abs(xp[...,0])**2+abs(xp[...,1])**2-zx.real).sum(),
        2*(abs(yp[...,0])**2+3*abs(yp[...,1])**2-zy.real).sum(),
        2*(xp[...,0].conj()*yp[...,1]+xp[...,1].conj()*yp[...,0]).real.sum()])


def finite_twist(a,gd,ns):
    estimates=[]
    for h in (.02,.01):
        f=lambda x,y:source_energy(a,x,y,gd,ns)
        f0=f(0,0);xp,xm,yp,ym=f(h,0),f(-h,0),f(0,h),f(0,-h)
        mixed=(f(h,h)-f(h,-h)-f(-h,h)+f(-h,-h))/(4*h*h)
        estimates.append(np.array([(xp-xm)/(2*h),(yp-ym)/(2*h),
            (xp-2*f0+xm)/h**2,(yp-2*f0+ym)/h**2,mixed]))
    return (4*estimates[1]-estimates[0])/3


def geometric_transform(a,R):
    # A'(r)=Rspin A(R^{-1}r) Rorb^T, with a proper spin extension.
    L=a.shape[0];indices=np.indices((L,L)).reshape(2,-1)
    old=R.T@indices;field=a[old[0]%L,old[1]%L].reshape(a.shape)
    S=np.zeros((3,3));S[:2,:2]=R;S[2,2]=round(np.linalg.det(R))
    return np.einsum('mu,xyua,ba->xymb',S,field,R)


def matrix(kx,ky):
    px,py=np.exp(1j*kx)-1,np.exp(1j*ky)-1
    return np.array([[3*abs(px)**2+abs(py)**2,2*(px.conjugate()*py).real],
                     [2*(px.conjugate()*py).real,abs(px)**2+3*abs(py)**2]])


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ns,sha=source_functions();rng=np.random.default_rng(30112026);L=8
    chiral=np.zeros((3,2),complex);chiral[2]=[1,1j];chiral/=np.sqrt(2)
    plus=np.zeros_like(chiral);plus[2]=[1,1];plus/=np.sqrt(2)
    minus=plus.copy();minus[:,1]*=-1
    planar=np.zeros_like(chiral);planar[0,1]=1/np.sqrt(2);planar[1,0]=-1/np.sqrt(2)
    arbitrary=rng.normal(size=(3,2))+1j*rng.normal(size=(3,2));arbitrary/=np.linalg.norm(arbitrary)
    uniform=dict(chiral_z=chiral,polar_plus_z=plus,polar_minus_z=minus,antisymmetric_planar=planar,arbitrary_uniform=arbitrary)
    fields={name:np.broadcast_to(a,(L,L,3,2)).copy() for name,a in uniform.items()}
    fields['inhomogeneous_full']=(rng.normal(size=(L,L,3,2))+1j*rng.normal(size=(L,L,3,2)))/3
    rows=[];checks=[];maximum=0.;direct=0.
    for name,a in fields.items():
        exact=twist_observables(a)/L**2
        for gd in (0.,.01,.1):
            measured=finite_twist(a,gd,ns)/L**2
            err=np.max(abs(exact-measured));maximum=max(maximum,float(err))
            direct=max(direct,abs(source_energy(a,0,0,gd,ns)-energy(a,gd=gd)['total']))
            checks.append(dict(field=name,g_d=gd,max_finite_twist_error_per_site=err))
        if name in uniform:
            a0=uniform[name];n=np.vdot(a0,a0).real;rho=2*(n*np.eye(2)+2*(a0.conj().T@a0).real)
            assert np.max(abs(rho-np.array([[exact[2],exact[4]],[exact[4],exact[3]]])))<1e-12
            f=local_fields(a0[None,None])[0,0]
            bpp=(f[7]+f[8]+2*f[9])/2;bmm=(f[7]+f[8]-2*f[9])/2
            rows.append(dict(field=name,density=n,chirality_per_density=f[0].real/n,
                spin_alignment=(3*sum(abs(a0[2])**2)/n-1)/2,
                abs_P=abs(f[1]),abs_B_plus=abs(bpp),abs_B_minus=abs(bmm),
                rho_xx=rho[0,0],rho_yy=rho[1,1],rho_xy=rho[0,1],
                eigen_low=np.linalg.eigvalsh(rho)[0],eigen_high=np.linalg.eigvalsh(rho)[1],
                geometric_stiffness=np.sqrt(np.linalg.det(rho))))
    symmetry=[];a=fields['inhomogeneous_full']
    operations={'identity':np.eye(2,dtype=int),'swap_xy':np.array([[0,1],[1,0]]),
        'inversion':-np.eye(2,dtype=int),'antidiagonal':np.array([[0,-1],[-1,0]]),
        'mirror_x':np.diag([-1,1]),'mirror_y':np.diag([1,-1]),'quarter_turn':np.array([[0,-1],[1,0]])}
    for gd in (0.,.01,.1):
        e0=energy(a,gd=gd)
        for name,R in operations.items():
            transformed=geometric_transform(a,R);ee=energy(transformed,gd=gd)
            symmetry.append(dict(operation=name,g_d=gd,onsite_difference=ee['onsite']-e0['onsite'],
                soc_difference=ee['dipole']-e0['dipole'],gradient_difference=ee['gradient']-e0['gradient']))
        for name,aa in [('conjugation',a.conj()),('phase',a*np.exp(.37j))]:
            ee=energy(aa,gd=gd)
            symmetry.append(dict(operation=name,g_d=gd,onsite_difference=ee['onsite']-e0['onsite'],
                soc_difference=ee['dipole']-e0['dipole'],gradient_difference=ee['gradient']-e0['gradient']))
    kx,ky=.7,1.1
    mirror=dict(kx=kx,ky=ky,original_eigenvalues=np.linalg.eigvalsh(matrix(kx,ky)).tolist(),
        mirrored_eigenvalues=np.linalg.eigvalsh(matrix(-kx,ky)).tolist(),
        cross_sum=float(matrix(kx,ky)[0,1]+matrix(-kx,ky)[0,1]),
        expected_cross_sum=float(4*(1-np.cos(kx))*(1-np.cos(ky))))
    # Conditional fixed-polar phase-only control, not free energy of full GL minima.
    waves=[]
    for Ls in (16,32,64,128,256):
        k=2*np.pi*np.fft.fftfreq(Ls);kx,ky=np.meshgrid(k,k,indexing='ij')
        px,py=np.exp(1j*kx)-1,np.exp(1j*ky)-1
        s=abs(px)**2+abs(py)**2;c=2*(px.conj()*py).real
        valid=s>0
        delta=.5*np.log((2*s[valid]+c[valid])/(2*s[valid]-c[valid])).sum()/Ls**2
        waves.append(dict(L=Ls,f_plus_minus_f_minus_per_T_site=delta))
    write_csv(HERE/'finite_twist_checks.csv',checks);write_csv(HERE/'uniform_tensors.csv',rows)
    write_csv(HERE/'symmetry_controls.csv',symmetry);write_csv(HERE/'fixed_polar_phase_control.csv',waves)
    output=dict(source_sha256=sha,finite_twist_max_error_per_site=maximum,
        full_source_energy_max_error=direct,mirror_spectral_counterexample=mirror)
    assert maximum<2e-7 and direct<1e-10
    (HERE/'checks.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
