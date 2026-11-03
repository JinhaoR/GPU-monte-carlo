"""Small deterministic full-field onsite landscape calculation; no Monte Carlo.

Real coordinates are Re(A).ravel(), Im(A).ravel(), with A of shape (3,2).
The exact quartic is represented as a sum of squared real quadratic forms,
which supplies analytic gradients and Hessians in all twelve coordinates.
"""
import ast
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize, root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from exploration.he3_hqv_stage2.energy import BETAS, energy_density
from models.he3.spatial import chiral_distance
C0=float(np.float32(2/3));BA=BETAS[1]+BETAS[3]+BETAS[4]
BP=BETAS[0]+BETAS[1]+(BETAS[2]+BETAS[3]+BETAS[4])/2


def pack(a):return np.r_[a.real.ravel(),a.imag.ravel()]
def unpack(x):return (np.asarray(x)[...,:6]+1j*np.asarray(x)[...,6:]).reshape(np.shape(x)[:-1]+(3,2))


def quadratics(x):
    a=unpack(x);ax,ay=a[:,0],a[:,1]
    xx,yy,xy=np.sum(ax*ax),np.sum(ay*ay),np.sum(ax*ay)
    o=np.vdot(ax,ay);nx=np.vdot(ax,ax).real;ny=np.vdot(ay,ay).real
    return np.array([(xx+yy).real,(xx+yy).imag,nx+ny,nx,ny,o.real,o.imag,
        xx.real,xx.imag,yy.real,yy.imag,xy.real,xy.imag])


def dipole(x):
    a=unpack(x)
    return abs(a[0,0]+a[1,1])**2+abs(a[0,0])**2+abs(a[1,1])**2+2*(a[0,1]*a[1,0].conjugate()).real-C0*np.vdot(a,a).real


EYE=np.eye(12)
Q=np.array([[(quadratics(u+v)-quadratics(u-v))/4 for v in EYE] for u in EYE]).transpose(2,0,1)
D=np.array([[(dipole(u+v)-dipole(u-v))/4 for v in EYE] for u in EYE])
b1,b2,b3,b4,b5=BETAS
W=np.array([b1,b1,b2,b3+b4,b3+b4,2*(b3+b4),2*(b4-b3),b5,b5,b5,b5,2*b5,2*b5])


def evaluate(x,g):
    qx=np.einsum('kij,...j->...ki',Q,x);q=np.einsum('...i,...ki->...k',x,qx)
    m=-2*EYE+g*D
    e=np.einsum('...i,ij,...j->...',x,m,x)+np.sum(W*q*q,axis=-1)
    grad=2*np.einsum('ij,...j->...i',m,x)+4*np.einsum('k,...k,...ki->...i',W,q,qx)
    return e,grad


def hessian(x,g):
    qx=np.einsum('kij,j->ki',Q,x);q=qx@x
    return 2*(-2*EYE+g*D)+4*np.einsum('k,k,kij->ij',W,q,Q)+8*np.einsum('k,ki,kj->ij',W,qx,qx)


def state(kind,g):
    a=np.zeros((3,2),complex)
    if kind=='chiral_z':
        n=(1+g*C0/2)/BA;a[2]=np.sqrt(n/2)*np.array([1,1j])
    elif kind=='planar_antisymmetric':
        n=(1+g*(1+C0)/2)/BP;a[0,1]=np.sqrt(n/2);a[1,0]=-np.sqrt(n/2)
    elif kind=='polar_plus_z':
        n=(1+g*C0/2)/BETAS.sum();a[2]=np.sqrt(n/2)
    else:raise ValueError(kind)
    return pack(a)


def describe(x,g):
    a=unpack(x);n=np.vdot(a,a).real;ap=(a[:,0]+a[:,1])/np.sqrt(2);am=(a[:,0]-a[:,1])/np.sqrt(2)
    ev=np.linalg.eigvalsh(hessian(x,g));v,grad=evaluate(x,g)
    eps=chiral_distance(a[None,None])['epsilon'].item()
    return dict(energy=float(v),density=n,dipole_per_density=dipole(x)/n,
        chirality=2*np.vdot(a[:,0],a[:,1]).imag/n,spin_alignment=(3*np.sum(abs(a[2])**2)/n-1)/2,
        chiral_residual=eps,A_plus_fraction=np.vdot(ap,ap).real/n,
        P_abs=abs(np.sum(a[:,0]**2-a[:,1]**2)),B_plus_abs=abs(np.sum(ap*ap)),B_minus_abs=abs(np.sum(am*am)),
        gradient_norm=float(np.linalg.norm(grad)),negative_modes=int(np.sum(ev< -1e-7)),zero_modes=int(np.sum(abs(ev)<=1e-7)),
        smallest_eigenvalue=float(ev[0]),smallest_positive_eigenvalue=float(ev[ev>1e-7][0]))


def csv_write(name,rows):
    with (HERE/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def validate():
    tree=ast.parse((ROOT/'models/he3/update.py').read_text(encoding='utf-8-sig'))
    fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('norm2','dipole_invariant','potential')]
    for f in fns:f.decorator_list=[]
    ns={'float32':np.float32};exec(compile(ast.Module(body=fns,type_ignores=[]),'production_onsite','exec'),ns)
    rng=np.random.default_rng(21061);errors=[];gradient_errors=[];hessian_errors=[]
    for g in (.01,.04,.1):
        for _ in range(10):
            x=rng.normal(0,.4,12);a=unpack(x);e,d=evaluate(x,g)
            errors.extend([abs(e-ns['potential'](a,BETAS,g)),abs(e-energy_density(a[None,None],gd=g)['total'].item())])
            h=1e-4
            central=(evaluate(x+EYE*h,g)[0]-evaluate(x-EYE*h,g)[0])/(2*h)
            central2=(evaluate(x+EYE*h/2,g)[0]-evaluate(x-EYE*h/2,g)[0])/h
            gradient_errors.append(np.max(abs((4*central2-central)/3-d)))
            hs=(evaluate(x+EYE*h,g)[1]-evaluate(x-EYE*h,g)[1])/(2*h)
            hs2=(evaluate(x+EYE*h/2,g)[1]-evaluate(x-EYE*h/2,g)[1])/h
            hessian_errors.append(np.max(abs((4*hs2-hs)/3-hessian(x,g))))
    answer=dict(energy_max_error=max(errors),gradient_richardson_max_error=max(gradient_errors),
        hessian_richardson_max_error=max(hessian_errors),dipole_eigenvalues=np.linalg.eigvalsh(D).tolist(),
        beta_A=BA,beta_P=BP,beta_polar=float(BETAS.sum()),C0=C0)
    assert max(errors)<1e-11 and max(gradient_errors)<1e-8 and max(hessian_errors)<1e-8
    return answer


def string(g,phase=0.,perturb=0.,nodes=41,iterations=12000):
    """Full 12-D steepest-descent string with fixed endpoint phase representatives."""
    a=state('chiral_z',g);b=pack(unpack(state('planar_antisymmetric',g))*np.exp(1j*phase))
    t=np.linspace(0,1,nodes);x=(1-t[:,None])*a+t[:,None]*b
    if perturb:x+=perturb*np.sin(np.pi*t[:,None])*np.random.default_rng(603).normal(size=x.shape)
    last=np.inf
    for it in range(iterations):
        e,grad=evaluate(x,g)
        tangent=x[2:]-x[:-2];tangent/=np.linalg.norm(tangent,axis=1)[:,None]
        perpendicular=grad[1:-1]-np.sum(grad[1:-1]*tangent,axis=1)[:,None]*tangent
        x[1:-1]-=.015*perpendicular
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(x,axis=0),axis=1))]
        x=np.array([np.interp(t*arc[-1],arc,x[:,i]) for i in range(12)]).T
        if it%100==0:
            change=abs(e.max()-last);last=e.max()
            if change<1e-12 and np.max(np.linalg.norm(perpendicular,axis=1))<2e-4:break
    e,grad=evaluate(x,g)
    saddle=root(lambda y:evaluate(y,g)[1],x[np.argmax(e)],jac=lambda y:hessian(y,g),method='lm',options={'ftol':1e-13,'xtol':1e-13,'gtol':1e-13,'maxiter':2000})
    return x,saddle.x,it+1


def segment_maximum(a,b,g):
    """Exact quartic along a line: include all real stationary points in [0,1]."""
    d=b-a;m=-2*EYE+g*D
    coefficients=np.zeros(5);coefficients[:3]=[a@m@a,2*a@m@d,d@m@d]
    for w,q in zip(W,Q):
        v=np.array([a@q@a,2*a@q@d,d@q@d]);coefficients+=w*np.convolve(v,v)
    roots=np.polynomial.polynomial.polyroots(np.arange(1,5)*coefficients[1:])
    t=[0.,1.]+[r.real for r in roots if abs(r.imag)<1e-9 and 0<r.real<1]
    return max(np.polynomial.polynomial.polyval(t,coefficients))


def connect(saddle,g,sign):
    """Break exact invariant-subspace trapping and retain a continuous path."""
    ev,vec=np.linalg.eigh(hessian(saddle,g))
    start=saddle+sign*.03*vec[:,0]+1e-5*np.random.default_rng(170+sign).normal(size=12)
    trajectory=[saddle.copy(),start.copy()]
    res=minimize(lambda y:evaluate(y,g),start,jac=True,method='BFGS',callback=lambda y:trajectory.append(y.copy()),
        options={'gtol':1e-9,'maxiter':1000})
    maximum=max(segment_maximum(a,b,g) for a,b in zip(trajectory[:-1],trajectory[1:]))
    return res.x,np.array(trajectory),maximum


def environment_checks():
    """One site in fixed uniform neighbors, distinct from a spatial barrier."""
    r=np.kron(np.eye(6),np.array([[8.,2.],[2.,8.]]));rows=[]
    for nominal in (.01,.02,.04,.06,.08,.1):
        g=float(np.float32(nominal))
        for kind,other in [('chiral_z','planar_antisymmetric'),('planar_antisymmetric','chiral_z')]:
            c=state(kind,g);x=state(other,g);difference=x-c
            uniform=np.broadcast_to(unpack(c),(5,5,3,2)).copy();changed=uniform.copy();changed[2,2]=unpack(x)
            before=energy_density(uniform,gd=g);after=energy_density(changed,gd=g)
            direct=float(np.sum(after['total']-before['total']))
            gradient=float(difference@r@difference);onsite=float(evaluate(x,g)[0]-evaluate(c,g)[0])
            def objective(y):
                e,d=evaluate(y,g);dy=y-c
                return e+dy@r@dy,d+2*r@dy
            result=minimize(objective,x,jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':500})
            error=abs(direct-onsite-gradient)
            assert error<1e-10 and np.linalg.norm(result.x-c)<1e-6
            rows.append(dict(g_d=nominal,environment=kind,replacement=other,onsite_energy_change=onsite,
                gradient_cost=gradient,total_single_site_cost=direct,identity_error=error,
                conditional_relaxation_distance_to_environment=float(np.linalg.norm(result.x-c))))
    csv_write('fixed_environment.csv',rows)


def main():
    checks=validate();rows=[];spectra=[];starts=[];paths=[];saddles=[];connections=[];fields={}
    rng=np.random.default_rng(20621101)
    couplings=(.01,.02,.04,.06,.08,.1)
    for nominal in couplings:
        g=float(np.float32(nominal))
        for kind in ('chiral_z','planar_antisymmetric','polar_plus_z'):
            x=state(kind,g);rows.append(dict(g_d=nominal,state=kind,**describe(x,g)))
            spectra.extend(dict(g_d=nominal,state=kind,mode=i,eigenvalue=e) for i,e in enumerate(np.linalg.eigvalsh(hessian(x,g))))
        seeds=[state('chiral_z',g),state('planar_antisymmetric',g),state('polar_plus_z',g)]
        seeds+=[seeds[i%3]+.12*rng.normal(size=12) for i in range(6)]
        seeds+=[rng.normal(0,.4,12) for _ in range(6)]
        for i,x in enumerate(seeds):
            result=minimize(lambda y:evaluate(y,g),x,jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':500})
            starts.append(dict(g_d=nominal,start=i,success=bool(result.success),**describe(result.x,g)))
        # A perturbed path can escape invariant ansatz subspaces.
        for label,phase,perturb in [('direct',0.,0.),('full_perturbed',0.,.03),('phase_rotated',np.pi/2,.03)]:
            images,saddle,iterations=string(g,phase,perturb)
            e,_=evaluate(images,g);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(images,axis=0),axis=1))]
            for j in range(len(e)):
                paths.append(dict(g_d=nominal,path=label,image=j,arc=arc[j]/arc[-1],energy=e[j]))
            d=describe(saddle,g);saddles.append(dict(g_d=nominal,path=label,string_iterations=iterations,
                image_max_energy=float(e.max()),barrier_from_chiral=d['energy']-evaluate(state('chiral_z',g),g)[0],
                barrier_from_planar=d['energy']-evaluate(state('planar_antisymmetric',g),g)[0],**d))
            fields[f'g={nominal:g}_{label}_images']=images;fields[f'g={nominal:g}_{label}_saddle']=saddle
            for sign in (-1,1):
                final,trajectory,maximum=connect(saddle,g,sign)
                fields[f'g={nominal:g}_{label}_descent{sign}']=trajectory
                connections.append(dict(g_d=nominal,path=label,sign=sign,continuous_path_maximum=maximum,**describe(final,g)))
        print('gD',nominal,'last saddle',saddles[-1]['energy'],'index',saddles[-1]['negative_modes'],flush=True)
    for name,r in [('stationary_states.csv',rows),('hessians.csv',spectra),('minimization_starts.csv',starts),
                    ('paths.csv',paths),('saddles.csv',saddles),('saddle_connections.csv',connections)]:csv_write(name,r)
    np.savez_compressed(HERE/'fields.npz',**fields)
    checks.update(saddle_gradient_max=max(r['gradient_norm'] for r in saddles),
        saddle_negative_modes=sorted(set(r['negative_modes'] for r in saddles)),
        endpoint_negative_modes=sorted(set(r['negative_modes'] for r in connections)),
        maximum_path_excess_over_saddle=max(r['continuous_path_maximum']-next(s['energy'] for s in saddles if s['g_d']==r['g_d'] and s['path']==r['path']) for r in connections),
        chiral_planar_ansatz_energy_crossing=float((np.sqrt(BP)-np.sqrt(BA))/((1+C0)/2*np.sqrt(BA)-C0/2*np.sqrt(BP))),
        zero_soc_planar=describe(state('planar_antisymmetric',0),0))
    assert checks['saddle_negative_modes']==[1] and checks['endpoint_negative_modes']==[0]
    assert checks['saddle_gradient_max']<1e-9 and checks['maximum_path_excess_over_saddle']<1e-8
    (HERE/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    plot(rows,fields)
    environment_checks()


def plot(rows,fields):
    certified=[];couplings=(.01,.02,.04,.06,.08,.1)
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for kind in ('chiral_z','planar_antisymmetric','polar_plus_z'):
        z=[r for r in rows if r['state']==kind]
        axs[0].plot([r['g_d'] for r in z],[r['energy'] for r in z],'-o',label=kind)
        if kind!='polar_plus_z':axs[1].plot([r['g_d'] for r in z],[r['smallest_positive_eigenvalue'] for r in z],'-o',label=kind)
    for g in couplings:
        one=fields[f'g={g:g}_full_perturbed_descent-1'];two=fields[f'g={g:g}_full_perturbed_descent1']
        actual=float(np.float32(g))
        if evaluate(one[-1],actual)[0]>evaluate(two[-1],actual)[0]:one,two=two,one
        path=np.concatenate((one[::-1],two[1:]));arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        t=np.unique(np.r_[np.linspace(0,1,401),arc/arc[-1]])
        field=np.array([np.interp(t*arc[-1],arc,path[:,i]) for i in range(12)]).T
        energies=evaluate(field,actual)[0];ea=evaluate(one[-1],actual)[0]
        certified.extend(dict(g_d=g,arc=x,energy=e) for x,e in zip(t,energies))
        axs[2].plot(t,energies-ea,label=f'{g:g}')
    axs[0].set(xlabel='g_D',ylabel='onsite energy',title='Stationary onsite structures');axs[0].legend(fontsize=7)
    axs[1].set(xlabel='g_D',ylabel='Hessian eigenvalue',title='Softest non-gauge modes of minima');axs[1].axhline(0,c='black',lw=.7)
    axs[1].text(.03,.82,'Polar saddle: 3 negative modes\nminimum = -3.32 to -3.42',transform=axs[1].transAxes,fontsize=9)
    axs[2].set(xlabel='normalized path arclength',ylabel='V - V_chiral',title='Verified full-field connecting paths');axs[2].legend(title='g_D',fontsize=7)
    fig.savefig(HERE/'landscape.png',dpi=180);plt.close(fig)
    csv_write('connected_paths.csv',certified)


if __name__=='__main__':main()
