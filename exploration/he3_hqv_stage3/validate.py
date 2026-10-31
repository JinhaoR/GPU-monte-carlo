"""Finite-difference and known-minimum checks before any HQV relaxation."""
import json
from pathlib import Path
import numpy as np
from relax import value_gradient,optimize,BETA_A

OUT=Path(__file__).resolve().parent/'results'


def main():
    rng=np.random.default_rng(39281);checks=[]
    A=.35*(rng.normal(size=(8,10,3,2))+1j*rng.normal(size=(8,10,3,2)))
    for gd in [0.,.001]:
        E,G=value_gradient(A,gd=gd)
        # All six complex channels, both real and imaginary, including wrap bonds.
        for i,j in [(0,0),(0,9),(7,0),(7,9),(3,4)]:
            for mu in range(3):
                for a in range(2):
                    for imaginary in [False,True]:
                        V=np.zeros_like(A);V[i,j,mu,a]=1j if imaginary else 1
                        h=2e-6
                        ep=value_gradient(A+h*V,gd=gd)[0];em=value_gradient(A-h*V,gd=gd)[0]
                        numeric=(ep-em)/(2*h)
                        analytic=2*(G[i,j,mu,a].imag if imaginary else G[i,j,mu,a].real)
                        error=abs(numeric-analytic)
                        assert error<2e-6,(gd,i,j,mu,a,imaginary,numeric,analytic)
                        checks.append(dict(gd=gd,coordinate=[i,j,mu,a],imaginary=imaginary,
                            analytic=float(analytic),numeric=float(numeric),absolute_error=float(error)))
        V=rng.normal(size=A.shape)+1j*rng.normal(size=A.shape);V/=np.linalg.norm(V)
        numeric=(value_gradient(A+1e-4*V,gd=gd)[0]-value_gradient(A-1e-4*V,gd=gd)[0])/2e-4
        analytic=2*np.vdot(G,V).real
        assert abs(numeric-analytic)<1e-6
        checks.append(dict(gd=gd,directional=True,analytic=float(analytic),numeric=float(numeric),absolute_error=float(abs(numeric-analytic))))
    # Known homogeneous minimum in a small, fixed-frame box.
    L=16;n0=1/BETA_A
    uniform=np.zeros((L,L,3,2),complex);uniform[...,2,0]=np.sqrt(n0/2);uniform[...,2,1]=1j*np.sqrt(n0/2)
    E,G=value_gradient(uniform)
    assert abs(E)<1e-10 and np.max(abs(G))<1e-12
    mask=np.zeros((L,L),bool);mask[:2]=mask[-2:]=True;mask[:,:2]=mask[:,-2:]=True
    initial=uniform.copy();initial[~mask]+=.03*(rng.normal(size=initial[~mask].shape)+1j*rng.normal(size=initial[~mask].shape))
    meta=dict(L=L,R=8,pin_radius_parameter=1,gd=0.,centers=((3.5,7.5),(11.5,7.5)),test='perturbed homogeneous field; no annular pins')
    final,info,history,top=optimize(initial,mask,meta,seconds=15,maxiter=800,gtol=1e-7)
    assert info['gradient_converged'] and info['constraint_error']==0,info
    assert abs(info['final_energy'])<1e-8,info
    assert np.max(abs(final-uniform))<1e-5
    assert info['max_accepted_energy_increase']<1e-9
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'validation.json').write_text(json.dumps(dict(passed=True,finite_difference_checks=checks,
        maximum_fd_error=max(r['absolute_error'] for r in checks),homogeneous_test=info,
        homogeneous_max_field_error=float(np.max(abs(final-uniform)))),indent=2),encoding='utf-8')
    print('Passed',len(checks),'finite differences; homogeneous relaxation',info['iterations'],'iterations',
        info['elapsed_seconds'],'s; final free gradient',info['gradient_inf'],flush=True)


if __name__=='__main__':main()
