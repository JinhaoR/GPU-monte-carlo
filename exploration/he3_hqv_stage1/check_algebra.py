"""Small CPU algebra checks against the actual production polynomial source.

Run: python -B exploration/he3_hqv_stage1/check_algebra.py
Reads four device helpers as Python AST, removes only CUDA decorators, and
evaluates them on the CPU. Does not import CUDA, mutate source, or relax fields.
"""
import ast
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'results'
BETAS=np.array([-.5165123343467712,.9420228004455566,.9875290393829346,
                .8790628910064697,-1.152514934539795])


def source_helpers():
    source=ast.parse((ROOT/'models/he3/update.py').read_text(encoding='utf-8'))
    keep=[]
    for node in source.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('norm2','dipole_invariant','potential','gradient'):
            node.decorator_list=[];keep.append(node)
    module=ast.fix_missing_locations(ast.Module(body=keep,type_ignores=[]))
    ns={'float32':np.float32}
    exec(compile(module,'read_only_production_polynomials','exec'),ns)
    return ns


def energy(A,gd,helpers):
    # Production has periodic boundaries and sums forward gradient cells.
    n=A.shape[0];m=A.shape[1]
    potential=sum(helpers['potential'](A[x,y],BETAS,gd) for x in range(n) for y in range(m))
    dx=np.roll(A,-1,axis=0)-A;dy=np.roll(A,-1,axis=1)-A
    gradient=np.sum(helpers['gradient'](dx[...,0],dx[...,1],dy[...,0],dy[...,1]))
    return float(potential+gradient)


def main():
    from defects import local_geometry
    from synthetic import chiral
    rng=np.random.default_rng(20261029);h=source_helpers()
    # Float64 fields isolate algebraic identities from production float32 errors.
    A=.4*(rng.normal(size=(11,12,3,2))+1j*rng.normal(size=(11,12,3,2)))
    Q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(Q)<0:Q[:,0]*=-1
    rot=np.einsum('ab,...bc->...ac',Q,A)
    checks={}
    for gd in (0.,.001,.01):
        base=energy(A,gd,h)
        gauge=energy(A*np.exp(.734j),gd,h)
        conjugate=energy(A.conj(),gd,h)
        er=energy(rot,gd,h)
        np.testing.assert_allclose(base,gauge,rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(base,conjugate,rtol=1e-12,atol=1e-10)
        if gd==0:np.testing.assert_allclose(base,er,rtol=1e-12,atol=1e-10)
        checks[f'gd={gd:g}']=dict(energy=base,gauge_difference=gauge-base,
            conjugation_difference=conjugate-base,spin_rotation_difference=er-base)
    # Check exact discrete identity in fixed chirality, including arbitrary
    # complex spin vector: G=2 (|Delta_x w|^2+|Delta_y w|^2).
    w=rng.normal(size=(11,12,3))+1j*rng.normal(size=(11,12,3))
    Ac=w[..., :,None]*np.array([1,1j])/np.sqrt(2)
    dx=np.roll(Ac,-1,axis=0)-Ac;dy=np.roll(Ac,-1,axis=1)-Ac
    actual=h['gradient'](dx[...,0],dx[...,1],dy[...,0],dy[...,1]).sum()
    expected=2*np.sum(abs(np.roll(w,-1,axis=0)-w)**2+abs(np.roll(w,-1,axis=1)-w)**2)
    np.testing.assert_allclose(actual,expected,rtol=1e-12)
    checks['chiral_gradient']=dict(actual=float(actual),expected=float(expected),difference=float(actual-expected))
    # Projection is checked by reconstructing the best fit, not only its eigenvalue.
    g=local_geometry(A);d=g['director'];a=g['amplitude'];sig=g['sigma']
    orbital=np.stack([np.ones_like(sig),1j*sig],axis=-1)/np.sqrt(2)
    fit=a[...,None,None]*d[..., :,None]*orbital[...,None,:]
    direct=np.sum(abs(A-fit)**2,axis=(-2,-1))/g['density']
    np.testing.assert_allclose(direct,g['residual'],atol=1e-14)
    np.testing.assert_allclose(g['wrong_chirality']+g['nonreal_spin'],g['residual'],atol=1e-14)
    bound=1-2*g['residual']
    assert np.all(g['sigma']*g['chirality']>=bound-1e-13)
    assert np.all(g['composite_ratio']>=bound-1e-13)
    assert np.all(g['gap']>=bound-1e-13)
    # Independent sampled-direction search cannot improve the analytic maximum.
    trial=rng.normal(size=(6000,3));trial/=np.linalg.norm(trial,axis=-1,keepdims=True)
    z=(A[...,0]-1j*sig[...,None]*A[...,1])/np.sqrt(2)
    sampled=np.max(abs(np.einsum('...i,ki->...k',z,trial))**2,axis=-1)
    analytic=abs(a)**2
    assert np.all(sampled<=analytic+1e-12)
    checks['projection']=dict(max_reconstruction_error=float(np.max(abs(direct-g['residual']))),
        directions_per_random_field=6000,random_fields=132,
        minimum_analytic_minus_sampled=float(np.min(analytic-sampled)))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'algebra_validation.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2),encoding='utf-8')
    print('Passed production-polynomial symmetry, exact gradient, projection and quality-bound checks.')


if __name__=='__main__':main()
