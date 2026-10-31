"""One tighter SAME-CONSTRAINT restart and a few local stability probes.

python -B exploration/he3_hqv_stage3/additional_checks.py
The curvature probes test only chosen directions, not the complete Hessian.
"""
import csv
import json
from pathlib import Path
import numpy as np
from relax import optimize,value_gradient

OUT=Path(__file__).resolve().parent/'results'


def main():
    case=OUT/'L64_R24_p2'
    with np.load(case/'fields.npz') as z:A=z['final'];mask=z['pinned'];meta=json.loads(z['metadata'].item())
    B,info,history,top=optimize(A,mask,meta,seconds=20,maxiter=1200,gtol=1e-7)
    info.update(case='same_pins_tighter_restart',starting_case='L64_R24_p2',
        energy_change_from_previous=float(info['final_energy']-value_gradient(A)[0]))
    assert info['gradient_converged'] and info['constraint_error']==0,info
    assert abs(info['energy_change_from_previous'])<1e-6,info
    rows=[];rng=np.random.default_rng(4821);L=A.shape[0]
    x,y=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
    envelope=np.sin(np.pi*x/(L-1))*np.sin(np.pi*y/(L-1))
    directions={}
    V=np.zeros_like(A);V[...,1,:]=envelope[...,None]*A[...,0,:]
    directions['smooth_out_of_plane_spin']=V
    directions['smooth_amplitude']=envelope[...,None,None]*A
    core=sum(np.exp(-((x-cx)**2+(y-cy)**2)/8) for cx,cy in meta['centers'])
    V=np.zeros_like(A);V[...,1,:]=core[...,None]*np.array([1,-1j])/np.sqrt(2)
    directions['opposite_chirality_core_spin']=V
    for j in range(3):directions[f'random_full_components_{j}']=rng.normal(size=A.shape)+1j*rng.normal(size=A.shape)
    E=value_gradient(B)[0]
    for name,V in directions.items():
        V[mask]=0;V/=np.linalg.norm(V);h=.001
        ep=value_gradient(B+h*V)[0];em=value_gradient(B-h*V)[0]
        curvature=(ep+em-2*E)/(h*h)
        rows.append(dict(direction=name,second_directional_derivative=float(curvature),
            positive=bool(curvature>0),step=h))
    assert all(r['positive'] for r in rows),rows
    # Exercise the stopping path independently of successful convergence.
    small=A[:16,:16].copy();small_mask=np.zeros((16,16),bool)
    small_mask[:2]=small_mask[-2:]=True;small_mask[:,:2]=small_mask[:,-2:]=True
    small[~small_mask]+=.01*(rng.normal(size=small[~small_mask].shape)+1j*rng.normal(size=small[~small_mask].shape))
    small_meta=dict(L=16,R=8,pin_radius_parameter=1,gd=0.,centers=((3.5,7.5),(11.5,7.5)))
    _,stopped,_,_=optimize(small,small_mask,small_meta,seconds=0.,maxiter=50)
    assert not stopped['gradient_converged'] and not stopped['scipy_success']
    assert 'StopIteration' in stopped['scipy_message'] and stopped['iterations']==1,stopped
    (OUT/'additional_validation.json').write_text(json.dumps(dict(passed=True,tighter_restart=info,
        selected_curvatures=rows,zero_time_cap_test=stopped,
        limitation='Selected positive curvatures do not prove positive semidefiniteness of the full Hessian or global optimality'),indent=2),encoding='utf-8')
    print('Tighter same-pin restart g_inf',info['gradient_inf'],'energy change',info['energy_change_from_previous'],
        'seconds',info['elapsed_seconds'],'six positive selected curvatures',flush=True)


if __name__=='__main__':main()
