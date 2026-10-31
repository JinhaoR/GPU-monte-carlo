"""Small explicit case lists; no automatic parameter expansion or MC.

python -B exploration/he3_hqv_stage3/run_cases.py            # three gD=0 cases
python -B exploration/he3_hqv_stage3/run_cases.py --controls # explicit modest controls
Individual completed cases are reused, never overwritten. Review convergence
and topology before choosing any additional cases or the optional SOC tests.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from relax import problem,optimize,topology,ROOT

HERE=Path(__file__).resolve().parent;OUT=HERE/'results'


def run_one(name,L,R,pin=2,initial='noisy',right_rotation=0.,gd=0.,seconds=60.,maxiter=3500,release=None):
    case=OUT/name
    if (case/'info.json').exists():
        print('Cached',name,flush=True);return json.loads((case/'info.json').read_text())
    if case.exists():raise FileExistsError(f'Incomplete case requires inspection: {case}')
    if release:
        previous=OUT/release
        with np.load(previous/'fields.npz',allow_pickle=False) as z:A=z['final'].copy();mask=z['pinned'].copy()
        meta=json.loads((previous/'info.json').read_text())
        meta={k:meta[k] for k in ['L','R','pin_radius_parameter','pin_chebyshev_radius','gd','initial','seed',
            'right_pin_spin_rotation','centers','outer_frame_width','pinned_sites','free_real_variables','constraint']}
        mask[:]=False;mask[:2]=mask[-2:]=True;mask[:,:2]=mask[:,-2:]=True
        meta.update(initial='released_previous_minimum',released_from=release,constraint='outer frame only; annular constraints released',
            pinned_sites=int(mask.sum()),free_real_variables=int((~mask).sum()*12))
    else:A,mask,meta=problem(L,R,pin=pin,initial=initial,right_rotation=right_rotation,gd=gd)
    print('Starting',name,'free variables',meta['free_real_variables'],flush=True)
    B,info,history,th=optimize(A,mask,meta,seconds=seconds,maxiter=maxiter)
    case.mkdir(parents=True)
    np.savez_compressed(case/'fields.npz',initial=A,final=B,pinned=mask,metadata=np.array(json.dumps(meta)))
    info.update(case=name,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        HERE/'relax.py',HERE/'run_cases.py',ROOT/'exploration/he3_hqv_stage2/energy.py',
        ROOT/'models/he3/update.py',ROOT/'exploration/he3_hqv_stage1/defects.py']})
    (case/'info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    with (case/'history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    (case/'topology_history.json').write_text(json.dumps(th,indent=2),encoding='utf-8')
    print(name,'E',info['initial_energy'],'->',info['final_energy'],'g_inf',info['gradient_inf'],
        'iterations',info['iterations'],'seconds',info['elapsed_seconds'],'status',info['scipy_message'],flush=True)
    print('Final loops',[(r['core'],r['radius_parameter'],r['accepted'],r['phase_charge'],round(r['max_residual'],4)) for r in th[-1]['loops']],flush=True)
    return info


def main():
    validation=json.loads((OUT/'validation.json').read_text())
    assert validation['passed'],'Run validate.py first'
    started=perf_counter()
    if '--controls' in sys.argv:
        plan=[('L96_R16_p2',96,16,2,'noisy',0.),('L96_R32_p2',96,32,2,'noisy',0.),
            ('L64_R16_p3',64,16,3,'noisy',0.),('L64_R32_p3',64,32,3,'noisy',0.),
            ('L64_R24_nonplanar',64,24,2,'nonplanar',0.),('L64_R24_rotated_pin',64,24,2,'noisy',np.pi/2)]
    else:plan=[(f'L64_R{R}_p2',64,R,2,'noisy',0.) for R in [16,24,32]]
    for name,L,R,pin,initial,rotation in plan:
        if perf_counter()-started>420:print('Batch time limit; remaining cases not started');break
        run_one(name,L,R,pin=pin,initial=initial,right_rotation=rotation)
    print('Batch seconds',perf_counter()-started,flush=True)


if __name__=='__main__':main()
