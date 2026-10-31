"""Reopen saved fields and verify energies, free gradients, pins and source hashes.

python -B exploration/he3_hqv_stage3/verify.py
This script never starts an optimizer. It also checks the report's topology
counts against the retained CSV, avoiding a summary detached from field data.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from relax import ROOT,value_gradient,problem

OUT=Path(__file__).resolve().parent/'results'


def main():
    checks=[];source_hashes_match=True
    for path in sorted(OUT.glob('*/info.json')):
        m=json.loads(path.read_text())
        if m.get('exclude_from_comparison'):continue
        for name,digest in m['source_sha256'].items():
            match=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
            source_hashes_match &= match
            assert match,(m['case'],name,'source changed since solve')
        with np.load(path.parent/'fields.npz') as z:A=z['initial'];B=z['final'];mask=z['pinned']
        E,G=value_gradient(B,gd=m['gd'])
        gin=float(np.max(abs(np.ascontiguousarray(2*G[~mask]).view(float))))
        np.testing.assert_allclose(E,m['final_energy'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(gin,m['gradient_inf'],rtol=1e-8,atol=1e-12)
        assert gin<=3e-6 and m['scipy_success']
        assert np.array_equal(A[mask],B[mask])
        if not m.get('released_from'):
            prescribed,prescribed_mask,_=problem(m['L'],m['R'],pin=m['pin_radius_parameter'],gd=m['gd'],
                initial='planar',right_rotation=m['right_pin_spin_rotation'])
            assert np.array_equal(mask,prescribed_mask)
            np.testing.assert_array_equal(B[mask],prescribed[mask])
        history=list(csv.DictReader((path.parent/'history.csv').open()))
        assert max(np.diff([float(r['energy']) for r in history]))<=1e-9
        checks.append(dict(case=m['case'],recomputed_excess_energy=E,recomputed_free_gradient_inf=gin,
            exact_fixed_variables=True,declared_prescription_verified=not bool(m.get('released_from'))))
    loops=list(csv.DictReader((OUT/'topology_checks.csv').open()))
    final=[r for r in loops if r['phase']=='final' and 'released' not in r['case']]
    assert len(final)==245 and all(r['accepted']=='True' and float(r['charge'])==float(r['expected']) for r in final)
    assert len(checks)==22
    for name in ['validation.json','additional_validation.json']:assert json.loads((OUT/name).read_text())['passed']
    result=dict(passed=True,verified_cases=len(checks),verified_final_pinned_loops=len(final),
        source_hashes_match=source_hashes_match,checks=checks,
        interpretation='Arithmetic, convergence and imposed constraints verified; global minimality and intrinsic-interaction claims remain limited')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Verified 22 saved full fields, 245 final pinned-loop checks, convergence, constraints and source hashes.')


if __name__=='__main__':main()
