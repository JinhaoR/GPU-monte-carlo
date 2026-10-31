"""Small numerical identities and artifact-consistency checks, no simulation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json
import numpy as np
from reweight import solve,evaluate
from prepare import NAMES,OUT


def main():
    e=np.array([0.,2.,5.]);deg=np.array([1.,2.,1.]);beta=np.array([1.,1.6,.7])
    H=np.exp(-beta[:,None]*e)*deg;H=10000*H/H.sum(1)[:,None];h=H.sum(0)
    vals={name:np.ones(3) for name in NAMES};vals['dE']=e;vals['dE2']=e*e;vals['density']=e**2
    S=np.array([vals[name]*h for name in NAMES]);f=solve(H,S,beta,e)
    T=np.array([1/1.4,1/.8]);p=deg*np.exp(-e[None,:]/T[:,None]);p/=p.sum(1)[:,None]
    a=evaluate(f,T,1);assert np.max(abs(a['density']-p@(e*e)))<1e-10
    assert np.max(abs(a['energy']-p@e))<1e-10
    # One-source solver reduces exactly to single-histogram reweighting.
    hs=H[:1];ss=np.array([vals[name]*hs.sum(0) for name in NAMES]);single=solve(hs,ss,beta[:1],e)
    assert np.max(abs(evaluate(single,T,1)['energy']-p@e))<1e-10
    # Changing the numerical energy reference cannot change normalized weights.
    shifted=solve(H,S,beta,e-2.);assert np.max(abs(evaluate(shifted,T,1)['density']-a['density']))<1e-10
    with (OUT/'crossings.csv').open() as f:r=list(csv.DictReader(f))
    assert all(int(x['failed_deletes'])==0 and np.isfinite(float(x['joint_pt_se'])) for x in r)
    summary=json.loads((OUT/'reweight_summary.json').read_text());assert summary['maximum_Binder_T_shift_bins_or_sources']<2e-7
    raw=json.loads((OUT/'reference_checks.json').read_text());assert raw['max_direct_moment_difference_except_energy']<2e-9
    result=dict(checks_passed=7,exact_multistate_moment=True,exact_multistate_energy=True,
        exact_single_histogram=True,reference_shift_invariance=True,finite_crossing_deletes=True,
        source_bin_sensitivity_bounded=True,independent_moment_cache_agrees=True)
    (OUT/'numerical_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)


if __name__=='__main__':main()
