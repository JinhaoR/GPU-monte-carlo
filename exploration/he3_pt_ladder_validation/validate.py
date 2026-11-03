"""Validate the saved pilot records and verify preservation of prior work."""
import ast,hashlib,json,subprocess
import numpy as np
from ladder_design import HERE,ROOT,OLD,OUT
from exploration.he3_pt_equilibration.transport import events

def main():
    paths=sorted(OUT.glob('L96_*.npz'))
    assert len(paths)==12, f'Expected 12 completed pilots, found {len(paths)}'
    checks=[];alias=[]
    for path in paths:
        with np.load(path) as z:
            m=json.loads(z['manifest'].item())['runs']['96']['metadata'];pt=m['parallel_tempering'];hist=pt['walker_history']
            T=np.array(m['temperatures']);R=len(T);t=np.array(hist['sweeps']);p=np.array(hist['temperature_indices']);ss=np.array(m['sample_sweeps'])
            np.testing.assert_array_equal(ss,np.arange(50,60001,50));np.testing.assert_array_equal(t,np.arange(0,60001,5))
            np.testing.assert_array_equal(np.sort(p,axis=1),np.broadcast_to(np.arange(R),p.shape))
            np.testing.assert_array_equal(np.array(pt['walker_to_temperature'])[pt['temperature_to_walker']],np.arange(R))
            np.testing.assert_array_equal(pt['temperature_betas'],(1/T).astype(np.float32))
            assert pt['sweeps']==60000 and pt['exchange_rounds']==12000 and pt['next_parity']==0
            assert np.all(np.array(pt['attempts'])==6000)
            assert np.all(np.array(m['sampling']['attempts'])==60000*96**2)
            assert not np.any(m['sampling']['nonfinite_rejections'])
            indices=np.array(m['spatial_sample_indices']);sp=np.array(m['spatial_sample_sweeps'])
            np.testing.assert_array_equal(indices,np.arange(19,1200,20));np.testing.assert_array_equal(sp,ss[indices])
            np.testing.assert_array_equal(z['L96_spatial_energy'],z['L96_energy'][indices])
            np.testing.assert_array_equal(z['L96_spatial_walkers'],np.argsort(p[np.searchsorted(t,sp)],axis=1))
            amp=z['L96_spatial_amplitudes']
            np.testing.assert_allclose(amp[:,:,0,0].real,z['L96_chirality'][indices],rtol=4e-6,atol=1e-3)
            np.testing.assert_allclose(amp[:,:,1,0],z['L96_phase_real'][indices]+1j*z['L96_phase_imag'][indices],rtol=4e-6,atol=1e-3)
            loops,passages,_,walkers,_=events(p,t)
            np.testing.assert_array_equal([w['loops'] for w in walkers],pt['walker_round_trips'])
            coarse=OUT/'coarse_history'/path.name
            if coarse.exists():
                with np.load(coarse) as old:
                    keys=[k for k in z.files if k!='manifest']
                    for k in keys:np.testing.assert_array_equal(z[k],old[k])
                    om=json.loads(old['manifest'].item())['runs']['96']['metadata']
                    np.testing.assert_array_equal(p[::10],om['parallel_tempering']['walker_history']['temperature_indices'])
                cp=events(p[::10],t[::10])[1]
                alias.append(dict(case=path.stem,observables_bitwise_identical=True,exact_all_passages=sum(r[1] for r in passages),coarse_all_passages=sum(r[1] for r in cp),exact_production_passages=sum(r[1] and r[5]>20000 for r in passages),coarse_production_passages=sum(r[1] and r[5]>20000 for r in cp)))
            checks.append(dict(case=path.stem,exact_loops=sum(pt['walker_round_trips']),permutation_and_sample_alignment=True,finite_measurements=all(np.all(np.isfinite(z[k])) for k in z.files if k!='manifest')))
    before=json.loads((HERE/'preservation_before.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in before['sources'].items())
    assert all((ROOT/p).stat().st_size==s['size'] and (ROOT/p).stat().st_mtime_ns==s['mtime_ns'] for p,s in before['datasets'].items())
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==before['branch']
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==before['head']
    assert not subprocess.check_output(['git','diff','--name-only'],text=True).strip()
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip()
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    record=dict(passed=True,cases=checks,history_alias_comparison=alias,unchanged_source_files=len(before['sources']),preserved_datasets=len(before['datasets']),branch_and_head_unchanged=True,tracked_and_staged_diff_empty=True)
    (OUT/'validation.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
