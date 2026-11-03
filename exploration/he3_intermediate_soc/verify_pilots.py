"""Independent CPU checks of saved thermal fields and physical walker mappings."""
from pathlib import Path
import json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from models.he3.spatial import numpy_amplitudes,chiral_distance
from exploration.he3_hqv_stage2.energy import energy
from pilot import extra


def main():
    rows=[]
    for p in sorted((HERE/'pilots').glob('*.npz')):
        m=json.loads(p.with_suffix('.json').read_text());N=m['L']**2
        with np.load(p) as z:
            a=z['snapshot_fields'][-1];amp=numpy_amplitudes(a)
            ix=np.searchsorted(z['walker_history_sweeps'],z['sweeps'])
            np.testing.assert_array_equal(z['walker_history_sweeps'][ix],z['sweeps'])
            np.testing.assert_array_equal(np.argsort(z['walker_history_indices'][ix],axis=1),z['temperature_to_walker'])
            ee=np.array([energy(f,betas=m['model']['betas'],gd=m['model']['g_d'])['total'] for f in a])
            errors=dict(energy_per_site=float(np.max(abs(ee-z['values'][-1,:,0]))/N),
                modes=float(np.max(abs(amp-z['amplitudes'][-1]))),
                mixed_curvature=float(np.max(abs(extra(a)[:,0]-z['extra'][-1,:,0]))))
            projection=[]
            for f in a:
                q=chiral_distance(f);projection.append((q['epsilon']*q['n']).sum())
            errors['chiral_residual_per_site']=float(np.max(abs(np.array(projection)-z['structure'][-1,:,6]))/N)
            assert errors['energy_per_site']<2e-6 and errors['modes']<1e-10
            assert errors['mixed_curvature']==0 and errors['chiral_residual_per_site']<1e-10
            assert np.isfinite(z['values']).all() and np.isfinite(z['structure']).all()
            assert np.isfinite(z['amplitudes']).all()
            assert not any(m['sampling']['nonfinite_rejections'])
            rows.append(dict(label=p.stem,mapping='exact',**errors))
    (HERE/'pilot_results/validation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('Validated saved fields, energies, modes, residuals and mappings for',len(rows),'pilots')


if __name__=='__main__':main()
