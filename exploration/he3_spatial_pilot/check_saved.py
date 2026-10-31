"""Verify synchronization, saved maps and unchanged sources without rerunning MC."""
import hashlib,json
from pathlib import Path
import numpy as np
from observables import numpy_amplitudes,powers
from ladder import OUT,ROOT


def main():
    rows=[]
    for label in ['ordered','hot']:
        folder=OUT/label;meta=json.loads((folder/'run.json').read_text())
        for name,h in meta['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
        with np.load(folder/'records.npz') as z:
            ids=np.searchsorted(z['global_sweeps'],z['spatial_sweeps'])
            np.testing.assert_array_equal(z['global_values'][ids],z['spatial_global_values'])
            np.testing.assert_array_equal(np.sort(z['temperature_to_walker'],axis=1),
                np.broadcast_to(np.arange(len(z['temperatures'])),z['temperature_to_walker'].shape))
            a=z['amplitudes'];g=z['spatial_global_values']
            assert np.all(powers(a,64)>=0)
            ck=float(np.max(abs(a[:,:,0,0].real-g[:,:,2])))
            cp=float(np.max(abs(a[:,:,1,0]-g[:,:,3]-1j*g[:,:,4])))
            assert ck<.001 and cp<.001
            assert np.sum(z['spatial_sweeps']>100000)==400
        maximum=0.
        for path in folder.glob('snapshot_*.npz'):
            with np.load(path) as z:
                err=float(np.max(abs(numpy_amplitudes(z['fields'])-z['amplitudes'])))
                assert err<1e-9;maximum=max(maximum,err)
        rows.append(dict(history=label,synchronized_records_exact=True,maps_are_permutations=True,
            source_hashes_unchanged=True,production_spatial_records=400,
            max_zero_mode_chirality_difference=ck,max_zero_mode_composite_difference=cp,
            max_sparse_snapshot_numpy_gpu_difference=maximum))
    (OUT/'saved_data_checks.json').write_text(json.dumps(dict(passed=True,histories=rows),indent=2),encoding='utf-8')
    print(rows)


if __name__=='__main__':main()
