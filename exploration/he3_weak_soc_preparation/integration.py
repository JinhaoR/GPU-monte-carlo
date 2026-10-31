"""Existing Simulation/PT/save_runs integration on static L8 fields.

Only local sweep calls are mocked to freeze the field. The original PT exchange,
measurement, Simulation loop and save/load code execute unchanged. No GL updates.
"""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
from numba import cuda
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.runtime import Runtime
from core.parallel_tempering import ParallelTempering
from core.simulation import Simulation
from models.he3.model import He3Model
from models.he3.spatial import add_spatial_samples, numpy_amplitudes, ensemble
from models.he3.spatial_analysis import check_spatial, summarize_spatial
from models.he3.analysis import He3Analysis
from utils.data_io import save_runs, load_runs
OUT=Path(__file__).resolve().parent/'results'


def main():
    if not cuda.is_available():
        print('CUDA unavailable: integration unverified');return
    results=[]
    for gd in (0.,.001,.01):
        model=He3Model(pressure=20,g_d=gd)
        runtime=Runtime(model,8,np.array([.365,.38,.395,.425]),threads=4,seed=123)
        # Distinct gauge phases, equal-energy uniform fields, nontrivial PT maps.
        a=runtime.state.copy_to_host()
        a*=np.exp(1j*np.arange(4)*.4)[:,None,None,None,None].astype(np.complex64)
        runtime.invalidate_energy_cache();runtime.state.copy_to_device(a)
        pt=ParallelTempering(runtime,exchange_interval=2,seed=124,tracked_walkers=range(4),history_stride=1)
        rng_before=runtime.rng_states.copy_to_host()
        model.spatial_every=2
        with patch.object(runtime,'sweep',return_value=None):
            run=Simulation(pt,thermalization_sweeps=4,measurement_sweeps=16,sample_interval=2).run()
        add_spatial_samples(run,model.spatial_samples)
        np.testing.assert_array_equal(runtime.state.copy_to_host(),a)
        np.testing.assert_array_equal(runtime.rng_states.copy_to_host(),rng_before)
        np.testing.assert_array_equal(run['metadata']['spatial_sample_sweeps'],[8,12,16,20])
        assert len(run['samples']['energy'])==8
        assert len(run['samples']['spatial_amplitudes'])==4
        for amp,walkers in zip(run['samples']['spatial_amplitudes'],run['samples']['spatial_walkers']):
            np.testing.assert_allclose(amp,numpy_amplitudes(a)[walkers],atol=1e-11)
        assert np.any(run['samples']['spatial_walkers']!=np.arange(4))
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            path=Path(tmp)/'run.npz';save_runs(path,{8:run});loaded=load_runs(path)[8]
            check=check_spatial(loaded)
            stats=summarize_spatial(loaded,block_size=1)
            He3Analysis({8:loaded},block_size=2)  # old scalar analysis ignores the new axes
            for connected,label in ((False,'full'),(True,'connected')):
                expected=ensemble(loaded['samples']['spatial_amplitudes'],8,connected)['spectra']
                np.testing.assert_allclose(stats[label+'_spectra'],expected,atol=1e-11)
            legacy={'metadata':loaded['metadata'],'samples':{k:v for k,v in loaded['samples'].items() if not k.startswith('spatial_')}}
            assert not check_spatial(legacy)['available']
        results.append(dict(g_d=gd,passed=True,local_GL_sweeps=0,static_exchange_rounds=pt.exchange_rounds,
                            maximum_residuals=check['maximum_absolute_residuals']))
    (OUT/'simple_integration.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('Passed existing Simulation/PT/NPZ integration at all three couplings; no local GL sweeps.')


if __name__=='__main__':main()
