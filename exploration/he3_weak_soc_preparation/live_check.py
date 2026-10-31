"""Tiny actual run.py execution: 3 couplings, L8, 6 replicas, 40+160 sweeps.

Only settings are reduced. Audit hooks observe the original Simulation/measure
calls; Gaussian updates, PT, all measurements and save_runs execute unchanged.
This is a correctness test, not an equilibrium or performance measurement.
"""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from numba import cuda

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run as he3
from core.simulation import Simulation
from models.he3.measure import MEASUREMENT_NAMES
from models.he3.spatial import numpy_amplitudes, local_fields
from models.he3.spatial_analysis import check_spatial, summarize_spatial, summarize_realspace
from models.he3.analysis import He3Analysis
from utils.data_io import load_runs

OUT = Path(__file__).resolve().parent / 'results'
spec = importlib.util.spec_from_file_location('old_numpy_reference', ROOT/'tests/test_he3.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def direct_profiles(a):
    """Direct pairs, including an independent full 3x3 Q contraction."""
    f = local_fields(a)[..., :7]
    L = len(a)
    c = np.empty((7, 2, L//2+1), complex)
    m = np.einsum('...ma,...na->...mn', a.astype(complex), a.conj().astype(complex)).real
    q = m - np.trace(m, axis1=-2, axis2=-1)[..., None, None]*np.eye(3)/3
    for axis in (0, 1):
        for r in range(L//2+1):
            c[:, axis, r] = (np.roll(f, -r, axis=axis)*f.conj()).mean(axis=(0, 1))
            full_q = (np.roll(q, -r, axis=axis)*q).sum(axis=(-2, -1)).mean()
            np.testing.assert_allclose(c[2:, axis, r].sum(), full_q, atol=1e-12)
    return c, f.mean(axis=(0, 1))


def main():
    if not cuda.is_available():
        raise RuntimeError('CUDA is required for this live check')
    destination = OUT / 'live_prelaunch'
    if destination.exists():
        raise FileExistsError('Keep existing evidence; choose another test destination before rerunning')
    original_run = Simulation.run
    audits = []

    def audited_run(sim):
        pt, rt = sim.runtime, sim.runtime.runtime
        model, L, R = rt.model, rt.L, rt.R
        assert (L, R, sim.thermalization_sweeps, sim.measurement_sweeps) == (8, 6, 40, 160)
        assert isinstance(rt, he3.He3GaussianRuntime)
        initial, initial_rng = rt.snapshot(), rt.rng_states.copy_to_host()
        scalar, maps, spatial, profiles = [], [], [], []
        errors = dict(scalar=0., modes=0., profiles=0., means=0.)
        original_measure = model.measure

        def audited_measure(state, temperature_indices=None, cached_energies=None):
            before = rt.snapshot()
            rng = rt.rng_states.copy_to_host()
            pt_rng = pt._rng_states.copy_to_host()
            mapping = temperature_indices.copy_to_host()
            walkers = np.argsort(mapping)
            versions = (rt._configuration_version, rt._energy_version)
            cache = cached_energies.copy_to_host()
            result = original_measure(state, temperature_indices, cached_energies)
            np.testing.assert_array_equal(rt.snapshot(), before)
            np.testing.assert_array_equal(rt.rng_states.copy_to_host(), rng)
            np.testing.assert_array_equal(pt._rng_states.copy_to_host(), pt_rng)
            np.testing.assert_array_equal(temperature_indices.copy_to_host(), mapping)
            np.testing.assert_array_equal(cached_energies.copy_to_host(), cache)
            assert versions == (rt._configuration_version, rt._energy_version)
            expected = reference.reference(before, model.K, model.betas, float(model.g_d))
            h = 2e-4
            for axis in (0, 1):
                twist = np.zeros(2); twist[axis] = h
                plus = reference.total_energy(before, model.K, model.betas, twist, float(model.g_d))
                minus = reference.total_energy(before, model.K, model.betas, -twist, float(model.g_d))
                expected['current_'+'xy'[axis]] = (plus-minus)/(2*h)
                expected['curvature_'+'xy'[axis]] = (plus+minus-2*expected['energy'])/h**2
            assert set(result) == set(MEASUREMENT_NAMES)
            for name, values in result.items():
                np.testing.assert_allclose(values, expected[name][walkers], rtol=3e-6, atol=3e-5)
                errors['scalar'] = max(errors['scalar'], float(np.max(abs(values-expected[name][walkers]))))
            scalar.append({k:v.copy() for k,v in result.items()})
            maps.append(walkers.copy())
            index = len(scalar)-1
            if (index+1) % model.spatial_every == 0:
                record = model.spatial_samples[-1]
                assert record[0] == index
                np.testing.assert_array_equal(record[3], walkers)
                modes = numpy_amplitudes(before)[walkers]
                np.testing.assert_allclose(record[1], modes, atol=1e-11, rtol=1e-12)
                errors['modes'] = max(errors['modes'], float(np.max(abs(record[1]-modes))))
                spatial.append(tuple(np.array(x, copy=True) for x in record))
            if (index+1) % model.realspace_every == 0:
                record = model.realspace_samples[-1]
                assert record[0] == index
                np.testing.assert_array_equal(record[3], walkers[model.realspace_indices])
                np.testing.assert_array_equal(record[4], model.realspace_indices)
                for j, w in enumerate(record[3]):
                    direct, means = direct_profiles(before[w])
                    np.testing.assert_allclose(record[1][j], direct, atol=1e-12)
                    np.testing.assert_allclose(record[2][j], means, atol=1e-12)
                    errors['profiles'] = max(errors['profiles'], float(np.max(abs(record[1][j]-direct))))
                    errors['means'] = max(errors['means'], float(np.max(abs(record[2][j]-means))))
                profiles.append(tuple(np.array(x, copy=True) for x in record))
            return result

        with patch.object(model, 'measure', side_effect=audited_measure):
            run = original_run(sim)
        assert not np.array_equal(rt.snapshot(), initial)
        assert not np.array_equal(rt.rng_states.copy_to_host(), initial_rng)
        diagnostics = rt.sampling_diagnostics()
        np.testing.assert_array_equal(diagnostics['attempts'], np.full(R, L*L*200))
        assert all(np.asarray(diagnostics['accepted']) > 0)
        assert not any(diagnostics['nonfinite_rejections'])
        assert pt.exchange_rounds == 20
        audits.append(dict(g_d=float(model.g_d), scalar=scalar, maps=np.array(maps),
                           spatial=spatial, profiles=profiles, errors=errors))
        return run

    original_sizes, original_production = he3.sizes[:], he3.measurement_sweeps
    small_ladder = dict(he3.temperature_ladder, n_T=6)
    options = dict(sizes=[8], thermalization_sweeps=40, measurement_sweeps=160,
                   spatial_interval=20, realspace_interval=40, history_stride=1,
                   temperature_ladder=small_ladder, data_root=OUT, campaign='live_prelaunch')
    # Leave sampler, rho, mass, threads, sample/exchange intervals and runner flow intact.
    with patch.multiple(he3, **options), patch.object(Simulation, 'run', audited_run):
        he3.main()
    rows = []
    for gd, audit in zip(he3.g_d_values, audits):
        path = destination / f'P=20_gd={gd:g}_gaussian' / 'L=8.npz'
        run = load_runs(path)[8]
        s, m = run['samples'], run['metadata']
        names = set(MEASUREMENT_NAMES) | {'spatial_amplitudes', 'spatial_structure', 'spatial_walkers',
                'spatial_energy', 'realspace_profiles', 'realspace_means', 'realspace_walkers', 'realspace_energy'}
        assert set(s) == names
        assert m['sampler'] == 'gaussian' and m['threads'] == 16
        np.testing.assert_array_equal(m['sample_sweeps'], np.arange(50, 201, 10))
        np.testing.assert_array_equal(m['temperatures'], he3.make_temperature_ladder(**small_ladder))
        for name in MEASUREMENT_NAMES:
            np.testing.assert_array_equal(s[name], np.stack([row[name] for row in audit['scalar']]))
        for prefix, keys, records in [('spatial', ('amplitudes','structure','walkers'), audit['spatial']),
                                     ('realspace', ('profiles','means','walkers'), audit['profiles'])]:
            for i, key in enumerate(keys, 1):
                np.testing.assert_array_equal(s[prefix+'_'+key], np.stack([r[i] for r in records]))
            ids = np.array([r[0] for r in records])
            np.testing.assert_array_equal(m[prefix+'_sample_indices'], ids)
            np.testing.assert_array_equal(m[prefix+'_sample_sweeps'], np.asarray(m['sample_sweeps'])[ids])
        np.testing.assert_array_equal(m['spatial_sample_indices'], np.arange(1,16,2))
        np.testing.assert_array_equal(m['realspace_sample_indices'], np.arange(3,16,4))
        probes = np.asarray(m['realspace_temperature_indices'])
        np.testing.assert_array_equal(m['realspace_temperatures'], np.asarray(m['temperatures'])[probes])
        np.testing.assert_array_equal(s['realspace_energy'], s['energy'][3::4][:,probes])
        np.testing.assert_array_equal(s['realspace_walkers'], audit['maps'][3::4][:,probes])
        pt = m['parallel_tempering']
        assert sum(pt['accepted']) > 0
        assert np.any(audit['maps'] != np.arange(6))
        np.testing.assert_array_equal(pt['attempts'], np.full(5,10))
        history = pt['walker_history']
        for sweep, mapping in zip(m['sample_sweeps'], audit['maps']):
            row = history['sweeps'].index(sweep)
            np.testing.assert_array_equal(mapping, np.argsort(history['temperature_indices'][row]))
        check = check_spatial(run)
        summarize_spatial(run, block_size=2)
        He3Analysis({8:run}, block_size=4)
        summary = summarize_realspace(run, block_size=1)
        raw, mean = s['realspace_profiles'], s['realspace_means']
        for connected, label in ((False,'full'), (True,'connected')):
            def estimate(p, u):
                c = p.mean(0) - (abs(u.mean(0))[...,None,None]**2 if connected else 0)
                return np.stack((c[:,0],c[:,1],c[:,2:7].sum(1)),axis=1)
            np.testing.assert_allclose(summary[label], estimate(raw, mean), atol=1e-12)
            deleted = np.stack([estimate(np.delete(raw,i,0), np.delete(mean,i,0)) for i in range(4)])
            for part in ('real','imag'):
                error = np.sqrt(3*getattr(deleted,part).var(0))
                np.testing.assert_allclose(summary[label+'_se_'+part], error, atol=1e-12)
        for name in names:
            assert np.all(np.isfinite(s[name])), name
        rows.append(dict(g_d=gd, passed=True, path=str(path.relative_to(ROOT)),
                         shape={k:list(v.shape) for k,v in s.items()},
                         realspace_temperature_indices=probes.tolist(),
                         gaussian_attempts=sum(m['sampling']['attempts']),
                         gaussian_accepted=sum(m['sampling']['accepted']),
                         pt_attempts=sum(pt['attempts']), pt_accepted=sum(pt['accepted']),
                         independent_reference_max_errors=audit['errors'],
                         spatial_consistency=check['maximum_absolute_residuals']))
    assert he3.sizes == original_sizes and he3.measurement_sweeps == original_production
    result = dict(passed=True, L=8, replicas=6, thermalization=40, production=160,
                  total_actual_GL_sweeps_across_cases=600, sample_interval=10, exchange_interval=10,
                  scalar_events_per_case=16, spatial_events_per_case=8, profile_events_per_case=4,
                  measurements_read_only=True, saved_channels_exact=True, rows=rows)
    (OUT/'live_prelaunch_results.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
    print('PASS: final run.py, actual Gaussian/PT, all saved channels and maps, direct references.')


if __name__ == '__main__':
    main()
