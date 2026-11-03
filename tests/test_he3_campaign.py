"""Bounded GPU validation; also run this on the cluster before the long reference."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from numba import cuda

from core.parallel_tempering import ParallelTempering
from core.simulation import Simulation
from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime
from models.he3.spatial import (numpy_amplitudes, charged_amplitudes, measurement_profiles,
                                add_spatial_samples, local_fields)
from models.he3.spatial_analysis import check_spatial
from tests.test_he3 import total_energy
from utils.checkpoint import read_checkpoint, write_checkpoint
from utils.data_io import load_runs, save_runs


def make_simulation(gd=.01, spatial=True):
    model = He3Model(pressure=20, g_d=gd)
    if spatial:
        model.spatial_every, model.realspace_every = 2, 3
        model.realspace_indices = np.array([1, 5], np.int32)
        model.charged_profiles = True
    runtime = He3GaussianRuntime(model, 8, np.linspace(.375, .405, 8), seed=752, threads=4)
    pt = ParallelTempering(runtime, exchange_interval=5, seed=753,
                           tracked_walkers=range(8), history_stride=3)
    # Force chunk rollover while keeping the already-recorded first row.
    pt._history_chunks = [cuda.to_device(pt._history_chunks[0][:3].copy_to_host())]
    pt._history_chunk_rows = 3
    return Simulation(pt, thermalization_sweeps=20, measurement_sweeps=60,
                      sample_interval=5, traffic_interval=20)


def assert_tree_equal(test, a, b):
    if isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a:
            assert_tree_equal(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_tree_equal(test, x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        test.assertEqual(a, b)


class ChargedTests(unittest.TestCase):
    def test_analysis_normalization_and_pseudocritical_fit(self):
        from exploration.he3_large_size_preparation.analysis_ready import transition_curves, direct_jump, crossing_summary
        from utils.Analysis.bkt import fit_pseudocritical_bkt
        from models.he3.spatial import FIELDS
        L, T = 160, np.array([.38, .382, .384])
        rng = np.random.default_rng(17)
        m = rng.normal(size=(400, 3))*.3
        # Exactly known anisotropic tensor, with random currents subtracted by hand.
        ix, iy = rng.normal(size=(2,400,3))
        hx, hy, xy = np.array([1.2,.9,.6]), np.array([1.1,.8,.5]), .2
        samples = dict(chirality=m, density=np.ones_like(m), current_x=ix, current_y=iy,
                       curvature_x=np.broadcast_to(hx*L**2+ix.var(0)/T,m.shape).copy(),
                       curvature_y=np.broadcast_to(hy*L**2+iy.var(0)/T,m.shape).copy(),
                       curvature_xy=np.broadcast_to(xy*L**2+((ix-ix.mean(0))*(iy-iy.mean(0))).mean(0)/T,m.shape).copy())
        a = np.ones((40,3,len(FIELDS),5), complex)
        a[...,0] = 2
        samples['spatial_amplitudes'] = a
        run = dict(samples=samples,metadata=dict(L=L,temperatures=T,thermalization_sweeps=0,
                   measurement_sweeps=400, sample_sweeps=np.arange(1,401),
                   spatial_sample_sweeps=np.arange(10,401,10)))
        c = transition_curves(run,100,20,seed=52)
        np.testing.assert_allclose(c['central']['helicity_x'],hx,atol=1e-12)
        np.testing.assert_allclose(c['central']['helicity_xy'],xy,atol=1e-12)
        np.testing.assert_allclose(c['central']['effective'],np.sqrt(hx*hy-xy**2),atol=1e-12)
        expected_xi = np.sqrt(3)/(2*L*np.sin(np.pi/L))
        np.testing.assert_allclose(c['central']['xi_over_L'][...,0],expected_xi)
        self.assertEqual(len(direct_jump(c,.5)['roots']),1)
        self.assertEqual(direct_jump(c,1.)['roots'],[])
        self.assertIsNone(crossing_summary(T,np.zeros(3),np.tile([-1,0,1],(20,1)))['statistical_se'])
        sizes = np.array([96,128,160,256,320])
        shift = .381+ .02/np.log(sizes/2.)**2
        fit = fit_pseudocritical_bkt(sizes,shift,np.full(5,1e-5))
        self.assertAlmostEqual(fit['temperature'],.381,places=7)
        self.assertLess(fit['chi2'],1e-8)
        with self.assertRaises(ValueError):
            fit_pseudocritical_bkt(sizes[:3],shift[:3],np.full(3,1e-5))

    def test_direct_fourier_and_realspace(self):
        rng = np.random.default_rng(23)
        a = rng.normal(size=(8, 8, 3, 2))+1j*rng.normal(size=(8, 8, 3, 2))
        plus, minus = (a[..., 0]+a[..., 1])/np.sqrt(2), (a[..., 0]-a[..., 1])/np.sqrt(2)
        transformed = charged_amplitudes(numpy_amplitudes(a))
        direct = dict(A_plus=plus, A_minus=minus, A_x=a[..., 0], A_y=a[..., 1],
                      B_plus_plus=np.sum(plus**2, -1)[..., None],
                      B_minus_minus=np.sum(minus**2, -1)[..., None],
                      B_plus_minus=np.sum(plus*minus, -1)[..., None])
        for name, field in direct.items():
            # Explicit site sum at all signed modes, independent of FFT code.
            x, y = np.indices((8, 8))
            modes = [np.sum(field*np.exp(-2j*np.pi*(kx*x+ky*y)/8)[..., None], axis=(0, 1))
                     for kx, ky in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]]
            np.testing.assert_allclose(transformed[name], np.stack(modes, -1), atol=1e-12)
        b = transformed
        np.testing.assert_allclose(abs(b['B_plus_plus'])**2+abs(b['B_minus_minus'])**2+2*abs(b['B_plus_minus'])**2,
                                   abs(b['B_xx'])**2+abs(b['B_yy'])**2+2*abs(b['B_xy'])**2, atol=1e-11)
        profiles, means = measurement_profiles(a, charged=True)
        fields = np.concatenate((local_fields(a)[..., :7], plus[..., 2:3], minus[..., 2:3],
                                 direct['B_plus_plus'], direct['B_minus_minus']), axis=-1)
        for axis in (0, 1):
            for r in range(5):
                expected = (np.roll(fields, -r, axis=axis)*fields.conj()).mean(axis=(0,1))
                np.testing.assert_allclose(profiles[:, axis, r], expected, atol=1e-12)
        np.testing.assert_allclose(means, fields.mean(axis=(0,1)), atol=1e-12)


@unittest.skipUnless(cuda.is_available(), 'CUDA required')
class CampaignGPUTests(unittest.TestCase):
    def test_sparse_history_has_exact_endpoint_labels(self):
        pt = make_simulation().runtime
        labels = np.full(pt.R,-1,np.int8)
        labels[0],labels[-1]=0,1
        for _ in range(80):
            pt.sweep()
            if pt.sweeps % pt.exchange_interval == 0:
                positions = pt.walker_to_temperature.copy_to_host()
                labels[positions==0]=0
                labels[positions==pt.R-1]=1
                history = pt.diagnostics()['walker_history']
                np.testing.assert_array_equal(history['last_endpoint'][-1],labels)

    def test_final_runner_partial_then_resume(self):
        import large_run
        with tempfile.TemporaryDirectory() as folder:
            options = dict(sizes=[8], allocations={8:(20,60)}, sample_interval=5,
                           spatial_interval=10, realspace_interval=15, traffic_interval=20,
                           output_dir=Path(folder), g_d_values=[0.], initializations=['random'])
            original = Simulation.run
            with patch.multiple(large_run, **options):
                with patch.object(Simulation,'run',lambda obj,**kw: original(obj,max_sweeps=37,**kw)):
                    large_run.main()
                large_run.main()
                output = load_runs(Path(folder)/'P=20_gd=0/L=8_random.npz')[8]
                self.assertTrue(output['metadata']['complete'])
                np.testing.assert_array_equal(output['metadata']['sample_sweeps'],np.arange(25,81,5))
                check_spatial(output)
                np.testing.assert_array_equal(output['samples']['realspace_energy'],output['samples']['energy'][[2,5,8,11]][:,output['metadata']['realspace_temperature_indices']])

    def test_full_twist_hessian(self):
        rng = np.random.default_rng(91)
        field = (.15*(rng.normal(size=(3,8,8,3,2))+1j*rng.normal(size=(3,8,8,3,2)))).astype(np.complex64)
        # Include a smooth correlated field and uniform diagonal orbital field.
        field[1] += (.3+.2j)
        field[2] = 0
        field[2, ..., 2, :] = .7/np.sqrt(2)
        for gd in (0., .01):
            model = He3Model(pressure=20, g_d=gd, K=.83)
            measured = model.measure(cuda.to_device(field))
            def energy(x, y):
                return total_energy(field, model.K, model.betas, (x,y), gd)
            levels = []
            for h in (.01, .005):
                zero = energy(0,0)
                levels.append(np.stack(((energy(h,0)-energy(-h,0))/(2*h),
                    (energy(0,h)-energy(0,-h))/(2*h),
                    (energy(h,0)-2*zero+energy(-h,0))/h**2,
                    (energy(0,h)-2*zero+energy(0,-h))/h**2,
                    (energy(h,h)-energy(h,-h)-energy(-h,h)+energy(-h,-h))/(4*h*h))))
            reference = (4*levels[1]-levels[0])/3
            for name, value in zip(('current_x','current_y','curvature_x','curvature_y','curvature_xy'), reference):
                np.testing.assert_allclose(measured[name], value, atol=2e-5, rtol=2e-6)
            n = measured['density'][2]
            np.testing.assert_allclose(measured['curvature_xy'][2], 2*model.K*n, rtol=2e-6)
            np.testing.assert_allclose(measured['curvature_x'][2], 4*model.K*n, rtol=2e-6)

    def test_checkpoint_exact_and_saved_channels(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/'state.npz'
            for gd in (0., .01):
                full = make_simulation(gd)
                expected = full.run()
                for stops in ([19], [20], [24], [25], [29], [30], [35], [37], [40], [80], [37,59]):
                    with self.subTest(gd=gd, stops=stops):
                        partial = make_simulation(gd)
                        for stop in stops:
                            partial.run(stop_requested=lambda obj=partial, end=stop: obj.completed_sweeps >= end,
                                        checkpoint_path=checkpoint,
                                        checkpoint_seconds=1e-9 if stop == 35 else 3600)
                            resumed = make_simulation(gd)
                            resumed.load_checkpoint(checkpoint)
                            assert_tree_equal(self, partial.runtime.state_dict(), resumed.runtime.state_dict())
                            partial = resumed
                        actual = partial.run()
                        assert_tree_equal(self, expected, actual)
                        assert_tree_equal(self, full.runtime.state_dict(), partial.runtime.state_dict())
                        assert_tree_equal(self, full.runtime.model.measurement_state(), partial.runtime.model.measurement_state())
                model = full.runtime.model
                add_spatial_samples(expected, model.spatial_samples, model.realspace_samples, charged_profiles=True, clear=False)
                check_spatial(expected)
                for binary in (False, True):
                    output = Path(folder)/f'output_{binary}.npz'
                    save_runs(output, {8: expected}, binary_metadata=binary)
                    loaded = load_runs(output)[8]
                    check_spatial(loaded)
                    assert_tree_equal(self, expected['samples'], loaded['samples'])
                    self.assertEqual(loaded['metadata']['realspace_fields'][-4:],
                                     ['A_plus_z','A_minus_z','B_plus_plus','B_minus_minus'])
                # No measurements can alter the physical chain or either RNG.
                bare = make_simulation(gd, spatial=False)
                bare.run()
                assert_tree_equal(self, full.runtime.state_dict(), bare.runtime.state_dict())

    def test_rejection_and_failed_write_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.npz'
            sim = make_simulation()
            sim.run(max_sweeps=37, checkpoint_path=path)
            original = path.read_bytes()
            with patch('utils.checkpoint.os.replace', side_effect=OSError('simulated failed replacement')):
                with self.assertRaises(OSError):
                    sim.save_checkpoint(path)
            self.assertEqual(original, path.read_bytes())
            saved = read_checkpoint(path)
            saved['samples'].pop('curvature_xy')
            write_checkpoint(path, saved)
            target = make_simulation()
            before = target.runtime.state_dict()
            with self.assertRaises(ValueError):
                target.load_checkpoint(path)
            assert_tree_equal(self, before, target.runtime.state_dict())
            path.write_bytes(original)
            target.runtime.model.spatial_every = 3
            with self.assertRaises(ValueError):
                target.load_checkpoint(path)
            saved = read_checkpoint(path)
            saved['runtime']['walker_to_temperature'][0] = -1
            write_checkpoint(path, saved)
            with self.assertRaises(ValueError):
                make_simulation().load_checkpoint(path)
            path.write_bytes(original[:len(original)//2])
            with self.assertRaises(Exception):
                make_simulation().load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
