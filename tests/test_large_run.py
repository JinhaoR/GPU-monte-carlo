"""Tiny runner checks: exact continuation, output conflicts and graceful stopping."""
import contextlib
import io
from pathlib import Path
import shutil
import signal
import tempfile
import unittest
from unittest.mock import patch
from zipfile import BadZipFile

from numba import cuda
import large_run
from core.simulation import Simulation
from models.he3.sampling import He3GaussianRuntime
from tests.test_he3_campaign import assert_tree_equal
from utils.checkpoint import read_checkpoint
from utils.data_io import load_runs, save_runs


@unittest.skipUnless(cuda.is_available(), 'CUDA required')
class LargeRunnerTests(unittest.TestCase):
    settings = dict(sizes=[8], pressures=[20.], g_d_values=[0.], initializations=['random'],
                    allocations={8:(20,60)}, sample_interval=5, spatial_interval=10,
                    realspace_interval=15, traffic_interval=20, resume=True)

    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.fixture.name)
        original = Simulation.run
        with patch.multiple(large_run, **cls.settings, output_dir=cls.folder), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(Simulation, 'run', lambda sim, **kw: original(sim, max_sweeps=37, **kw)):
                large_run.main()
            cls.early_checkpoint = (cls.folder/'P=20_gd=0/L=8_random.checkpoint.npz').read_bytes()
            large_run.main()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        folder = self.root/'P=20_gd=0'
        folder.mkdir()
        for source in (self.folder/'P=20_gd=0').glob('*.npz'):
            shutil.copy2(source, folder/source.name)
        self.output = folder/'L=8_random.npz'
        self.checkpoint = folder/'L=8_random.checkpoint.npz'
        self.partial = folder/'L=8_random.partial.npz'

    def run_script(self, **overrides):
        with patch.multiple(large_run, **(self.settings | dict(output_dir=self.root) | overrides)), contextlib.redirect_stdout(io.StringIO()):
            large_run.main()

    def contents(self):
        return {p.name:p.read_bytes() for p in self.output.parent.glob('*.npz')}

    def assert_refused_without_writes(self, **overrides):
        before = self.contents()
        with patch.object(He3GaussianRuntime, 'sweep', side_effect=AssertionError('Unexpected local update')):
            with self.assertRaises((ValueError, KeyError, OSError, BadZipFile)):
                self.run_script(**overrides)
        self.assertEqual(before, self.contents())

    def test_completed_cases_skip_with_or_without_checkpoint(self):
        for remove_checkpoint in (False, True):
            if remove_checkpoint:
                self.checkpoint.unlink()
            before = self.contents()
            with patch.object(He3GaussianRuntime, 'sweep', side_effect=AssertionError('Completed case ran')):
                self.run_script(resume=False)
            self.assertEqual(before, self.contents())

    def test_resumed_random_trajectory_equals_uninterrupted(self):
        reference = read_checkpoint(self.checkpoint)
        samples = load_runs(self.output)[8]['samples']
        self.output.unlink()
        self.checkpoint.write_bytes(self.early_checkpoint)
        self.run_script()
        assert_tree_equal(self, reference, read_checkpoint(self.checkpoint))
        assert_tree_equal(self, samples, load_runs(self.output)[8]['samples'])
        # A fresh uninterrupted runner also agrees, including the initialization RNG.
        with tempfile.TemporaryDirectory() as fresh:
            self.run_script(output_dir=Path(fresh))
            assert_tree_equal(self, reference, read_checkpoint(Path(fresh)/'P=20_gd=0/L=8_random.checkpoint.npz'))

    def test_orphan_partial_and_disabled_resume_are_refused(self):
        self.output.unlink()
        self.assert_refused_without_writes(resume=False)
        self.checkpoint.unlink()
        self.assert_refused_without_writes()

    def test_completed_output_and_old_checkpoint_are_ambiguous(self):
        self.checkpoint.write_bytes(self.early_checkpoint)
        self.assert_refused_without_writes()

    def test_incompatible_settings_and_missing_channels_are_refused(self):
        self.assert_refused_without_writes(allocations={8:(20,80)})
        runs = load_runs(self.output)
        runs[8]['samples'].pop('curvature_xy')
        save_runs(self.output, runs, binary_metadata=True)
        self.assert_refused_without_writes()

    def test_conflicting_partial_or_corrupt_output_is_not_overwritten(self):
        runs = load_runs(self.partial)
        runs[8]['samples']['energy'][0,0] += 1
        save_runs(self.partial, runs, binary_metadata=True)
        self.assert_refused_without_writes()
        self.output.write_bytes(b'not an NPZ archive')
        self.assert_refused_without_writes()

    def test_profile_selection_checked_without_checkpoint(self):
        self.checkpoint.unlink()
        self.assert_refused_without_writes(profile_temperatures={0.:[.38,.39,.4]})

    def test_stop_signal_saves_checkpoint_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        original = Simulation.run
        def stop_on_entry(simulation, **kwargs):
            signal.raise_signal(signal.SIGTERM)
            return original(simulation, **kwargs)
        with tempfile.TemporaryDirectory() as fresh:
            with patch.object(Simulation, 'run', stop_on_entry):
                self.run_script(output_dir=Path(fresh))
            checkpoint = read_checkpoint(Path(fresh)/'P=20_gd=0/L=8_random.checkpoint.npz')
            self.assertEqual(checkpoint['completed_sweeps'], 0)
            self.assertFalse((Path(fresh)/'P=20_gd=0/L=8_random.npz').exists())
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_duplicate_case_seeds_are_refused(self):
        self.assert_refused_without_writes(sizes=[8,8])


if __name__ == '__main__':
    unittest.main()
