"""The existing sweep/measure loop, with resumable progress at sweep boundaries."""
from time import monotonic
import numpy as np
from core.parallel_tempering import ParallelTempering


class Simulation:
    def __init__(self, runtime, thermalization_sweeps=1000, measurement_sweeps=4000,
                 sample_interval=10, traffic_interval=0):
        if thermalization_sweeps < 0:
            raise ValueError('Thermalization sweeps cannot be negative')
        if measurement_sweeps < 1:
            raise ValueError('Measurement sweeps must be positive')
        if sample_interval < 1 or sample_interval > measurement_sweeps:
            raise ValueError('Invalid sampling interval')
        if traffic_interval < 0:
            raise ValueError('Invalid traffic interval')
        self.runtime = runtime
        self.thermalization_sweeps = thermalization_sweeps
        self.measurement_sweeps = measurement_sweeps
        self.sample_interval = sample_interval
        self.traffic_interval = traffic_interval
        self.completed_sweeps = 0
        self.samples = {}
        self.sample_sweeps = []
        self.traffic = []
        self._initial_pt_sweeps = runtime.sweeps if isinstance(runtime, ParallelTempering) else 0

    @property
    def complete(self):
        return self.completed_sweeps == self.thermalization_sweeps + self.measurement_sweeps

    def _record_traffic(self):
        if self.traffic_interval and isinstance(self.runtime, ParallelTempering):
            if not self.traffic or self.traffic[-1]['sweep'] != self.runtime.sweeps:
                self.traffic.append(self.runtime.traffic_counts())

    def run(self, *, max_sweeps=None, checkpoint_path=None,
            checkpoint_seconds=3600, stop_requested=None):
        """Continue at the next unfinished sweep, returning accumulated data.

        Optional stopping/checkpointing happens AFTER that sweep's exchanges and
        measurements. Repeated calls continue the same allocation, not a new run.
        A fresh Simulation retains the original one-shot run() behavior.
        """
        if max_sweeps is not None and max_sweeps < 0:
            raise ValueError('max_sweeps must be nonnegative')
        if checkpoint_seconds <= 0:
            raise ValueError('checkpoint_seconds must be positive')
        rt = self.runtime
        stop = self.thermalization_sweeps + self.measurement_sweeps
        if max_sweeps is not None:
            stop = min(stop, self.completed_sweeps + max_sweeps)
        last_checkpoint = monotonic()
        if self.completed_sweeps == 0:
            self._record_traffic()
        while self.completed_sweeps < stop:
            if stop_requested is not None and stop_requested():
                break
            rt.sweep()
            self.completed_sweeps += 1
            step = self.completed_sweeps - self.thermalization_sweeps
            if step > 0 and step % self.sample_interval == 0:
                measurement = rt.measure()
                if not measurement:
                    raise ValueError('Model returned no measurements')
                if not self.samples:
                    self.samples = {name: [] for name in measurement}
                if measurement.keys() != self.samples.keys():
                    raise ValueError('Measurement channels changed')
                for name, value in measurement.items():
                    value = np.asarray(value)
                    if value.ndim < 1 or value.shape[0] != rt.R:
                        raise ValueError(f'Invalid replica dimension for {name}')
                    self.samples[name].append(value.copy())
                self.sample_sweeps.append(self.completed_sweeps)
            if (self.traffic_interval and self.completed_sweeps % self.traffic_interval == 0
                    or step == 0 or self.complete):
                self._record_traffic()
            if checkpoint_path is not None and monotonic()-last_checkpoint >= checkpoint_seconds:
                self.save_checkpoint(checkpoint_path)
                last_checkpoint = monotonic()
        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)
        return self.result()

    def result(self):
        rt = self.runtime
        metadata = dict(L=rt.L, temperatures=rt.temperatures.copy(), seed=rt.seed,
                        model_parameters=rt.model.parameters(), **self.settings(),
                        sample_sweeps=np.asarray(self.sample_sweeps, dtype=np.int64),
                        completed_sweeps=self.completed_sweeps, complete=self.complete)
        if isinstance(rt, ParallelTempering):
            metadata['parallel_tempering'] = rt.diagnostics()
        if self.traffic:
            metadata['traffic_blocks'] = {k: np.stack([r[k] for r in self.traffic]) for k in self.traffic[0]}
        return dict(samples={k: np.stack(v) for k, v in self.samples.items()}, metadata=metadata)

    def settings(self):
        return dict(thermalization_sweeps=self.thermalization_sweeps,
                    measurement_sweeps=self.measurement_sweeps, sample_interval=self.sample_interval,
                    traffic_interval=self.traffic_interval)

    def save_checkpoint(self, path):
        from utils.checkpoint import write_checkpoint, checkpoint_stack
        from numba import cuda
        cuda.synchronize()
        state = dict(settings=self.settings(), runtime=self.runtime.state_dict(),
                     completed_sweeps=self.completed_sweeps, initial_pt_sweeps=self._initial_pt_sweeps,
                     samples={k: np.stack(v) for k, v in self.samples.items()},
                     sample_sweeps=np.asarray(self.sample_sweeps, dtype=np.int64),
                     traffic={k: np.stack([r[k] for r in self.traffic]) for k in self.traffic[0]} if self.traffic else {},
                     stack=checkpoint_stack())
        if hasattr(self.runtime.model, 'measurement_state'):
            state['measurements'] = self.runtime.model.measurement_state()
        write_checkpoint(path, state)

    def load_checkpoint(self, path):
        from utils.checkpoint import read_checkpoint, same_settings, checkpoint_stack
        saved = read_checkpoint(path)
        stack = checkpoint_stack()
        if saved['stack'] != stack:
            raise ValueError('Checkpoint GPU/Numba/NumPy stack differs; exact continuation is unvalidated')
        if not same_settings(saved['settings'], self.settings()):
            raise ValueError('Simulation checkpoint settings mismatch')
        count = saved['completed_sweeps']
        if not 0 <= count <= self.thermalization_sweeps+self.measurement_sweeps:
            raise ValueError('Invalid completed sweep count')
        expected = np.arange(self.thermalization_sweeps+self.sample_interval, count+1, self.sample_interval)
        if not np.array_equal(expected, saved['sample_sweeps']):
            raise ValueError('Invalid scalar sample coordinates')
        if (len(expected) and not saved['samples']) or any(len(v) != len(expected) or v.shape[1] != self.runtime.R for v in saved['samples'].values()):
            raise ValueError('Invalid scalar sample arrays')
        if isinstance(self.runtime, ParallelTempering) and saved['runtime']['sweeps'] != saved['initial_pt_sweeps']+count:
            raise ValueError('PT/Simulation progress mismatch')
        self.runtime.validate_state(saved['runtime'])
        if 'measurements' in saved:
            self.runtime.model.validate_measurement_state(saved['measurements'], saved['samples'], len(expected), self.runtime.R, self.runtime.L)
        # Validation above precedes all live-state mutation.
        self.runtime.load_state_dict(saved['runtime'])
        if 'measurements' in saved:
            self.runtime.model.load_measurement_state(saved['measurements'])
        self.completed_sweeps = count
        self._initial_pt_sweeps = saved['initial_pt_sweeps']
        self.samples = {k: list(v) for k, v in saved['samples'].items()}
        self.sample_sweeps = saved['sample_sweeps'].tolist()
        self.traffic = [dict(zip(saved['traffic'], row)) for row in zip(*saved['traffic'].values())]
