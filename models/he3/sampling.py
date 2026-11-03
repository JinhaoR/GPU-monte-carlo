
import numpy as np
from core.runtime import Runtime

from .model import He3Model
from .gaussian_fast import FastGaussianUpdater


class He3GaussianRuntime(Runtime):

    def __init__(
        self, model, L, temperatures,
        threads=16, seed=42,
        rho=-0.9, proposal_mass=0.0,
    ):
        if not isinstance(model, He3Model):
            raise TypeError("Requires He3Model")

        super().__init__(
            model, L, temperatures,
            threads=threads, seed=seed,
        )

        self.gaussian = FastGaussianUpdater(
            self,
            rho=rho,
            mass=proposal_mass,
        )

    def sweep(self):
        self.gaussian.step()

    def sampling_diagnostics(self):
        return self.gaussian.diagnostics()

    def checkpoint_settings(self):
        return dict(super().checkpoint_settings(), rho=float(self.gaussian.rho),
                    proposal_mass=float(self.gaussian.mass))

    def state_dict(self):
        return dict(super().state_dict(), gaussian_counts=self.gaussian.counts.copy_to_host(),
                    gaussian_sweeps=self.gaussian.sweeps)

    def validate_state(self, saved):
        from utils.checkpoint import check_array
        super().validate_state(saved)
        check_array(saved['gaussian_counts'], self.gaussian.counts, 'gaussian_counts')
        if saved['gaussian_sweeps'] < 0:
            raise ValueError('Invalid Gaussian sweep count')
        counts = saved['gaussian_counts']
        if (np.any(counts < 0) or not np.all(counts[0] == saved['gaussian_sweeps'])
                or np.any(counts[1:] > counts[0])):
            raise ValueError('Invalid Gaussian attempt/acceptance counters')

    def load_state_dict(self, saved):
        super().load_state_dict(saved)
        self.gaussian.counts.copy_to_device(saved['gaussian_counts'])
        self.gaussian.sweeps = saved['gaussian_sweeps']
