
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
