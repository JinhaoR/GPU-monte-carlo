
import numpy as np

from core.runtime import Runtime
from core.simulation import Simulation

from models.xy.model import XYModel
from models.xy.analysis import XYAnalysis


sizes = [32, 48, 64]
temperatures = np.linspace(0.78, 1.02, 25)

runs = {}

for L in sizes:
    print(f"Running XY model: L = {L}")

    model = XYModel(J=1.0, delta=1.5)

    runtime = Runtime(
        model=model,
        L=L,
        temperatures=temperatures,
        threads=16,
        seed=42 + L
    )

    simulation = Simulation(
        runtime=runtime,
        thermalization_sweeps=10000,
        measurement_sweeps=100000,
        sample_interval=10
    )

    runs[L] = simulation.run()


analysis = XYAnalysis(runs)

analysis.print_crossings()
analysis.plot()
