
import numpy as np

from core.runtime import Runtime
from core.simulation import Simulation

from models.ising.model import IsingModel
from models.ising.analysis import IsingAnalysis


sizes = [24, 32, 48, 64]
temperatures = np.linspace(1.5, 3.5, 21)

runs = {}

for L in sizes:
    print(f"Running Ising model: L = {L}")

    model = IsingModel(J=1.0)

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


analysis = IsingAnalysis(runs)

analysis.print_crossings()
analysis.plot()
