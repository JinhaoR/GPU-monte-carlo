
import numpy as np

from IsingModel import IsingModel
from IsingRuntime import IsingRuntime
from IsingSimulation import IsingSimulation
from IsingAnalysis import IsingAnalysis


L = 32
temperatures = np.linspace(1.5, 3.5, 21)

# Physical model
model = IsingModel(J=1.0)

# GPU runtime
runtime = IsingRuntime(
    model=model,
    L=L,
    temperatures=temperatures,
    threads=16,
    seed=42
)

# Simulation
simulation = IsingSimulation(
    runtime=runtime,
    thermalization_sweeps=1000,
    measurement_sweeps=4000,
    sample_interval=10
)

E, M = simulation.run()

# Analysis
analysis = IsingAnalysis(
    L=L,
    temperatures=temperatures,
    E=E,
    M=M
)

analysis.plot()
