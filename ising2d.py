
import numpy as np
import math
from numba import cuda
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float32,
)
import matplotlib.pyplot as plt


@cuda.jit
def initialize_spins(spins):
    i, j = cuda.grid(2)

    if i < spins.shape[0] and j < spins.shape[1]:
        spins[i, j] = 1


@cuda.jit
def calculate_energy(spins, energies):
    i, j = cuda.grid(2)

    L = spins.shape[0] #Assume equal x and y length

    if i < L and j < L:
        s = spins[i, j]

        right = spins[i, (j + 1) % L]
        bottom = spins[(i + 1) % L, j]

        energies[i, j] = -s * (right + bottom)




@cuda.jit
def metropolis_color(spins, rng_states, betas, color):
    row, col, r = cuda.grid(3)

    L = spins.shape[1]

    i = 2 * row + color // 2
    j = 2 * col + color % 2

    if r < spins.shape[0] and i < L and j < L:
        s = spins[r, i, j]

        up = spins[r, (i - 1) % L, j]
        down = spins[r, (i + 1) % L, j]
        left = spins[r, i, (j - 1) % L]
        right = spins[r, i, (j + 1) % L]

        dE = 2 * s * (up + down + left + right)

        rng_index = r * L * L + i * L + j
        random_value = xoroshiro128p_uniform_float32(
            rng_states, rng_index
        )

        if dE <= 0 or random_value < math.exp(-betas[r] * dE):
            spins[r, i, j] = -s




def measure(spins_gpu):
    spins = spins_gpu.copy_to_host().astype(np.int32)

    M = np.sum(spins, axis=(1, 2))

    E = -np.sum(
        spins * (
            np.roll(spins, -1, axis=1)
            + np.roll(spins, -1, axis=2)
        ),
        axis=(1, 2)
    )

    return E, M



L = 32

temperatures = np.linspace(1.5, 3.5, 21)
R = len(temperatures)

betas_gpu = cuda.to_device(
    (1.0 / temperatures).astype(np.float32)
)

# One ordered lattice per temperature
spins_gpu = cuda.to_device(
    np.ones((R, L, L), dtype=np.int8)
)

# Separate random-number states for every spin
rng_states = create_xoroshiro128p_states(
    R * L * L, seed=42
)

threads_per_block = (16, 16, 1)

half_L = L // 2

blocks_per_grid = (
    (half_L + 15) // 16,
    (half_L + 15) // 16,
    R
)

thermalization_sweeps = 1000
measurement_sweeps = 4000
sample_interval = 10

energies = []
magnetizations = []

total_sweeps = thermalization_sweeps + measurement_sweeps

for sweep in range(total_sweeps):

    # All temperatures update concurrently on the GPU
    for color in range(4):
        metropolis_color[blocks_per_grid, threads_per_block](
            spins_gpu, rng_states, betas_gpu, color
        )

    # Only measure after thermalization
    if sweep >= thermalization_sweeps:
        sampling_step = sweep - thermalization_sweeps + 1

        if sampling_step % sample_interval == 0:
            E, M = measure(spins_gpu)

            energies.append(E)
            magnetizations.append(M)

E = np.asarray(energies, dtype=np.float64)
M = np.asarray(magnetizations, dtype=np.float64)


N = L * L

mean_E = np.mean(E, axis=0)
mean_E2 = np.mean(E**2, axis=0)

mean_M = np.mean(M, axis=0)
mean_abs_M = np.mean(np.abs(M), axis=0)
mean_M2 = np.mean(M**2, axis=0)
mean_M4 = np.mean(M**4, axis=0)

energy = mean_E / N
magnetization = mean_abs_M / N

heat_capacity = (
    (mean_E2 - mean_E**2) / (N * temperatures**2)
)

susceptibility = (
    (mean_M2 - mean_M**2) / (N * temperatures)
)

binder = 1 - mean_M4 / (3 * mean_M2**2)


Tc = 2 / np.log(1 + np.sqrt(2))

observables = [
    (energy, "Energy per spin"),
    (magnetization, "Absolute magnetization per spin"),
    (heat_capacity, "Heat capacity per spin"),
    (susceptibility, "Susceptibility per spin"),
    (binder, "Binder cumulant"),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 8))

for ax, (values, label) in zip(axes.flat, observables):
    ax.plot(temperatures, values, "o-")
    ax.axvline(Tc, linestyle="--", color="gray")

    ax.set_xlabel("Temperature")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)

axes.flat[-1].axis("off")

fig.suptitle(f"2D Ising model, L = {L}")
fig.tight_layout()

plt.show()



