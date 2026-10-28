
import numpy as np
import matplotlib.pyplot as plt

from models.he3.model import He3Model
from models.he3.sampling import He3GaussianRuntime


# Settings
L = 64
g_d = 0.001
temperatures = [0.375, 0.381, 0.385]
seeds = [42, 123]

sweeps = 30_000
sample_interval = 200


# Simulation
results = {}

for direction in ["z", "x"]:
    for seed in seeds:

        print(f"Running: initial spin={direction}, seed={seed}")

        model = He3Model(
            K=1.0,
            delta=0.15,
            pressure=20.0,
            reduced_temperature=1.0,
            g_d=g_d,
        )

        rt = He3GaussianRuntime(
            model=model,
            L=L,
            temperatures=temperatures,
            seed=seed,
            rho=-0.9,
            proposal_mass=0.0,
        )

        # Initialize spin orientation
        if direction == "x":
            state = model.initialize_state(L, len(temperatures), seed)
            state[..., 0, :] = state[..., 2, :]
            state[..., 2, :] = 0

            rt.state.copy_to_device(state)
            rt.invalidate_energy_cache()

        # Record Qzz throughout the simulation
        history = [rt.measure()["spin_alignment"]]

        for step in range(1, sweeps + 1):
            rt.sweep()

            if step % sample_interval == 0:
                history.append(rt.measure()["spin_alignment"])

        results[(direction, seed)] = np.asarray(history)


# Plot
times = np.arange(0, sweeps + 1, sample_interval)

fig, axes = plt.subplots(
    1, len(temperatures),
    figsize=(14, 4),
    sharey=True,
)

for i, T in enumerate(temperatures):
    ax = axes[i]

    for direction in ["z", "x"]:
        for seed in seeds:
            qzz = results[(direction, seed)][:, i]

            ax.plot(
                times,
                qzz,
                label=f"Start {direction}, seed={seed}",
            )

    ax.set_title(f"T = {T}")
    ax.set_xlabel("Monte Carlo sweeps")
    ax.grid(alpha=0.3)

axes[0].set_ylabel(r"$Q_{zz}$")
axes[-1].legend(fontsize=8)

fig.suptitle(f"Spin alignment relaxation, L={L}, g_D={g_d}")
plt.tight_layout()
plt.show()
