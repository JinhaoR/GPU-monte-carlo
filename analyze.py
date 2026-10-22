
import json
from pathlib import Path

import matplotlib.pyplot as plt

from models.xy.analysis import XYAnalysis
from utils.Analysis.parallel_tempering import PTAnalysis
from utils.data_io import load_runs


# Settings
input_file = "data/xy_run.npz"
output_dir = Path("analysis_results/xy")
block_size = 200
n_resamples = 400

# Load data
runs = load_runs(input_file)

# XY analysis
xy = XYAnalysis(runs, block_size=block_size, n_resamples=n_resamples)

print(xy.format_summary())

figures = {
    "xy_observables": xy.plot(show=False),
    "xy_transitions": xy.plot_transitions(show=False),
}

# Parallel tempering analysis
pt = PTAnalysis(runs)

print(pt.format_summary())

pt_figure = pt.plot(show=False)
if pt_figure is not None:
    figures["parallel_tempering"] = pt_figure


# Save results
output_dir.mkdir(parents=True, exist_ok=True)

for name, fig in figures.items():
    fig.savefig(output_dir / f"{name}.png", dpi=170)
    fig.savefig(output_dir / f"{name}.pdf")

with open(output_dir / "analysis.json", "w") as f:
    json.dump({"xy": xy.summary(), "parallel_tempering": pt.summary()}, f, indent=2)


plt.show()
