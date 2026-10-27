
from pathlib import Path

import matplotlib.pyplot as plt

from models.ising.analysis import IsingAnalysis
from utils.Analysis.parallel_tempering import PTAnalysis
from utils.data_io import load_runs


# Settings
input_file = "data/ising_run.npz"
output_dir = Path("analysis_results/ising")
block_size = 200

# Load data
runs = load_runs(input_file)

# Ising analysis
ising = IsingAnalysis(runs, block_size=block_size)

figures = {
    "ising_observables": ising.plot(show=False),
}

# Parallel tempering analysis
pt = PTAnalysis(runs)

print(pt.format_summary())

pt_figure = pt.plot(show=False)
if pt_figure is not None:
    figures["parallel_tempering"] = pt_figure


# Save results as Markdown
output_dir.mkdir(parents=True, exist_ok=True)

report = [
    "# Ising Analysis",
    "",
    f"**Data:** `{input_file}`",
    f"**System sizes:** {ising.sizes}",
    f"**Block size:** {block_size}",
    "",
]

# Save plots and embed them in Markdown
for name, fig in figures.items():
    image_path = output_dir / f"{name}.png"
    fig.savefig(image_path, dpi=170, bbox_inches="tight")

    report.extend([
        f"## {name.replace('_', ' ').title()}",
        "",
        f"![{name}](./{image_path.name})",
        "",
    ])

# Parallel tempering summary
report.extend([
    "## Parallel Tempering Summary",
    "",
    "```text",
    pt.format_summary(),
    "```",
    "",
])

# Write Markdown file
(output_dir / "analysis.md").write_text(
    "\n".join(report),
    encoding="utf-8"
)

plt.show()
