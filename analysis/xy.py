
from pathlib import Path

import matplotlib.pyplot as plt

from models.xy.analysis import XYAnalysis
from utils.Analysis.parallel_tempering import PTAnalysis
from utils.data_io import load_runs


# Settings
input_file = "data/xy_run.npz"
output_dir = Path("analysis_results/xy")
block_size = 200

# Load data
runs = load_runs(input_file)

# XY analysis
xy = XYAnalysis(runs, block_size=block_size)

# Weber–Minnhagen fit
fit = xy.fit_wm()

Tc = fit["temperature"]
C = fit["C"]
chi2_dof = fit["chi2"] / fit["degrees_of_freedom"]

print(f"Weber–Minnhagen: Tc = {Tc:.6f}, C = {C:.6f}")
print(f"Chi2/dof = {chi2_dof:.3f}")

# Plot observables with WM-corrected helicity
fig = xy.plot(C=C, show=False)

fig.axes[2].axvline(
    Tc,
    color="gray",
    linestyle=":",
    label=rf"$T_c={Tc:.5f}$"
)
fig.axes[2].legend()

figures = {"xy_observables": fig}

# Parallel tempering analysis
pt = PTAnalysis(runs)

print(pt.format_summary())

pt_figure = pt.plot(show=False)
if pt_figure is not None:
    figures["parallel_tempering"] = pt_figure


# Save results as Markdown
output_dir.mkdir(parents=True, exist_ok=True)

report = [
    "# XY Analysis",
    "",
    f"**Data:** `{input_file}`",
    f"**System sizes:** {xy.sizes}",
    f"**Block size:** {block_size}",
    "",
    "## Weber–Minnhagen Fit",
    "",
    f"- Critical temperature: $T_c = {Tc:.6f}$",
    f"- Correction constant: $C = {C:.6f}$",
    f"- Reduced chi-square: {chi2_dof:.3f}",
    "",
    "The helicity modulus is rescaled using:",
    "",
    r"$$\widetilde{\Upsilon}_L(T) = "
    r"\frac{\Upsilon_L(T)}{1 + 1/(2\ln L+C)}$$",
    "",
]

# Save figures and embed them in Markdown
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

# Write Markdown report
(output_dir / "analysis.md").write_text(
    "\n".join(report),
    encoding="utf-8"
)

plt.show()
