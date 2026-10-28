
from pathlib import Path

import matplotlib.pyplot as plt

from models.he3.analysis import He3Analysis
from utils.Analysis.parallel_tempering import PTAnalysis
from utils.data_io import load_runs

import numpy as np

from utils.Analysis.joint_histogram import (
    joint_histogram,
    plot_joint_histogram,
)


# Settings
input_file = "data/he3_run.npz"
output_dir = Path("analysis_results/he3")
block_size = 200

# Vortex charge hypothesis
q = 0.5  # Ordinary vortices: 1.0; half-quantum vortices: 0.5

# Optional fit window
wm_temperature_range = None


# Joint probability distributions
joint_L = 96
joint_temperature = 0.39
joint_bins = 40

# Load data
runs = load_runs(input_file)


# He3 analysis
he3 = He3Analysis(runs, block_size=block_size)


# Weber–Minnhagen fit
fit = he3.fit_wm(
    temperature_range=wm_temperature_range,
    q=q,
)

Tc = fit["temperature"]
C = fit["C"]

dof = fit["degrees_of_freedom"]
chi2_dof = fit["chi2"] / dof if dof > 0 else float("nan")

print(f"Weber–Minnhagen (q={q:g}): Tc = {Tc:.6f}, C = {C:.6f}")
print(f"Chi2/dof = {chi2_dof:.3f}")



# Four independent figures
figures = he3.plot(C=C, q=q, show=False)

# Add fitted transition temperature to the helicity figure
ax = figures["helicity"].axes[0]

ax.legend(frameon=False)


# ----------------------------------------
# Joint probability distributions
# ----------------------------------------

run = runs[joint_L]

temperatures = np.asarray(run["metadata"]["temperatures"])
idx = np.argmin(np.abs(temperatures - joint_temperature))
actual_T = float(temperatures[idx])

print(
    f"Joint distributions: L={joint_L}, "
    f"requested T={joint_temperature:.5f}, "
    f"actual T={actual_T:.5f}"
)

s = run["samples"]
N = joint_L**2

# Extract simultaneous measurements at the selected temperature
E = s["energy"][:, idx]
D = s["density"][:, idx]
C_chi = s["chirality"][:, idx]
R = s["phase_real"][:, idx]
I = s["phase_imag"][:, idx]

# Keep valid measurements
valid = (
    np.isfinite(E)
    & np.isfinite(D)
    & (D > 0)
    & np.isfinite(C_chi)
    & np.isfinite(R)
    & np.isfinite(I)
)

E, D, C_chi, R, I = (
    array[valid] for array in (E, D, C_chi, R, I)
)

# Construct physical observables
e = E / N
n = D / N

chi = C_chi / D
abs_chi = np.abs(chi)

p2 = np.hypot(R, I) / N


# ----------------------------------------
# Select observable pairs
# ----------------------------------------

pairs = {
    "energy_chirality": (
        e, chi,
        r"Energy density $e$",
        r"Signed chirality $m_\chi$",
    ),
    "energy_phase": (
        e, p2,
        r"Energy density $e$",
        r"Composite phase magnitude $p_2$",
    ),
}


# ----------------------------------------
# Generate 2D and 3D plots
# ----------------------------------------

for name, (x, y, x_label, y_label) in pairs.items():

    hist = joint_histogram(
        x,
        y,
        bins=joint_bins,
    )

    plots = plot_joint_histogram(
        hist,
        x_label=x_label,
        y_label=y_label,
        title=f"He3: {name}, L={joint_L}, T={actual_T:.5f}",
    )

    for plot_name, fig in plots.items():
        figures[f"joint_L{joint_L}_{name}_{plot_name}"] = fig


# Parallel tempering analysis
pt = PTAnalysis(runs)

pt_summary = pt.format_summary()
print(pt_summary)

pt_figure = pt.plot(show=False)

if pt_figure is not None:
    figures["parallel_tempering"] = pt_figure

walker_figure = pt.plot_walkers(show=False)
if walker_figure is not None:
    figures["walker_temperature_history"] = walker_figure


# Save results as Markdown
output_dir.mkdir(parents=True, exist_ok=True)

report = [
    "# Full Thin-Film He3 Analysis",
    "",
    f"**Data:** `{input_file}`",
    f"**System sizes:** {he3.sizes}",
    f"**Block size:** {block_size}",
    f"**Vortex charge hypothesis:** $q = {q:g}$",
    "",
    "## Weber–Minnhagen Fit",
    "",
    f"- Critical temperature: $T_{{BKT}} = {Tc:.6f}$",
    f"- Correction constant: $C = {C:.6f}$",
    f"- Reduced chi-square: {chi2_dof:.3f}",
    "",
    "The helicity modulus is rescaled using:",
    "",
    r"$$",
    r"\widetilde{\Upsilon}_L(T) = "
    r"\frac{\Upsilon_L(T)}{1 + 1/(2\ln L+C)}",
    r"$$",
    "",
    "The corresponding candidate universal jump is:",
    "",
    r"$$",
    rf"\Upsilon(T_{{BKT}}^-) = \frac{{2T_{{BKT}}}}{{\pi ({q:g})^2}}",
    r"$$",
    "",
    "The WM fit is exploratory. The chirality Binder cumulants "
    "are inspected visually without fitting their crossings.",
    "",
]


# Save figures and embed them in Markdown
for name, fig in figures.items():
    image_path = output_dir / f"{name}.png"

    fig.savefig(
        image_path,
        dpi=170,
        bbox_inches="tight",
    )

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
    pt_summary,
    "```",
    "",
])


# Write Markdown report
report_path = output_dir / "analysis.md"

report_path.write_text(
    "\n".join(report),
    encoding="utf-8",
)

print(f"Saved analysis to {report_path}")

plt.show()
