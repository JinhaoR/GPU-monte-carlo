
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models.he3.analysis import He3Analysis
from utils.Analysis.parallel_tempering import PTAnalysis
from utils.Analysis.joint_histogram import (
    joint_histogram,
    plot_joint_histogram,
)
from utils.data_io import load_runs


# ============================================================
# SETTINGS
# ============================================================

# General
input_file = "data/he3_gaussian_run.npz"
output_dir = Path("analysis_results/he3")
figure_dpi = 170

# Enable/disable analysis sections
run_pt = False
run_joint = True
run_thermo = False


# ------------------------------------------------------------
# 1. Parallel Tempering
# ------------------------------------------------------------

# No additional settings


# ------------------------------------------------------------
# 2. Joint Probability Distributions
# ------------------------------------------------------------

joint_L = 96
joint_temperature = 0.384
joint_bins = 40


# ------------------------------------------------------------
# 3. Thermodynamic Observables
# ------------------------------------------------------------

block_size = 200

# 1.0 = ordinary vortices
# 0.5 = half-quantum vortices
q = 0.5

wm_temperature_range = None


# ============================================================
# INITIALIZATION
# ============================================================

runs = load_runs(input_file)
figures = {}

report = [
    "# Full Thin-Film He3 Analysis",
    "",
    f"**Data:** `{input_file}`",
    f"**System sizes:** {sorted(runs)}",
    "",
]


# ============================================================
# 1. PARALLEL TEMPERING
# ============================================================

if run_pt:

    pt = PTAnalysis(runs)

    pt_summary = pt.format_summary()
    print(pt_summary)

    report.extend([
        "## Parallel Tempering Summary",
        "",
        "```text",
        pt_summary,
        "```",
        "",
    ])

    # Swap acceptance
    fig = pt.plot(show=False)
    if fig is not None:
        figures["parallel_tempering"] = fig

    # Walker histories
    fig = pt.plot_walkers(show=False)
    if fig is not None:
        figures["walker_temperature_history"] = fig


# ============================================================
# 2. JOINT PROBABILITY DISTRIBUTIONS
# ============================================================

if run_joint:

    run = runs[joint_L]

    temperatures = np.asarray(
        run["metadata"]["temperatures"]
    )

    idx = np.argmin(
        np.abs(temperatures - joint_temperature)
    )
    actual_T = float(temperatures[idx])

    print(
        f"Joint distributions: L={joint_L}, "
        f"requested T={joint_temperature:.5f}, "
        f"actual T={actual_T:.5f}"
    )

    s = run["samples"]
    N = joint_L**2

    # Simultaneous measurements
    E = s["energy"][:, idx]
    D = s["density"][:, idx]
    C_chi = s["chirality"][:, idx]
    R = s["phase_real"][:, idx]
    I = s["phase_imag"][:, idx]

    # Valid measurements
    valid = (
        np.isfinite(E)
        & np.isfinite(D)
        & (D > 0)
        & np.isfinite(C_chi)
        & np.isfinite(R)
        & np.isfinite(I)
    )

    E, D, C_chi, R, I = (
        array[valid]
        for array in (E, D, C_chi, R, I)
    )

    if E.size == 0:
        raise ValueError("No valid joint-distribution samples")

    # Physical observables
    e = E / N
    chi = C_chi / D
    p2 = np.hypot(R, I) / N

    # Observable pairs
    pairs = {
        "energy_chirality": (
            e,
            chi,
            r"Energy density $e$",
            r"Signed chirality $m_\chi$",
        ),
        "energy_phase": (
            e,
            p2,
            r"Energy density $e$",
            r"Composite phase magnitude $p_2$",
        ),
    }

    # Generate 2D and 3D figures
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

    report.extend([
        "## Joint Probability Distributions",
        "",
        f"- System size: L = {joint_L}",
        f"- Temperature: {actual_T:.5f}",
        f"- Histogram bins: {joint_bins}",
        "",
    ])


# ============================================================
# 3. THERMODYNAMIC OBSERVABLES
# ============================================================

if run_thermo:

    he3 = He3Analysis(runs, block_size=block_size)

    # Weber–Minnhagen fit
    fit = he3.fit_wm(
        temperature_range=wm_temperature_range,
        q=q,
    )

    Tc = fit["temperature"]
    C = fit["C"]

    dof = fit["degrees_of_freedom"]
    chi2_dof = (
        fit["chi2"] / dof
        if dof > 0
        else float("nan")
    )

    print(
        f"Weber–Minnhagen (q={q:g}): "
        f"Tc = {Tc:.6f}, C = {C:.6f}"
    )
    print(f"Chi2/dof = {chi2_dof:.3f}")

    # Thermodynamic figures
    thermo_figures = he3.plot(
        C=C,
        q=q,
        show=False,
    )

    thermo_figures["helicity"].axes[0].legend(
        frameon=False
    )

    figures.update(thermo_figures)

    report.extend([
        "## Thermodynamic Observables",
        "",
        f"- Block size: {block_size}",
        f"- Vortex charge hypothesis: q = {q:g}",
        "",
        "### Weber–Minnhagen Fit",
        "",
        f"- Critical temperature: $T_{{BKT}} = {Tc:.6f}$",
        f"- Correction constant: $C = {C:.6f}$",
        f"- Reduced chi-square: {chi2_dof:.3f}",
        "",
        "WM-corrected helicity:",
        "",
        "$$",
        r"\widetilde{\Upsilon}_L(T) = "
        r"\frac{\Upsilon_L(T)}{1 + 1/(2\ln L+C)}",
        "$$",
        "",
        "Candidate universal jump:",
        "",
        "$$",
        rf"\Upsilon(T_{{BKT}}^-) = "
        rf"\frac{{2T_{{BKT}}}}{{\pi ({q:g})^2}}",
        "$$",
        "",
        "The WM fit is exploratory.",
        "",
    ])


# ============================================================
# SAVE RESULTS
# ============================================================

output_dir.mkdir(parents=True, exist_ok=True)

for name, fig in figures.items():

    image_path = output_dir / f"{name}.png"

    fig.savefig(
        image_path,
        dpi=figure_dpi,
        bbox_inches="tight",
    )

    report.extend([
        f"## {name.replace('_', ' ').title()}",
        "",
        f"![{name}](./{image_path.name})",
        "",
    ])


report_path = output_dir / "analysis.md"

report_path.write_text(
    "\n".join(report),
    encoding="utf-8",
)

print(f"Saved analysis to {report_path}")

if figures:
    plt.show()
