
"""Analyze He3 simulation files. Run from the project root."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models.he3.analysis import He3Analysis
from utils.Analysis.parallel_tempering import PTAnalysis
from utils.Analysis.joint_histogram import joint_histogram, plot_joint_histogram
from utils.data_io import load_runs


# ============================================================
# SETTINGS
# ============================================================

pressure = 20.0
g_d = 0.01
sampler = "gaussian"

folder = f"P={pressure:g}_gd={g_d:g}_{sampler}"

# Input files
input_dir = Path("data/he3") / folder
file_pattern = "L=*.npz"

# Output
output_dir = Path("analysis_results/he3") / folder
figure_dpi = 170

# Analysis sections
run_pt = True
run_joint = True
run_thermo = True

# Joint distributions
joint_L = 96
joint_temperature = 0.381
joint_bins = 40

# Thermodynamic observables
block_size = 200
q = 0.5
wm_temperature_range = None


# ============================================================
# ANALYSIS
# ============================================================

def main():

    # Load matching files
    files = (
        [input_dir]
        if input_dir.is_file()
        else sorted(input_dir.glob(file_pattern))
    )

    if not files:
        raise FileNotFoundError(
            f"No simulation files found in {input_dir}"
        )

    runs = {}

    for path in files:
        loaded = load_runs(path)

        for L, run in loaded.items():
            if L in runs:
                raise ValueError(f"Duplicate lattice size L={L}")
            runs[L] = run

        print(f"Loaded: {path}")

    runs = dict(sorted(runs.items()))

    figures = {}
    report = [
        "# Full Thin-Film He3 Analysis",
        "",
        f"**Input:** `{input_dir}`",
        f"**Files:** {', '.join(p.name for p in files)}",
        f"**System sizes:** {list(runs)}",
        "",
    ]

    # ========================================================
    # 1. PARALLEL TEMPERING
    # ========================================================

    if run_pt:

        pt = PTAnalysis(runs)

        summary = pt.format_summary()
        print(summary)

        report += [
            "## Parallel Tempering Summary",
            "",
            "```text",
            summary,
            "```",
            "",
        ]

        fig = pt.plot(show=False)
        if fig is not None:
            figures["parallel_tempering"] = fig

        fig = pt.plot_walkers(show=False)
        if fig is not None:
            figures["walker_temperature_history"] = fig

    # ========================================================
    # 2. JOINT PROBABILITY DISTRIBUTIONS
    # ========================================================

    if run_joint:

        if joint_L not in runs:
            raise ValueError(
                f"L={joint_L} not found. Available: {list(runs)}"
            )

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

        samples = run["samples"]

        E = samples["energy"][:, idx]
        D = samples["density"][:, idx]
        C = samples["chirality"][:, idx]
        R = samples["phase_real"][:, idx]
        I = samples["phase_imag"][:, idx]

        valid = (
            np.isfinite(E)
            & np.isfinite(D)
            & (D > 0)
            & np.isfinite(C)
            & np.isfinite(R)
            & np.isfinite(I)
        )

        if not np.any(valid):
            raise ValueError("No valid joint-distribution samples")

        N = joint_L**2

        e = E[valid] / N
        chi = C[valid] / D[valid]
        p2 = np.hypot(R[valid], I[valid]) / N

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
                title=(
                    f"He3: {name}, "
                    f"L={joint_L}, T={actual_T:.5f}"
                ),
            )

            for plot_name, fig in plots.items():
                figures[
                    f"joint_L{joint_L}_{name}_{plot_name}"
                ] = fig

        report += [
            "## Joint Probability Distributions",
            "",
            f"- System size: L = {joint_L}",
            f"- Requested temperature: {joint_temperature:.5f}",
            f"- Actual temperature: {actual_T:.5f}",
            f"- Histogram bins: {joint_bins}",
            "",
        ]

    # ========================================================
    # 3. THERMODYNAMIC OBSERVABLES
    # ========================================================

    if run_thermo:

        he3 = He3Analysis(runs, block_size=block_size)

        fit = he3.fit_wm(
            temperature_range=wm_temperature_range,
            q=q,
        )

        Tc = fit["temperature"]
        C = fit["C"]

        dof = fit["degrees_of_freedom"]
        chi2_dof = (
            fit["chi2"] / dof if dof > 0 else float("nan")
        )

        print(
            f"Weber–Minnhagen (q={q:g}): "
            f"Tc = {Tc:.6f}, C = {C:.6f}"
        )
        print(f"Chi2/dof = {chi2_dof:.3f}")

        thermo_figures = he3.plot(
            C=C,
            q=q,
            show=False,
        )

        thermo_figures["helicity"].axes[0].legend(
            frameon=False
        )

        figures.update(thermo_figures)

        report += [
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
        ]

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():

        image_path = output_dir / f"{name}.png"

        fig.savefig(
            image_path,
            dpi=figure_dpi,
            bbox_inches="tight",
        )

        report += [
            f"## {name.replace('_', ' ').title()}",
            "",
            f"![{name}](./{image_path.name})",
            "",
        ]

    report_path = output_dir / "analysis.md"

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"Saved analysis to {report_path}")

    if figures:
        plt.show()


if __name__ == "__main__":
    main()
