"""Analyze Kawamura model simulations. Run from the project root.

Compatible with models.kawamura.analysis.KawamuraAnalysis and the shared
PTAnalysis, joint_histogram, and load_runs utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models.kawamura.analysis import KawamuraAnalysis
from utils.Analysis.parallel_tempering import PTAnalysis
from utils.Analysis.joint_histogram import joint_histogram, plot_joint_histogram
from utils.data_io import load_runs


# ============================================================
# SETTINGS
# ============================================================

# Can be a folder containing one L=*.npz file per size, or a single .npz file.
input_source = Path("data/kawamura")
file_pattern = "L=*.npz"

output_dir = Path("analysis_results/kawamura")
figure_dpi = 170

run_pt = True
run_joint = True
run_thermo = True
run_wm = True

# Joint distributions; None automatically selects the largest simulated size.
joint_L = None
joint_bins = 40
# Illustrative temperatures (in units of J), near transitions reported in
# Kawamura (1999). Set to temperatures relevant to your simulated grid.
joint_targets_over_J = {
    "chirality": 0.753,
    "phase": 0.692,
    "spin": 0.692,
}

block_size = 200
q = 1.0  # Vortex-charge hypothesis for theta (compare q=0.5).
wm_temperature_range = None


# ============================================================
# HELPERS
# ============================================================

def load_data():
    files = (
        [input_source]
        if input_source.is_file()
        else sorted(input_source.glob(file_pattern))
    )
    if not files:
        raise FileNotFoundError(
            f"No simulation files at {input_source} (pattern: {file_pattern})"
        )

    runs = {}
    for path in files:
        for L, run in load_runs(path).items():
            if L in runs:
                raise ValueError(f"Duplicate lattice size L={L}")
            runs[L] = run
        print(f"Loaded: {path}")

    runs = dict(sorted(runs.items()))
    reference = runs[next(iter(runs))]["metadata"]
    reference_T = np.asarray(reference["temperatures"])
    reference_J = float(reference["model_parameters"]["J"])
    for L, run in runs.items():
        meta = run["metadata"]
        T = np.asarray(meta["temperatures"])
        J = float(meta["model_parameters"]["J"])
        if T.shape != reference_T.shape or not np.allclose(
            T, reference_T, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f"Different temperature ladder for L={L}")
        if not np.isclose(J, reference_J):
            raise ValueError(f"Different coupling J for L={L}")
    return files, runs, reference_J


def main():
    files, runs, J = load_data()
    figures = {}
    report = [
        "# Kawamura Model Analysis",
        "",
        "Kawamura, Phys. Rev. Lett. 82, 964 (1999), Eq. (3).",
        "",
        f"**Input:** `{input_source}`",
        f"**Files:** {', '.join(p.name for p in files)}",
        f"**System sizes:** {list(runs)}",
        f"**Coupling:** J = {J:g}",
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
            "## Parallel Tempering", "", "```text", summary, "```", ""
        ]

        fig = pt.plot(show=False)
        if fig is not None:
            figures["parallel_tempering"] = fig

        fig = pt.plot_walkers(show=False)
        if fig is not None:
            figures["walker_temperature_history"] = fig

    # ========================================================
    # 2. JOINT DISTRIBUTIONS
    # ========================================================
    if run_joint:
        selected_L = max(runs) if joint_L is None else joint_L
        if selected_L not in runs:
            raise ValueError(
                f"Joint-distribution L={selected_L} not found; available: {list(runs)}"
            )

        run = runs[selected_L]
        T = np.asarray(run["metadata"]["temperatures"])
        samples = run["samples"]
        N = selected_L**2

        # Signed chirality is essential for seeing +/- Ising domains.
        # Phase and spin observables are gauge-invariant magnitudes.
        joint_variables = {
            "chirality": (
                samples["chirality"] / N,
                r"Signed chirality $m_\tau$",
            ),
            "phase": (
                np.hypot(samples["px"], samples["py"]) / N,
                r"Doubled-phase magnitude $|P_2|$",
            ),
            "spin": (
                np.sqrt(np.maximum(samples["spin2"], 0.0)) / N,
                r"Spin-nematic magnitude $\sqrt{D^2}$",
            ),
        }

        report += ["## Joint Probability Distributions", ""]
        for name, target_over_J in joint_targets_over_J.items():
            if name not in joint_variables:
                raise ValueError(f"Unknown joint observable: {name}")
            requested_T = J * target_over_J
            idx = int(np.argmin(np.abs(T - requested_T)))
            actual_T = float(T[idx])
            print(
                f"Joint {name}: L={selected_L}, requested T/J={target_over_J:.5f}, "
                f"actual T={actual_T:.5f} (T/J={actual_T / J:.5f})"
            )

            E = samples["energy"][:, idx] / N
            Y, ylabel = joint_variables[name]
            Y = Y[:, idx]
            valid = np.isfinite(E) & np.isfinite(Y)
            if not np.any(valid):
                raise ValueError(f"No finite samples for joint {name}")

            hist = joint_histogram(E[valid], Y[valid], bins=joint_bins)
            plots = plot_joint_histogram(
                hist,
                x_label=r"Energy density $H/L^2$",
                y_label=ylabel,
                title=f"Kawamura: energy vs {name}, L={selected_L}, T/J={actual_T / J:.5f}",
            )
            for plot_name, fig in plots.items():
                figures[f"joint_L{selected_L}_{name}_{plot_name}"] = fig

            report += [
                f"- **{name}:** L={selected_L}, requested T/J={target_over_J:.5f}, "
                f"actual T/J={actual_T / J:.5f}, bins={joint_bins}",
            ]
        report.append("")

    # ========================================================
    # 3. THERMODYNAMIC OBSERVABLES AND OPTIONAL WM FIT
    # ========================================================
    if run_thermo:
        kawamura = KawamuraAnalysis(runs, block_size=block_size)
        figures.update(kawamura.plot(C=None, q=q, show=False))
        report += [
            "## Thermodynamic Observables", "",
            f"- Block size: {block_size} saved samples",
            "- Helicity: x-direction only (current GPU measurement)",
            f"- Candidate vortex charge in theta: q={q:g}",
            "",
        ]

        if run_wm:
            try:
                fit = kawamura.fit_wm(
                    temperature_range=wm_temperature_range, q=q
                )
                Tc = float(fit["temperature"])
                C = float(fit["C"])
                dof = fit["degrees_of_freedom"]
                chi2_dof = fit["chi2"] / dof if dof > 0 else float("nan")

                # Keep the raw helicity figure and save the corrected one separately.
                corrected = kawamura.plot(
                    C=C, q=q, show=False, names=("helicity",)
                )["helicity"]
                figures["helicity_wm_corrected"] = corrected

                print(
                    f"WM (q={q:g}): Tc/J={Tc / J:.6f}, "
                    f"Tc={Tc:.6f}, C={C:.6f}, chi2/dof={chi2_dof:.3f}"
                )
                report += [
                    "### Exploratory Weber--Minnhagen Fit", "",
                    f"- Critical temperature: T = {Tc:.6f}, T/J = {Tc / J:.6f}",
                    f"- Correction constant: C = {C:.6f}",
                    f"- Chi-squared per degree of freedom: {chi2_dof:.3f}",
                    "",
                    r"$$\widetilde{\Upsilon}_{x,L}(T) "
                    r"= \frac{\Upsilon_{x,L}(T)}{1+1/(2\ln L+C)}$$",
                    "",
                    r"$$\Upsilon(T_{\mathrm{BKT}}^-)=\frac{2T_{\mathrm{BKT}}}{\pi q^2}$$",
                    "",
                    "This is a test of a BKT scaling hypothesis, not proof of "
                    "a BKT transition or of the vortex charge.",
                    "",
                ]
            except ValueError as exc:
                print(f"WM fit/correction skipped: {exc}")
                report += ["### Weber--Minnhagen Fit", "", f"Skipped: {exc}", ""]

    # ========================================================
    # 4. SAVE FIGURES AND REPORT
    # ========================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        image_path = output_dir / f"{name}.png"
        fig.savefig(image_path, dpi=figure_dpi, bbox_inches="tight")
        report += [
            f"## {name.replace('_', ' ').title()}",
            "", f"![{name}](./{image_path.name})", "",
        ]

    report_path = output_dir / "analysis.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved analysis to {report_path}")

    if figures:
        plt.show()


if __name__ == "__main__":
    main()
