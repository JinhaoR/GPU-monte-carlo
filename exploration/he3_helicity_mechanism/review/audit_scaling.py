"""Deterministic assumption audit of published tables; no fits or Monte Carlo.

Run from any directory with Python's standard library. Inputs are read only.
Numbers here illustrate the size of omitted/assumed corrections, not estimates
of physical spin stiffness or a short-range background in the He3 simulation.
"""
from pathlib import Path
import csv
import json
import math


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "he3_transition_study" / "results"
OUT = HERE / "results"


def rows(name):
    with (SOURCE / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save(name, values):
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corrections = []
    for row in rows("wm_fits.csv"):
        if row["sizes"] != "64:96:128:160":
            continue
        C = float(row["C"])
        for L in (64, 96, 128, 160, 256):
            denominator = 2 * math.log(L) + C
            corrections.append(dict(
                g_d=float(row["g_d"]), q=float(row["q"]), L=L,
                fitted_temperature=float(row["temperature"]), C=C,
                denominator=denominator, leading_relative_correction=1/denominator,
                formal_pole_length=math.exp(-C/2),
                scope="held_C_extrapolation" if L == 256 else "fitted_size",
            ))
    save("wm_correction_sizes.csv", corrections)

    background = []
    for row in rows("phase_consistency.csv"):
        L1, L2 = int(row["L1"]), int(row["L2"])
        s = L2 / L1
        observed = float(row["eta_P_effective"])
        reference = float(row["gaussian_from_helicity"])
        # M2(L)=A L^-eta + b L^-2. Solve the TWO-SIZE ratio for the
        # background fraction at L1. Each pair has its own eta/reference.
        # This construction does NOT validate one common A,b across sizes.
        fraction = (s**(-observed)-s**(-reference)) / (s**(-2)-s**(-reference))
        background.append(dict(
            g_d=float(row["g_d"]), L1=L1, L2=L2,
            observed_effective_eta=observed, reference_eta=reference,
            eta_difference=observed-reference,
            two_size_white_background_fraction_at_L1=fraction,
            hypothetical_spin_over_mass_from_observed_eta=4*observed-1,
            hypothetical_spin_over_mass_from_helicity_eta=4*reference-1,
        ))
    save("conditional_consistency_algebra.csv", background)
    # Exact checks of the algebra, including an exactly zero background.
    eta, s, f = 0.23, 1.5, 0.17
    ratio = (1-f)*s**(-eta)+f*s**(-2)
    recovered = (ratio-s**(-eta))/(s**(-2)-s**(-eta))
    assert abs(recovered-f) < 1e-14
    assert abs((s**(-eta)-s**(-eta))/(s**(-2)-s**(-eta))) < 1e-14
    checks = dict(
        inputs=[str(SOURCE / name) for name in ("wm_fits.csv", "phase_consistency.csv")],
        no_new_fit=True, no_monte_carlo=True,
        algebra_check_max_error=abs(recovered-f),
        cautions=[
            "WM pole length is an ansatz scale, not a measured core radius.",
            "L256 values hold old fitted C fixed and are not predictions with uncertainty.",
            "Two-size background fractions are independent conditional illustrations; no joint background fit was made.",
            "Spin ratios assume a decoupled mass/spin HQV marginality relation; they are not measured stiffnesses.",
            "Published effective exponents and Gaussian references have correlated uncertainty not propagated here.",
        ],
    )
    (OUT / "audit_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(dict(wm_rows=len(corrections), conditional_rows=len(background),
                          algebra_check_max_error=abs(recovered-f))))


if __name__ == "__main__":
    main()
