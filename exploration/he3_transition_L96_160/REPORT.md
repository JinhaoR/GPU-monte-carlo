# Transition estimates using L = 96, 128, 160

Analysis of `data/weak_soc_spatial_simple_01`, 1 November 2026. Only the six archives for these three sizes and g_D = 0, 0.01 enter this analysis. L = 64 is excluded from preparation, crossings, fits, and estimates. No simulations were run or production files changed.

The half-quantum-vortex Weber–Minnhagen (WM) hypothesis fits the three sizes at both couplings. The integer-vortex hypothesis fails at zero SOC and is marginal and method-sensitive at g_D = 0.01. **The g_D = 0.01 Binder crossing exhibits substantial production-time drift: its full-history bootstrap error is not a reliable total uncertainty.**

| g_D | Chirality T_c proxy: crossing (128,160) | Helicity T_c, q=1 | Helicity T_c, q=0.5: WM |
| --- | --- | --- | --- |
| 0 | 0.381837 ± 0.000030 | No acceptable WM fit | 0.381502 ± 0.000019 |
| 0.01 | 0.385584 ± 0.000039; time drift ≈ 0.00028 | No robust estimate; marginal candidate ≈ 0.3895 | 0.385245 ± 0.000023 |

All ± values here and below are **conditional one-standard-error block-bootstrap statistical uncertainties**. They exclude finite-size/model uncertainty and failure of stationarity. Chirality entries are explicitly finite-size proxies, not thermodynamic extrapolations. WM entries assume the stated scalar BKT scaling form; a good fit does not identify vortex charge in the full multicomponent model.

## Data and uncertainty method

Each archive contains 60,000 scalar measurements at 200 temperatures spanning 0.37–0.42, following 3 million thermalization sweeps and covering 3 million production sweeps, sampled every 50 sweeps. Thus raw histories, rather than only summarized means, are available. There is one history per size/coupling. [The inventory](results/inventory.json) records sources and checks against earlier archive-member CRC metadata; moments recomputed from raw histories agree with the previous spatial-analysis cache exactly.

The primary analysis resamples 30 nonoverlapping blocks of 100,000 sweeps, with 600 bootstrap replicates. A block contains **all temperature slots and all channels together**, preserving contemporaneous PT correlations and correlations between the Binder and helicity estimators. Different lattice-size histories are resampled independently. Every replicate recomputes nonlinear moments, energy reweighting, crossings, and WM fits. Sensitivity analyses use 250,000- and 500,000-sweep blocks (12 and 6 blocks), and separate production halves, with 200 replicates each. These checks assess robustness but cannot certify independence or equilibration with so few long blocks.

For Binder and WM estimates, local multiple-histogram energy reweighting (WHAM) reuses `exploration/he3_transition_study/reweight.py`. The ordering window uses 13 recorded source temperatures; the integer-jump window uses 11. Histograms retain 800 extensive-energy bins and source-temperature counts, using the actual float32 inverse temperatures. The WHAM equations are solved anew after block resampling. Dense evaluation curves are interpolants of the same histories, **not independent measurements**. A WM fit has just three size-dependent observations at a trial temperature and two parameters, hence one nominal degree of freedom. It does not fit hundreds of interpolated points as independent data.

Direct universal-jump crossings instead use linear interpolation of the original temperature-grid means, with whole-curve bootstrap errors. This distinction allows an explicit check on interpolation dependence.

## Chirality crossings and finite-size drift

The existing normalized chirality is m = global chirality / global density, evaluated configuration by configuration. Its Binder cumulant is

\[
U_L=1-\frac{\langle m^4\rangle}{3\langle m^2\rangle^2}.
\]

| g_D | Size pair | Crossing temperature ± statistical SE | Bootstrap 95% percentile interval |
| --- | --- | --- | --- |
| 0 | 96,128 | 0.381849 ± 0.000022 | [0.381808, 0.381892] |
| 0 | 128,160 | 0.381837 ± 0.000030 | [0.381778, 0.381899] |
| 0 | 96,160 | 0.381843 ± 0.000015 | [0.381814, 0.381874] |
| 0.01 | 96,128 | 0.385552 ± 0.000022 | [0.385513, 0.385597] |
| 0.01 | 128,160 | 0.385584 ± 0.000039 | [0.385504, 0.385653] |
| 0.01 | 96,160 | 0.385567 ± 0.000016 | [0.385534, 0.385596] |

Each selected ordering-region crossing is unique in the search window and survives all 600 main replicates. The windows are centered at 0.38183 and 0.38558 with half-width 0.00065. Additional raw-curve crossings elsewhere, including in the negative-dip region and high-temperature tail, are catalogued with bootstrap matching diagnostics in [all_raw_binder_roots.csv](results/all_raw_binder_roots.csv). These are not silently averaged into an ordering temperature. Root matching there uses a finite neighborhood and can be ambiguous; its survival fraction is not evidence for another transition.

The difference T_(128,160) − T_(96,128) is −0.000012 ± 0.000042 at zero SOC and +0.000031 ± 0.000054 at g_D = 0.01. No finite-size crossing drift is resolved by the full-history statistical calculation. The three pair estimates share histories and are correlated. Without a justified correction exponent and more sizes, averaging them as independent estimates or extrapolating them to infinite size would manufacture precision. The summary therefore uses the largest pair as a labelled proxy.

**Stationarity is a stronger limitation than these pairwise differences.** For g_D = 0.01, the (128,160) crossing is 0.385726 ± 0.000045 in the first production half and 0.385447 ± 0.000040 in the second. Their difference is 0.000279, about 4.6 errors if the half estimates were independent. The (96,160) crossing also shifts by 0.000103. The two halves are not independent equilibrium runs, so this is a diagnostic rather than a calibrated hypothesis test. It nevertheless prevents interpreting ±0.000039 as equilibrium precision. The half-span ≈0.00014 is a useful warning scale, not an additional Gaussian error bar. At zero SOC the largest-pair half estimates, 0.381826 ± 0.000049 and 0.381846 ± 0.000038, are consistent.

Changing block length from 100k to 250k to 500k sweeps gives largest-pair errors 0.000030, 0.000033, 0.000024 at g_D=0 and 0.000039, 0.000048, 0.000055 at g_D=0.01. Bootstrap resampling cannot repair nonstationarity.

![Binder ordering crossings and negative dips](results/binder_crossings_and_dips.png)

## Minimum negative Binder cumulant

These are minima over the entire **saved temperature grid**, not continuous-temperature minima. The displayed errors are pointwise bootstrap SE at the observed minimizing temperature.

| g_D | L | Temperature of grid minimum | Minimum U ± pointwise SE |
| --- | --- | --- | --- |
| 0 | 96 | 0.384268 | −0.19404 ± 0.00817 |
| 0 | 128 | 0.383958 | −0.15678 ± 0.00947 |
| 0 | 160 | 0.383185 | −0.14994 ± 0.00926 |
| 0.01 | 96 | 0.389710 | −0.16093 ± 0.00790 |
| 0.01 | 128 | 0.391081 | −0.09310 ± 0.00719 |
| 0.01 | 160 | 0.391636 | −0.07746 ± 0.00884 |

Reselecting the minimum in each joint bootstrap gives changes U_min(160) − U_min(96) of **+0.04410 ± 0.01112** and **+0.08347 ± 0.00945**, respectively: the overall dip becomes clearly shallower. The 128→160 changes are +0.00684 ± 0.01225 and +0.01564 ± 0.00855, so evidence for the final step is absent at zero SOC and weak at nonzero SOC. Strict monotonicity across all three sizes occurs in 69.7% and 97.5% of resamples; these are bootstrap descriptions, not posterior probabilities.

Searching for a minimum produces selection bias. Bootstrap mean reselected minima are approximately 0.006–0.010 more negative than the original minima. [main_dip.csv](results/main_dip.csv) includes reselected-minimum distributions and minimizing-temperature ranges, but these should not be mistaken for guaranteed simultaneous confidence intervals for the underlying continuous minimum. These sizes support a diminishing dip overall; they do not establish its thermodynamic disappearance or exclude a weak first-order transition.

![Grid minimum versus size](results/dip_size_trend.png)

## Helicity normalization and the two charge hypotheses

Using the saved total currents I and curvatures K, the existing per-site modulus averaged over spatial directions is

\[
Y=\frac{\langle K_x\rangle+\langle K_y\rangle-[\operatorname{Var}(I_x)+\operatorname{Var}(I_y)]/T}{2L^2}.
\]

It is not divided by density. For winding charge q in units of a 2π phase winding, the tested universal jump is Y_c = 2T_c/(πq²): **2T/π for q=1 and 8T/π for q=0.5**. This q is not the charge of the composite order parameter. The conditional WM form is

\[
Y(L,T_c)=\frac{2T_c}{\pi q^2}\left[1+\frac{1}{2\ln L+C}\right].
\]

The existing `utils/Analysis/bkt.py::fit_weber_minnhagen` is reused with q²Y and q²SE, followed by bounded refinement and additional starts to avoid raw-grid interpolation-knot minima. Finite corrections are constrained positive. Internal-spin stiffness and coupled defects can invalidate this scalar ansatz even when a numerical crossing exists.

| g_D | q | WM optimum ± conditional statistical SE | χ² / dof | Approximate p | Relative correction, L96 → L160 | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5 | 0.381502 ± 0.000019 | 0.186 / 1 | 0.666 | 21.5% → 17.6% | Acceptable conditional fit |
| 0 | 1 | 0.383412 ± 0.000011 | 147.09 / 1 | 7.5×10⁻³⁴ | 129% → 55.7% | Rejected; optimizer precision is not a T_c uncertainty |
| 0.01 | 0.5 | 0.385245 ± 0.000023 | 0.039 / 1 | 0.843 | 12.1% → 10.8% | Acceptable conditional fit |
| 0.01 | 1 | 0.389497 ± 0.000033 | 4.154 / 1 | 0.0415 | 66.9% → 39.7% | Marginal/rejected at nominal 5%; not robust |

The p values use estimated errors and an asymptotic χ² approximation, with only one degree of freedom. Tiny residuals for q=0.5 are not strong model-selection evidence by themselves. Its bootstrap 95% intervals are [0.381464,0.381539] and [0.385200,0.385290]. No selected main bootstrap optimum hits the fitting boundary.

For g_D=0.01, raw-grid linear interpolation instead gives q=1 T≈0.389398 with χ²≈0.004. Such a change in apparent goodness of fit, together with large corrections and the reweighted marginal fit, makes a precise integer-vortex T_c unjustified. At g_D=0, the raw-grid integer fit still fails (χ²≈21.05). For q=0.5 the raw-linear estimates are 0.381519 and 0.385320; the latter shift of 0.000075 exceeds the primary statistical error and is an important interpolation/reweighting systematic.

![Helicity data, jumps, direct crossings, and WM predictions](results/helicity_crossings_and_WM.png)

![WM profile diagnostics](results/WM_profiles.png)

## Direct universal-jump crossings are finite-size results

All six central curves have one direct crossing for each charge hypothesis over the measured range, and all main resamples retain a matched crossing. These temperatures need not equal WM estimates: WM includes the finite-size correction above the thermodynamic jump.

| g_D | L | Direct Y=2T/π crossing, q=1 | Direct Y=8T/π crossing, q=0.5 |
| --- | --- | --- | --- |
| 0 | 96 | 0.385882 ± 0.000080 | 0.382068 ± 0.000015 |
| 0 | 128 | 0.384556 ± 0.000075 | 0.381952 ± 0.000009 |
| 0 | 160 | 0.383929 ± 0.000027 | 0.381875 ± 0.000007 |
| 0.01 | 96 | 0.391242 ± 0.000078 | 0.385635 ± 0.000018 |
| 0.01 | 128 | 0.390634 ± 0.000046 | 0.385536 ± 0.000014 |
| 0.01 | 160 | 0.390208 ± 0.000067 | 0.385510 ± 0.000013 |

These errors exclude interpolation bias. The strong systematic size drift, especially for q=1, prohibits averaging the direct crossings into a thermodynamic transition estimate. A direct crossing exists even where WM fails.

## Are chirality and helicity transitions separated?

Using joint resamples to retain their covariance, the largest-pair Binder proxy minus the q=0.5 WM temperature is:

| g_D | Difference ± statistical SE | Bootstrap 95% interval |
| --- | --- | --- |
| 0 | +0.000335 ± 0.000033 | [+0.000271,+0.000405] |
| 0.01 | +0.000339 ± 0.000042 | [+0.000254,+0.000419] |

Thus the **chosen finite-size estimators are statistically separated conditional on the histories and WM model**. This does not establish two thermodynamic transitions: Binder finite-size corrections are unextrapolated, the WM charge hypothesis is conditional, interpolation changes the nonzero-SOC estimate appreciably, and its Binder history drifts by almost the inferred separation.

For comparison, Binder minus the L160 *direct* q=0.5 crossing is −0.000039 ± 0.000027 at zero SOC and +0.000074 ± 0.000036 at g_D=0.01. The first is compatible with coincidence; the second is only marginal before the stationarity limitation. No reliable separation claim can be based on the rejected/marginal q=1 WM fits.

## Validation, limitations, and reproduction

Synthetic tests pass for the Binder formula, helicity variance subtraction, batched bootstrap axes, and known WM curves for both q values. Coarsening 800 energy bins to 400 shifts crossing/WM temperatures by less than 2.4×10⁻⁸. Trimming the WHAM source windows shifts Binder crossings by at most 2.8×10⁻⁶, q=0.5 WM estimates by at most 1.0×10⁻⁵, and q=1 optima by at most 2.5×10⁻⁵. Solver residuals satisfy the numerical tolerance. Leaving out individual central source temperatures changes predicted observables by up to approximately 2.7 original-grid standard errors; these are correlated comparisons, not independent z tests. Numerical convergence is therefore much better established than equilibrium precision or model validity.

Raw-linear and PCHIP Binder roots differ from WHAM by order 10⁻⁵. Their agreement does not remove the g_D=0.01 time drift. Longer blocks do not monotonically enlarge errors because there are only six 500k blocks. No stationary-bootstrap method can infer missing equilibration from one drifting history. Independent equilibrium histories and further size information would be needed for a defensible thermodynamic extrapolation; none were generated here.

Reproduce from the repository root, using the existing scientific Python environment:

```powershell
python -B exploration/he3_transition_L96_160/prepare.py
python -B exploration/he3_transition_L96_160/analyze.py main
python -B exploration/he3_transition_L96_160/analyze.py 250k
python -B exploration/he3_transition_L96_160/analyze.py 500k
python -B exploration/he3_transition_L96_160/analyze.py first
python -B exploration/he3_transition_L96_160/analyze.py second
python -B exploration/he3_transition_L96_160/validate.py
python -B exploration/he3_transition_L96_160/figures.py
```

Numerical results, bootstrap arrays, and all figures are in [results/](results/). Particularly useful tables are [Binder crossings](results/main_binder.csv), [WM diagnostics](results/main_wm.csv), [direct crossings](results/main_direct_jump.csv), [paired contrasts](results/main_contrasts.csv), and [numerical sensitivities](results/systematics.csv). The scripts read the existing analysis implementations and prior provenance metadata but write only within this dedicated directory. The simulation engine, production configuration, and earlier exploratory work are preserved.
