# What L=256 adds at zero SOC

Analysis dated 1 November 2026. This extends the [previous transition analysis](../he3_transition_L96_160/REPORT.md) using **only L=96,128,160,256 at g_D=0**. L=64 contributes to no crossing, fit, plotted curve, or estimate. The new source is `data/weak_soc_spatial_simple_01/P=20_gd=0_gaussian/L=256.npz`.

The new size strengthens the evidence for a diminishing negative Binder dip and leaves the chirality ordering temperature near **0.38182**. It also exposes tension with the previously successful leading-order half-charge Weber–Minnhagen fit. Consequently, the earlier precise helicity estimate and the transition separation inferred from it are less secure after including L=256.

| Quantity | Updated result | Interpretation |
| --- | --- | --- |
| Chirality, Binder crossing (160,256) | **0.381818 ± 0.000021** | Finite-size transition proxy; ±0.000028 with 250k blocks |
| Direct q=0.5 jump crossing, L256 | **0.381809 ± 0.000008** | Finite-size crossing, statistically compatible with the Binder proxy |
| Direct q=1 jump crossing, L256 | **0.383099 ± 0.000044** | Finite-size crossing; does not validate integer-vortex scaling |
| q=0.5 WM, four sizes | Formal optimum 0.381548 ± 0.000013 | Poor primary fit: χ²/dof=10.07/2, p≈0.0065; no robust thermodynamic T_c |
| q=1 WM, four sizes | No acceptable transition estimate | χ²/dof≈1212.64/2; strongly inconsistent |

All quoted ± values are **conditional one-standard-error statistical uncertainties**. They exclude unmeasured equilibration bias, finite-size corrections, and model uncertainty. A fitted temperature with poor goodness of fit is a diagnostic optimizer location, not a validated physical transition estimate.

## Data, methods, and equilibrium limitations

The new archive has the same model parameters and 200-temperature ladder as the smaller runs: T=0.37–0.42, 3 million burn-in sweeps followed by 3 million production sweeps, Gaussian local sampling, and PT exchange interval 5. There are 60,000 scalar measurements, 3,000 spatial measurements, and 300 real-space profile measurements. Initialization was ordered positive chirality with d along z. Model and temperature-grid equality were checked explicitly.

The new analysis reuses the earlier Binder/helicity estimators, WHAM solver, WM fitting function, and validated spatial/PT reduction. The previous three-size bootstrap arrays are reused unchanged. L256 is resampled independently, preserving all its temperature slots and channels within each time block. The primary bootstrap uses 30 blocks of 100,000 sweeps and 600 replicates; checks use 250k and 500k blocks, plus separate production halves, with 200 replicates each. Every new-size WHAM curve and every extended fit is recalculated within the bootstrap. Dense interpolated temperatures are never treated as independent observations.

Transport is a material limitation:

| Diagnostic | L160, previous data | L256 |
| --- | --- | --- |
| Minimum recorded swap acceptance | 66.1% | **43.9%**, at T midpoint 0.381874 |
| Mean exact completed round-trip duration, including burn-in | 1.42 million sweeps | **3.18 million sweeps** |
| Walkers with a production sampled round trip | 171/200 | **41/200** |
| Walkers observed at both endpoints during production | 184/200 | **97/200** |

L256 has 123 exact completed loops over the whole six-million-sweep run and 43 loops identifiable from sampled production trajectories. Its median sampled production loop count per walker is zero. Completed-trip durations are censored by the finite observation window; the shorter production-only sampled mean cannot be interpreted as better transport. Good neighboring acceptance does not imply that the ladder mixes globally.

Near T=0.381796, descriptive scalar autocorrelation times are approximately 3,066 sweeps for energy and 4,102 sweeps for m². A 100k block is much longer than these observed local times, but these estimates cannot certify equilibration of slow walker or topological sectors. The replica-flow curve retains a sharp bottleneck near the ordering region.

![PT acceptance, flow, and time-block histories](results/PT_and_stationarity.png)

## Chirality crossings

As before, m is the configurationwise ratio of global chirality to global density, and U=1−〈m⁴〉/(3〈m²〉²). Ordering-region WHAM crossings are:

| Size pair | Crossing ± statistical SE | Bootstrap 95% percentile interval |
| --- | --- | --- |
| 96,128 | 0.381849 ± 0.000022 | [0.381808,0.381892] |
| 128,160 | 0.381837 ± 0.000030 | [0.381778,0.381899] |
| 96,160 | 0.381843 ± 0.000015 | [0.381814,0.381874] |
| 96,256 | **0.381827 ± 0.000013** | [0.381801,0.381850] |
| 128,256 | **0.381822 ± 0.000015** | [0.381789,0.381850] |
| 160,256 | **0.381818 ± 0.000021** | [0.381774,0.381858] |

All selected main roots are unique in the previous ordering search window, 0.38118–0.38248, and survive all 600 bootstrap realizations. Additional raw-curve crossings outside it are saved in [all_raw_crossings_with_L256.csv](results/all_raw_crossings_with_L256.csv); negative-dip and hot-tail crossings are not merged into the ordering estimate.

The largest-pair proxy moves downward by **0.000019 ± 0.000042** relative to the old (128,160) proxy, using the correlated bootstrap difference. This is unresolved finite-size drift. Pairwise crossings share histories; they must not be averaged as independent measurements. Four sizes still do not establish a correction exponent or justify a precise thermodynamic extrapolation.

For (160,256), the statistical SE is 0.000021, 0.000028, and 0.000024 with 100k, 250k, and 500k blocks. The two production halves give 0.381845 ± 0.000028 and 0.381789 ± 0.000029. Their shift, 0.000056, is approximately 1.4 combined errors if the halves were independent. This is not strong evidence of drift, but the halves are not independent equilibrium runs and transport remains limited. A readable working estimate is therefore **T_chi≈0.38182, with conditional statistical precision of a few ×10⁻⁵ and unquantified finite-size/equilibration uncertainty**.

## The negative dip weakens substantially

These are minima over the full saved temperature grid, with pointwise errors at the observed minimizing temperature:

| L | Temperature of grid minimum | Minimum U ± pointwise SE |
| --- | --- | --- |
| 96 | 0.384268 | −0.19404 ± 0.00817 |
| 128 | 0.383958 | −0.15678 ± 0.00947 |
| 160 | 0.383185 | −0.14994 ± 0.00926 |
| 256 | **0.382722** | **−0.08282 ± 0.01679** |

Reselecting each minimum in every bootstrap gives U_min(256)−U_min(160)=**+0.06712 ± 0.01493**, with percentile range [+0.03706,+0.09567]. Relative to L96 the change is +0.11122 ± 0.01309. L256 provides substantially clearer evidence of a shallowing trend than the previously unresolved 128→160 step.

Minimum selection introduces bias: the L256 bootstrap mean reselected minimum is −0.09060, below the observed −0.08282. Its reselected-minimum SD is 0.01132, which differs from the pointwise error above. Neither error includes uncertainty about the continuous-temperature minimum between recorded points. The dip also shifts toward the ordering region as size grows. These observations support a finite-size explanation for much of the negative excursion, but do not prove its infinite-size disappearance or rule out a weak first-order transition.

![Binder crossings, dip, helicity curves, and direct jump crossings](results/transitions.png)

## Helicity: the simple half-charge fit loses its previous precision

The normalization is unchanged:

\[
Y=\frac{\langle K_x\rangle+\langle K_y\rangle-[\mathrm{Var}(I_x)+\mathrm{Var}(I_y)]/T}{2L^2},\qquad
Y_c=\frac{2T_c}{\pi q^2}.
\]

Thus the direct criteria are **Y=2T/π for q=1** and **Y=8T/π for q=0.5**. Y is the per-site, direction-averaged modulus without density normalization. The tested leading WM form is

\[
Y(L,T_c)=\frac{2T_c}{\pi q^2}\left[1+\frac{1}{2\ln L+C}\right].
\]

Fits use one modulus value per size at the trial temperature, not independent values at every interpolation point. The existing fitter is reused with q²Y and q²SE and positive finite-size corrections.

| q | Sizes | Formal optimum ± statistical SE | χ²/dof | Approximate p | Assessment |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 96,128,160, previous result | 0.381502 ± 0.000019 | 0.186/1 | 0.666 | Previously acceptable |
| 0.5 | **96,128,160,256** | **0.381548 ± 0.000013** | **10.07/2** | **0.0065** | Primary fit is poor |
| 0.5 | 128,160,256 | 0.381580 ± 0.000019 | 4.96/1 | 0.0259 | Marginal and sensitivity-dependent |
| 1 | 96,128,160,256 | 0.383047 ± 0.000014 | 1212.64/2 | ≈4.8×10⁻²⁶⁴ | Strongly rejected |
| 1 | 128,160,256 | Boundary optimum; no valid T_c error | 192.09/1 | ≈1.1×10⁻⁴³ | Rejected within the local window |

The q=1 largest-three-size fit hits the lower boundary of its inherited local reweighting window, T≈0.382876; its full-grid direct-linear fit also fails, χ²≈44.36/1. A boundary optimizer is not reported as a transition estimate. There is no reason to expand an expensive fit search merely to extract precision from this inconsistent model.

For the four-size half-charge fit, corrections are about 19.9% at L96 and 14.3% at L256. Its poor primary fit persists with 250k and 500k blocks: χ²≈9.31/2 and 8.73/2, respectively. Removing the outer source temperatures gives χ²≈8.80/2. A null-centered joint bootstrap, retaining the sampled temperature dependence and refitting both parameters, gives an empirical tail probability ≈0.0067, consistent with the approximate χ² result.

There are nevertheless important sensitivities. The four-size raw-grid linear fit gives T≈0.381554 and χ²≈1.65/2; it is acceptable with its larger direct-grid errors. Keeping WHAM means but substituting the direct-grid error estimates gives χ²≈2.25/2. Therefore much of the stronger WHAM diagnostic comes from the information gain and smaller errors provided by reweighting, rather than a large temperature shift. The main fit also has χ²≈10.91/2 in the first production half versus 2.26/2 in the second. The largest-three-size half-charge fit changes from nominal p≈0.026 with 100k blocks to p≈0.056–0.067 with longer blocks. These facts prohibit treating rejection of the leading form as estimator-independent or as proof against HQVs.

The new size also supplies an independent prediction test of the **old** three-size fit. At its fitted T=0.381502, the predicted L256 modulus is 1.11841, whereas the measured/reweighted value is 1.13386. Their joint-bootstrap difference is **+0.01546 ± 0.00471**, with 95% range [+0.00633,+0.02436]. This tension remains about three conditional errors with the longer blocks. The earlier fit therefore does not predict the new size within its claimed statistical precision.

![Old-fit prediction and extended WM size dependence](results/WM_size_tests.png)

The defensible conclusion is that **no robust thermodynamic helicity T_c can currently be assigned from this leading WM form**. The half-charge optima around 0.38155–0.38158 are useful model-dependent candidate scales, not a calibrated confidence interval. Larger corrections, imperfect equilibration, and the multicomponent character of the field remain possible explanations. The much worse integer fit does not identify the actual defect population.

## Direct crossings and the proposed transition separation

Direct crossings use linear interpolation of the original grid with the same block bootstrap:

| L | q=0.5: Y=8T/π | q=1: Y=2T/π |
| --- | --- | --- |
| 96 | 0.382068 ± 0.000015 | 0.385882 ± 0.000080 |
| 128 | 0.381952 ± 0.000009 | 0.384556 ± 0.000075 |
| 160 | 0.381875 ± 0.000007 | 0.383929 ± 0.000027 |
| 256 | **0.381809 ± 0.000008** | **0.383099 ± 0.000044** |

All central curves have one direct crossing per charge hypothesis and all main resamples retain a matched crossing. These are finite-size crossings, with interpolation bias outside their statistical bars.

The largest Binder crossing minus the L256 direct half-charge crossing is **+0.000009 ± 0.000018**, with 95% interval [−0.000027,+0.000045]: consistent with coincidence. In contrast, subtracting the formal four-size half-charge WM optimum gives +0.000270 ± 0.000023. This latter difference is statistically nonzero **conditional on a fit that is poor**. It cannot establish thermodynamic separation.

L256 consequently weakens the earlier interpretation of a well-resolved chiral/helicity split. It does not prove simultaneous transitions either: direct finite-size jump temperatures and Binder crossings can coincide before their thermodynamic limits do. The remaining physical question is how the large finite-size corrections behave, rather than which tiny formal fitting error should be preferred.

## Spatial and energy observables provide context

The validated spatial reduction was applied to L256 without changing definitions. At T=0.381796, full-correlation second-moment ratios are:

| L | ξ_kappa/L | ξ_P/L | ξ_Q/L |
| --- | --- | --- | --- |
| 96 | 1.080 ± 0.015 | 0.614 ± 0.005 | 0.580 ± 0.003 |
| 128 | 1.059 ± 0.015 | 0.599 ± 0.004 | 0.565 ± 0.003 |
| 160 | 1.042 ± 0.019 | 0.591 ± 0.005 | 0.557 ± 0.003 |
| 256 | **1.056 ± 0.033** | **0.582 ± 0.009** | **0.538 ± 0.004** |

The chirality ratios remain consistent with a common ordering-region intersection. P shows increasing ξ_P/L with size at the lower probe T=0.381486 (0.679 at L96 versus 0.732 at L256), and decreasing ratios at T=0.381796 and above. This preserves evidence for a crossover or transition in the composite-order sector in this narrow region, but these curves alone do not supply an infinite-size T_c. Q decreases with size at these temperatures, with no comparable common crossing established. Full correlations are used in this comparison; finite-history mean subtraction can be sensitive to slow spin rotations at zero SOC.

The density-weighted departure from the ideal local chiral manifold remains appreciable: epsilon≈0.3530±0.0004 at T=0.381796, comprising orbital loss≈0.2839 and spin loss≈0.0692. L256 therefore does not approach an everywhere ideal restricted chiral field near the transition simply by increasing system size. This supports retaining the full-field interpretation of stiffness and defects.

![Spatial correlation ratios and chiral-manifold departure](results/spatial_correlations.png)

The existing decomposition Y=D−S also remains relevant. At T=0.381486, (D,S,Y)≈(2.4605,1.3209,1.1396); at T=0.382568 it is approximately (2.2544,1.7930,0.4614). Over this interval about 30% of the modulus drop comes from decreasing curvature D and 70% from increasing current-fluctuation subtraction S. This is a descriptive decomposition, not a causal separation into vortex and non-vortex contributions. The large energy-fluctuation peak sharpens with size and lies in the same narrow region. These measurements continue to caution against interpreting the entire drop as a simple universal-jump event in a rigid-amplitude phase-only model.

![Helicity contributions and energy fluctuations](results/helicity_components.png)

## Validation and saved outputs

Spatial/current synchronization, walker maps, zero-mode identities, energy-component identities, profile normalization checks, and finite-value checks pass using the existing reduction. Critical-region adjacent energy-histogram overlap at L256 has a minimum ≈0.573; this is a measured histogram overlap, distinct from swap acceptance. Holding out individual central source temperatures predicts the direct L256 Binder and modulus within 0.72 and 0.56 direct-grid standard errors, respectively. These comparisons are correlated diagnostics, not independent significance tests.

Changing 800 bins to 400 shifts the new Binder roots by at most about 1.1×10⁻⁸ and the tested half-charge WM optima by less than 9×10⁻⁸. Trimming WHAM sources shifts the new Binder roots by less than 4×10⁻⁷ and the four-size half-charge optimum by 6.4×10⁻⁶. Synthetic normalization and WM recovery tests pass for both charges and both selected size sets. Numerical controls do not remove the physical finite-size or equilibration limitations.

All new work is contained in this directory. Reproduce from the repository root:

```powershell
python -B exploration/he3_transition_L256/prepare.py
python -B exploration/he3_transition_L256/analyze.py main
python -B exploration/he3_transition_L256/analyze.py 250k
python -B exploration/he3_transition_L256/analyze.py 500k
python -B exploration/he3_transition_L256/analyze.py first
python -B exploration/he3_transition_L256/analyze.py second
python -B exploration/he3_transition_L256/spatial_checks.py
python -B exploration/he3_transition_L256/validate.py
python -B exploration/he3_transition_L256/figures.py
```

The earlier `he3_transition_L96_160/results` and `he3_weak_soc_spatial_01/results` caches are explicit read-only dependencies. Useful tables include [Binder crossings](results/main_binder.csv), [WM fits](results/main_wm.csv), [direct jump crossings](results/main_direct_jump.csv), [dip minima](results/main_dip.csv), [old-fit prediction test](results/main_holdout.csv), [paired differences](results/main_contrasts.csv), [numerical controls](results/numerical_controls.csv), and [PT diagnostics](results/pt_summary.json). The new spatial summary retains the existing observable channels for later inspection. No simulation, Slurm submission, production edit, or change to earlier exploratory code was made.
