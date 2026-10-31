# Relationship between the U(1) and Z2 transitions in the full thin-film He3 model

**Outcome:** local multiple-histogram reweighting materially improves the analysis of the existing campaign. It changes some finite-size estimates substantially and exposes limitations of a simple BKT interpretation. **The available sizes and observables still do not determine whether the thermodynamic transitions coincide or enclose a narrow intermediate phase.** A targeted simulation with spatial correlations, independently initialized histories and larger size ratios is justified.

All calculations in this directory analyze existing records. Production code, the Hamiltonian, original datasets and earlier exploratory directories were left unchanged. No new Monte Carlo or GPU campaign was launched. The proposed settings in [simulation_proposal.json](results/simulation_proposal.json) are a design document, not a job submission.

## 1. Established observations, conditional results and unresolved questions

### What the existing histories now establish

1. **Reweighting works locally and removes significant temperature interpolation bias.** Across twelve datasets, 216 withheld-temperature comparisons were made using paired time-block errors. All 72 chirality/composite Binder comparisons are within 1.79 errors; one helicity comparison reaches 3.47 errors. Histogram resolution and contributing-temperature sensitivity are much smaller than history errors near the reported Binder roots.
2. **The two Binder crossings track each other exceptionally closely.** At zero SOC, the L=128,160 roots become 0.3818980 and 0.3818978. Their difference is only 0.0000003, with a paired conditional error of 0.0000070. This is a finite-size statement. The global magnitudes are strongly correlated, and their Binder agreement is not two independent determinations of a thermodynamic critical temperature.
3. **The previous zero-SOC crossing near 0.381655 was appreciably biased by straight-line interpolation.** Reweighting shifts the chirality crossing by +0.0002426. The old grid spacing near this feature is approximately 0.00064. The apparent precision of a root between two straight segments did not measure the interpolation error.
4. **Finite-size estimator differences survive the improved temperature analysis.** The zero-SOC q=1/2 WM diagnostic is 0.3815189(141), whereas the largest chirality Binder root is 0.3818980(755). Their paired difference is 0.0003791(784). These are conditional history errors, not uncertainty on a thermodynamic transition separation.
5. **There is no resolved energy coexistence signal at the tested sizes.** Reweighted energy distributions at the zero-SOC heat-capacity maxima have one prominent mode under all three stated smoothing choices. Peak growth is much slower than asymptotic L² coexistence scaling. Very weak first order remains possible below the accessible crossover scale.

### What remains conditional

The q=1/2 leading WM form fits the helicity size dependence much better than q=1 in the tested region. The zero-SOC q=1/2 fit nevertheless has chi-square/dof = 3.45, and its temperature changes by 0.000063 when L=64 is excluded. A good fit at weak SOC does not validate a mass-only universal jump, identify the vortices, or establish the asymptotic effective theory.

An additional consistency check challenges the simplest interpretation: at the zero-SOC WM temperature, charge-two size exponents are approximately 0.33–0.43, whereas a Gaussian phase-only relation using the measured helicity predicts approximately 0.21. Finite-size amplitude/internal-sector effects, crossover physics and sampling remain possible explanations. This is a reason to measure spatial correlations before assigning a universal jump.

### Which scenarios remain compatible?

| Scenario | Assessment from this investigation | Evidence needed to distinguish it |
|---|---|---|
| One coincident continuous/coupled transition | Compatible | Both sectors must approach one limiting temperature with a stable account of corrections and spatial scaling |
| Two extremely close transitions | Compatible | Two stable large-L limits, with an interval whose spatial correlations identify different ordering in the two sectors |
| A finite intermediate phase | No intermediate regime has been demonstrated; a narrow one remains compatible | Show an interval that persists as L grows, with one sector ordered or algebraic and the other short ranged |
| Weak first order | Unresolved; current energy histograms and peak growth supply no positive coexistence evidence | Repeated equilibrium switching, a stable nonzero energy-density separation and an interface barrier growing with L |
| The simple asymptotic HQV/BKT description already applies at L=64–160 | Not established; the composite/helicity consistency check raises a concrete concern | Spatial charge-two correlations, the spin crossover scale and calibrated thermal defect evidence |

No defensible numerical upper bound on the thermodynamic splitting follows from the paired Binder error alone. An asymptotic gap cannot be bounded without controlling the finite-size shifts and identifying the U(1) critical behavior.

## 2. Model, observable definitions and provenance

The source review covers the [Hamiltonian](../../models/he3/update.py), [measurements](../../models/he3/measure.py), [original analysis](../../models/he3/analysis.py), the original WM helper, the comprehensive [SOC report](../he3_soc/REPORT.md), its [model audit](../he3_soc/model_audit.md), and the three [HQV](../he3_hqv_stage1/REPORT.md) [stage](../he3_hqv_stage2/REPORT.md) [reports](../he3_hqv_stage3/REPORT.md). The attached thesis was consulted, especially its full-model results, discussion and helicity-estimator appendix. Its earlier L<=96 calculations and reduced-model comparisons provide context; their temperatures are not additional samples of this campaign.

The thesis also gives a useful methodological warning: fixing amplitudes, freezing spin, and allowing the full matrix can change the relative ordering inferred for the two transitions. Its full-model discussion leaves a narrow intermediate phase unresolved, with estimator ordering changing between pressures. A transition sequence from a restricted London or spin-frozen theory therefore cannot be imposed on the present full-field data. The present analysis follows that warning by testing both sectors and the effective-theory assumptions independently.

The field A is an unrestricted complex 3-by-2 matrix, with twelve real variables per site. The implemented dimensionless energy is

\[
H=\sum_r\left[-2n+\sum_{i=1}^5\beta_i I_i+g_D D
       +K\sum_\mu G_\mu\right],\qquad n=\sum_{\mu,a}|A_{\mu a}|^2.
\]

For a=Delta_x A_x, b=Delta_x A_y, c=Delta_y A_x, d=Delta_y A_y,

\[
G=3(|a|^2+|d|^2)+|b|^2+|c|^2+2\operatorname{Re}(a^*d+b^*c).
\]

The full quartic polynomial and SOC invariant are retained in the sampled data. The gD=0 theory has exact common U(1), independent spin rotations and time-reversal/chirality symmetry. Continuum orbital/spatial rotations need not be exact for the implemented forward-difference lattice. Positive SOC explicitly breaks the independent spin symmetry. No projection onto the chiral manifold is made by this analysis.

The [inventory](results/inventory.json), regenerated from actual NPZ manifests, verifies the following for all twelve analyzed files:

| Setting | Saved value |
|---|---|
| Pressure; K; coefficient reduced_temperature | 20 bar; 1; 1 |
| SOC strengths analyzed | 0, 0.001, 0.01 |
| L | 64, 96, 128, 160 |
| Quartic coefficients, rounded in the simulation | (-0.5165123343, 0.9420228004, 0.9875290394, 0.8790628910, -1.1525149345) |
| Thermalization; production | 2,000,000; 3,000,000 sweeps |
| Measurement interval; records per temperature | 50 sweeps; 60,000 |
| PT exchange interval | **5 sweeps** |
| Temperatures | 160, spanning 0.33–0.47 |
| Local sampler | float32 conditional-Gaussian full-Hamiltonian MH, rho approximately -0.9, proposal mass zero |
| Local seeds | 106, 138, 170, 202 for increasing L; reused across SOC strengths |

The temperature changes exp(-H/T) with fixed coefficients. It is distinct from the phenomenological coefficient variable `reduced_temperature`. The saved energy is **the full extensive H**, including gradients and SOC. Reweighting different gD values together with exp[-Delta(beta) H] would be invalid; each coupling was treated separately. Actual float32 inverse temperatures in the manifests were used for source ensembles. The associated nominal-temperature rounding is about 10^-8 here, far smaller than the identified interpolation effects.

Let N=L² and define the following per configuration:

\[
D_A=\sum_{r,\mu,a}|A_{\mu a}|^2,\quad
C=2\sum_{r,\mu}\operatorname{Im}(A_{\mu x}^*A_{\mu y}),
\]
\[
m=C/D_A,\quad c=C/N,\quad
P=\sum_{r,\mu}(A_{\mu x}^2-A_{\mu y}^2),\quad p=|P|/N.
\]

The original chirality Binder uses m, a configuration-by-configuration ratio. The alternative raw-density Binder uses c. They have the same odd chirality symmetry, but different amplitude-related corrections. The complex P is spin-rotation invariant and has gauge charge two. The observables reconstructed here are

\[
U_m=1-\frac{\langle m^4\rangle}{3\langle m^2\rangle^2},\quad
U_c=1-\frac{\langle c^4\rangle}{3\langle c^2\rangle^2},\quad
U_P=2-\frac{\langle p^4\rangle}{\langle p^2\rangle^2},
\]
\[
C_V=\frac{\operatorname{Var}(H)}{NT^2},\qquad
\Upsilon_\nu=\frac{\langle K_\nu\rangle-
 (\langle I_\nu^2\rangle-\langle I_\nu\rangle^2)/T}{N},\quad
\Upsilon=(\Upsilon_x+\Upsilon_y)/2.
\]

I and K are the exact saved common-phase twist derivatives. Their variances are formed after reweighting their individual moments. The reported chirality response N(\<m²\>-\<|m|\>²)/T and composite response N(\<p²\>-\<p\>²)/T are **magnitude-fluctuation responses**. The former is not the signed linear susceptibility to a chiral field. This distinction matters when comparing critical scaling.

## 3. Reweighting construction and validation

### Estimating equations

For a single source beta_i, the exact importance-weight identity is

\[
\langle O\rangle_\beta=
\frac{\langle O\exp[-(\beta-\beta_i)H]\rangle_i}
     {\langle\exp[-(\beta-\beta_i)H]\rangle_i}.
\]

Its usefulness is controlled by sampled overlap, rather than the algebraic identity. Multiple-histogram analysis combines overlapping neighboring ensembles. These methods originate with [Ferrenberg and Swendsen's single-histogram work](https://doi.org/10.1103/PhysRevLett.61.2635) and [multiple-histogram analysis](https://doi.org/10.1103/PhysRevLett.63.1195). Here the following estimating equations were implemented independently in [reweight.py](reweight.py), with shifted energies and log-sum-exp arithmetic.

For source counts n_i, bin counts h_ib, h_b=sum_i h_ib, and f_i=-log Z_i up to an arbitrary constant,

\[
d_b=\sum_i n_i e^{f_i-\beta_i E_b},\qquad
g_b=\frac{h_b}{d_b},\qquad
e^{-f_i}=\sum_b g_b e^{-\beta_i E_b}.
\]

An individual observation in bin b has normalized target weight proportional to exp(-beta E_b)/d_b. If S_Ob is the sum of **raw O values** in the bin, its target mean is

\[
\widehat{\langle O\rangle}_\beta=
\frac{\sum_b S_{Ob}e^{-\beta E_b}/d_b}
     {\sum_b h_be^{-\beta E_b}/d_b}.
\]

Every moment entering a Binder, response or helicity is handled this way. A previously averaged Binder or helicity is never assigned a fictitious configuration weight. Centering H before forming its variance avoids cancellation between large extensive numbers. The energy derivative identity d\<O\>/dT=Cov(O,H)/T² explains why straight lines can be inaccurate through a narrow peak.

Each dataset uses fifteen neighboring temperatures, centered at the main feature, and 800 equal-width energy bins. Preparation retains 1,600 bins for resolution checks. The zero/0.001 source interval is 0.377781–0.386742; the 0.01 interval is 0.380935–0.390044. The published smooth curves are restricted to the interior thirteen-temperature interval. Seven/eleven-source variants are interpreted only within their own contributing-temperature intervals. Source windows of seven and eleven temperatures and 400/1,600 bins were checked. Boundary optima are rejected, including the local gD=0.01 q=1 WM attempt.

An unbinned MBAR calculation on four raw neighboring ensembles, with a fifth withheld, supplies an independent reference at gD=0,L=160. MBAR avoids energy-bin discretization; see [Shirts and Chodera](https://arxiv.org/abs/0801.1426). The reference uses 240,000 raw records and independently evaluates the weights. Its largest 800-bin differences across three target temperatures are 0.000045 for U_m, 0.000097 for U_P, 0.000055 for helicity and 0.0053 for Cv. These are small against the history errors. Full Binder crossing shifts under bin/source-window changes are at most 1.12e-7. This validates the small binned calculation for the present purpose; a large MBAR framework is unnecessary.

### Correlated histories and uncertainty

Main errors use a **joint delete-one-time-block jackknife**, with thirty blocks of 2,000 measurements = 100,000 sweeps. A deletion removes the same time interval from every contributing PT temperature and every moment, and solves the histogram equations again. This preserves contemporaneous PT correlations. Separate L histories contribute separate jackknife variance/covariance terms; their block indices are not paired artificially. Differences of crossing estimators use their within-run covariance.

Sensitivity analyses use 4,000/6,000-measurement blocks, with fifteen/ten deletes, and first/second halves. Binned counts are actual record counts. Replacing these counts alone by an estimated ESS would change the equations inconsistently. Correlated observations can give consistent point estimating equations under stationary ergodic sampling, but the independence-based optimality and analytic-error claims of histogram/MBAR likelihoods do not apply directly. The joint-block errors address observed correlations; they cannot measure an unvisited sector or preparation bias.

The raw Kish weight count 1/sum_n w_n² diagnoses weight concentration. **It is not an autocorrelation-corrected number of independent physical observations.** The previous SOC investigation estimates only about 440–810 effective critical energy/order records in the largest weak-SOC histories. Reweighted raw weight counts can be tens of thousands or more while those histories remain correlated.

### Withheld-temperature results

For every dataset, three central temperatures are omitted individually. All normalizers and moments are recomputed using the remaining fourteen ensembles. The held-out actual measurement and prediction share a PT history, so their difference error is obtained from paired joint block deletions.

| Diagnostic | Result |
|---|---:|
| Datasets; omitted temperatures per dataset | 12; 3 |
| Observables per test | energy, U_m, U_P, helicity, Cv, Qzz |
| Total comparisons | 216 |
| Largest absolute Binder difference / paired error | 1.787 |
| Largest value over all observables | 3.474 |
| Comparisons exceeding 3 errors | 1, helicity at gD=0.01,L=64,T=0.384787 |

These comparisons are correlated and are not a set of 216 independent significance tests. The isolated helicity discrepancy is retained in the table; it is not removed as an outlier. The tests support local predictive usefulness, with residual sampling/estimator uncertainty, rather than exact agreement or certified equilibrium.

Minimum neighboring energy overlaps sum_b min(p_ib,p_(i+1)b) in the fifteen-temperature windows are:

| gD | L=64 | 96 | 128 | 160 |
|---:|---:|---:|---:|---:|
| 0 | 0.702 | 0.534 | 0.359 | 0.212 |
| 0.001 | 0.699 | 0.512 | 0.354 | 0.237 |
| 0.01 | 0.703 | 0.535 | 0.397 | 0.267 |

The largest systems have marginal overlap across the sharpest edge, but a connected local histogram network and successful withheld predictions. A single neighboring histogram is less efficient: at zero SOC,L=160, predicting the central omitted temperature from either immediate neighbor gives raw Kish counts of only 511 or 609. The fourteen-source prediction has about 25,786 raw weight records. Its helicity prediction error is 0.0148, compared with 0.0722/0.0522 for the two single-histogram predictions. Multiple histograms are therefore useful here.

![Overlap and held-out prediction checks](results/reweight_validation.png)

Detailed tables: [withheld tests](results/withheld_validation.csv), [single-histogram comparison](results/single_histogram.csv), [unbinned reference](results/unbinned_reference.csv), [overlap](results/energy_overlap.csv), [block errors](results/block_errors.csv). Independent agreement with the earlier unbinned moment cache is better than 1.2e-10 in the retained non-energy moments. Seven numerical identity/artifact checks pass; see [checks](results/numerical_checks.json).

## 4. Improved finite-size estimates

### Binder crossings and their covariance

The table reports the selected ordered-side root for L=128,160. Root selection chooses the nearest crossing to a stated feature center (0.3817,0.3823,0.3855 for increasing gD), within 0.0015 and Binder values between 0.1 and 0.95. All thirty deletes in each contributing run retain the selected roots. This criterion avoids adopting unrelated disordered-side intersections.

| gD | U_m crossing | U_P crossing | Difference U_m minus U_P | Paired conditional error of difference |
|---:|---:|---:|---:|---:|
| 0 | 0.3818980 ± 0.0000755 | 0.3818978 ± 0.0000786 | +0.0000003 | 0.0000070 |
| 0.001 | 0.3823561 ± 0.0001100 | 0.3823558 ± 0.0001139 | +0.0000003 | 0.0000076 |
| 0.01 | 0.3854941 ± 0.0000632 | 0.3855014 ± 0.0000650 | -0.0000072 | 0.0000062 |

At zero SOC, treating the two crossing errors as independent would give 0.000109 for the difference. The correct paired value is about sixteen times smaller. This strong correlation is physically relevant: the reweighted raw-record correlation of |m| and p at T=0.381898 is 0.964. Their joint distribution is a narrow rising band. Both moment ratios respond strongly to a common fluctuation mode. Their agreement is evidence of closely coupled finite-size behavior, while offering limited independent discrimination of two limiting transitions.

Raw chirality-density Binder crossings are slightly higher: 0.3819056,0.3823639,0.3855057. Shifts of 0.000008–0.000012 illustrate observable-dependent finite-size corrections even within the chirality sector.

Zero-SOC adjacent-size U_m crossings are **nonmonotonic**:

| Pair | Reweighted crossing |
|---|---:|
| 64,96 | 0.3818889 ± 0.0000314 |
| 96,128 | 0.3817297 ± 0.0000454 |
| 128,160 | 0.3818980 ± 0.0000755 |

These pairs share runs and are mutually correlated. They do not support a stable extrapolation with free critical temperature, correction exponent and amplitudes. A four-size fit can return small numerical errors while being governed by the assumed corrections. No thermodynamic Binder extrapolation is reported.

At zero SOC, changing the largest-pair block length from 2,000 to 4,000/6,000 measurements gives chirality crossing errors 0.0000755,0.0000852,0.0000750. The first/second-half roots differ by less than 1e-6. At gD=0.001 those halves give 0.3824504 and 0.3822678, separated by 0.0001825. That difference is a useful reliability sensitivity; its proximity to the full-history error scale is not an independent equilibration test.

### Weber–Minnhagen diagnostics

The tested leading relation is

\[
q^2\Upsilon_L(T_*)=\frac{2T_*}{\pi}
\left[1+\frac{1}{2\ln L+C}\right].
\]

Temperature and C are fitted to four size values at a trial temperature, leaving **two nominal degrees of freedom**. Hundreds of temperatures along a reweighted curve do not create hundreds of independent size observations. The fitting code profiles the window to find the relevant optimum, avoiding the original helper's sparse-start failure. The relation is a hypothesis adapted from [Weber and Minnhagen](https://doi.org/10.1103/PhysRevB.37.5986), with a chosen charge and long-distance stiffness interpretation.

| gD | q=1/2 T_* | Conditional block SE | chi-square/dof | q=1 result in local window |
|---:|---:|---:|---:|---|
| 0 | 0.3815189 | 0.0000141 | 3.45 | 0.3837955; chi-square/dof 272.7 |
| 0.001 | 0.3819485 | 0.0000175 | 0.88 | 0.3844245; chi-square/dof 157.2 |
| 0.01 | 0.3852066 | 0.0000239 | 0.19 | Boundary optimum; no accepted temperature |

The boundary result at 0.01 does not invalidate the earlier wider-window q=1 calculation; this local reweighting window cannot supply an interior optimum for it. Neither its boundary temperature nor its tiny numerical variation is treated as an uncertainty estimate.

Removing L=64 moves the q=1/2 temperatures to 0.3814559,0.3819857,0.3851768. The zero-SOC shift is about 4.5 full-fit conditional SEs. The zero-SOC three-size fit has chi-square = 0.0096 with only one nominal degree of freedom. That small number illustrates how little a flexible three-size fit alone establishes.

At zero SOC, 4,000/6,000-record blocking gives WM SEs 0.0000131/0.0000141. Bin changes shift T_* by less than 1e-7, seven-source restriction by 0.0000032, and history halves by about 0.0000071 relative to each other. The remaining size-selection dependence is appreciably larger than these local numerical choices.

### How much did reweighting improve the estimates?

| Zero-SOC quantity | Previous straight-line result | Reweighted result | Interpretation |
|---|---:|---:|---|
| Largest chirality crossing | 0.3816554 | 0.3818980 | Removes a +0.0002426 interpolation correction |
| Largest composite crossing | 0.3816682 | 0.3818978 | Removes a +0.0002296 correction |
| q=1/2 WM diagnostic | about 0.3814929 | 0.3815189 | Smaller temperature shift; conditional error improves from about 0.000025 to 0.000014 |
| L=160 Cv peak height | 73.854 at a grid point | 88.938 at the continuous maximum | The grid missed about 17% of the reconstructed maximum |

The WM comparison also changes statistical weighting. Holding the reweighted error weights fixed and replacing only the central curves by direct-data straight lines gives 0.3815011, so the controlled curve-shape change is +0.0000178. At 0.001/0.01 the analogous changes are +0.0001000/+0.0001152.

The **largest-pair Binder error did not shrink**: the zero-SOC conditional chirality error is about twice the earlier straight-line bootstrap SD. Reweighting changes the local curve slopes, and hence the crossing uncertainty. Its benefit is a validated continuous estimate and removal of discretization bias, rather than an automatic reduction in every error bar.

The zero-SOC Binder-minus-WM difference is now 0.0003791 ± 0.0000784, with covariance included. Therefore the two finite-size estimators are more clearly separated than the old summary implied. **This difference cannot be interpreted as T_Z2-T_U1:** the leading WM hypothesis has unresolved systematic corrections and the Binder root has unresolved size drift. A common critical temperature with different finite-size corrections remains possible.

![Zero-SOC Binder, helicity and composite consistency diagnostics](results/zero_soc_diagnostics.png)

Tables: [crossings](results/crossings.csv), [paired differences](results/paired_gaps.csv), [WM fits](results/wm_fits.csv), [analysis sensitivities](results/temperature_sensitivity.csv).

## 5. Scaling, distributions and alternatives to a simple BKT picture

### Response maxima and first-order tests

Reweighted zero-SOC maxima are:

| L | T(Cv maximum) | Cv maximum | Chirality magnitude-response maximum | Composite magnitude-response maximum |
|---:|---:|---:|---:|---:|
| 64 | 0.3824478 | 45.907 ± 0.174 | 165.84 ± 0.65 | 93.62 ± 0.39 |
| 96 | 0.3822100 | 62.974 ± 0.433 | 315.26 ± 2.33 | 159.48 ± 1.09 |
| 128 | 0.3820467 | 78.078 ± 0.850 | 500.29 ± 4.98 | 233.17 ± 2.46 |
| 160 | 0.3819994 | 88.938 ± 1.619 | 678.56 ± 10.46 | 293.87 ± 4.81 |

The maxima of different responses occur at different temperatures; the full table retains their positions and conditional errors. A local quadratic refinement of a 601-point curve avoids claiming zero peak-temperature variance when all deletes fall in one fine-grid cell.

Endpoint effective powers between 64 and 160 are 0.72 for Cv, 1.54 for the chirality magnitude response and 1.25 for the composite magnitude response. These are descriptive slopes, not measured asymptotic exponents. Ordinary 2D Ising benchmarks include logarithmic Cv growth, susceptibility exponent 7/4 and nu=1; the observed values do not establish that regime. A weighted Cv=a+b log L fit has chi-square/dof 4.64, while a+b L² gives 65.8. Neither simple form is a satisfactory complete finite-size model at the quoted conditional errors. Adding correction parameters to four points would not provide a convincing universality test.

At a resolved first-order transition, a nonzero latent heat produces Cv maximum proportional to N, and in two dimensions an equilibrium interfacial barrier grows proportionally to L. Energy-distribution analysis is the appropriate direct test; see [Challa, Landau and Binder](https://doi.org/10.1103/PhysRevB.34.1841). Here all four zero-SOC energy distributions at their own Cv maxima have one prominent mode after Gaussian smoothing by 1,2,4 histogram bins. The stated mode criterion requires prominence at least 8% of the maximum and separation at least half the energy standard deviation. It is a reproducible descriptive check, not a confidence limit on a latent heat.

There is consequently no resolved pair of coexistence peaks from which to estimate a latent heat or an interface tension. Weak first order with a correlation length larger than these boxes remains possible. Unsampled metastability also cannot be excluded by a unimodal recorded distribution. A future coexistence claim must show repeated phase switching, preparation independence, peak-weight control and barrier growth; a negative order-parameter Binder alone is insufficient.

The very negative Binders in the gD=0.1 study are not used here. The earlier report identified rare-event moment domination and nonstationary branches in that campaign. Those observations do not establish the order of the zero/weak-SOC transition.

### An independent charge-two scaling diagnostic already available

Although spatial correlations are absent, the saved P supplies S_P(0)=N\<p²\>. If a charge-two correlation has a single algebraic regime G_P(r) proportional to r^-eta_P with stable amplitude, then \<p²\> scales approximately as L^-eta_P. Define the two-size diagnostic

\[
\eta_{P,\mathrm{eff}}(L_1,L_2;T)=
-\frac{\ln[\langle p^2\rangle_{L_2}/\langle p^2\rangle_{L_1}]}{\ln(L_2/L_1)}.
\]

For a Gaussian action F=(rho_c/2) integral (grad theta)² and P proportional to exp(2i theta), the spin-wave result is

\[
\eta_P=\frac{2T}{\pi\rho_c}.
\]

It follows from the logarithmic phase variance and the charge two of P. A mass-only q=1/2 BKT jump would imply the limiting eta_P=1/4; a q=1 transition would instead give eta_P=1 for this charge-two composite, assuming this same simple effective description. BKT logarithmic corrections and finite-box/winding effects must be included for precision comparisons. [Hasenbusch's XY study](https://arxiv.org/abs/cond-mat/0502556) demonstrates the need for such corrections and independent correlation-length tests even in a simpler one-phase model.

At the **conditional zero-SOC WM temperature**, the measured diagnostic is:

| Pair | eta_P,eff | Conditional SE at fixed temperature | 2T/(pi Upsilon at larger L) |
|---|---:|---:|---:|
| 64,96 | 0.412 | 0.014 | 0.205 |
| 96,128 | 0.434 | 0.025 | 0.211 |
| 128,160 | 0.333 | 0.040 | 0.214 |

This is a substantial consistency concern, especially at the smaller sizes. At 0.001 the largest-pair value is 0.258(53), but at 0.01 it is 0.333(53); this limited comparison does not identify the cause. The errors quoted in this diagnostic hold the fitted trial temperature fixed. They do not include uncertainty in the effective theory or all finite-size corrections.

Possible explanations include scale-dependent spin/internal fluctuations, changes in the amplitude of the composite correlation near chirality ordering, preasymptotic BKT flow and incomplete sector sampling. The full field need not remain close to a fixed-amplitude chiral representation. These alternatives cannot be separated by fitting another exponent to four global values. An eta_P=1/4 crossing of two sizes would itself be another conditional pseudocritical estimator; it should not replace the disputed WM temperature as an assumed exact answer.

Binder slopes and effective chirality exponents are also supplied in the tables. The zero-SOC chirality Binder slope magnitudes at the largest-root temperature are approximately 100,176,316,401. Their size growth does not provide a stable nu determination. These quantities reinforce the need to study scaling flow rather than force a four-size extrapolation.

![Peak growth, distributions, crossing drift and joint order fluctuations](results/finite_size_and_distributions.png)

Tables: [response maxima](results/response_peaks.csv), [energy distributions](results/energy_distributions.csv), [mode check](results/energy_modes.csv), [descriptive fits](results/descriptive_scaling.csv), [effective exponents](results/effective_exponents.csv), [phase consistency](results/phase_consistency.csv), [Binder slopes](results/binder_slopes.csv), [order distributions](results/order_distributions.csv), [order covariance](results/order_covariance.csv).

## 6. What HQVs do and do not explain

Within an ideal chiral region,

\[
A=\sqrt{n/2}\,e^{i\theta}\hat d(1,i\sigma),\qquad
(\hat d,\theta)\sim(-\hat d,\theta+\pi),\quad\sigma=\pm1.
\]

The connected fixed-chirality manifold is (S² x U(1))/Z2. Its HQV generator combines a pi phase change with a director sign change. The parity of a real three-dimensional director lift is meaningful; an independent signed planar spin winding is not generally available in the full model. The Stage-1 contour diagnostics and manifold-quality checks remain necessary to distinguish HQVs from zeros of P, integer vortices and chirality features.

For an illustrative long-wavelength action

\[
F=\frac12\int d^2r[\rho_c(\nabla\theta)^2+\rho_d(\nabla\hat d)^2],
\]

a simple HQV has logarithmic energy pi(rho_c+rho_d) log(R/a)/4. An integer phase vortex has pi rho_c log(R/a). Comparing the HQV energy with its positional entropy gives a mass-only rho_c=8T/pi criterion only after any additional long-distance spin logarithm is screened, and if HQVs remain unconfined. Persisting spin stiffness, multiple coupled modes or SOC strings change that argument. The bare chiral GL reduction has rho_c=rho_d=4Kn; their renormalized finite-temperature values need not remain equal.

The spin-1 polar-condensate theory with the same connected manifold provides a useful example of a spin-disordered paired superfluid and an 8T/pi jump, but it is not a mapping of this full GL theory with its chirality sector. [Mukerjee, Xu and Moore](https://arxiv.org/abs/cond-mat/0605102) explicitly distinguish the combined spin/phase defect from its spin-disordered finite-temperature limit. The present campaign lacks the spin correlation length and spin twist response needed to establish that scale hierarchy.

For positive weak SOC, the bare chiral dipole length sqrt(2K/gD) is 44.7 at 0.001 and 14.1 at 0.01. These numbers are not renormalized thermal confinement lengths. SOC may change the asymptotic allowed defects even if the common-phase thermodynamic curves change only slightly; the limits gD to zero and L to infinity need not commute. Applying q=1/2 independently at both nonzero couplings therefore remains a finite-size comparison, not proof of a common asymptotic mechanism.

Within an easy-axis locked chiral minimum, one can fix the spin representative along +z and describe the remaining ground-state circle by the ordinary complex phase of A. A half-phase loop then requires leaving that minimum, so an integer-vortex BKT transition is a plausible long-distance candidate if the connecting texture remains costly. Deconfinement, coupled transitions or another internal phase could change this conclusion. The better finite-size q=1/2 fit does not rule out that asymptotic alternative.

The earlier HQV work supplies useful boundaries on interpretation. Stage 1 validates synthetic topology detection. The small Stage-2 thermal pilot could not confirm convincing microscopic HQVs because most surrounding contours failed manifold-quality criteria; this does not establish their absence. Stage 3 shows that constrained HQV pairs survive full twelve-component energy relaxation, while their energies remain affected by pins and boundaries. These are deterministic zero-temperature constrained excitations. They do not measure equilibrium vortex fugacity, a finite-temperature pair free energy, a transition temperature or real-time dynamics.

Evidence that would support an HQV-mediated thermal transition should combine: calibrated odd composite winding with director parity on valid larger-scale loops; spin correlation/stiffness flow showing the appropriate screening regime; charge-two algebraic correlations and their loss; and common-phase stiffness consistent with the same long-distance theory. An HQV count or a good WM curve on its own would leave central ambiguities. Further vortex pin optimization is not the most informative next step for the phase-diagram question.

## 7. Spatial observables that would change the inference

The global scalars cannot reconstruct spatial correlations. In particular, neither S(k_min) nor a second-moment correlation length can be recovered by reweighting the currently saved arrays. Reweighting adds temperature information within sampled configurations; it cannot supply missing spatial information.

### Minimum useful addition

Use the raw local fields

\[
\kappa(r)=2\operatorname{Im}\sum_\mu A_{\mu x}^*(r)A_{\mu y}(r),\qquad
\mathcal P(r)=\sum_\mu[A_{\mu x}^2(r)-A_{\mu y}^2(r)].
\]

For each, save the complex Fourier amplitudes at (2pi/L,0) and (0,2pi/L), alongside the existing zero modes C,P and the simultaneous full H. This requires **eight additional real scalars per measurement**. For complex P, also saving the two negative-wavevector modes adds four real scalars and permits explicit inversion/current-sector checks and averaging over both signs. They are not determined by the positive modes on each configuration, unlike the real chirality field. The eight-scalar minimum uses the equilibrium inversion symmetry for their expectation values; the twelve-scalar option is preferable if inexpensive. Retaining amplitudes rather than only averaged powers allows later blocking and symmetry checks. Squared Fourier magnitudes provide

\[
S_\kappa(k)=N^{-1}\langle|\kappa_k|^2\rangle,\qquad
S_P(k)=N^{-1}\langle|\mathcal P_k|^2\rangle.
\]

With S(k_min) averaged over the two directions, define

\[
\xi_2/L=\frac{\sqrt{S(0)/S(k_{\min})-1}}{2L\sin(\pi/L)}.
\]

Also retain the two directional ratios before averaging, to reveal lattice anisotropy or statistical inconsistency. Use a declared finite-volume definition consistently. In an unbiased finite box the signed/complex means vanish by symmetry; the unconnected S(0) is appropriate for standard finite-volume ratios. In the chirality-ordered region it contains the ordering contribution and is not a connected bulk correlation length. Subtracting a finite-history mean or an absolute magnetization ad hoc would create another estimator and can hide slow sign sampling.

These two ratios can show different large-L flows without assuming a vortex charge. Spatial G_kappa(r) and G_P(r), measured less often, would test whether an apparent intermediate interval has long-range chirality and exponential phase correlations, or algebraic phase correlations and short-range chirality. Ratio crossings alone still require corrections and multiple sizes; BKT ratios typically merge and drift along an algebraic phase, rather than behaving like an ordinary isolated Ising crossing. Importing conventional XY universal ratio numbers without verifying the effective field and boundary conventions would be premature.

The Fourier additions are O(N) reductions, with no full-field transfer. Measure them initially every 250 sweeps, simultaneously with H and all existing channels. Benchmark their cost and save enough cadence to estimate autocorrelation rather than assuming a longer interval creates independence. Selected-temperature FFT correlations every 5,000 sweeps can be reduced to compact direction-resolved/radial arrays; full configurations need not be saved at that frequency.

### Spin and internal-sector checks

The next priority is the density-weighted spin nematic tensor

\[
Q_{\mu\nu}(r)=\operatorname{Re}\sum_a A_{\mu a}A_{\nu a}^*
                     -\frac{n(r)}{3}\delta_{\mu\nu}.
\]

Its five independent real components are gauge invariant; within an ideal chiral region it equals n(d_mu d_nu-delta_mu_nu/3). Its structure factor/correlation length probes orientational scales without normalizing every small-amplitude site or assuming a real director everywhere. At gD=0, spin twist derivatives of the **full Hamiltonian** would supply a complementary scale-dependent spin response. At nonzero SOC, that response no longer has the same conserved-spin interpretation, so it should not be inserted into an unmodified universal criterion.

If P correlations and helicity remain inconsistent, measure the complete spin-singlet charge-two orbital tensor B_ab=sum_mu A_mu,a A_mu,b. Its trace, P=B_xx-B_yy and 2B_xy reveal cancellations or a different internal ordering channel. The chiral combinations (P minus/plus i 2B_xy)/2 are useful diagnostics, not replacements for the unrestricted field. This addition guards against declaring all U(1) coherence absent solely because one composite projection becomes small.

Occasional full matrices at three or four critical/control temperatures would permit manifold-quality, domain-wall and contour checks after coarse-graining. A practical cap is twenty configurations per selected temperature per independent run, separated by at least 200,000 sweeps initially. Such a sparse set diagnoses structures; it is not sufficient for precise equilibrium defect populations. Loop size and coarse-graining must be varied and calibrated before using microscopic composite zeros as HQVs.

## 8. Statistical reliability and the next simulation

The existing SOC audit remains relevant after reweighting. At L=160, exact completed PT loops are 65,38,98 for gD=0,0.001,0.01, including thermalization; mean completed-loop times are approximately 2.8–3.1 million sweeps, comparable to production. The critical histories contain hundreds of effective observations rather than 60,000. At 0.001 a low-temperature signed-chirality drift persists under long blocking while magnitude observables look much steadier. One seed/preparation per parameter set prevents a between-run check. A small jackknife error or successful withheld test cannot repair these limitations.

The dominant remaining uncertainties are the accessible length scales, the effective U(1) theory, and sector/preparation reliability. The original temperature grid also mattered substantially, but its interpolation uncertainty is now much reduced locally. Saving more global scalar records at the same sizes would not supply the missing spatial or independent-history information.

### A targeted, staged campaign

The proposed first stage is **gD=0 at L=64,128,160,256**. Use two new histories at 64 as an inexpensive spatial reference, and four independently seeded histories at each larger size. The 64,128,256 sequence gives factor-two comparisons over a factor-four span; 160 tests reproducibility against the current largest history. Four starts should include ordered positive chirality, ordered negative chirality, and two independent hot full-matrix configurations. At gD=0 vary the initial spin orientation as well. All starts subsequently sample the same unrestricted model.

If these histories agree but scaling remains ambiguous, add L=192 and 384, four seeds each. This gives 128/256 and 192/384 factor-two pairs and tests whether drift persists at larger absolute scales. Do not interpret L=256 or 384 as guaranteed asymptotic sizes. Continue to still larger boxes only if the observed flows and correlation lengths justify that cost.

Use gD=0.001 as the first SOC comparison after establishing the zero-SOC protocol: L=128 and 256, four independent histories each, with a starting temperature-window shift of about +0.00045. The current 0.01 results suffice as an additional comparison for this analysis; a strong-SOC campaign is not required to answer the zero-SOC question.

### Temperature ladder and sampling settings

Start with a PT interval **0.37–0.40**, concentrating the decision window at **0.3795–0.3840** for gD=0. Check that the hot end is disordered and refreshes chirality/internal sectors; extend to 0.42 if it does not. The cold end should supply an ordered control. Global acceptance alone is not the ladder-design target.

For a fixed Hamiltonian, sigma_H approximately equals T L sqrt(Cv). Choose neighboring temperatures so Delta(beta) sigma_H is about 0.7 or smaller, and inspect actual histogram overlap/flow in a pilot. This gives a physical spacing criterion as L increases. A reproducible planning ladder is generated by [proposal.py](proposal.py), using the measured fine-grid Cv_160 and a conservative positive logarithmic increment for larger L. This extrapolation is a planning assumption, not a result about the transition universality.

| L | Proposed temperature count | Decision-window spacing cap | PT interval |
|---:|---:|---:|---|
| 64 | 63 | 0.000100 | 0.37–0.40 |
| 128 | 83 | 0.000100 | 0.37–0.40 |
| 160 | 95 | 0.000100 | 0.37–0.40 |
| 256 | 183 | 0.000100, further overlap-limited if needed | 0.37–0.40 |
| 192, optional | 125 | 0.000100 | 0.37–0.40 |
| 384, optional | 326 | 0.0000667, further overlap-limited | 0.37–0.40 |

The conservative 384 ladder is costly and should be redesigned from actual 256/192 histograms before allocating it. Narrowing the ladder to the verified thermalization range is more informative than keeping 160 temperatures over a broad interval by default. A pilot can remove excessive density away from the feature while preserving overlap and internal-sector refresh at the endpoints.

Retain the validated full-Hamiltonian sampler and exchange every 5 sweeps initially. Keep existing global scalar sampling every 50 sweeps; save the eight new Fourier scalars with simultaneous H/global moments every 250 sweeps, selected compact spatial correlations every 5,000 sweeps, checkpoints every 100,000 and walker history every 100 sweeps. Record initialization, RNG streams, source revision, temperature betas, complete run boundaries and PT counters by production segment.

Provisional burn-in/production allocations are 2M/4M sweeps per history up to L=160 and 4M/8M at 256, with 1M-sweep extension batches. **These are initial allocations, not equilibration guarantees.** Aim for at least 1,000 effective records per independent history for the slow critical symmetry-even observables and spatial ratios, measured from autocorrelation and a blocking plateau, at several temperatures bracketing each feature. Seek repeated completed PT trips across most walkers and repeated sector renewal near/above the ordering region. Persistent sign at a deeply ordered temperature can be an equilibrium finite-size barrier; compare even observables and appropriate symmetry mixtures rather than demanding arbitrary frequent flips everywhere.

The strongest stop/continue test is agreement between independent hot/ordered histories, together with stable half-history results, block estimates and temperature flow. If four runs disagree, extending all lattice sizes immediately is inefficient; first resolve the sampling failure. Four independent preparations can identify a bias that a four-times-longer single history and its conditional jackknife may conceal. Actual cost and larger-L autocorrelation should be measured in the pilot; no credible GPU-hour estimate follows from the saved scalar files alone.

### What would count as resolving the phase diagram?

For a chiral non-superfluid interval, increasing L should retain chirality order while the composite correlations acquire a finite correlation length and common-phase stiffness flows to zero at temperatures strictly below chirality loss. For the reverse intermediate phase, chirality must be short ranged while a charge-two or other identified U(1) channel remains algebraic with a nonzero limiting stiffness. Demonstrate a temperature interval wider than the combined finite-size, preparation and statistical uncertainty; two unlike pseudocritical estimators separated by several error bars are insufficient.

For a coincident continuous scenario, show that both spatial sectors approach one scale with compatible correction flow and no stable interval as larger sizes are added. Report a detectable splitting threshold under explicit scaling assumptions, rather than claiming a proof of exact equality. For first order, require stable coexistence weights, an energy-density split and growing barriers with repeated equilibrium switching. If crossover lengths still exceed the largest boxes, the scientifically correct result remains a bound conditional on the tested corrections or an unresolved phase structure.

The immediate aim should be to make splittings around 10^-4–10^-3 testable with independent spatial evidence. The present reweighting can locate finite-size features more finely than the old 0.00064 grid, but it does not establish a 10^-5 thermodynamic resolution. More precise temperature sampling alone cannot settle that distinction.

## 9. Reproduction and artifact guide

Run from the repository root with Python, NumPy, SciPy and Matplotlib available:

```powershell
python -B exploration/he3_transition_study/prepare.py
python -B exploration/he3_transition_study/study.py
python -B exploration/he3_transition_study/scaling.py
python -B exploration/he3_transition_study/reference.py
python -B exploration/he3_transition_study/figures.py
python -B exploration/he3_transition_study/proposal.py
python -B exploration/he3_transition_study/checks.py
```

`prepare.py` reads one original channel at a time and writes disposable binned caches under `results/cache/`, ignored by Git. It records ZIP member CRC32 values and file sizes to detect source changes; these are practical change checks, not cryptographic provenance. The original NPZ files are read-only. `reference.py` and `proposal.py` also read the earlier `he3_soc/results/blocks/` moment caches, which are already present in this worktree. No script writes into that directory.

The three PNG figures above contain the principal visual evidence. CSVs retain the numerical values, chosen roots, conditional errors, failed-delete counts, boundary rejection and analysis sensitivities. `curves_gd=...npz` files hold smooth moments-derived curves, joint deletes, grouped-block deletes and numerical variants, so downstream comparisons do not require repeatedly decompressing the large original datasets. In `temperature_sensitivity.csv`, rows ending `_se` store the conditional error in the `temperature` column; they are explicitly labeled and are not temperatures to fit. NaN boundary estimates are intentional.

The tested environment is Python 3.12.10, NumPy 2.4.6, SciPy 1.18.1 and Matplotlib 3.11.2; see [environment.json](results/environment.json). The current tracked source revision is `84fe77bc5c90822a1c74a613fb75a1ce04ac58fe`. The original campaign manifests contain no historical source revision, so this audit cannot certify the exact historical source text independently. All numerical conclusions are conditional on the saved model/measurement definitions and the sampling limitations described above.
