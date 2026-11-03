# Old weak-SOC helicity: measured decomposition and its limits

The saved histories identify two contributions to the response loss: curvature decreases substantially while current fluctuations increasingly screen it. In a specified narrow transition window at L=160, screening accounts for approximately 58–67% of the helicity loss, curvature for 33–42%. About 59% of the curvature decrease is assigned to the decrease of mean density by an exact symmetric product decomposition; the remaining 41% comes from the decrease of curvature per unit density. These are measured shares for finite-size windows, not universal numbers, independent physical causes or defect identifications.

The current distributions are close to Gaussian in shape while their width increases. The sampled variance is not dominated by a few extreme records. Coarse partitions by chirality sign, absolute chirality, density or spin alignment account for at most 0.0261% of the variance through differences of conditional mean current. Conditional current variances do vary with density and chirality. However, the apparent current-power/chirality association largely disappears after a linear density adjustment at the representative central temperatures. These scalar records therefore do not isolate domain walls, half vortices, spin textures or winding sectors as the physical origin of screening.

## Data, definitions and uncertainty

Only the old `soc_scan_02` files `data/he3/P=20_gd={0,0.001,0.01}_gaussian/L={64,96,128,160}.npz` were read. No new simulation, current campaign, cluster data or existing output was modified. Each dataset has 60,000 saved records at 50-sweep intervals, following two million burn-in sweeps, with parallel-tempering exchange attempts every five sweeps. The compact old caches contain 600 means of 100 saved records. Raw checks used ten selected temperature columns per dataset, reading one archive member at a time. Source member CRCs and file sizes match the earlier cached inventory. The maximum difference between selected raw means and old cache means is 1.46e-11.

With extensive twist derivatives I_mu and K_mu, N=L², fixed model coupling K=1, define

```
D_mu = <K_mu>/N
S_mu = (<I_mu²> - <I_mu>²)/(T N)
Y_mu = D_mu - S_mu
D = (D_x+D_y)/2, S = (S_x+S_y)/2, Y = D-S
M = (<I_x>²+<I_y>²)/(2 T N)
```

`M` is the amount added to the uncentered estimator by subtracting the squared sampled means. The model coupling K and twist curvature K_mu are different quantities.

Unless stated otherwise, errors are paired delete-one jackknife standard errors from 30 joint blocks of 2,000 saved records, each spanning 100,000 sweeps. A whole time block is deleted simultaneously for all temperatures and moments within a dataset. This retains within-run covariance, including x/y and endpoint covariance. It does not estimate unknown equilibration bias or correlations between separately archived runs. Block sizes 100, 500 and 5,000 records and time halves are also reported. Fixed-temperature records contain exchanged walkers; their sequence is not physical time evolution.

## How much falls through curvature, and how much through screening?

The primary window uses saved indices 68–74 for g_D=0 and 0.001, and 73–79 for g_D=0.01. Its center is the saved grid point close to the previously identified response peak. This is a descriptive common window within each coupling, not an estimate of a thermodynamic transition temperature. No interpolation is needed for the endpoint results.

| g_D, L=160 | T interval | D: low → high | S: low → high | Y: low → high | Curvature share of Y loss | Screening share |
|---|---|---|---|---|---|---|
| 0 | 0.38030047–0.38413969 | 2.55476 → 2.18328 | 1.19559 → 1.96497 | 1.35917 → 0.21832 | 32.562 ± 0.244% | 67.438 ± 0.244% |
| 0.001 | 0.38030047–0.38413969 | 2.57084 → 2.19667 | 1.17493 → 1.86811 | 1.39591 → 0.32856 | 35.055 ± 0.357% | 64.945 ± 0.357% |
| 0.01 | 0.38349459–0.38739762 | 2.55700 → 2.21471 | 1.17661 → 1.64707 | 1.38039 → 0.56764 | 42.115 ± 0.466% | 57.885 ± 0.466% |

The exact definitions are curvature share = −ΔD/(−ΔY) and screening share = ΔS/(−ΔY). Their errors are anticorrelated because the two shares sum to one. The underlying Y drops are 1.14085±0.00899, 1.06735±0.01149 and 0.81276±0.00917.

Ordinary least-squares slopes over all seven temperatures in each primary window give:

| g_D, L=160 | slope of D | slope of S | slope of Y |
|---|---|---|---|
| 0 | −108.05 ± 0.19 | +223.21 ± 2.05 | −331.26 ± 2.13 |
| 0.001 | −109.31 ± 0.20 | +198.67 ± 2.59 | −307.98 ± 2.68 |
| 0.01 | −97.54 ± 0.07 | +132.71 ± 1.69 | −230.25 ± 1.69 |

These are descriptive slopes of a strongly curved function, not claims that the response is linear. An independent local canonical derivative avoids the window convention. For T-independent Hamiltonian parameters,

```
dD_mu/dT = Cov(K_mu,E)/(N T²)
dS_mu/dT = [Cov(I_mu²,E) - 2<I_mu>Cov(I_mu,E)]/(N T³)
           - Var(I_mu)/(N T²).
```

At the central saved temperatures, the directional averages give (dD/dT,dS/dT) = (−199.29±6.36,+407.41±22.00), (−223.15±9.94,+376.86±27.29), and (−180.81±4.41,+245.76±21.08). The associated curvature shares are 0.3285±0.0104, 0.3719±0.0128 and 0.4239±0.0199. Local derivatives are steeper than broad fitted slopes because the response peak is narrow. Ratio estimates where the total derivative is unresolved are flagged in `canonical_derivatives.csv` and should not be interpreted.

The shares depend on size. In the primary window, curvature fractions for L=64,96,128,160 are respectively:

- g_D=0: 0.4154, 0.3863, 0.3491, 0.3256.
- g_D=0.001: 0.4268, 0.3903, 0.3663, 0.3506.
- g_D=0.01: 0.4573, 0.4296, 0.4253, 0.4211.

The size dependence is mainly a more rapidly increasing screening contribution; curvature also changes with L. Full uncertainties and directions appear in the CSVs.

![Curvature, screening and helicity versus temperature](decomposition.png)

### Density and normalized curvature

Let n=<total density>/N and R=D/(4 K n), with K=1 here. The reference 4Kn is the uniform ideal chiral-manifold curvature. For the actual full-field histories, R is only a normalized curvature diagnostic; it is not uniquely a phase, spin or bond-coherence observable. The exact product identity is

```
D = 4 n R
ΔD = 4 R_mid Δn + 4 n_mid ΔR,
```

where the midpoint means the arithmetic mean of the two endpoint values. This symmetric allocation is additive and exact but is an accounting convention, not a causal intervention.

| g_D, L=160 | n: low → high | R: low → high | Density share of curvature loss |
|---|---|---|---|
| 0 | 0.893561 → 0.813951 | 0.714770 → 0.670582 | 59.3776 ± 0.0037% |
| 0.001 | 0.897085 → 0.816960 | 0.716441 → 0.672209 | 59.4745 ± 0.0039% |
| 0.01 | 0.895888 → 0.822965 | 0.713538 → 0.672783 | 59.0694 ± 0.0040% |

The unusually small error on the last column reflects very strong density/curvature covariance, not a universal 59% law. The corresponding shares of total helicity loss are approximately 19.33%,20.85%,24.88% from the density term, 13.23%,14.21%,17.24% from the R term, and the remaining screening shares in the first table. A fixed-amplitude treatment would miss a meaningful component of the measured change. A density-only treatment would also miss the R decrease and the dominant screening term.

### The hotter shoulder and window dependence

The primary upper endpoint retains Y=0.568 at g_D=0.01, compared with Y=0.218 at zero SOC. Consequently, the larger curvature share at g_D=0.01 cannot by itself establish a different asymptotic mechanism. There is a broader hotter region in which curvature has mostly fallen but screening continues to increase. For g_D=0.01,L=160, T=0.38739762–0.39340368 gives Y=0.56764→0.02189: 27.48±0.61% of this additional loss comes from curvature and 72.52±0.61% from screening. This separation of shapes is a finite-size observation, not proof of an intermediate thermodynamic phase.

For L=160, changing the symmetric halfwidth to 1,2,3,5,8 saved-grid steps gives curvature shares:

- g_D=0: 0.3324,0.3254,0.3256,0.3273,0.3555.
- g_D=0.001: 0.3770,0.3560,0.3506,0.3379,0.3586.
- g_D=0.01: 0.4349,0.4245,0.4211,0.4242,0.3895.

This window sensitivity is distinct from the smaller conditional jackknife errors.

## Current distributions, size dependence and conditional associations

Define j_mu=I_mu/sqrt(TN), whose variance equals S_mu. At selected temperatures away from the transition, Var(I_mu) is approximately proportional to L²: extensive current scales in width like L, rather than like N. Across the central finite-size crossover, descriptive log–log slopes over the four sizes rise to about 2.2 because S itself grows with L. These slopes are not critical exponents or evidence for a particular vortex charge.

Across 240 directional distributions, standardized excess kurtosis ranges from −0.0379 to 0.0906. The largest value is 0.0906±0.0247 at g_D=0,L=64,T=0.38157194; it is a selected maximum across many tests. The top 1% of absolute centered currents contribute 8.223–8.831% of the total variance, compared with 8.449% for a Gaussian. No single 2,000-record time block supplies more than 6.33% of these tail records. This is incompatible with a picture in which the measured screening is numerically carried by one or two extreme records. It does not bound unsampled tails or exclude topological events hidden inside a nearly Gaussian extensive sum. Tail error bars use the observed full-history quantile boundaries as fixed thresholds.

![Standardized current distributions and Gaussian reference](current_distributions.png)

For any saved scalar partition C, the law of total variance is exact:

```
Var(j) = E[Var(j | C)] + Var(E[j | C]).
```

Partitions used here are chirality sign, quartiles of |chi|, density, and Qzz. The largest measured between-group mean fraction among all 960 directional partitions is 0.0002604, or 0.02604% of total variance. At L=160 central temperatures, chirality-sign fractions for the x current are 7.25e-7,4.01e-8,1.90e-5 for g_D=0,0.001,0.01. Thus mixing the *observed scalar classes* does not generate appreciable screening through different mean currents. This does not constrain unobserved winding or defect labels.

Conditional variances themselves do change. For g_D=0,L=160,T=0.38221074, the lowest/highest |chi| quartiles have x-current variances 1.6753/1.4224, and conditional curvatures 2.2653/2.3926. Both components move in the direction of reduced response for the lower-order records. The difference of conditional moments shown in the figure is **not** automatically a thermodynamic conditional helicity: the conditioning membership may itself change under a twist.

![Conditional curvature and current variance](conditional_moments.png)

To check whether the association isolates chirality, define the direction-averaged centered current power P=[(I_x−<I_x>)²+(I_y−<I_y>)²]/(2TN). At L=160 central temperatures:

| g_D | corr(P,|chi|) | corr(n,|chi|) | partial corr(P,|chi| given n) | partial corr(P,n given |chi|) |
|---|---|---|---|---|
| 0 | −0.0629±0.0034 | 0.8607 | −0.0030±0.0035 | −0.0338±0.0037 |
| 0.001 | −0.0649±0.0036 | 0.9038 | −0.0023±0.0032 | −0.0283±0.0040 |
| 0.01 | −0.0490±0.0044 | 0.8851 | −0.0088±0.0047 | −0.0158±0.0042 |

The apparent chirality association is largely shared with density. Linear partial correlation does not remove nonlinear dependence or establish causality, but it directly challenges an inference that this scalar association uniquely identifies chirality walls. It does not exclude walls acting through density: density can be an intermediate variable in a causal chain, and conditioning can also introduce selection effects. The partial-correlation jackknife recomputes both current means within every deleted-block sample using joint raw moments.

## Directions, cancellation and stationarity

The diagonal x/y responses agree within sampled errors. Across all 12 datasets in 0.375<T<0.390, the largest |Y_x−Y_y| divided by its paired error is 3.00, a selected maximum over many temperatures. At L=160 and the central temperatures, Y_x−Y_y equals −0.00535±0.01388, −0.00930±0.00805, and +0.00627±0.01091.

This does not establish a completely isotropic response tensor. The saved currents have nonzero cross covariance. At those same central temperatures, corr(I_x,I_y) = −0.06333±0.00454, −0.07737±0.00512, and −0.09072±0.00330. Below the transition the correlation is about −0.09; at T=0.40031 it is approximately +0.02. The forward mixed-gradient stencil preserves x↔y but need not preserve separate mirror symmetries. The off-diagonal helicity requires <K_xy>/N−Cov(I_x,I_y)/(TN), and K_xy was not saved. Current cross covariance alone therefore cannot identify off-diagonal stiffness or defects; the separate theory audit verifies the required mixed derivative.

The mean-current correction M is at most 0.0002311 over all 1,920 weak-SOC grid points, and at most 0.0328 of the corresponding Y standard error. Consequently, centered versus uncentered current moments cannot account for the observed stiffness fall, despite slow chirality-sign sampling in some runs. This small correction says nothing about unvisited topological sectors.

Cancellation becomes statistically severe as Y approaches zero. For g_D=0,L=160 at T=0.38414, D=2.18328 and S=1.96497, giving Y=0.21832±0.00720 and (|D|+|S|)/|Y|=19.0. At T=0.38544, Y=0.06582±0.01015 and the ratio is 64.2. At higher temperatures the ratio becomes hundreds or thousands and may be undefined near a sign change. This is primarily cancellation of uncertain ensemble estimates; double-precision arithmetic is ample for subtracting quantities of order two. Small negative estimates consistent with their errors are not evidence for negative equilibrium stiffness.

The first/second-half Y difference in the three selected transition-region temperatures exceeds two diagnostic errors only once among 36 comparisons: g_D=0.001,L=128 at the center, with z=2.22. In contrast, low-temperature signed chirality shows coherent drift. At g_D=0.001,L=160,T=0.34956910, its mean changes 0.29516→0.15723, z=−7.09. At g_D=0.01,L=96 the corresponding z is −5.24, and at g_D=0,L=160 it is −3.94. These observations reinforce the distinction between reasonably stable sampled even moments and certification of equilibration of every sector.

Coarsening blocks from 100 to 5,000 records does not produce a systematic large growth in the primary decomposition-share errors. There is not a clean universal error plateau: only 12 blocks remain at 5,000 records, and some errors decrease substantially. For example the g_D=0,L=160 central Y error changes 0.00824,0.00891,0.00805,0.00432 across the four block lengths. The 2,000-record choice is an explicit working convention, not a proof of independent blocks.

## Conclusions that the data do and do not support

**Exact accounting:** the response drops because the measured curvature decreases and the measured current variance increasingly cancels it. Density accounts for part of the curvature loss, and normalized curvature also falls. The estimator, variance decompositions, canonical derivatives and product identities are independently checkable from saved moments.

**Measured finite-size behavior:** screening supplies the larger contribution in the stated windows and becomes relatively more important at larger size. The current distribution broadens without developing a dominant sampled rare tail. Lower-density/lower-order records tend to have higher current variance. The g_D=0.01 curve has a broader hotter screening shoulder. These are observations of the sampled ensemble, with the uncertainty and stationarity qualifications above.

**Not identified:** which defects or internal textures produce that broadening; whether phase, spin and chirality decouple asymptotically; a thermodynamic split between transitions; an HQV charge; or a mass-only universal-jump relation. A favorable q=1/2 finite-size fit and a broadening Gaussian-looking current do not answer these questions. Spatial configurations, gauge-invariant winding/defect diagnostics, spin and phase correlations, mixed stiffness and independently equilibrated histories would distinguish mechanisms more directly.

## Reproduction and files

Run the four short scripts with `python -B` in this directory or from the repository root, in this order: `decompose.py`, `current_histories.py`, `conditional_associations.py`, `summarize_evidence.py`. They write only files in this data investigation directory. Existing caches are read, not rewritten. The raw-history script verifies campaign metadata and cache provenance before reporting results.

- `curves.csv`, `transition_windows.csv`, `hotter_shoulders.csv`: directional components, endpoint accounting, density accounting and local window slopes.
- `canonical_derivatives.csv`: selected raw covariance derivatives with paired errors and a resolved-slope flag.
- `current_statistics.csv`: current moments, tails, x/y covariance, correlations and block-length checks.
- `conditional_moments.csv`, `total_variance.csv`, `partial_associations.csv`: scalar conditional moments and variance decomposition, including the density-confounding check.
- `time_halves.csv`, `block_sensitivity.csv`, `current_size_scaling.csv`: finite-history and size diagnostics.
- `raw_provenance.json`, `method.json`, `raw_cache_checks.csv`, `numerical_checks.json`: provenance, conventions and numerical consistency checks. Exact algebraic residuals are at most 8.9e-16, apart from the separately quoted raw/cache mean comparison.
- `current_histograms.npz`, `joint_histograms.npz`: compact distribution summaries, not new configurations or Monte Carlo output.
