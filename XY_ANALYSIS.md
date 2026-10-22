# Analysis utilities and verification

The numerical helpers trust the Model -> Runtime -> Simulation -> saved-data
contracts. They contain the estimator, not input schemas or reporting machinery.
CLI boundary checks remain in `analyze.py`; its command-line interface is unchanged.

```powershell
python -B analyze.py data/xy_run.npz --output-dir analysis_results/xy --no-show
python -B analyze.py data/other_model.npz --pt-only --no-show
```

## Numerical interfaces

- `binder_cumulant(m2_samples, gaussian_factor)` returns the Binder curve:
  `1 - mean(m2**2)/(g*mean(m2)**2)`, with g=3 for Ising and g=2 for XY.
- `block_means` averages consecutive time blocks and drops an incomplete tail.
  `blocked_mean_error` returns the mean and its block standard error.
- `blocked_jackknife` / `jackknife_from_blocks` return `(estimate, error)`.
  Nonlinear observables are recomputed after deleting each block, using
  `sqrt((B-1)/B * sum((theta_deleted - mean(theta_deleted))**2))`.
- `bootstrap_block_means` returns resampled moments. One multinomial count
  vector is shared across every temperature and channel of a time block.
  Independent size runs use independent RNG streams. It does not resample
  individual correlated measurements or final error bars.
- `linear_crossings(T, y1, y2=0, temperature_range=...)` returns temperatures.
  Multiple roots remain separate. A NaN marks a flat interval or undefined
  curve; JSON exports represent it as `null`. Roots are never extrapolated.
- `bootstrap_crossings(T, left, right=0, temperature_range=...)` returns one
  temperature per paired bootstrap draw, with NaN if there is no unique root.
  It replaces `crossing_statistics`; there are no root objects, status strings,
  midpoint branch assignments, confidence thresholds, or per-branch metadata.
- `bootstrap_error(draws)` returns the sample standard deviation (`ddof=1`),
  or `None` if any draw is undefined or fewer than two draws exist.

At least two complete blocks are required mathematically for an error estimate;
this replaces the arbitrary minimum of 20. Two blocks do not establish adequate
sampling. Choose block lengths that cover autocorrelation and test stability.
Binder normalization and the connected helicity fluctuation term are unchanged.
Undefined moment ratios still produce NumPy warnings/NaNs rather than being hidden.

## Weber-Minnhagen

The fitted leading form is
`rho(L,T) = (2T/pi) * [1 + 1/(2 log(L) + C)]`, with free shared C.
The weighted chi-square is minimized over C at each temperature, followed by
local refinement of the temperature minimum. This numerical search is unchanged.
The positive correction uses `a=1/(2 log(L_min)+C)>=0`; the zero endpoint is
C=infinity. Minimum-size cuts and a temperature window remain supported.

`fit_weber_minnhagen` now returns eight useful fields: `temperature`, `C`,
`chi2`, `degrees_of_freedom`, `min_size`, `profile_temperatures`, `profile_chi2`,
and `reason`. A failed estimate has `temperature=None` and a plain reason.
The status taxonomy, duplicated profile diagnostics, and validation layers
were removed. Insufficient sizes/temperatures, a flat profile, a boundary
minimum, nonpositive weights, and an infinite C still prevent a claimed fit.
Zero-error columns are not silently dropped or interpolated across: select
an appropriate contiguous window instead.

`XYAnalysis` repeats the fit on joint bootstrap curves with fixed nominal
standard-error weights. The final fit has n_sizes-2 nominal degrees of freedom;
each profile point has n_sizes-1. Chi-square is a fit diagnostic, not a calibrated
p-value. Leading-log scaling bias is outside the bootstrap error.

## Model callers and PT

`XYAnalysis.results` keeps the raw observables and their jackknife errors.
Crossing groups now contain a temperature list, one `error`, and internal
`bootstrap_temperatures`. A single uncertainty requires a unique central root
and every requested bootstrap draw to be defined. Observable plots retain
their error bars. The helicity panel divides values and errors by
`1 + 1/(2 log(L) + C)` using one shared C from the primary WM fit, and marks
`(T_fit, 2T_fit/pi)`. It shows fitted sizes; if no fit is available it stays raw.
These plotted errors hold C fixed; fit uncertainty is reported separately.
Raw results are unchanged. Transition plots mark unavailable errors with x.

This deliberately replaces conditional branch matching and the 80% rule.
Unresolved branches keep their central roots but receive no scalar error.
This is a conservative reporting change, not evidence that the uncertainty
vanished. Bootstrap draws remain available through `estimate_transitions()`;
large draw arrays are omitted from the JSON summary. Percentile intervals and
administrative validity counters have been removed.

`PTAnalysis.summary()`, `.format_summary()` and `.plot()` remain. Only runs
with PT metadata appear in the summary. Rates, paired counts, walker trip
counts, per-walker and pooled completed-trip means, temperatures and counter
scope are retained. No mapping validation, metadata consistency checks, deep
copies, obsolete-format fallbacks, or duplicate aggregate records remain.
Unattempted rates and means without completed trips are `None`. Counters
include thermalization; completed-trip averages exclude unfinished trips.
Neither acceptance nor trips prove equilibration.

## Changes and line counts (simplification snapshot)

Physical lines, including comments and blank lines; captured before editing on
2026-10-21. Every Python module under `utils/Analysis/` was inspected and changed.

| File | Before | After |
|---|---:|---:|
| `utils/Analysis/binder.py` | 17 | 7 |
| `utils/Analysis/bkt.py` | 158 | 102 |
| `utils/Analysis/crossings.py` | 160 | 44 |
| `utils/Analysis/parallel_tempering.py` | 172 | 63 |
| `utils/Analysis/statistics.py` | 73 | 49 |
| `models/xy/analysis.py` | 279 | 258 |
| `tests/test_analysis_statistics.py` | 132 | 116 |
| `tests/test_bkt.py` | 130 | 120 |
| `tests/test_crossings.py` | 158 | 53 |
| `tests/test_pt_analysis.py` | 141 | 81 |
| `tests/test_xy_analysis.py` | 268 | 172 |
| `tests/test_analyze_cli.py` | 167 | 167 |
| `tests/test_ising_analysis.py` | 0 | 43 |
| `README.md` | 81 | 81 |
| `XY_ANALYSIS.md` | 182 | 152 |

The five utility modules total **580 -> 265 lines**. Complexity was deleted,
not moved into the model: XY analysis also shrank. Tests for removed validation
and obsolete schemas were replaced by checks of the numerical algorithms.
`README.md` links this guide and matches the current plotting behavior.
`analysis_results/xy/` contains regenerated PNG/PDF figures and `analysis.json`.

## Verification against the captured baseline

The initial 61-test suite reported 2 failures and 11 errors because several tests
expected model features absent from the code already in the workspace. The
updated **45 CPU tests all pass**, including Ising and foreign-model PT analysis.
Run them without the existing GPU test modules:

```powershell
$files = @('test_analysis_statistics.py', 'test_bkt.py', 'test_crossings.py', 'test_pt_analysis.py', 'test_xy_analysis.py', 'test_ising_analysis.py', 'test_analyze_cli.py')
foreach ($file in $files) { python -B -m unittest discover -s tests -p $file -v }
```

The real `data/xy_run.npz` and default CLI were run before and after using
block_size=200, 400 resamples, seed=42. Raw XY observables, jackknife errors,
all joint bootstrap curves, all 400 fitted temperatures per size cut, WM
parameters/errors/chi-square, and PT rates/counts/trip means are **exactly equal**.

| WM cut | T before = after | Bootstrap SE | C | Chi-square |
|---|---:|---:|---:|---:|
| L >= 48 | 0.8918920561741602 | 0.002359009826689857 | 1.594776482507883 | 0.20200844715448785 |
| L >= 64 | 0.8919522553604652 | 0.0050409329553602064 | 1.663507122825056 | 0.02666049347435543 |

All 104 Binder roots and four raw jump roots agree within 1.12e-16.
Previously five Binder branches and three jump curves received conditional
errors. They now receive unavailable errors: Binder has multiple central
roots, and every jump has some nonunique bootstrap draws. Well-defined
crossing bootstrap standard deviations are preserved in synthetic regression
checks. This change is intentional and separate from unchanged curve errors.

Saved input bytes, core/GPU modules, simulation scripts, data IO, `analyze.py`
and Ising analysis are unchanged by this refactor. Hash verification detected
an external edit to `utils/temperature_ladder.py` during the task; it was left
untouched. No GPU tests, simulations, benchmarks, dependencies, commits or pushes
were used. Source snapshots and numerical baselines are retained in a temporary
directory for this session.
