# Parallel-tempering transport and its limits

This section is generated from a separate read-only investigation of all 16 NPZ
manifests, using `python exploration/he3_soc/pt_diagnostics.py`. It uses no local
configuration evolution or new Monte Carlo simulation. Numeric evidence is in
`results/pt_summary.csv`, `pt_edges.csv`, `pt_walkers.csv`,
`pt_endpoint_events.csv`, and `pt_time_windows.csv`.

## What was recorded

All 16 runs used a **5-sweep exchange interval**, not the 10 sweeps stated in the
request. Every run contains 1,000,000 alternating even/odd exchange rounds during
5,000,000 local sweeps. Each adjacent pair therefore has 500,000 attempted
exchanges. Acceptance counters and exact completed round trips include the
2,000,000 thermalization sweeps as well as the 3,000,000 production sweeps.
All 160 physical walkers were recorded at 500-sweep intervals, including sweeps
0 and 5,000,000: 10,001 history rows per run. The script checked that every
history row is a permutation of all 160 temperature indices, that sweep
coordinates increase, and that all walker IDs are present.

The code keeps configurations and their random-number streams in physical walker
slots. Only temperature assignments change. Measurement arrays are scattered
into fixed temperature order; consecutive observations at a temperature can
therefore come from different physical configurations. See
`core/parallel_tempering.py:59` for the acceptance ratio, `:79`–`:91` for trip
bookkeeping, `:248` for exchange scheduling, `:268` for history downsampling,
`:273` for measurement scattering, and `:304` for diagnostics scope.

A stored completed trip means: first reach an endpoint, reach the opposite
endpoint, then return to the original endpoint. Walkers initially in the
interior only start their clock when they first reach an endpoint. A one-way
passage is insufficient. Mean stored trip durations are conditional on completed
trips and exclude unfinished trips; this censoring matters especially at large
L. The exact counters are authoritative for completed loops. History-derived
passages and side changes are **lower bounds**, because visits or multiple
crossings between snapshots can be missed.

## Overall acceptance conceals a qualitative change in transport

| g_D | L=64 acceptance / exact trips | L=96 | L=128 | L=160 |
|---:|---:|---:|---:|---:|
| 0 | 0.76918 / 1920 | 0.66149 / 542 | 0.56111 / 183 | 0.46976 / 65 |
| 0.001 | 0.76910 / 1747 | 0.66127 / 493 | 0.56097 / 150 | 0.46959 / 38 |
| 0.01 | 0.76788 / 1787 | 0.65922 / 596 | 0.55792 / 252 | 0.46570 / 98 |
| 0.1 | 0.76312 / 0 | 0.65300 / 0 | 0.55225 / 0 | 0.46208 / 0 |

The acceptance is attempted-exchange weighted; here all edge attempt counts
are equal. At fixed L its change with SOC is small, yet transport changes
sharply at g_D=0.1. In particular, L=64 has 76.3% acceptance and zero completed
loops despite 5,000,000 sweeps. The explanation cannot be reduced to generally
small acceptance.

The weak-SOC L=160 cases are also far from unrestricted mixing. Only 61, 35,
and 84 walkers complete any exact trip at g_D=0, 0.001, and 0.01 respectively.
Their mean completed-trip times are approximately 3.06, 2.97, and 2.77 million
sweeps, compared with a 3-million-sweep production interval. These numbers do
not prove failure of a particular thermodynamic observable, but they preclude
treating the 160 temperature histories as independently equilibrated chains.

![Acceptance and sampled endpoint flow](results/pt_acceptance_flow.png)

## Strong SOC has sampled one-way escape, not zero endpoint travel

| L | Exact round trips | Sampled cold→hot passages, full run | Sampled hot→cold passages | Cold→hot completions during production | Both endpoint visits strictly within production |
|---:|---:|---:|---:|---:|---:|
| 64 | 0 | 14 | 0 | 2 | 2 |
| 96 | 0 | 6 | 0 | 1 | 1 |
| 128 | 0 | 5 | 0 | 1 | 1 |
| 160 | 0 | 3 | 0 | 2 | 1 |

The zero-trip statement must not be paraphrased as no walker reaching the
other end. Sampled histories directly show opposite-endpoint visits. Every
observed full endpoint passage is cold→hot; no hot→cold passage is resolved in
these snapshots. Unresolved short endpoint visits remain possible, so the
directional statement refers to the saved histories, not an unavailable exact
one-way counter. The exact zero-loop counters independently establish the
absence of full returns for the entire run.

The six sampled production completion events are:

| L | Walker | Sweep reaching hot endpoint |
|---:|---:|---:|
| 64 | 35 | 2,995,500 |
| 64 | 15 | 3,401,000 |
| 96 | 7 | 3,835,000 |
| 128 | 10 | 3,363,000 |
| 160 | 17 | 2,065,000 |
| 160 | 30 | 2,085,000 |

These are snapshot arrival times, with at most a 500-sweep recording resolution
and possible earlier unrecorded visits. They should be compared with the
independent physical-walker branch conversion analysis. In particular, branch
conversion and arrival at the hot endpoint need not coincide; L=128 shows a
substantial delay. The observed one-way pattern is consistent with irreversible
loss of the high-Qzz branch in this run, rather than repeated equilibrium
conversion between branches.

![All-coupling walker histories](results/pt_walker_histories_L160.png)

## The strongest spin-front bottleneck is distinct from the lowest swap acceptance

For each snapshot the script labels a walker by its most recently **observed**
endpoint. The production cold-labelled fraction at each temperature is the
number of cold-labelled observations divided by all endpoint-labelled
observations there; walkers with no observed endpoint yet are excluded. This
is a descriptive transport statistic. Under nonstationarity and coarse history
recording it is not an equilibrium diffusivity estimator.

At strong SOC its steepest adjacent drop lies at the same temperature pair as
the largest spin-alignment step:

| L | Temperature pair | Cold-labelled fraction drop | Swap acceptance on that edge |
|---:|---:|---:|---:|
| 64 | 0.355607–0.356420 | 0.19480 | 0.706848 |
| 96 | 0.364834–0.365461 | 0.24263 | 0.642936 |
| 128 | 0.367933–0.368546 | 0.33619 | 0.551888 |
| 160 | 0.370379–0.370990 | 0.52366 | 0.456216 |

The global acceptance minimum in every strong-SOC run is instead near
T=0.465949, with values 0.559892, 0.366478, 0.221984, and 0.121484 for increasing
L. The spin-front edges exhibit local depressions, but they retain substantial
acceptance. All 160 walkers are observed on both sides of each of the four
listed front edges during production. Consequently there is no single
permanently uncrossed temperature edge separating the walkers. The obstruction
concerns sustained transport between the whole low- and high-temperature
configuration sectors, rather than an impermeable geometrical cut in the
ladder. Successful local swaps can exchange two configurations of the same
long-lived branch, or return an opposing-branch walker before it traverses the
whole ladder.

For g_D=0 and 0.001 the large-L minimum acceptance and strongest interior flow
drop occur near T≈0.382; for g_D=0.01 they move to T≈0.38576. These identify
sampling bottlenecks near the weak-SOC transition region, not independently
determined critical temperatures. Flow estimates have artificial sharp changes
at the endpoints because labels are reset there, especially visible for L=64;
the script's literal maximum-drop statistic should therefore only be read as
an interior bottleneck where the corresponding pair is interior.

## Broad local motion persists within restricted sectors

During production, median per-walker index spans for strong SOC are 132, 120,
113, and 108 out of the possible 159, versus 159 for nearly all weak-SOC runs
(157.5 for g_D=0.001,L=160). The minimum number of distinct walkers seen at a
fixed temperature is only 33, 44, 50, and 54 for strong SOC. At g_D=0 the
corresponding numbers are 160, 156, 136, and 113. Thus the large-L weak runs
also have finite-run coverage limitations, but their exchange of endpoint
cohorts is qualitatively different.

The normalized Shannon entropy of a walker's empirical temperature occupation
distribution, averaged over walkers, is 0.906, 0.887, 0.879, and 0.873 for
strong SOC. At g_D=0 it is 0.992, 0.976, 0.947, and 0.916. An entropy close to
one would describe a broad flat temperature occupation histogram, but this
statistic cannot establish decorrelation of spin textures or topological
defects. Even broad motion within one long-lived configuration sector can
produce high occupation entropy.

![Strong-SOC occupation distributions](results/pt_strong_occupation.png)

The time-window table resolves 500,000-sweep intervals separately. The strong
runs continue to show broad temperature motion, but no observed reverse
endpoint passage in any interval. The pronounced dependence of the retained
low-temperature branch population on L, its slow depletion, and the
temperature-space sorting of those configurations make a thermodynamic reading
of the low-temperature Qzz step unsafe without evidence of reverse branch
conversion. High swap acceptance and fixed-temperature switching do not supply
that evidence.

## Consequences and useful follow-up

PT metadata alone do not prove stationarity of any observable or quantify an
independent-run error. Exact loops test mobility between endpoints; they do
not test whether a defect population is equilibrated within a walker. The
present strongest conclusion is a clear failure of bidirectional, repeatable
configuration-sector exploration at strong SOC, corroborated by branch
inventories. Production-only round-trip counters cannot be reconstructed
exactly, because only cumulative exact counters were saved; coarse endpoint
passages should not be silently substituted.

The most informative next evidence would be configuration/branch histories
from deliberately different initial sectors, demonstrating both conversion
directions and stationary sector populations in the narrow region of
competition. For future diagnostics, preserve exact one-way endpoint passage
counters and production-start snapshots of cumulative counters. Such additions
would disambiguate censoring and time scope, but they would not themselves
repair insufficient exploration of the competing states. This investigation
does not modify the PT implementation or initiate further simulation.
