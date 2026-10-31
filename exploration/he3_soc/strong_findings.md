# Strong-SOC investigation: branch conservation, rare excursions and stiffness

All results in this note refer to the four `data/he3/P=20_gd=0.1_gaussian/L=*.npz` files. Reproduce with `python exploration/he3_soc/strong_soc.py`; outputs are `results/strong_*.csv` and four composite PNGs. Temperatures are the actual saved ladder values. The analysis uses measured samples at fixed temperature, and separately aligns samples with physical walker identities at exactly matching saved PT-history sweeps. No interpolation of walker identities is used.

## 1. The clearest failure of equilibrium sampling is irreversible branch depletion

There are two extremely well separated configuration populations near the apparent low-temperature anomaly. Define a diagnostic branch indicator

\[
b(t,T)=\mathbf 1[Q_{zz}(t,T)>0.3].
\]

This threshold is an empirical classification, not a microscopic phase definition. Varying it to 0.2 or 0.4 leaves the initial/final branch counts and number/direction of inventory changes unchanged. The alternative classifier \(|m_\chi|>0.1\) agrees with it in all 9,600,000 measured configurations per size apart from 1, 0, 1 and 2 samples, respectively. These exceptional points occur during the rapid conversions. The low-Q population is therefore almost nonchiral, while the high-Q population has large chirality; a pure spin rotation of the same chiral state cannot account for their difference.

PT exchanges only permute configurations among temperatures, so the total branch inventory

\[
B(t)=\sum_{k=1}^{160} b(t,T_k)
\]

is unchanged by an exchange. Its change directly detects branch conversion by the local dynamics, independently of apparent switching at a fixed temperature. During all 3,000,000 production sweeps:

| L | Initial high-Q inventory | Final inventory | Physical walker high→low conversions | Physical walker low→high conversions |
|---:|---:|---:|---:|---:|
|64|33|31|2|0|
|96|44|43|1|0|
|128|50|49|1|0|
|160|55|53|2|0|

Every measured inventory change is a loss; there are no observed reversals. The 500-sweep history snapshots identify exactly the same six losses in the physical walker coordinates. They belong to walkers 35 and 15 (L64), 7 (L96), 10 (L128), and 17 and 30 (L160). At those snapshots each initially high-Q walker remains high-Q until its one observed conversion; the remaining physical walkers retain their branch labels throughout production. Short unresolved excursions between saved measurements cannot be ruled out. Conversion is bracketed by the following exact recorded sweeps:

|L|Walker|Last high-Q history sweep|First low-Q history sweep|Temperatures at those two snapshots|
|---:|---:|---:|---:|---:|
|64|35|2,994,500|2,995,000|0.389379 → 0.466753|
|64|15|3,400,500|3,401,000|0.422881 → 0.470000|
|96|7|3,834,500|3,835,000|0.396135 → 0.470000|
|128|10|3,178,500|3,179,000|0.404589 → 0.465146|
|160|17|2,064,000|2,064,500|0.381572 → 0.461965|
|160|30|2,084,000|2,084,500|0.377781 → 0.438065|

The precise crossing of the Q=0.3 threshold can be located to 50 sweeps using total inventory, but its physical walker identity is established only at the saved 500-sweep snapshot times. The temperatures above bracket the conversion; they do not locate its instantaneous temperature. A walker can make a rapid upward excursion while the order parameter relaxes.

![Irreversible branch inventories](results/strong_branch_inventory.png)

These observations directly invalidate interpreting fixed-temperature branch switching as repeated equilibrium transformations. Most switching is exchange of nearly immutable branch-labelled configurations. The PT endpoint analysis independently finds only cold→hot endpoint passages, no hot→cold ones, and zero completed round trips; the branch loss counts also equal the numbers of production cold→hot passages, although a branch conversion need not occur at the endpoint-passage sweep. The L128 conversion, for example, precedes its eventual endpoint passage.

This is strong evidence of slow branch equilibration and a persistent memory of the existing branch population. It does **not** prove that one branch is unphysical, that a transition is absent, or that the limiting equilibrium branch weights are known. Equilibrium coexistence requires stationary weights and reversible exploration, neither of which is established here. There is only one production trajectory per size; a within-run standard error cannot bound independent-run or initialization dependence.

## 2. The apparent spin/chirality anomaly moves with size and with branch population

The largest adjacent-temperature step in mean Q occurs in these brackets. These are descriptive locations of a sampled front, not equilibrium transition estimates:

|L|Largest mean-Q step bracket|
|---:|---:|
|64|0.355607–0.356420|
|96|0.364834–0.365461|
|128|0.367933–0.368546|
|160|0.370379–0.370990|

At the saved temperature closest to 50% branch occupancy (the narrow L160 front jumps over 50%), the conditional populations have:

|L|T|High fraction, first half → second half|Q high / low|Density high / low|Energy/site high / low|\(\langle|m_\chi|\rangle\) high / low|
|---:|---:|---:|---:|---:|---:|---:|
|64|0.356420|0.68317 → 0.35457|0.64608 / −0.28038|1.08744 / 1.04670|0.81743 / 0.84518|0.67897 / 0.00638|
|96|0.364834|0.61803 → 0.39943|0.62528 / −0.27109|1.06836 / 1.03480|0.89065 / 0.91003|0.65739 / 0.00443|
|128|0.368546|0.59420 → 0.28283|0.61524 / −0.26680|1.05930 / 1.02933|0.92401 / 0.93896|0.64676 / 0.00335|
|160|0.370990|0.25760 → 0.22297|0.60837 / −0.26395|1.05336 / 1.02570|0.94629 / 0.95811|0.63948 / 0.00272|

The conditional numbers characterize the recorded populations, with no equilibrium uncertainty assigned. The large first/second-half changes at L64–128 are far larger than short-block errors and coincide with slow depletion of branch-labelled walkers. At L160 most depletion happens near the start of production, so half agreement is much better despite the same absence of reverse conversion. This is an example of a half-split check missing a persistent equilibration problem.

Joint contiguous-block jackknife confirms the limitation of a short block. At the listed L64 point, the branch-fraction standard error grows from 0.01084 at 200 samples/block to 0.02411, 0.05473 and 0.08062 at 1,000, 5,000 and 10,000 samples/block. The L128 sequence is 0.01141, 0.02521, 0.05660, 0.07976. No plateau is established before only six blocks remain. These are diagnostics of the observed trajectory; nonstationary drift means even the larger error is not an equilibrium error bound. Each sample represents 50 sweeps.

![Separated configuration populations](results/strong_joint_branches.png)

The Q–chirality correlations at the representative mixed temperatures are 0.99984, 0.99992, 0.99995 and 0.99995. These correlations mostly reflect the two-population mixture, not an independently established continuous spin–chirality relation. Energy conditional distributions overlap substantially even though their means differ. The density and chirality gaps are much more visually separated. A planar or otherwise nonchiral competitor is a promising interpretation of the low-Q branch, but Q, global chirality and density do not identify its full order-parameter tensor. Local spin/orbital invariants or configurations are required to distinguish a planar state, textures, and spatially disordered chiral domains. The Q estimator is density weighted and is not a local director measurement.

## 3. The enormous negative Binder minima are almost entirely sparse chiral intrusions

The saved estimator is \(U_\chi=1-\langle m_\chi^4\rangle/(3\langle m_\chi^2\rangle^2)\), with \(m_\chi=C/D\) evaluated separately in each configuration. Searching the full temperature ladder gives:

|L|T at minimum|U|\(\langle m^2\rangle\)|\(\langle m^4\rangle\)|\(|m|>0.1\) samples / 60,000|Fraction of fourth moment in these samples|Binder after deleting those samples|
|---:|---:|---:|---:|---:|---:|---:|---:|
|64|0.368546|−485.20|1.63855e−4|3.91612e−5|13|99.9606%|+0.00244|
|96|0.373441|−1008.53|6.66447e−5|1.34515e−5|5|99.9753%|−0.00674|
|128|0.375291|−1662.87|3.20815e−5|5.13749e−6|2|99.9789%|+0.00114|
|160|0.377781|−2548.21|2.51827e−5|4.84990e−6|2|99.9903%|−0.01087|

Deletion is an influence diagnostic, **not** a corrected equilibrium estimator: the large-chirality configurations are real measured samples and should not simply be discarded from thermodynamic averages. Their equilibrium probability is precisely what is not known. The negative minimum is a mathematically valid value of the moment ratio for the recorded mixture. It is unsafe to interpret its size or growth with L as first-order finite-size scaling.

At L128 and L160 both large events occur in the first half. The first/second-half Binders are −1673.59/−0.00108 and −2242.39/−0.02342, respectively. At L160 both rare samples occupy a single 200-sample time block, so the 200-sample block-jackknife standard error is about 2540, comparable to the magnitude 2548 itself. A distribution sampled only twice does not support a Gaussian uncertainty or a reliable tail probability. The minimum was selected after inspecting all temperatures, so an error at that selected temperature also omits selection effects.

A simple mixture explains the sensitivity. A nearly Gaussian background of variance \(\sigma^2\) plus sparse chiral records of magnitude a and fraction p has \(\langle m^2\rangle\simeq(1-p)\sigma^2+pa^2\) and \(\langle m^4\rangle\simeq3(1-p)\sigma^4+pa^4\). Very small p can dominate the fourth moment while contributing only a comparable amount to the second. Narrower background fluctuations at larger L can then deepen a Binder minimum without determining a transition order. Here the actual branch inventory also drifts, so p cannot be assumed to be an equilibrium mixing weight.

![Moment concentration and raw rare events](results/strong_binder_rare_events.png)

The reproducible tables include every rare sample's sweep, Q, chirality, energy, density and currents (`strong_rare_events.csv`), as well as deletion of the largest 1, 2, 5, 10 and 20 records. The results survive changing the branch cutoff; they do not rely on a visual histogram bin choice.

## 4. Strong stiffness persists in the low-Q nearly nonchiral population

The raw isotropic helicity at T=0.442331 is 1.19391, 1.17989, 1.18561 and 1.18607 for L64,96,128,160, respectively. There are no high-Q samples at this temperature in any size. At T=0.388056 the values are 2.19667, 2.20194, 2.19262 and 2.18802; only one of the 60,000 L128 records is high-Q and all other sizes have none. Thus the high-temperature stiffness feature around the provisional WM value 0.443 is measured in a different configuration population from the low-temperature chiral high-Q branch.

This is a physical distinction that a single WM temperature misses. It raises the possibility of a superfluid but globally almost nonchiral regime, but the existing sampling cannot establish an equilibrium phase sequence. In particular, it does not identify an elementary vortex charge, since the global U(1) stiffness is not a direct defect measurement and the full order-parameter manifold of the low-Q branch is unresolved.

The \(\langle I_\mu\rangle^2/(TN)\) contribution is only approximately 10^−7–1.5×10^−4 at T=0.442331. Neglecting versus retaining the sampled mean current therefore cannot explain the approximately 1.18 stiffness there. Directional stiffnesses are comparable and positive at the representative temperatures; this diagnostic does not suggest a large sampled-current offset or one-direction artifact. This does not independently validate the production twist implementation, for which a separate code/formula audit is needed.

Near the low-temperature front, branch-conditioned stiffnesses are both substantial. At the tabulated representative temperatures the high/low conditional values are 2.825/2.623, 2.676/2.520, 2.601/2.464 and 2.568/2.434. Conditional estimators are diagnostic because selecting a branch changes the ensemble; they should not be presented as equilibrium phase stiffnesses or used without its selection-boundary response. The full mixture's current variance obeys the usual within/between-population decomposition. Differences of conditional mean currents here contribute only small corrections and do not generate the main stiffness gap.

![Helicity and current-mean contribution](results/strong_helicity_components.png)

In this figure conditional curves are shown only where a branch has at least 0.5% of the recorded samples. Sparse conditional curves outside that range would have artificially small current variance and cannot be physically interpreted. `strong_temperature.csv` retains all raw conditional estimates and occupancies to make this limitation auditable.

## 5. Most informative follow-up

1. **Distinguish the competing configuration populations before assigning a transition.** Save or reconstruct local order-parameter tensors and compare unitary/nonunitary and chiral/planar invariants, spatial chirality, spin alignment tensor eigenvalues, and domain structure. The present global channels cannot identify the low-Q manifold or distinguish a uniform nonchiral state from textured/domain configurations. Different outcomes change the relevant topological defects and the meaning of a helicity jump.
2. **Test reversibility of the branch weights using targeted independent starts.** Short, scientifically designed runs started in identified high-Q chiral and low-Q candidate states at a few temperatures around the anomaly would directly test whether both preparations converge and whether conversions occur in both directions. Current runs show only losses of an initial branch inventory, so simply reducing within-run errors would not answer this question. No new runs were launched here.
3. **Inspect escaping physical walkers and the absence of branch regeneration.** Save sufficient configurations/invariants through a few branch conversion paths and compare their free-energy barriers. The existing 500-sweep histories bracket six loss events but do not resolve whether a texture, domain wall, amplitude collapse, or another route causes conversion.
4. **Interpret the high-temperature stiffness in the identified low-Q manifold.** Direct circulation/spin-texture and defect statistics, including interactions under imposed separated defects, are needed to decide whether HQVs, integer vortices, or another composite excitation controls it. The current thermodynamic measurements cannot demonstrate SOC confinement.

All of these are higher priority than fitting an equilibrium latent heat, first-order exponent, or critical temperature to the present mixed trajectories.
