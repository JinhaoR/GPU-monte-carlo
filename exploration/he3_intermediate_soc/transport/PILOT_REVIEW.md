# Independent review of the 15 bounded local pilots

All 15 completed pilots were reviewed without new Monte Carlo. The independent script [review_pilots.py](review_pilots.py) reads the saved fields/mappings and the preparation-resolved analysis, recomputes production-only physical conversions, and compares preparation means, production halves, block sizes and empirical distributions. It does not pool different preparations into an equilibrium average.

## Scope and uncertainty

The three L=8,g_D=.1 histories each contain 16k total sweeps, of which the last 12k enter the analysis. At L=16 the four couplings .04,.06,.08,.1 each have three independent preparations, 32k total sweeps and 24k analyzed sweeps. Each uses the same 24 temperatures .28–.65. Scalar events are every20 sweeps and spatial events every100. The initializations and independent seeds are retained in the pilot JSONs.

The supplied full-production errors use 12 joint blocks: 1k sweeps at L8 and 2k at L16. Half-history estimates use12 blocks within each half. The review also groups full-history blocks into6 and4 blocks, reaching3k at L8 and6k at L16. Reported comparisons divide differences by the quadrature sum of conditional errors. These are diagnostics of finite sampled histories, not calibrated significance tests under metastability. Repeated temperatures/channels are correlated and no multiple-testing claim is made. Empirical CDF distances are reported without iid Kolmogorov–Smirnov p-values.

Mapping checks independently verify that the inverse saved temperature-to-walker arrays equal the physical-walker histories at the exact same sweeps. Production-only conversion counting starts afresh after burn: both accepted structural visits must occur wholly within production. No initial basin label is carried from burn. A chiral-like visit requires s_z>.25 and |mχ|>.25; a low-Q visit requires s_z<−.1 and |mχ|<.15. These are structural descriptors, not proof of local ideal chirality or a phase assignment. Results for100,200,400-sweep persistence are retained; the table below uses400.

## Conversions are possible on L8, but their weights are not yet stationary

At L8,g_D=.1 the hot and planar preparations show both conversion directions during production. This establishes that the unchanged local sampler can regenerate a chiral-like structure on the small box with the extended ladder. It is a meaningful positive result, but few long visits do not establish equilibrium weights.

For example, at T=.28 the planar-start history changes between halves from s_z=−.3172±.0034 to +.1426±.0832, and |mχ|=.0402±.0017 to .3739±.0578. The corresponding shifts are5.5 and5.8 diagnostic errors. Its full-history s_z error grows .077→.138 when blocks increase from1k to3k sweeps. The T=.376522 distributions agree much better across preparations: maximum empirical CDF differences are .040 for s_z, .053 for |mχ|, and .070 for energy/site. Thus successful conversion and regional agreement coexist with poor cold-region stationarity.

## A cold preparation trap is already present at g_D=.06 on L16

At T=.28, means from the three independent starts are:

| g_D | s_z: chiral / hot / planar start | Y: chiral / hot / planar start | Assessment |
|---:|---|---|---|
|.04|.7373 / .7332 / .7317|3.669 / 3.657 / 3.641|Good cold agreement in the sampled histories|
|.06|.7514 / .7542 / −.3198|3.715 / 3.715 / 3.155|Planar preparation remains in a different structure|
|.08|.7627 / .7672 / −.3378|3.786 / 3.773 / 3.341|Same strong preparation dependence|
|.10|.7720 / −.3455 / −.3463|3.855 / 3.456 / 3.445|Hot and planar preparations agree with each other, not with chiral start|

The spin/chirality empirical CDF distance is exactly1 between the trapped cold populations for each coupling .06,.08,.1; their sampled supports do not overlap. These gaps are far larger than any within-history error. Cold production halves can nevertheless agree extremely well: at .08 the largest s_z half difference is only .31 conditional errors. Stationarity inside one basin is not equilibration between basins.

The cold planar-like histories have local antisymmetric-planar weights .743,.766,.777 for .06,.08,.1, versus ≈.04 in the competing chiral histories. At .1, the coherent antisymmetric spin–orbital projection has S(0)≈212 while the z-spin plus-orbital projection has S(0)≈1.4–1.6. This is a concrete structural distinction, not just a change in spin alignment.

The onset of this **finite-time, finite-size preparation trap** is bracketed between the tested .04 and.06 under this particular protocol. It is not a thermodynamic transition or a coupling where a bare onsite minimum first appears.

## Region-local agreement is stronger above the cold trap

At the common T=.376522 probe:

| g_D | s_z: chiral / hot / planar start | Maximum preparation CDF distance in s_z | Interpretation |
|---:|---|---:|---|
|.04|.304±.063 / .352±.043 / .438±.023|.219|Broad uncertainty and different sampled mixtures near the feature|
|.06|−.078±.060 / −.203±.003 / −.201±.001|.174|Chiral start has a visibly evolving mixture|
|.08|−.229±.003 / −.236±.001 / −.236±.002|.066|Useful agreement in the planar-like region|
|.10|−.249±.003 / −.252±.001 / −.253±.001|.039|Useful agreement in the planar-like region|

The .04 means differ by at most2 conditional errors; that alone is not a decisive preparation discrepancy. Nevertheless the box has only2–4 full-run PT loops and very little persistent production structural conversion. Near-feature equilibrium is not certified by those broad overlapping errors.

The .06 chiral-start evolution is more concrete. At this probe, s_z changes −.1979±.0040→+.0417±.0797 and |mχ| .0397±.0028→.2224±.0597 between halves. Energy and planar weight move coherently by about3 diagnostic errors. Its full s_z error grows .0595→.1096 with2k→6k blocks. Meanwhile Y remains near1.98. This is an example where helicity agreement conceals changing structural weights.

At .08, Y=2.167±.023,2.166±.026,2.178±.024, with local planar weights≈.60 and planar-projection S(0)=128–131 versus z-plus S(0)=2.5–3.2. At .1, Y=2.410±.020,2.392±.011,2.358±.026 and the corresponding S(0) values are≈142 versus2.3–2.5. These are convincing finite-box descriptions of a coherent planar-like channel sampled from different starts at that temperature. They do not prove that the complete ladder samples correct basin weights, that correlations are algebraic as L→∞, or that this region belongs to a thermodynamic intermediate phase.

## Round trips and structural exploration answer different questions

Each entry below gives production-only chiral-like→low-Q / low-Q→chiral-like persistent visits. The adjacent loop counts are exact but include burn.

| L,g_D | Chiral start: visits; loops | Hot start: visits; loops | Planar start: visits; loops |
|---|---|---|---|
|8,.1|0/2;34|2/2;31|2/3;43|
|16,.04|0/1;2|0/0;4|0/0;3|
|16,.06|0/2;0|0/1;0|0/0;14|
|16,.08|1/0;0|0/0;0|0/0;9|
|16,.1|1/0;0|0/0;14|0/0;9|

Most importantly, the L16 planar starts at .06,.08,.1 complete14,9,9 PT loops while producing no400-sweep persistent structural conversion in production. At .1 the hot start similarly has14 loops and no such conversion. The low-basin configurations can traverse the entire hotter/broader ladder without regenerating the competing structure. Conversely, the .06 chiral/hot runs show some low→chiral-like visits while completing no full PT loops. Neither count can replace the other.

![Production-only chiral-like inventories, averaged over500 sweeps. Legend loop counts include burn; the structural curves do not.](pilot_inventories.png)

At .08 the chiral, hot and planar starts retain inventories near4,2,0. At .1 the chiral start retains about3 while hot and planar remain0. At .06 the chiral-start inventory grows from≈5.05 in the first half to≈6.38 in the second, consistent with its changing fixed-temperature mixture. These patterns directly parallel the old large-box inventory problem even though the new hot endpoint is .65. Since the pilots also differ in size, duration and temperature placement from the old campaign, this is not a controlled estimate of the effect of raising the endpoint alone.

## Smallest useful next investigation

The most promising equilibrium-oriented next couplings are .02 and.04 at L8 and16, with the same three preparations. The cold .04 result is encouraging; near-feature mixing, distribution agreement and block stability should be demonstrated before increasing L. A targeted .05 case would refine the observed trapping bracket between .04 and.06, but that bracket describes algorithmic accessibility on these boxes and times, not a phase boundary.

The current24 uniformly spaced temperatures give minimum acceptances as low as .019 on L16. A modest, data-informed redistribution of a bounded ladder around the observed features can improve temperature transport, but cannot be assumed to fix structure conversion. Keep a sufficiently hot region to test forgetting, monitor both directions in physical configurations, and regard .06 or .1 as failure controls rather than production-ready targets. Expansion to L24 is justified only after the smaller boxes show stable preparation-independent distributions and repeated persistent conversions where both structures have appreciable probability. No larger simulation or new implementation is authorized by this review.

Tables: [preparation comparisons](pilot_preparation_comparisons.csv), [production halves](pilot_half_comparisons.csv), [production-only walkers](pilot_production_walkers.csv), [block sensitivity](pilot_block_sensitivity.csv), and [empirical distribution comparisons](pilot_distribution_comparisons.csv). They retain all prespecified cold/intermediate/hot probes; the narrative highlights the physically discriminating cases instead of treating every isolated two-error discrepancy as evidence.
