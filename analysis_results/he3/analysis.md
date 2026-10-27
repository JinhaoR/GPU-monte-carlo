# Full Thin-Film He3 Analysis

**Data:** `data/he3_run.npz`
**System sizes:** [32, 48, 64, 80]
**Block size:** 200
**Vortex charge hypothesis:** $q = 0.5$

## Weber–Minnhagen Fit

- Critical temperature: $T_{BKT} = 0.381283$
- Correction constant: $C = -4.641613$
- Reduced chi-square: 10.043

The helicity modulus is rescaled using:

$$
\widetilde{\Upsilon}_L(T) = \frac{\Upsilon_L(T)}{1 + 1/(2\ln L+C)}
$$

The corresponding candidate universal jump is:

$$
\Upsilon(T_{BKT}^-) = \frac{2T_{BKT}}{\pi (0.5)^2}
$$

The WM fit is exploratory. The chirality Binder cumulants are inspected visually without fitting their crossings.

## Binder Chirality

![binder_chirality](./binder_chirality.png)

## Binder Phase

![binder_phase](./binder_phase.png)

## Helicity

![helicity](./helicity.png)

## Density

![density](./density.png)

## Parallel Tempering

![parallel_tempering](./parallel_tempering.png)

## Parallel Tempering Summary

```text
Parallel tempering diagnostics (includes thermalization)
L=32: 1495377/1770000 accepted, 30 trips, mean 342392.0 sweeps/trip
L=48: 1365070/1770000 accepted, 1 trips, mean 354080.0 sweeps/trip
L=64: 1237626/1770000 accepted, 0 trips, mean None sweeps/trip
L=80: 1114123/1770000 accepted, 0 trips, mean None sweeps/trip
```
