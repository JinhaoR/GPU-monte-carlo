"""Read old L160 data and design ONE conservative L256 ladder. No simulation.

This offline calculation does not change run.py or the ladder implementation.
Old per-coupling text files/results remain historical evidence, not run inputs.
"""
import csv
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run as settings
from utils.temperature_ladder import make_temperature_ladder

COUPLINGS = (0., .001, .01)
GRID = np.linspace(.365, .425, 60001)
CRITICAL = (.380, .388)
TARGET = .65  # Delta(beta)*sigma_H; same target as the validated L64 pilot


def load_fluctuations():
    curves, sources = [], []
    inventory = json.loads((HERE/'results/inventory.json').read_text())
    for gd in COUPLINGS:
        path = ROOT/f'data/he3/P=20_gd={gd:g}_gaussian/L=160.npz'
        previous = next(r for r in inventory if r['g_d'] == gd and r['L'] == 160)
        with zipfile.ZipFile(path) as z:
            crc = ';'.join(f'{i.filename}:{i.CRC:08x}' for i in z.infolist())
        assert crc == previous['crc32'], 'Old source changed; reassess the cached curves'
        with np.load(path) as z:
            meta = json.loads(z['manifest'].item())['runs']['160']['metadata']
            energy = z['L160_energy']
        T = np.asarray(meta['temperatures'])
        raw_cv = energy.var(axis=0)/(160**2*T**2)
        del energy
        block_path = ROOT/f'exploration/he3_soc/results/blocks/gd={gd:g}_L=160.npz'
        with np.load(block_path) as z:
            np.testing.assert_array_equal(T, z['T'])
            e, e2 = z['e'], z['e2']  # energy density, 600 blocks of 5000 sweeps
        cv = (e2.mean(0)-e.mean(0)**2)*160**2/T**2
        np.testing.assert_allclose(cv, raw_cv, rtol=1e-7, atol=1e-8)
        errors = []
        for width in (20, 40, 60):  # 100k, 200k, 300k sweeps; match reweighted checks
            a = e.reshape(-1, width, len(T)).mean(1)
            b = e2.reshape(-1, width, len(T)).mean(1)
            B = len(a)
            deleted = ((b.sum(0)-b)/(B-1)-((a.sum(0)-a)/(B-1))**2)*160**2/T**2
            errors.append(np.sqrt((B-1)*deleted.var(0)))
        upper = np.interp(GRID, T, cv+2*np.maximum.reduce(errors))
        fine_path = ROOT/f'exploration/he3_transition_study/results/curves_gd={gd:g}_L=160.npz'
        with np.load(fine_path) as z:
            F = z['T']; i = list(z['keys']).index('specific_heat'); fine = z['point'][i]
            se = np.maximum.reduce([np.sqrt((len(z[k])-1)*z[k][:, i].var(0))
                                    for k in ('jk', 'jk4000', 'jk6000')])
            fine_upper = np.maximum(fine+2*se, z['variants'][:, i].max(0))
        inside = (GRID >= F[0]) & (GRID <= F[-1])
        upper[inside] = np.maximum(upper[inside], np.interp(GRID[inside], F, fine_upper))
        # Cover a shifted/narrower dip anywhere in the entire requested region.
        # Pad by one old critical-grid gap, rounded UP to 0.0001 for readability.
        gaps = np.diff(T)[(T[:-1] >= .379) & (T[1:] <= .389)]
        guard = math.ceil(gaps.max()*10000)/10000
        peak = upper[(GRID >= CRITICAL[0]) & (GRID <= CRITICAL[1])].max()
        upper[(GRID >= CRITICAL[0]-guard) & (GRID <= CRITICAL[1]+guard)] = peak
        # Area growth exceeds the observed peak growth. This is a stress envelope,
        # not an asserted transition exponent or a rigorous upper bound.
        upper256 = upper * 1.10 * (256/160)**2
        curves.append(upper256)
        sources.append(dict(g_d=gd, raw_source=str(path.relative_to(ROOT)),
            block_source=str(block_path.relative_to(ROOT)), fine_source=str(fine_path.relative_to(ROOT)),
            raw_cv_peak=float(cv.max()), reweighted_cv_peak=float(fine.max()),
            peak_temperature=float(F[fine.argmax()]), upper160_peak=float(peak),
            upper256_peak=float(upper256.max()), critical_guard=guard,
            old_min_swap=previous['min_acceptance'], old_min_histogram_overlap=previous['min_overlap'],
            old_mean_completed_round_trip=previous['mean_completed_trip_sweeps'],
            raw_cache_max_cv_difference=float(abs(cv-raw_cv).max())))
    return np.asarray(curves), sources


def edge_metrics(temperatures, sigma):
    """Maximum sigma over EACH full edge, including interior grid points."""
    starts = np.searchsorted(GRID, temperatures, side='left')
    assert np.all(np.diff(starts) > 0)
    maxima = np.maximum.reduceat(sigma, starts[:-1])
    endpoints = np.interp(temperatures, GRID, sigma)
    maxima = np.maximum(maxima, np.maximum(endpoints[:-1], endpoints[1:]))
    return np.diff(-1/temperatures)*maxima


def choose_ladder(curves):
    sigma = GRID*256*np.sqrt(curves.max(0))
    choices = []
    # Center the protected critical interval; search the existing function's
    # shoulder width and density allocation, using the full measured envelope.
    for width in np.round(np.arange(.010, .0201, .0005), 4):
        for fraction in np.round(np.arange(.20, .651, .01), 2):
            p = dict(T_min=.365, T_max=.425, n_T=401, Tc=.384,
                     tc_window=float(width), tc_fraction=float(fraction))
            try:
                trial = make_temperature_ladder(**p)
            except ValueError:
                continue
            guess = math.ceil(400*edge_metrics(trial, sigma).max()/TARGET)+1
            # Recheck actual edges; inverse-replica scaling is only a search seed.
            for n in range(max(2, guess-4), guess+8):
                p['n_T'] = n
                metric = edge_metrics(make_temperature_ladder(**p), sigma).max()
                if metric <= TARGET:
                    choices.append((n, float(metric), p.copy()))
                    break
            else:
                raise AssertionError('Expand the bounded offline parameter search')
    # Fewest replicas; among ties take the largest overlap margin.
    n, metric, parameters = min(choices, key=lambda row: (row[0], row[1]))
    return parameters, sorted(choices, key=lambda row: (row[0], row[1]))[:8]


def resource_estimates(R):
    rows = []
    burn, production = settings.thermalization_sweeps, settings.measurement_sweeps
    scalar = production//settings.sample_interval
    spatial = production//settings.spatial_interval
    profiles = production//settings.realspace_interval if settings.realspace_interval else 0
    history = (burn+production)//(settings.exchange_interval*settings.history_stride)+1
    history_gpu = math.ceil(history/1024)*1024*R*4
    # Existing L128/R47 loop: local updates + PT every5 + globals every50.
    with (ROOT/'exploration/he3_spatial_pilot/results/benchmark.csv').open() as f:
        old = list(csv.DictReader(f))
    base = float(next(x for x in old if x['L'] == '128')['baseline_us_per_sweep'])*1e-6
    timings = json.loads((HERE/'results/benchmark.json').read_text())['timings']['256']
    for L in (64, 96, 128, 256):
        sites = R*L*L
        meter = 16*L + 8*R*((3*52+68)*math.ceil(L*L/1024)+26*3*2+68)+4*R
        # State48, site RNG16, Gaussian counters24 bytes/site; small PT arrays included.
        gpu = 88*sites + meter + history_gpu + 104*R
        scalar_bytes = scalar*R*10*8
        spatial_bytes = spatial*R*(16*5*16+68*8+4+8)
        profile_bytes = profiles*5*(7*2*(L//2+1)*16+7*16+4+8)
        raw = scalar_bytes+spatial_bytes+profile_bytes
        peak_arrays = 2*scalar_bytes+spatial_bytes+profile_bytes+history*R*8
        # Site-work scaling only; no new GPU benchmark or simulation.
        scale = (R/47)*(L/128)**2
        # Conservatively add ALL current observations to the older baseline.
        measurement_scale = (R/253)*(L/256)**2
        globals_s = timings['old_globals_cached']['median_seconds']*measurement_scale
        spatial_s = (timings['low_modes_host']['median_seconds']+
                     timings['structure_energy_host']['median_seconds'])*measurement_scale
        profile_s = (timings['selected_field_transfer']['median_seconds']+
                     timings['seven_CPU_FFTs']['median_seconds'])*(5/7)*(L/256)**2
        seconds = (burn+production)*base*scale + scalar*globals_s + spatial*spatial_s + profiles*profile_s
        rows.append(dict(L=L, replicas=R, model_sites_per_sweep=sites,
            state_GiB=48*sites/2**30, estimated_live_GPU_GiB=gpu/2**30,
            meter_MiB=meter/2**20, PT_history_GPU_MiB=history_gpu/2**20,
            raw_arrays_GiB=raw/2**30, PT_history_logical_GiB=history*R*4/2**30,
            array_stacking_peak_GiB=peak_arrays/2**30,
            estimated_loop_hours=seconds/3600,
            sweeps=burn+production, exchange_rounds=(burn+production)//settings.exchange_interval))
    return rows


def main():
    curves, sources = load_fluctuations()
    parameters, candidates = choose_ladder(curves)
    temperatures = make_temperature_ladder(**parameters)
    assert np.all(np.diff(temperatures) > 0)
    assert np.all(np.diff((1/temperatures).astype(np.float32)) < 0)
    np.savetxt(HERE/'ladders/common_L256.txt', temperatures, fmt='%.17g',
               header='One common ladder: all g_D=0,.001,.01 and L=64,96,128,256')
    edges, summary, metrics = [], [], []
    middle = (temperatures[:-1]+temperatures[1:])/2
    for source, cv in zip(sources, curves):
        q = edge_metrics(temperatures, GRID*256*np.sqrt(cv))
        assert q.max() <= TARGET
        metrics.append(q)
        overlaps = np.array([math.erfc(d/(2*np.sqrt(2))) for d in q])
        acceptance = np.array([math.erfc(d/2) for d in q])
        critical = (temperatures[:-1] <= CRITICAL[1]) & (temperatures[1:] >= CRITICAL[0])
        worst = q.argmax()
        summary.append(dict(g_d=source['g_d'], maximum_delta_beta_sigma=float(q.max()),
            minimum_Gaussian_overlap=float(overlaps.min()), minimum_Gaussian_swap=float(acceptance.min()),
            critical_minimum_Gaussian_overlap=float(overlaps[critical].min()),
            critical_minimum_Gaussian_swap=float(acceptance[critical].min()),
            worst_edge=[float(temperatures[worst]), float(temperatures[worst+1])]))
        for i in range(len(q)):
            edges.append(dict(g_d=source['g_d'], edge=i, T_left=temperatures[i], T_right=temperatures[i+1],
                spacing=temperatures[i+1]-temperatures[i], delta_beta_sigma=q[i],
                Gaussian_overlap=overlaps[i], Gaussian_swap=acceptance[i]))
    with (HERE/'results/common_ladder_edges.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(edges[0]));writer.writeheader();writer.writerows(edges)
    resources = resource_estimates(len(temperatures))
    result = dict(parameters=parameters, minimum_spacing=float(np.diff(temperatures).min()),
        maximum_spacing=float(np.diff(temperatures).max()),
        temperatures_in_critical_region=int(np.sum((temperatures >= .380) & (temperatures <= .388))),
        target_delta_beta_sigma=TARGET, comparison_size=256, grid_spacing=float(GRID[1]-GRID[0]),
        Cv_growth_exponent=2, extra_Cv_margin=1.10, sources=sources, predictions=summary,
        best_grid_search_candidates=[dict(parameters=p, maximum_delta_beta_sigma=m) for n,m,p in candidates],
        resource_estimates=resources, current_thermalization=settings.thermalization_sweeps,
        current_production=settings.measurement_sweeps, simulations_run=0)
    (HERE/'results/common_ladder.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for source, cv, q in zip(sources, curves, metrics):
        label=f"g_D={source['g_d']:g}"
        axes[0,0].plot(GRID,cv,label=label)
        overlap=[math.erfc(d/(2*np.sqrt(2))) for d in q]
        axes[1,0].plot(middle,overlap,label=label)
        axes[1,1].plot(middle,overlap,label=label)
    axes[0,0].set(ylabel='L256 Cv stress envelope',ylim=(0,None))
    axes[0,1].plot(middle,np.diff(temperatures));axes[0,1].set(ylabel='Temperature spacing')
    axes[1,0].set(ylabel='Predicted Gaussian energy overlap',ylim=(.7,1))
    axes[1,1].set(xlim=(.379,.389),ylabel='Predicted overlap: critical region',ylim=(.7,1))
    for ax in axes.flat:
        ax.axvspan(.380,.388,color='grey',alpha=.12);ax.set_xlabel('T');ax.grid(alpha=.2)
    axes[0,0].legend();axes[1,0].legend()
    fig.suptitle(f'Common {len(temperatures)}-temperature ladder: predictions, no L256 simulation')
    fig.savefig(HERE/'results/common_ladder.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(parameters=parameters,minimum_spacing=result['minimum_spacing'],
                         predictions=summary,resources=resources),indent=2))


if __name__ == '__main__':
    main()
