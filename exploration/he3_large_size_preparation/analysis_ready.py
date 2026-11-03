"""Small analysis functions for the new archives, reusing existing estimators.

No automatic phase classification. Different sizes/preparations are resampled
independently; all temperatures and channels of one run share time blocks.
"""
import numpy as np
from models.he3.spatial import charged_amplitudes, contracted_components, correlation_ratio
from utils.Analysis.statistics import bootstrap_block_means
from utils.Analysis.crossings import linear_crossings, bootstrap_crossings
from utils.Analysis.bkt import fit_weber_minnhagen, fit_pseudocritical_bkt


def endpoint_flow(run):
    """Flow at stored times, using exact last-visited labels (not inferred visits).

    Position/duration resolution is still the saved stride; unknown walkers are
    counted separately. This diagnostic alone is not an equilibration test.
    """
    meta = run['metadata']
    history = meta['parallel_tempering']['walker_history']
    select = np.asarray(history['sweeps']) > meta['thermalization_sweeps']
    positions = np.asarray(history['temperature_indices'])[select].ravel()
    labels = np.asarray(history['last_endpoint'])[select].ravel()
    R = len(meta['temperatures'])
    cold = np.bincount(positions[labels==0], minlength=R)
    hot = np.bincount(positions[labels==1], minlength=R)
    unknown = np.bincount(positions[labels==-1], minlength=R)
    fraction = np.divide(cold, cold+hot, out=np.full(R,np.nan), where=cold+hot>0)
    return dict(cold_fraction=fraction, cold=cold, hot=hot, unknown=unknown)


def transition_curves(run, block_sweeps=100_000, n_bootstrap=400, seed=None):
    s, meta = run['samples'], run['metadata']
    T, L = np.asarray(meta['temperatures']), meta['L']
    burn = meta['thermalization_sweeps']
    completed = meta.get('completed_sweeps', burn+meta['measurement_sweeps'])
    B = (completed-burn)//block_sweeps
    if B < 4:
        raise ValueError('Fewer than four complete blocks; uncertainty is not assessable')
    moments = {}
    def add(name, values, sweeps):
        ids = (np.asarray(sweeps)-burn-1)//block_sweeps
        if any(not np.any(ids == b) for b in range(B)):
            raise ValueError('A block has no measurements at this cadence')
        moments[name] = np.stack([values[ids == b].mean(0) for b in range(B)])
    coords = meta['sample_sweeps']
    m = s['chirality']/s['density']
    for name, values in dict(m2=m**2, m4=m**4, ix=s['current_x'], iy=s['current_y'],
            ix2=s['current_x']**2, iy2=s['current_y']**2, ixiy=s['current_x']*s['current_y'],
            kx=s['curvature_x'], ky=s['curvature_y']).items():
        add(name, values, coords)
    if 'curvature_xy' in s:
        add('kxy', s['curvature_xy'], coords)
    spatial_names = []
    if 'spatial_amplitudes' in s:
        a = s['spatial_amplitudes']
        # Keep Q mean amplitudes so SOC-connected subtraction is inside each resample.
        add('qmean', a[..., 2:7, :], meta['spatial_sample_sweeps'])
        full = contracted_components(abs(a)**2)/L**2
        spectra = {name: full[..., i, :] for i, name in enumerate(('kappa','P','Q','B','A','Q_m0','Q_m1','Q_m2'))}
        for name, values in charged_amplitudes(a).items():
            spectra[name] = np.sum(abs(values)**2, axis=-2)/L**2
            if name.startswith('A_'):
                for j, spin in enumerate('xyz'):
                    spectra[name+'_'+spin] = abs(values[..., j, :])**2/L**2
        spatial_names = list(spectra)
        add('spectra', np.stack(list(spectra.values()), axis=-2), meta['spatial_sample_sweeps'])
    def estimate(m):
        hx = (m['kx']-(m['ix2']-m['ix']**2)/T)/L**2
        hy = (m['ky']-(m['iy2']-m['iy']**2)/T)/L**2
        out = dict(binder=1-m['m4']/(3*m['m2']**2), helicity=(hx+hy)/2, helicity_x=hx, helicity_y=hy)
        if 'kxy' in m:
            xy = (m['kxy']-(m['ixiy']-m['ix']*m['iy'])/T)/L**2
            det = hx*hy-xy**2
            out.update(helicity_xy=xy, determinant=det,
                       effective=np.sqrt(np.where((hx>0) & (hy>0) & (det>0), det, np.nan)))
        if 'spectra' in m:
            power = m['spectra']
            # Q is the third full sector, before extra orbital fields.
            qconnected = power[..., 2, :]-np.sum(abs(m['qmean'])**2, axis=-2)/L**2
            power = np.concatenate((power, qconnected[..., None, :]), axis=-2)
            ratio, xi, _ = correlation_ratio(power[..., 0], power[..., 1:].mean(-1), L)
            out.update(spectra=power, ratio=ratio, xi_over_L=xi)
        return out
    central = estimate({k: v.mean(0) for k,v in moments.items()})
    boot = estimate(bootstrap_block_means(moments, n_bootstrap, rng=seed))
    return dict(L=L, temperatures=T, blocks=B, block_sweeps=block_sweeps,
                spatial_channels=spatial_names+(['Q_connected'] if spatial_names else []),
                central=central, bootstrap=boot,
                errors={k: v.std(0, ddof=1) for k,v in boot.items()})


def crossing_summary(T, left, draws, right=0., right_draws=0., window=None):
    roots = linear_crossings(T, left, right, temperature_range=window)
    samples = bootstrap_crossings(T, draws, right_draws, temperature_range=window)
    valid = np.isfinite(samples)
    # Do not silently condition the error on successful roots; report failures.
    return dict(roots=roots, valid_fraction=float(valid.mean()),
                statistical_se=float(samples.std(ddof=1)) if valid.all() and len(roots)==1 and np.isfinite(roots[0]) else None,
                bootstrap=samples)


def direct_jump(curves, q, stiffness='helicity', window=None):
    if q not in (1., .5):
        raise ValueError('Compare the requested integer and half-charge hypotheses')
    T = curves['temperatures']
    return crossing_summary(T, curves['central'][stiffness], curves['bootstrap'][stiffness],
                            right=2*T/(np.pi*q*q), right_draws=2*T/(np.pi*q*q), window=window)


def wm_comparison(curves, q=.5, min_size=96, stiffness='helicity', window=None):
    selected = sorted((c for c in curves if c['L'] >= min_size and c['L'] != 64), key=lambda c: c['L'])
    if len(selected) < 3:
        raise ValueError('WM requires at least three sizes for positive degrees of freedom')
    T = selected[0]['temperatures']
    if any(not np.array_equal(c['temperatures'], T) for c in selected):
        raise ValueError('Use the existing transition interpolation/WHAM analysis for different ladders')
    mask = np.ones(len(T), bool) if window is None else (T>=window[0]) & (T<=window[1])
    errors = q*q*np.stack([c['errors'][stiffness] for c in selected])[:,mask]
    if mask.sum()<2 or not np.all(np.isfinite(errors) & (errors>0)):
        raise ValueError('Invalid WM window/errors')
    sizes = np.array([c['L'] for c in selected])
    value = q*q*np.stack([c['central'][stiffness] for c in selected])[:,mask]
    fit = fit_weber_minnhagen(T[mask], sizes, value, errors)
    draws = np.stack([c['bootstrap'][stiffness] for c in selected], axis=1)
    estimates = [fit_weber_minnhagen(T[mask], sizes, q*q*d[:,mask], errors, profile=False)['temperature'] for d in draws]
    estimates = np.array([np.nan if t is None else t for t in estimates])
    fit.update(bootstrap=estimates, valid_fraction=float(np.isfinite(estimates).mean()),
               statistical_se=float(estimates.std(ddof=1)) if np.isfinite(estimates).all() else None)
    return fit


def charged_eta(left, right, channel='A_plus_z'):
    """Pairwise eta from S(0)~L^(2-eta), with joint-within-run bootstrap draws."""
    if not np.array_equal(left['temperatures'], right['temperatures']):
        raise ValueError('Align temperature curves before comparing sizes')
    i, j = left['spatial_channels'].index(channel), right['spatial_channels'].index(channel)
    def estimate(a,b):
        return 2-np.log(b[...,j,0]/a[...,i,0])/np.log(right['L']/left['L'])
    value = estimate(left['central']['spectra'], right['central']['spectra'])
    draws = estimate(left['bootstrap']['spectra'], right['bootstrap']['spectra'])
    return value, draws.std(0, ddof=1)


def pseudocritical_comparison(curves, q=.5, min_size=96, stiffness='helicity', L0=None, window=None):
    """Refit the size drift in each time-block bootstrap; no interpolation-point fit."""
    selected = sorted((c for c in curves if c['L'] >= min_size and c['L'] != 64), key=lambda c: c['L'])
    crossings = [direct_jump(c, q, stiffness, window) for c in selected]
    if any(len(c['roots']) != 1 or c['statistical_se'] is None or c['statistical_se'] <= 0 for c in crossings):
        return dict(temperature=None, reason='Missing/ambiguous/unsupported direct crossing')
    sizes = np.array([c['L'] for c in selected])
    errors = np.array([c['statistical_se'] for c in crossings])
    central = np.array([c['roots'][0] for c in crossings])
    fit = fit_pseudocritical_bkt(sizes, central, errors, L0=L0)
    draws = np.stack([c['bootstrap'] for c in crossings], axis=1)
    estimates = [fit_pseudocritical_bkt(sizes, d, errors, L0=L0)['temperature'] for d in draws]
    estimates = np.array([np.nan if t is None else t for t in estimates])
    fit.update(bootstrap=estimates, valid_fraction=float(np.isfinite(estimates).mean()),
               statistical_se=float(estimates.std(ddof=1)) if np.isfinite(estimates).all() else None)
    return fit
