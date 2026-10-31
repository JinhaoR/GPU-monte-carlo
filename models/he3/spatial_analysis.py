"""Small analysis hook for the existing NPZ run dictionaries."""
import warnings
import numpy as np
from utils.Analysis.statistics import block_means, jackknife_from_blocks
from .spatial import SECTORS, contracted_components, correlation_ratio


def check_spatial(run):
    samples, meta = run['samples'], run['metadata']
    if 'spatial_amplitudes' not in samples:
        return {'available': False, 'reason': 'legacy file has no spatial channels'}
    ids = np.asarray(meta['spatial_sample_indices'])
    a, s = samples['spatial_amplitudes'], samples['spatial_structure']
    assert np.array_equal(np.asarray(meta['sample_sweeps'])[ids], meta['spatial_sample_sweeps'])
    assert np.array_equal(samples['spatial_energy'], samples['energy'][ids])
    maps = samples['spatial_walkers']
    assert np.all(np.sort(maps, axis=-1) == np.arange(maps.shape[-1]))
    residuals = dict(kappa=a[...,0,0]-samples['chirality'][ids],
        P=a[...,1,0]-samples['phase_real'][ids]-1j*samples['phase_imag'][ids],
        Qzz=-2*a[...,3,0]/np.sqrt(6)-(2/3)*samples['density'][ids]*samples['spin_alignment'][ids],
        energy_parts=s[...,13]+s[...,14]+s[...,16]-samples['energy'][ids])
    errors = {key: float(np.max(abs(value))) for key, value in residuals.items()}
    assert max(errors.values()) < 5e-6*meta['L']**2
    return {'available': True, 'maximum_absolute_residuals': errors}


def summarize_spatial(run, block_size=200):
    """Delete the same temporal block across all temperatures and observables.

    block_size counts SPATIAL events (200 = 100,000 sweeps at cadence 500).
    Mean subtraction is recomputed in each delete; no per-field spatial mean
    is removed. Invalid/singular xi remains NaN with a status and valid count.
    """
    if not check_spatial(run)['available']:
        warnings.warn('Spatial channels absent in this legacy file')
        return None
    samples, meta = run['samples'], run['metadata']
    a, structure = samples['spatial_amplitudes'], samples['spatial_structure']
    L = meta['L']; T = np.asarray(meta['temperatures'])
    if block_size < 1 or len(a)//block_size < 2:
        raise ValueError('At least two complete spatial blocks are required')
    indices = np.asarray(meta['spatial_sample_indices'])
    moments = {'amplitude': block_means(a, block_size),
               'power': block_means(abs(a)**2, block_size),
               'structure': block_means(structure, block_size)}
    for name in ('current_x', 'current_y', 'curvature_x', 'curvature_y'):
        values = samples[name][indices]
        moments[name] = block_means(values, block_size)
        if name.startswith('current'): moments[name+'_square'] = block_means(values**2, block_size)
    numerator = structure[..., [6,6,7,8,2]]
    denominator = structure[..., [0,1,0,0,0]]
    local = np.divide(numerator, denominator, out=np.full(numerator.shape,np.nan), where=denominator>0)
    moments['local_ratios'] = block_means(local, block_size)
    B = len(moments['amplitude'])
    means = {key: value.mean(0) for key, value in moments.items()}

    def estimate(m):
        result = dict(mean_amplitudes=m['amplitude'], structure=m['structure'], local_ratios=m['local_ratios'])
        for connected, label in ((False,'full'),(True,'connected')):
            power = m['power']-abs(m['amplitude'])**2 if connected else m['power']
            spectra = contracted_components(power)/(L*L)
            result[label+'_spectra'] = spectra
            for direction, sk in [('x',spectra[...,1:3].mean(-1)),('y',spectra[...,3:5].mean(-1)),
                                  ('xy',spectra[...,1:].mean(-1))]:
                r, xi, status = correlation_ratio(spectra[...,0],sk,L)
                result[label+'_ratio_'+direction] = r
                result[label+'_xi_'+direction] = xi
                result[label+'_status_'+direction] = status
        for axis in ('x','y'):
            current = m['current_'+axis]
            result['helicity_'+axis] = (m['curvature_'+axis]-(m['current_'+axis+'_square']-current**2)/T)/(L*L)
        return result

    out = estimate(means)
    deleted = [estimate({key:(B*means[key]-value[b])/(B-1) for key,value in moments.items()}) for b in range(B)]
    for key, value in list(out.items()):
        if '_status_' in key: continue
        values = np.stack([d[key] for d in deleted])
        valid = np.all(np.isfinite(values),axis=0)
        error = np.full(value.shape,np.nan)
        error[valid] = np.sqrt((B-1)*np.var(values[:,valid],axis=0))
        out[key+'_se'] = error
        if '_xi_' in key: out[key+'_valid_deletes'] = np.isfinite(values).sum(0)
    out.update(temperatures=T,sectors=np.array(SECTORS),blocks=np.array(B),
        block_size_spatial_events=np.array(block_size),dropped_events=np.array(len(a)-B*block_size),
        local_ratio_names=np.array(['epsilon_total_density','epsilon_valid_density','orbital_loss_total_density',
                                   'spin_loss_total_density','excluded_density_fraction']))
    return out


def summarize_realspace(run, block_size=10):
    """Full and homogeneous connected kappa/P/Q profiles, without a decay fit.

    Blocks count profile events (10 = 100,000 sweeps at the default cadence).
    Subtract the ENSEMBLE mean, recomputed in each delete, not each snapshot's
    spatial mean. Q contracts all five orthonormal components. P stays complex.
    """
    samples, meta = run['samples'], run['metadata']
    if 'realspace_profiles' not in samples:
        warnings.warn('Real-space profiles absent in this file')
        return None
    profiles, means = samples['realspace_profiles'], samples['realspace_means']
    if block_size < 1 or len(profiles)//block_size < 2:
        raise ValueError('At least two complete real-space blocks are required')
    blocks = dict(profiles=block_means(profiles, block_size), means=block_means(means, block_size))
    out = {}
    for connected, label in ((False, 'full'), (True, 'connected')):
        def estimate(m):
            c = m['profiles']
            if connected:
                c = c - abs(m['means'])[..., None, None]**2
            sectors = np.stack((c[:, 0], c[:, 1], c[:, 2:7].sum(axis=1)), axis=1)
            return np.stack((sectors.real, sectors.imag), axis=-1)
        value, error = jackknife_from_blocks(blocks, estimate)
        out[label] = value[..., 0] + 1j*value[..., 1]
        out[label+'_se_real'], out[label+'_se_imag'] = error[..., 0], error[..., 1]
    out.update(temperatures=np.asarray(meta['realspace_temperatures']),
               displacements=np.asarray(meta['realspace_displacements']),
               sectors=np.array(['kappa', 'P', 'Q']), directions=np.array(['x', 'y']),
               mean_fields=blocks['means'].mean(0), blocks=np.array(len(blocks['means'])),
               block_size_profile_events=np.array(block_size),
               dropped_events=np.array(len(profiles) % block_size))
    return out
