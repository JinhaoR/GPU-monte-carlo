"""Full-matrix spatial observables and NumPy analysis, with no CUDA dependency.

The kappa/P/Q basis, Fourier signs and ratio flags extend the validated
exploration/he3_spatial_pilot/observables.py implementation.
"""
import numpy as np

FIELDS = ('chirality', 'composite', 'q1', 'q2', 'q3', 'q4', 'q5',
          'B_xx', 'B_yy', 'B_xy', 'A_xx', 'A_xy', 'A_yx', 'A_yy', 'A_zx', 'A_zy')
MOMENTA = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
SECTORS = ('kappa', 'P', 'Q', 'B', 'A', 'Q_m0', 'Q_m1', 'Q_m2')
# Q_m0=q2, Q_m1=(q4,q5), Q_m2=(q1,q3), resolved about the film normal z.
# These are rank-two rotation channels, not three spin-vector components.
WEIGHTS = np.zeros((8, 16))
WEIGHTS[0, 0] = WEIGHTS[1, 1] = 1
WEIGHTS[2, 2:7] = 1
WEIGHTS[3, 7:10] = (1, 1, 2)
WEIGHTS[4, 10:] = 1
WEIGHTS[5, 3] = 1
WEIGHTS[6, 5:7] = 1
WEIGHTS[7, [2, 4]] = 1
DENSITY_FLOOR = 1e-12  # absolute in the existing dimensionless A normalization
HIST_EDGES = np.linspace(0, 1, 11)
STRUCTURE_NAMES = (
    'density', 'valid_density', 'excluded_density', 'valid_sites',
    'excluded_sites', 'nonfinite_sites', 'n_epsilon', 'n_orbital_loss',
    'n_spin_loss', 'n_plus', 'n_minus', 'abs_s_plus', 'abs_s_minus',
    'energy_bulk', 'energy_gradient', 'dipole_bare', 'energy_soc',
    'energy_direct', 'epsilon_outside_roundoff', 'abs_kappa',
)
HIST_NAMES = ('epsilon_count', 'epsilon_density', 'orbital_loss_count', 'spin_loss_count')


def add_spatial_samples(run, records, realspace_records=()):
    """Attach occasional measurements to the existing NPZ format, with their own axis.

    Record indices refer to scalar measurement calls in this single Simulation.
    Enable recording immediately before Simulation.run(); do not insert extra
    measure() calls into that run. A fresh model is used for each lattice size.
    """
    samples, meta = run['samples'], run['metadata']
    if records:
        indices, amplitudes, structure, walkers = zip(*records)
        indices = np.asarray(indices)
        samples['spatial_amplitudes'] = np.stack(amplitudes)
        samples['spatial_structure'] = np.stack(structure)
        samples['spatial_walkers'] = np.stack(walkers)
        samples['spatial_energy'] = samples['energy'][indices]
        meta.update(spatial_sample_indices=indices, spatial_sample_sweeps=meta['sample_sweeps'][indices],
                    spatial_fields=FIELDS, spatial_momenta=MOMENTA, structure_names=STRUCTURE_NAMES,
                    structure_histograms=HIST_NAMES, structure_histogram_edges=HIST_EDGES,
                    structure_density_floor=DENSITY_FLOOR)
        records.clear()
    if realspace_records:
        indices, profiles, means, walkers, temperatures = zip(*realspace_records)
        indices, temperatures = np.asarray(indices), np.asarray(temperatures)
        if not np.all(temperatures == temperatures[0]):
            raise ValueError('Real-space temperature indices changed during the run')
        probes = temperatures[0]
        samples['realspace_profiles'] = np.stack(profiles)
        samples['realspace_means'] = np.stack(means)
        samples['realspace_walkers'] = np.stack(walkers)
        samples['realspace_energy'] = samples['energy'][indices][:, probes]
        meta.update(realspace_sample_indices=indices, realspace_sample_sweeps=meta['sample_sweeps'][indices],
                    realspace_temperature_indices=probes, realspace_temperatures=meta['temperatures'][probes],
                    realspace_fields=FIELDS[:7], realspace_directions=('x', 'y'),
                    realspace_displacements=np.arange(meta['L']//2+1),
                    realspace_normalization='G_c(r)=mean_x f_c(x+r) conj(f_c(x)); no mean subtraction')
        realspace_records.clear()


def local_fields(state):
    a = np.asarray(state, dtype=np.complex128)
    x, y = a[..., 0], a[..., 1]
    kappa = 2 * np.sum((x.conj() * y).imag, axis=-1)
    b = np.einsum('...ma,...mb->...ab', a, a)
    m = np.einsum('...ma,...na->...mn', a, a.conj()).real
    q = np.stack(((m[..., 0, 0] - m[..., 1, 1]) / np.sqrt(2),
                  (m[..., 0, 0] + m[..., 1, 1] - 2*m[..., 2, 2]) / np.sqrt(6),
                  np.sqrt(2)*m[..., 0, 1], np.sqrt(2)*m[..., 0, 2],
                  np.sqrt(2)*m[..., 1, 2]), axis=-1)
    return np.concatenate((kappa[..., None], (b[..., 0, 0]-b[..., 1, 1])[..., None],
                           q, b[..., 0, 0, None], b[..., 1, 1, None],
                           b[..., 0, 1, None], a.reshape(a.shape[:-2]+(6,))), axis=-1)


def numpy_amplitudes(state):
    a = np.asarray(state)
    single = a.ndim == 4
    f = local_fields(a[None] if single else a)
    ft = np.fft.fftn(f, axes=(1, 2))
    result = np.stack([ft[:, x, y, :] for x, y in MOMENTA], axis=-1)
    return result[0] if single else result


def unpack_real_modes(raw):
    """26 real fields at 0,+x,+y -> 16 complex fields at all five modes."""
    z = raw[..., 0] + 1j*raw[..., 1]
    out = np.empty(z.shape[:-2]+(16, 5), dtype=np.complex128)
    for dst, src in ((0, 0), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)):
        out[..., dst, :] = z[..., src, :][..., [0, 1, 1, 2, 2]]
        out[..., dst, [2, 4]] = out[..., dst, [2, 4]].conj()
    for dst, src in [(1, 1)] + [(k, 8+2*(k-7)) for k in range(7, 16)]:
        out[..., dst, 0] = z[..., src, 0] + 1j*z[..., src+1, 0]
        for mode, pos, neg in ((1, 1, 2), (2, 3, 4)):
            out[..., dst, pos] = z[..., src, mode] + 1j*z[..., src+1, mode]
            out[..., dst, neg] = z[..., src, mode].conj() + 1j*z[..., src+1, mode].conj()
    return out


def chiral_distance(state, density_floor=DENSITY_FLOOR):
    """Closed-form projection; no clipping, no arbitrary division at zero norm.

    For each sign, Re(v v^dagger)=u u^T+t t^T has largest eigenvalue
    (v^dagger v+|v dot v|)/2. Maximizing this eigenvalue gives the nearest
    z*d*w^T. orbital_loss + spin_loss = epsilon on the maximizing branch.
    """
    a = np.asarray(state, np.complex128)
    n = np.sum(abs(a)**2, axis=(-2, -1))
    v = np.stack([(a[..., 0]-1j*s*a[..., 1])/np.sqrt(2) for s in (1, -1)], axis=-2)
    ns = np.sum(abs(v)**2, axis=-1)
    ss = abs(np.sum(v*v, axis=-1))
    winner = np.argmax(ns+ss, axis=-1)
    selected_n = np.take_along_axis(ns, winner[..., None], -1)[..., 0]
    selected_s = np.take_along_axis(ss, winner[..., None], -1)[..., 0]
    valid = np.isfinite(n) & (n > density_floor) & np.all(np.isfinite(ns+ss), axis=-1)
    eps = np.full(n.shape, np.nan)
    orbital = eps.copy(); spin = eps.copy()
    eps[valid] = 1-(selected_n[valid]+selected_s[valid])/(2*n[valid])
    orbital[valid] = 1-selected_n[valid]/n[valid]
    spin[valid] = (selected_n[valid]-selected_s[valid])/(2*n[valid])
    return dict(n=n, valid=valid, epsilon=eps, orbital_loss=orbital,
                spin_loss=spin, n_sigma=ns, abs_s_sigma=ss, winner=winner)


def histogram_with_tails(values, weights=None):
    """12 bins: underflow, ten [0,.1)...[.9,1], overflow. NaNs excluded."""
    a = np.asarray(values)
    good = np.isfinite(a)
    index = np.searchsorted(HIST_EDGES, a[good], side='right')
    index[a[good] == 1] = 10
    w = None if weights is None else np.asarray(weights)[good]
    return np.bincount(index, weights=w, minlength=12)


def contracted_components(values):
    """Input (...,component,momentum_or_distance), output (...,sector,coordinate)."""
    return np.einsum('sc,...ck->...sk', WEIGHTS, values)


def correlation_ratio(s0, sk, L):
    # Preserved pilot invalid/singular policy, including the roundoff floor.
    s0, sk = np.broadcast_arrays(np.asarray(s0, float), np.asarray(sk, float))
    ratio = np.full(s0.shape, np.nan); xi = ratio.copy()
    status = np.full(s0.shape, 3, dtype=np.int8)
    floor = 256*np.finfo(float).eps**2*np.maximum(abs(s0), abs(sk))
    good = np.isfinite(s0) & np.isfinite(sk) & (sk > floor) & (s0 >= 0)
    ratio[good] = s0[good]/sk[good]; status[good] = 1
    valid = good & (ratio >= 1); status[valid] = 0
    xi[valid] = np.sqrt(ratio[valid]-1)/(2*L*np.sin(np.pi/L))
    singular = np.isfinite(s0) & np.isfinite(sk) & (s0 > 0) & (sk >= 0) & (sk <= floor)
    ratio[singular] = np.inf; status[singular] = 2
    return ratio, xi, status


def ensemble(amplitudes, L, connected=False):
    """Sample axis first; preserve all temperature/component/momentum axes.

    Connected powers use the ensemble mean at EACH mode; use this function
    again inside every resample. A two-pass variance avoids cancellation.
    """
    a = np.asarray(amplitudes)
    mean = a.mean(axis=0)
    p = abs(a-mean if connected else a)**2
    spectra = contracted_components(p.mean(axis=0))/(L*L)
    result = dict(spectra=spectra, mean_amplitudes=mean, connected=connected)
    for axis, sk in [('x', spectra[..., 1:3].mean(-1)),
                     ('y', spectra[..., 3:5].mean(-1)),
                     ('xy', spectra[..., 1:].mean(-1))]:
        result['ratio_'+axis], result['xi_over_L_'+axis], result['status_'+axis] = correlation_ratio(spectra[..., 0], sk, L)
    return result


def compact_correlations(state):
    """Per configuration, component-resolved periodic correlations via NumPy FFT.

    G_c(r)=N^-1 sum_x f_c(x+r) conj(f_c(x)). Full signed x/y displacements
    0..L-1 are retained. Radial shell j uses j<=minimum-image distance<j+1,
    including diagonal displacements. Raw means support homogeneous-ensemble
    connected correlations; NO per-configuration spatial mean is subtracted.
    """
    a = np.asarray(state)
    if a.ndim != 4:
        raise ValueError('Pass one (L,L,3,2) configuration at a time')
    L = a.shape[0]; n = L*L
    f = local_fields(a)
    ft = np.fft.fftn(f, axes=(0, 1))
    c = np.fft.ifftn(abs(ft)**2, axes=(0, 1))/n
    d = np.minimum(np.arange(L), L-np.arange(L))
    shell = np.floor(np.hypot(d[:, None], d[None, :])).astype(int)
    counts = np.bincount(shell.ravel())
    radial = np.stack([(np.bincount(shell.ravel(), weights=c[..., k].real.ravel())
                       + 1j*np.bincount(shell.ravel(), weights=c[..., k].imag.ravel()))/counts
                      for k in range(len(FIELDS))])
    return dict(x=c[:, 0, :].T, y=c[0, :, :].T, radial=radial,
                means=f.mean(axis=(0, 1)), radial_counts=counts,
                radial_edges=np.arange(len(counts)+1), displacements=np.arange(L))
