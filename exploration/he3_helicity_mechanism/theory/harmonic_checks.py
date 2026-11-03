"""Bounded CPU algebra/harmonic checks of the actual full3x2GL model.

No Monte Carlo, minimization, GPU, production writes, or old-result writes.
Run from repository root: python exploration/he3_helicity_mechanism/theory/harmonic_checks.py
The covariance is a quadratic expansion about the zero-temperature homogeneous
chiral branch, not a thermal prediction close to the observed transition.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BETAS = np.array([-.5165123343467712, .9420228004455566,
                  .9875290393829346, .8790628910064697, -1.152514934539795])
C0 = float(np.float32(2 / 3))
BETA_A = BETAS[1] + BETAS[3] + BETAS[4]


def from_coordinates(q):
    """Orthonormal real coordinates: Re u, Im u, Re v, Im v (each x,y,z)."""
    q = np.asarray(q)
    u = q[..., :3] + 1j * q[..., 3:6]
    v = q[..., 6:9] + 1j * q[..., 9:12]
    return np.stack([(u + v) / np.sqrt(2), 1j * (u - v) / np.sqrt(2)], axis=-1)


BASIS = from_coordinates(np.eye(12))


def invariants(a):
    x, y = a[..., 0], a[..., 1]
    nx, ny = np.sum(abs(x)**2), np.sum(abs(y)**2)
    xx, yy, xy = np.sum(x*x), np.sum(y*y), np.sum(x*y)
    o = np.vdot(x, y)
    return np.array([abs(xx+yy)**2, (nx+ny)**2,
                     nx*nx+ny*ny+2*(o*o).real,
                     nx*nx+ny*ny+2*abs(o)**2,
                     abs(xx)**2+abs(yy)**2+2*abs(xy)**2])


def dipole(a):
    return (abs(a[0, 0]+a[1, 1])**2 + abs(a[0, 0])**2
            + abs(a[1, 1])**2 + 2*(a[0, 1]*a[1, 0].conjugate()).real
            - C0*np.vdot(a, a).real)


def potential(a, gd=0.):
    return -2*np.vdot(a, a).real + BETAS @ invariants(a) + gd*dipole(a)


def hessian(gd):
    n = (1 + gd*C0/2) / BETA_A
    common = -4 + 4*n*BETA_A
    lam_nu = common - 8*n*BETAS[4]
    lam_vt = -4 + 4*n*(BETAS[1]+BETAS[2])
    lam_vz = common + 8*n*(BETAS[0]+BETAS[2])
    h = np.diag([common, common, common+8*n*BETA_A,
                 lam_nu, lam_nu, common,
                 lam_vt, lam_vt, lam_vz, lam_vt, lam_vt, lam_vz])
    # Since D is quadratic, this polarization computes its exact real Hessian.
    d = np.array([[(dipole(a+b)-dipole(a-b))/2 for b in BASIS] for a in BASIS])
    return h + gd*d, n


def gradient(a, b, c, d):
    return (3*(abs(a)**2+abs(d)**2)+abs(b)**2+abs(c)**2
            + 2*(a.conjugate()*d+b.conjugate()*c).real)


def gradient_chiral(dx_u, dx_v, dy_u, dy_v):
    return (2*(abs(dx_u)**2+abs(dx_v)**2+abs(dy_u)**2+abs(dy_v)**2)
            + 2*(dx_u.conjugate()*dx_v-dy_u.conjugate()*dy_v
                   - 1j*(dx_u.conjugate()*dy_v+dy_u.conjugate()*dx_v)).real)


def fourier_hessians(L, gd):
    k = 2*np.pi*np.fft.fftfreq(L)
    px, py = np.meshgrid(np.exp(1j*k)-1, np.exp(1j*k)-1, indexing='ij')
    px2, py2 = abs(px)**2, abs(py)**2
    orbital = np.empty((L, L, 2, 2))
    orbital[..., 0, 0], orbital[..., 1, 1] = 3*px2+py2, px2+3*py2
    orbital[..., 0, 1] = orbital[..., 1, 0] = 2*(px.conjugate()*py).real
    g = np.einsum('ima,xyab,jmb->xyij', BASIS.conjugate(), orbital, BASIS).real
    h0, n = hessian(gd)
    return h0[None, None, :, :] + 2*g, n


def lattice_parts(a):
    xp, yp = np.roll(a, -1, axis=0), np.roll(a, -1, axis=1)
    da, db = (xp-a)[..., 0], (xp-a)[..., 1]
    dc, dd = (yp-a)[..., 0], (yp-a)[..., 1]
    zx = (3*da.conjugate()+dd.conjugate())*xp[..., 0] + (db.conjugate()+dc.conjugate())*xp[..., 1]
    zy = (dc.conjugate()+db.conjugate())*yp[..., 0] + (3*dd.conjugate()+da.conjugate())*yp[..., 1]
    curvature = 2*np.sum(3*abs(xp[..., 0])**2+abs(xp[..., 1])**2-zx.real
                        + abs(yp[..., 0])**2+3*abs(yp[..., 1])**2-zy.real)
    grad = np.sum(gradient(da, db, dc, dd))
    diagonal_interference = np.sum(((xp[..., 0]-yp[..., 0]).conjugate()
                                  * (xp[..., 1]-yp[..., 1])).real)
    norm = np.vdot(a, a).real
    return float(curvature), float(8*norm-grad+2*diagonal_interference)


def twist_derivatives(a):
    xp, yp = np.roll(a, -1, axis=0), np.roll(a, -1, axis=1)
    da, db = (xp-a)[..., 0], (xp-a)[..., 1]
    dc, dd = (yp-a)[..., 0], (yp-a)[..., 1]
    zx = (3*da.conjugate()+dd.conjugate())*xp[..., 0] + (db.conjugate()+dc.conjugate())*xp[..., 1]
    zy = (dc.conjugate()+db.conjugate())*yp[..., 0] + (3*dd.conjugate()+da.conjugate())*yp[..., 1]
    kx = 2*np.sum(3*abs(xp[..., 0])**2+abs(xp[..., 1])**2-zx.real)
    ky = 2*np.sum(abs(yp[..., 0])**2+3*abs(yp[..., 1])**2-zy.real)
    kxy = 2*np.sum((xp[..., 0].conjugate()*yp[..., 1]
                   + xp[..., 1].conjugate()*yp[..., 0]).real)
    return np.array([2*np.sum(zx.imag), 2*np.sum(zy.imag), kx, ky, kxy])


def twisted_hamiltonian(a, qx=0., qy=0., gd=0.):
    dx = np.exp(-1j*qx)*np.roll(a, -1, axis=0)-a
    dy = np.exp(-1j*qy)*np.roll(a, -1, axis=1)-a
    onsite = sum(potential(site, gd) for site in a.reshape(-1, 3, 2))
    return onsite + np.sum(gradient(dx[..., 0], dx[..., 1], dy[..., 0], dy[..., 1]))


def finite_twists(a, gd):
    # Richardson central derivatives, using the full Hamiltonian at each twist.
    levels = []
    for h in (.01, .005):
        zero = twisted_hamiltonian(a, gd=gd)
        xp, xm = twisted_hamiltonian(a, h, 0., gd), twisted_hamiltonian(a, -h, 0., gd)
        yp, ym = twisted_hamiltonian(a, 0., h, gd), twisted_hamiltonian(a, 0., -h, gd)
        xy = (twisted_hamiltonian(a, h, h, gd)-twisted_hamiltonian(a, h, -h, gd)
              -twisted_hamiltonian(a, -h, h, gd)+twisted_hamiltonian(a, -h, -h, gd))/(4*h*h)
        levels.append(np.array([(xp-xm)/(2*h), (yp-ym)/(2*h),
                                (xp-2*zero+xm)/h**2, (yp-2*zero+ym)/h**2, xy]))
    return (4*levels[1]-levels[0])/3


def main():
    start = time.perf_counter()
    rng = np.random.default_rng(20261101)
    source_path = ROOT / 'models/he3/update.py'
    source = source_path.read_text(encoding='utf-8')
    parsed = ast.parse(source)
    wanted = {'norm2', 'dipole_invariant', 'potential', 'gradient'}
    funcs = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
             and node.name in wanted]
    for node in funcs:
        node.decorator_list = []
    ns = {'float32': np.float32}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source_path), 'exec'), ns)
    errors = {'onsite_source': 0., 'gradient_source': 0., 'chiral_gradient': 0.,
              'hessian_finite_difference': 0., 'hessian_halfstep': 0.,
              'uniform_trace': 0., 'phase_schur': 0., 'lattice_curvature_identity': 0.,
              'finite_twist_derivatives': 0.}
    for _ in range(20):
        a = rng.normal(size=(3, 2)) + 1j*rng.normal(size=(3, 2))
        for gd in (0., .001, .01, .1):
            errors['onsite_source'] = max(errors['onsite_source'],
                abs(potential(a, gd)-ns['potential'](a, BETAS, gd)))
        p = rng.normal(size=4) + 1j*rng.normal(size=4)
        errors['gradient_source'] = max(errors['gradient_source'],
            abs(gradient(*p)-ns['gradient'](*p)))
        dxu, dxv, dyu, dyv = p
        a, b = (dxu+dxv)/np.sqrt(2), 1j*(dxu-dxv)/np.sqrt(2)
        c, d = (dyu+dyv)/np.sqrt(2), 1j*(dyu-dyv)/np.sqrt(2)
        errors['chiral_gradient'] = max(errors['chiral_gradient'],
            abs(gradient(a, b, c, d)-gradient_chiral(*p)))
        aa = rng.normal(size=(3, 2)) + 1j*rng.normal(size=(3, 2))
        n = np.vdot(aa, aa).real
        rho = 2*(n*np.eye(2)+2*(aa.conjugate().T @ aa).real)
        errors['uniform_trace'] = max(errors['uniform_trace'], abs(np.trace(rho)-8*n))
        assert np.linalg.eigvalsh(rho)[0] >= 2*n-1e-10
        field = rng.normal(size=(4, 5, 3, 2)) + 1j*rng.normal(size=(4, 5, 3, 2))
        lhs, rhs = lattice_parts(field)
        errors['lattice_curvature_identity'] = max(errors['lattice_curvature_identity'], abs(lhs-rhs))

    spectra = []
    for gd in (0., .001, .01, .1):
        h, n = hessian(gd)
        q0 = np.zeros(12); q0[2] = np.sqrt(n)
        fd = []
        for step in (1e-3, 5e-4):
            e = np.eye(12)*step
            hs = np.array([[(potential(from_coordinates(q0+x+y), gd)
                            - potential(from_coordinates(q0+x-y), gd)
                            - potential(from_coordinates(q0-x+y), gd)
                            + potential(from_coordinates(q0-x-y), gd))/(4*step*step)
                           for y in e] for x in e])
            # Richardson cancellation removes the quartic central-difference error.
            fd.append(hs)
        extrapolated = (4*fd[1]-fd[0])/3
        errors['hessian_finite_difference'] = max(errors['hessian_finite_difference'],
                                                  float(np.max(abs(extrapolated-h))))
        errors['hessian_halfstep'] = max(errors['hessian_halfstep'],
                                           float(np.max(abs(fd[1]-fd[0]))))
        eigs, vectors = np.linalg.eigh(h)
        for j, value in enumerate(eigs):
            spectra.append(dict(gd=gd, mode=j, eigenvalue=float(value),
                                dominant_coordinate=int(np.argmax(abs(vectors[:, j])))))

    covariances = []
    for L in (32, 64, 128):
        hh, n = fourier_hessians(L, 0.)
        vals, vecs = np.linalg.eigh(hh)
        invvals = np.divide(1., vals, out=np.zeros_like(vals), where=vals > 1e-9)
        local_cov = np.einsum('xyij,xyj,xykj->ik', vecs, invvals, vecs)/(L*L)
        normal_indices = [3, 4, 6, 7, 8, 9, 10, 11]
        wrong = float(np.trace(local_cov[6:12, 6:12]))
        nonunit = float(local_cov[3, 3]+local_cov[4, 4])
        for temperature in (.05, .35, .3815):
            covariances.append(dict(L=L, T=temperature, n_reference=n,
                wrong_chirality_epsilon=temperature*wrong/n,
                nonunitary_epsilon=temperature*nonunit/n,
                total_epsilon_leading=temperature*(wrong+nonunit)/n,
                amplitude_variance=temperature*local_cov[2, 2],
                phase_angle_variance=temperature*local_cov[5, 5]/n,
                transverse_spin_angle_variance=temperature*(local_cov[0, 0]+local_cov[1, 1])/n,
                removed_zero_modes=int(np.sum(vals <= 1e-9))))
        assert len(normal_indices) == 8

    dispersion = []
    m_vz = hessian(0.)[0][8, 8]
    # Algebraic integration of all other coordinates at nonzero k.
    for kx, ky in ((.001, 0.), (.01, .02), (.1, 0.), (.3, .4), (.7, -.4)):
        px, py = np.exp(1j*kx)-1, np.exp(1j*ky)-1
        s = abs(px)**2+abs(py)**2
        c = abs(px)**2-abs(py)**2-2j*(px.conjugate()*py).real
        m = np.array([[3*abs(px)**2+abs(py)**2, 2*(px.conjugate()*py).real],
                      [2*(px.conjugate()*py).real, abs(px)**2+3*abs(py)**2]])
        hh = hessian(0.)[0] + 2*np.einsum('ima,ab,jmb->ij', BASIS.conjugate(), m, BASIS).real
        other = [j for j in range(12) if j != 5]
        schur = hh[5, 5] - hh[5, other] @ np.linalg.solve(hh[np.ix_(other, other)], hh[other, 5])
        formula = 4*s - 4*abs(c)**2/(m_vz+4*s)
        errors['phase_schur'] = max(errors['phase_schur'], abs(schur-formula))
        dispersion.append(dict(kx=kx, ky=ky, phase_kernel=schur,
                               phase_kernel_over_4s=schur/(4*s), formula=formula))

    controls = {}
    n0 = 1/BETA_A
    q0 = np.zeros(12); q0[2] = np.sqrt(n0)
    controls['uniform_chiral'] = np.broadcast_to(from_coordinates(q0), (8, 8, 3, 2)).copy()
    aa = rng.normal(size=(3, 2)) + 1j*rng.normal(size=(3, 2))
    aa *= np.sqrt(n0/np.vdot(aa, aa).real)
    controls['uniform_full_tensor'] = np.broadcast_to(aa, (8, 8, 3, 2)).copy()
    qnu = q0.copy(); qnu[3] = np.sqrt(n0)
    controls['uniform_nonunitary'] = np.broadcast_to(from_coordinates(qnu)/np.sqrt(2), (8, 8, 3, 2)).copy()
    wall = controls['uniform_chiral'].copy(); wall[4:, ..., 1] *= -1
    controls['straight_opposite_wall_pair'] = wall
    coords = np.zeros((8, 8, 12)); coords[..., 2] = np.sqrt(n0)
    coords[..., 0] = .2*np.sin(2*np.pi*np.arange(8)/8)[:, None]
    controls['smooth_spin'] = from_coordinates(coords)
    controls['inhomogeneous_full_tensor'] = controls['uniform_chiral'] + .15*(rng.normal(size=(8, 8, 3, 2))+1j*rng.normal(size=(8, 8, 3, 2)))
    twist_rows = []
    for name, field in controls.items():
        analytic = twist_derivatives(field)
        for gd in (0., .01):
            fd = finite_twists(field, gd)
            err = float(np.max(abs(fd-analytic)))
            errors['finite_twist_derivatives'] = max(errors['finite_twist_derivatives'], err)
            twist_rows.append(dict(case=name, L=8, gd=gd, current_x=analytic[0],
                current_y=analytic[1], curvature_x=analytic[2], curvature_y=analytic[3],
                curvature_xy=analytic[4], finite_twist_max_error=err))

    for name, rows in [('hessian_modes.csv', spectra), ('harmonic_covariance.csv', covariances),
                       ('phase_kernel.csv', dispersion), ('finite_twist_controls.csv', twist_rows)]:
        with (HERE/name).open('w', newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    assert max(errors.values()) < 1e-4, errors
    metadata = dict(description='Deterministic quadratic-branch calculation; no thermal sampling',
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    betas=BETAS.tolist(), K=1., errors=errors,
                    zero_soc_density=1/BETA_A, elapsed_seconds=time.perf_counter()-start,
                    covariance_scope='Expansion about bare homogeneous n0, no self-consistent thermal mass or density; no defects; q=0 Goldstone modes removed')
    (HERE/'checks.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    print('T=.3815 covariance:', [r for r in covariances if r['T'] == .3815])


if __name__ == '__main__':
    main()
