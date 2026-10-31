"""CPU-only checks of the saved homogeneous onsite GL potential.

Run from the repository root: python exploration/he3_soc/local_potential.py
This evaluates algebraic trial states and small deterministic minimizations;
it does not sample configurations, alter datasets, or execute CUDA code.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results" / "tables"


def invariants(a):
    n = np.vdot(a, a).real
    nx, ny = np.sum(abs(a)**2, axis=0)
    xx, yy = np.sum(a*a, axis=0)
    xy = np.sum(a[:, 0]*a[:, 1])
    overlap = np.vdot(a[:, 0], a[:, 1])
    return np.array([abs(xx+yy)**2, n*n,
                     nx*nx+ny*ny+2*(overlap*overlap).real,
                     nx*nx+ny*ny+2*abs(overlap)**2,
                     abs(xx)**2+abs(yy)**2+2*abs(xy)**2])


def dipole(a):
    n = np.vdot(a, a).real
    return (abs(a[0, 0]+a[1, 1])**2+abs(a[0, 0])**2
            +abs(a[1, 1])**2+2*(a[0, 1]*a[1, 0].conjugate()).real
            -2*n/3)


def pack(a):
    return np.r_[a.real.ravel(), a.imag.ravel()]


def unpack(x):
    return (x[:6]+1j*x[6:]).reshape(3, 2)


def potential(x, betas, gd):
    a = unpack(x)
    return -2*np.vdot(a, a).real+betas@invariants(a)+gd*dipole(a)


def hessian(fun, x, step=1e-4):
    # Smooth quartic polynomial; evaluate all mixed central differences.
    basis = np.eye(len(x))*step
    out = np.empty((len(x), len(x)))
    for i, ei in enumerate(basis):
        for j, ej in enumerate(basis):
            out[i, j] = (fun(x+ei+ej)-fun(x+ei-ej)
                         -fun(x-ei+ej)+fun(x-ei-ej))/(4*step*step)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    spectra = []
    rng = np.random.default_rng(20261029)
    # Hermitian quadratic form D=A^dagger M A, flattening by spin row.
    matrix = np.diag([4/3, -2/3, -2/3, 4/3, -2/3, -2/3])
    matrix[0, 3] = matrix[3, 0] = 1
    matrix[1, 2] = matrix[2, 1] = 1
    assert np.allclose(np.linalg.eigvalsh(matrix), [-5/3, -2/3, -2/3, 1/3, 1/3, 7/3])
    check_rng = np.random.default_rng(33)
    for _ in range(100):
        a = check_rng.normal(size=(3, 2))+1j*check_rng.normal(size=(3, 2))
        assert np.allclose(invariants(a)[2], np.linalg.norm(a@a.T)**2)
        assert np.allclose(dipole(a), np.vdot(a.ravel(), matrix@a.ravel()).real)
    for gd in [0, .001, .01, .1]:
        dataset = ROOT / "data" / "he3" / f"P=20_gd={gd:g}_gaussian" / "L=64.npz"
        with np.load(dataset, allow_pickle=False) as z:
            info = next(iter(json.loads(z["manifest"].item())["runs"].values()))
            params = info["metadata"]["model_parameters"]
        betas = np.asarray(params["betas"])
        actual_gd = params["g_d"]
        beta_a = betas[1]+betas[3]+betas[4]
        beta_p = betas[0]+betas[1]+(betas[2]+betas[3]+betas[4])/2
        n_a = (1+actual_gd/3)/beta_a
        n_p = (1+5*actual_gd/6)/beta_p
        chiral = np.zeros((3, 2), dtype=complex)
        chiral[2] = np.sqrt(n_a/2)*np.array([1, 1j])
        planar = np.zeros((3, 2), dtype=complex)
        planar[0, 1], planar[1, 0] = np.sqrt(n_p/2), -np.sqrt(n_p/2)
        fun = lambda x: potential(x, betas, actual_gd)
        candidates = [pack(chiral), pack(planar)]
        candidates.extend(rng.normal(0, .4, 12) for _ in range(16))
        opt = [minimize(fun, x, method="BFGS", options={"gtol": 1e-7})
               for x in candidates]
        best = min(opt, key=lambda r: r.fun)
        for label, a in [("chiral_z", chiral), ("antisymmetric_planar", planar)]:
            x = pack(a)
            eig = np.linalg.eigvalsh(hessian(fun, x))
            eig_halfstep = np.linalg.eigvalsh(hessian(fun, x, 5e-5))
            n = np.vdot(a, a).real
            rows.append(dict(gd=gd, dataset=str(dataset.relative_to(ROOT)),
                             state=label, beta_chiral=beta_a, beta_planar=beta_p,
                             density=n, dipole_per_density=dipole(a)/n,
                             potential=fun(x), min_hessian_eigenvalue=eig[0],
                             first_nonzero_hessian_eigenvalue=eig[eig > 1e-5][0],
                             negative_hessian_modes=int(np.sum(eig < -1e-5)),
                             zero_hessian_modes=int(np.sum(abs(eig) < 1e-5)),
                             hessian_step_change=np.max(abs(eig-eig_halfstep)),
                             multistart_best_potential=best.fun,
                             multistart_gap_to_chiral=best.fun-fun(pack(chiral)),
                             starts=len(opt), successful_optimizations=sum(r.success for r in opt)))
            spectra.extend(dict(gd=gd, state=label, mode=i, eigenvalue=v)
                           for i, v in enumerate(eig))
        if abs(best.fun-fun(pack(chiral))) > 1e-6:
            print(f"gd={gd:g}: best candidate differs from chiral by {best.fun-fun(pack(chiral)):.8g}")
    for name, data in [("local_potential.csv", rows), ("local_hessian.csv", spectra)]:
        with (OUT/name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print("Dipole spectrum:", np.linalg.eigvalsh(matrix))
    for row in rows:
        print(row["gd"], row["state"], "f=", row["potential"],
              "minimum Hessian=", row["min_hessian_eigenvalue"])
    print("Saved", OUT/"local_potential.csv")


if __name__ == "__main__":
    main()
