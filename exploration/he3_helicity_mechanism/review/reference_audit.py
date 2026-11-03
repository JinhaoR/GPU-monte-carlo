"""Independent algebra controls of the root finite-volume reference module."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("root_correlation_reference", HERE.parent / "correlation_checks.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

T = 0.3815
rho_pair = 8*T/np.pi
rho_single = rho_pair/4
yp, wp, pp = reference.winding(rho_pair, T, np.pi)
ys, ws, ps = reference.winding(rho_single, T, 2*np.pi)
# Same winding distribution after phi=2 theta; responses differ by charge^2.
np.testing.assert_allclose(yp/4, ys, atol=1e-14, rtol=0)
np.testing.assert_allclose(pp, ps, atol=1e-14, rtol=0)
np.testing.assert_allclose(yp/(4*T), 0.63650817819, atol=1e-11, rtol=0)

c_pair, inverted_rho, _ = reference.compact_reference(T, yp, np.pi)
np.testing.assert_allclose(inverted_rho, rho_pair, atol=1e-12, rtol=0)
# Evaluate charge-one phi covariance independently, including integer sectors.
L = reference.L
k = 2*np.pi*np.fft.fftfreq(L)
laplacian = 4*np.sin(k[:, None]/2)**2 + 4*np.sin(k[None, :]/2)**2
laplacian[0, 0] = np.inf
green = np.fft.ifft2(1/laplacian).real
c_oscillator_phi = np.exp(-T/rho_single*(green[0, 0]-green))
m = np.arange(-30, 31)
w = np.exp(-rho_single*(2*np.pi)**2*m*m/(2*T)); w /= w.sum()
factor_phi = np.cos(2*np.pi*m[:, None]*np.arange(L)[None, :]/L).T@w
c_phi = c_oscillator_phi*factor_phi[:, None]*factor_phi[None, :]
np.testing.assert_allclose(c_pair, c_phi, atol=1e-13, rtol=0)

# A positive synthetic contact background must be recovered in spectral units.
A, B = 0.2, 0.1
c = A*c_pair.copy(); c[0, 0] += B
s, sref = np.fft.fft2(c).real, np.fft.fft2(c_pair).real
def lowmode(a):
    return (a[1, 0]+a[-1, 0]+a[0, 1]+a[0, -1])/4
recovered_A = (s[0, 0]-lowmode(s))/(sref[0, 0]-lowmode(sref))
recovered_B = lowmode(s)-recovered_A*lowmode(sref)
np.testing.assert_allclose([recovered_A, recovered_B], [A, B], atol=1e-13, rtol=0)
np.testing.assert_allclose(s.sum()/L**2, A+B, atol=1e-14, rtol=0)

result = dict(
    all_checks_pass=True, simulations_run=0,
    critical_paired_Y_over_4T=yp/(4*T),
    critical_paired_rho_over_T=rho_pair/T,
    critical_L64_pair_xi_over_L=reference.xi_from_spectrum(np.fft.fft2(c_pair)),
    pair_to_single_covariance_max_error=float(abs(c_pair-c_phi).max()),
    contact_recovered_A=float(recovered_A), contact_recovered_B=float(recovered_B),
    limitations="No test here of the effective scalar reduction for real GL data or sparse Kxy sampling uncertainty.",
)
(HERE / "results" / "reference_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
