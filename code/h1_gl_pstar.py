"""
Analytical recovery target p* for the Guyon-Lekeufack cell (Objective 1, cell 5).

New file under the frozen-protocol rule: nothing in the frozen cells changes; this
script only COMPUTES the analytical recovery target that replaces the old
reference-simulation (ACF-tail) estimate.

Derivation (dissertation front matter v9, Section 3.1, eqs. 15-16), all quantities
computed from the frozen parameters in gl_ngrc_sanity.py, no simulation:

    x_t = r_t^2,  S_t = sum_j K2_j x_{t-j},  sigma* = beta0 + beta2 * sqrt(S_bar)

    d_j = (sigma* * beta2 / sqrt(S_bar)) * K2_j + beta1^2 * K1_j^2

    fixed point:  x_bar * (1 - beta1^2 * sum_j K1_j^2)
                      = (beta0 + beta2 * sqrt(x_bar * sum_j K2_j))^2

    p* = largest root in (0, 1) of  sum_j d_j p^{-j} = 1

With y = sqrt(x_bar) and c = sum_j K2_j the fixed point is the quadratic
    (A - beta2^2 c) y^2 - 2 beta0 beta2 sqrt(c) y - beta0^2 = 0,
A = 1 - beta1^2 sum_j K1_j^2, whose positive root gives x_bar.

Verification: power iteration of the companion (delay) form of the linear map
converges to the same dominant contraction factor; printed side by side with the
characteristic-root solve.

Run: python3 h1_gl_pstar.py
"""

import numpy as np

from gl_ngrc_sanity import (
    BETA0, BETA1, BETA2, ALPHA1, DELTA1, ALPHA2, DELTA2, C_CUT, tspl_kernel,
)

OLD_REF_SIM_PSTAR = 0.9975521276790771  # ACF-tail measurement this replaces


def analytical_coefficients():
    K1 = tspl_kernel(ALPHA1, DELTA1, C_CUT)
    K2 = tspl_kernel(ALPHA2, DELTA2, C_CUT)
    A = 1.0 - BETA1**2 * float(K1 @ K1)
    c = float(K2.sum())
    # quadratic in y = sqrt(x_bar): (A - beta2^2 c) y^2 - 2 beta0 beta2 sqrt(c) y - beta0^2 = 0
    aq = A - BETA2**2 * c
    bq = -2.0 * BETA0 * BETA2 * np.sqrt(c)
    cq = -BETA0**2
    disc = bq * bq - 4.0 * aq * cq
    y = (-bq + np.sqrt(disc)) / (2.0 * aq)  # positive root
    x_bar = y * y
    S_bar = x_bar * c
    sigma_star = BETA0 + BETA2 * np.sqrt(S_bar)
    d = (sigma_star * BETA2 / np.sqrt(S_bar)) * K2 + BETA1**2 * K1**2
    return d, x_bar, S_bar, sigma_star, A, c


def characteristic_root(d, tol=1e-15):
    """Largest root in (0,1) of sum_j d_j p^{-j} = 1 by bisection (f decreasing)."""
    j = np.arange(1, len(d) + 1, dtype=float)
    log_d = np.log(d)

    def f(p):
        # log( sum_j d_j p^{-j} ) via log-sum-exp, stable for tiny p
        lt = log_d - j * np.log(p)
        m = lt.max()
        return m + np.log(np.exp(lt - m).sum())  # compare against log(1) = 0

    lo, hi = 1e-12, 1.0 - 1e-15
    assert f(lo) > 0.0 and f(hi) < 0.0, "no root in (0,1): process not mean-stationary?"
    for _ in range(1000):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def companion_power_iteration(d, iters=200000, seed=7):
    """Independent check: iterate v_{t+1} = companion(d) v_t from a fixed non-degenerate
    start; the asymptotic per-step contraction ratio converges to the spectral radius."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(len(d))
    v /= np.linalg.norm(v)
    ratio = None
    for _ in range(iters):
        new = np.empty_like(v)
        new[0] = d @ v
        new[1:] = v[:-1]
        nrm = np.linalg.norm(new)
        ratio = nrm  # v is unit norm, so ||new|| is the one-step contraction factor
        v = new / nrm
    return ratio


def main():
    d, x_bar, S_bar, sigma_star, A, c = analytical_coefficients()
    print("frozen inputs: beta0=%.4f beta1=%.2f beta2=%.2f  K1~(j+%.1f)^-%.1f K2~(j+%.1f)^-%.1f cutoff=%d"
          % (BETA0, BETA1, BETA2, DELTA1, ALPHA1, DELTA2, ALPHA2, C_CUT))
    print(f"sum K2 = {c:.12f}  (normalised)")
    print(f"beta1^2 * sum K1^2 = {1.0 - A:.12f}")
    print(f"x_bar = E[r^2] = {x_bar:.10e}   (daily rms return {np.sqrt(x_bar)*100:.4f}%)")
    print(f"sigma* = {sigma_star:.10f}")
    print(f"magnitude channel sum: (sigma* beta2 / sqrt(S_bar)) * sum K2 = {(sigma_star * BETA2 / np.sqrt(S_bar)) * c:.12f}")
    print(f"signed channel sum:    beta1^2 * sum K1^2                    = {BETA1**2 * float(d[0]*0 + 0) + (1.0 - A):.12f}")
    print(f"sum d_j = {d.sum():.12f}  (< 1: mean-stationary)")

    p_star = characteristic_root(d)
    p_check = companion_power_iteration(d)
    print()
    print(f"p* (characteristic root)   = {p_star:.13f}")
    print(f"p* (companion power iter)  = {p_check:.13f}")
    print(f"implied relaxation time    = {-1.0 / np.log(p_star):.1f} days")
    print(f"old reference-sim estimate = {OLD_REF_SIM_PSTAR:.13f}")
    print(f"abs difference             = {abs(p_star - OLD_REF_SIM_PSTAR):.3e}")
    print(f"locked band around new p*: |rho - p*| <= 0.10*(1-p*) = ±{0.10*(1.0-p_star):.6f}")
    print(f"  -> rho-hat must lie in [{p_star - 0.10*(1-p_star):.9f}, {p_star + 0.10*(1-p_star):.9f}]")


if __name__ == "__main__":
    main()
