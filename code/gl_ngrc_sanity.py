"""
Guyon-Lekeufack path-dependent volatility generator + sanity checks
(Objective 1, second generator; companion to garch_ngrc_pilot.py).

Model (the corrected dissertation Eq. 14, verified against Guyon & Lekeufack 2023,
"Volatility is (mostly) path-dependent", and Andres & Jourdain 2024, arXiv:2408.02477):

    r_t     = sigma_t * z_t,            z_t ~ i.i.d. N(0,1)
    sigma_t = beta0 + beta1 * R1_t + beta2 * sqrt(R2_t)
    R1_t    = sum_{j=1..C} K1_j * r_{t-j}        (trend feature,  beta1 <= 0: leverage)
    R2_t    = sum_{j=1..C} K2_j * r_{t-j}^2      (activity feature, beta2 >= 0: clustering)

Kernels are time-shifted power laws (TSPL), the parametric family G-L calibrate on
index data (their numerical experiments truncate at C = 1000 business days):

    K_i,j = (j + delta_i)^(-alpha_i) / sum_{m=1..C} (m + delta_i)^(-alpha_i)

so each kernel is normalised to sum to one over the window. The normalisation makes
R2 a weighted average of past squared returns, so the unconditional mean of sigma is
approximately beta0 / (1 - beta2).

What this file does:
  --check : simulate the G-L cell, report positivity of sigma_t, basic moments,
            the ACF of r^2 next to a GARCH(1,1) reference, a reference relaxation
            rate measured from a long simulation (the dissertation's frozen recovery
            target for this cell), and a light k=44 linear NG-RC latch check
            (same readout as the pilot). Writes a two-panel figure to results/.

NOT a certification run: no replications, no pass/fail. Settings below are hand-set
working values inside the G-L family (signs and normalisation as calibrated in the
paper; magnitudes chosen so the unconditional daily vol is ~1.2% and persistence is
comparable with the GARCH cells, whose alpha+beta are 0.95 and 0.98). Every seed is
fixed and recorded, as the protocol requires.

Run: python3 gl_ngrc_sanity.py --check
"""

import argparse
import numpy as np

# ---------------- frozen working parameters (hand-set, recorded here) ----------------
C_CUT = 1000          # kernel cut-off lag in days, as in G-L's own numerical experiments
BETA0 = 0.0024        # intercept floor; mean sigma ~ BETA0 / (1 - BETA2)
BETA1 = -1.5          # trend sensitivity (leverage), < 0
BETA2 = 0.80          # activity sensitivity (clustering), in (0, 1)
ALPHA1, DELTA1 = 1.5, 2.0   # trend kernel: fast decay, weeks-scale leverage memory
ALPHA2, DELTA2 = 1.2, 2.0   # activity kernel: slow decay, long vol memory
BURN = 2000           # >= kernel memory, so start-up transients are gone (pilot uses 1000)
SEEDS = [20261120 + i for i in range(8)]   # fixed, recorded

K0 = 44               # same warm-up as the pilot so train/test rows match the GARCH cells


def tspl_kernel(alpha, delta, cutoff=C_CUT):
    """Time-shifted power-law kernel, normalised to sum to one over j=1..cutoff."""
    j = np.arange(1, cutoff + 1, dtype=float)
    w = (j + delta) ** (-alpha)
    return w / w.sum()


def sim_gl(n, beta0=BETA0, beta1=BETA1, beta2=BETA2,
           a1=ALPHA1, d1=DELTA1, a2=ALPHA2, d2=DELTA2,
           seed=0, burn=BURN, cutoff=C_CUT):
    """Simulate the G-L path-dependent generator. Returns (x, sig): x = r^2 (what the
    model sees, matching sim_garch) and sig = latent sigma path (the recovery target
    reference; never shown to the model)."""
    rng = np.random.default_rng(seed)
    K1 = tspl_kernel(a1, d1, cutoff)
    K2 = tspl_kernel(a2, d2, cutoff)
    T = n + burn
    z = rng.standard_normal(T)
    r = np.zeros(T)
    sig = np.zeros(T)
    n_neg = 0
    for t in range(T):
        lo = max(0, t - cutoff)
        hist_r = r[lo:t][::-1][:cutoff]          # r_{t-1}, r_{t-2}, ...
        m = len(hist_r)
        R1 = K1[:m] @ hist_r
        R2 = K2[:m] @ (hist_r * hist_r)
        s = beta0 + beta1 * R1 + beta2 * np.sqrt(R2) if m else beta0
        if s <= 0:
            n_neg += 1
            s = 1e-8                             # count, then floor to keep the run alive
        sig[t] = s
        r[t] = s * z[t]
    if n_neg:
        print(f"  WARNING: sigma_t hit the floor {n_neg} times in this path")
    x = r * r
    return x[burn:], sig[burn:], r[burn:]


def acf(v, max_lag):
    v = v - v.mean()
    denom = (v * v).sum()
    return np.array([1.0] + [(v[:-k] * v[k:]).sum() / denom for k in range(1, max_lag + 1)])


def sim_garch(n, omega, alpha, beta, seed, burn=1000):
    """Same as the pilot's, for the reference ACF."""
    rng = np.random.default_rng(seed)
    z2 = rng.standard_normal(n + burn) ** 2
    h = omega / (1 - alpha - beta)
    hs = np.empty(n + burn); x = np.empty(n + burn)
    for t in range(n + burn):
        hs[t] = h; x[t] = h * z2[t]
        h = omega + alpha * x[t] + beta * h
    return x[burn:], hs[burn:]


def states(x, k):
    """Pilot convention: rows t = K0-1..N-2, [x_t,...,x_{t-k+1}] -> target x_{t+1}."""
    T = len(x); ts = np.arange(K0 - 1, T - 1)
    return np.column_stack([x[ts - j] for j in range(k)]), x[ts + 1]


def fit_ridge(F, y, lam=1e-3):
    mu, sd = F.mean(0), F.std(0) + 1e-12
    Fs = (F - mu) / sd
    Fs = np.column_stack([np.ones(len(Fs)), Fs])
    A = Fs.T @ Fs + lam * np.eye(Fs.shape[1]); A[0, 0] -= lam
    ws = np.linalg.solve(A, Fs.T @ y)
    w = np.empty_like(ws); w[1:] = ws[1:] / sd; w[0] = ws[0] - np.sum(ws[1:] * mu / sd)
    return w


def relaxation_rate(sig, lags=range(40, 250)):
    """Dominant relaxation rate of the generator's volatility map, measured from the
    ACF tail of the latent sigma path of a long reference simulation: fit
    acf(m) ~ rho_ref^m over the given lags. This is the frozen recovery target the
    dissertation defines for the G-L cell."""
    a = acf(sig, max(lags))
    m = np.array(list(lags), dtype=float)
    y = np.log(np.maximum(a[list(lags)], 1e-12))
    slope = np.polyfit(m, y, 1)[0]
    return np.exp(slope)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    if not args.check:
        ap.error("nothing to do; pass --check")

    print(f"G-L cell, frozen working parameters: beta0={BETA0}, beta1={BETA1}, beta2={BETA2}")
    print(f"kernels: TSPL, K1 ~ (j+{DELTA1})^-{ALPHA1}, K2 ~ (j+{DELTA2})^-{ALPHA2}, cutoff {C_CUT}, burn {BURN}")

    # ---- positivity + moments over several fixed seeds ----
    print("\n[1] positivity and moments (8 seeds, N=20000 each):")
    neg_total = 0
    for s in SEEDS:
        x, sig, r = sim_gl(args.n, seed=s)
        neg = (sig <= 0).sum()
        neg_total += neg
        print(f"  seed {s}: min sigma={sig.min():.5f}, mean sigma={sig.mean():.5f}, "
              f"sd sigma={sig.std():.5f}, kurtosis r={float(((r - r.mean())**4).mean() / r.var()**2):.2f}")
    print(f"  non-positive sigma_t across all paths: {neg_total}")

    # ---- ACF of r^2 vs GARCH reference ----
    print("\n[2] ACF of r^2 (one long path, seed", SEEDS[0], "):")
    x, sig, r = sim_gl(60000, seed=SEEDS[0])
    a_gl = acf(x, 250)
    xg, hg = sim_garch(60000, 1e-5, 0.08, 0.90, seed=SEEDS[0])
    a_g = acf(xg, 250)
    for m in (1, 5, 22, 66, 132, 250):
        print(f"  lag {m:4d}: G-L {a_gl[m]:.3f}   GARCH(0.08,0.90) {a_g[m]:.3f}")

    # ---- reference relaxation rate (recovery target for this cell) ----
    rho_ref = relaxation_rate(sig)
    print(f"\n[3] reference relaxation rate from long-run ACF tail of sigma: rho_ref = {rho_ref:.4f}")
    print("    (freeze this with the protocol before any G-L certification run)")

    # ---- light k=44 NG-RC latch check (pilot readout) ----
    print("\n[4] k=44 linear NG-RC latch check on x = r^2:")
    X, y = states(x, K0)
    n_train = int(0.7 * len(X))
    w = fit_ridge(X[:n_train], y[:n_train])
    g = w[1:1 + K0]
    J = np.zeros((K0, K0)); J[0] = g
    J[1:, :-1] = np.eye(K0 - 1)
    rho_hat = max(abs(np.linalg.eigvals(J)))
    print(f"  rho_hat (spectral radius of delay-map Jacobian at train state) = {rho_hat:.4f}")
    print(f"  vs rho_ref = {rho_ref:.4f}  ->  bias {rho_hat - rho_ref:+.4f}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    a1.plot(sig[:2000], lw=0.6, label="G-L $\\sigma_t$")
    a1.plot(np.sqrt(hg[:2000]), lw=0.6, alpha=0.8, label="GARCH(1,1) $\\sigma_t$")
    a1.set_title("latent volatility paths (first 2000 days)")
    a1.set_xlabel("day"); a1.legend(fontsize=8)
    lags = np.arange(1, 251)
    a2.loglog(lags, a_gl[1:], ".", ms=2.5, label="G-L, ACF of $r^2$")
    a2.loglog(lags, a_g[1:], ".", ms=2.5, alpha=0.6, label="GARCH, ACF of $r^2$")
    a2.set_title("memory: power-law-ish vs geometric")
    a2.set_xlabel("lag (days)"); a2.legend(fontsize=8)
    fig.tight_layout()
    import os
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/gl_sanity_check.png", dpi=150)
    print("\nfigure written to results/gl_sanity_check.png")


if __name__ == "__main__":
    main()
