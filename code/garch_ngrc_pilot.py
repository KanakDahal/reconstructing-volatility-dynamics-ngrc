"""
EXPLORATORY PILOT - GARCH(1,1) control part of Objective 1 only (not the whole objective).

Original objective/hypothesis wording (Objectives and Research Plan, 13 Nov 2026 draft):
  Objective 1: validate recovery of dynamics on controlled systems (GARCH control and a
    Guyon-Lekeufack path-dependent process); test whether NG-RC recovers relaxation rates,
    feedback asymmetry, memory, and whether Jacobian relaxation times / Lyapunov exponents match
    the analytically known values.
  Hypothesis 1: "The spectral radius of the learned Jacobian recovers the known persistence
    alpha+beta of the GARCH control to within estimation error."
  (A later revised draft qualifies this as a conditional-mean control experiment.)
NOT covered here: Guyon-Lekeufack process, relaxation times, Lyapunov exponents, asymmetry.

What this does: for each (alpha, beta, N, k, library) simulate GARCH(1,1), fit NG-RC (delay
features of x_t = r_t^2, ridge) to predict x_{t+1}, take the spectral radius of the learned
delay-map Jacobian at the training-mean state, compare with alpha+beta.
Descriptive output only: bias, sd and RMSE of rho_hat - (alpha+beta) over simulated datasets
(Monte Carlo spread), plus a percentile range of the estimates. There is NO hypothesis test and
no claim that H1 is "supported". Settings (omega, parameter pairs, k grid, ridge lambda) are
exploratory choices. With finite k, a regression on r^2 is a truncated approximation of the
ARMA(1,1) full-history conditional mean, so bias in rho_hat is expected and is itself a result.

Fair comparison: every (k, library, HAR) uses the SAME simulated series (seed per replication),
the SAME rows: state at time t predicts x_{t+1}; train on t in [K0-1, n_train-1], test on t in [n_train, N-2],
n_train = int(0.7*N), K0 = 44 for every model (shorter-k models just ignore the extra lags).
Run: python3 garch_ngrc_pilot.py [--reps 30 default] [--quick]; reported run used --reps 50   -> prints summary, writes raw_results.csv
"""
import argparse, csv, itertools, time
import numpy as np

K0 = 44  # common warm-up so all models share identical train/test rows

def sim_garch(n, omega, alpha, beta, seed, burn=1000):
    rng = np.random.default_rng(seed)
    z2 = rng.standard_normal(n + burn) ** 2
    h = omega / (1 - alpha - beta); hs = np.empty(n + burn); x = np.empty(n + burn)
    for t in range(n + burn):
        hs[t] = h; x[t] = h * z2[t]
        h = omega + alpha * x[t] + beta * h
    return x[burn:], hs[burn:]          # x = r^2

def states(x, k):                        # rows t=K0-1..N-2 ; [x_t,...,x_{t-k+1}] -> target x_{t+1}
    T = len(x); ts = np.arange(K0 - 1, T - 1)
    return np.column_stack([x[ts - j] for j in range(k)]), x[ts + 1]

def feats(X, quad):
    c = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])]
    if quad:
        c += [X[:, i] * X[:, j] for i, j in itertools.combinations_with_replacement(range(X.shape[1]), 2)]
    return np.column_stack(c)

def fit(F, y, lam):
    mu, sd = F[:, 1:].mean(0), F[:, 1:].std(0) + 1e-12
    Fs = np.column_stack([F[:, 0], (F[:, 1:] - mu) / sd])
    A = Fs.T @ Fs + lam * np.eye(F.shape[1]); A[0, 0] -= lam
    ws = np.linalg.solve(A, Fs.T @ y)
    w = np.empty_like(ws); w[1:] = ws[1:] / sd; w[0] = ws[0] - np.sum(ws[1:] * mu / sd)
    return w

def jac(w, k, quad, s0):
    g = w[1:1 + k].copy()
    if quad:
        i0 = 1 + k
        for n, (i, j) in enumerate(itertools.combinations_with_replacement(range(k), 2)):
            c = w[i0 + n]
            if i == j: g[i] += 2 * c * s0[i]
            else: g[i] += c * s0[j]; g[j] += c * s0[i]
    J = np.zeros((k, k)); J[0] = g
    if k > 1: J[1:, :-1] = np.eye(k - 1)
    return J

def har_feats(x):
    T = len(x); ts = np.arange(K0 - 1, T - 1)
    cs = np.concatenate([[0], np.cumsum(x)])
    m = lambda L: np.array([(cs[t + 1] - cs[t + 1 - L]) / L for t in ts])
    return np.column_stack([np.ones(len(ts)), x[ts], m(5), m(22)])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    reps = 5 if a.quick else a.reps
    Ns = [2000] if a.quick else [2000, 20000]
    pairs = [(0.05, 0.90), (0.10, 0.85), (0.08, 0.90), (0.15, 0.80)]
    omega, lam, ks = 0.05, 1e-3, [5, 22, 44]
    t0 = time.time(); rows = []
    for (al, be) in pairs:
        for N in Ns:
            for rep in range(reps):
                seed = 1000 * rep + N % 997                    # same series for every model below
                x, _ = sim_garch(N, omega, al, be, seed)
                ntr = int(0.7 * len(x))
                Xk, y = states(x, 44); tr = slice(0, ntr - (K0 - 1)); te = slice(ntr - (K0 - 1), None)
                Hf = har_feats(x); wh = np.linalg.lstsq(Hf[tr], y[tr], rcond=None)[0]
                har = float(np.sqrt(np.mean((Hf[te] @ wh - y[te]) ** 2)))
                for k, quad in [(5, False), (22, False), (44, False), (5, True)]:
                    X = Xk[:, :k]; F = feats(X, quad); w = fit(F[tr], y[tr], lam)
                    rho = float(np.max(np.abs(np.linalg.eigvals(jac(w, k, quad, X[tr].mean(0))))))
                    rm = float(np.sqrt(np.mean((F[te] @ w - y[te]) ** 2)))
                    rows.append(dict(alpha=al, beta=be, true_persistence=al + be, N=N, rep=rep, seed=seed,
                                     k=k, library="quad" if quad else "linear", rho_hat=rho,
                                     oos_rmse=rm, har_rmse=har, n_test=int(len(y[te]))))
    el = time.time() - t0
    with open("raw_results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"EXPLORATORY descriptive summary ({reps} replications per cell; runtime {el:.0f}s)")
    print("bias=mean(rho_hat)-(alpha+beta); sd,[p2.5,p97.5] = Monte Carlo spread of rho_hat across datasets (not a CI, not a test)")
    print(f"{'alpha':>5} {'beta':>5} {'a+b':>5} {'N':>6} {'k':>3} {'lib':>6} {'bias':>8} {'sd':>7} {'RMSE':>7} {'[p2.5,p97.5]':>16} {'oosRMSE':>8} {'HAR':>8}")
    for key in sorted({(r['alpha'], r['beta'], r['N'], r['k'], r['library']) for r in rows}):
        g = [r for r in rows if (r['alpha'], r['beta'], r['N'], r['k'], r['library']) == key]
        rh = np.array([r['rho_hat'] for r in g]); t = key[0] + key[1]; lo, hi = np.percentile(rh, [2.5, 97.5])
        print(f"{key[0]:5.2f} {key[1]:5.2f} {t:5.2f} {key[2]:6d} {key[3]:3d} {key[4]:>6} {rh.mean()-t:8.4f} {rh.std():7.4f} "
              f"{np.sqrt(np.mean((rh-t)**2)):7.4f} [{lo:6.3f},{hi:6.3f}] {np.mean([r['oos_rmse'] for r in g]):8.4f} {np.mean([r['har_rmse'] for r in g]):8.4f}")
    # Check: regression of h_{t+1} on (h_t, x_t) with the TRUE conditional variance observed.
    print("\nCheck (not a noise-free process; data are stochastic): OLS of h_{t+1} on (h_t, r_t^2) uses the exact GARCH")
    print("recursion with the latent variance supplied, so it recovers beta, alpha up to float error. Tests code, not NG-RC.")
    for (al, be) in pairs:
        x, h = sim_garch(5000, omega, al, be, 7)
        F = np.column_stack([np.ones(4999), h[:-1], x[:-1]]); w = np.linalg.lstsq(F, h[1:], rcond=None)[0]
        print(f"  true alpha={al:.2f} beta={be:.2f} | OLS alpha={w[2]:.4f} beta={w[1]:.4f}")
if __name__ == "__main__": main()
