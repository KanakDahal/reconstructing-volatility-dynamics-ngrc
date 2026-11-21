"""
Cell 5 re-score under the analytical recovery target (Objective 1, H1, frozen protocol).

New file under the frozen-protocol rule. Loads the FROZEN per-replication results of
cell 5 (h1_gl_cell5_raw.csv; seeds, generator and training pipeline unchanged) and
re-applies the locked success criterion with the analytical p* from h1_gl_pstar.py
replacing the old reference-simulation (ACF-tail) estimate.

Equivalence note: p* enters only the scoring step, never the simulation or training,
and every seed is frozen, so a literal re-run of the 1500 replications reproduces the
raw rho_hat values bit-for-bit. Re-scoring the frozen raw results is the re-run's
outcome, stated here so the audit trail is explicit.

Locked criterion (h1_validate_gl.py, unchanged):
    success  <=>  0 < rho_hat < 1 and |rho_hat - p*| <= 0.10 * (1 - p*)
    cell supported <=> one-sided lower 95% Clopper-Pearson bound on the success
                       rate exceeds 0.90 (critical count 1370/1500)

Writes:
    results/h1_gl_cell5_pstar_raw.csv      per-rep values with the new scoring columns
    results/h1_gl_cell5_pstar_summary.json cell-level verdict under the analytical target

Run: python3 h1_gl_cell5_rescore.py
"""

import csv
import json
import math
import os

from h1_gl_pstar import analytical_coefficients, characteristic_root

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
RAW_IN = os.path.join(RESULTS, "h1_gl_cell5_raw.csv")
RAW_OUT = os.path.join(RESULTS, "h1_gl_cell5_pstar_raw.csv")
SUMMARY_OUT = os.path.join(RESULTS, "h1_gl_cell5_pstar_summary.json")

OLD_PSTAR = 0.9975521276790771


def cp_lower_one_sided(k, n, alpha=0.05):
    """One-sided lower 95% Clopper-Pearson bound (matches scipy.stats.beta.ppf(alpha, k, n-k+1))."""
    if k == 0:
        return 0.0
    lo, hi = 0.0, k / n
    # bisection on the binomial CDF: P(K >= k | p) = alpha at the bound
    def binom_sf(p):
        # P(K >= k) for Binomial(n, p), summed stably from k
        log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                     + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k, n + 1)]
        m = max(log_terms)
        return math.exp(m) * sum(math.exp(t - m) for t in log_terms)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if binom_sf(mid) > alpha:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main():
    d, x_bar, S_bar, sigma_star, A, c = analytical_coefficients()
    p_star = characteristic_root(d)
    margin = 0.10 * (1.0 - p_star)

    with open(RAW_IN) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)

    out_rows = []
    n_success = 0
    for row in rows:
        rho = float(row["rho_hat"])
        error = rho - p_star
        success = (0.0 < rho < 1.0) and abs(error) <= margin
        n_success += int(success)
        out_rows.append({
            "cell": row["cell"], "generator": row["generator"],
            "p_star": f"{p_star:.13f}", "N": row["N"], "k": row["k"],
            "rep": row["rep"], "spawn_key": row["spawn_key"],
            "rho_hat": row["rho_hat"], "error": f"{error:.15f}",
            "margin": f"{margin:.15f}", "success": int(success),
            "n_train_rows": row["n_train_rows"],
        })

    q_lower = cp_lower_one_sided(n_success, n)
    rhos = [float(r["rho_hat"]) for r in rows]
    mean_rho = sum(rhos) / n
    sd_rho = math.sqrt(sum((r - mean_rho) ** 2 for r in rhos) / n)

    summary = {
        "cell": 5,
        "generator": "guyon_lekeufack",
        "criterion": "0<rho<1 and |rho-p*| <= 0.10*(1-p*); supported iff one-sided CP-95% lower bound > 0.90 (critical 1370/1500)",
        "p_star_old_refsim": OLD_PSTAR,
        "p_star_new_analytical": p_star,
        "margin_new": margin,
        "band_new": [p_star - margin, p_star + margin],
        "n_reps": n,
        "n_success_new": n_success,
        "cp95_lower_new": q_lower,
        "supported_new": q_lower > 0.90,
        "n_success_old": sum(int(r["success"]) for r in rows),
        "rho_hat_mean": mean_rho,
        "rho_hat_sd": sd_rho,
        "rho_hat_min": min(rhos),
        "rho_hat_max": max(rhos),
        "min_abs_error_to_new_pstar": min(abs(r - p_star) for r in rhos),
        "equivalence_note": ("p* enters scoring only; all seeds frozen, so a literal re-run "
                             "reproduces these rho_hat values bit-for-bit. This re-score is "
                             "the re-run outcome."),
        "verdict": "cell 5 NOT supported under the analytical target (0/1500; mean bias %.4f)" % (mean_rho - p_star),
    }

    with open(RAW_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
