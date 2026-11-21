#!/usr/bin/env python3
"""H1 second-model certification: HAR-RV under the frozen protocol.

New file under the frozen-protocol rule: the frozen NG-RC cells, code, and results
are untouched. This runner certifies the HAR-RV benchmark (Corsi 2009) on the SAME
paired replication datasets as the frozen NG-RC run (same master seed and spawn keys,
so each replication regenerates the identical series bit-for-bit), under the same
locked criterion:

    success iff 0 < rho_hat < 1 and |rho_hat - p*| <= 0.10 * (1 - p*),
    cell supported iff the exact one-sided lower 95% binomial bound on the pass
    rate exceeds 0.90 (critical count 1370 of 1500).

Model (per replication):
    x_{m} = c + b_d x_{m-1}
              + b_w * (1/5)  sum_{j=1..5}  x_{m-j}
              + b_m * (1/22) sum_{j=1..22} x_{m-j} + e_m,        x = r^2,
    fit by OLS on the same 13957 training targets as the frozen readout.

Jacobian: the HAR recursion is AR(22) with coefficients
    phi_1 = b_d + b_w/5 + b_m/22,
    phi_j = b_w/5 + b_m/22,  j = 2..5,
    phi_j = b_m/22,          j = 6..22,
so rho_hat is the exact spectral radius of its 22x22 companion form. No NG-RC
delay embedding, no warm-up features: the model class is the point of comparison.

Targets: cells 1-4 use alpha+beta of the frozen GARCH cells; cell 5 uses the
analytical p* (h1_gl_pstar.py), NOT the superseded reference-simulation estimate.

Outputs (refuses to overwrite):
    results/h1_har_cell{1..5}_raw.csv   per-replication rows
    results/h1_har_summary.json         combined verdict

Run from the experiment folder:
    OPENBLAS_NUM_THREADS=1 python code/h1_validate_har.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import csv, json, sys, time, platform
from pathlib import Path
import numpy as np
from scipy.stats import beta as betadist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from garch_ngrc_pilot import sim_garch
import gl_ngrc_sanity as gl
from h1_gl_pstar import analytical_coefficients, characteristic_root

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'

MASTER_SEED = 193859540053032125584074185598345864826  # same as frozen protocol
R = 1500
N = 20000
K = 44               # frozen warm-up: HAR targets are aligned to the NG-RC rows
OMEGA = 0.05
GARCH_BURN = 1000
N_TRAIN_ROWS = 13957

PAIRS = [(0.05, 0.90), (0.10, 0.85), (0.08, 0.90), (0.15, 0.80)]  # cells 1-4
CELL5_SPAWN = (4,)

FIELDS = ['cell', 'generator', 'target', 'N', 'rep', 'spawn_key', 'rho_hat',
          'error', 'margin', 'success', 'failure', 'n_train_rows',
          'b_const', 'b_daily', 'b_weekly', 'b_monthly']


def har_fit_rho(x):
    """Fit HAR-RV by OLS on the frozen-aligned training targets; return
    (rho_hat, coefficients). Targets are x[m] for m = K .. K+N_TRAIN_ROWS-1,
    identical to the frozen readout's training rows."""
    m0 = K
    m1 = K + N_TRAIN_ROWS
    idx = np.arange(m0, m1)
    yv = x[idx]
    daily = x[idx - 1]
    weekly = np.mean([x[idx - j] for j in range(1, 6)], axis=0)
    monthly = np.mean([x[idx - j] for j in range(1, 23)], axis=0)
    A = np.column_stack([np.ones_like(yv), daily, weekly, monthly])
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    c, b_d, b_w, b_m = coef
    phi = np.zeros(22)
    phi[0] = b_d + b_w / 5.0 + b_m / 22.0
    phi[1:5] = b_w / 5.0 + b_m / 22.0
    phi[5:22] = b_m / 22.0
    J = np.zeros((22, 22))
    J[0, :] = phi
    J[1:, :-1] = np.eye(21)
    rho = float(np.max(np.abs(np.linalg.eigvals(J))))
    return rho, (float(c), float(b_d), float(b_w), float(b_m))


def _one_rep(args):
    rep, child, cell, target = args
    margin = 0.10 * (1.0 - target)
    generator = 'guyon_lekeufack' if cell == 5 else 'garch'
    row = dict(cell=cell, generator=generator, target=target, N=N, rep=rep,
               spawn_key=f'{cell-1}/{rep}', rho_hat='', error='', margin=margin,
               success=0, failure='', n_train_rows=N_TRAIN_ROWS,
               b_const='', b_daily='', b_weekly='', b_monthly='')
    try:
        if cell == 5:
            x, sig, r = gl.sim_gl(N, seed=child, burn=gl.BURN)
        else:
            al, be = PAIRS[cell - 1]
            x, _ = sim_garch(N, OMEGA, al, be, child, burn=GARCH_BURN)
        rho, (c, b_d, b_w, b_m) = har_fit_rho(x)
        row['b_const'], row['b_daily'], row['b_weekly'], row['b_monthly'] = c, b_d, b_w, b_m
        if not np.isfinite(rho):
            row['failure'] = 'nonfinite_radius'
        else:
            row['rho_hat'] = rho
            row['error'] = rho - target
            row['success'] = int(0 < rho < 1 and abs(rho - target) <= margin)
            if rho >= 1:
                row['failure'] = 'radius_ge_one'
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError, ValueError) as exc:
        row['failure'] = type(exc).__name__ + ': ' + str(exc)
    return row


def run_cell(cell, target, pool):
    raw = OUT / f'h1_har_cell{cell}_raw.csv'
    if raw.exists():
        raise RuntimeError(f'Refusing to overwrite {raw}; inspect/recover explicitly')
    spawn = CELL5_SPAWN if cell == 5 else (cell - 1,)
    children = np.random.SeedSequence(MASTER_SEED, spawn_key=spawn).spawn(R)
    start = time.time()
    with raw.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for done, row in enumerate(pool.imap(_one_rep,
                                             ((rep, ch, cell, target) for rep, ch in enumerate(children)),
                                             chunksize=10), 1):
            writer.writerow(row)
            if done % 300 == 0:
                handle.flush()
                print(f'  cell {cell}: {done}/{R}; {time.time()-start:.1f}s', flush=True)
    rows = list(csv.DictReader(raw.open()))
    assert len(rows) == R and [int(r['rep']) for r in rows] == list(range(R))
    return rows


def cell_stats(rows, target):
    S = sum(int(r['success']) for r in rows)
    lower = float(betadist.ppf(.05, S, R - S + 1)) if S else 0.
    upper = float(betadist.ppf(.95, S + 1, R - S)) if S < R else 1.
    vals = np.array([float(r['rho_hat']) for r in rows if r['rho_hat'] != ''])
    e = vals - target
    return dict(
        target=target, margin=0.10 * (1.0 - target), R=R, successes=S,
        q_hat=S / R, q_lower95_one_sided=lower, q_upper95_one_sided=upper,
        support=lower > .9, evidence_below_target=upper < .9, critical_count=1370,
        failure_count=sum(bool(r['failure']) for r in rows),
        finite_estimate_count=len(vals),
        mean_rho=float(vals.mean()), bias=float(e.mean()),
        sd=float(vals.std(ddof=1)), rmse=float(np.sqrt(np.mean(e ** 2))),
        mean_b_daily=float(np.mean([float(r['b_daily']) for r in rows])),
        mean_b_weekly=float(np.mean([float(r['b_weekly']) for r in rows])),
        mean_b_monthly=float(np.mean([float(r['b_monthly']) for r in rows])),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / 'h1_har_summary.json'
    if summary_path.exists():
        raise RuntimeError(f'Refusing to overwrite {summary_path}')

    d, *_ = analytical_coefficients()
    p_star_gl = characteristic_root(d)
    targets = {c: PAIRS[c - 1][0] + PAIRS[c - 1][1] for c in range(1, 5)}
    targets[5] = p_star_gl

    import multiprocessing as mp
    summary = dict(model='HAR-RV (Corsi 2009), OLS on x=r^2, aggregates include day t',
                   criterion='0<rho<1 and |rho-target| <= 0.10*(1-target); supported iff '
                             'one-sided CP-95% lower bound > 0.90 (critical 1370/1500)',
                   paired='same master seed and spawn keys as the frozen NG-RC run: '
                          'each replication regenerates the identical series',
                   n_train_rows=N_TRAIN_ROWS,
                   cell5_target_method='analytical p* (h1_gl_pstar.py)',
                   cells={})
    with mp.Pool(min(2, mp.cpu_count())) as pool:
        for cell in range(1, 6):
            print(f'cell {cell}: target {targets[cell]:.13f}', flush=True)
            rows = run_cell(cell, targets[cell], pool)
            summary['cells'][str(cell)] = cell_stats(rows, targets[cell])
            st = summary['cells'][str(cell)]
            print(f"  -> {st['successes']}/{R} in band; CP95 lower {st['q_lower95_one_sided']:.4f}; "
                  f"mean rho {st['mean_rho']:.6f}; supported={st['support']}", flush=True)

    summary['python'] = platform.python_version()
    summary['numpy'] = np.__version__
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
