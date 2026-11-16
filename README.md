# Reconstructing Volatility Dynamics with Next Generation Reservoir Computing

Research workspace (code, results, report).

Status: **exploratory**. Nothing here is a final result or a test of a hypothesis.

## Contents
- `code/garch_ngrc_pilot.py` - GARCH(1,1) control pilot for Objective 1 / H1 (NG-RC Jacobian spectral radius vs known alpha+beta).
- `results/raw_results.csv` - per-replication output (1600 rows). `results/full_output.txt` - printed summary of the same run.
- `report/` - LaTeX report of that run (`garch_pilot_report.tex/.pdf`), `make_report.py` (builds tables/figure from the CSV), `fig_rho.png`.

## Reproduce
```
pip install -r requirements.txt
cd code && python3 garch_ngrc_pilot.py --reps 50      # ~12 s; writes raw_results.csv in the current directory
mv raw_results.csv ../results/ && cd ../report && python3 make_report.py && pdflatex garch_pilot_report.tex
```

## Scope and caveats
- Covers only the GARCH-control part of Objective 1. Not done: Guyon-Lekeufack process, relaxation times, Lyapunov exponents, feedback asymmetry.
- Parameter grid (omega=0.05, four alpha/beta pairs, k in {5,22,44}, ridge 1e-3, N in {2000,20000}), Gaussian innovations and 50 replications are exploratory choices, not taken from the objectives draft.
- Spread columns are Monte Carlo spread of the estimator, not confidence intervals; no tolerance for "within estimation error" has been set, so no verdict on H1.

## License
No license has been specified; all rights reserved by the author for now.
