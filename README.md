# Reconstructing Volatility Dynamics with Next Generation Reservoir Computing

Research workspace for the dissertation: a simulation-based certification study of learned volatility dynamics (working title; final title TBD).

Status: **H1 certification run complete under the frozen protocol; second-model certification in progress.**

## Contents
- `code/garch_ngrc_pilot.py` - GARCH(1,1) control pilot for Objective 1 / H1 (NG-RC Jacobian spectral radius vs known alpha+beta).
- `results/raw_results.csv` - per-replication pilot output (1600 rows). `results/full_output.txt` - printed summary of the same run.
- `report/` - LaTeX report of the pilot (`garch_pilot_report.tex/.pdf`), `make_report.py`, `fig_rho.png`.
- `front_matter/` - dissertation front matter v9 (abstract, introduction, literature review, methodology, objectives & hypotheses): `Dahal_Dissertation_FrontMatter.tex/.pdf` with `dissertation_refs.bib` (87 verified references).

## Where the study stands (Nov 21, 2026)
- The H1 protocol is locked: five frozen cells (four GARCH(1,1) persistence levels + one Guyon-Lekeufack path-dependent generator), 1500 replications each, pass iff the one-sided lower 95% Clopper-Pearson bound on the in-band rate exceeds 0.90 (critical count 1370/1500). Frozen code and raw results live in the private working repo; this repo carries the pilot and the dissertation text.
- Result, stated plainly: NG-RC fails the locked band on all five cells in this experiment. The GARCH QMLE positive control on the same G-L data shows the same direction of bias, so the bias is not unique to NG-RC in this experiment. No stronger claim is made.
- The recovery target for the G-L cell (p*) is now derived analytically - a first-order linearisation of the generator around its stationary point (front_matter, Section 3.1) - replacing the earlier reference-simulation measurement.

## In progress
- Recompute p* from the analytical formula and re-run cell 5 under the unchanged locked protocol.
- Certify a second model under the same frozen protocol: HAR-RV (the econometric RV benchmark) and a small MLP (learned-forecaster comparison). LSTM deferred to a post-exam stretch.

## Reproduce the pilot
```
pip install -r requirements.txt
cd code && python3 garch_ngrc_pilot.py --reps 50      # ~12 s; writes raw_results.csv in the current directory
mv raw_results.csv ../results/ && cd ../report && python3 make_report.py && pdflatex garch_pilot_report.tex
```

## Scope and caveats
- The pilot covers only the GARCH-control part of Objective 1 and used exploratory settings (50 replications, no locked tolerance), so it makes no verdict on H1; the verdict comes from the frozen run described above.
- Pilot spread columns are Monte Carlo spread of the estimator, not confidence intervals.

## License
No license has been specified; all rights reserved by the author for now.
