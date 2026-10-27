# DMU-NACI — Provenance-Controlled Residual Learning with Adaptive Conformal Calibration for Cross-Year PV Prediction Intervals

Code archive for the manuscript:

> **Provenance-Controlled Residual Learning with Adaptive Conformal
> Calibration for Cross-Year PV Prediction Intervals** — C. Shi, C. Jin,
> S. Zhang, Z. Xiao. Manuscript in preparation for submission to
> *Solar Energy* (draft v3, October 2026).

DMU-NACI (**D**ecoupled **M**ean–**U**ncertainty Learning with **N**ormalized
**A**daptive **C**onformal **I**nference) produces 15-minute-ahead probabilistic
forecasts of photovoltaic power on DKASC Alice Springs **Site 18** (5.2 kW,
fixed tilt) with cross-system replication on **Site 1B** (23.4 kW, dual-axis
tracking). This repository contains the principal experimental pipeline and the
archived evidence supporting the reported results — the frozen model
checkpoints and the aggregate result files behind each table and figure.

> **Release status: pipeline and frozen-artifact release — not fully
> end-to-end reproducible from raw CSV.**
> Everything downstream of the frozen data universes is verified to reproduce
> the paper exactly (bit-identical reruns; `docs/REPRODUCIBILITY_REPORT.md`).
> However, the frozen 2024/2025 window universes (inputs of `s2`–`s7`) were
> built by earlier-project tooling whose full dependency closure is not
> redistributable here, and the DKASC raw exports themselves must be downloaded
> from the official source. See `data/README.md` and `docs/MISSING_FILES.md`
> for the precise boundary and `docs/CHANGELOG.md` for every deviation from
> the original project.
>
> **License: not yet open-source.** The `LICENSE` file is an all-rights-
> reserved placeholder pending the rights holders' decision between MIT and
> Apache-2.0 (candidate texts in `docs/license_candidates/`); until then no
> redistribution or modification is licensed.

---

## 1. Method overview

For every 5-minute forecast origin *t*, the input window is `x_t ∈ R^{72×5}`
(six hours of PV active power, GHI, DHI, air temperature, relative humidity)
and the target is the power 15 minutes ahead (`y_{t+3}`).

1. **Mean branch (Section 2.1).** A multi-scale causal TCN encodes the window
   at 5-min, 15-min (3× avg-pool), and 60-min (12× avg-pool) scales — two-layer
   TCN branches with 32 channels each — fuses them into a 64-d embedding
   (36,705 parameters), and outputs the point forecast, clipped to
   `[0, P_rated]`.
2. **Error-scale branch (Section 2.2).** An independent one-layer GRU (hidden
   size 32; 3,777 parameters) reads the standardized raw window and maps it
   through `Linear(32→1)` + `softplus` (+1e-6) to the conditional
   predictive-error *variance* in standardized target space. It is trained with
   the Gaussian heteroscedastic NLL on a residual training pool, so it learns
   (in the ideal limit) the conditional second moment of the standardized
   residuals.
3. **Provenance-controlled residual pools (Section 2.3).** Candidate pools:
   rolling out-of-fold residuals from four expanding-window 2024 folds
   (`E24_OOF`, 28,418), frozen-model archive replay over 2023/2022
   (`E23/E22_replay`, 99,371 / 102,570), and diagnostic controls
   (`T3-match`, `T1-in`, vintage pools `V1/V2/V3`). Nested pools
   `T1' ⊂ T2' ⊂ T3'` are compared on SELECT
   (`median |Std(z)−1| ≤ 0.35`, then minimum NLL); **T3′ (230,359 residuals) is
   the frozen winner** (`results/rigorous_v2/s6_gru_select/selection_decision.json`).
4. **Normalized conformal calibration + delayed ACI (Section 2.4).** Scores
   `s_i = |e_i|/(σ_i+1e-6)` on the CALIB span give the finite-sample quantile
   `Q(a)`; at deployment, delayed ACI updates the miscoverage level with a
   FIFO queue that consumes only **matured** feedback (3 steps = 15 min):
   `α_t = clip(α_{t−1} + γ(α − err_{t−3}), 1e-3, 0.5)`, with γ selected per
   seed on SELECT. The issued interval is
   `C = [μ − q_tσ, μ + q_tσ]` with `q_t = Q(α_t)` (Algorithm 1).

The `2×2` ablation isolates the two mechanisms: A0 (absolute, fixed),
A1 (absolute, ACI), A2 (normalized, fixed), **A3 (normalized, delayed ACI =
DMU-NACI)**. Canonical CQR and CQR+ACI are external comparators.

## 2. Repository layout

```
DMU-NACI/
├── README.md, LICENSE, CITATION.cff, requirements.txt, .gitignore
├── configs/            frozen protocol constants (mirrored by tests)
├── data/               data source/policy README + synthetic format example
├── src/
│   ├── data/           raw schema + frozen 2024/2025 preopen loader
│   ├── models/         MultiScaleTCN, TCN blocks, GRU scale encoders
│   ├── features/       difficulty-state features (dynamic index, historical
│   │                   familiarity) — evaluation only
│   └── eval/           causal-safe window audit machinery
├── experiments/
│   ├── rigorous_v2/    Site 18 pipeline s1–s10 (common_v2.py = protocol core)
│   ├── site1b_validation/  Site 1B replication (reuses common_v2 algorithms)
│   └── cqr_aci/        CQR + CQR-ACI baselines for both sites
├── figures/            paper-figure scripts + shippable source data
├── legacy/preopen_builders/  frozen 2024/2025 universe builders (reference)
├── scripts/            synthetic-data generator, verification helpers
├── tests/              unit, checkpoint, config-consistency, smoke tests
├── checkpoints/        frozen trained weights (site18/, site1b/)
├── results/            frozen aggregate evidence for every paper number
└── docs/               audit reports (mapping, reproducibility, changelog…)
```

The layout follows the original research project (pipeline-first) rather than
the conventional `src/models|data|...` split; the reasons are documented in
`docs/CHANGELOG.md` (§ Layout).

## 3. Environment and installation

```bash
python -m venv .venv && source .venv/bin/activate   # Python ≥ 3.11
pip install -r requirements.txt
# optional: install a CUDA build of torch for training-speed parity
```

The archived results were produced with Python 3.11.15 / torch 2.13.0+cu126 /
numpy 2.4.4 / pandas 3.0.2 / scipy 1.17.1 on a single consumer GPU. All scripts
fall back to CPU automatically. Checkpoint verification (71 tests) and the
calibration-semantics tests run on CPU in seconds.

## 4. Data acquisition

The pipeline reads yearly DKASC CSV exports that must be placed (unmodified
names) in `data/raw/`:

```
data/raw/Alice_Springs_2021.csv … data/raw/Alice_Springs_2025.csv
```

Source: **https://dkasolarcentre.com.au** (Data → Download → yearly CSV).
Both the Site 18 and the Site 1B power channels are contained in the same
site-18-system export, which also carries the shared weather station. Exact
column names and the full data policy: [data/README.md](data/README.md).
`data/synthetic_example.csv` is a small synthetic file in the same schema for
smoke tests.

## 5. Preprocessing and splits (paper Section 3.1, Table 1)

Cleaning (frozen QC rules, no target interpolation) → causal-safe 72×5 windows
with target = origin+15 min → chronological splits: DEV
`[2024-01-01, 2024-08-08)` (internal early-stop tail from 2024-07-17), SELECT
`[2024-08-08, 2024-09-16)`, CALIB `[2024-09-16, 2024-10-02)`, SAME
`[2024-10-02, 2025-01-01)`, CROSS = 2025 (98,239 valid Site 18 windows;
87,839 on Site 1B). 2025 is opened exactly once per pipeline (`s7` / `p8`) and
is used report-only plus 3-step-delayed ACI feedback. Difficulty-state labels
(`HIGH_N`, `HARD_DF`; Section 3.2) are computed from inputs and 2021/2022
references and never enter training or calibration.

## 6. Running the pipeline

All commands run from the repository root. Site 18 (main method):

```bash
python experiments/rigorous_v2/s1_build_data.py        # 2021+2022 cores/windows from raw
python experiments/rigorous_v2/s2_train_final_mean.py  # final mean TCN, 5 seeds
python experiments/rigorous_v2/s3_e24_oof.py           # E24_OOF residuals (4 folds x 5 seeds)
python experiments/rigorous_v2/s4_replay.py            # E22/E23 replay residuals
python experiments/rigorous_v2/s5_vintage.py           # vintage residuals M22/M23/M24
python experiments/rigorous_v2/s6_gru_select.py        # pools -> GRU -> selection -> freeze
python experiments/rigorous_v2/s7_final_eval.py        # CALIB q -> SAME/CROSS + A0-A3
python experiments/rigorous_v2/s8_report.py            # consolidated audit report
python experiments/cqr_aci/s9_cqr_aci_site18.py        # CQR gate -> CQR+ACI
python experiments/rigorous_v2/s10_two_level_bootstrap.py  # two-level bootstrap CIs
```

Site 1B (same protocol, site-specific training):

```bash
python experiments/site1b_validation/phase01_audit.py
python experiments/site1b_validation/p2_clean.py
python experiments/site1b_validation/p34_mean_resid.py
python experiments/site1b_validation/p567_gru_freeze.py
python experiments/site1b_validation/p8_eval.py
python experiments/site1b_validation/p8b_site18_cqr.py
python experiments/site1b_validation/p9_report.py
```

Notes: `s1`/`p2` regenerate cleaned data from the raw downloads; `s2`–`s10`
write into `results/` and skip work whose checkpoint already exists. The
2024/2025 preopen universes (windows + difficulty labels) are frozen inputs of
`s2`/`s7`; they are regenerated from raw data by the legacy builders under
`legacy/preopen_builders/`, whose extra dependency closure is documented in
`docs/MISSING_FILES.md`.

## 7. Tables and figures

| Paper item | Evidence file | Regeneration |
|---|---|---|
| Table 1 (splits) | `results/rigorous_v2/data/*_windows_audit.json`, run manifests | `s1` + `s2`/`s7` manifests |
| Table 2 (point, both sites) | `results/rigorous_v2/s7_final_eval/run_manifest.json`, `results/site1b_val/site1b_eval_per_seed.csv` | `s7` / `p8` |
| Table 3 (residual pools) | `results/rigorous_v2/s7_final_eval/metrics_per_seed.csv` (A2 columns) | `s7` |
| Table 4 (calibration ablation) | same file (A0–A3 columns) | `s7` |
| Table 5 (cross-system A0–A3 + CQR) | `results/rigorous_v2/s9_cqr_aci/cqr_aci_per_seed.csv`, `results/site1b_val/cqr_aci/cqr_aci_per_seed.csv`, `results/site1b_val/site1b_eval_per_seed.csv` | `s9_*` / `p8` |
| Fig. 3 (input representation), Fig. 4 (residual pools), Fig. 6 (difficult states), Fig. 7 (cross-system) | `figures/figure_source_data/` + pipeline CSVs | verified byte-identical — see `figures/README.md` (asset file names are legacy labels; the mapping there is authoritative) |
| Fig. 5 (ablation), Fig. S1 (representative days) | need per-sample artifacts that are excluded from the archive (data policy) | rerun `s7` (+`s0_replay_intervals.py`) on your own data |
| Tables S1, S2 | `results/supplementary/*.csv` | legacy stage generators (see `docs/MISSING_FILES.md`) |

`python scripts/verify_paper_numbers.py` recomputes Tables 3/4/5 medians from
the evidence CSVs and compares them against the values printed in the
manuscript.

## 8. Tests and verification

```bash
python -m pytest tests/ -q          # 100+ tests: semantics, architecture, configs, checkpoints
python -m pytest tests/test_smoke_synthetic_pipeline.py -q   # full mini-pipeline on synthetic data (needs torch; ~1 min)
```

What is covered: conformal finite-sample quantile rule; delayed-ACI FIFO
semantics (3-step maturity, α clipping, no-future-update assert); interval
score; per-seed γ selection rule; model parameter counts vs the paper
(36,705 / 3,777); every archived checkpoint loads into its exact architecture
and matches the freeze-manifest SHA-256 registry; configs match code constants;
end-to-end synthetic smoke run. Execution evidence: `docs/REPRODUCIBILITY_REPORT.md`.

## 9. Verified vs unverified scope

**Re-executed during this archive's preparation (PASS evidence in
`docs/REPRODUCIBILITY_REPORT.md`):** paper Figs. 3/4/6/7 regenerate
byte-identically from the shipped evidence (final-manuscript numbering);
the s7/s9 evaluations and the Site 1B p8 evaluation re-run from the frozen
checkpoints against the original data bit-identically (max |Δ| = 0); all
unit/checkpoint/config/smoke tests pass.

**Frozen-artifact level (archived, not retrained here):** full s2–s5 training
runs are seeded and repeatable in principle but take GPU-hours; the archived
checkpoints (SHA-256-registered in `results/rigorous_v2/FREEZE_MANIFEST.json`)
are the ones that produced the paper numbers.

**Not runnable from this archive alone:** regeneration of the 2024/2025
preopen universes from raw (legacy closure — `docs/MISSING_FILES.md`);
Fig. 5 (ablation) and Fig. S1 (representative days) without per-sample
artifacts; Table S1/S2 regeneration (legacy diagnostic generators).

## 10. License and citation

**This repository is not yet open-source software.** `LICENSE` is an
all-rights-reserved placeholder: running the code locally for verification of
the manuscript is fine, but redistribution and modification are not licensed
until the rights holders choose a license. The candidate texts (MIT,
Apache-2.0) are provided verbatim in `docs/license_candidates/` for that
decision; the selected text will replace `LICENSE` and the license field in
`CITATION.cff` at that point. Third-party packages keep their own licenses.

Data: DKASC Alice Springs, https://dkasolarcentre.com.au (data made available
on request per the paper's Data Availability statement).

Citation: see [CITATION.cff](CITATION.cff) (BibTeX will be added on
publication).

## 11. Archive provenance

This repository was produced by a controlled archival pass over the original
research project: every code file is either a verbatim copy or carries a
line-level change record in `docs/CHANGELOG.md`; the mapping from paper
sections to code files is `docs/PAPER_CODE_MAPPING.md`; the security/dependency
audit is `docs/CODE_AUDIT_REPORT.md`; missing modules and weights are listed in
`docs/MISSING_FILES.md`. The original project remains the authoritative record.
