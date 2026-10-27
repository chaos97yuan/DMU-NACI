# Figures — provenance and regeneration

**Numbering below follows the FINAL manuscript** (`史超远初稿_v3.pdf` /
`latex_v8`, 7 main figures + supplementary Fig. S1 + Tables S1–S2). Note that
the *file names* of several assets (`figS3_representation.png`,
`fig3_provenance.png`, …) are legacy labels from earlier manuscript versions
and do NOT match the final figure numbers; the mapping table is authoritative.
Scripts are kept under their original names (verbatim archive copies).

| Final paper item | Asset in `latex_v8/figs/` | Script in this repo | Inputs | Status in this archive |
|---|---|---|---|---|
| **Fig. 1** temporal provenance schematic | `Fig1_Temporal_Provenance_Revised(1).jpg` | — (drawing-tool asset, not code-generated) | — | manuscript asset |
| **Fig. 2** framework schematic | `fig1_framework.jpg` | — (drawing-tool asset) | — | manuscript asset |
| **Fig. 3** input representation (R0/R1/R2) | `figS3_representation.png` | `latex_v4_repair/fig5_representation_repair.py` | `results/stage8_final_architecture/05_stage1_confirmatory/metrics_per_seed.csv` | **verified byte-identical** (md5 `3830038d…`) |
| **Fig. 4** residual pools | `fig3_provenance.png` | `latex_v4_repair/fig6_provenance_pools_repair.py` | `results/rigorous_v2/s7_final_eval/metrics_per_seed.csv`, `monthly_cross_year.csv` | **verified byte-identical** (md5 `f80c83ad…`) |
| **Fig. 5** A0–A3 ablation | `fig4_ablation.png` | `latex_v4_repair/fig8_ablation_dynamic_repair.py` | per-sample interval parquets (`figure_scripts/s0_replay_intervals.py` from `s7` per-sample parquets) | script included; needs per-sample artifacts (data policy: excluded) |
| **Fig. 6** difficult-state (HARD_DF) | `fig5_harddf.png` | `latex_v5_repair/fig5_harddf_conditional_v5.py` | `figure_source_data/Fig9_*.csv` | **verified byte-identical** (md5 `eda45479…`) |
| **Fig. 7** cross-system | `fig6_crosssystem.png` | `latex_v5_repair/fig6_crosssystem_v5.py` | `figure_source_data/Fig10_cross_system_source.csv`, `results/rigorous_v2/s9_cqr_aci/cqr_aci_per_seed.csv`, `results/site1b_val/cqr_aci/cqr_aci_per_seed.csv` | **verified byte-identical** (md5 `5357f8ae…`) |
| **Fig. S1** representative interval days | `figS4_examples.png` | `latex_v4_repair/fig7_interval_examples_repair.py` | `figure_source_data/Fig7_case_days.csv` + per-sample intervals parquet | script included; needs per-sample artifacts |
| Table S1 (backbones), Table S2 (representation/encoders) | — (values in `results/supplementary/`) | generators are legacy stage tools (see `docs/MISSING_FILES.md` §3) | archived evidence CSVs | evidence archived |

Not referenced in the final manuscript (kept as auxiliary asset generators):
`latex_v4_repair/fig1_data_overview_repair.py` (data overview,
`figS1_dataoverview.png` in earlier drafts), `latex_v4_repair/
fig2_3_schematics_repair.py` (multi-scale TCN schematic, `figS2_tcn.png`),
and `latex_v5_repair/figS5_rolling_v5.py` (rolling-coverage figure). All run,
but the final PDF cites none of them. The `latex_v4_repair/
fig4_pipeline_timeline_repair.py` and `latex_v4_repair/
fig9_conditional_harddf_repair.py` scripts belong to the same earlier-draft
generation chain (Fig.1-precursor and Fig.6-predecessor respectively).

The `stage9_5 figure_scripts/` directory preserves the original generation
scripts from the pre-repair era (same data, earlier layout); the
`latex_v4_repair`/`latex_v5_repair` scripts produced the exact PNGs shipped
with the manuscript. Outputs go to `figures/output_stage95/`, `output_v4/`,
`output_v5/` respectively (git-ignored, regenerated on demand).

Shipped source CSVs under `figure_source_data/` are aggregates (per-seed
metrics, daily rolling coverage, day-selection metadata) and contain no
per-sample observations. The per-sample parquets/interval CSVs are excluded
under the data policy in `data/README.md`; rerunning `s7` (with the DKASC data)
regenerates them.

Verification record (this archive, 2026-10-27): the four main-text figures
marked byte-identical (Fig. 3/4/6/7) were regenerated inside `DMU-NACI/` and
compared by MD5 against `manuscript/latex_v8/figs/` of the original project —
all matched exactly.
