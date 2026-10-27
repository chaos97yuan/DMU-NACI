# Figures — provenance and regeneration

|Final paper item|Asset in `latex\_v8/figs/`|Script in this repo|Inputs|Status in this archive|
|-|-|-|-|-|
|**Fig. 1** temporal provenance schematic|`Fig1\_Temporal\_Provenance\_Revised(1).jpg`|— (drawing-tool asset, not code-generated)|—|manuscript asset|
|**Fig. 2** framework schematic|`fig1\_framework.jpg`|— (drawing-tool asset)|—|manuscript asset|
|**Fig. 3** input representation (R0/R1/R2)|`figS3\_representation.png`|`latex\_v4\_repair/fig5\_representation\_repair.py`|`results/stage8\_final\_architecture/05\_stage1\_confirmatory/metrics\_per\_seed.csv`|**verified byte-identical** (md5 `3830038d…`)|
|**Fig. 4** residual pools|`fig3\_provenance.png`|`latex\_v4\_repair/fig6\_provenance\_pools\_repair.py`|`results/rigorous\_v2/s7\_final\_eval/metrics\_per\_seed.csv`, `monthly\_cross\_year.csv`|**verified byte-identical** (md5 `f80c83ad…`)|
|**Fig. 5** A0–A3 ablation|`fig4\_ablation.png`|`latex\_v4\_repair/fig8\_ablation\_dynamic\_repair.py`|per-sample interval parquets (`figure\_scripts/s0\_replay\_intervals.py` from `s7` per-sample parquets)|script included; needs per-sample artifacts (data policy: excluded)|
|**Fig. 6** difficult-state (HARD\_DF)|`fig5\_harddf.png`|`latex\_v5\_repair/fig5\_harddf\_conditional\_v5.py`|`figure\_source\_data/Fig9\_\*.csv`|**verified byte-identical** (md5 `eda45479…`)|
|**Fig. 7** cross-system|`fig6\_crosssystem.png`|`latex\_v5\_repair/fig6\_crosssystem\_v5.py`|`figure\_source\_data/Fig10\_cross\_system\_source.csv`, `results/rigorous\_v2/s9\_cqr\_aci/cqr\_aci\_per\_seed.csv`, `results/site1b\_val/cqr\_aci/cqr\_aci\_per\_seed.csv`|**verified byte-identical** (md5 `5357f8ae…`)|
|**Fig. S1** representative interval days|`figS4\_examples.png`|`latex\_v4\_repair/fig7\_interval\_examples\_repair.py`|`figure\_source\_data/Fig7\_case\_days.csv` + per-sample intervals parquet|script included; needs per-sample artifacts|
|Table S1 (backbones), Table S2 (representation/encoders)|— (values in `results/supplementary/`)|generators are legacy stage tools (see `docs/MISSING\_FILES.md` §3)|archived evidence CSVs|evidence archived|

Not referenced in the final manuscript (kept as auxiliary asset generators):
`latex\_v4\_repair/fig1\_data\_overview\_repair.py` (data overview,
`figS1\_dataoverview.png` in earlier drafts), `latex\_v4\_repair/ fig2\_3\_schematics\_repair.py` (multi-scale TCN schematic, `figS2\_tcn.png`),
and `latex\_v5\_repair/figS5\_rolling\_v5.py` (rolling-coverage figure). All run,
but the final PDF cites none of them. The `latex\_v4\_repair/ fig4\_pipeline\_timeline\_repair.py` and `latex\_v4\_repair/ fig9\_conditional\_harddf\_repair.py` scripts belong to the same earlier-draft
generation chain (Fig.1-precursor and Fig.6-predecessor respectively).

The `stage9\_5 figure\_scripts/` directory preserves the original generation
scripts from the pre-repair era (same data, earlier layout); the
`latex\_v4\_repair`/`latex\_v5\_repair` scripts produced the exact PNGs shipped
with the manuscript. Outputs go to `figures/output\_stage95/`, `output\_v4/`,
`output\_v5/` respectively (git-ignored, regenerated on demand).

Shipped source CSVs under `figure\_source\_data/` are aggregates (per-seed
metrics, daily rolling coverage, day-selection metadata) and contain no
per-sample observations. The per-sample parquets/interval CSVs are excluded
under the data policy in `data/README.md`; rerunning `s7` (with the DKASC data)
regenerates them.

Verification record (this archive, 2026-10-27): the four main-text figures
marked byte-identical (Fig. 3/4/6/7) were regenerated inside `DMU-NACI/` and
compared by MD5 against `manuscript/latex\_v8/figs/` of the original project —
all matched exactly.

