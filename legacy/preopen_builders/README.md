# Legacy preopen builders (reference only)

These two scripts are the **original generators of the frozen 2024/2025 window
universes** that the DMU-NACI pipeline consumes as inputs:

- `37rh_2024_preopen.py` — built `data/processed/external_2024_preopen/`
  (`X_2024_preopen.npy`, `2024_core.csv`, `2024_flags.csv`,
  `2024_context_preopen.csv` with the `N_group`/`HARD_DF` labels) and the
  aligned `outputs/predictions/stage37r_2024_preopen/*.csv` manifests.
- `37rj_2025_preopen.py` — same for 2025.

They are preserved verbatim for provenance because the paper's difficulty-state
labels (Section 3.2) and window counts (Table 1) originate here.

**They are NOT runnable inside this release as-is.** Beyond the modules that
ship here (`src/data`, `src/features`, `src/eval`,
`src/models/multiscale_tcn`), they import components of the legacy chain that
are not part of this archive — `src/models/bscn.py`,
`src/eval/compact_final_reliability.py`, `src/features/scaler.py` users in the
legacy calibration step — and read frozen artifacts of earlier project stages
(causal-clean TCN checkpoint, historical-memory CSV, dynamic-index /
familiarity threshold JSONs, protocol lock files). The full closure is
itemized in `docs/MISSING_FILES.md`.

Consequences for reproducibility:

- Everything downstream (s2–s10, Site 1B pipeline, all paper tables) is fully
  reproducible given (a) the DKASC downloads and (b) the preopen artifacts,
  which any user can rebuild with these scripts once the missing legacy pieces
  are obtained from the original project.
- The window construction itself (cleaning + causal-safe windows) is fully
  self-contained in the release and is exercised by `s1` for 2021/2022 and by
  the synthetic smoke test.
