"""configs/*.yaml must stay consistent with the authoritative code constants.

The YAML files are documentation mirrors; this test is what keeps them honest.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from experiments.rigorous_v2 import common_v2 as C
from experiments.site1b_validation import common_1b as C1B
from src.data.schema import FEATURE_COLUMNS, HISTORY_STEPS, HORIZON_STEPS, SAMPLING_MINUTES
from src.models.multiscale_tcn import MultiScaleTCN, count_trainable_parameters
from src.models.uncertainty_encoders import ScaleModel

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_site18_yaml_matches_code():
    cfg = _load("site18.yaml")
    assert cfg["site"]["rated_power_kw"] == C.P_RATED
    assert cfg["seeds"] == C.SEEDS
    assert cfg["sampling"]["interval_minutes"] == C.SAMPLING_MIN
    assert cfg["sampling"]["history_steps"] == C.HISTORY_STEPS == HISTORY_STEPS
    assert cfg["sampling"]["horizon_steps"] == C.HORIZON_STEPS == HORIZON_STEPS
    assert cfg["sampling"]["input_features"] == FEATURE_COLUMNS
    spans = cfg["spans"]
    assert spans["mean_gradient"] == list(C.MEAN_GRAD)
    assert spans["mean_early_stop"] == list(C.MEAN_ES)
    assert spans["select"] == list(C.SELECT)
    assert spans["calib"] == list(C.CALIB)
    assert spans["same"] == list(C.SAME)
    assert spans["cross"] == list(C.CROSS)
    folds = [{"fold": f["fold"], "es_start": f["es_start"], "train_end": f["train_end"],
              "fc": [str(f["fc"][0]), str(f["fc"][1])]} for f in C.OOF_FOLDS]
    assert cfg["oof_folds"] == folds
    vint = [{"tag": v["tag"], "train_years": v["years"], "forecast_year": v["fc_year"],
             "es_start": v["es_start"]} for v in C.VINTAGE]
    assert cfg["vintage"] == vint
    ca = cfg["conformal_aci"]
    assert ca["alpha"] == C.ALPHA
    assert ca["alpha_bounds"] == [C.ALPHA_MIN, C.ALPHA_MAX]
    assert ca["epsilon"] == C.EPS
    assert ca["feedback_delay_minutes"] == C.ACI_DELAY_MIN
    assert ca["gamma_grid"] == C.GAMMA_GRID
    ev = cfg["evaluation"]
    assert ev["daylight_rule"] == f"origin GHI > {C.DAYLIGHT_GHI_THRESHOLD:g} W/m2"
    assert ev["bootstrap"]["replicates"] == C.N_BOOTSTRAP
    assert ev["bootstrap"]["draw_seed"] == 2026  # s10_two_level_bootstrap.RNG_SEED
    assert cfg["models"]["mean_tcn"]["params"] == count_trainable_parameters(MultiScaleTCN())
    assert cfg["models"]["gru_scale"]["params"] == ScaleModel("GRU").parameter_report()["total_params"]


def test_site1b_yaml_matches_code():
    cfg = _load("site1b.yaml")
    assert cfg["site"]["rated_power_kw"] == C1B.P_RATED_1B
    assert cfg["site"]["raw_columns"] == C1B.RAW_COLS
    assert cfg["site"]["feature_order"] == C1B.FEAT_ORDER
    assert cfg["protocol_inherited_from_site18"]["seeds"] == C.SEEDS
    assert cfg["cleaning"]["power_cap_kw"] == C1B.POWER_CAP_KW
    assert cfg["evaluation"]["rated_normalization"] == C1B.P_RATED_1B
