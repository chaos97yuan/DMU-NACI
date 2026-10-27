#!/usr/bin/env python3
"""
Conservative, traceable cleaning for DKASC Alice Springs 2022 Site 18 data.

This script creates cleaned 2022 data products and quality flags. It does not
split datasets, normalize variables, create time features, build windows,
compute D_t, or train any model.
"""

from pathlib import Path
import json
import logging
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
    import pandas as pd

    MPLCONFIGDIR = PROJECT_ROOT / "outputs" / ".matplotlib"
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Missing required package. Please install only the current cleaning dependencies:\n"
        "    pip install pandas numpy matplotlib\n"
        f"Original import error: {exc}"
    )

from src.data.schema import (
    COLUMN_MAP,
    NUMERIC_COLUMNS,
    OUTPUT_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    SITE_RATED_POWER_KW as SHARED_SITE_RATED_POWER_KW,
)


# 工程初始清洗参数，不是论文最终不可变参数。
YEAR = 2022
FREQ = "5min"
EXPECTED_ROWS = 365 * 24 * 12
YEAR_START = pd.Timestamp(f"{YEAR}-01-01 00:00:00")
NEXT_YEAR_START = pd.Timestamp(f"{YEAR + 1}-01-01 00:00:00")
YEAR_END = NEXT_YEAR_START - pd.Timedelta(minutes=5)

# 工程初始清洗参数：Site 18额定功率近似值，仅用于标记和强异常候选。
SITE_RATED_POWER_KW = SHARED_SITE_RATED_POWER_KW
POWER_HARD_MAX_KW = 6.0

# 工程初始清洗参数：当前只用于清洗辅助判断，不是论文最终昼夜定义。
DAYLIGHT_GHI_THRESHOLD = 20.0
NIGHT_NEGATIVE_POWER_TOLERANCE_KW = 0.05

# 工程初始清洗参数：辐照度小负数容差与强异常上限。
RADIATION_SMALL_NEGATIVE_TOLERANCE = 5.0
RADIATION_HIGH_FLAG = 1200.0
RADIATION_HARD_MAX = 1500.0

# 工程初始清洗参数：相对湿度按百分比处理；超过硬阈值需人工确认。
RH_MIN = 0.0
RH_MAX = 100.0
RH_SANITY_HARD_MAX = 110.0

# 工程初始清洗参数：只允许非常局部、双边短缺失插值。
MAX_GAP_POWER = 2
MAX_GAP_GHI = 2
MAX_GAP_DHI = 2
MAX_GAP_TEMP = 6
MAX_GAP_HUMIDITY = 6
MAX_GAP_TILTED = 3

SHOW_PLOTS = False

CORE_COLUMNS = [
    "timestamp",
    "Active_Power",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
]

EXTENDED_COLUMNS = [
    "timestamp",
    "Active_Power",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Weather_Daily_Rainfall",
]

QUALITY_FLAG_COLUMNS = [
    "timestamp",
    "timestamp_was_missing",
    "power_imputed",
    "power_corrected_night_negative",
    "power_invalid_negative",
    "power_above_rated",
    "power_invalid_high",
    "ghi_imputed",
    "dhi_imputed",
    "temperature_imputed",
    "humidity_imputed",
    "humidity_clipped_high",
    "Global_Tilted_imputed",
    "Diffuse_Tilted_imputed",
    "radiation_corrected_small_negative",
    "radiation_invalid_negative",
    "ghi_high_flag",
    "radiation_global_tilted_high_flag",
    "humidity_invalid_low",
    "rainfall_invalid_negative",
    "row_has_core_missing",
    "row_has_any_imputation",
    "segment_id",
]


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def setup_logging(project_root: Path) -> logging.Logger:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "02_clean_data.log"

    logger = logging.getLogger("clean_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(stream_handler)

    logger.info("Script started")
    logger.info("Log file: %s", log_path)
    return logger


def ensure_output_dirs(project_root: Path) -> tuple[Path, Path, Path]:
    processed_dir = project_root / "data" / "processed"
    metrics_dir = project_root / "outputs" / "metrics"
    figures_dir = project_root / "outputs" / "figures" / "data_cleaning"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir, metrics_dir, figures_dir


def find_raw_csv(project_root: Path) -> Path:
    raw_dir = project_root / "data" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in raw data directory: {raw_dir}")
    if len(csv_files) > 1:
        print_section("Multiple CSV Files Found")
        for path in csv_files:
            print(f"- {path.name}")
        raise RuntimeError("Multiple CSV files found. Please specify the intended raw data file explicitly.")
    return csv_files[0]


def validate_required_columns(csv_path: Path) -> None:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in header]
    if missing:
        missing_text = "\n".join(f"- {column}" for column in missing)
        raise ValueError(
            "The raw CSV is missing required DKASC columns:\n"
            f"{missing_text}\n"
            "Do not guess replacement fields. Check data version and field names."
        )


def load_required_data(csv_path: Path, logger: logging.Logger) -> pd.DataFrame:
    validate_required_columns(csv_path)
    logger.info("Start reading selected 11 columns from %s", csv_path)
    print("开始读取数据...")
    raw = pd.read_csv(csv_path, usecols=REQUIRED_RAW_COLUMNS)
    print("读取完成")
    logger.info("Finished reading raw data. Rows=%s, columns=%s", len(raw), len(raw.columns))

    df = raw.rename(columns=COLUMN_MAP).loc[:, OUTPUT_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def restrict_to_2022(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, dict]:
    valid_ts = df["timestamp"].dropna()
    original_start = valid_ts.min() if not valid_ts.empty else pd.NaT
    original_end = valid_ts.max() if not valid_ts.empty else pd.NaT
    in_2022 = (df["timestamp"] >= YEAR_START) & (df["timestamp"] < NEXT_YEAR_START)
    dropped_2023_rows = int((df["timestamp"] >= NEXT_YEAR_START).sum())
    dropped_other_rows = int((~in_2022).sum() - dropped_2023_rows)
    df_2022 = df.loc[in_2022].copy()

    duplicate_count = int(df_2022["timestamp"].duplicated(keep=False).sum())
    if duplicate_count:
        raise ValueError(
            f"2022 data contains {duplicate_count} duplicate timestamp rows. "
            "No duplicate-resolution cleaning rule is defined for 02_clean_data.py."
        )

    info = {
        "original_start": original_start,
        "original_end": original_end,
        "raw_total_rows": int(len(df)),
        "raw_2022_rows": int(len(df_2022)),
        "dropped_2023_rows": dropped_2023_rows,
        "dropped_other_out_of_range_rows": dropped_other_rows,
    }

    print_section("Restrict To 2022")
    for key, value in info.items():
        print(f"{key}: {value}")
    logger.info("Dropped 2023 rows: %s", dropped_2023_rows)
    logger.info("2022 raw rows after restriction: %s", len(df_2022))
    return df_2022, info


def build_full_2022_timeline() -> pd.DatetimeIndex:
    timeline = pd.date_range(YEAR_START, YEAR_END, freq=FREQ)
    if len(timeline) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected 2022 timeline length {EXPECTED_ROWS}, got {len(timeline)}")
    return timeline


def mark_original_missing_timestamps(df_2022: pd.DataFrame, timeline: pd.DatetimeIndex, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_timestamps = pd.DatetimeIndex(df_2022["timestamp"].dropna().sort_values())
    missing_timestamps = timeline.difference(existing_timestamps)

    reindexed = df_2022.sort_values("timestamp").set_index("timestamp").reindex(timeline)
    reindexed.index.name = "timestamp"
    reindexed = reindexed.reset_index()

    flags = pd.DataFrame({"timestamp": reindexed["timestamp"]})
    flags["timestamp_was_missing"] = flags["timestamp"].isin(missing_timestamps).astype(int)

    missing_count = int(flags["timestamp_was_missing"].sum())
    print_section("Reindex To Full 2022 Timeline")
    print(f"Expected rows: {EXPECTED_ROWS}")
    print(f"Reindexed rows: {len(reindexed)}")
    print(f"Original missing timestamps restored as NaN rows: {missing_count}")
    if missing_count != 2217:
        logger.warning("Restored missing timestamp count is %s, expected audit reference was 2217", missing_count)
        print(f"WARNING: restored missing timestamp count is {missing_count}, audit reference was 2217.")
    logger.info("Reindexed rows=%s, timestamp_was_missing=%s", len(reindexed), missing_count)
    return reindexed, flags


def initialize_quality_flags(flags: pd.DataFrame) -> pd.DataFrame:
    initialized = flags.copy()
    for column in QUALITY_FLAG_COLUMNS:
        if column not in initialized.columns:
            initialized[column] = 0
    initialized["segment_id"] = -1
    return initialized.loc[:, QUALITY_FLAG_COLUMNS].copy()


def identify_nan_runs(series: pd.Series) -> pd.DataFrame:
    is_nan = series.isna().to_numpy()
    rows = []
    start = None
    for idx, missing in enumerate(is_nan):
        if missing and start is None:
            start = idx
        elif not missing and start is not None:
            rows.append({"start_pos": start, "end_pos": idx - 1, "length": idx - start})
            start = None
    if start is not None:
        rows.append({"start_pos": start, "end_pos": len(is_nan) - 1, "length": len(is_nan) - start})
    return pd.DataFrame(rows, columns=["start_pos", "end_pos", "length"])


def generate_gap_summary(flags: pd.DataFrame, metrics_dir: Path) -> pd.DataFrame:
    missing_marker = flags["timestamp_was_missing"].replace({0: 0.0, 1: np.nan})
    runs = identify_nan_runs(missing_marker)
    rows = []
    for gap_id, row in enumerate(runs.itertuples(index=False), start=1):
        missing_points = int(row.length)
        if missing_points <= 2:
            category = "short"
        elif missing_points <= 6:
            category = "small"
        elif missing_points <= 12:
            category = "medium"
        elif missing_points <= 72:
            category = "long"
        else:
            category = "very_long"
        rows.append(
            {
                "gap_id": gap_id,
                "start_timestamp": flags.loc[row.start_pos, "timestamp"],
                "end_timestamp": flags.loc[row.end_pos, "timestamp"],
                "missing_points": missing_points,
                "duration_minutes": missing_points * 5,
                "duration_hours": missing_points * 5 / 60,
                "category": category,
            }
        )
    result = pd.DataFrame(
        rows,
        columns=[
            "gap_id",
            "start_timestamp",
            "end_timestamp",
            "missing_points",
            "duration_minutes",
            "duration_hours",
            "category",
        ],
    )
    summarized_missing_points = int(result["missing_points"].sum()) if not result.empty else 0
    flagged_missing_points = int(flags["timestamp_was_missing"].sum())
    if summarized_missing_points != flagged_missing_points:
        raise RuntimeError("Gap summary missing_points does not match timestamp_was_missing total.")
    result.to_csv(metrics_dir / "cleaning_gap_summary.csv", index=False)
    return result


def drop_unusable_features() -> list[dict]:
    return [
        {"column": "Wind_Speed", "reason": "100% missing in 2022"},
        {
            "column": "Wind_Direction",
            "reason": "negative values present, not a current core PV predictor, requires separate periodic encoding if reintroduced",
        },
    ]


def audit_preclean_values(df: pd.DataFrame, metrics_dir: Path) -> dict:
    negative_power = df.loc[df["Active_Power"] < 0, "Active_Power"].dropna()
    ghi = df["Global_Horizontal_Radiation"]
    rh = df["Weather_Relative_Humidity"]
    night = ghi <= DAYLIGHT_GHI_THRESHOLD
    day = ghi > DAYLIGHT_GHI_THRESHOLD

    if negative_power.empty:
        negative_stats = {"min": np.nan, "1%": np.nan, "5%": np.nan, "median": np.nan, "max": np.nan}
    else:
        negative_stats = {
            "min": negative_power.min(),
            "1%": negative_power.quantile(0.01),
            "5%": negative_power.quantile(0.05),
            "median": negative_power.median(),
            "max": negative_power.max(),
        }

    rh_stats = {
        "rh_above_100": int((rh > 100).sum()),
        "rh_100_101": int(((rh > 100) & (rh <= 101)).sum()),
        "rh_101_102": int(((rh > 101) & (rh <= 102)).sum()),
        "rh_102_105": int(((rh > 102) & (rh <= 105)).sum()),
        "rh_above_105": int((rh > 105).sum()),
        "rh_max": rh.max(),
        "rh_95": rh.quantile(0.95),
        "rh_99": rh.quantile(0.99),
        "rh_99_9": rh.quantile(0.999),
    }

    result = {
        "power_negative_total": int((df["Active_Power"] < 0).sum()),
        "power_negative_night_count": int(((df["Active_Power"] < 0) & night).sum()),
        "power_negative_day_count": int(((df["Active_Power"] < 0) & day).sum()),
        "power_negative_stats": negative_stats,
        "humidity_above_100_stats": rh_stats,
    }

    report_rows = []
    for key, value in negative_stats.items():
        report_rows.append({"section": "negative_power", "metric": key, "value": value})
    report_rows.extend(
        [
            {"section": "negative_power", "metric": "night_count", "value": result["power_negative_night_count"]},
            {"section": "negative_power", "metric": "day_count", "value": result["power_negative_day_count"]},
        ]
    )
    for key, value in rh_stats.items():
        report_rows.append({"section": "humidity_above_100", "metric": key, "value": value})
    pd.DataFrame(report_rows).to_csv(metrics_dir / "preclean_value_audit.csv", index=False)
    return result


def clean_active_power(df: pd.DataFrame, flags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cleaned = df.copy()
    power = cleaned["Active_Power"]
    ghi = cleaned["Global_Horizontal_Radiation"]
    negative = power < 0
    obvious_negative = power < -NIGHT_NEGATIVE_POWER_TOLERANCE_KW
    night_small_negative = (
        (ghi <= DAYLIGHT_GHI_THRESHOLD)
        & negative
        & (power >= -NIGHT_NEGATIVE_POWER_TOLERANCE_KW)
    )
    daytime_negative = (ghi > DAYLIGHT_GHI_THRESHOLD) & negative
    invalid_negative = obvious_negative | daytime_negative

    above_rated = (power > SITE_RATED_POWER_KW) & (power <= POWER_HARD_MAX_KW)
    invalid_high = power > POWER_HARD_MAX_KW

    flags.loc[night_small_negative.fillna(False), "power_corrected_night_negative"] = 1
    flags.loc[invalid_negative.fillna(False), "power_invalid_negative"] = 1
    flags.loc[above_rated.fillna(False), "power_above_rated"] = 1
    flags.loc[invalid_high.fillna(False), "power_invalid_high"] = 1

    cleaned.loc[night_small_negative.fillna(False), "Active_Power"] = 0.0
    cleaned.loc[invalid_negative.fillna(False), "Active_Power"] = np.nan
    cleaned.loc[invalid_high.fillna(False), "Active_Power"] = np.nan

    stats = {
        "power_negative_total": int(negative.sum()),
        "power_night_negative_to_zero": int(night_small_negative.sum()),
        "power_invalid_negative_to_nan": int(invalid_negative.sum()),
        "power_above_rated_kept": int(above_rated.sum()),
        "power_extreme_high_to_nan": int(invalid_high.sum()),
        "power_max_before_qc": power.max(),
    }
    return cleaned, flags, stats


def clean_radiation_values(df: pd.DataFrame, flags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cleaned = df.copy()
    stats = {}
    radiation_columns = [
        "Global_Horizontal_Radiation",
        "Diffuse_Horizontal_Radiation",
        "Radiation_Global_Tilted",
        "Radiation_Diffuse_Tilted",
    ]

    corrected_small_total = 0
    invalid_negative_total = 0
    for column in radiation_columns:
        values = cleaned[column]
        small_negative = (values < 0) & (values >= -RADIATION_SMALL_NEGATIVE_TOLERANCE)
        invalid_negative = values < -RADIATION_SMALL_NEGATIVE_TOLERANCE
        corrected_small_total += int(small_negative.sum())
        invalid_negative_total += int(invalid_negative.sum())
        cleaned.loc[small_negative.fillna(False), column] = 0.0
        cleaned.loc[invalid_negative.fillna(False), column] = np.nan

    flags.loc[:, "radiation_corrected_small_negative"] = 0
    flags.loc[:, "radiation_invalid_negative"] = 0
    any_small_negative = pd.Series(False, index=cleaned.index)
    any_invalid_negative = pd.Series(False, index=cleaned.index)
    for column in radiation_columns:
        original = df[column]
        any_small_negative |= ((original < 0) & (original >= -RADIATION_SMALL_NEGATIVE_TOLERANCE)).fillna(False)
        any_invalid_negative |= (original < -RADIATION_SMALL_NEGATIVE_TOLERANCE).fillna(False)
    flags.loc[any_small_negative, "radiation_corrected_small_negative"] = 1
    flags.loc[any_invalid_negative, "radiation_invalid_negative"] = 1

    ghi_high = cleaned["Global_Horizontal_Radiation"] > RADIATION_HIGH_FLAG
    ghi_invalid_high = cleaned["Global_Horizontal_Radiation"] > RADIATION_HARD_MAX
    tilted_high = cleaned["Radiation_Global_Tilted"] > RADIATION_HIGH_FLAG
    tilted_invalid_high = cleaned["Radiation_Global_Tilted"] > RADIATION_HARD_MAX

    flags.loc[ghi_high.fillna(False), "ghi_high_flag"] = 1
    flags.loc[tilted_high.fillna(False), "radiation_global_tilted_high_flag"] = 1
    cleaned.loc[ghi_invalid_high.fillna(False), "Global_Horizontal_Radiation"] = np.nan
    cleaned.loc[tilted_invalid_high.fillna(False), "Radiation_Global_Tilted"] = np.nan

    stats.update(
        {
            "radiation_small_negative_to_zero": corrected_small_total,
            "radiation_invalid_negative_to_nan": invalid_negative_total,
            "ghi_high_flagged": int(ghi_high.sum()),
            "ghi_extreme_high_to_nan": int(ghi_invalid_high.sum()),
            "radiation_global_tilted_high_flagged": int(tilted_high.sum()),
            "radiation_global_tilted_extreme_high_to_nan": int(tilted_invalid_high.sum()),
        }
    )
    return cleaned, flags, stats


def clean_humidity(df: pd.DataFrame, flags: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cleaned = df.copy()
    rh = cleaned["Weather_Relative_Humidity"]
    above_100 = rh > RH_MAX
    below_0 = rh < RH_MIN
    above_hard = rh > RH_SANITY_HARD_MAX

    stats = {
        "humidity_above_100": int(above_100.sum()),
        "humidity_above_hard_max": int(above_hard.sum()),
        "humidity_clipped": 0,
        "humidity_low_to_nan": int(below_0.sum()),
        "humidity_max_before_qc": rh.max(),
        "humidity_99_9_before_qc": rh.quantile(0.999),
    }

    flags.loc[below_0.fillna(False), "humidity_invalid_low"] = 1
    cleaned.loc[below_0.fillna(False), "Weather_Relative_Humidity"] = np.nan

    if int(above_hard.sum()) > 0:
        logger.warning("RH > %s detected. Automatic RH clipping skipped; manual field-definition check required.", RH_SANITY_HARD_MAX)
        print(f"WARNING: RH > {RH_SANITY_HARD_MAX} detected. RH > 100 values are kept unchanged for manual review.")
    else:
        flags.loc[above_100.fillna(False), "humidity_clipped_high"] = 1
        cleaned.loc[above_100.fillna(False), "Weather_Relative_Humidity"] = RH_MAX
        stats["humidity_clipped"] = int(above_100.sum())

    return cleaned, flags, stats


def clean_rainfall(df: pd.DataFrame, flags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cleaned = df.copy()
    negative = cleaned["Weather_Daily_Rainfall"] < 0
    flags.loc[negative.fillna(False), "rainfall_invalid_negative"] = 1
    cleaned.loc[negative.fillna(False), "Weather_Daily_Rainfall"] = np.nan
    return cleaned, flags, {"rainfall_negative_to_nan": int(negative.sum())}


def interpolate_short_gaps(series: pd.Series, max_gap_points: int) -> tuple[pd.Series, pd.Series]:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("interpolate_short_gaps requires a DatetimeIndex for time interpolation.")

    result = series.copy()
    imputed = pd.Series(False, index=series.index)
    runs = identify_nan_runs(series)

    for run in runs.itertuples(index=False):
        start = int(run.start_pos)
        end = int(run.end_pos)
        length = int(run.length)
        if length > max_gap_points:
            continue
        if start == 0 or end == len(series) - 1:
            continue
        if pd.isna(series.iloc[start - 1]) or pd.isna(series.iloc[end + 1]):
            continue

        local = series.iloc[start - 1 : end + 2].copy()
        interpolated = local.interpolate(method="time", limit_area="inside")
        fill_values = interpolated.iloc[1:-1]
        fill_mask = fill_values.notna()
        positions = series.index[start : end + 1][fill_mask.to_numpy()]
        result.loc[positions] = fill_values.loc[positions]
        imputed.loc[positions] = True

    return result, imputed


def assert_no_partial_long_run_fill(before: pd.Series, after: pd.Series, max_gap_points: int, column: str) -> None:
    runs = identify_nan_runs(before)
    for run in runs.itertuples(index=False):
        if int(run.length) <= max_gap_points:
            continue
        after_slice = after.iloc[int(run.start_pos) : int(run.end_pos) + 1]
        if after_slice.notna().any():
            raise RuntimeError(
                f"Long NaN run was partially filled in {column}: "
                f"positions {run.start_pos}-{run.end_pos}, length {run.length}"
            )


def apply_variable_specific_interpolation(df: pd.DataFrame, flags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cleaned = df.copy().set_index("timestamp")
    stats = {}
    rules = {
        "Active_Power": (MAX_GAP_POWER, "power_imputed", "active_power_interpolated"),
        "Global_Horizontal_Radiation": (MAX_GAP_GHI, "ghi_imputed", "GHI_short_interpolated"),
        "Diffuse_Horizontal_Radiation": (MAX_GAP_DHI, "dhi_imputed", "DHI_short_interpolated"),
        "Weather_Temperature_Celsius": (MAX_GAP_TEMP, "temperature_imputed", "temperature_interpolated"),
        "Weather_Relative_Humidity": (MAX_GAP_HUMIDITY, "humidity_imputed", "humidity_interpolated"),
        "Radiation_Global_Tilted": (MAX_GAP_TILTED, "Global_Tilted_imputed", "Global_Tilted_short_interpolated"),
        "Radiation_Diffuse_Tilted": (MAX_GAP_TILTED, "Diffuse_Tilted_imputed", "Diffuse_Tilted_short_interpolated"),
    }

    for column, (max_gap, flag_column, stat_key) in rules.items():
        before = cleaned[column].copy()
        after, imputed = interpolate_short_gaps(before, max_gap)
        assert_no_partial_long_run_fill(before, after, max_gap, column)
        cleaned[column] = after
        flags.loc[imputed.to_numpy(), flag_column] = 1
        stats[stat_key] = int(imputed.sum())

    cleaned = cleaned.reset_index()
    stats["tilted_short_interpolated"] = (
        stats["Global_Tilted_short_interpolated"] + stats["Diffuse_Tilted_short_interpolated"]
    )
    return cleaned, flags, stats


def build_segment_ids(df: pd.DataFrame, flags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    core_value_columns = [column for column in CORE_COLUMNS if column != "timestamp"]
    row_has_core_missing = df[core_value_columns].isna().any(axis=1)
    flags["row_has_core_missing"] = row_has_core_missing.astype(int)
    imputation_columns = [
        "power_imputed",
        "ghi_imputed",
        "dhi_imputed",
        "temperature_imputed",
        "humidity_imputed",
        "Global_Tilted_imputed",
        "Diffuse_Tilted_imputed",
    ]
    flags["row_has_any_imputation"] = (flags[imputation_columns].sum(axis=1) > 0).astype(int)

    segment_ids = np.full(len(df), -1, dtype=int)
    current_segment = 0
    previous_valid = False
    previous_timestamp = None

    for idx, row in df.iterrows():
        is_valid = not bool(row_has_core_missing.iloc[idx])
        timestamp = row["timestamp"]
        if not is_valid:
            previous_valid = False
            previous_timestamp = timestamp
            continue
        if (not previous_valid) or previous_timestamp is None or (timestamp - previous_timestamp != pd.Timedelta(minutes=5)):
            current_segment += 1
        segment_ids[idx] = current_segment
        previous_valid = True
        previous_timestamp = timestamp

    flags["segment_id"] = segment_ids
    return df, flags


def build_core_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, CORE_COLUMNS].copy()


def build_extended_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, EXTENDED_COLUMNS].copy()


def generate_segment_summary(flags: pd.DataFrame, metrics_dir: Path) -> pd.DataFrame:
    valid = flags.loc[flags["segment_id"] > 0, ["timestamp", "segment_id"]]
    if valid.empty:
        result = pd.DataFrame(columns=["segment_id", "start_timestamp", "end_timestamp", "points", "duration_hours"])
    else:
        grouped = valid.groupby("segment_id")["timestamp"]
        result = pd.DataFrame(
            {
                "segment_id": grouped.min().index.astype(int),
                "start_timestamp": grouped.min().values,
                "end_timestamp": grouped.max().values,
                "points": grouped.size().values,
            }
        )
        result["duration_hours"] = result["points"] * 5 / 60
    result.to_csv(metrics_dir / "segment_summary.csv", index=False)
    return result


def missing_rates(df: pd.DataFrame, columns: list[str]) -> dict:
    return {column: float(df[column].isna().mean()) for column in columns}


def generate_cleaning_summary(
    metrics_dir: Path,
    restrict_info: dict,
    flags: pd.DataFrame,
    gap_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    core: pd.DataFrame,
    active_stats: dict,
    radiation_stats: dict,
    humidity_stats: dict,
    rainfall_stats: dict,
    interpolation_stats: dict,
) -> Path:
    core_value_columns = [column for column in CORE_COLUMNS if column != "timestamp"]
    summary = {
        "raw_2022_rows": restrict_info["raw_2022_rows"],
        "expected_rows": EXPECTED_ROWS,
        "reindexed_rows": int(len(flags)),
        "original_missing_timestamps": int(flags["timestamp_was_missing"].sum()),
        "dropped_2023_rows": restrict_info["dropped_2023_rows"],
        "dropped_columns": drop_unusable_features(),
        **active_stats,
        "humidity_above_100": humidity_stats["humidity_above_100"],
        "humidity_clipped": humidity_stats["humidity_clipped"],
        "GHI_short_interpolated": interpolation_stats["GHI_short_interpolated"],
        "DHI_short_interpolated": interpolation_stats["DHI_short_interpolated"],
        "temperature_interpolated": interpolation_stats["temperature_interpolated"],
        "humidity_interpolated": interpolation_stats["humidity_interpolated"],
        "active_power_interpolated": interpolation_stats["active_power_interpolated"],
        "tilted_short_interpolated": interpolation_stats["tilted_short_interpolated"],
        "long_gaps_not_interpolated": int((gap_summary["category"].isin(["long", "very_long"])).sum()) if not gap_summary.empty else 0,
        "final_core_missing_rows": int(core[core_value_columns].isna().any(axis=1).sum()),
        "final_core_missing_rate_by_variable": missing_rates(core, core_value_columns),
        "number_of_segments": int(len(segment_summary)),
        "max_segment_length_points": int(segment_summary["points"].max()) if not segment_summary.empty else 0,
        "segments_ge_75_points": int((segment_summary["points"] >= 75).sum()) if not segment_summary.empty else 0,
        **radiation_stats,
        **rainfall_stats,
    }
    path = metrics_dir / "cleaning_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, indent=2, ensure_ascii=False)
    return path


def to_python(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python(v) for v in value]
    return value


def generate_cleaning_report(
    metrics_dir: Path,
    restrict_info: dict,
    preclean_audit: dict,
    gap_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    core_before_imputation: pd.DataFrame,
    core: pd.DataFrame,
    active_stats: dict,
    humidity_stats: dict,
    interpolation_stats: dict,
) -> Path:
    core_value_columns = [column for column in CORE_COLUMNS if column != "timestamp"]
    gap_counts = gap_summary["category"].value_counts().to_dict() if not gap_summary.empty else {}
    max_gap = gap_summary.sort_values("missing_points", ascending=False).head(1)
    max_gap_text = "(none)"
    if not max_gap.empty:
        row = max_gap.iloc[0]
        max_gap_text = (
            f"{row['missing_points']} points, {row['duration_hours']:.2f} hours, "
            f"{row['start_timestamp']} to {row['end_timestamp']}"
        )

    core_complete_rows = int(core[core_value_columns].notna().all(axis=1).sum())
    core_missing_rows = int(core[core_value_columns].isna().any(axis=1).sum())
    valid_segments = int(len(segment_summary))
    long_segments = int((segment_summary["points"] >= 75).sum()) if not segment_summary.empty else 0

    lines = [
        "CLEANING REPORT",
        "",
        "TIME RANGE",
        f"Original time range: {restrict_info['original_start']} to {restrict_info['original_end']}",
        f"Dropped 2023 rows: {restrict_info['dropped_2023_rows']}",
        f"2022 final timeline length: {EXPECTED_ROWS}",
        "",
        "DROPPED FEATURES",
    ]
    lines.extend(f"- {item['column']}: {item['reason']}" for item in drop_unusable_features())
    lines.extend(
        [
            "",
            "TIMESTAMP GAPS",
            f"Gap count: {len(gap_summary)}",
            f"Gap category counts: {gap_counts}",
            f"Max gap: {max_gap_text}",
            "",
            "ACTIVE POWER",
            f"Negative raw count: {preclean_audit['power_negative_total']}",
            f"Negative distribution: {preclean_audit['power_negative_stats']}",
            f"Night small negative corrected to zero: {active_stats['power_night_negative_to_zero']}",
            f"Invalid negative converted to NaN: {active_stats['power_invalid_negative_to_nan']}",
            f"> {SITE_RATED_POWER_KW} kept and flagged: {active_stats['power_above_rated_kept']}",
            f"> {POWER_HARD_MAX_KW} converted to NaN: {active_stats['power_extreme_high_to_nan']}",
            f"Short missing interpolated: {interpolation_stats['active_power_interpolated']}",
            "",
            "WEATHER",
            f"GHI interpolated: {interpolation_stats['GHI_short_interpolated']}",
            f"DHI interpolated: {interpolation_stats['DHI_short_interpolated']}",
            f"Temperature interpolated: {interpolation_stats['temperature_interpolated']}",
            f"RH > 100 count: {humidity_stats['humidity_above_100']}",
            f"RH clipped to 100: {humidity_stats['humidity_clipped']}",
            f"RH interpolated: {interpolation_stats['humidity_interpolated']}",
            f"RH audit stats: max={humidity_stats['humidity_max_before_qc']}, 99.9%={humidity_stats['humidity_99_9_before_qc']}",
            "",
            "TILTED RADIATION",
            "Tilted radiation is treated as candidate extended feature, not a first-version core input.",
            f"Only <= {MAX_GAP_TILTED} point short missing runs are interpolated.",
            "Long missing runs remain NaN; the audit found a longest run around 2389 points.",
            f"Global tilted interpolated: {interpolation_stats['Global_Tilted_short_interpolated']}",
            f"Diffuse tilted interpolated: {interpolation_stats['Diffuse_Tilted_short_interpolated']}",
            "",
            "FINAL DATASET",
            f"Core rows: {len(core)}",
            f"Core complete rows: {core_complete_rows}",
            f"Core rows with NaN: {core_missing_rows}",
            f"Core missing rate before imputation: {missing_rates(core_before_imputation, core_value_columns)}",
            f"Core missing rate after imputation: {missing_rates(core, core_value_columns)}",
            f"Valid segment count: {valid_segments}",
            f"Segments with >=75 points: {long_segments}",
            "",
            "DOWNSTREAM RULE REMINDER",
            "If target timestamp has power_imputed=1, it must not be used as a formal prediction label.",
            "If a window crosses a core missing row or different segment_id, it must not be constructed.",
            "These checks belong in 04_build_windows.py, not in this cleaning script.",
        ]
    )

    path = metrics_dir / "cleaning_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def choose_window(mask: pd.Series, timestamps: pd.Series, fallback_start: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    if mask.any():
        center = timestamps.loc[mask].iloc[0]
    else:
        center = fallback_start + pd.Timedelta(days=14)
    return center - pd.Timedelta(hours=12), center + pd.Timedelta(hours=12)


def plot_power_before_after(before: pd.DataFrame, after: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "power_before_after.png"
    anomaly = (before["Active_Power"] < 0) | (before["Active_Power"] > SITE_RATED_POWER_KW)
    start, end = choose_window(anomaly.fillna(False), before["timestamp"], YEAR_START)
    mask = before["timestamp"].between(start, end)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(before.loc[mask, "timestamp"], before.loc[mask, "Active_Power"], label="Before QC", linewidth=1.2)
    ax.plot(after.loc[mask, "timestamp"], after.loc[mask, "Active_Power"], label="After cleaning", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(SITE_RATED_POWER_KW, color="tab:red", linestyle="--", linewidth=0.9, label="Rated 5.2 kW")
    ax.set_title("Active Power Before / After Cleaning")
    ax.set_xlabel("Time")
    ax.set_ylabel("Active Power")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return path


def plot_humidity_before_after(before: pd.DataFrame, after: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "humidity_before_after.png"
    anomaly = before["Weather_Relative_Humidity"] > RH_MAX
    start, end = choose_window(anomaly.fillna(False), before["timestamp"], YEAR_START)
    mask = before["timestamp"].between(start, end)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(before.loc[mask, "timestamp"], before.loc[mask, "Weather_Relative_Humidity"], label="Before QC", linewidth=1.2)
    ax.plot(after.loc[mask, "timestamp"], after.loc[mask, "Weather_Relative_Humidity"], label="After cleaning", linewidth=1.2)
    ax.axhline(100, color="tab:red", linestyle="--", linewidth=0.9, label="100%")
    ax.set_title("Relative Humidity Before / After Cleaning")
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative Humidity")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return path


def plot_gap_distribution(gap_summary: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "gap_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    if gap_summary.empty:
        ax.text(0.5, 0.5, "No missing timestamp gaps", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(gap_summary["missing_points"], bins=30, edgecolor="black")
        ax.set_title("Missing Timestamp Gap Distribution")
        ax.set_xlabel("Gap length (5-min points)")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return path


def plot_missingness_before_after(before: pd.DataFrame, after: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "missingness_before_after.png"
    columns = [column for column in CORE_COLUMNS if column != "timestamp"]
    before_rates = [before[column].isna().mean() * 100 for column in columns]
    after_rates = [after[column].isna().mean() * 100 for column in columns]
    x = np.arange(len(columns))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, before_rates, width, label="QC before imputation")
    ax.bar(x + width / 2, after_rates, width, label="After short-gap imputation")
    ax.set_title("Core Missingness Before / After Short-Gap Imputation")
    ax.set_ylabel("Missing rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return path


def plot_tilted_missing_period(before: pd.DataFrame, after: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "tilted_missing_period.png"
    runs = identify_nan_runs(before["Radiation_Global_Tilted"])
    if runs.empty:
        start, end = YEAR_START, YEAR_START + pd.Timedelta(days=1)
    else:
        longest = runs.sort_values("length", ascending=False).iloc[0]
        start = before.loc[int(longest["start_pos"]), "timestamp"] - pd.Timedelta(hours=12)
        end = before.loc[int(longest["end_pos"]), "timestamp"] + pd.Timedelta(hours=12)
    mask = before["timestamp"].between(start, end)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(before.loc[mask, "timestamp"], before.loc[mask, "Radiation_Global_Tilted"], label="Before imputation", linewidth=1.1)
    axes[0].plot(after.loc[mask, "timestamp"], after.loc[mask, "Radiation_Global_Tilted"], label="After short-gap imputation", linewidth=1.1)
    axes[0].set_ylabel("Global Tilted")
    axes[0].legend()
    axes[0].set_title("Longest Tilted Radiation Missing Period")
    axes[1].plot(before.loc[mask, "timestamp"], before.loc[mask, "Radiation_Global_Tilted"].isna().astype(int), linewidth=1.0)
    axes[1].set_ylabel("Missing flag")
    axes[1].set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return path


def plot_segment_length_distribution(segment_summary: pd.DataFrame, figures_dir: Path) -> Path:
    path = figures_dir / "segment_length_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    if segment_summary.empty:
        ax.text(0.5, 0.5, "No valid core segments", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(segment_summary["points"], bins=30, edgecolor="black")
        ax.axvline(75, color="tab:red", linestyle="--", linewidth=1.0, label="75 points")
        ax.set_title("Valid Segment Length Distribution")
        ax.set_xlabel("Segment length (5-min points)")
        ax.set_ylabel("Count")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return path


def generate_cleaning_plots(
    before_qc: pd.DataFrame,
    qc_before_imputation: pd.DataFrame,
    after: pd.DataFrame,
    gap_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    return [
        plot_power_before_after(before_qc, after, figures_dir),
        plot_humidity_before_after(before_qc, after, figures_dir),
        plot_gap_distribution(gap_summary, figures_dir),
        plot_missingness_before_after(qc_before_imputation, after, figures_dir),
        plot_tilted_missing_period(qc_before_imputation, after, figures_dir),
        plot_segment_length_distribution(segment_summary, figures_dir),
    ]


def save_processed_outputs(
    processed_dir: Path,
    qc_before_imputation: pd.DataFrame,
    core: pd.DataFrame,
    extended: pd.DataFrame,
    flags: pd.DataFrame,
) -> dict:
    paths = {
        "qc_before_imputation": processed_dir / "site18_qc_before_imputation.csv",
        "core": processed_dir / "site18_clean_core.csv",
        "extended": processed_dir / "site18_clean_extended.csv",
        "quality_flags": processed_dir / "site18_quality_flags.csv",
    }
    qc_before_imputation.loc[:, EXTENDED_COLUMNS].to_csv(paths["qc_before_imputation"], index=False)
    core.to_csv(paths["core"], index=False)
    extended.to_csv(paths["extended"], index=False)
    flags.loc[:, QUALITY_FLAG_COLUMNS].to_csv(paths["quality_flags"], index=False)
    return paths


def validate_outputs(
    raw_path: Path,
    raw_size_before: int,
    core: pd.DataFrame,
    extended: pd.DataFrame,
    flags: pd.DataFrame,
    qc_before_imputation: pd.DataFrame,
) -> None:
    if len(core) != EXPECTED_ROWS or len(extended) != EXPECTED_ROWS or len(flags) != EXPECTED_ROWS:
        raise RuntimeError("Final outputs do not preserve the full 105120-row 2022 timeline.")
    if (core["timestamp"] >= NEXT_YEAR_START).any() or (extended["timestamp"] >= NEXT_YEAR_START).any():
        raise RuntimeError("Final outputs contain timestamps >= 2023-01-01.")
    for column in ["Wind_Speed", "Wind_Direction"]:
        if column in core.columns or column in extended.columns:
            raise RuntimeError(f"{column} should not be present in core or extended clean datasets.")

    for before_column, after_column, max_gap in [
        ("Active_Power", "Active_Power", MAX_GAP_POWER),
        ("Global_Horizontal_Radiation", "Global_Horizontal_Radiation", MAX_GAP_GHI),
        ("Diffuse_Horizontal_Radiation", "Diffuse_Horizontal_Radiation", MAX_GAP_DHI),
        ("Weather_Temperature_Celsius", "Weather_Temperature_Celsius", MAX_GAP_TEMP),
        ("Weather_Relative_Humidity", "Weather_Relative_Humidity", MAX_GAP_HUMIDITY),
        ("Radiation_Global_Tilted", "Radiation_Global_Tilted", MAX_GAP_TILTED),
        ("Radiation_Diffuse_Tilted", "Radiation_Diffuse_Tilted", MAX_GAP_TILTED),
    ]:
        assert_no_partial_long_run_fill(
            qc_before_imputation.set_index("timestamp")[before_column],
            extended.set_index("timestamp")[after_column],
            max_gap,
            after_column,
        )

    imputation_columns = [
        "power_imputed",
        "ghi_imputed",
        "dhi_imputed",
        "temperature_imputed",
        "humidity_imputed",
        "Global_Tilted_imputed",
        "Diffuse_Tilted_imputed",
    ]
    if flags["row_has_any_imputation"].sum() < flags[imputation_columns].max(axis=1).sum():
        raise RuntimeError("Some imputation flags are not reflected in row_has_any_imputation.")
    if raw_path.stat().st_size != raw_size_before:
        raise RuntimeError("Raw CSV file size changed during cleaning, which should never happen.")


def main() -> None:
    logger = setup_logging(PROJECT_ROOT)
    processed_dir, metrics_dir, figures_dir = ensure_output_dirs(PROJECT_ROOT)
    raw_path = find_raw_csv(PROJECT_ROOT)
    raw_size_before = raw_path.stat().st_size
    logger.info("Input file: %s", raw_path)

    raw_df = load_required_data(raw_path, logger)
    df_2022, restrict_info = restrict_to_2022(raw_df, logger)
    timeline = build_full_2022_timeline()
    reindexed, flags = mark_original_missing_timestamps(df_2022, timeline, logger)
    flags = initialize_quality_flags(flags)
    gap_summary = generate_gap_summary(flags, metrics_dir)
    dropped_features = drop_unusable_features()
    logger.info("Dropped features from model datasets: %s", dropped_features)

    before_qc = reindexed.copy()
    preclean_audit = audit_preclean_values(before_qc, metrics_dir)
    cleaned, flags, active_stats = clean_active_power(before_qc, flags)
    cleaned, flags, radiation_stats = clean_radiation_values(cleaned, flags)
    cleaned, flags, humidity_stats = clean_humidity(cleaned, flags, logger)
    cleaned, flags, rainfall_stats = clean_rainfall(cleaned, flags)
    qc_before_imputation = cleaned.copy()

    cleaned, flags, interpolation_stats = apply_variable_specific_interpolation(cleaned, flags)
    cleaned, flags = build_segment_ids(cleaned, flags)
    core = build_core_dataset(cleaned)
    extended = build_extended_dataset(cleaned)
    segment_summary = generate_segment_summary(flags, metrics_dir)

    processed_paths = save_processed_outputs(processed_dir, qc_before_imputation, core, extended, flags)
    summary_path = generate_cleaning_summary(
        metrics_dir,
        restrict_info,
        flags,
        gap_summary,
        segment_summary,
        core,
        active_stats,
        radiation_stats,
        humidity_stats,
        rainfall_stats,
        interpolation_stats,
    )
    report_path = generate_cleaning_report(
        metrics_dir,
        restrict_info,
        preclean_audit,
        gap_summary,
        segment_summary,
        build_core_dataset(qc_before_imputation),
        core,
        active_stats,
        humidity_stats,
        interpolation_stats,
    )
    figure_paths = generate_cleaning_plots(before_qc, qc_before_imputation, cleaned, gap_summary, segment_summary, figures_dir)

    validate_outputs(raw_path, raw_size_before, core, extended, flags, qc_before_imputation)

    core_value_columns = [column for column in CORE_COLUMNS if column != "timestamp"]
    core_missing_rates = missing_rates(core, core_value_columns)
    max_segment_length = int(segment_summary["points"].max()) if not segment_summary.empty else 0
    ge_75_segments = int((segment_summary["points"] >= 75).sum()) if not segment_summary.empty else 0

    print_section("Final Cleaning Check")
    print(f"Final 2022 rows: {len(core)}")
    print(f"Dropped 2023 rows: {restrict_info['dropped_2023_rows']}")
    print(f"Original missing timestamps restored: {int(flags['timestamp_was_missing'].sum())}")
    print(f"Core missing rates: {core_missing_rates}")
    print(f"Valid segment count: {len(segment_summary)}")
    print(f"Max segment length points: {max_segment_length}")
    print(f"Segments >=75 points: {ge_75_segments}")

    print_section("Saved Outputs")
    for label, path in processed_paths.items():
        print(f"{label}: {path}")
    print(f"gap summary: {metrics_dir / 'cleaning_gap_summary.csv'}")
    print(f"segment summary: {metrics_dir / 'segment_summary.csv'}")
    print(f"cleaning summary: {summary_path}")
    print(f"cleaning report: {report_path}")
    for path in figure_paths:
        print(f"figure: {path}")
    print(f"log: {PROJECT_ROOT / 'logs' / '02_clean_data.log'}")

    logger.info("Final 2022 rows: %s", len(core))
    logger.info("Original missing timestamps restored: %s", int(flags["timestamp_was_missing"].sum()))
    logger.info("Core missing rates: %s", core_missing_rates)
    logger.info("Valid segment count: %s", len(segment_summary))
    logger.info("Max segment length points: %s", max_segment_length)
    logger.info("Segments >=75 points: %s", ge_75_segments)
    logger.info("Script finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.getLogger("clean_data").exception("Cleaning failed")
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
