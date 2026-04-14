#!/usr/bin/env python3
"""Run a 2020 joint consumer-grid simulation with load shifting and cost accounting."""
# pylint: disable=too-many-lines

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from dataset_exploration.infer_energy_transformer import (
    append_predictions_to_raw,
    build_model_from_checkpoint,
    load_weather_frame,
    predict_all_rows,
    preprocess_house_for_inference,
)
from grid_energy_data.test_frequency_models import import_storenet_helpers


# ---------------------------
# Global configuration values
# ---------------------------
SIM_YEAR = 2020
RNG_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent
ROOT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "consumer_grid"

GRID_NORMALIZED_DIR = Path("grid_energy_data/normalized")
GRID_DATASET_ROOT = Path("grid_energy_data/dataset")
GRID_DEMAND_2020_PATH = GRID_DATASET_ROOT / "demand" / "demanddata_2020.csv"
GRID_GENERATION_PATH = GRID_DATASET_ROOT / "df_fuel_ckan.csv"
GRID_MODEL_RESULTS_PATH = Path("grid_energy_data/artifacts/frequency_model_test_results.json")
GRID_TRANSFORMER_ARTIFACT_PATH = Path(
    "grid_energy_data/artifacts/frequency_model_artifacts/transformer_best.pt"
)

CONSUMER_DATA_DIR = Path("dataset_exploration/ireland_data")
CONSUMER_INFERRED_DIR = Path("dataset_exploration/ireland_data_inferred")
CONSUMER_WEATHER_PATH = CONSUMER_DATA_DIR / "weather.csv"
CONSUMER_MODEL_CHECKPOINT_PATH = Path("dataset_exploration/models/shared_energy_transformer.pt")
CONSUMER_ONNX_MODEL_PATH = Path("dataset_exploration/models/shared_energy_transformer.onnx")
CONSUMER_END_USE_DIR = Path("dataset_exploration/artifacts/end_use_estimates")
HOUSE_IDS = [f"H{i}" for i in range(1, 21)]

OUTPUT_CSV_PATH = ROOT_ARTIFACT_DIR / "consumer_grid_joint_2020.csv.gz"
OUTPUT_FIRST_WEEK_CSV_PATH = ROOT_ARTIFACT_DIR / "consumer_grid_joint_2020_first_week.csv"
OUTPUT_SUMMARY_JSON_PATH = ROOT_ARTIFACT_DIR / "consumer_grid_joint_2020_summary.json"
OUTPUT_GZIP_COMPRESSLEVEL = 9

LCOE_USD_PER_MWH = {
    "Biomass": 95.0,
    "Coal (ultra-supercritical)": 76.0,
    "Natural Gas (combined cycle)": 38.0,
    "Natural Gas (combustion turbine)": 67.0,
    "Nuclear": 82.0,
    "Solar": 36.0,
    "Wind": 40.0,
}

GRID_READ_CHUNKSIZE = 350_000
GRID_FREQ_INFER_BATCH_SIZE = 8_192
CONSUMER_INFER_BATCH_SIZE = 8_192

USE_PRECOMPUTED_CONSUMER_PREDICTIONS = False
USE_ONNX_CONSUMER_INFERENCE = False
CONSUMER_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
RECOMPUTE_MISSING_END_USE = False

CONSUMER_GROUP_MULTIPLIER = 12000.0
ACTION_EFFECTIVENESS_PROB = 0.85

PEAK_LOAD_QUANTILE = 0.75
EXPENSIVE_PRICE_QUANTILE = 0.75
CHEAP_PRICE_QUANTILE = 0.35

MIN_ACTIONABLE_REDUCTION_WH = 20.0
MAX_CONSUMPTION_REDUCTION_FRAC = 0.45
SHIFTABLE_REDUCTION_FRACTION = 0.80
OTHER_PLUG_SHIFTABLE_FRACTION = 0.35
WATER_HEATING_SHIFTABLE_FRACTION = 0.75
HVAC_ADJUSTABLE_FRACTION = 0.50
REDUCTION_RELEASE_RATE = 0.12
MAX_REDUCTION_WH_PER_MIN = 250.0
MAX_REBOUND_WH_PER_MIN = 250.0
REBOUND_RELEASE_RATE = 0.10
REBOUND_MAX_CONSUMPTION_FRAC = 0.18
REBOUND_RAMP_DELTA_WH_PER_MIN = 200.0

BATTERY_CAPACITY_WH = 10_000.0
BATTERY_MIN_SOC_PCT = 20.0
BATTERY_MAX_SUPPORT_WH_PER_MIN = 180.0
BATTERY_SUPPORT_NET_LOAD_FRAC = 0.30
BATTERY_MAX_REBOUND_WH_PER_MIN = 180.0
BATTERY_REBOUND_RELEASE_RATE = 0.08
BATTERY_REBOUND_MAX_CONSUMPTION_FRAC = 0.12
BATTERY_REBOUND_RAMP_DELTA_WH_PER_MIN = 800.0

DEVICE_ON_THRESHOLD_W = 120.0
GRID_FREQ_RESPONSE_HZ_PER_MW = 0.00002

UNMAPPED_LCOE_SOURCE_NAME = "Natural Gas (combustion turbine)"
LCOE_BY_GENERATION_COLUMN = {
    "generation_biomass": "Biomass",
    "generation_coal": "Coal (ultra-supercritical)",
    "generation_gas": "Natural Gas (combined cycle)",
    "generation_nuclear": "Nuclear",
    "generation_solar": "Solar",
    "generation_wind": "Wind",
    "generation_wind_emb": "Wind",
}
GENERATION_SOURCE_COLUMNS = [
    "generation_gas",
    "generation_coal",
    "generation_nuclear",
    "generation_wind",
    "generation_wind_emb",
    "generation_hydro",
    "generation_imports",
    "generation_biomass",
    "generation_other",
    "generation_solar",
    "generation_storage",
]
PROTECTED_GENERATION_COLUMNS = {"generation_nuclear"}
CSV_EXPORT_COLUMNS_TO_DROP = [
    "demand_england_wales_demand",
    "demand_embedded_wind_generation",
    "demand_embedded_wind_capacity",
    "demand_embedded_solar_generation",
    "demand_embedded_solar_capacity",
    "demand_non_bm_stor",
    "demand_pump_storage_pumping",
    "demand_scottish_transfer",
    "demand_ifa_flow",
    "demand_ifa2_flow",
    "demand_britned_flow",
    "demand_moyle_flow",
    "demand_east_west_flow",
    "demand_nemo_flow",
    "demand_nsl_flow",
    "demand_eleclink_flow",
    "demand_viking_flow",
    "demand_greenlink_flow",
    "generation_low_carbon",
    "generation_zero_carbon",
    "generation_renewable",
    "generation_fossil",
    "generation_gas_perc",
    "generation_coal_perc",
    "generation_nuclear_perc",
    "generation_wind_perc",
    "generation_wind_emb_perc",
    "generation_hydro_perc",
    "generation_imports_perc",
    "generation_biomass_perc",
    "generation_other_perc",
    "generation_solar_perc",
    "generation_storage_perc",
    "generation_generation_perc",
    "generation_low_carbon_perc",
    "generation_zero_carbon_perc",
    "generation_renewable_perc",
    "generation_fossil_perc",
]


@dataclass
class ConsumerInferenceContext:
    """Shared consumer-model inference state."""

    model: torch.nn.Module | None
    weather_frame: pd.DataFrame
    feature_columns: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    seq_len: int
    device: torch.device
    use_onnx: bool
    onnx_session: object | None = None
    onnx_input_x_name: str | None = None
    onnx_input_house_name: str | None = None
    onnx_output_name: str | None = None


def load_grid_2020_minute_frame(normalized_dir: Path, chunksize: int) -> pd.DataFrame:
    """Load and minute-aggregate normalized 2020 grid rows.

    :param normalized_dir: Directory with normalized frequency partitions.
    :param chunksize: Stream chunk size used for CSV reading.
    :return: Minute-level frame with frequency, demand, and generation columns.
    """
    # pylint: disable=too-many-locals
    files = sorted(normalized_dir.glob("*2020*_normalized.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No 2020 normalized files found in {normalized_dir}.")

    monthly_frames: list[pd.DataFrame] = []
    for path in tqdm(files, desc="Loading grid 2020 files", unit="file"):
        per_chunk: list[pd.DataFrame] = []
        reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
        for chunk in reader:
            if "timestamp_utc" not in chunk.columns:
                continue
            timestamps = pd.to_datetime(chunk["timestamp_utc"], errors="coerce", utc=True)
            valid = timestamps.notna()
            if not valid.any():
                continue
            filtered = chunk.loc[valid].copy()
            filtered["timestamp_utc"] = timestamps.loc[valid]
            numeric_columns = [column for column in filtered.columns if column != "timestamp_utc"]
            for column in numeric_columns:
                filtered[column] = pd.to_numeric(filtered[column], errors="coerce")
            filtered["minute_utc"] = filtered["timestamp_utc"].dt.floor("min")
            grouped = filtered.groupby("minute_utc", as_index=False)[numeric_columns].mean()
            per_chunk.append(grouped)

        if not per_chunk:
            continue
        month = pd.concat(per_chunk, ignore_index=True)
        month = month.groupby("minute_utc", as_index=False).mean(numeric_only=True)
        monthly_frames.append(month)

    if not monthly_frames:
        raise RuntimeError("Failed to load any minute-level rows from 2020 normalized files.")

    frame = pd.concat(monthly_frames, ignore_index=True)
    frame = frame.groupby("minute_utc", as_index=False).mean(numeric_only=True)
    frame = frame.sort_values("minute_utc").reset_index(drop=True)
    frame = frame.rename(columns={"minute_utc": "timestamp_utc"})

    if "frequency_hz" not in frame.columns:
        raise ValueError("Expected 'frequency_hz' in normalized grid data.")
    frame["frequency_hz"] = frame["frequency_hz"].interpolate(
        method="linear",
        limit_direction="both",
    )

    frame = enrich_grid_with_raw_fallback(frame)

    if "demand_tsd" not in frame.columns:
        raise ValueError("Expected 'demand_tsd' in grid data.")
    frame["demand_tsd"] = pd.to_numeric(frame["demand_tsd"], errors="coerce")
    frame["demand_tsd"] = frame["demand_tsd"].interpolate(method="linear", limit_direction="both")
    if frame["demand_tsd"].isna().any():
        raise RuntimeError("Demand series still has missing values after fallback/interpolation.")
    return frame


def load_raw_2020_demand() -> pd.DataFrame:
    """Load raw 2020 demand file and return half-hour anchor points.

    :return: Dataframe with ``timestamp_utc``, ``demand_tsd``, and ``demand_nd``.
    """
    if not GRID_DEMAND_2020_PATH.exists():
        raise FileNotFoundError(f"Missing raw demand file: {GRID_DEMAND_2020_PATH}")

    demand = pd.read_csv(GRID_DEMAND_2020_PATH, low_memory=False)
    demand.columns = [column.strip() for column in demand.columns]
    required = {"SETTLEMENT_DATE", "SETTLEMENT_PERIOD"}
    if not required.issubset(demand.columns):
        raise ValueError(f"Raw demand file missing columns: {sorted(required)}")

    settlement_date = pd.to_datetime(
        demand["SETTLEMENT_DATE"].astype("string").str.strip(),
        format="%d-%b-%Y",
        errors="coerce",
        utc=True,
    )
    settlement_period = pd.to_numeric(demand["SETTLEMENT_PERIOD"], errors="coerce")
    offset_minutes = (settlement_period - 1.0) * 30.0
    demand["timestamp_utc"] = settlement_date + pd.to_timedelta(offset_minutes, unit="m")

    output = demand[["timestamp_utc"]].copy()
    output["demand_tsd"] = pd.to_numeric(demand.get("TSD"), errors="coerce")
    output["demand_nd"] = pd.to_numeric(demand.get("ND"), errors="coerce")
    output = output.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    output = output[output["timestamp_utc"].dt.year == SIM_YEAR]
    output = output.groupby("timestamp_utc", as_index=False).mean(numeric_only=True)
    return output


def load_raw_2020_generation() -> pd.DataFrame:
    """Load raw generation mix and return 2020 half-hour anchor points.

    :return: Dataframe with timestamp and mapped ``generation_*`` columns.
    """
    if not GRID_GENERATION_PATH.exists():
        raise FileNotFoundError(f"Missing raw generation file: {GRID_GENERATION_PATH}")

    generation = pd.read_csv(GRID_GENERATION_PATH, low_memory=False)
    generation.columns = [column.strip() for column in generation.columns]
    if "DATETIME" not in generation.columns:
        raise ValueError("Raw generation file is missing DATETIME column.")

    generation["timestamp_utc"] = pd.to_datetime(
        generation["DATETIME"].astype("string").str.strip(),
        errors="coerce",
        utc=True,
    )
    generation = generation.dropna(subset=["timestamp_utc"]).copy()
    generation = generation[generation["timestamp_utc"].dt.year == SIM_YEAR]

    mapping = {
        "GAS": "generation_gas",
        "COAL": "generation_coal",
        "NUCLEAR": "generation_nuclear",
        "WIND": "generation_wind",
        "WIND_EMB": "generation_wind_emb",
        "HYDRO": "generation_hydro",
        "IMPORTS": "generation_imports",
        "BIOMASS": "generation_biomass",
        "OTHER": "generation_other",
        "SOLAR": "generation_solar",
        "STORAGE": "generation_storage",
        "GENERATION": "generation_generation",
    }

    output = generation[["timestamp_utc"]].copy()
    for source_column, target_column in mapping.items():
        output[target_column] = pd.to_numeric(generation.get(source_column), errors="coerce")

    output = output.sort_values("timestamp_utc")
    output = output.groupby("timestamp_utc", as_index=False).mean(numeric_only=True)
    return output


def interpolate_anchors_to_target(
    target_timestamps: pd.Series,
    anchors: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Linearly interpolate anchor columns onto target timestamps.

    :param target_timestamps: Target timestamp series.
    :param anchors: Source anchor dataframe with ``timestamp_utc``.
    :param columns: Anchor value columns to interpolate.
    :return: Dataframe indexed to target timestamps with interpolated columns.
    """
    result = pd.DataFrame({"timestamp_utc": target_timestamps})
    target_seconds = target_timestamps.astype("int64").to_numpy(dtype=np.int64) // 1_000_000_000

    source_seconds = anchors["timestamp_utc"].astype("int64").to_numpy(dtype=np.int64)
    source_seconds = source_seconds // 1_000_000_000
    for column in columns:
        values = anchors[column].to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        if valid.sum() < 2:
            result[column] = np.nan
            continue
        result[column] = np.interp(
            target_seconds.astype(np.float64),
            source_seconds[valid].astype(np.float64),
            values[valid],
            left=np.nan,
            right=np.nan,
        ).astype(np.float32)
    return result


def enrich_grid_with_raw_fallback(grid_frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing demand/generation fields from raw 2020 source files.

    :param grid_frame: Minute-level grid frame built from normalized frequency files.
    :return: Enriched grid frame.
    """
    output = grid_frame.copy()
    target_ts = output["timestamp_utc"]

    demand_anchors = load_raw_2020_demand()
    demand_interp = interpolate_anchors_to_target(
        target_timestamps=target_ts,
        anchors=demand_anchors,
        columns=["demand_tsd", "demand_nd"],
    )

    if "demand_tsd" not in output.columns:
        output["demand_tsd"] = np.nan
    output["demand_tsd"] = output["demand_tsd"].fillna(demand_interp["demand_tsd"])
    if "demand_nd" not in output.columns:
        output["demand_nd"] = np.nan
    output["demand_nd"] = output["demand_nd"].fillna(demand_interp["demand_nd"])
    output["demand_tsd"] = output["demand_tsd"].fillna(output["demand_nd"])
    output["demand_nd"] = pd.to_numeric(output["demand_nd"], errors="coerce")
    output["demand_nd"] = output["demand_nd"].interpolate(
        method="linear",
        limit_direction="both",
    )

    generation_columns = [
        "generation_gas",
        "generation_coal",
        "generation_nuclear",
        "generation_wind",
        "generation_wind_emb",
        "generation_hydro",
        "generation_imports",
        "generation_biomass",
        "generation_other",
        "generation_solar",
        "generation_storage",
        "generation_generation",
    ]
    generation_anchors = load_raw_2020_generation()
    generation_interp = interpolate_anchors_to_target(
        target_timestamps=target_ts,
        anchors=generation_anchors,
        columns=generation_columns,
    )
    for column in generation_columns:
        if column not in output.columns:
            output[column] = np.nan
        output[column] = output[column].fillna(generation_interp[column])
        output[column] = pd.to_numeric(output[column], errors="coerce")
        output[column] = output[column].interpolate(
            method="linear",
            limit_direction="both",
        )
        output[column] = output[column].fillna(0.0)

    return output


def compute_lcoe_price_series(grid_frame: pd.DataFrame, lcoe: dict[str, float]) -> pd.Series:
    """Compute blended $/MWh from generation mix using README LCOE values.

    :param grid_frame: Minute-level grid frame.
    :param lcoe: LCOE mapping parsed from README.
    :return: Price series in USD per MWh.
    """
    known_columns = [column for column in LCOE_BY_GENERATION_COLUMN if column in grid_frame.columns]
    if not known_columns:
        raise ValueError("No mapped generation columns found for LCOE pricing.")

    known_total = np.zeros(len(grid_frame), dtype=np.float64)
    weighted_sum = np.zeros(len(grid_frame), dtype=np.float64)
    for column in known_columns:
        values = pd.to_numeric(grid_frame[column], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=False,
        )
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        known_total += values
        weighted_sum += values * float(lcoe[LCOE_BY_GENERATION_COLUMN[column]])

    if "generation_generation" in grid_frame.columns:
        total_generation = pd.to_numeric(
            grid_frame["generation_generation"],
            errors="coerce",
        ).to_numpy(
            dtype=np.float64,
            copy=False,
        )
        total_generation = np.nan_to_num(total_generation, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        total_generation = known_total.copy()
    unknown_generation = np.clip(total_generation - known_total, a_min=0.0, a_max=None)

    fallback = float(lcoe[UNMAPPED_LCOE_SOURCE_NAME])
    weighted_sum += unknown_generation * fallback
    denominator = known_total + unknown_generation
    price = np.full_like(denominator, fallback, dtype=np.float64)
    np.divide(weighted_sum, denominator, out=price, where=denominator > 0.0)
    return pd.Series(
        price.astype(np.float32),
        index=grid_frame.index,
        name="grid_lcoe_usd_per_mwh",
    )


def build_export_frame(grid_frame: pd.DataFrame) -> pd.DataFrame:
    """Build the final CSV export frame.

    :param grid_frame: Simulation results frame.
    :return: Frame without internal demand and generation source columns.
    """
    return grid_frame.drop(columns=CSV_EXPORT_COLUMNS_TO_DROP, errors="ignore")


def build_first_week_export_frame(grid_frame: pd.DataFrame) -> pd.DataFrame:
    """Build the uncompressed first-week export frame.

    :param grid_frame: Simulation results frame.
    :return: Frame filtered to the first seven days of timestamps.
    """
    export_frame = build_export_frame(grid_frame)
    if export_frame.empty or "timestamp_utc" not in export_frame.columns:
        return export_frame.copy()

    timestamps = pd.to_datetime(export_frame["timestamp_utc"], errors="coerce", utc=True)
    valid_timestamps = timestamps.notna()
    if not valid_timestamps.any():
        return export_frame.iloc[0:0].copy()

    first_timestamp = timestamps.loc[valid_timestamps].min()
    cutoff_timestamp = first_timestamp + pd.Timedelta(days=7)
    return export_frame.loc[timestamps < cutoff_timestamp].copy()


def apply_generation_merit_reduction(
    grid_frame: pd.DataFrame,
    lcoe: dict[str, float],
    demand_original_mw: np.ndarray,
    demand_adjusted_mw: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reduce generation stack by merit order for demand reductions.

    Higher-cost generation is curtailed first, while protected sources such as nuclear
    are not selected for curtailment.

    :param grid_frame: Grid frame with ``generation_*`` columns.
    :param lcoe: LCOE data from README.
    :param demand_original_mw: Original demand vector.
    :param demand_adjusted_mw: Adjusted demand vector.
    :return: Per-column adjusted generation and reduction arrays.
    """
    # pylint: disable=too-many-locals
    fallback_cost = float(lcoe[UNMAPPED_LCOE_SOURCE_NAME])
    source_columns = [
        column for column in GENERATION_SOURCE_COLUMNS if column in grid_frame.columns
    ]
    if not source_columns:
        return {
            "remaining_unserved_reduction_mw": np.zeros_like(demand_original_mw, dtype=np.float64),
        }

    adjusted_by_column: dict[str, np.ndarray] = {}
    reduced_by_column: dict[str, np.ndarray] = {}
    for column in source_columns:
        values = pd.to_numeric(grid_frame[column], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=False,
        )
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.clip(values, a_min=0.0, a_max=None)
        adjusted_by_column[column] = values.copy()
        reduced_by_column[column] = np.zeros_like(values)

    required_reduction_mw = np.clip(demand_original_mw - demand_adjusted_mw, a_min=0.0, a_max=None)
    remaining = required_reduction_mw.copy()

    reducible_columns: list[tuple[str, float]] = []
    for column in source_columns:
        if column in PROTECTED_GENERATION_COLUMNS:
            continue
        source_name = LCOE_BY_GENERATION_COLUMN.get(column)
        source_cost = fallback_cost if source_name is None else float(lcoe[source_name])
        reducible_columns.append((column, source_cost))
    reducible_columns.sort(key=lambda item: item[1], reverse=True)

    for column, _ in reducible_columns:
        available = adjusted_by_column[column]
        reduction = np.minimum(available, remaining)
        adjusted_by_column[column] = available - reduction
        reduced_by_column[column] = reduction
        remaining = np.clip(remaining - reduction, a_min=0.0, a_max=None)

    result: dict[str, np.ndarray] = {}
    adjusted_total = np.zeros_like(demand_original_mw, dtype=np.float64)
    reduced_total = np.zeros_like(demand_original_mw, dtype=np.float64)
    for column in source_columns:
        adjusted = adjusted_by_column[column]
        reduced = reduced_by_column[column]
        result[f"{column}_adjusted_mw"] = adjusted
        result[f"{column}_reduced_mw"] = reduced
        adjusted_total += adjusted
        reduced_total += reduced

    result["generation_generation_adjusted_mw"] = adjusted_total
    result["generation_total_reduced_mw"] = reduced_total
    result["remaining_unserved_reduction_mw"] = remaining
    return result


def load_consumer_inference_context(device: torch.device) -> ConsumerInferenceContext:
    """Load the consumer transformer model/checkpoint and inference metadata.

    :param device: Torch device for inference.
    :return: Initialized consumer inference context.
    """
    checkpoint = torch.load(
        CONSUMER_MODEL_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    weather = load_weather_frame(CONSUMER_WEATHER_PATH)

    common_kwargs = {
        "weather_frame": weather,
        "feature_columns": list(checkpoint["feature_columns"]),
        "feature_mean": np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        "feature_std": np.asarray(checkpoint["feature_std"], dtype=np.float32),
        "target_mean": np.asarray(checkpoint["target_mean"], dtype=np.float32),
        "target_std": np.asarray(checkpoint["target_std"], dtype=np.float32),
        "seq_len": int(checkpoint["config"]["seq_len"]),
        "device": device,
    }

    if USE_ONNX_CONSUMER_INFERENCE:
        # pylint: disable=import-outside-toplevel
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for USE_ONNX_CONSUMER_INFERENCE=True."
            ) from exc

        if not CONSUMER_ONNX_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing ONNX model: {CONSUMER_ONNX_MODEL_PATH}. "
                "Run dataset_exploration/quantize_energy_transformer_onnx.py first."
            )

        available = set(ort.get_available_providers())
        print(f"Available ONNX Runtime providers: {sorted(available)}")
        providers = [provider for provider in CONSUMER_ONNX_PROVIDERS if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(CONSUMER_ONNX_MODEL_PATH),
            providers=providers,
            sess_options=session_options,
        )
        inputs = session.get_inputs()
        if len(inputs) < 2:
            raise ValueError(
                f"ONNX model expected 2 inputs, found {len(inputs)}: {CONSUMER_ONNX_MODEL_PATH}"
            )
        outputs = session.get_outputs()
        if not outputs:
            raise ValueError(f"ONNX model has no outputs: {CONSUMER_ONNX_MODEL_PATH}")

        return ConsumerInferenceContext(
            model=None,
            use_onnx=True,
            onnx_session=session,
            onnx_input_x_name=inputs[0].name,
            onnx_input_house_name=inputs[1].name,
            onnx_output_name=outputs[0].name,
            **common_kwargs,
        )

    model = build_model_from_checkpoint(checkpoint=checkpoint, device=device)
    return ConsumerInferenceContext(
        model=model,
        use_onnx=False,
        **common_kwargs,
    )


def predict_all_rows_consumer_onnx(
    context: ConsumerInferenceContext,
    processed_frame: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    """Predict production/consumption using ONNX Runtime for every row.

    :param context: Consumer inference context with ONNX session metadata.
    :param processed_frame: Model-ready frame.
    :param batch_size: Batch size.
    :return: Denormalized predictions with shape ``(n_rows, 2)``.
    """
    # pylint: disable=too-many-locals
    if (
        context.onnx_session is None
        or context.onnx_input_x_name is None
        or context.onnx_input_house_name is None
        or context.onnx_output_name is None
    ):
        raise RuntimeError("ONNX context is not fully initialized.")

    feature_values = processed_frame[context.feature_columns].to_numpy(dtype=np.float32)
    normalized = (feature_values - context.feature_mean) / context.feature_std
    row_count = normalized.shape[0]
    history_offsets = np.arange(context.seq_len, dtype=np.int64) - context.seq_len
    house_id = int(processed_frame["house_id"].iloc[0])
    predictions = np.empty((row_count, len(context.target_mean)), dtype=np.float32)

    for batch_start in tqdm(
        range(0, row_count, batch_size),
        leave=False,
        desc=f"{processed_frame['house_name'].iloc[0]} inference (onnx)",
    ):
        batch_end = min(batch_start + batch_size, row_count)
        indices = np.arange(batch_start, batch_end, dtype=np.int64)
        window_indices = (indices[:, None] + history_offsets[None, :]) % row_count
        windows = normalized[window_indices].astype(np.float32, copy=False)
        house_ids = np.full((len(indices),), house_id, dtype=np.int64)

        predicted_norm = context.onnx_session.run(
            [context.onnx_output_name],
            {
                context.onnx_input_x_name: windows,
                context.onnx_input_house_name: house_ids,
            },
        )[0]
        predictions[batch_start:batch_end] = (
            predicted_norm * context.target_std + context.target_mean
        )

    return predictions


def _load_precomputed_predictions(house: str) -> pd.DataFrame | None:
    """Load existing per-house inferred predictions when available.

    :param house: House id (for example H1).
    :return: Prediction frame or ``None`` when unavailable/incomplete.
    """
    path = CONSUMER_INFERRED_DIR / f"{house}_Wh.csv"
    if not path.exists():
        return None

    header = pd.read_csv(path, nrows=0).columns.tolist()
    stripped_to_raw = {column.strip(): column for column in header}
    required = [
        "date",
        "predicted_production_wh",
        "predicted_consumption_wh",
        "State of Charge(%)",
        "Discharge(Wh)",
    ]
    if any(column not in stripped_to_raw for column in required):
        return None

    usecols = [stripped_to_raw[column] for column in required]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame.columns = [column.strip() for column in frame.columns]
    frame["timestamp_utc"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp_utc"]).copy()
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["predicted_production_wh", "predicted_consumption_wh"])
    return frame[
        [
            "timestamp_utc",
            "predicted_production_wh",
            "predicted_consumption_wh",
            "State of Charge(%)",
            "Discharge(Wh)",
        ]
    ]


def infer_house_predictions(
    house: str,
    context: ConsumerInferenceContext,
) -> pd.DataFrame:
    """Run consumer-model inference for one house and return needed columns.

    :param house: House id (for example H1).
    :param context: Shared consumer inference context.
    :return: Per-row prediction frame with timestamps and battery fields.
    """
    raw_frame, processed_frame = preprocess_house_for_inference(
        energy_path=CONSUMER_DATA_DIR / f"{house}_Wh.csv",
        weather=context.weather_frame,
        feature_columns=context.feature_columns,
    )
    if context.use_onnx:
        predictions = predict_all_rows_consumer_onnx(
            context=context,
            processed_frame=processed_frame,
            batch_size=CONSUMER_INFER_BATCH_SIZE,
        )
    else:
        if context.model is None:
            raise RuntimeError("Torch consumer model context is not initialized.")
        predictions = predict_all_rows(
            model=context.model,
            processed_frame=processed_frame,
            feature_columns=context.feature_columns,
            feature_mean=context.feature_mean,
            feature_std=context.feature_std,
            target_mean=context.target_mean,
            target_std=context.target_std,
            seq_len=context.seq_len,
            batch_size=CONSUMER_INFER_BATCH_SIZE,
            device=context.device,
        )
    output = append_predictions_to_raw(
        raw_frame=raw_frame,
        processed_frame=processed_frame,
        predictions=predictions,
    )
    output.columns = [column.strip() for column in output.columns]
    output["timestamp_utc"] = pd.to_datetime(output["date"], errors="coerce", utc=True)
    output = output.dropna(subset=["timestamp_utc"]).copy()
    needed_columns = [
        "predicted_production_wh",
        "predicted_consumption_wh",
        "State of Charge(%)",
        "Discharge(Wh)",
    ]
    for column in needed_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["predicted_production_wh", "predicted_consumption_wh"])
    return output[["timestamp_utc", *needed_columns]]


def load_house_prediction_frame(
    house: str,
    context: ConsumerInferenceContext | None,
) -> pd.DataFrame:
    """Load (or infer) one house's production/consumption prediction frame.

    :param house: House id (for example H1).
    :param context: Consumer inference context, required when inference is needed.
    :return: Minute-level per-house frame with prediction and battery columns.
    """
    prediction_frame = None
    if USE_PRECOMPUTED_CONSUMER_PREDICTIONS:
        prediction_frame = _load_precomputed_predictions(house)
    if prediction_frame is None:
        if context is None:
            raise RuntimeError(
                f"{house}: missing precomputed predictions and no inference context provided."
            )
        prediction_frame = infer_house_predictions(house=house, context=context)

    prediction_frame = prediction_frame.copy()
    prediction_frame["minute_utc"] = prediction_frame["timestamp_utc"].dt.floor("min")
    prediction_frame = prediction_frame[
        prediction_frame["minute_utc"].dt.year == SIM_YEAR
    ].copy()
    prediction_frame = prediction_frame.groupby("minute_utc", as_index=False).mean(
        numeric_only=True
    )
    prediction_frame = prediction_frame.rename(columns={"minute_utc": "timestamp_utc"})
    return prediction_frame


def load_house_end_use_frame(house: str) -> pd.DataFrame:
    """Load one house's end-use estimate frame with only needed columns.

    :param house: House id (for example H1).
    :return: Minute-level frame with end-use columns.
    """
    path = CONSUMER_END_USE_DIR / f"{house}_end_use_estimates.csv"
    if not path.exists():
        if not RECOMPUTE_MISSING_END_USE:
            raise FileNotFoundError(f"Missing end-use file for {house}: {path}")
        # pylint: disable=import-outside-toplevel
        from dataset_exploration.estimate_house_end_uses import (
            estimate_end_uses,
            read_house_and_weather,
        )

        source = read_house_and_weather(data_dir=CONSUMER_DATA_DIR, house=house)
        rebuilt = estimate_end_uses(frame=source, event_threshold_w=300.0)
        rebuilt.to_csv(path, index=False)

    requested_columns = [
        "date",
        "cooking_wh",
        "laundry_wh",
        "other_plug_wh",
        "water_heating_wh",
        "device_hvac_heating_wh",
        "device_hvac_cooling_wh",
        "device_water_heater_w",
        "device_cooker_w",
        "device_kettle_w",
        "device_microwave_w",
        "device_washing_machine_w",
        "device_dishwasher_w",
        "device_tumble_dryer_w",
        "device_misc_event_w",
    ]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    available = [column for column in requested_columns if column in header]
    frame = pd.read_csv(path, usecols=available, low_memory=False)
    frame["timestamp_utc"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp_utc"]).copy()

    numeric_columns = [column for column in available if column != "date"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["minute_utc"] = frame["timestamp_utc"].dt.floor("min")
    frame = frame[frame["minute_utc"].dt.year == SIM_YEAR].copy()
    frame = frame.groupby("minute_utc", as_index=False).mean(numeric_only=True)
    frame = frame.rename(columns={"minute_utc": "timestamp_utc"})
    return frame


def simulate_house_response(
    house: str,
    grid_timestamps: pd.DatetimeIndex,
    policy_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    end_use_frame: pd.DataFrame,
    seed: int,
) -> dict[str, np.ndarray]:
    """Simulate one house's load-reduction and rebound behavior.

    :param house: House id (for example H1).
    :param grid_timestamps: Target minute index used by grid frame.
    :param policy_frame: Grid policy frame indexed by timestamp.
    :param prediction_frame: Per-house predicted production/consumption frame.
    :param end_use_frame: Per-house end-use estimate frame.
    :param seed: RNG seed for probabilistic action effectiveness.
    :return: Dictionary of per-minute arrays.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    del house
    pred = prediction_frame.set_index("timestamp_utc").reindex(grid_timestamps)
    end_use = end_use_frame.set_index("timestamp_utc").reindex(grid_timestamps)

    pred_columns = [
        "predicted_production_wh",
        "predicted_consumption_wh",
        "State of Charge(%)",
        "Discharge(Wh)",
    ]
    for column in pred_columns:
        if column not in pred.columns:
            pred[column] = 0.0
        pred[column] = pred[column].interpolate(method="time", limit_direction="both").fillna(0.0)

    end_use_defaults = [
        "cooking_wh",
        "laundry_wh",
        "other_plug_wh",
        "water_heating_wh",
        "device_hvac_heating_wh",
        "device_hvac_cooling_wh",
        "device_water_heater_w",
        "device_cooker_w",
        "device_kettle_w",
        "device_microwave_w",
        "device_washing_machine_w",
        "device_dishwasher_w",
        "device_tumble_dryer_w",
        "device_misc_event_w",
    ]
    for column in end_use_defaults:
        if column not in end_use.columns:
            end_use[column] = 0.0
        end_use[column] = end_use[column].fillna(0.0)

    predicted_consumption_wh = pred["predicted_consumption_wh"].to_numpy(dtype=np.float32)
    predicted_production_wh = pred["predicted_production_wh"].to_numpy(dtype=np.float32)
    soc_pct = pred["State of Charge(%)"].to_numpy(dtype=np.float32)
    original_discharge_wh = pred["Discharge(Wh)"].to_numpy(dtype=np.float32)
    original_discharge_wh = np.nan_to_num(original_discharge_wh, nan=0.0)
    soc_pct = np.clip(np.nan_to_num(soc_pct, nan=0.0), 0.0, 100.0)

    shiftable_wh = (
        end_use["cooking_wh"].to_numpy(dtype=np.float32)
        + end_use["laundry_wh"].to_numpy(dtype=np.float32)
        + end_use["other_plug_wh"].to_numpy(dtype=np.float32) * OTHER_PLUG_SHIFTABLE_FRACTION
        + end_use["water_heating_wh"].to_numpy(dtype=np.float32)
        * WATER_HEATING_SHIFTABLE_FRACTION
    )
    hvac_adjustable_wh = (
        end_use["device_hvac_heating_wh"].to_numpy(dtype=np.float32)
        + end_use["device_hvac_cooling_wh"].to_numpy(dtype=np.float32)
    ) * HVAC_ADJUSTABLE_FRACTION

    potential_reduction_wh = shiftable_wh * SHIFTABLE_REDUCTION_FRACTION + hvac_adjustable_wh
    potential_reduction_wh = np.minimum(
        potential_reduction_wh,
        predicted_consumption_wh * MAX_CONSUMPTION_REDUCTION_FRAC,
    )
    potential_reduction_wh = np.clip(potential_reduction_wh, a_min=0.0, a_max=None)

    is_peak_or_expensive = policy_frame["is_peak_or_expensive"].to_numpy(dtype=bool)
    is_cheap = policy_frame["is_cheap"].to_numpy(dtype=bool)
    trigger = is_peak_or_expensive & (potential_reduction_wh >= MIN_ACTIONABLE_REDUCTION_WH)

    device_on_columns = [
        "device_water_heater_w",
        "device_cooker_w",
        "device_kettle_w",
        "device_microwave_w",
        "device_washing_machine_w",
        "device_dishwasher_w",
        "device_tumble_dryer_w",
        "device_misc_event_w",
    ]
    device_on_count = np.zeros(len(grid_timestamps), dtype=np.float32)
    for column in device_on_columns:
        active = end_use[column].to_numpy(dtype=np.float32) >= DEVICE_ON_THRESHOLD_W
        device_on_count += active.astype(np.float32)

    achieved_reduction_wh = np.zeros(len(grid_timestamps), dtype=np.float32)
    reduction_rebound_wh = np.zeros(len(grid_timestamps), dtype=np.float32)
    battery_support_wh = np.zeros(len(grid_timestamps), dtype=np.float32)
    battery_rebound_wh = np.zeros(len(grid_timestamps), dtype=np.float32)
    action_triggered = trigger.astype(np.int8)
    action_success = np.zeros(len(grid_timestamps), dtype=np.int8)

    deferred_pool_wh = 0.0
    reduction_pending_wh = 0.0
    battery_debt_wh = 0.0
    previous_rebound_wh = 0.0
    previous_battery_rebound_wh = 0.0
    rng = np.random.default_rng(seed=seed)

    for index in range(len(grid_timestamps)):
        requested_reduction_wh = 0.0
        if trigger[index] and rng.random() < ACTION_EFFECTIVENESS_PROB:
            requested_reduction_wh = float(potential_reduction_wh[index])
            action_success[index] = 1
        reduction_pending_wh += requested_reduction_wh

        reduction_now = 0.0
        if is_peak_or_expensive[index] and reduction_pending_wh > 0.0:
            reduction_cap = min(
                float(potential_reduction_wh[index]),
                float(predicted_consumption_wh[index]) * MAX_CONSUMPTION_REDUCTION_FRAC,
                MAX_REDUCTION_WH_PER_MIN,
            )
            reduction_now = min(
                reduction_pending_wh,
                max(reduction_pending_wh * REDUCTION_RELEASE_RATE, 0.0),
                reduction_cap,
            )
            reduction_pending_wh -= reduction_now
        achieved_reduction_wh[index] = reduction_now
        deferred_pool_wh += reduction_now

        rebound_now = 0.0
        if is_cheap[index] and deferred_pool_wh > 0.0:
            rebound_cap = min(
                MAX_REBOUND_WH_PER_MIN,
                float(predicted_consumption_wh[index]) * REBOUND_MAX_CONSUMPTION_FRAC,
            )
            rebound_cap = min(
                rebound_cap,
                previous_rebound_wh + REBOUND_RAMP_DELTA_WH_PER_MIN,
            )
            rebound_now = min(
                deferred_pool_wh,
                max(deferred_pool_wh * REBOUND_RELEASE_RATE, 0.0),
                rebound_cap,
            )
            deferred_pool_wh -= rebound_now
        reduction_rebound_wh[index] = rebound_now
        previous_rebound_wh = rebound_now

        support_now = 0.0
        if trigger[index]:
            available_wh = (
                max(soc_pct[index] - BATTERY_MIN_SOC_PCT, 0.0) * BATTERY_CAPACITY_WH / 100.0
            )
            available_wh = max(available_wh - battery_debt_wh, 0.0)
            support_cap = predicted_consumption_wh[index] * BATTERY_SUPPORT_NET_LOAD_FRAC
            support_now = min(available_wh, BATTERY_MAX_SUPPORT_WH_PER_MIN, support_cap)
            if support_now > 0.0:
                battery_debt_wh += support_now
        battery_support_wh[index] = support_now

        battery_payback_now = 0.0
        if is_cheap[index] and battery_debt_wh > 0.0 and original_discharge_wh[index] > 0.0:
            battery_payback_cap = min(
                BATTERY_MAX_REBOUND_WH_PER_MIN,
                float(predicted_consumption_wh[index]) * BATTERY_REBOUND_MAX_CONSUMPTION_FRAC,
            )
            battery_payback_cap = min(
                battery_payback_cap,
                previous_battery_rebound_wh + BATTERY_REBOUND_RAMP_DELTA_WH_PER_MIN,
            )
            battery_payback_now = min(
                battery_debt_wh,
                max(battery_debt_wh * BATTERY_REBOUND_RELEASE_RATE, 0.0),
                battery_payback_cap,
            )
            battery_debt_wh -= battery_payback_now
        battery_rebound_wh[index] = battery_payback_now
        previous_battery_rebound_wh = battery_payback_now

    if reduction_pending_wh > 0.0:
        peak_indices = np.flatnonzero(is_peak_or_expensive)
        if len(peak_indices) > 0:
            per_slot_reduction = reduction_pending_wh / len(peak_indices)
            for index in peak_indices:
                reduction_cap = min(
                    float(potential_reduction_wh[index]),
                    float(predicted_consumption_wh[index]) * MAX_CONSUMPTION_REDUCTION_FRAC,
                    MAX_REDUCTION_WH_PER_MIN,
                )
                remaining_cap = max(reduction_cap - float(achieved_reduction_wh[index]), 0.0)
                refill = min(per_slot_reduction, remaining_cap, reduction_pending_wh)
                achieved_reduction_wh[index] += refill
                reduction_pending_wh -= refill
                deferred_pool_wh += refill
                if reduction_pending_wh <= 0.0:
                    break

    if deferred_pool_wh > 0.0 or battery_debt_wh > 0.0:
        cheap_indices = np.flatnonzero(is_cheap)
        if len(cheap_indices) > 0:
            per_slot_rebound = deferred_pool_wh / len(cheap_indices)
            per_slot_battery = battery_debt_wh / len(cheap_indices)
            for index in cheap_indices:
                if deferred_pool_wh > 0.0:
                    rebound_cap = min(
                        MAX_REBOUND_WH_PER_MIN,
                        float(predicted_consumption_wh[index]) * REBOUND_MAX_CONSUMPTION_FRAC,
                    )
                    previous_value = 0.0 if index == 0 else float(reduction_rebound_wh[index - 1])
                    ramp_headroom = max(
                        previous_value + REBOUND_RAMP_DELTA_WH_PER_MIN
                        - float(reduction_rebound_wh[index]),
                        0.0,
                    )
                    rebound_remaining_cap = max(
                        rebound_cap - float(reduction_rebound_wh[index]),
                        0.0,
                    )
                    refill = min(
                        per_slot_rebound,
                        ramp_headroom,
                        rebound_remaining_cap,
                        deferred_pool_wh,
                    )
                    reduction_rebound_wh[index] += refill
                    deferred_pool_wh -= refill
                if battery_debt_wh > 0.0 and original_discharge_wh[index] > 0.0:
                    battery_payback_cap = min(
                        BATTERY_MAX_REBOUND_WH_PER_MIN,
                        float(predicted_consumption_wh[index])
                        * BATTERY_REBOUND_MAX_CONSUMPTION_FRAC,
                    )
                    previous_value = 0.0 if index == 0 else float(battery_rebound_wh[index - 1])
                    ramp_headroom = max(
                        previous_value + BATTERY_REBOUND_RAMP_DELTA_WH_PER_MIN
                        - float(battery_rebound_wh[index]),
                        0.0,
                    )
                    battery_remaining_cap = max(
                        battery_payback_cap - float(battery_rebound_wh[index]),
                        0.0,
                    )
                    refill = min(
                        per_slot_battery,
                        ramp_headroom,
                        battery_remaining_cap,
                        battery_debt_wh,
                    )
                    battery_rebound_wh[index] += refill
                    battery_debt_wh -= refill

    adjusted_consumption_wh = (
        predicted_consumption_wh
        - achieved_reduction_wh
        + reduction_rebound_wh
        + battery_rebound_wh
    )
    adjusted_consumption_wh = np.clip(adjusted_consumption_wh, a_min=0.0, a_max=None)

    original_net_wh = predicted_consumption_wh - predicted_production_wh
    adjusted_net_wh = adjusted_consumption_wh - predicted_production_wh - battery_support_wh
    net_delta_wh = adjusted_net_wh - original_net_wh

    return {
        "predicted_consumption_wh": predicted_consumption_wh.astype(np.float32),
        "predicted_production_wh": predicted_production_wh.astype(np.float32),
        "original_net_wh": original_net_wh.astype(np.float32),
        "adjusted_net_wh": adjusted_net_wh.astype(np.float32),
        "net_delta_wh": net_delta_wh.astype(np.float32),
        "potential_reduction_wh": potential_reduction_wh.astype(np.float32),
        "achieved_reduction_wh": achieved_reduction_wh.astype(np.float32),
        "reduction_rebound_wh": reduction_rebound_wh.astype(np.float32),
        "battery_support_wh": battery_support_wh.astype(np.float32),
        "battery_rebound_wh": battery_rebound_wh.astype(np.float32),
        "shiftable_estimate_wh": shiftable_wh.astype(np.float32),
        "device_on_count": device_on_count.astype(np.float32),
        "action_triggered": action_triggered.astype(np.float32),
        "action_success": action_success.astype(np.float32),
    }


def load_grid_transformer_for_frequency(
    device: torch.device,
    seq_len: int,
) -> tuple[torch.nn.Module, float, float]:
    """Load and compile the grid frequency transformer model.

    :param device: Torch device.
    :param seq_len: Sequence length expected by trained model.
    :return: Tuple ``(model, feature_mean, feature_std)``.
    """
    artifact = torch.load(
        GRID_TRANSFORMER_ARTIFACT_PATH,
        map_location=device,
        weights_only=False,
    )
    helper = import_storenet_helpers()
    model = helper["SharedEnergyTransformer"](
        feature_dim=1,
        num_houses=1,
        max_seq_len=seq_len,
        d_model=int(artifact["model_params"]["d_model"]),
        house_embedding_dim=int(artifact["model_params"]["house_embedding_dim"]),
        num_layers=int(artifact["model_params"]["num_layers"]),
        num_heads=int(artifact["model_params"]["num_heads"]),
        feedforward_dim=int(artifact["model_params"]["feedforward_dim"]),
        dropout=float(artifact["model_params"]["dropout"]),
    ).to(device)
    model.load_state_dict(artifact["best_state_dict"])
    model = torch.compile(model)
    model.eval()
    return model, float(artifact["feature_mean"]), float(artifact["feature_std"])


def predict_grid_frequency(
    model: torch.nn.Module,
    values: np.ndarray,
    feature_mean: float,
    feature_std: float,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predict frequency for every minute using a transformer with clipped history windows.

    :param model: Compiled transformer model.
    :param values: Actual frequency values.
    :param feature_mean: Feature mean for normalization.
    :param feature_std: Feature std for normalization.
    :param seq_len: History window length.
    :param batch_size: Batch size.
    :param device: Torch device.
    :return: Predicted frequency vector.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    normalized = (values.astype(np.float32) - np.float32(feature_mean)) / np.float32(feature_std)
    row_count = len(normalized)
    offsets = np.arange(seq_len, dtype=np.int64) - seq_len
    output = np.empty(row_count, dtype=np.float32)

    with torch.inference_mode():
        for start in tqdm(
            range(0, row_count, batch_size),
            desc="Predicting grid frequency",
            unit="batch",
        ):
            end = min(start + batch_size, row_count)
            indices = np.arange(start, end, dtype=np.int64)
            window_indices = indices[:, None] + offsets[None, :]
            np.clip(window_indices, 0, row_count - 1, out=window_indices)
            windows = normalized[window_indices].reshape(len(indices), seq_len, 1)

            x = torch.from_numpy(windows).to(device, non_blocking=True)
            house_ids = torch.zeros((len(indices),), dtype=torch.int64, device=device)
            predicted = model(x, house_ids).squeeze(-1).detach().cpu().numpy()
            output[start:end] = predicted * feature_std + feature_mean
    return output


def main() -> None:
    """Run the full simulation and write output artifacts."""
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    lcoe_values = dict(LCOE_USD_PER_MWH)
    ROOT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    grid_frame = load_grid_2020_minute_frame(
        normalized_dir=GRID_NORMALIZED_DIR,
        chunksize=GRID_READ_CHUNKSIZE,
    )
    grid_frame = grid_frame[grid_frame["timestamp_utc"].dt.year == SIM_YEAR].reset_index(
        drop=True
    )
    grid_frame["grid_lcoe_usd_per_mwh"] = compute_lcoe_price_series(grid_frame, lcoe_values)
    if grid_frame["grid_lcoe_usd_per_mwh"].isna().any():
        raise RuntimeError("LCOE price series contains NaN after fallback handling.")

    load_threshold = float(grid_frame["demand_tsd"].quantile(PEAK_LOAD_QUANTILE))
    expensive_threshold = float(
        grid_frame["grid_lcoe_usd_per_mwh"].quantile(EXPENSIVE_PRICE_QUANTILE)
    )
    cheap_threshold = float(grid_frame["grid_lcoe_usd_per_mwh"].quantile(CHEAP_PRICE_QUANTILE))

    grid_frame["is_peak_load"] = (grid_frame["demand_tsd"] >= load_threshold).astype(np.int8)
    grid_frame["is_expensive"] = (
        grid_frame["grid_lcoe_usd_per_mwh"] >= expensive_threshold
    ).astype(np.int8)
    grid_frame["is_peak_or_expensive"] = (
        (grid_frame["is_peak_load"] == 1) | (grid_frame["is_expensive"] == 1)
    ).astype(np.int8)
    grid_frame["is_cheap"] = (
        grid_frame["grid_lcoe_usd_per_mwh"] <= cheap_threshold
    ).astype(np.int8)

    timestamps = pd.DatetimeIndex(grid_frame["timestamp_utc"])
    policy_frame = grid_frame.set_index("timestamp_utc")[
        ["is_peak_or_expensive", "is_cheap"]
    ].copy()
    policy_frame["is_peak_or_expensive"] = policy_frame["is_peak_or_expensive"].astype(bool)
    policy_frame["is_cheap"] = policy_frame["is_cheap"].astype(bool)

    needs_consumer_inference = not USE_PRECOMPUTED_CONSUMER_PREDICTIONS
    if USE_PRECOMPUTED_CONSUMER_PREDICTIONS:
        for house in HOUSE_IDS:
            if _load_precomputed_predictions(house) is None:
                needs_consumer_inference = True
                break

    consumer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    consumer_context = None
    if needs_consumer_inference:
        consumer_context = load_consumer_inference_context(device=consumer_device)

    aggregate_columns = [
        "predicted_consumption_wh",
        "predicted_production_wh",
        "original_net_wh",
        "adjusted_net_wh",
        "net_delta_wh",
        "potential_reduction_wh",
        "achieved_reduction_wh",
        "reduction_rebound_wh",
        "battery_support_wh",
        "battery_rebound_wh",
        "shiftable_estimate_wh",
        "device_on_count",
        "action_triggered",
        "action_success",
    ]
    consumer_aggregate = {
        column: np.zeros(len(grid_frame), dtype=np.float64) for column in aggregate_columns
    }

    for house_index, house in enumerate(tqdm(HOUSE_IDS, desc="Simulating houses", unit="house")):
        prediction_frame = load_house_prediction_frame(house=house, context=consumer_context)
        end_use_frame = load_house_end_use_frame(house=house)
        simulation = simulate_house_response(
            house=house,
            grid_timestamps=timestamps,
            policy_frame=policy_frame,
            prediction_frame=prediction_frame,
            end_use_frame=end_use_frame,
            seed=RNG_SEED + house_index * 101,
        )
        for column in aggregate_columns:
            consumer_aggregate[column] += simulation[column]

    for column in aggregate_columns:
        grid_frame[f"consumer_{column}_raw"] = consumer_aggregate[column]
        grid_frame[f"consumer_{column}_scaled"] = (
            consumer_aggregate[column] * CONSUMER_GROUP_MULTIPLIER
        )

    grid_frame["consumer_group_multiplier"] = float(CONSUMER_GROUP_MULTIPLIER)

    demand_original_mw = grid_frame["demand_tsd"].to_numpy(dtype=np.float64)
    net_delta_wh_scaled = grid_frame["consumer_net_delta_wh_scaled"].to_numpy(dtype=np.float64)
    grid_delta_mw = net_delta_wh_scaled * 60.0 / 1_000_000.0
    demand_adjusted_mw = np.clip(demand_original_mw + grid_delta_mw, a_min=0.0, a_max=None)

    grid_frame["grid_demand_tsd_original_mw"] = demand_original_mw.astype(np.float32)
    grid_frame["grid_demand_tsd_adjusted_mw"] = demand_adjusted_mw.astype(np.float32)
    grid_frame["grid_delta_mw_from_consumers"] = grid_delta_mw.astype(np.float32)

    generation_adjustment = apply_generation_merit_reduction(
        grid_frame=grid_frame,
        lcoe=lcoe_values,
        demand_original_mw=demand_original_mw,
        demand_adjusted_mw=demand_adjusted_mw,
    )
    for column, values in generation_adjustment.items():
        grid_frame[column] = values.astype(np.float32)

    adjusted_pricing_frame = pd.DataFrame(index=grid_frame.index)
    for column in LCOE_BY_GENERATION_COLUMN:
        adjusted_column = f"{column}_adjusted_mw"
        if adjusted_column in grid_frame.columns:
            adjusted_pricing_frame[column] = pd.to_numeric(
                grid_frame[adjusted_column],
                errors="coerce",
            ).fillna(0.0)
        elif column in grid_frame.columns:
            adjusted_pricing_frame[column] = pd.to_numeric(
                grid_frame[column],
                errors="coerce",
            ).fillna(0.0)
    if "generation_generation_adjusted_mw" in grid_frame.columns:
        adjusted_pricing_frame["generation_generation"] = pd.to_numeric(
            grid_frame["generation_generation_adjusted_mw"],
            errors="coerce",
        ).fillna(0.0)
    elif "generation_generation" in grid_frame.columns:
        adjusted_pricing_frame["generation_generation"] = pd.to_numeric(
            grid_frame["generation_generation"],
            errors="coerce",
        ).fillna(0.0)

    price_original_per_mwh = pd.to_numeric(
        grid_frame["grid_lcoe_usd_per_mwh"],
        errors="coerce",
    ).fillna(0.0)
    price_adjusted_per_mwh = compute_lcoe_price_series(adjusted_pricing_frame, lcoe_values)
    price_adjusted_per_mwh = pd.to_numeric(price_adjusted_per_mwh, errors="coerce").fillna(0.0)
    grid_frame["grid_lcoe_adjusted_usd_per_mwh"] = price_adjusted_per_mwh.astype(np.float32)
    grid_frame["grid_lcoe_delta_usd_per_mwh"] = (
        price_adjusted_per_mwh - price_original_per_mwh
    ).astype(np.float32)

    grid_results = json.loads(GRID_MODEL_RESULTS_PATH.read_text(encoding="utf-8"))
    seq_len = int(grid_results["data_params"]["seq_len"])
    freq_model, freq_mean, freq_std = load_grid_transformer_for_frequency(
        device=consumer_device,
        seq_len=seq_len,
    )
    frequency_actual = grid_frame["frequency_hz"].to_numpy(dtype=np.float32)
    frequency_predicted = predict_grid_frequency(
        model=freq_model,
        values=frequency_actual,
        feature_mean=freq_mean,
        feature_std=freq_std,
        seq_len=seq_len,
        batch_size=GRID_FREQ_INFER_BATCH_SIZE,
        device=consumer_device,
    )
    frequency_adjusted = frequency_predicted - grid_delta_mw.astype(np.float32) * np.float32(
        GRID_FREQ_RESPONSE_HZ_PER_MW
    )

    grid_frame["grid_frequency_actual_hz"] = frequency_actual
    grid_frame["grid_frequency_predicted_hz"] = frequency_predicted
    grid_frame["grid_frequency_predicted_adjusted_hz"] = frequency_adjusted

    price_original_mwh = price_original_per_mwh.to_numpy(dtype=np.float64, copy=False)
    price_adjusted_mwh = price_adjusted_per_mwh.to_numpy(dtype=np.float64, copy=False)
    grid_cost_original = demand_original_mw / 60.0 * price_original_mwh
    grid_cost_adjusted = demand_adjusted_mw / 60.0 * price_adjusted_mwh
    grid_cost_original = np.nan_to_num(grid_cost_original, nan=0.0, posinf=0.0, neginf=0.0)
    grid_cost_adjusted = np.nan_to_num(grid_cost_adjusted, nan=0.0, posinf=0.0, neginf=0.0)
    grid_frame["grid_cost_original_usd_per_min"] = grid_cost_original.astype(np.float32)
    grid_frame["grid_cost_adjusted_usd_per_min"] = grid_cost_adjusted.astype(np.float32)
    grid_frame["grid_cost_delta_usd_per_min"] = (grid_cost_adjusted - grid_cost_original).astype(
        np.float32
    )

    consumer_import_original_wh = np.clip(
        grid_frame["consumer_original_net_wh_scaled"].to_numpy(dtype=np.float64),
        a_min=0.0,
        a_max=None,
    )
    consumer_import_adjusted_wh = np.clip(
        grid_frame["consumer_adjusted_net_wh_scaled"].to_numpy(dtype=np.float64),
        a_min=0.0,
        a_max=None,
    )
    consumer_cost_original = consumer_import_original_wh / 1_000_000.0 * price_original_mwh
    consumer_cost_adjusted = consumer_import_adjusted_wh / 1_000_000.0 * price_adjusted_mwh
    consumer_cost_original = np.nan_to_num(
        consumer_cost_original,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    consumer_cost_adjusted = np.nan_to_num(
        consumer_cost_adjusted,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    grid_frame["consumer_cost_original_usd_per_min"] = consumer_cost_original.astype(np.float32)
    grid_frame["consumer_cost_adjusted_usd_per_min"] = consumer_cost_adjusted.astype(np.float32)
    grid_frame["consumer_cost_delta_usd_per_min"] = (
        consumer_cost_adjusted - consumer_cost_original
    ).astype(np.float32)

    grid_frame["grid_cost_original_usd_cumulative"] = np.cumsum(grid_cost_original).astype(
        np.float64
    )
    grid_frame["grid_cost_adjusted_usd_cumulative"] = np.cumsum(grid_cost_adjusted).astype(
        np.float64
    )
    grid_frame["consumer_cost_original_usd_cumulative"] = np.cumsum(consumer_cost_original).astype(
        np.float64
    )
    grid_frame["consumer_cost_adjusted_usd_cumulative"] = np.cumsum(consumer_cost_adjusted).astype(
        np.float64
    )

    export_frame = build_export_frame(grid_frame)
    export_frame.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        compression={
            "method": "gzip",
            "compresslevel": OUTPUT_GZIP_COMPRESSLEVEL,
        },
    )
    first_week_export_frame = build_first_week_export_frame(grid_frame)
    first_week_export_frame.to_csv(OUTPUT_FIRST_WEEK_CSV_PATH, index=False)

    summary = {
        "sim_year": SIM_YEAR,
        "rows_output": int(len(grid_frame)),
        "rows_output_first_week": int(len(first_week_export_frame)),
        "consumer_group_multiplier": float(CONSUMER_GROUP_MULTIPLIER),
        "action_effectiveness_probability": float(ACTION_EFFECTIVENESS_PROB),
        "peak_load_threshold_mw": load_threshold,
        "expensive_price_threshold_usd_per_mwh": expensive_threshold,
        "cheap_price_threshold_usd_per_mwh": cheap_threshold,
        "consumer_total_original_import_kwh_scaled": float(
            np.nansum(consumer_import_original_wh) / 1000.0
        ),
        "consumer_total_adjusted_import_kwh_scaled": float(
            np.nansum(consumer_import_adjusted_wh) / 1000.0
        ),
        "consumer_total_reduction_kwh_scaled": float(
            np.nansum(grid_frame["consumer_achieved_reduction_wh_scaled"]) / 1000.0
        ),
        "consumer_total_rebound_kwh_scaled": float(
            np.nansum(grid_frame["consumer_reduction_rebound_wh_scaled"]) / 1000.0
        ),
        "consumer_total_battery_support_kwh_scaled": float(
            np.nansum(grid_frame["consumer_battery_support_wh_scaled"]) / 1000.0
        ),
        "consumer_total_battery_rebound_kwh_scaled": float(
            np.nansum(grid_frame["consumer_battery_rebound_wh_scaled"]) / 1000.0
        ),
        "consumer_actions_triggered_scaled": float(
            np.nansum(grid_frame["consumer_action_triggered_scaled"])
        ),
        "consumer_actions_success_scaled": float(
            np.nansum(grid_frame["consumer_action_success_scaled"])
        ),
        "grid_total_cost_original_usd": float(np.nansum(grid_cost_original)),
        "grid_total_cost_adjusted_usd": float(np.nansum(grid_cost_adjusted)),
        "grid_total_cost_savings_usd": float(
            np.nansum(grid_cost_original - grid_cost_adjusted)
        ),
        "grid_avg_lcoe_original_usd_per_mwh": float(np.nanmean(price_original_mwh)),
        "grid_avg_lcoe_adjusted_usd_per_mwh": float(np.nanmean(price_adjusted_mwh)),
        "grid_total_generation_reduced_mwh": float(
            np.nansum(grid_frame.get("generation_total_reduced_mw", 0.0)) / 60.0
        ),
        "grid_unserved_reduction_mwh": float(
            np.nansum(grid_frame.get("remaining_unserved_reduction_mw", 0.0)) / 60.0
        ),
        "grid_nuclear_reduced_mwh": float(
            np.nansum(grid_frame.get("generation_nuclear_reduced_mw", 0.0)) / 60.0
        ),
        "consumer_total_cost_original_usd": float(np.nansum(consumer_cost_original)),
        "consumer_total_cost_adjusted_usd": float(np.nansum(consumer_cost_adjusted)),
        "consumer_total_cost_savings_usd": float(
            np.nansum(consumer_cost_original - consumer_cost_adjusted)
        ),
        "output_csv_gz": str(OUTPUT_CSV_PATH.resolve()),
        "output_csv_first_week": str(OUTPUT_FIRST_WEEK_CSV_PATH.resolve()),
    }

    OUTPUT_SUMMARY_JSON_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Joint 2020 consumer-grid simulation complete.")
    print(f"Rows written: {summary['rows_output']}")
    print(f"First-week rows written: {summary['rows_output_first_week']}")
    print(f"Output CSV: {OUTPUT_CSV_PATH}")
    print(f"First-week CSV: {OUTPUT_FIRST_WEEK_CSV_PATH}")
    print(f"Summary JSON: {OUTPUT_SUMMARY_JSON_PATH}")
    print(
        "Consumer cost (USD): "
        f"{summary['consumer_total_cost_original_usd']:,.2f} -> "
        f"{summary['consumer_total_cost_adjusted_usd']:,.2f} "
        f"(savings {summary['consumer_total_cost_savings_usd']:,.2f})"
    )
    print(
        "Grid cost (USD): "
        f"{summary['grid_total_cost_original_usd']:,.2f} -> "
        f"{summary['grid_total_cost_adjusted_usd']:,.2f} "
        f"(savings {summary['grid_total_cost_savings_usd']:,.2f})"
    )


if __name__ == "__main__":
    main()
