#!/usr/bin/env python3
"""Flask web UI for running one-week consumer-grid simulations."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from flask import Flask, jsonify, render_template, request

import plot_consumer_grid_first_week as plot_helpers
import run_consumer_grid_2020_simulation as sim


matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PROJECT_ROOT / "static"
TEMPLATE_ROOT = PROJECT_ROOT / "templates"
WEEK_DURATION = pd.Timedelta(days=7)

AGGREGATE_COLUMNS = [
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


@dataclass(frozen=True)
class NumericOptionSpec:
    """Numeric option metadata exposed to the UI."""

    key: str
    label: str
    module_var: str
    min_value: float
    max_value: float
    step: float


NUMERIC_OPTION_SPECS = (
    NumericOptionSpec(
        key="consumer_group_multiplier",
        label="Consumer Group Multiplier",
        module_var="CONSUMER_GROUP_MULTIPLIER",
        min_value=1.0,
        max_value=1_000_000.0,
        step=1.0,
    ),
    NumericOptionSpec(
        key="action_effectiveness_prob",
        label="Action Effectiveness Probability",
        module_var="ACTION_EFFECTIVENESS_PROB",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    ),
    NumericOptionSpec(
        key="peak_load_quantile",
        label="Peak Load Quantile",
        module_var="PEAK_LOAD_QUANTILE",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    ),
    NumericOptionSpec(
        key="expensive_price_quantile",
        label="Expensive Price Quantile",
        module_var="EXPENSIVE_PRICE_QUANTILE",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    ),
    NumericOptionSpec(
        key="cheap_price_quantile",
        label="Cheap Price Quantile",
        module_var="CHEAP_PRICE_QUANTILE",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    ),
    NumericOptionSpec(
        key="min_actionable_reduction_wh",
        label="Min Actionable Reduction (Wh)",
        module_var="MIN_ACTIONABLE_REDUCTION_WH",
        min_value=0.0,
        max_value=10_000.0,
        step=1.0,
    ),
    NumericOptionSpec(
        key="max_consumption_reduction_frac",
        label="Max Consumption Reduction Fraction",
        module_var="MAX_CONSUMPTION_REDUCTION_FRAC",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    ),
    NumericOptionSpec(
        key="shiftable_reduction_fraction",
        label="Shiftable Reduction Fraction",
        module_var="SHIFTABLE_REDUCTION_FRACTION",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    ),
    NumericOptionSpec(
        key="battery_support_net_load_frac",
        label="Battery Support Net-Load Fraction",
        module_var="BATTERY_SUPPORT_NET_LOAD_FRAC",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    ),
    NumericOptionSpec(
        key="grid_freq_response_hz_per_mw",
        label="Grid Frequency Response (Hz per MW)",
        module_var="GRID_FREQ_RESPONSE_HZ_PER_MW",
        min_value=0.0,
        max_value=0.01,
        step=0.000001,
    ),
)

DEFAULT_NUMERIC_OPTIONS = {
    spec.key: float(getattr(sim, spec.module_var)) for spec in NUMERIC_OPTION_SPECS
}


@dataclass
class CachedSimulationData:
    """In-memory inputs reused across week simulations."""

    consumer_context: sim.ConsumerInferenceContext
    grid_frame: pd.DataFrame
    house_inference_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]]
    house_end_use_frame: dict[str, pd.DataFrame]
    week_ranges: list[tuple[pd.Timestamp, pd.Timestamp]]
    frequency_model: torch.nn.Module
    frequency_mean: float
    frequency_std: float
    frequency_seq_len: int
    device: torch.device
    grid_week_frames: dict[int, pd.DataFrame]
    house_end_use_week_frames: dict[tuple[str, int], pd.DataFrame]


_CACHE: CachedSimulationData | None = None
_CACHE_LOCK = threading.Lock()
_SIMULATION_LOCK = threading.Lock()


def _parse_numeric_options(payload: dict[str, Any]) -> dict[str, float]:
    """Parse and validate numeric simulation options from the request payload.

    :param payload: Request JSON dictionary.
    :return: Sanitized numeric options.
    :raises ValueError: If any option is out of range or non-numeric.
    """
    options: dict[str, float] = {}
    for spec in NUMERIC_OPTION_SPECS:
        raw_value = payload.get(spec.key, DEFAULT_NUMERIC_OPTIONS[spec.key])
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{spec.key} must be numeric.") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"{spec.key} must be finite.")
        if parsed < spec.min_value or parsed > spec.max_value:
            raise ValueError(
                f"{spec.key} must be between {spec.min_value} and {spec.max_value}."
            )
        options[spec.key] = parsed

    if options["cheap_price_quantile"] >= options["expensive_price_quantile"]:
        raise ValueError("cheap_price_quantile must be less than expensive_price_quantile.")
    return options


def _build_week_ranges(timestamps: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Build contiguous seven-day ranges from timestamps.

    :param timestamps: Timestamp series from the grid frame.
    :return: List of ``(week_start, week_end)`` tuples.
    """
    parsed = pd.to_datetime(timestamps, errors="coerce", utc=True)
    parsed = parsed.dropna()
    if parsed.empty:
        return []

    first_timestamp = parsed.min().floor("D")
    last_timestamp = parsed.max()
    one_minute = pd.Timedelta(minutes=1)

    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = first_timestamp
    while start + WEEK_DURATION <= last_timestamp + one_minute:
        ranges.append((start, start + WEEK_DURATION))
        start += WEEK_DURATION
    return ranges


def _build_sim_year_week_ranges(year: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Build complete seven-day ranges for a fixed simulation year.

    :param year: Simulation year.
    :return: List of complete ``(week_start, week_end)`` ranges in UTC.
    """
    year_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    year_end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = year_start
    while start + WEEK_DURATION <= year_end:
        ranges.append((start, start + WEEK_DURATION))
        start += WEEK_DURATION
    return ranges


STATIC_WEEK_RANGES = _build_sim_year_week_ranges(sim.SIM_YEAR)


def _slice_frame_for_week(
    frame: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> pd.DataFrame:
    """Return one-week frame slice.

    :param frame: Source frame containing ``timestamp_utc``.
    :param week_start: Inclusive UTC start.
    :param week_end: Exclusive UTC end.
    :return: Week-limited copy.
    """
    timestamps = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    mask = (timestamps >= week_start) & (timestamps < week_end)
    return frame.loc[mask].copy()


def _load_grid_frame_for_year() -> pd.DataFrame:
    """Load the full simulation-year grid frame once.

    :return: Minute-level grid frame for the configured simulation year.
    """
    frame = sim.load_grid_2020_minute_frame(
        normalized_dir=sim.GRID_NORMALIZED_DIR,
        chunksize=sim.GRID_READ_CHUNKSIZE,
    )
    timestamps = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    frame = frame.loc[timestamps.dt.year == sim.SIM_YEAR].copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp_utc"]).reset_index(drop=True)
    return frame


def _load_house_cache_entry(
    house: str,
    consumer_context: sim.ConsumerInferenceContext,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cached inference and end-use inputs for one house.

    :param house: House identifier.
    :param consumer_context: Shared consumer inference context.
    :return: House id, raw frame, processed frame, and end-use frame.
    """
    raw_frame, processed_frame = sim.preprocess_house_for_inference(
        energy_path=sim.CONSUMER_DATA_DIR / f"{house}_Wh.csv",
        weather=consumer_context.weather_frame,
        feature_columns=consumer_context.feature_columns,
    )
    raw_frame = raw_frame.copy()
    raw_frame.columns = [column.strip() for column in raw_frame.columns]
    raw_frame["timestamp_utc"] = pd.to_datetime(raw_frame["date"], errors="coerce", utc=True)

    processed_frame = processed_frame.copy()
    processed_frame["timestamp_utc"] = pd.to_datetime(
        processed_frame["date"], errors="coerce", utc=True
    )

    end_use_frame = sim.load_house_end_use_frame(house=house)
    return house, raw_frame, processed_frame, end_use_frame


def _load_cache() -> CachedSimulationData:
    """Load reusable model/state data into process memory.

    :return: Cached simulation data.
    """
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is not None:
            return _CACHE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        consumer_context = sim.load_consumer_inference_context(device=device)
        grid_frame = _load_grid_frame_for_year()

        grid_results = json.loads(sim.GRID_MODEL_RESULTS_PATH.read_text(encoding="utf-8"))
        frequency_seq_len = int(grid_results["data_params"]["seq_len"])
        frequency_model, frequency_mean, frequency_std = sim.load_grid_transformer_for_frequency(
            device=device,
            seq_len=frequency_seq_len,
        )

        house_inference_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        house_end_use_frame: dict[str, pd.DataFrame] = {}
        max_workers = min(len(sim.HOUSE_IDS), max(1, (os.cpu_count() or 1)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_load_house_cache_entry, house, consumer_context)
                for house in sim.HOUSE_IDS
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Preloading house data",
                unit="house",
            ):
                house, raw_frame, processed_frame, end_use_frame = future.result()
                house_inference_data[house] = (raw_frame, processed_frame)
                house_end_use_frame[house] = end_use_frame

        _CACHE = CachedSimulationData(
            consumer_context=consumer_context,
            grid_frame=grid_frame,
            house_inference_data=house_inference_data,
            house_end_use_frame=house_end_use_frame,
            week_ranges=STATIC_WEEK_RANGES,
            frequency_model=frequency_model,
            frequency_mean=frequency_mean,
            frequency_std=frequency_std,
            frequency_seq_len=frequency_seq_len,
            device=device,
            grid_week_frames={},
            house_end_use_week_frames={},
        )
        return _CACHE


def _apply_numeric_options(options: dict[str, float]) -> None:
    """Apply request numeric options to simulation globals.

    :param options: Parsed option values.
    :return: None.
    """
    for spec in NUMERIC_OPTION_SPECS:
        setattr(sim, spec.module_var, float(options[spec.key]))


def _get_grid_frame_for_week(
    cache: CachedSimulationData,
    week_number: int,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> pd.DataFrame:
    """Return cached week grid frame sliced from preloaded year data.

    :param cache: Shared simulation cache.
    :param week_number: 1-based week number.
    :param week_start: Inclusive week start.
    :param week_end: Exclusive week end.
    :return: Week grid frame copy.
    """
    if week_number not in cache.grid_week_frames:
        cache.grid_week_frames[week_number] = _slice_frame_for_week(
            cache.grid_frame,
            week_start=week_start,
            week_end=week_end,
        )
    return cache.grid_week_frames[week_number].copy()


def _infer_house_predictions_for_week(
    raw_frame: pd.DataFrame,
    processed_frame: pd.DataFrame,
    house: str,
    context: sim.ConsumerInferenceContext,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> pd.DataFrame:
    """Run consumer-model inference for one house on a selected week only.

    :param raw_frame: Preloaded raw house frame.
    :param processed_frame: Preloaded processed house frame.
    :param house: House identifier.
    :param context: Shared consumer inference context.
    :param week_start: Inclusive week start.
    :param week_end: Exclusive week end.
    :return: Week-limited minute-level prediction frame.
    """
    week_processed = processed_frame.loc[
        (processed_frame["timestamp_utc"] >= week_start)
        & (processed_frame["timestamp_utc"] < week_end)
    ].copy()
    if week_processed.empty:
        raise RuntimeError(f"{house}: no consumer feature rows found for selected week.")

    if context.use_onnx:
        predictions = sim.predict_all_rows_consumer_onnx(
            context=context,
            processed_frame=week_processed,
            batch_size=sim.CONSUMER_INFER_BATCH_SIZE,
        )
    else:
        if context.model is None:
            raise RuntimeError("Torch consumer model context is not initialized.")
        predictions = sim.predict_all_rows(
            model=context.model,
            processed_frame=week_processed,
            feature_columns=context.feature_columns,
            feature_mean=context.feature_mean,
            feature_std=context.feature_std,
            target_mean=context.target_mean,
            target_std=context.target_std,
            seq_len=context.seq_len,
            batch_size=sim.CONSUMER_INFER_BATCH_SIZE,
            device=context.device,
        )

    raw_week = raw_frame.loc[
        (raw_frame["timestamp_utc"] >= week_start) & (raw_frame["timestamp_utc"] < week_end)
    ].copy()
    raw_week = raw_week.drop_duplicates(subset=["timestamp_utc"], keep="last")
    raw_week = raw_week.set_index("timestamp_utc")

    output = week_processed[["timestamp_utc"]].copy()
    output["predicted_production_wh"] = predictions[:, 0]
    output["predicted_consumption_wh"] = predictions[:, 1]

    output["State of Charge(%)"] = pd.to_numeric(
        raw_week.reindex(output["timestamp_utc"])["State of Charge(%)"],
        errors="coerce",
    )
    output["Discharge(Wh)"] = pd.to_numeric(
        raw_week.reindex(output["timestamp_utc"])["Discharge(Wh)"],
        errors="coerce",
    )
    output["State of Charge(%)"] = output["State of Charge(%)"].interpolate(
        method="linear",
        limit_direction="both",
    ).fillna(0.0)
    output["Discharge(Wh)"] = output["Discharge(Wh)"].interpolate(
        method="linear",
        limit_direction="both",
    ).fillna(0.0)

    output["minute_utc"] = output["timestamp_utc"].dt.floor("min")
    output = output.groupby("minute_utc", as_index=False).mean(numeric_only=True)
    return output.rename(columns={"minute_utc": "timestamp_utc"})


def _load_house_end_use_for_week(
    house_end_use_frame: pd.DataFrame,
    house: str,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> pd.DataFrame:
    """Load one house's end-use estimates for a selected week only.

    :param house_end_use_frame: Preloaded full-year end-use frame.
    :param house: House identifier.
    :param week_start: Inclusive week start.
    :param week_end: Exclusive week end.
    :return: Week-limited minute-level end-use frame.
    """
    frame = _slice_frame_for_week(
        house_end_use_frame,
        week_start=week_start,
        week_end=week_end,
    )
    if frame.empty:
        raise RuntimeError(f"{house}: no end-use rows found for selected week.")
    return frame


def _get_house_prediction_frame_for_week(
    cache: CachedSimulationData,
    house: str,
    week_number: int,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> pd.DataFrame:
    """Run and return one house's prediction frame for one week.

    :param cache: Shared simulation cache.
    :param house: House identifier.
    :param week_number: 1-based week number.
    :param week_start: Inclusive week start.
    :param week_end: Exclusive week end.
    :return: Week-limited prediction frame.
    """
    del week_number
    raw_frame, processed_frame = cache.house_inference_data[house]
    return _infer_house_predictions_for_week(
        raw_frame=raw_frame,
        processed_frame=processed_frame,
        house=house,
        context=cache.consumer_context,
        week_start=week_start,
        week_end=week_end,
    )


def _get_house_end_use_frame_for_week(
    cache: CachedSimulationData,
    house: str,
    week_number: int,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> pd.DataFrame:
    """Return cached per-house end-use frame for one week.

    :param cache: Shared simulation cache.
    :param house: House identifier.
    :param week_number: 1-based week number.
    :param week_start: Inclusive week start.
    :param week_end: Exclusive week end.
    :return: Week-limited end-use frame.
    """
    cache_key = (house, week_number)
    if cache_key not in cache.house_end_use_week_frames:
        cache.house_end_use_week_frames[cache_key] = _load_house_end_use_for_week(
            house_end_use_frame=cache.house_end_use_frame[house],
            house=house,
            week_start=week_start,
            week_end=week_end,
        )
    return cache.house_end_use_week_frames[cache_key].copy()


def _figure_grid_demand_base64(frame: pd.DataFrame, week_number: int) -> str:
    """Render grid-demand figure and return a base64 data URL.

    :param frame: Week frame with plotting columns.
    :param week_number: UI week number.
    :return: PNG data URL string.
    """
    timestamps = frame["timestamp_utc"]
    figure, axis = plt.subplots(1, 1, figsize=plot_helpers.FIGSIZE, constrained_layout=True)
    axis_diff = axis.twinx()
    axis.plot(
        timestamps,
        pd.to_numeric(frame["grid_demand_tsd_original_mw"], errors="coerce").fillna(0.0),
        color="black",
        linewidth=1.2,
        label="Grid Demand (Original)",
    )
    axis.plot(
        timestamps,
        pd.to_numeric(frame["grid_demand_tsd_adjusted_mw"], errors="coerce").fillna(0.0),
        color="#e31a1c",
        linewidth=1.2,
        label="Grid Demand (Adjusted)",
    )
    axis_diff.plot(
        timestamps,
        frame["grid_demand_difference_mw"],
        color="#1f78b4",
        linewidth=1.2,
        label="Demand Difference (Original - Adjusted)",
    )
    axis.set_ylabel("Power (MW)")
    axis_diff.set_ylabel("Demand Difference (MW)")
    axis.set_title(f"Week {week_number}: Grid Demand")
    left_handles, left_labels = axis.get_legend_handles_labels()
    right_handles, right_labels = axis_diff.get_legend_handles_labels()
    axis.legend(left_handles + right_handles, left_labels + right_labels, loc="upper right", ncol=2)
    axis.grid(alpha=0.25)
    plot_helpers.apply_time_axis(axis)

    image = io.BytesIO()
    figure.savefig(image, dpi=200, format="png")
    plt.close(figure)
    image.seek(0)
    encoded = base64.b64encode(image.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _figure_consumer_action_base64(frame: pd.DataFrame, week_number: int) -> str:
    """Render consumer-action figure and return a base64 data URL.

    :param frame: Week frame with plotting columns.
    :param week_number: UI week number.
    :return: PNG data URL string.
    """
    timestamps = frame["timestamp_utc"]
    figure, axis = plt.subplots(1, 1, figsize=plot_helpers.FIGSIZE, constrained_layout=True)
    axis.plot(
        timestamps,
        frame["consumer_reduction_mw"],
        color="#33a02c",
        linewidth=1.2,
        label="Consumer Reduction",
    )
    axis.plot(
        timestamps,
        frame["consumer_rebound_mw"],
        color="#ff7f00",
        linewidth=1.1,
        label="Consumer Rebound",
    )
    axis.plot(
        timestamps,
        frame["generation_total_reduced_mw"],
        color="#6a3d9a",
        linewidth=1.1,
        label="Generation Curtailed",
    )
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.set_ylabel("Power (MW)")
    axis.set_title(f"Week {week_number}: Consumer Action")
    axis.legend(loc="upper right", ncol=2)
    axis.grid(alpha=0.25)
    plot_helpers.apply_time_axis(axis)

    image = io.BytesIO()
    figure.savefig(image, dpi=200, format="png")
    plt.close(figure)
    image.seek(0)
    encoded = base64.b64encode(image.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _run_week_simulation(
    cache: CachedSimulationData,
    week_number: int,
    options: dict[str, float],
) -> dict[str, Any]:
    """Run the simulation for one selected week.

    :param cache: Cached simulation sources.
    :param week_number: 1-based week number from UI.
    :param options: Parsed numeric options.
    :return: API response payload.
    """
    # pylint: disable=too-many-locals,too-many-statements
    if week_number < 1 or week_number > len(cache.week_ranges):
        raise ValueError(f"week_number must be between 1 and {len(cache.week_ranges)}.")

    _apply_numeric_options(options)
    week_start, week_end = cache.week_ranges[week_number - 1]
    week_frame = _get_grid_frame_for_week(
        cache=cache,
        week_number=week_number,
        week_start=week_start,
        week_end=week_end,
    )
    if week_frame.empty:
        raise RuntimeError("Selected week has no rows after filtering.")

    lcoe_values = dict(sim.LCOE_USD_PER_MWH)
    week_frame["grid_lcoe_usd_per_mwh"] = sim.compute_lcoe_price_series(week_frame, lcoe_values)
    load_threshold = float(week_frame["demand_tsd"].quantile(sim.PEAK_LOAD_QUANTILE))
    expensive_threshold = float(
        week_frame["grid_lcoe_usd_per_mwh"].quantile(sim.EXPENSIVE_PRICE_QUANTILE)
    )
    cheap_threshold = float(
        week_frame["grid_lcoe_usd_per_mwh"].quantile(sim.CHEAP_PRICE_QUANTILE)
    )

    week_frame["is_peak_load"] = (week_frame["demand_tsd"] >= load_threshold).astype(np.int8)
    week_frame["is_expensive"] = (
        week_frame["grid_lcoe_usd_per_mwh"] >= expensive_threshold
    ).astype(np.int8)
    week_frame["is_peak_or_expensive"] = (
        (week_frame["is_peak_load"] == 1) | (week_frame["is_expensive"] == 1)
    ).astype(np.int8)
    week_frame["is_cheap"] = (
        week_frame["grid_lcoe_usd_per_mwh"] <= cheap_threshold
    ).astype(np.int8)

    timestamps = pd.DatetimeIndex(week_frame["timestamp_utc"])
    policy_frame = week_frame.set_index("timestamp_utc")[
        ["is_peak_or_expensive", "is_cheap"]
    ].copy()
    policy_frame["is_peak_or_expensive"] = policy_frame["is_peak_or_expensive"].astype(bool)
    policy_frame["is_cheap"] = policy_frame["is_cheap"].astype(bool)

    consumer_aggregate = {
        column: np.zeros(len(week_frame), dtype=np.float64) for column in AGGREGATE_COLUMNS
    }

    for house_index, house in enumerate(sim.HOUSE_IDS):
        prediction_frame = _get_house_prediction_frame_for_week(
            cache=cache,
            house=house,
            week_number=week_number,
            week_start=week_start,
            week_end=week_end,
        )
        end_use_frame = _get_house_end_use_frame_for_week(
            cache=cache,
            house=house,
            week_number=week_number,
            week_start=week_start,
            week_end=week_end,
        )
        simulation = sim.simulate_house_response(
            house=house,
            grid_timestamps=timestamps,
            policy_frame=policy_frame,
            prediction_frame=prediction_frame,
            end_use_frame=end_use_frame,
            seed=sim.RNG_SEED + house_index * 101,
        )
        for column in AGGREGATE_COLUMNS:
            consumer_aggregate[column] += simulation[column]

    for column in AGGREGATE_COLUMNS:
        week_frame[f"consumer_{column}_raw"] = consumer_aggregate[column]
        week_frame[f"consumer_{column}_scaled"] = (
            consumer_aggregate[column] * sim.CONSUMER_GROUP_MULTIPLIER
        )

    demand_original_mw = week_frame["demand_tsd"].to_numpy(dtype=np.float64)
    net_delta_wh_scaled = week_frame["consumer_net_delta_wh_scaled"].to_numpy(dtype=np.float64)
    grid_delta_mw = net_delta_wh_scaled * 60.0 / 1_000_000.0
    demand_adjusted_mw = np.clip(demand_original_mw + grid_delta_mw, a_min=0.0, a_max=None)

    week_frame["grid_demand_tsd_original_mw"] = demand_original_mw.astype(np.float32)
    week_frame["grid_demand_tsd_adjusted_mw"] = demand_adjusted_mw.astype(np.float32)
    week_frame["grid_delta_mw_from_consumers"] = grid_delta_mw.astype(np.float32)

    generation_adjustment = sim.apply_generation_merit_reduction(
        grid_frame=week_frame,
        lcoe=lcoe_values,
        demand_original_mw=demand_original_mw,
        demand_adjusted_mw=demand_adjusted_mw,
    )
    for column, values in generation_adjustment.items():
        week_frame[column] = values.astype(np.float32)

    adjusted_pricing_frame = pd.DataFrame(index=week_frame.index)
    for column in sim.LCOE_BY_GENERATION_COLUMN:
        adjusted_column = f"{column}_adjusted_mw"
        if adjusted_column in week_frame.columns:
            adjusted_pricing_frame[column] = pd.to_numeric(
                week_frame[adjusted_column], errors="coerce"
            ).fillna(0.0)
        elif column in week_frame.columns:
            adjusted_pricing_frame[column] = pd.to_numeric(
                week_frame[column], errors="coerce"
            ).fillna(0.0)
    if "generation_generation_adjusted_mw" in week_frame.columns:
        adjusted_pricing_frame["generation_generation"] = pd.to_numeric(
            week_frame["generation_generation_adjusted_mw"], errors="coerce"
        ).fillna(0.0)
    elif "generation_generation" in week_frame.columns:
        adjusted_pricing_frame["generation_generation"] = pd.to_numeric(
            week_frame["generation_generation"], errors="coerce"
        ).fillna(0.0)

    price_original_per_mwh = pd.to_numeric(
        week_frame["grid_lcoe_usd_per_mwh"],
        errors="coerce",
    ).fillna(0.0)
    price_adjusted_per_mwh = sim.compute_lcoe_price_series(adjusted_pricing_frame, lcoe_values)
    price_adjusted_per_mwh = pd.to_numeric(price_adjusted_per_mwh, errors="coerce").fillna(0.0)
    week_frame["grid_lcoe_adjusted_usd_per_mwh"] = price_adjusted_per_mwh.astype(np.float32)

    frequency_actual = week_frame["frequency_hz"].to_numpy(dtype=np.float32)
    frequency_predicted = sim.predict_grid_frequency(
        model=cache.frequency_model,
        values=frequency_actual,
        feature_mean=cache.frequency_mean,
        feature_std=cache.frequency_std,
        seq_len=cache.frequency_seq_len,
        batch_size=sim.GRID_FREQ_INFER_BATCH_SIZE,
        device=cache.device,
    )
    frequency_adjusted = frequency_predicted - grid_delta_mw.astype(np.float32) * np.float32(
        sim.GRID_FREQ_RESPONSE_HZ_PER_MW
    )
    week_frame["grid_frequency_predicted_hz"] = frequency_predicted
    week_frame["grid_frequency_predicted_adjusted_hz"] = frequency_adjusted

    price_original_mwh = price_original_per_mwh.to_numpy(dtype=np.float64, copy=False)
    price_adjusted_mwh = price_adjusted_per_mwh.to_numpy(dtype=np.float64, copy=False)
    grid_cost_original = demand_original_mw / 60.0 * price_original_mwh
    grid_cost_adjusted = demand_adjusted_mw / 60.0 * price_adjusted_mwh
    grid_cost_original = np.nan_to_num(grid_cost_original, nan=0.0, posinf=0.0, neginf=0.0)
    grid_cost_adjusted = np.nan_to_num(grid_cost_adjusted, nan=0.0, posinf=0.0, neginf=0.0)

    consumer_import_original_wh = np.clip(
        week_frame["consumer_original_net_wh_scaled"].to_numpy(dtype=np.float64),
        a_min=0.0,
        a_max=None,
    )
    consumer_import_adjusted_wh = np.clip(
        week_frame["consumer_adjusted_net_wh_scaled"].to_numpy(dtype=np.float64),
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

    consumer_original_total = float(np.nansum(consumer_cost_original))
    consumer_adjusted_total = float(np.nansum(consumer_cost_adjusted))
    grid_original_total = float(np.nansum(grid_cost_original))
    grid_adjusted_total = float(np.nansum(grid_cost_adjusted))
    power_difference_mw = float(np.nansum(grid_delta_mw))
    average_consumer_savings_usd = float(np.nanmean(consumer_cost_original - consumer_cost_adjusted))

    plot_frame = plot_helpers.add_plot_columns(week_frame)
    grid_demand_plot = _figure_grid_demand_base64(plot_frame, week_number=week_number)
    consumer_action_plot = _figure_consumer_action_base64(plot_frame, week_number=week_number)

    return {
        "week_number": week_number,
        "week_start_utc": week_start.isoformat(),
        "week_end_utc": week_end.isoformat(),
        "row_count": int(len(week_frame)),
        "thresholds": {
            "peak_load_threshold_mw": load_threshold,
            "expensive_price_threshold_usd_per_mwh": expensive_threshold,
            "cheap_price_threshold_usd_per_mwh": cheap_threshold,
        },
        "consumer_cost": {
            "original_usd": consumer_original_total,
            "adjusted_usd": consumer_adjusted_total,
            "difference_usd": consumer_adjusted_total - consumer_original_total,
            "savings_usd": consumer_original_total - consumer_adjusted_total,
            "average_savings_usd": average_consumer_savings_usd,
        },
        "grid_cost": {
            "original_usd": grid_original_total,
            "adjusted_usd": grid_adjusted_total,
            "difference_usd": grid_adjusted_total - grid_original_total,
            "savings_usd": grid_original_total - grid_adjusted_total,
        },
        "power": {
            "difference_mw": power_difference_mw,
        },
        "figures": {
            "grid_demand_png_data_url": grid_demand_plot,
            "consumer_action_png_data_url": consumer_action_plot,
        },
    }


def _build_week_options(
    week_ranges: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> list[dict[str, Any]]:
    """Convert week ranges into frontend-friendly dictionaries.

    :param week_ranges: List of week ranges.
    :return: Week dictionaries for template rendering.
    """
    output: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(week_ranges, start=1):
        output.append(
            {
                "number": index,
                "start": start.strftime("%Y-%m-%d"),
                "end": (end - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d"),
            }
        )
    return output


APP = Flask(
    __name__,
    template_folder=str(TEMPLATE_ROOT),
    static_folder=str(STATIC_ROOT),
    static_url_path="/static",
)
APP.config["TEMPLATES_AUTO_RELOAD"] = True
APP.jinja_env.auto_reload = True


@APP.get("/")
def index() -> str:
    """Render the main simulation UI page.

    :return: HTML page.
    """
    week_options = _build_week_options(STATIC_WEEK_RANGES)
    option_specs = [
        {
            "key": spec.key,
            "label": spec.label,
            "min": spec.min_value,
            "max": spec.max_value,
            "step": spec.step,
        }
        for spec in NUMERIC_OPTION_SPECS
    ]
    return render_template(
        "consumer_grid_simulation.html",
        week_options=week_options,
        option_specs=option_specs,
        default_options=DEFAULT_NUMERIC_OPTIONS,
    )


@APP.post("/api/simulate-week")
def simulate_week() -> Any:
    """Run selected week simulation and return metrics and figures.

    :return: JSON simulation result payload.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        week_number = int(payload.get("week_number"))
    except (TypeError, ValueError):
        return jsonify({"error": "week_number must be an integer."}), 400

    try:
        options = _parse_numeric_options(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        cache = _load_cache()
        with _SIMULATION_LOCK:
            result = _run_week_simulation(cache, week_number=week_number, options=options)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for server runtime mode.

    :return: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Run consumer-grid simulation web UI.")
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Run with waitress (2 threads) instead of Flask development server.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Bind port (default: 5000).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the web server in development or production mode."""
    args = _parse_args()
    if args.prod:
        _load_cache()
        # pylint: disable=import-outside-toplevel
        from waitress import serve

        serve(APP, host=args.host, port=args.port, threads=2)
        return
    APP.run(host=args.host, port=args.port, debug=True, use_reloader=True)


if __name__ == "__main__":
    main()
