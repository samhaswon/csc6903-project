#!/usr/bin/env python3
"""Estimate end-use draws with event-aware heuristics and HVAC split."""
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_exploration.storenet_ml.config import DATA_DIR


@dataclass
class Segment:
    """High-load segment metadata used for appliance classification."""

    start: int
    end: int
    duration_minutes: int
    peak_w: float
    mean_w: float
    start_hour: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :return: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Estimate appliance and HVAC end uses for one house or all houses from the "
            "consumption stream and weather context."
        )
    )
    parser.add_argument(
        "--house",
        type=str,
        default="all",
        help="House identifier (for example H1) or 'all' to process every H*_Wh.csv file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing H*_Wh.csv and weather.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_exploration/artifacts/end_use_estimates"),
        help="Directory for tabular output artifacts.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("dataset_exploration/figures/end_use_estimates"),
        help="Directory for visualization outputs.",
    )
    parser.add_argument(
        "--event-threshold-w",
        type=float,
        default=300.0,
        help="Absolute minute-to-minute power change threshold for event detection (W).",
    )
    return parser.parse_args()


def read_house_and_weather(data_dir: Path, house: str) -> pd.DataFrame:
    """Read house Wh and weather data, then align on a 1-minute timeline.

    :param data_dir: Base data directory.
    :param house: House id, for example H1.
    :return: Minute-aligned dataframe.
    """
    house_path = data_dir / f"{house}_Wh.csv"
    weather_path = data_dir / "weather.csv"

    house_frame = pd.read_csv(house_path, dtype="string", low_memory=False)
    house_frame.columns = [column.strip() for column in house_frame.columns]
    house_frame["date"] = pd.to_datetime(house_frame["date"], errors="coerce")
    numeric_columns = [
        "Discharge(Wh)",
        "Charge(Wh)",
        "Production(Wh)",
        "Consumption(Wh)",
        "Feed-in(Wh)",
        "From grid(Wh)",
        "State of Charge(%)",
    ]
    for column in numeric_columns:
        house_frame[column] = pd.to_numeric(house_frame[column], errors="coerce")
    house_frame = house_frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    house_frame = house_frame.set_index("date")

    full_index = pd.date_range(house_frame.index.min(), house_frame.index.max(), freq="1min")
    house_frame = house_frame.reindex(full_index)
    house_frame.index.name = "date"
    house_frame["Consumption(Wh)"] = house_frame["Consumption(Wh)"].interpolate(
        method="linear",
        limit_direction="both",
    )

    weather_frame = pd.read_csv(weather_path, dtype="string", low_memory=False)
    weather_frame["date"] = pd.to_datetime(
        weather_frame["date"].astype("string").str.strip(),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    for column in ["drybulb", "soltot"]:
        weather_frame[column] = pd.to_numeric(weather_frame[column], errors="coerce")
    weather_frame = weather_frame.dropna(subset=["date"]).sort_values("date")
    weather_frame = weather_frame.drop_duplicates("date")
    weather_frame = weather_frame.set_index("date")
    weather_frame = weather_frame.reindex(full_index)
    weather_frame[["drybulb", "soltot"]] = weather_frame[["drybulb", "soltot"]].interpolate(
        method="linear",
        limit_direction="both",
    )

    merged = house_frame.join(weather_frame[["drybulb", "soltot"]], how="left")
    merged = merged.reset_index().rename(columns={"index": "date"})
    merged["house"] = house
    return merged


def detect_events(consumption_w: np.ndarray, threshold_w: float) -> np.ndarray:
    """Detect event timestamps from absolute power deltas.

    :param consumption_w: Per-minute power signal in W.
    :param threshold_w: Event threshold in W.
    :return: Boolean event mask.
    """
    delta = np.diff(consumption_w, prepend=consumption_w[0])
    return np.abs(delta) >= threshold_w


def estimate_baseline(consumption_w: pd.Series) -> pd.Series:
    """Estimate always-on baseline via rolling lower quantile.

    :param consumption_w: Consumption power series in W.
    :return: Baseline estimate series.
    """
    baseline = consumption_w.rolling(
        window=24 * 60,
        min_periods=60,
        center=True,
    ).quantile(0.1)
    baseline = baseline.bfill().ffill()
    baseline = baseline.clip(lower=0.0)
    return baseline


def fit_hvac_components(
    frame: pd.DataFrame,
    residual_w: np.ndarray,
    event_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit heating and cooling HVAC components from temperature sensitivity.

    :param frame: Input dataframe with `drybulb`.
    :param residual_w: Positive residual load after baseline subtraction (W).
    :param event_mask: Event mask used to avoid fitting on sharp transients.
    :return: Tuple ``(hvac_heating_w, hvac_cooling_w)``.
    """
    temp = frame["drybulb"].to_numpy(dtype=np.float32)
    heating_degree = np.maximum(0.0, 17.0 - temp)
    cooling_degree = np.maximum(0.0, temp - 20.0)
    design = np.column_stack([heating_degree, cooling_degree]).astype(np.float32)

    fit_mask = (~event_mask) & (residual_w > 0.0) & (residual_w <= np.quantile(residual_w, 0.9))
    if np.count_nonzero(fit_mask) < 100:
        fit_mask = residual_w > 0.0

    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(design[fit_mask], residual_w[fit_mask].astype(np.float32))
    hvac_heating = model.coef_[0] * heating_degree
    hvac_cooling = model.coef_[1] * cooling_degree

    hvac_total = hvac_heating + hvac_cooling
    scale = np.ones_like(hvac_total, dtype=np.float32)
    over = hvac_total > residual_w
    scale[over] = residual_w[over] / np.maximum(hvac_total[over], 1e-6)
    hvac_heating *= scale
    hvac_cooling *= scale
    return hvac_heating.astype(np.float32), hvac_cooling.astype(np.float32)


def extract_high_load_segments(non_hvac_w: np.ndarray, threshold_w: float) -> list[Segment]:
    """Extract contiguous high-load segments for appliance classification.

    :param non_hvac_w: Non-HVAC residual load in W.
    :param threshold_w: Segment activation threshold in W.
    :return: Segment list.
    """
    active = non_hvac_w >= threshold_w
    segments: list[Segment] = []
    start = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        if not flag and start is not None:
            end = index
            segment = non_hvac_w[start:end]
            segments.append(
                Segment(
                    start=start,
                    end=end,
                    duration_minutes=end - start,
                    peak_w=float(np.max(segment)),
                    mean_w=float(np.mean(segment)),
                    start_hour=int((start % (24 * 60)) // 60),
                )
            )
            start = None
    if start is not None:
        end = len(non_hvac_w)
        segment = non_hvac_w[start:end]
        segments.append(
            Segment(
                start=start,
                end=end,
                duration_minutes=end - start,
                peak_w=float(np.max(segment)),
                mean_w=float(np.mean(segment)),
                start_hour=int((start % (24 * 60)) // 60),
            )
        )
    return segments


def classify_device_segments(
    high_load_w: np.ndarray,
    segments: list[Segment],
) -> dict[str, np.ndarray]:
    """Classify high-load segments into per-device power channels.

    :param high_load_w: High-load non-HVAC residual signal in W.
    :param segments: Candidate event segments.
    :return: Mapping from `device_*_w` column name to power series.
    """
    device_names = [
        "device_water_heater_w",
        "device_cooker_w",
        "device_kettle_w",
        "device_microwave_w",
        "device_washing_machine_w",
        "device_dishwasher_w",
        "device_tumble_dryer_w",
        "device_misc_event_w",
    ]
    device_series = {name: np.zeros_like(high_load_w, dtype=np.float32) for name in device_names}

    for segment in segments:
        start, end = segment.start, segment.end
        peak = segment.peak_w
        duration = segment.duration_minutes
        hour = segment.start_hour

        in_morning_evening = hour in set(range(5, 10)).union(range(17, 23))
        in_meal_window = hour in set(range(6, 10)).union(range(11, 15), range(17, 22))
        in_evening_or_night = hour >= 18 or hour <= 6

        if peak >= 1800.0 and duration <= 8:
            label = "device_kettle_w"
        elif 500.0 <= peak < 1800.0 and 1 <= duration <= 20 and in_meal_window:
            label = "device_microwave_w"
        elif 1600.0 <= peak <= 4500.0 and 20 <= duration <= 180 and in_meal_window:
            label = "device_cooker_w"
        elif 2000.0 <= peak <= 4000.0 and duration <= 90 and in_morning_evening:
            label = "device_water_heater_w"
        elif 500.0 <= peak <= 2500.0 and 40 <= duration <= 180 and not in_evening_or_night:
            label = "device_washing_machine_w"
        elif 700.0 <= peak <= 2200.0 and 60 <= duration <= 240 and in_evening_or_night:
            label = "device_dishwasher_w"
        elif 1200.0 <= peak <= 3500.0 and 30 <= duration <= 180:
            label = "device_tumble_dryer_w"
        elif 1400.0 <= peak <= 3500.0 and duration <= 12 and in_morning_evening:
            label = "device_kettle_w"
        elif 700.0 <= peak <= 2200.0 and duration <= 15 and in_meal_window:
            label = "device_microwave_w"
        else:
            label = "device_misc_event_w"

        device_series[label][start:end] = high_load_w[start:end]

    return device_series


def estimate_end_uses(frame: pd.DataFrame, event_threshold_w: float) -> pd.DataFrame:
    """Estimate end-use draws and add event-aware columns.

    :param frame: Minute-level house/weather frame.
    :param event_threshold_w: Event detection threshold in W.
    :return: Frame with estimated end-use columns.
    """
    # pylint: disable=too-many-locals
    consumption_wh = frame["Consumption(Wh)"].to_numpy(dtype=np.float32)
    consumption_w = consumption_wh * 60.0
    event_mask = detect_events(consumption_w, threshold_w=event_threshold_w)

    baseline_w = estimate_baseline(pd.Series(consumption_w)).to_numpy(dtype=np.float32)
    baseline_w = np.minimum(baseline_w, consumption_w)
    residual_w = np.clip(consumption_w - baseline_w, a_min=0.0, a_max=None)

    hvac_heating_w, hvac_cooling_w = fit_hvac_components(
        frame=frame,
        residual_w=residual_w,
        event_mask=event_mask,
    )
    hvac_total_w = hvac_heating_w + hvac_cooling_w
    non_hvac_w = np.clip(residual_w - hvac_total_w, a_min=0.0, a_max=None)

    night = (frame["date"].dt.hour >= 1) & (frame["date"].dt.hour <= 5)
    fridge_level = float(np.quantile(non_hvac_w[night.to_numpy()], 0.5))
    fridge_w = np.minimum(non_hvac_w, fridge_level).astype(np.float32)
    high_load_w = np.clip(non_hvac_w - fridge_w, a_min=0.0, a_max=None)

    segment_threshold_w = float(max(400.0, np.quantile(high_load_w, 0.75)))
    segments = extract_high_load_segments(high_load_w, threshold_w=segment_threshold_w)
    device_series = classify_device_segments(high_load_w, segments)

    output = frame.copy()
    output["consumption_wh"] = consumption_wh
    output["consumption_w"] = consumption_w
    output["event_flag"] = event_mask.astype(np.int8)
    output["device_baseline_w"] = baseline_w
    output["device_hvac_heating_w"] = hvac_heating_w
    output["device_hvac_cooling_w"] = hvac_cooling_w
    output["device_fridge_w"] = fridge_w
    for name, values in device_series.items():
        output[name] = values

    output["water_heating_w"] = output["device_water_heater_w"]
    output["cooking_w"] = (
        output["device_cooker_w"] + output["device_kettle_w"] + output["device_microwave_w"]
    )
    output["laundry_w"] = (
        output["device_washing_machine_w"]
        + output["device_dishwasher_w"]
        + output["device_tumble_dryer_w"]
    )
    output["other_plug_w"] = output["device_misc_event_w"]

    assigned_total = (
        output["device_baseline_w"].to_numpy(dtype=np.float32)
        + output["device_hvac_heating_w"].to_numpy(dtype=np.float32)
        + output["device_hvac_cooling_w"].to_numpy(dtype=np.float32)
        + output["device_fridge_w"].to_numpy(dtype=np.float32)
        + output["device_water_heater_w"].to_numpy(dtype=np.float32)
        + output["device_cooker_w"].to_numpy(dtype=np.float32)
        + output["device_kettle_w"].to_numpy(dtype=np.float32)
        + output["device_microwave_w"].to_numpy(dtype=np.float32)
        + output["device_washing_machine_w"].to_numpy(dtype=np.float32)
        + output["device_dishwasher_w"].to_numpy(dtype=np.float32)
        + output["device_tumble_dryer_w"].to_numpy(dtype=np.float32)
        + output["device_misc_event_w"].to_numpy(dtype=np.float32)
    )
    output["device_unknown_w"] = np.clip(consumption_w - assigned_total, a_min=0.0, a_max=None)
    output["unassigned_w"] = output["device_unknown_w"]

    w_columns = [
        "consumption_w",
        "device_baseline_w",
        "device_hvac_heating_w",
        "device_hvac_cooling_w",
        "device_fridge_w",
        "device_water_heater_w",
        "device_cooker_w",
        "device_kettle_w",
        "device_microwave_w",
        "device_washing_machine_w",
        "device_dishwasher_w",
        "device_tumble_dryer_w",
        "device_misc_event_w",
        "device_unknown_w",
        "water_heating_w",
        "cooking_w",
        "laundry_w",
        "other_plug_w",
        "unassigned_w",
    ]
    for column in w_columns:
        if not column.endswith("_w"):
            continue
        base_name = column[:-2]
        output[f"{base_name}_kw"] = output[column] / 1000.0
        output[f"{base_name}_wh"] = output[column] / 60.0
        output[f"{base_name}_kwh"] = output[column] / 60000.0
    return output


def plot_week_stack(end_use_frame: pd.DataFrame, figure_path: Path) -> None:
    """Plot one week of stacked end-use estimates with total consumption.

    :param end_use_frame: Estimated end-use frame.
    :param figure_path: Output image path.
    """
    first_week = end_use_frame.iloc[: 7 * 24 * 60].copy()
    time = first_week["date"]
    plt.figure(figsize=(18, 8))
    plt.plot(time, first_week["consumption_kw"], color="black", linewidth=1.0, label="Total")
    plt.stackplot(
        time,
        first_week["device_baseline_kw"],
        first_week["device_hvac_heating_kw"],
        first_week["device_hvac_cooling_kw"],
        first_week["device_fridge_kw"],
        first_week["water_heating_kw"],
        first_week["cooking_kw"],
        first_week["laundry_kw"],
        first_week["other_plug_kw"],
        labels=[
            "Baseline",
            "HVAC Heating",
            "HVAC Cooling",
            "Fridge",
            "Water Heating",
            "Cooking",
            "Laundry",
            "Other Plug",
        ],
        alpha=0.6,
    )
    plt.legend(loc="upper right", ncol=3)
    plt.title("Estimated End Uses - First Week")
    plt.ylabel("Power (kW)")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()


def plot_event_timing(end_use_frame: pd.DataFrame, figure_path: Path) -> None:
    """Plot event timing distribution across hour-of-day.

    :param end_use_frame: Estimated end-use frame.
    :param figure_path: Output image path.
    """
    events = end_use_frame[end_use_frame["event_flag"] == 1].copy()
    events["hour"] = events["date"].dt.hour
    counts = events.groupby("hour").size().reindex(range(24), fill_value=0)

    plt.figure(figsize=(14, 5))
    plt.bar(counts.index, counts.values, color="#1f77b4")
    plt.xticks(range(24))
    plt.title("Power Event Counts by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Event Count")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()


def plot_monthly_energy(end_use_frame: pd.DataFrame, figure_path: Path) -> None:
    """Plot monthly energy share for key end-use categories.

    :param end_use_frame: Estimated end-use frame.
    :param figure_path: Output image path.
    """
    monthly = end_use_frame.copy()
    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
    columns = [
        "device_hvac_heating_kwh",
        "device_hvac_cooling_kwh",
        "water_heating_kwh",
        "cooking_kwh",
        "laundry_kwh",
        "device_fridge_kwh",
        "other_plug_kwh",
        "device_baseline_kwh",
    ]
    by_month = monthly.groupby("month")[columns].sum()

    plt.figure(figsize=(18, 8))
    by_month.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="tab20")
    plt.title("Monthly Estimated End-Use Energy")
    plt.ylabel("Energy (kWh)")
    plt.xlabel("Month")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()


def discover_houses(data_dir: Path) -> list[str]:
    """Discover available house ids from ``H*_Wh.csv`` files.

    :param data_dir: Base data directory.
    :return: Sorted house id list.
    """
    houses = []
    for path in sorted(data_dir.glob("H*_Wh.csv")):
        house_id = path.stem.replace("_Wh", "")
        if house_id.startswith("H"):
            houses.append(house_id)
    return houses


def process_house(
    house: str,
    data_dir: Path,
    output_dir: Path,
    figure_dir: Path,
    event_threshold_w: float,
) -> None:
    """Process one house and emit end-use artifacts.

    :param house: House id, for example H1.
    :param data_dir: Base data directory.
    :param output_dir: Output directory for CSV artifacts.
    :param figure_dir: Output directory for figures.
    :param event_threshold_w: Event detection threshold in W.
    :return: None.
    """
    frame = read_house_and_weather(data_dir=data_dir, house=house)
    end_use = estimate_end_uses(frame=frame, event_threshold_w=event_threshold_w)

    csv_path = output_dir / f"{house}_end_use_estimates.csv"
    end_use.to_csv(csv_path, index=False)
    summary_columns = [
        "device_baseline_kwh",
        "device_hvac_heating_kwh",
        "device_hvac_cooling_kwh",
        "device_fridge_kwh",
        "device_water_heater_kwh",
        "device_cooker_kwh",
        "device_kettle_kwh",
        "device_microwave_kwh",
        "device_washing_machine_kwh",
        "device_dishwasher_kwh",
        "device_tumble_dryer_kwh",
        "device_misc_event_kwh",
        "device_unknown_kwh",
    ]
    summary = end_use[summary_columns].sum().rename("estimated_kwh").to_frame()
    summary_path = output_dir / f"{house}_device_consumption_summary.csv"
    summary.to_csv(summary_path)

    power_columns = [column.replace("_kwh", "_kw") for column in summary_columns]
    power_summary = pd.DataFrame(
        {
            "mean_kw": end_use[power_columns].mean(),
            "p95_kw": end_use[power_columns].quantile(0.95),
            "max_kw": end_use[power_columns].max(),
            "event_mean_kw": end_use.loc[end_use["event_flag"] == 1, power_columns].mean(),
        }
    )
    power_summary_path = output_dir / f"{house}_device_power_summary.csv"
    power_summary.to_csv(power_summary_path)

    plot_week_stack(
        end_use_frame=end_use,
        figure_path=figure_dir / f"{house}_week_stack.png",
    )
    plot_event_timing(
        end_use_frame=end_use,
        figure_path=figure_dir / f"{house}_event_timing.png",
    )
    plot_monthly_energy(
        end_use_frame=end_use,
        figure_path=figure_dir / f"{house}_monthly_energy.png",
    )


def main() -> None:
    """Estimate end uses for one house or all houses and emit artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    available_houses = discover_houses(args.data_dir)
    if not available_houses:
        raise ValueError(f"No house files matching H*_Wh.csv found in {args.data_dir}.")

    selected_houses = available_houses
    if args.house.lower() != "all":
        selected_houses = [args.house]
        missing = [house for house in selected_houses if house not in available_houses]
        if missing:
            raise ValueError(
                f"Requested houses not found in {args.data_dir}: {', '.join(missing)}."
            )

    failures: list[tuple[str, str]] = []
    for house in tqdm(selected_houses, desc="Processing houses", unit="house"):
        try:
            process_house(
                house=house,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                figure_dir=args.figure_dir,
                event_threshold_w=args.event_threshold_w,
            )
        except Exception as exc:  # pylint: disable=broad-except
            failures.append((house, str(exc)))

    success_count = len(selected_houses) - len(failures)
    print(
        f"Completed {success_count}/{len(selected_houses)} houses. "
        f"Outputs in {args.output_dir} and {args.figure_dir}."
    )
    if failures:
        for house, error in failures:
            print(f"[FAILED] {house}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
