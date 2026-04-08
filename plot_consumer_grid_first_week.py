#!/usr/bin/env python3
"""Plot first-week grid-demand and consumer-action views from simulation output."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "consumer_grid"
INPUT_CSV_GZ = ROOT_ARTIFACT_DIR / "consumer_grid_joint_2020.csv.gz"
OUTPUT_GRID_DEMAND_PNG = ROOT_ARTIFACT_DIR / "consumer_grid_joint_2020_first_week_grid_demand.png"
OUTPUT_CONSUMER_ACTION_PNG = (
    ROOT_ARTIFACT_DIR / "consumer_grid_joint_2020_first_week_consumer_action.png"
)

WEEK_ROWS = 7 * 24 * 60
FIGSIZE = (18, 6)


def load_first_week(path: Path, rows: int) -> pd.DataFrame:
    """Load the first-week slice from the simulation CSV.

    :param path: Input simulation CSV path.
    :param rows: Number of rows to load.
    :return: First-week dataframe with parsed UTC timestamps.
    """
    frame = pd.read_csv(path, nrows=rows, low_memory=False)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp_utc"]).copy()
    return frame


def add_plot_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute derived line-series columns needed for plotting.

    :param frame: Input first-week frame.
    :return: Frame with derived columns.
    """
    output = frame.copy()
    output["grid_demand_difference_mw"] = (
        pd.to_numeric(output["grid_demand_tsd_original_mw"], errors="coerce").fillna(0.0)
        - pd.to_numeric(output["grid_demand_tsd_adjusted_mw"], errors="coerce").fillna(0.0)
    )
    output["consumer_reduction_mw"] = (
        pd.to_numeric(output["consumer_achieved_reduction_wh_scaled"], errors="coerce").fillna(0.0)
        * 60.0
        / 1_000_000.0
    )
    output["consumer_rebound_mw"] = (
        pd.to_numeric(output["consumer_reduction_rebound_wh_scaled"], errors="coerce").fillna(0.0)
        * 60.0
        / 1_000_000.0
    )
    output["generation_total_reduced_mw"] = pd.to_numeric(
        output["generation_total_reduced_mw"],
        errors="coerce",
    ).fillna(0.0)
    return output


def apply_time_axis(ax) -> None:
    """Apply common first-week datetime axis formatting.

    :param ax: Matplotlib axis.
    :return: None.
    """
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")


def plot_grid_demand(frame: pd.DataFrame, output_path: Path) -> None:
    """Plot first-week grid demand lines.

    :param frame: First-week frame with derived plotting columns.
    :param output_path: Destination PNG path.
    :return: None.
    """
    timestamps = frame["timestamp_utc"]
    fig, axis = plt.subplots(1, 1, figsize=FIGSIZE, constrained_layout=True)
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
    axis.set_title("First Week: Grid Demand")
    left_handles, left_labels = axis.get_legend_handles_labels()
    right_handles, right_labels = axis_diff.get_legend_handles_labels()
    axis.legend(left_handles + right_handles, left_labels + right_labels, loc="upper right", ncol=2)
    axis.grid(alpha=0.25)
    apply_time_axis(axis)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350)
    plt.close(fig)


def plot_consumer_action(frame: pd.DataFrame, output_path: Path) -> None:
    """Plot first-week consumer action lines.

    :param frame: First-week frame with derived plotting columns.
    :param output_path: Destination PNG path.
    :return: None.
    """
    timestamps = frame["timestamp_utc"]
    fig, axis = plt.subplots(1, 1, figsize=FIGSIZE, constrained_layout=True)
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
    axis.set_title("First Week: Consumer Action")
    axis.legend(loc="upper right", ncol=2)
    axis.grid(alpha=0.25)
    apply_time_axis(axis)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350)
    plt.close(fig)


def main() -> None:
    """Load first-week simulation data and write separate demand/action plots."""
    first_week = load_first_week(INPUT_CSV_GZ, rows=WEEK_ROWS)
    first_week = add_plot_columns(first_week)
    plot_grid_demand(first_week, OUTPUT_GRID_DEMAND_PNG)
    plot_consumer_action(first_week, OUTPUT_CONSUMER_ACTION_PNG)
    print(f"Wrote first-week grid-demand plot to {OUTPUT_GRID_DEMAND_PNG}")
    print(f"Wrote first-week consumer-action plot to {OUTPUT_CONSUMER_ACTION_PNG}")


if __name__ == "__main__":
    main()
