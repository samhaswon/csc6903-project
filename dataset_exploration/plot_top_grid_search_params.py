#!/usr/bin/env python3
"""Summarize and visualize top grid-search hyperparameters by model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from storenet_ml.config import ARTIFACT_DIR, PROJECT_ROOT

DEFAULT_RESULTS_PATH = ARTIFACT_DIR / "grid_search_results.jsonl"
DEFAULT_SUMMARY_PATH = ARTIFACT_DIR / "top3_params_by_model.json"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures"
TOP_K = 10


@dataclass(frozen=True)
class AxisSpec:
    """Encodes one selected plotting axis.

    :param key: Parameter key used for the axis.
    :param mapping: Optional category-to-float mapping for non-numeric values.
    """

    key: str
    mapping: dict[Any, float] | None


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    :return: Parsed namespace with input/output paths.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Find top-3 parameter sets per model from grid-search results and create a 3D plot."
        )
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to grid_search_results.jsonl",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to write top-3 parameter summary JSON",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory where per-model 3D figures are written",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines records.

    :param path: JSONL source path.
    :return: Parsed records.
    :raises FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            records.append(json.loads(payload))
    return records


def latest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the latest record for each trial id.

    :param records: Chronological result records.
    :return: Latest record per trial id.
    """
    latest_by_trial: dict[str, dict[str, Any]] = {}
    for record in records:
        latest_by_trial[record["trial_id"]] = record
    return list(latest_by_trial.values())


def merge_params(record: dict[str, Any]) -> dict[str, Any]:
    """Merge and namespace trial parameters.

    :param record: One result record.
    :return: Flattened parameter mapping with namespaced keys.
    """
    merged: dict[str, Any] = {}
    for key, value in record.get("data_params", {}).items():
        merged[f"data.{key}"] = value
    for key, value in record.get("model_params", {}).items():
        merged[f"model.{key}"] = value
    return merged


def is_numeric_value(value: Any) -> bool:
    """Check whether a value is numeric or boolean.

    :param value: Candidate value.
    :return: ``True`` if value can be plotted directly as a float.
    """
    return isinstance(value, (int, float, bool)) and not isinstance(value, complex)


def key_priority(
    key: str,
    top_unique: int,
    model_unique: int,
) -> tuple[int, int, int, int, int, str]:
    """Build a sort key for axis relevance.

    :param key: Namespaced parameter key.
    :param top_unique: Unique values seen in top-k records.
    :param model_unique: Unique values seen in all model records.
    :return: Tuple where lower values indicate higher priority.
    """
    return (
        -int(top_unique > 1),
        -int(model_unique > 1),
        -int(key.startswith("model.")),
        -top_unique,
        -model_unique,
        key,
    )


def classify_key(
    key: str,
    model_records: list[dict[str, Any]],
    top_records: list[dict[str, Any]],
) -> tuple[bool, tuple[int, int, int, int, int, str]] | None:
    """Classify one key as numeric/categorical and compute its priority.

    :param key: Namespaced parameter key.
    :param model_records: All successful records for the model.
    :param top_records: Top-k records for this model.
    :return: ``(is_numeric, priority)`` or ``None`` when invalid for all records.
    """
    values: list[Any] = []
    numeric_for_all = True
    for record in model_records:
        value = merge_params(record).get(key)
        if value is None:
            return None
        values.append(value)
        if not is_numeric_value(value):
            numeric_for_all = False

    top_unique = len({str(merge_params(record)[key]) for record in top_records})
    model_unique = len({str(value) for value in values})
    return numeric_for_all, key_priority(key, top_unique, model_unique)


def select_axis_specs(
    model_records: list[dict[str, Any]],
    top_records: list[dict[str, Any]],
) -> list[AxisSpec]:
    """Choose three parameter axes for one model.

    :param model_records: All successful records for a model.
    :param top_records: Top-k records for the model.
    :return: Exactly three axis specifications.
    :raises ValueError: If fewer than three common parameters are available.
    """
    common_keys = set(merge_params(top_records[0]).keys())
    for record in top_records[1:]:
        common_keys.intersection_update(merge_params(record).keys())

    if len(common_keys) < 3:
        raise ValueError("Not enough shared parameters among top records to build 3D axes.")

    numeric_keys: list[tuple[tuple[int, int, int, int, int, str], str]] = []
    categorical_keys: list[tuple[tuple[int, int, int, int, int, str], str]] = []

    for key in sorted(common_keys):
        classified = classify_key(key, model_records, top_records)
        if classified is None:
            continue
        numeric_for_all, priority = classified
        if numeric_for_all:
            numeric_keys.append((priority, key))
        else:
            categorical_keys.append((priority, key))

    numeric_keys.sort(key=lambda item: item[0])
    categorical_keys.sort(key=lambda item: item[0])
    axis_specs: list[AxisSpec] = [
        AxisSpec(key=key, mapping=None)
        for _, key in numeric_keys[:3]
    ]

    for _, key in categorical_keys:
        if len(axis_specs) >= 3:
            break
        categories = sorted({str(merge_params(record)[key]) for record in model_records})
        mapping = {category: float(index) for index, category in enumerate(categories)}
        axis_specs.append(AxisSpec(key=key, mapping=mapping))

    if len(axis_specs) < 3:
        raise ValueError("Could not select three parameters suitable for 3D plotting.")

    return axis_specs[:3]


def axis_value(record: dict[str, Any], spec: AxisSpec) -> float:
    """Compute numeric axis value for one trial record.

    :param record: Trial result record.
    :param spec: Axis spec with optional categorical mapping.
    :return: Float value to plot.
    """
    value = merge_params(record)[spec.key]
    if spec.mapping is None:
        return float(value)
    return spec.mapping[str(value)]


def top_by_model(records: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    """Group and rank trials by model and score.

    :param records: Successful latest trial records.
    :param top_k: Number of top records to keep per model.
    :return: Mapping from model name to sorted top-k records.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["model"], []).append(record)

    ranked: dict[str, list[dict[str, Any]]] = {}
    for model_name, candidates in grouped.items():
        ranked[model_name] = sorted(candidates, key=lambda item: float(item["score"]))[:top_k]
    return ranked


def write_top_summary(
    output_path: Path,
    top_records_by_model: dict[str, list[dict[str, Any]]],
) -> None:
    """Write top-k parameter sets per model as JSON.

    :param output_path: Target JSON path.
    :param top_records_by_model: Top trials grouped by model.
    """
    payload: dict[str, Any] = {"top_k": TOP_K, "models": {}}
    for model_name, records in sorted(top_records_by_model.items()):
        payload["models"][model_name] = [
            {
                "rank": index,
                "score": float(record["score"]),
                "trial_id": record["trial_id"],
                "data_params": record.get("data_params", {}),
                "model_params": record.get("model_params", {}),
            }
            for index, record in enumerate(records, start=1)
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def draw_model_subplot(
    subplot,
    figure,
    model_name: str,
    ranked_records: list[dict[str, Any]],
    axis_specs: list[AxisSpec],
) -> None:
    """Draw one model's top-k trials in a 3D subplot.

    :param figure: Parent Matplotlib figure.
    :param subplot: Matplotlib axis object for plotting.
    :param model_name: Model family name.
    :param ranked_records: Ranked trial records for the model.
    :param axis_specs: Selected axis specifications.
    """
    # pylint: disable=too-many-locals
    axis_points = [
        (
            axis_value(record, axis_specs[0]),
            axis_value(record, axis_specs[1]),
            axis_value(record, axis_specs[2]),
            float(record["score"]),
        )
        for record in ranked_records
    ]
    x_vals = [point[0] for point in axis_points]
    y_vals = [point[1] for point in axis_points]
    z_vals = [point[2] for point in axis_points]
    scores = [float(record["score"]) for record in ranked_records]

    scatter = subplot.scatter(
        x_vals,
        y_vals,
        z_vals,
        c=scores,
        cmap="viridis_r",
        s=80,
        depthshade=True,
    )
    subplot.set_title(f"{model_name}: top 3 params, top {TOP_K} runs")
    subplot.set_xlabel(axis_specs[0].key)
    subplot.set_ylabel(axis_specs[1].key)
    subplot.set_zlabel(axis_specs[2].key)

    for rank, (x_val, y_val, z_val, score) in enumerate(axis_points, start=1):
        subplot.text(x_val, y_val, z_val, f"#{rank}\n{score:.4f}", fontsize=8)

    colorbar = figure.colorbar(scatter, ax=subplot, fraction=0.045, pad=0.08)
    colorbar.set_label("Validation Joint MAE (lower is better)")


def plot_top_3d_by_model(
    figure_dir: Path,
    successful_records: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, Path]]:
    """Create one per-model 3D scatter plot for top-k trials.

    :param figure_dir: Destination directory for PNG files.
    :param successful_records: Successful latest trial records.
    :return: Axis labels by model and output PNG path by model.
    """
    top_records_by_model = top_by_model(successful_records, TOP_K)
    model_names = sorted(top_records_by_model.keys())
    axis_labels_by_model: dict[str, list[str]] = {}
    figure_paths_by_model: dict[str, Path] = {}
    figure_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        model_records = [record for record in successful_records if record["model"] == model_name]
        ranked_records = top_records_by_model[model_name]
        axis_specs = select_axis_specs(model_records, ranked_records)
        axis_labels_by_model[model_name] = [spec.key for spec in axis_specs]
        output_path = figure_dir / f"top_k_params_3d_{model_name}.png"
        figure_paths_by_model[model_name] = output_path

        figure = plt.figure(figsize=(7, 6), constrained_layout=True)
        subplot = figure.add_subplot(1, 1, 1, projection="3d")
        draw_model_subplot(
            subplot=subplot,
            figure=figure,
            model_name=model_name,
            ranked_records=ranked_records,
            axis_specs=axis_specs,
        )
        figure.savefig(output_path, dpi=220)
        plt.close(figure)
    return axis_labels_by_model, figure_paths_by_model


def main() -> None:
    """Run top-k extraction and 3D plotting pipeline."""
    args = parse_args()

    records = load_jsonl(args.results_path)
    latest = latest_records(records)
    successful = [record for record in latest if record.get("status") == "ok"]

    if not successful:
        raise RuntimeError("No successful trials were found in the provided results file.")

    top_records_by_model = top_by_model(successful, TOP_K)
    write_top_summary(args.summary_path, top_records_by_model)
    axis_labels_by_model, figure_paths_by_model = plot_top_3d_by_model(
        args.figure_dir,
        successful,
    )

    print(f"Summary written to: {args.summary_path}")
    for model_name in sorted(axis_labels_by_model):
        labels = ", ".join(axis_labels_by_model[model_name])
        print(f"{model_name} figure: {figure_paths_by_model[model_name]}")
        print(f"{model_name} axes: {labels}")


if __name__ == "__main__":
    main()
