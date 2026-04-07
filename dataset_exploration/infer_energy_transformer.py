#!/usr/bin/env python3
"""
Run transformer inference over house Wh datasets and emit predicted targets.

python dataset_exploration/infer_energy_transformer.py \
  --output-dir dataset_exploration/ireland_data_inferred \
  --device cuda \
  --batch-size 8192
"""
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_exploration.storenet_ml.config import (
    DATA_DIR,
    ENERGY_COLUMNS,
    MODEL_DIR,
    WEATHER_COLUMNS,
)
from dataset_exploration.storenet_ml.data_loaders import (
    add_calendar_features,
    strip_and_coerce_numeric,
)
from dataset_exploration.storenet_ml.models import SharedEnergyTransformer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Load the trained transformer checkpoint and emit inferred house CSVs "
            "with predicted production/consumption columns."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing source house CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_exploration/ireland_data_inferred"),
        help="Directory to write inferred house CSV files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODEL_DIR / "shared_energy_transformer.pt",
        help="Transformer checkpoint path.",
    )
    parser.add_argument(
        "--weather-file",
        type=Path,
        default=DATA_DIR / "weather.csv",
        help="Weather CSV file path.",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="H*_Wh.csv",
        help="Glob pattern selecting house files to run inference on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Inference batch size (windows per forward pass).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cpu, or cuda.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve runtime device from user argument.

    :param device_arg: User-specified device option.
    :return: Torch device.
    :raises ValueError: If the provided option is unsupported.
    """
    candidate = device_arg.strip().lower()
    if candidate == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if candidate in {"cpu", "cuda"}:
        if candidate == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device(candidate)
    raise ValueError(f"Unsupported --device value: {device_arg}")


def load_weather_frame(weather_file: Path) -> pd.DataFrame:
    """Load weather data and align it to 1-minute granularity.

    :param weather_file: Weather CSV path.
    :return: Indexed minute-level weather frame.
    """
    weather = pd.read_csv(
        weather_file,
        dtype="string",
        low_memory=False,
        na_values=["", " ", "  ", "   "],
    )
    weather.columns = weather.columns.str.strip()
    weather["date"] = pd.to_datetime(
        weather["date"].astype("string").str.strip(),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    weather = strip_and_coerce_numeric(weather, WEATHER_COLUMNS)
    weather = weather.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    weather = weather.set_index("date")

    full_index = pd.date_range(weather.index.min(), weather.index.max(), freq="1min")
    weather = weather.reindex(full_index)
    weather.index.name = "date"
    weather[WEATHER_COLUMNS] = weather[WEATHER_COLUMNS].interpolate(
        method="linear",
        limit_direction="both",
    )
    weather = weather.dropna(subset=WEATHER_COLUMNS)
    return weather


def circular_linear_fill(values: np.ndarray) -> np.ndarray:
    """Fill NaN values by linear interpolation on a circular timeline.

    Missing spans are filled between their nearest known points. The segment at the
    beginning of the file is connected to the final known point from the file end.

    :param values: Float vector with possible NaNs.
    :return: Filled float vector.
    """
    output = values.astype(np.float32, copy=True)
    finite_indices = np.flatnonzero(np.isfinite(output))
    if len(finite_indices) == 0:
        return np.zeros_like(output, dtype=np.float32)
    if len(finite_indices) == 1:
        output[:] = output[finite_indices[0]]
        return output

    for index, start_position in enumerate(finite_indices):
        start_idx = int(start_position)
        end_idx = int(finite_indices[(index + 1) % len(finite_indices)])
        if index == len(finite_indices) - 1:
            end_idx += len(output)

        start_value = float(output[start_idx])
        end_value = float(output[end_idx % len(output)])
        gap = end_idx - start_idx
        if gap <= 1:
            continue

        for step in range(1, gap):
            position = (start_idx + step) % len(output)
            interpolated = start_value + (end_value - start_value) * (step / gap)
            if not np.isfinite(output[position]):
                output[position] = np.float32(interpolated)
    return output


def circular_fill_frame_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply circular linear interpolation to selected frame columns.

    :param frame: Dataframe containing numeric columns.
    :param columns: Column names to fill.
    :return: Updated dataframe.
    """
    for column in columns:
        values = frame[column].to_numpy(dtype=np.float32)
        frame[column] = circular_linear_fill(values)
    return frame


def preprocess_house_for_inference(
    energy_path: Path,
    weather: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build model-ready house frame while preserving raw rows for output.

    :param energy_path: House CSV path.
    :param weather: Preprocessed weather frame.
    :param feature_columns: Model feature column names from checkpoint.
    :return: Tuple `(raw_frame, processed_frame)`.
    """
    raw_frame = pd.read_csv(
        energy_path,
        dtype="string",
        low_memory=False,
        na_values=["", " ", "  ", "   "],
    )

    processed = raw_frame.copy()
    processed.columns = processed.columns.str.strip()
    processed["date"] = pd.to_datetime(
        processed["date"].astype("string").str.strip(),
        errors="coerce",
    )
    processed = strip_and_coerce_numeric(processed, ENERGY_COLUMNS)
    processed = processed.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    processed = processed.set_index("date")

    full_index = pd.date_range(processed.index.min(), processed.index.max(), freq="1min")
    processed = processed.reindex(full_index)
    processed.index.name = "date"
    processed = circular_fill_frame_columns(processed, ENERGY_COLUMNS)

    processed = processed.join(weather, how="left")
    processed = circular_fill_frame_columns(processed, WEATHER_COLUMNS)
    processed = processed.reset_index().rename(columns={"index": "date"})
    processed = add_calendar_features(processed)

    house_name = energy_path.stem.split("_")[0]
    processed["house_name"] = house_name
    processed["house_id"] = int(house_name[1:]) - 1

    missing_features = [column for column in feature_columns if column not in processed.columns]
    if missing_features:
        raise KeyError(f"{energy_path.name} is missing feature columns: {missing_features}")

    return raw_frame, processed


def predict_all_rows(
    model: SharedEnergyTransformer,
    processed_frame: pd.DataFrame,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predict production/consumption for every row using circular history windows.

    :param model: Transformer model in eval mode.
    :param processed_frame: Model-ready frame.
    :param feature_columns: Ordered model feature columns.
    :param feature_mean: Feature normalization mean vector.
    :param feature_std: Feature normalization std vector.
    :param target_mean: Target denormalization mean vector.
    :param target_std: Target denormalization std vector.
    :param seq_len: Sequence length expected by checkpoint.
    :param batch_size: Inference batch size.
    :param device: Inference device.
    :return: Array of denormalized predictions with shape ``(n_rows, 2)``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    feature_values = processed_frame[feature_columns].to_numpy(dtype=np.float32)
    normalized = (feature_values - feature_mean) / feature_std
    row_count = normalized.shape[0]
    history_offsets = np.arange(seq_len, dtype=np.int64) - seq_len
    house_ids = np.full((batch_size,), int(processed_frame["house_id"].iloc[0]), dtype=np.int64)
    predictions = np.empty((row_count, len(target_mean)), dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for batch_start in tqdm(
            range(0, row_count, batch_size),
            leave=False,
            desc=f"{processed_frame['house_name'].iloc[0]} inference",
        ):
            batch_end = min(batch_start + batch_size, row_count)
            indices = np.arange(batch_start, batch_end, dtype=np.int64)
            window_indices = (indices[:, None] + history_offsets[None, :]) % row_count
            windows = normalized[window_indices]

            x = torch.from_numpy(windows).to(device, non_blocking=True)
            hid = torch.from_numpy(house_ids[: len(indices)]).to(device, non_blocking=True)
            predicted_norm = model(x, hid).cpu().numpy()
            predictions[batch_start:batch_end] = predicted_norm * target_std + target_mean

    return predictions


def append_predictions_to_raw(
    raw_frame: pd.DataFrame,
    processed_frame: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Attach predicted columns to the original raw frame while preserving original columns.

    :param raw_frame: Original raw house dataframe.
    :param processed_frame: Processed minute-level frame used for inference.
    :param predictions: Prediction matrix shaped `(n_rows, 2)`.
    :return: Output dataframe containing all original columns plus prediction columns.
    """
    production_predictions = predictions[:, 0].astype(np.float32, copy=False)
    house_production = processed_frame["Production(Wh)"].to_numpy(dtype=np.float32)
    has_any_solar = bool(np.any(np.isfinite(house_production) & (house_production > 0.0)))
    if not has_any_solar:
        production_predictions[:] = 0.0
    else:
        no_solar_mask = processed_frame["soltot"].to_numpy(dtype=np.float32) == 0.0
        production_predictions[no_solar_mask] = 0.0

    prediction_frame = pd.DataFrame(
        {
            "date": processed_frame["date"],
            "predicted_production_wh": production_predictions,
            "predicted_consumption_wh": predictions[:, 1],
        }
    )

    output = raw_frame.copy()
    parsed_dates = pd.to_datetime(output["date"].astype("string").str.strip(), errors="coerce")
    lookup = prediction_frame.set_index("date")
    output["predicted_production_wh"] = (
        lookup.reindex(parsed_dates)["predicted_production_wh"].to_numpy()
    )
    output["predicted_consumption_wh"] = (
        lookup.reindex(parsed_dates)["predicted_consumption_wh"].to_numpy()
    )
    return output


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> SharedEnergyTransformer:
    """Instantiate and load transformer weights from checkpoint payload.

    :param checkpoint: Transformer checkpoint dictionary.
    :param device: Runtime device.
    :return: Loaded transformer model.
    """
    config = checkpoint["config"]
    feature_columns = checkpoint["feature_columns"]
    house_order = checkpoint["house_order"]
    model = SharedEnergyTransformer(
        feature_dim=len(feature_columns),
        num_houses=len(house_order),
        max_seq_len=int(config["seq_len"]),
        d_model=int(config["d_model"]),
        house_embedding_dim=int(config["house_embedding_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        feedforward_dim=int(config["feedforward_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = torch.compile(model)
    return model


def main() -> None:
    """Run transformer inference and emit inferred files."""
    # pylint: disable=too-many-locals
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, device=device)

    feature_columns = list(checkpoint["feature_columns"])
    target_columns = list(checkpoint["target_columns"])
    if target_columns != ["Production(Wh)", "Consumption(Wh)"]:
        raise ValueError(f"Unexpected checkpoint targets: {target_columns}")

    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    seq_len = int(checkpoint["config"]["seq_len"])

    house_files = sorted(args.input_dir.glob(args.file_pattern))
    if not house_files:
        raise FileNotFoundError(
            f"No files matching {args.file_pattern} were found in {args.input_dir}"
        )

    weather = load_weather_frame(args.weather_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for house_path in tqdm(house_files, desc="Inferring house files", unit="file"):
        raw_frame, processed_frame = preprocess_house_for_inference(
            energy_path=house_path,
            weather=weather,
            feature_columns=feature_columns,
        )
        predictions = predict_all_rows(
            model=model,
            processed_frame=processed_frame,
            feature_columns=feature_columns,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            seq_len=seq_len,
            batch_size=args.batch_size,
            device=device,
        )
        output_frame = append_predictions_to_raw(
            raw_frame=raw_frame,
            processed_frame=processed_frame,
            predictions=predictions,
        )
        output_path = args.output_dir / house_path.name
        output_frame.to_csv(output_path, index=False)

    print(f"Inference complete. Wrote {len(house_files)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
