#!/usr/bin/env python3
"""Export and statically quantize the consumer transformer model to int8 ONNX."""
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_exploration.storenet_ml.config import MODEL_DIR
from dataset_exploration.storenet_ml.models import SharedEnergyTransformer
from dataset_exploration.infer_energy_transformer import (
    load_weather_frame,
    preprocess_house_for_inference,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    :return: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Export shared consumer transformer to ONNX and statically quantize to int8."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODEL_DIR / "shared_energy_transformer.pt",
        help="Input PyTorch checkpoint path.",
    )
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=MODEL_DIR / "shared_energy_transformer.onnx",
        help="Output path for exported float ONNX model.",
    )
    parser.add_argument(
        "--int8-output",
        type=Path,
        default=MODEL_DIR / "shared_energy_transformer_int8.onnx",
        help="Output path for quantized int8 ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="ONNX opset version used for export.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset_exploration/ireland_data"),
        help="Directory containing source house CSV files for calibration data.",
    )
    parser.add_argument(
        "--weather-file",
        type=Path,
        default=Path("dataset_exploration/ireland_data/weather.csv"),
        help="Weather CSV path used by preprocessing.",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="H*_Wh.csv",
        help="Glob pattern for selecting house files for calibration.",
    )
    parser.add_argument(
        "--calibration-files",
        type=int,
        default=8,
        help="Maximum number of house files to use for calibration windows.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=8192,
        help="Total number of calibration windows sampled across selected files.",
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=256,
        help="Calibration batch size provided to ONNX Runtime.",
    )
    return parser.parse_args()


def build_model_for_export(checkpoint: dict) -> tuple[SharedEnergyTransformer, int, int]:
    """Build the consumer transformer model from checkpoint for ONNX export.

    :param checkpoint: Loaded checkpoint dictionary.
    :return: Tuple ``(model, seq_len, feature_dim)``.
    """
    config = checkpoint["config"]
    feature_columns = list(checkpoint["feature_columns"])
    house_order = list(checkpoint["house_order"])

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
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, int(config["seq_len"]), len(feature_columns)


def export_float_onnx(
    model: SharedEnergyTransformer,
    seq_len: int,
    feature_dim: int,
    output_path: Path,
    opset: int,
) -> None:
    """Export float ONNX model from the PyTorch transformer.

    :param model: Consumer transformer model.
    :param seq_len: Sequence length.
    :param feature_dim: Feature dimension.
    :param output_path: Destination ONNX path.
    :param opset: ONNX opset version.
    :return: None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_x = torch.zeros((1, seq_len, feature_dim), dtype=torch.float32)
    dummy_house_ids = torch.zeros((1,), dtype=torch.int64)

    torch.onnx.export(
        model,
        (dummy_x, dummy_house_ids),
        str(output_path),
        input_names=["x", "house_ids"],
        output_names=["predictions"],
        dynamic_axes={
            "x": {0: "batch"},
            "house_ids": {0: "batch"},
            "predictions": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


class _WindowCalibrationDataReader:
    """ONNX Runtime calibration reader backed by pre-batched numpy arrays.

    :param x_name: ONNX input name for feature windows.
    :param house_name: ONNX input name for house ids.
    :param batches: List of `(windows, house_ids)` batches.
    """

    def __init__(
        self,
        x_name: str,
        house_name: str,
        batches: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self._x_name = x_name
        self._house_name = house_name
        self._iterator = iter(batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return next calibration batch for ORT.

        :return: Input dictionary or ``None`` when exhausted.
        """
        try:
            windows, house_ids = next(self._iterator)
        except StopIteration:
            return None
        return {
            self._x_name: windows,
            self._house_name: house_ids,
        }


def build_calibration_batches(
    checkpoint: dict,
    input_dir: Path,
    weather_file: Path,
    file_pattern: str,
    calibration_files: int,
    calibration_samples: int,
    calibration_batch_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct calibration batches from real house windows.

    :param checkpoint: Loaded checkpoint dictionary.
    :param input_dir: House data directory.
    :param weather_file: Weather CSV path.
    :param file_pattern: File-selection glob pattern.
    :param calibration_files: Max number of house files used.
    :param calibration_samples: Total window samples target.
    :param calibration_batch_size: Batch size per calibration feed.
    :return: List of `(window_batch, house_id_batch)` tuples.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    feature_columns = list(checkpoint["feature_columns"])
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    seq_len = int(checkpoint["config"]["seq_len"])
    house_files = sorted(input_dir.glob(file_pattern))
    if calibration_files > 0:
        house_files = house_files[:calibration_files]
    if not house_files:
        raise FileNotFoundError(
            f"No calibration files found in {input_dir} with pattern {file_pattern!r}."
        )
    if calibration_samples <= 0:
        raise ValueError("--calibration-samples must be positive.")
    if calibration_batch_size <= 0:
        raise ValueError("--calibration-batch-size must be positive.")

    weather = load_weather_frame(weather_file)
    selected_windows: list[np.ndarray] = []
    selected_house_ids: list[np.ndarray] = []
    samples_per_file = max(1, calibration_samples // len(house_files))
    history_offsets = np.arange(seq_len, dtype=np.int64) - seq_len

    for house_path in house_files:
        _, processed_frame = preprocess_house_for_inference(
            energy_path=house_path,
            weather=weather,
            feature_columns=feature_columns,
        )
        feature_values = processed_frame[feature_columns].to_numpy(dtype=np.float32)
        normalized = (feature_values - feature_mean) / feature_std
        row_count = normalized.shape[0]
        if row_count == 0:
            continue

        take_count = min(samples_per_file, row_count)
        sample_indices = np.linspace(
            0,
            row_count - 1,
            num=take_count,
            dtype=np.int64,
        )
        window_indices = (sample_indices[:, None] + history_offsets[None, :]) % row_count
        windows = normalized[window_indices].astype(np.float32, copy=False)
        house_id = int(processed_frame["house_id"].iloc[0])
        house_ids = np.full((take_count,), house_id, dtype=np.int64)
        selected_windows.append(windows)
        selected_house_ids.append(house_ids)

    if not selected_windows:
        raise RuntimeError("Failed to construct any calibration windows.")

    windows_all = np.concatenate(selected_windows, axis=0)
    house_ids_all = np.concatenate(selected_house_ids, axis=0)
    if len(windows_all) > calibration_samples:
        windows_all = windows_all[:calibration_samples]
        house_ids_all = house_ids_all[:calibration_samples]

    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, len(windows_all), calibration_batch_size):
        end = min(start + calibration_batch_size, len(windows_all))
        batches.append((windows_all[start:end], house_ids_all[start:end]))
    return batches


def quantize_to_int8_static(
    onnx_path: Path,
    int8_path: Path,
    calibration_batches: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Statically quantize exported ONNX model to int8 with ONNX Runtime.

    :param onnx_path: Source float ONNX path.
    :param int8_path: Destination int8 ONNX path.
    :param calibration_batches: Calibration inputs batched for ORT reader.
    :return: None.
    """
    # pylint: disable=import-outside-toplevel
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime and onnxruntime-tools quantization support are required."
        ) from exc

    _ = CalibrationDataReader
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) < 2:
        raise ValueError(f"ONNX model expected 2 inputs, found {len(inputs)}: {onnx_path}")
    x_name = inputs[0].name
    house_name = inputs[1].name
    calibration_reader = _WindowCalibrationDataReader(
        x_name=x_name,
        house_name=house_name,
        batches=calibration_batches,
    )

    quantize_static(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["MatMul", "Gemm"],
    )


def validate_onnx_runtime(int8_path: Path, seq_len: int, feature_dim: int) -> None:
    """Run one sanity inference with ONNX Runtime.

    :param int8_path: Quantized ONNX model path.
    :param seq_len: Sequence length.
    :param feature_dim: Feature dimension.
    :return: None.
    """
    # pylint: disable=import-outside-toplevel
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnxruntime is required for ONNX validation.") from exc

    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    x = np.zeros((2, seq_len, feature_dim), dtype=np.float32)
    house_ids = np.zeros((2,), dtype=np.int64)
    outputs = session.run(["predictions"], {"x": x, "house_ids": house_ids})
    if not outputs:
        raise RuntimeError(f"ONNX runtime validation produced no outputs for {int8_path}")


def main() -> None:
    """Export and statically quantize consumer transformer model."""
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, seq_len, feature_dim = build_model_for_export(checkpoint)
    calibration_batches = build_calibration_batches(
        checkpoint=checkpoint,
        input_dir=args.input_dir,
        weather_file=args.weather_file,
        file_pattern=args.file_pattern,
        calibration_files=args.calibration_files,
        calibration_samples=args.calibration_samples,
        calibration_batch_size=args.calibration_batch_size,
    )

    export_float_onnx(
        model=model,
        seq_len=seq_len,
        feature_dim=feature_dim,
        output_path=args.onnx_output,
        opset=args.opset,
    )
    quantize_to_int8_static(
        onnx_path=args.onnx_output,
        int8_path=args.int8_output,
        calibration_batches=calibration_batches,
    )
    validate_onnx_runtime(args.int8_output, seq_len=seq_len, feature_dim=feature_dim)

    print(f"Wrote float ONNX model: {args.onnx_output}")
    print(f"Wrote int8 ONNX model: {args.int8_output}")


if __name__ == "__main__":
    main()
