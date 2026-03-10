#!/usr/bin/env python3
"""Train and evaluate a shared XGBoost model for energy targets."""

from __future__ import annotations

import json
import pickle

from sklearn.multioutput import MultiOutputRegressor

from storenet_ml.config import ARTIFACT_DIR, MODEL_DIR
from storenet_ml.pipelines import build_tabular_splits
from storenet_ml.training import compute_metrics

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is required for this script. Install it with `pip install xgboost`."
    ) from exc


def maybe_to_gpu_arrays(x_train, x_val, x_test, device: str):
    """Move feature matrices to CuPy arrays when GPU XGBoost is requested.

    :param x_train: Training feature matrix.
    :param x_val: Validation feature matrix.
    :param x_test: Test feature matrix.
    :param device: XGBoost device string.
    :return: Tuple ``(x_train, x_val, x_test, cp_module_or_none)``.
    :raises ImportError: If CuPy is required but unavailable.
    """
    # pylint: disable=import-outside-toplevel
    if not str(device).startswith("cuda"):
        return x_train, x_val, x_test, None

    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "GPU XGBoost prediction requires CuPy for GPU-backed arrays. "
            "Install a matching build (for example `pip install cupy-cuda12x`)."
        ) from exc

    return cp.asarray(x_train), cp.asarray(x_val), cp.asarray(x_test), cp


# Training configuration
SEQ_LEN = 60
HORIZON = 1
STRIDE = 15
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
MAX_INTERP_GAP = 5
SEED = 42

# XGBoost configuration
N_ESTIMATORS = 600
LEARNING_RATE = 0.05
MAX_DEPTH = 8
MIN_CHILD_WEIGHT = 5
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
REG_ALPHA = 0.0
REG_LAMBDA = 1.0
TREE_METHOD = "hist"
DEVICE = "cuda"


def main() -> None:
    """Train and evaluate the shared XGBoost energy model."""
    # pylint: disable=too-many-locals
    x_train, y_train, x_val, y_val, x_test, y_test = build_tabular_splits(
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        stride=STRIDE,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        max_interp_gap=MAX_INTERP_GAP,
    )

    print(
        "Tabular split sizes:",
        {
            "train": len(x_train),
            "val": len(x_val),
            "test": len(x_test),
        },
    )

    x_train_device, x_val_device, x_test_device, cp = maybe_to_gpu_arrays(
        x_train,
        x_val,
        x_test,
        DEVICE,
    )

    base_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_child_weight=MIN_CHILD_WEIGHT,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        tree_method=TREE_METHOD,
        device=DEVICE,
        random_state=SEED,
        n_jobs=0,
    )
    model = MultiOutputRegressor(base_model)
    model.fit(x_train_device, y_train)

    val_predictions = model.predict(x_val_device)
    if cp is not None:
        val_predictions = cp.asnumpy(val_predictions)
    val_metrics = compute_metrics(val_predictions, y_val)
    print("Validation metrics:")
    print(json.dumps(val_metrics, indent=2))

    test_predictions = model.predict(x_test_device)
    if cp is not None:
        test_predictions = cp.asnumpy(test_predictions)
    test_metrics = compute_metrics(test_predictions, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "shared_energy_xgboost.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "config": {
                    "seq_len": SEQ_LEN,
                    "horizon": HORIZON,
                    "stride": STRIDE,
                    "train_frac": TRAIN_FRAC,
                    "val_frac": VAL_FRAC,
                    "max_interp_gap": MAX_INTERP_GAP,
                    "seed": SEED,
                    "n_estimators": N_ESTIMATORS,
                    "learning_rate": LEARNING_RATE,
                    "max_depth": MAX_DEPTH,
                    "min_child_weight": MIN_CHILD_WEIGHT,
                    "subsample": SUBSAMPLE,
                    "colsample_bytree": COLSAMPLE_BYTREE,
                    "reg_alpha": REG_ALPHA,
                    "reg_lambda": REG_LAMBDA,
                    "tree_method": TREE_METHOD,
                    "device": DEVICE,
                },
            },
            handle,
        )

    metrics_path = ARTIFACT_DIR / "shared_energy_xgboost_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("Saved model to:", model_path)
    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    main()
