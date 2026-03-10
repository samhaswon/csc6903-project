#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle

from sklearn.multioutput import MultiOutputRegressor

from storenet_ml.config import ARTIFACT_DIR, MODEL_DIR
from storenet_ml.pipelines import build_tabular_splits
from storenet_ml.training import compute_metrics

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError("lightgbm is required for this script. Install it with `pip install lightgbm`.") from exc


# Training configuration
SEQ_LEN = 60
HORIZON = 1
STRIDE = 15
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
MAX_INTERP_GAP = 5
SEED = 42

# LightGBM configuration
N_ESTIMATORS = 400
LEARNING_RATE = 0.05
NUM_LEAVES = 63
MAX_DEPTH = -1
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
MIN_CHILD_SAMPLES = 50
REG_ALPHA = 0.0
REG_LAMBDA = 0.0
DEVICE_TYPE = "cpu"


def main() -> None:
    """Train and evaluate the shared LightGBM energy model."""
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

    base_model = LGBMRegressor(
        objective="regression",
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        min_child_samples=MIN_CHILD_SAMPLES,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        random_state=SEED,
        n_jobs=-1,
        device_type=DEVICE_TYPE,
        verbosity=-1,
    )
    model = MultiOutputRegressor(base_model)
    model.fit(x_train, y_train)

    val_predictions = model.predict(x_val)
    val_metrics = compute_metrics(val_predictions, y_val)
    print("Validation metrics:")
    print(json.dumps(val_metrics, indent=2))

    test_predictions = model.predict(x_test)
    test_metrics = compute_metrics(test_predictions, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "shared_energy_lightgbm.pkl"
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
                    "num_leaves": NUM_LEAVES,
                    "max_depth": MAX_DEPTH,
                    "subsample": SUBSAMPLE,
                    "colsample_bytree": COLSAMPLE_BYTREE,
                    "min_child_samples": MIN_CHILD_SAMPLES,
                    "reg_alpha": REG_ALPHA,
                    "reg_lambda": REG_LAMBDA,
                    "device_type": DEVICE_TYPE,
                },
            },
            handle,
        )

    metrics_path = ARTIFACT_DIR / "shared_energy_lightgbm_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("Saved model to:", model_path)
    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    main()
