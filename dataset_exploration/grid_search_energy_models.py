#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from sklearn.model_selection import ParameterGrid
from sklearn.multioutput import MultiOutputRegressor
from torch import nn
from tqdm.auto import tqdm

from storenet_ml.config import ARTIFACT_DIR, HOUSE_ORDER, INPUT_FEATURES
from storenet_ml.datasets import create_dataloader
from storenet_ml.models import SharedEnergyRNN, SharedEnergyTCN, SharedEnergyTransformer
from storenet_ml.pipelines import build_rnn_datasets, build_tabular_splits
from storenet_ml.training import collect_predictions, compute_metrics, set_seed, train_one_epoch

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


RESULTS_PATH = ARTIFACT_DIR / "grid_search_results.jsonl"
SUMMARY_PATH = ARTIFACT_DIR / "grid_search_summary.json"
SEED = 42
SKIP_FAILED_TRIALS = False

# Data parameter grids
NEURAL_DATA_GRID = {
    "seq_len": [48, 60],
    "horizon": [1],
    "stride": [15],
    "train_frac": [0.7],
    "val_frac": [0.15],
    "max_interp_gap": [5],
}
TABULAR_DATA_GRID = {
    "seq_len": [60],
    "horizon": [1],
    "stride": [15],
    "train_frac": [0.7],
    "val_frac": [0.15],
    "max_interp_gap": [5],
}

# Model parameter grids
RNN_GRID = {
    "batch_size": [256],
    "hidden_size": [96, 128, 160],
    "house_embedding_dim": [8, 16],
    "num_layers": [1, 2, 4, 8, 16],
    "dropout": [0.15, 0.25],
    "epochs": [20],
    "learning_rate": [5e-4, 1e-3],
    "weight_decay": [1e-5],
    "patience": [5],
}
TCN_GRID = {
    "batch_size": [256],
    "hidden_size": [64, 128, 256],
    "house_embedding_dim": [8, 16],
    "num_layers": [2, 3, 4, 8, 16],
    "kernel_size": [3],
    "dropout": [0.15, 0.25],
    "epochs": [30],
    "learning_rate": [1e-3],
    "weight_decay": [1e-5],
    "patience": [5],
}
TRANSFORMER_GRID = {
    "batch_size": [128],
    "d_model": [48, 64],
    "house_embedding_dim": [8],
    "num_layers": [1, 2, 4, 8, 16],
    "num_heads": [4],
    "feedforward_dim": [96, 128],
    "dropout": [0.1, 0.2],
    "epochs": [50],
    "learning_rate": [3e-4, 5e-4],
    "weight_decay": [1e-4, 3e-4],
    "patience": [5],
}
LIGHTGBM_GRID = {
    "n_estimators": [300, 500],
    "learning_rate": [0.05],
    "num_leaves": [31, 63],
    "max_depth": [-1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "min_child_samples": [25, 50],
    "reg_alpha": [0.0],
    "reg_lambda": [0.0],
    "device_type": ["cpu"],
}
XGBOOST_GRID = {
    "n_estimators": [400, 600],
    "learning_rate": [0.05],
    "max_depth": [6, 8],
    "min_child_weight": [3, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_alpha": [0.0],
    "reg_lambda": [1.0],
    "tree_method": ["hist"],
    "device": ["cuda"],
}


def stable_params(params: dict) -> dict:
    """Return a JSON-stable copy of a parameter dictionary.

    :param params: Arbitrary parameter mapping.
    :return: Deterministic JSON-serializable mapping.
    """
    return json.loads(json.dumps(params, sort_keys=True, default=str))


def trial_id(model_name: str, params: dict) -> str:
    """Build a stable identifier for one grid-search trial.

    :param model_name: Model family name.
    :param params: Combined data/model parameter mapping.
    :return: Stable JSON string used as trial id.
    """
    payload = {"model": model_name, "params": stable_params(params)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_previous_results() -> list[dict]:
    """Load all prior result records from the JSONL results file.

    :return: List of result dictionaries.
    """
    if not RESULTS_PATH.exists():
        return []

    results = []
    with RESULTS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def should_skip(existing: dict[str, dict], current_trial_id: str) -> bool:
    """Decide whether a trial should be skipped.

    :param existing: Latest result records keyed by trial id.
    :param current_trial_id: Trial id for the candidate run.
    :return: ``True`` when this trial should be skipped.
    """
    previous = existing.get(current_trial_id)
    if previous is None:
        return False
    if previous.get("status") == "ok":
        return True
    return SKIP_FAILED_TRIALS


def append_result(result: dict) -> None:
    """Append one result record to the JSONL results file.

    :param result: Trial result dictionary.
    """
    ARTIFACT_DIR.mkdir(exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")


def latest_results(results: list[dict]) -> list[dict]:
    """Keep only the latest record for each trial id.

    :param results: Chronological list of result records.
    :return: De-duplicated latest records keyed by trial id.
    """
    by_id = {}
    for result in results:
        by_id[result["trial_id"]] = result
    return list(by_id.values())


def write_summary(results: list[dict]) -> None:
    """Write a summary JSON file with best trial per model family.

    :param results: Result records collected so far.
    """
    current_results = latest_results(results)
    successful = [result for result in current_results if result.get("status") == "ok"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_path": str(RESULTS_PATH),
        "completed_trials": len(current_results),
        "successful_trials": len(successful),
        "failed_trials": len(current_results) - len(successful),
        "best_by_model": {},
    }

    for model_name in sorted({result["model"] for result in successful}):
        candidates = [result for result in successful if result["model"] == model_name]
        best = min(candidates, key=lambda result: result["score"])
        summary["best_by_model"][model_name] = {
            "score": best["score"],
            "trial_id": best["trial_id"],
            "data_params": best["data_params"],
            "model_params": best["model_params"],
            "val_metrics": best["val_metrics"],
            "test_metrics": best["test_metrics"],
        }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def neural_data_key(data_params: dict) -> tuple:
    """Build a hashable cache key for neural-data pipeline parameters.

    :param data_params: Data-parameter dictionary.
    :return: Tuple key suitable for dictionary caching.
    """
    return (
        data_params["seq_len"],
        data_params["horizon"],
        data_params["stride"],
        data_params["train_frac"],
        data_params["val_frac"],
        data_params["max_interp_gap"],
    )


def tabular_data_key(data_params: dict) -> tuple:
    """Build a hashable cache key for tabular-data pipeline parameters.

    :param data_params: Data-parameter dictionary.
    :return: Tuple key suitable for dictionary caching.
    """
    return neural_data_key(data_params)


def get_neural_bundle(cache: dict, data_params: dict):
    """Get or build cached neural datasets for a parameter set.

    :param cache: Cache mapping from data keys to dataset bundles.
    :param data_params: Data-parameter dictionary.
    :return: Cached tuple from ``build_rnn_datasets``.
    """
    key = neural_data_key(data_params)
    if key not in cache:
        cache[key] = build_rnn_datasets(
            seq_len=data_params["seq_len"],
            horizon=data_params["horizon"],
            stride=data_params["stride"],
            train_frac=data_params["train_frac"],
            val_frac=data_params["val_frac"],
            max_interp_gap=data_params["max_interp_gap"],
        )
    return cache[key]


def get_tabular_bundle(cache: dict, data_params: dict):
    """Get or build cached tabular arrays for a parameter set.

    :param cache: Cache mapping from data keys to tabular bundles.
    :param data_params: Data-parameter dictionary.
    :return: Cached tuple from ``build_tabular_splits``.
    """
    key = tabular_data_key(data_params)
    if key not in cache:
        cache[key] = build_tabular_splits(
            seq_len=data_params["seq_len"],
            horizon=data_params["horizon"],
            stride=data_params["stride"],
            train_frac=data_params["train_frac"],
            val_frac=data_params["val_frac"],
            max_interp_gap=data_params["max_interp_gap"],
        )
    return cache[key]


def train_neural_trial(
    model_name: str,
    model,
    datasets,
    model_params: dict,
    stats,
    device: torch.device,
) -> dict:
    """Train one neural-model trial with early stopping and test evaluation.

    :param model_name: Name used for progress labels.
    :param model: Initialized model instance.
    :param datasets: Tuple of ``(train_dataset, val_dataset, test_dataset)``.
    :param model_params: Hyperparameter dictionary for training.
    :param stats: Normalization statistics with target mean/std.
    :param device: Device used for training/evaluation.
    :return: Dictionary with ``score``, ``val_metrics``, and ``test_metrics``.
    :raises RuntimeError: If datasets are empty or no best state is found.
    """
    train_dataset, val_dataset, test_dataset = datasets

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError("At least one split has zero windows.")

    train_loader = create_dataloader(train_dataset, batch_size=model_params["batch_size"], shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=model_params["batch_size"], shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=model_params["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_params["learning_rate"],
        weight_decay=model_params["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_val_metrics = None
    patience_left = model_params["patience"]

    epoch_progress = tqdm(range(1, model_params["epochs"] + 1), desc=f"{model_name} trial", leave=False)
    for epoch in epoch_progress:
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch,
            model_params["epochs"],
        )
        val_loss, val_predictions, val_targets = collect_predictions(
            model,
            val_loader,
            loss_fn,
            device,
            stats.target_mean,
            stats.target_std,
            desc=f"{model_name} epoch {epoch} [val]",
        )
        val_metrics = compute_metrics(val_predictions, val_targets)
        epoch_progress.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            mae=f"{val_metrics['joint_mae']:.4f}",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            patience_left = model_params["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None or best_val_metrics is None:
        raise RuntimeError("Trial completed without producing a best model state.")

    model.load_state_dict(best_state)
    test_loss, test_predictions, test_targets = collect_predictions(
        model,
        test_loader,
        loss_fn,
        device,
        stats.target_mean,
        stats.target_std,
        desc=f"{model_name} [test]",
    )
    test_metrics = compute_metrics(test_predictions, test_targets)
    test_metrics["test_loss"] = float(test_loss)

    return {
        "score": float(best_val_metrics["joint_mae"]),
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }


def run_rnn_trial(data_params: dict, model_params: dict, cache: dict, device: torch.device) -> dict:
    """Run one RNN hyperparameter trial.

    :param data_params: Data-parameter dictionary.
    :param model_params: RNN hyperparameter dictionary.
    :param cache: Dataset cache.
    :param device: Device used for model training.
    :return: Trial outcome dictionary.
    """
    train_dataset, val_dataset, test_dataset, stats = get_neural_bundle(cache, data_params)
    model = SharedEnergyRNN(
        feature_dim=len(INPUT_FEATURES),
        num_houses=len(HOUSE_ORDER),
        hidden_size=model_params["hidden_size"],
        house_embedding_dim=model_params["house_embedding_dim"],
        num_layers=model_params["num_layers"],
        dropout=model_params["dropout"],
    ).to(device)
    return train_neural_trial("rnn", model, (train_dataset, val_dataset, test_dataset), model_params, stats, device)


def run_tcn_trial(data_params: dict, model_params: dict, cache: dict, device: torch.device) -> dict:
    """Run one TCN hyperparameter trial.

    :param data_params: Data-parameter dictionary.
    :param model_params: TCN hyperparameter dictionary.
    :param cache: Dataset cache.
    :param device: Device used for model training.
    :return: Trial outcome dictionary.
    """
    train_dataset, val_dataset, test_dataset, stats = get_neural_bundle(cache, data_params)
    model = SharedEnergyTCN(
        feature_dim=len(INPUT_FEATURES),
        num_houses=len(HOUSE_ORDER),
        hidden_size=model_params["hidden_size"],
        house_embedding_dim=model_params["house_embedding_dim"],
        num_layers=model_params["num_layers"],
        kernel_size=model_params["kernel_size"],
        dropout=model_params["dropout"],
    ).to(device)
    return train_neural_trial("tcn", model, (train_dataset, val_dataset, test_dataset), model_params, stats, device)


def run_transformer_trial(data_params: dict, model_params: dict, cache: dict, device: torch.device) -> dict:
    """Run one transformer hyperparameter trial.

    :param data_params: Data-parameter dictionary.
    :param model_params: Transformer hyperparameter dictionary.
    :param cache: Dataset cache.
    :param device: Device used for model training.
    :return: Trial outcome dictionary.
    """
    train_dataset, val_dataset, test_dataset, stats = get_neural_bundle(cache, data_params)
    model = SharedEnergyTransformer(
        feature_dim=len(INPUT_FEATURES),
        num_houses=len(HOUSE_ORDER),
        max_seq_len=data_params["seq_len"],
        d_model=model_params["d_model"],
        house_embedding_dim=model_params["house_embedding_dim"],
        num_layers=model_params["num_layers"],
        num_heads=model_params["num_heads"],
        feedforward_dim=model_params["feedforward_dim"],
        dropout=model_params["dropout"],
    ).to(device)
    return train_neural_trial(
        "transformer",
        model,
        (train_dataset, val_dataset, test_dataset),
        model_params,
        stats,
        device,
    )


def run_lightgbm_trial(data_params: dict, model_params: dict, cache: dict) -> dict:
    """Run one LightGBM hyperparameter trial.

    :param data_params: Data-parameter dictionary.
    :param model_params: LightGBM hyperparameter dictionary.
    :param cache: Tabular-data cache.
    :return: Trial outcome dictionary.
    :raises ImportError: If LightGBM is not installed.
    """
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed.")

    x_train, y_train, x_val, y_val, x_test, y_test = get_tabular_bundle(cache, data_params)
    base_model = LGBMRegressor(
        objective="regression",
        n_estimators=model_params["n_estimators"],
        learning_rate=model_params["learning_rate"],
        num_leaves=model_params["num_leaves"],
        max_depth=model_params["max_depth"],
        subsample=model_params["subsample"],
        colsample_bytree=model_params["colsample_bytree"],
        min_child_samples=model_params["min_child_samples"],
        reg_alpha=model_params["reg_alpha"],
        reg_lambda=model_params["reg_lambda"],
        random_state=SEED,
        n_jobs=-1,
        device_type=model_params["device_type"],
        verbosity=-1,
    )
    model = MultiOutputRegressor(base_model)
    model.fit(x_train, y_train)

    val_metrics = compute_metrics(model.predict(x_val), y_val)
    test_metrics = compute_metrics(model.predict(x_test), y_test)
    return {
        "score": float(val_metrics["joint_mae"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def run_xgboost_trial(data_params: dict, model_params: dict, cache: dict) -> dict:
    """Run one XGBoost hyperparameter trial.

    :param data_params: Data-parameter dictionary.
    :param model_params: XGBoost hyperparameter dictionary.
    :param cache: Tabular-data cache.
    :return: Trial outcome dictionary.
    :raises ImportError: If required XGBoost/CuPy dependencies are unavailable.
    """
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed.")

    x_train, y_train, x_val, y_val, x_test, y_test = get_tabular_bundle(cache, data_params)
    cp = None
    if str(model_params["device"]).startswith("cuda"):
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError(
                "GPU XGBoost prediction requires CuPy for GPU-backed arrays. "
                "Install a matching build (for example `pip install cupy-cuda12x`)."
            ) from exc
        x_train_device = cp.asarray(x_train)
        x_val_device = cp.asarray(x_val)
        x_test_device = cp.asarray(x_test)
    else:
        x_train_device = x_train
        x_val_device = x_val
        x_test_device = x_test

    base_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=model_params["n_estimators"],
        learning_rate=model_params["learning_rate"],
        max_depth=model_params["max_depth"],
        min_child_weight=model_params["min_child_weight"],
        subsample=model_params["subsample"],
        colsample_bytree=model_params["colsample_bytree"],
        reg_alpha=model_params["reg_alpha"],
        reg_lambda=model_params["reg_lambda"],
        tree_method=model_params["tree_method"],
        device=model_params["device"],
        random_state=SEED,
        n_jobs=0,
    )
    model = MultiOutputRegressor(base_model)
    model.fit(x_train_device, y_train)

    val_predictions = model.predict(x_val_device)
    test_predictions = model.predict(x_test_device)
    if cp is not None:
        val_predictions = cp.asnumpy(val_predictions)
        test_predictions = cp.asnumpy(test_predictions)

    val_metrics = compute_metrics(val_predictions, y_val)
    test_metrics = compute_metrics(test_predictions, y_test)
    return {
        "score": float(val_metrics["joint_mae"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def enumerate_trials():
    """Yield every model/data parameter combination for grid search.

    :return: Generator of ``(model_name, data_params, model_params)`` tuples.
    """
    for data_params in ParameterGrid(NEURAL_DATA_GRID):
        for model_params in ParameterGrid(RNN_GRID):
            yield "rnn", stable_params(data_params), stable_params(model_params)

    for data_params in ParameterGrid(NEURAL_DATA_GRID):
        for model_params in ParameterGrid(TCN_GRID):
            yield "tcn", stable_params(data_params), stable_params(model_params)

    for data_params in ParameterGrid(NEURAL_DATA_GRID):
        for model_params in ParameterGrid(TRANSFORMER_GRID):
            yield "transformer", stable_params(data_params), stable_params(model_params)

    for data_params in ParameterGrid(TABULAR_DATA_GRID):
        for model_params in ParameterGrid(LIGHTGBM_GRID):
            yield "lightgbm", stable_params(data_params), stable_params(model_params)

    for data_params in ParameterGrid(TABULAR_DATA_GRID):
        for model_params in ParameterGrid(XGBOOST_GRID):
            yield "xgboost", stable_params(data_params), stable_params(model_params)


def main() -> None:
    """Execute the full cross-model grid search and write result artifacts."""
    set_seed(SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device for neural models: {device}")

    ARTIFACT_DIR.mkdir(exist_ok=True)
    existing_results = load_previous_results()
    existing_by_id = {result["trial_id"]: result for result in existing_results}

    neural_cache = {}
    tabular_cache = {}
    all_trials = list(enumerate_trials())
    trial_progress = tqdm(all_trials, desc="Grid search")

    for model_name, data_params, model_params in trial_progress:
        current_trial_id = trial_id(model_name, {"data": data_params, "model": model_params})
        trial_progress.set_postfix(model=model_name)

        if should_skip(existing_by_id, current_trial_id):
            continue

        result_record = {
            "trial_id": current_trial_id,
            "model": model_name,
            "data_params": data_params,
            "model_params": model_params,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            if model_name == "rnn":
                if device.type != "cuda":
                    raise RuntimeError("RNN grid search requires CUDA.")
                outcome = run_rnn_trial(data_params, model_params, neural_cache, device)
            elif model_name == "tcn":
                if device.type != "cuda":
                    raise RuntimeError("TCN grid search requires CUDA.")
                outcome = run_tcn_trial(data_params, model_params, neural_cache, device)
            elif model_name == "transformer":
                if device.type != "cuda":
                    raise RuntimeError("Transformer grid search requires CUDA.")
                outcome = run_transformer_trial(data_params, model_params, neural_cache, device)
            elif model_name == "lightgbm":
                outcome = run_lightgbm_trial(data_params, model_params, tabular_cache)
            elif model_name == "xgboost":
                outcome = run_xgboost_trial(data_params, model_params, tabular_cache)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            result_record.update(
                {
                    "status": "ok",
                    "score": outcome["score"],
                    "val_metrics": outcome["val_metrics"],
                    "test_metrics": outcome["test_metrics"],
                }
            )
        except Exception as exc:
            result_record.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

        result_record["finished_at"] = datetime.now(timezone.utc).isoformat()
        append_result(result_record)
        existing_results.append(result_record)
        existing_by_id[current_trial_id] = result_record
        write_summary(existing_results)

    print("Grid search complete.")
    print("Results:", RESULTS_PATH)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
