"""Static configuration constants for data locations and feature schema."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "ireland_data"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

ENERGY_COLUMNS = [
    "Discharge(Wh)",
    "Charge(Wh)",
    "Production(Wh)",
    "Consumption(Wh)",
    "Feed-in(Wh)",
    "From grid(Wh)",
    "State of Charge(%)",
]
WEATHER_COLUMNS = ["speed", "dir", "drybulb", "cbl", "soltot", "rain"]

INPUT_FEATURES = [
    "Production(Wh)",
    "Consumption(Wh)",
    *[column for column in ENERGY_COLUMNS if column not in {"Production(Wh)", "Consumption(Wh)"}],
    *WEATHER_COLUMNS,
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]
TARGET_COLUMNS = ["Production(Wh)", "Consumption(Wh)"]
HOUSE_ORDER = [f"H{i}" for i in range(1, 21)]
