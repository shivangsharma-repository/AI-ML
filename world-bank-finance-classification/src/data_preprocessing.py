"""Data preprocessing for the World Bank finance classification project.

This module loads long-format World Bank Enterprise Surveys indicator data,
filters to aggregate observations, pivots the data to country-year rows,
creates a binary access-to-finance target, and writes reproducible train/test
splits.

Author: Shivang Sharma
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "fin16"
BINARY_TARGET = "high_finance_constraint"
METADATA_COLUMNS = ["country", "country_code", "year", TARGET_COLUMN, BINARY_TARGET]


def load_config(config_path: Path | None = None) -> dict:
    """Load project configuration.

    Parameters
    ----------
    config_path : Path | None
        Optional path to config.yaml. If omitted, the project-level config is used.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw long-format World Bank indicator data."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")
    return pd.read_csv(filepath)


def filter_aggregate_level(data: pd.DataFrame) -> pd.DataFrame:
    """Filter to aggregate-level rows where cut='All' and subcut='All'."""
    required_columns = {"cut", "subcut"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data[(data["cut"] == "All") & (data["subcut"] == "All")].copy()


def pivot_to_wide_format(data: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format indicator rows into wide country-year rows."""
    required_columns = {"country", "country_code", "year", "indicator", "value"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return (
        data.pivot_table(
            index=["country", "country_code", "year"],
            columns="indicator",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )


def create_binary_target(data: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Create binary target using the median of the continuous target column."""
    if target_col not in data.columns:
        raise ValueError(f"Target column not found: {target_col}")

    output = data.copy()
    threshold = output[target_col].median()
    output[BINARY_TARGET] = (output[target_col] > threshold).astype(int)
    return output


def select_features(data: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """Select metadata, target, and configured modeling features."""
    selected_columns = METADATA_COLUMNS + feature_list
    missing = [column for column in selected_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Configured columns not found in data: {missing}")
    return data[selected_columns].copy()


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Apply complete-case filtering for reproducible modeling sample."""
    return data.dropna().copy()


def split_train_test(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create stratified train/test split."""
    return train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[BINARY_TARGET],
    )


def run_preprocessing(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete preprocessing workflow and save outputs."""
    raw_data = load_raw_data(config["data"]["raw_path"])
    aggregate_data = filter_aggregate_level(raw_data)
    wide_data = pivot_to_wide_format(aggregate_data)
    data_with_target = create_binary_target(wide_data, config["features"]["target"])
    feature_data = select_features(data_with_target, config["features"]["selected"])
    analytic_sample = handle_missing_values(feature_data)

    processed_dir = Path(config["data"]["processed_dir"])
    split_dir = processed_dir / "train_test_split"
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    wide_data.to_csv(processed_dir / "wide_format_data.csv", index=False)
    analytic_sample.to_csv(processed_dir / "analytic_sample.csv", index=False)

    train_data, test_data = split_train_test(
        analytic_sample,
        test_size=config["modeling"]["test_size"],
        random_state=config["modeling"]["random_state"],
    )
    train_data.to_csv(split_dir / "train.csv", index=False)
    test_data.to_csv(split_dir / "test.csv", index=False)
    return train_data, test_data


if __name__ == "__main__":
    project_config = load_config()
    train, test = run_preprocessing(project_config)
    print(f"Preprocessing complete. Train={train.shape}, Test={test.shape}")
