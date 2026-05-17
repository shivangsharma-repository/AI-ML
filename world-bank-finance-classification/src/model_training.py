"""Model training for the World Bank finance classification project.

This module trains Logistic Regression and Random Forest classifiers, evaluates
holdout performance, and saves model metrics for reproducible comparison.

Author: Shivang Sharma
"""

from pathlib import Path
import json
from typing import Dict, Tuple

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BINARY_TARGET = "high_finance_constraint"
DROP_COLUMNS = ["country", "country_code", "year", "fin16", BINARY_TARGET]


def load_config(config_path: Path | None = None) -> dict:
    """Load project configuration."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def prepare_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split modeling dataframe into features and target."""
    missing = [column for column in DROP_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    features = data.drop(columns=DROP_COLUMNS)
    target = data[BINARY_TARGET]
    return features, target


def build_logistic_regression(config: dict) -> Pipeline:
    """Build standardized Logistic Regression model pipeline."""
    params = config["modeling"]["logistic_regression"]
    model = LogisticRegression(
        C=params["C"],
        solver=params["solver"],
        max_iter=params["max_iter"],
        class_weight=params["class_weight"],
    )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def build_random_forest(config: dict) -> RandomForestClassifier:
    """Build Random Forest classifier from config."""
    params = config["modeling"]["random_forest"]
    return RandomForestClassifier(**params)


def evaluate_classifier(model, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate fitted classifier on holdout set."""
    predicted_labels = model.predict(x_test)
    predicted_probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predicted_labels),
        "precision": precision_score(y_test, predicted_labels, zero_division=0),
        "recall": recall_score(y_test, predicted_labels, zero_division=0),
        "f1": f1_score(y_test, predicted_labels, zero_division=0),
        "roc_auc": roc_auc_score(y_test, predicted_probabilities),
    }


def train_models(config: dict) -> Dict[str, Dict[str, float]]:
    """Train and evaluate Logistic Regression and Random Forest models."""
    split_dir = Path(config["data"]["processed_dir"]) / "train_test_split"
    train_data = pd.read_csv(split_dir / "train.csv")
    test_data = pd.read_csv(split_dir / "test.csv")

    x_train, y_train = prepare_features(train_data)
    x_test, y_test = prepare_features(test_data)

    models = {
        "logistic_regression": build_logistic_regression(config),
        "random_forest": build_random_forest(config),
    }

    metrics = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        metrics[model_name] = evaluate_classifier(model, x_test, y_test)
        if hasattr(model, "oob_score_"):
            metrics[model_name]["oob_score"] = float(model.oob_score_)

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


if __name__ == "__main__":
    project_config = load_config()
    model_metrics = train_models(project_config)
    print(json.dumps(model_metrics, indent=2))
