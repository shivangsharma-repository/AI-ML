#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean loan default prediction workflow.

Repository-friendly version of the loan default prediction notebook.
It loads a loan dataset, preprocesses numeric and categorical features, trains
classification models, evaluates model performance, and exports a comparison file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DATA_PATH = Path("path/to/your/HMEQ.csv")
TARGET = "BAD"
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' not found.")
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    X = df.drop(columns=[TARGET])
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer(transformers=[("num", numeric_pipeline, numeric_features), ("cat", categorical_pipeline, categorical_features)])


def build_models(preprocessor: ColumnTransformer) -> dict:
    return {
        "Logistic Regression": Pipeline(steps=[("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        "Gradient Boosting": Pipeline(steps=[("preprocessor", preprocessor), ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
    }


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model.named_steps["model"], "predict_proba") else None
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("ROC-AUC:", roc_auc)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }


def main():
    df = load_data(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)

    preprocessor = build_preprocessor(df)
    models = build_models(preprocessor)

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate_model(name, model, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    print("\nModel comparison:\n", results_df)
    results_df.to_csv("loan_default_model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
