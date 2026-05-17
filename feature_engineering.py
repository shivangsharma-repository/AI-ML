#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 5 10:24:06 2024
@author: shivang.sharma
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import catboost as cb

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "path/to/your/data"   # <-- update this

# ── Interaction feature helpers ───────────────────────────────────────────────

def one_hot_mat_mul(data, cols):
    """Outer product of one-hot encoded columns."""
    df1 = pd.get_dummies(data[cols])
    n   = df1.shape[1]
    x   = np.repeat(df1.values, n, axis=1)
    y   = np.tile(df1.values, n)
    z   = x * y
    colnames = [a + b for a, b in zip(
        np.repeat(df1.columns, n), np.tile(df1.columns, n)
    )]
    return pd.concat([data, pd.DataFrame(z, columns=colnames)], axis=1)


def _pairwise_interaction(data, cols, op):
    """Generic helper: apply *op* between all column pairs."""
    df  = data[cols].copy()
    n   = df.shape[1]
    x   = np.repeat(df.values, n, axis=1)
    y   = np.tile(df.values, n)
    z   = op(x, y)
    colnames = [a + b for a, b in zip(
        np.repeat(df.columns, n), np.tile(df.columns, n)
    )]
    return pd.concat([data, pd.DataFrame(z, columns=colnames)], axis=1)


def mul_interaction(data, cols):
    return _pairwise_interaction(data, cols, lambda x, y: x * y)

def add_interaction(data, cols):
    return _pairwise_interaction(data, cols, lambda x, y: x + y)

def div_interaction(data, cols):
    return _pairwise_interaction(data, cols, lambda x, y: x / y)

def minus_interaction(data, cols):
    return _pairwise_interaction(data, cols, lambda x, y: x - y)


# ── Target / mean encoding helpers ───────────────────────────────────────────

def _kfold_encode(x_train, x_test, col, target, agg_fn, suffix):
    """
    K-fold target encoding using *agg_fn* (e.g. 'mean', 'median',
    or a lambda for quantiles).
    Returns (encoded_train, encoded_test).
    """
    kf      = model_selection.KFold(n_splits=5, shuffle=True, random_state=123)
    new_col = f"{col}_{suffix}_{target}"
    t1      = pd.DataFrame()

    for i, (tr_idx, val_idx) in enumerate(kf.split(x_train)):
        x_tr, x_val = x_train.iloc[tr_idx], x_train.iloc[val_idx]
        mapping = x_tr.groupby(col)[target].agg(agg_fn)
        x_val   = x_val.copy()
        x_val[new_col] = x_val[col].map(mapping)
        t1[f"test_encoding_{i}"] = x_test[col].map(mapping)
        if i == 0:
            train_new = x_val
        else:
            train_new = pd.concat([train_new, x_val])

    x_test = x_test.copy()
    x_test[new_col] = t1.mean(axis=1)
    return train_new, x_test


def mean_encoding_single(x_train, x_test, col="work_type", target="stroke"):
    return _kfold_encode(x_train, x_test, col, target, "mean", "mean_encoding")

def median_encoding_single(x_train, x_test, col="work_type", target="stroke"):
    return _kfold_encode(x_train, x_test, col, target, "median", "median_encoding")

def first_quantile_encoding_single(x_train, x_test, col="work_type", target="stroke"):
    return _kfold_encode(x_train, x_test, col, target,
                         lambda s: s.quantile(0.25), "1quantile_encoding")

def third_quantile_encoding_single(x_train, x_test, col="work_type", target="stroke"):
    return _kfold_encode(x_train, x_test, col, target,
                         lambda s: s.quantile(0.75), "3quantile_encoding")


def double_mean(train2, val2,
                col=("LOSS_POSTCODE", "MNTH_OCC_DT"), target="CAT_FLAG"):
    """Two-column grouped mean encoding with K-fold OOF."""
    col  = list(col)
    name = f"{'_'.join(col)}_{target}_mean"
    kf   = model_selection.KFold(n_splits=5, shuffle=True, random_state=123)
    t1   = pd.DataFrame()

    for i, (tr_idx, val_idx) in enumerate(kf.split(train2)):
        x_tr, x_val = train2.iloc[tr_idx], train2.iloc[val_idx]
        t = x_tr.groupby(col)[target].mean().reset_index().rename(
            columns={target: name}
        )
        x_val = pd.merge(x_val, t, on=col, how="left")
        t1[f"test_encoding_{i}"] = pd.merge(val2[col], t, on=col, how="left")[name]
        if i == 0:
            train_new = x_val
        else:
            train_new = pd.concat([train_new, x_val])

    val2 = val2.copy()
    val2[name] = t1.mean(axis=1)
    return train_new, val2


# ── Load data ─────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA_PATH + "/train.csv")
test  = pd.read_csv(DATA_PATH + "/test.csv")

# ── Pre-processing ────────────────────────────────────────────────────────────
num_cols = ["age", "length_of_service", "avg_training_score"]
for col in num_cols:
    train[col] = train[col] * 1.0
    test[col]  = test[col]  * 1.0

train["previous_year_rating"].fillna(0, inplace=True)
test["previous_year_rating"].fillna(0, inplace=True)
train["previous_year_rating"] = train["previous_year_rating"].astype(int)
test["previous_year_rating"]  = test["previous_year_rating"].astype(int)

train["education"].fillna("unknown", inplace=True)
test["education"].fillna("unknown", inplace=True)

print("Missing values — train:\n", train.isnull().sum())
print("Missing values — test:\n",  test.isnull().sum())

# ── Train / validation split ──────────────────────────────────────────────────
x = train.drop(columns=["is_promoted", "employee_id"])
y = train["is_promoted"]

x_tr1, X_val,   y_tr1, y_val   = train_test_split(x,    y,    test_size=0.1,  random_state=123)
x_train, X_test, y_train, y_test = train_test_split(x_tr1, y_tr1, test_size=0.3, random_state=123)

# ── Baseline CatBoost model ───────────────────────────────────────────────────
model = cb.CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=15,
    loss_function="CrossEntropy",
    feature_border_type="MinEntropy",
    thread_count=-1,           # use all available cores
    random_seed=123,
    use_best_model=True,
    has_time=False,
    random_strength=1,
    bagging_temperature=0.8,
    eval_metric="Logloss",
    verbose=False,
)

cat_feature_idx = np.where(x_train.dtypes != float)[0]   # np.float removed in NumPy 1.24
bst = model.fit(
    x_train, y_train,
    cat_features=cat_feature_idx,
    eval_set=(X_test, y_test),
)

# ── Evaluation ───────────────────────────────────────────────────────────────
pred_test  = bst.predict_proba(X_test)[:, 1]
pred_val   = bst.predict_proba(X_val)[:, 1]

auc_test   = roc_auc_score(y_test, pred_test)
auc_val    = roc_auc_score(y_val,  pred_val)

f1_test = f1_score(np.where(pred_test > 0.26,  1, 0), y_test, pos_label=1, average="binary")
f1_val  = f1_score(np.where(pred_val  > 0.291, 1, 0), y_val,  pos_label=1, average="binary")

print(f"Test  — AUC: {auc_test:.4f}  F1: {f1_test:.4f}")
print(f"Val   — AUC: {auc_val:.4f}  F1: {f1_val:.4f}")

# ── Feature importances ───────────────────────────────────────────────────────
importances = pd.Series(bst.feature_importances_, index=x_train.columns)
importances.sort_values(ascending=False, inplace=True)
importances.plot(kind="barh", figsize=(10, 8), title="Feature Importance")

# ── Submission ────────────────────────────────────────────────────────────────
test1 = test.drop(columns=["employee_id"])
pred_prob = bst.predict_proba(test1)[:, 1]

test["pred_prob"]   = pred_prob
test["is_promoted"] = np.where(pred_prob > 0.296, 1, 0)
test[["employee_id", "is_promoted"]].to_csv(DATA_PATH + "/submission.csv", index=False)

# ── Important-variable analysis ───────────────────────────────────────────────
imp = (
    importances
    .reset_index()
    .rename(columns={"index": "var", 0: "importance"})
)
imp["cumulative"] = imp["importance"].cumsum()
imp = imp[imp["importance"] > 0]
cols_imp = list(imp["var"].unique())
top_20   = list(imp["var"].head(20))
