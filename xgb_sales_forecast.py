#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 5 17:50:58 2024
@author: shivang.sharma
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import xgboost as xgb

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "path/to/input"     # <-- update
OUTPUT_PATH = "path/to/output"    # <-- update

IP_FILE   = "merged"
KEYS_FILE = "Keys_zero_sales_v2"

# ── Load data ─────────────────────────────────────────────────────────────────
df               = pd.read_csv(f"{DATA_PATH}/{IP_FILE}.csv")
df_key_zero      = pd.read_excel(f"{DATA_PATH}/{KEYS_FILE}.xlsx")
keys_zero        = list(df_key_zero["Fus"])

# ── Train / test split (by date) ──────────────────────────────────────────────
df_train = df[df["Date"] <  201649].reset_index(drop=True)
df_test  = df[df["Date"] >= 201649].reset_index(drop=True)

# ── Per-product XGBoost parameters ───────────────────────────────────────────
# "reg:linear" was removed in XGBoost ≥ 1.0 — replaced with "reg:squarederror"
_BASE = {
    "objective":        "reg:squarederror",
    "min_child_weight": 1,
    "subsample":        0.7,
    "colsample_bytree": 0.6,
    "silent":           1,
    "max_depth":        6,
    "seed":             1,
    "nthread":          -1,
}

FU_PARAMS = {
    "FGB0748": {**_BASE, "eta": 0.750},
    "FGB0751": {**_BASE, "eta": 0.110},
    "FGB0737": {**_BASE, "eta": 0.736},
    "FGB6596": {**_BASE, "eta": 0.007},
    "FGB0727": {**_BASE, "eta": 0.010},
    "FGB6299": {**_BASE, "eta": 0.003},
    "FGB6542": {**_BASE, "eta": 0.008},
    "FGB0726": {**_BASE, "eta": 0.003},
    "FGB0723": {**_BASE, "eta": 0.200},
    "FGB6543": {**_BASE, "eta": 0.400},
    "FGB0754": {**_BASE, "eta": 0.050},
    "FGB0735": {**_BASE, "eta": 0.010},
}

FU_NR = {
    "FGB0748":   20, "FGB0751":   13, "FGB0737":   15,
    "FGB6596": 2800, "FGB0727":  800, "FGB6299": 2500,
    "FGB6542":  500, "FGB0726": 1500, "FGB0723":   35,
    "FGB6543":   40, "FGB0754":  300, "FGB0735": 1000,
}

PROD_KEYS = list(FU_PARAMS.keys())
DROP_COLS = ["Product_Key", "Date", "yhat_final"]

# ── Model training helper ─────────────────────────────────────────────────────
def run_xgb(train_X, train_y, params, n_rounds):
    dtrain = xgb.DMatrix(train_X, label=train_y)
    return xgb.train(params, dtrain, n_rounds)

# ── Per-product predictions ───────────────────────────────────────────────────
df_test_out = df_test.copy()

for prod in PROD_KEYS:
    mask_tr = df_train["Product_Key"] == prod
    mask_te = df_test["Product_Key"]  == prod

    X_tr = df_train.loc[mask_tr].drop(columns=["Sales"] + DROP_COLS)
    y_tr = df_train.loc[mask_tr, "Sales"].values
    X_te = df_test.loc[mask_te].drop(columns=["Sales"] + DROP_COLS)

    model   = run_xgb(X_tr, y_tr, FU_PARAMS[prod], FU_NR[prod])
    y_pred  = model.predict(xgb.DMatrix(X_te))
    df_test_out.loc[mask_te, "XGB_Pred"] = y_pred

# Fall back to prophet baseline where XGB has no prediction
df_test_out["XGB_Pred"].fillna(df_test_out["yhat_final"], inplace=True)

# Zero-out known zero-sales products
df_test_out.loc[df_test_out["Sales"] == 0, "XGB_Pred"] = 0
df_test_out.loc[df_test_out["Product_Key"].isin(keys_zero), "XGB_Pred"] = 0

# ── Accuracy (MAPE-style) ─────────────────────────────────────────────────────
accuracy = 1 - (
    df_test_out["XGB_Pred"].sub(df_test_out["Sales"]).abs().sum()
    / df_test_out["Sales"].sum()
)
print(f"Accuracy: {accuracy:.4f}")

# ── Output ────────────────────────────────────────────────────────────────────
df_test_out.to_excel(f"{OUTPUT_PATH}/Final_Output.xlsx", index=False)

xgb.plot_importance(model)
