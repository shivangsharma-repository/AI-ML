#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 5 10:24:06 2024
@author: shivang.sharma
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
from prophet import Prophet          # fbprophet was deprecated; use 'prophet'

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "path/to/input"       # <-- update
OUTPUT_PATH  = "path/to/output"      # <-- update

WTHR_FILE    = "weather_data"
SLS_FILE     = "prophet_output"
PROMO_FILE   = "promo_data"

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Data preparation (merge sales, weather, promo + feature engineering)
# ══════════════════════════════════════════════════════════════════════════════

weather = pd.read_csv(f"{DATA_PATH}/{WTHR_FILE}.csv")
sales   = pd.read_csv(f"{DATA_PATH}/{SLS_FILE}.csv")
promo   = pd.read_csv(f"{DATA_PATH}/{PROMO_FILE}.csv")

# Merge weather onto sales
sales["Date"] = sales["Date"].astype(int)
m1 = sales.merge(weather, on="Date", how="left")

# Build join key for promo
m1["ID"]    = m1["Date"].astype(str) + "_" + m1["Product_Key"].astype(str)
promo["ID"] = promo["Date"].astype(str) + "_" + promo["Product_Key"].astype(str)
m2 = m1.merge(promo, on="ID", how="left")

# Date decomposition
m2["Date"]  = m2["Date_x"].astype(str)
m2["Year"]  = m2["Date"].str[:4]
m2["Week"]  = m2["Date"].str[-2:]
m2["Month"] = pd.DatetimeIndex(m2["ds"]).month

m2 = (
    m2
    .drop(columns=["Date_y", "ID", "Date_x", "Product_Key_y"])
    .rename(columns={"Product_Key_x": "Product_Key"})
)

# Lagged sales features
m3 = m2.sort_values(["Product_Key", "Date"]).copy()
for lag in range(4, 8):
    col = f"Sales_shift_{lag}"
    m3[col] = m3.groupby("Product_Key")["Sales"].shift(lag).bfill()

# Weather fill (forward-fill by week)
m4 = m3.copy()
weather_cols = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]
for w in weather_cols:
    m4[w] = m4.groupby("Week")[w].ffill()

# Promo lag features
for lag in (1, 2):
    col = f"Promo_shift_{lag}"
    m4[col] = m4.groupby("Product_Key")["Promo_Count"].shift(lag).bfill()

# Drop unneeded Prophet output columns
drop_cols = [
    "trend_lower", "trend_upper",
    "seasonal_lower", "seasonal_upper",
    "seasonalities", "seasonalities_lower", "seasonalities_upper",
    "weekly_lower", "weekly_upper",
    "yearly_lower", "yearly_upper",
    "ds",
]
m5 = m4.drop(columns=drop_cols, errors="ignore")

m5.to_csv(f"{OUTPUT_PATH}/merged.csv", index=False)   # written once only

print("Missing values after merge:\n", m5.isnull().sum())

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Prophet modelling
# ══════════════════════════════════════════════════════════════════════════════

SLS_INPUT  = "sales_data_v2"
DATE_MAP   = "Date_mapping"

d1       = pd.read_excel(f"{DATA_PATH}/{SLS_INPUT}.xlsx")
date_map = pd.read_excel(f"{DATA_PATH}/{DATE_MAP}.xlsx")
date_map["ds"] = pd.to_datetime(date_map["ds"])

d1 = d1.merge(date_map, on="Date", how="left")
d1["ID"] = d1["ds"].astype(str) + "_" + d1["Product_Key"].astype(str)
d1 = d1.sort_values(["Product_Key", "ds"]).reset_index(drop=True)

# Training cut-off
d2 = d1[d1["Date"] < 201648].reset_index(drop=True)

# ── Model 1: MCMC (captures uncertainty better) ───────────────────────────────
forecasts_mcmc = []
for prod in d2["Product_Key"].unique():
    d_prod = d2[d2["Product_Key"] == prod][["ds", "Sales"]].rename(columns={"Sales": "y"})
    m = Prophet(weekly_seasonality=True, mcmc_samples=300)
    m.fit(d_prod)
    future   = m.make_future_dataframe(periods=5, freq="W")
    forecast = m.predict(future)
    forecast["PK"] = prod
    forecasts_mcmc.append(forecast)

res = pd.concat(forecasts_mcmc)
res["ID"] = res["ds"].astype(str) + "_" + res["PK"].astype(str)

# ── Model 2: MAP / linear growth (faster) ────────────────────────────────────
forecasts_map = []
for prod in d2["Product_Key"].unique():
    d_prod = d2[d2["Product_Key"] == prod][["ds", "Sales"]].rename(columns={"Sales": "y"})
    m = Prophet(weekly_seasonality=True, growth="linear")
    m.fit(d_prod)
    future   = m.make_future_dataframe(periods=5, freq="W")
    forecast = m.predict(future)
    forecast["PK"] = prod
    forecasts_map.append(forecast)

res1 = pd.concat(forecasts_map)
res1["ID"] = res1["ds"].astype(str) + "_" + res1["PK"].astype(str)

# ── Ensemble ──────────────────────────────────────────────────────────────────
a = res[["ID", "yhat"]].merge(res1, on="ID", how="left")
a["yhat"]       = 0.9 * a["yhat_x"] + 0.1 * a["yhat_y"]
a["yhat_final"] = 0.78 * a["yhat"]
a.loc[a["yhat_final"] < 0, ["yhat_lower", "yhat_final"]] = 0

a["ds"] = pd.to_datetime(a["ds"])
a = (
    a
    .merge(date_map, on="ds", how="left")
    .merge(d1, on="ID", how="left")
    .loc[lambda df: df["Date_y"] != 201648]
    .drop(columns=["Date_y", "ds_y", "ID", "yhat", "yhat_x", "yhat_y", "Product_Key"])
    .rename(columns={"Date_x": "Date", "PK": "Product_Key", "ds_x": "ds"})
)

a.to_csv(f"{OUTPUT_PATH}/prophet_output.csv", index=False)
