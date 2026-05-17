# Forecasting and Feature Engineering

## Problem Statement

Build reusable Python workflows for product-level demand forecasting and tabular machine learning feature engineering.

This project demonstrates how sales history, calendar mappings, weather variables, promotion data, lagged features, and model outputs can be combined into a structured forecasting pipeline.

## Project Components

| Component | Description |
|---|---|
| Data preparation | Merges sales, weather, promotion, and calendar mapping files into an analytics-ready table |
| Prophet forecasting | Builds product-level weekly time-series forecasts |
| XGBoost regression | Uses lagged sales, promo, weather, and calendar features for product-level demand prediction |
| Feature engineering | Creates lag features, promo shifts, interaction features, and target encodings |

## Current Files

- `../data_prep_and_prophet.py`: cleaned data preparation and Prophet forecasting workflow
- `../xgb_sales_forecast.py`: cleaned XGBoost product-level sales forecasting workflow
- `../feature_engineering.py`: reusable feature engineering and CatBoost classification workflow

## Modeling Techniques

- Prophet time-series forecasting
- XGBoost regression
- Lagged sales features
- Promotion shift features
- Weather joins
- Calendar decomposition
- K-fold target encoding
- Pairwise numerical interactions
- One-hot interaction features

## How to Run

```bash
pip install -r requirements.txt
python ../data_prep_and_prophet.py
python ../xgb_sales_forecast.py
```

Update `DATA_PATH` and `OUTPUT_PATH` before running locally.

## Author

Shivang Sharma  
MS Data Science Candidate, Purdue University  
MIT PE Applied Data Science Program Graduate
