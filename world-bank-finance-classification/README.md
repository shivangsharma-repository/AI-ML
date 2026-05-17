# World Bank Enterprise Surveys: Access-to-Finance Classification

## Problem Statement

Predict whether a country-year observation has above-median access-to-finance constraints using business environment indicators from the World Bank Enterprise Surveys.

This project is designed as a policy-facing supervised classification workflow: it compares an interpretable linear model with a non-linear ensemble model and translates model performance into practical business and policy insight.

## Dataset

- Source: World Bank Enterprise Surveys global indicators
- Unit of analysis: country-year observation
- Observations: 288 country-year observations in the final analytic sample
- Target variable: `fin16`, the percentage of firms identifying access to finance as a major or severe constraint
- Binary target: `high_finance_constraint`
  - `1`: above-median finance constraint
  - `0`: at or below median finance constraint
- Feature set: 18 business environment indicators across regulation, infrastructure, institutional quality, labor, competition, finance, and firm performance

## Methodology

### Data Preprocessing

1. Filtered raw long-format indicator data to aggregate-level observations.
2. Pivoted indicator-value pairs into a wide country-year modeling table.
3. Created a binary target using the median of `fin16`.
4. Selected 18 theoretically motivated predictors.
5. Applied complete-case filtering for the final analytic sample.
6. Created a 70/30 stratified train-test split.

### Models Compared

#### Logistic Regression

- Interpretable baseline classifier
- Standardized features
- L2 regularization
- Useful for policy-facing coefficient interpretation

#### Random Forest

- Non-linear ensemble classifier
- 500 trees
- Class-balanced weighting
- Out-of-bag validation
- Useful for maximizing F1 and recall for high-constraint cases

## Results

| Model | ROC-AUC | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.906 | 0.805 | 0.805 | 0.805 | 0.805 |
| Random Forest | 0.901 | 0.839 | 0.844 | 0.822 | 0.861 |

### Cross-Validation

| Model | Mean ROC-AUC | Std. Dev. |
|---|---:|---:|
| Logistic Regression | 0.909 | 0.032 |
| Random Forest | 0.906 | 0.028 |

### Interpretation

- Logistic Regression provided the strongest ROC-AUC and is preferred when interpretability is the priority.
- Random Forest provided stronger F1 and recall, making it more useful when the goal is to identify high-constraint cases.
- Random Forest out-of-bag accuracy was close to test accuracy, supporting generalization and reducing overfitting concerns.

## Project Structure

```text
world-bank-finance-classification/
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── tests/
│   ├── test_data_preprocessing.py
│   └── test_model_training.py
├── results/
│   ├── figures/
│   └── metrics/
└── data/
    ├── raw/
    └── processed/
```

## How to Run

```bash
cd world-bank-finance-classification
pip install -r requirements.txt
python src/data_preprocessing.py
python src/model_training.py
python src/evaluation.py
```

## Author

Shivang Sharma  
MS Data Science Candidate, Purdue University  
MIT PE Applied Data Science Program Graduate
