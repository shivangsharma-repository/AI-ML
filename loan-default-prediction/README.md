# Loan Default Prediction

## Problem Statement

Predict whether a home equity loan applicant will default or become severely delinquent. The project is framed as an imbalanced binary classification problem where recall for the default class is important because missed defaults can create financial risk.

## Dataset

- Dataset: Home Equity dataset (HMEQ)
- Observations: 5,960 recent home equity loan records
- Target: `BAD`
  - `0`: loan was repaid / not severely delinquent
  - `1`: applicant defaulted or became severely delinquent
- Class imbalance: approximately 80% non-default and 20% default

## Methodology

### Data Preparation

- Loaded and profiled the HMEQ dataset
- Separated predictors from the binary target variable
- Handled missing values and categorical predictors through preprocessing
- Split data into training and testing sets
- Evaluated models with precision, recall, F1-score, accuracy, and confusion matrices

### Models Compared

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline for binary classification |
| Decision Tree | Non-linear baseline with high interpretability but overfitting risk |
| Random Forest with Class Weights | Ensemble model to improve generalization and handle imbalance |
| Tuned Random Forest | Final model selected using grid search and F1-focused evaluation |

## Results

The tuned Random Forest provided the strongest balance between overall accuracy and minority-class detection.

| Model | Test Accuracy | Test Recall | Test Precision |
|---|---:|---:|---:|
| Logistic Regression | 0.621 | 0.605 | 0.287 |
| Decision Tree | 0.861 | 0.613 | 0.666 |
| Random Forest with Class Weights | 0.901 | 0.653 | 0.815 |
| Tuned Random Forest | 0.909 | 0.695 | 0.821 |
| Default Random Forest | 0.904 | 0.667 | 0.818 |

## Key Findings

- Logistic Regression provided an interpretable baseline but struggled with the complexity and imbalance in the dataset.
- Decision Tree overfit the training data and generalized less effectively.
- Random Forest models performed substantially better, especially after class weighting and hyperparameter tuning.
- The tuned Random Forest achieved the strongest overall test performance and was selected as the final model.

## How to Run

```bash
cd loan-default-prediction
pip install -r requirements.txt
python src/train_model.py
```

## Author

Shivang Sharma  
MS Data Science Candidate, Purdue University  
MIT Applied Data Science Program Graduate
