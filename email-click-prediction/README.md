# Email Click Prediction

## Problem Statement

Predict whether a user will click an email campaign using campaign metadata, email subject/body text, user history, communication patterns, and time-based engagement features.

This project is positioned as a marketing analytics and product analytics classification workflow. It demonstrates feature engineering for text, user behavior, campaign metadata, and imbalanced binary classification.

## Methodology

### Text Preprocessing

- Cleaned HTML email bodies using BeautifulSoup
- Removed punctuation, excess whitespace, stop words, and filtered common greeting terms
- Applied stemming using NLTK SnowballStemmer
- Created cleaned subject and body text fields

### Feature Engineering

- Subject-line features: length, capitalization ratio, punctuation, symbols, unique-word ratio, prize/money indicators
- Email-body features: body length, capitalization ratio, punctuation, symbol counts, word diversity
- TF-IDF features for subject and body text
- User-level historical behavior: open rate, click rate, click-if-opened rate
- Communication-type behavior features
- Time features: hour, month, office-hours flag, day-of-week dummies
- Interactive content features: image/link ratios, section size, subject-to-body ratio

### Modeling

- XGBoost binary classification
- Out-of-fold prediction workflow for intermediate `is_open` probability
- Final click model using engineered behavioral, text, and campaign features
- ROC-AUC used as a primary ranking metric

## Current Files

- `../email_click_prediction_clean.py`: cleaned Python pipeline template

## How to Run

```bash
pip install -r requirements.txt
python ../email_click_prediction_clean.py
```

Update the `DATA_PATH` constant before running locally.

## Author

Shivang Sharma  
MS Data Science Candidate, Purdue University  
MIT PE Applied Data Science Program Graduate
