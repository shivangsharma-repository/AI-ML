# ── Imports ──────────────────────────────────────────────────────────────────
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

import nltk
from nltk.stem.snowball import SnowballStemmer

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import xgboost as xgb

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "path/to/data"  # update as needed

# ── Load data ─────────────────────────────────────────────────────────────────
df_train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
df_campaign = pd.read_csv(os.path.join(DATA_PATH, "campaign_data.csv"))
df_test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

# ── Text preprocessing ────────────────────────────────────────────────────────
ENG_STOP_WORDS = nltk.corpus.stopwords.words("english")
FILTER_LIST = ["dear", "hi", "avians"]
_stemmer = SnowballStemmer("english")
_tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")


def clean_up_text(query):
    soup = BeautifulSoup(query, "lxml")
    text = soup.get_text()
    return text.strip("\n").strip("\t").strip("\r").replace("\n", " ").replace("\t", " ").replace("\r", " ")


def clean(mail):
    text = re.sub(r"\.+", ". ", mail)
    text = re.sub(r"([.,!#?()=<>*/&}~^@|{])", r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return clean_up_text(text)


def preprocessing(text):
    tokens = _tokenizer.tokenize(clean(str(text)))
    tokens = [t.lower() for t in tokens if t.lower() not in FILTER_LIST]
    tokens = [t for t in tokens if t not in ENG_STOP_WORDS]
    tokens = [re.sub(r"\d+", "", t) for t in tokens]
    tokens = [t for t in tokens if t]
    tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


# ── Feature engineering ───────────────────────────────────────────────────────
def clean_text_columns(df):
    df = df.copy()
    df["clean_subject"] = df["subject"].fillna("").apply(preprocessing)
    df["clean_email"] = df["email_body"].fillna("").apply(preprocessing)
    return df


def subject_line_features(df):
    df = df.copy()
    subj = df["subject"].fillna("")
    df["len_subject"] = df["clean_subject"].str.len().replace(0, 1)
    df["capitals"] = subj.apply(lambda s: sum(1 for c in s if c.isupper()))
    df["caps_vs_length"] = df["capitals"] / df["len_subject"]
    df["num_exclamation_marks"] = subj.apply(lambda s: s.count("!"))
    df["num_question_marks"] = subj.apply(lambda s: s.count("?"))
    df["num_punctuation"] = subj.apply(lambda s: sum(s.count(c) for c in ".,;:"))
    df["num_symbols"] = subj.apply(lambda s: sum(s.count(c) for c in "*&$%"))
    df["num_words"] = subj.apply(lambda s: max(len(s.split()), 1))
    df["num_unique_words"] = subj.apply(lambda s: len(set(s.split())))
    df["words_vs_unique"] = df["num_unique_words"] / df["num_words"]
    df["prize_money"] = subj.apply(lambda s: int(bool(re.search(r"\binr|win|rs|prizes|prize\b", s.lower()))))
    return df


def email_body_features(df):
    df = df.copy()
    body = df["email_body"].fillna("")
    df["len_email"] = df["clean_email"].str.len().replace(0, 1)
    df["total_length_b"] = body.apply(len).replace(0, 1)
    df["capitals_b"] = body.apply(lambda s: sum(1 for c in s if c.isupper()))
    df["caps_vs_length_b"] = df["capitals_b"] / df["total_length_b"]
    df["num_exclamation_marks_b"] = body.apply(lambda s: s.count("!"))
    df["num_question_marks_b"] = body.apply(lambda s: s.count("?"))
    df["num_punctuation_b"] = body.apply(lambda s: sum(s.count(c) for c in ".,;:"))
    df["num_symbols_b"] = body.apply(lambda s: sum(s.count(c) for c in "*&$%"))
    df["num_words_b"] = body.apply(lambda s: max(len(s.split()), 1))
    df["num_unique_words_b"] = body.apply(lambda s: len(set(s.split())))
    df["words_vs_unique_b"] = df["num_unique_words_b"] / df["num_words_b"]
    return df


def add_tfidf(df_train, df_test, col):
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[col] = df_train[col].fillna("text na")
    df_test[col] = df_test[col].fillna("text na")

    ngram_range = (1, 3) if col == "clean_subject" else (1, 1)
    vect = TfidfVectorizer(stop_words="english", max_df=0.95, ngram_range=ngram_range)
    vect.fit(list(df_train[col]))

    suffix = col[-1]
    cols = [f"{f}_{suffix}" for f in vect.get_feature_names_out()]
    train_tfidf = pd.DataFrame(vect.transform(list(df_train[col])).toarray(), columns=cols)
    test_tfidf = pd.DataFrame(vect.transform(list(df_test[col])).toarray(), columns=cols)
    return pd.concat([df_train, train_tfidf], axis=1), pd.concat([df_test, test_tfidf], axis=1)


def create_dummies(df, col):
    return pd.concat([df.copy(), pd.get_dummies(df[col])], axis=1)


def user_patterns(df):
    agg = df.groupby("user_id").agg({"is_open": ["count", "sum"], "is_click": "sum"}).reset_index()
    agg.columns = ["user_id", "user_id_count", "is_open", "is_click"]
    agg["open_rate_user"] = agg["is_open"] / agg["user_id_count"]
    agg["click_rate_user"] = agg["is_click"] / agg["user_id_count"]
    agg["click_rate_if_opened_user"] = agg.apply(lambda r: r["is_click"] / r["is_open"] if r["is_open"] > 0 else 0, axis=1)
    return agg[["user_id", "open_rate_user", "click_rate_if_opened_user", "click_rate_user"]]


def communication_patterns(df):
    agg = df.groupby("communication_type").agg({"is_open": ["count", "sum"], "is_click": "sum"}).reset_index()
    agg.columns = ["communication_type", "communication_type_count", "is_open", "is_click"]
    agg["open_rate_commun"] = agg["is_open"] / agg["communication_type_count"]
    agg["click_rate_commun"] = agg["is_click"] / agg["communication_type_count"]
    agg["click_rate_if_opened_commun"] = agg.apply(lambda r: r["is_click"] / r["is_open"] if r["is_open"] > 0 else 0, axis=1)
    return agg[["communication_type", "open_rate_commun", "click_rate_if_opened_commun", "click_rate_commun"]]


def time_variables(df):
    df = df.copy()
    dates = pd.to_datetime(df["send_date"], dayfirst=True)
    df["hour_var"] = dates.dt.hour
    df["month_var"] = dates.dt.month
    df["office_hours"] = df["hour_var"].between(10, 18).astype(int)
    df["day_of_week"] = dates.dt.weekday.map({0: "monday_week", 1: "tuesday_week", 2: "wednesday_week", 3: "thursday_week", 4: "friday_week", 5: "saturday_week", 6: "sunday_week"})
    return create_dummies(df, "day_of_week").drop(["send_date_2", "send_date"], axis=1, errors="ignore")


def interactive_features(df):
    df = df.copy()
    for c in ["total_links", "no_of_sections", "no_of_internal_links", "len_email"]:
        if c in df.columns:
            df[c] = df[c].replace(0, 1)
    df["internal_link_percent"] = df["no_of_internal_links"] / df["total_links"]
    df["image_per_section"] = df["no_of_images"] / df["no_of_sections"]
    df["image_to_links"] = df["no_of_images"] / df["total_links"]
    df["link_per_section"] = df["total_links"] / df["no_of_sections"]
    df["image_per_internal"] = df["no_of_images"] / df["no_of_internal_links"]
    df["section_size"] = df["no_of_sections"] / df["len_email"]
    df["sub_to_body"] = df["len_subject"] / df["len_email"]
    return df


# ── Campaign processing ───────────────────────────────────────────────────────
def campaign_file_processing(df, cmp_id=54, cmp_id_2=80):
    df = create_dummies(df.copy(), "communication_type")
    df_tr = df[df["campaign_id"] <= cmp_id].reset_index(drop=True)
    df_te = df[(df["campaign_id"] > cmp_id) & (df["campaign_id"] <= cmp_id_2)].reset_index(drop=True)

    def _process(tr, te):
        tr, te = clean_text_columns(tr), clean_text_columns(te)
        tr, te = add_tfidf(tr, te, "clean_subject")
        tr, te = subject_line_features(tr), subject_line_features(te)
        tr, te = add_tfidf(tr, te, "clean_email")
        tr, te = email_body_features(tr), email_body_features(te)
        tr, te = interactive_features(tr), interactive_features(te)
        return tr, te

    tr_full, te_full = _process(df_tr, df_te)
    drop_shared = ["email_body", "subject", "email_url", "clean_subject", "clean_email"]
    drop_isopen = ["total_links", "no_of_internal_links", "no_of_images", "no_of_sections"] + drop_shared
    return tr_full.drop(drop_isopen, axis=1, errors="ignore"), te_full.drop(drop_isopen, axis=1, errors="ignore"), tr_full.drop(drop_shared, axis=1, errors="ignore"), te_full.drop(drop_shared, axis=1, errors="ignore")


# ── XGBoost utilities ─────────────────────────────────────────────────────────
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eta": 0.01,
    "min_child_weight": 1,
    "subsample": 0.7,
    "colsample_bytree": 0.2,
    "max_depth": 5,
    "seed": 1,
    "n_jobs": -1,
}


def get_oof(x_train, y_train, x_test, n_folds=3, n_rounds=500):
    kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
    oof_train = np.zeros(x_train.shape[0])
    oof_test_skf = np.empty((n_folds, x_test.shape[0]))
    auc_scores = []

    for i, (tr_idx, val_idx) in enumerate(kf.split(x_train)):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_val, y_val = x_train[val_idx], y_train[val_idx]
        model = xgb.train(XGB_PARAMS, xgb.DMatrix(x_tr, y_tr), n_rounds)
        y_pred = model.predict(xgb.DMatrix(x_val))
        oof_train[val_idx] = y_pred
        auc_scores.append(metrics.roc_auc_score(y_val, y_pred))
        oof_test_skf[i] = model.predict(xgb.DMatrix(x_test))

    print("AUC cross-val scores:", auc_scores)
    return oof_train.reshape(-1, 1), oof_test_skf.mean(axis=0).reshape(-1, 1)


# ── Main model training template ──────────────────────────────────────────────
def train_click_model(x_train, y_train, x_test):
    params = {**XGB_PARAMS, "subsample": 0.4, "scale_pos_weight": 99, "gamma": 1}
    xgtrain = xgb.DMatrix(x_train, y_train)
    xgtest = xgb.DMatrix(x_test)
    model = xgb.train(params, xgtrain, num_boost_round=2000)
    return model, model.predict(xgtrain), model.predict(xgtest)


if __name__ == "__main__":
    print("Loaded email click prediction pipeline template.")
    print("Update DATA_PATH and run the feature engineering functions before fitting the final model.")
