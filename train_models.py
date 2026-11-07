"""
ML Assignment 2 - model training

Trains five classifiers on the UCI Online Shoppers Purchasing Intention dataset,
evaluates them on a held-out 20% test split and saves:
  - model/<name>.joblib   one full pipeline (preprocessing + classifier) per model
  - model/metrics.csv     test metrics for all five models
  - test_data.csv         the raw test rows, used for the Streamlit app

Run:  python train_models.py
"""
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import CAT_COLS, FEATURES, NUM_COLS, TARGET, clean

SEED = 42
DATA_PATH = "data/online_shoppers_intention.csv"
MODEL_DIR = "model"


def positive_scores(pipe, X):
    # AUC only needs a score that ranks the positive class higher. SVC is trained
    # without probability=True (deprecated in recent sklearn), so use its decision function.
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def evaluate(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ---------------- load and split ----------------
raw = pd.read_csv(DATA_PATH)
print("Raw shape:", raw.shape)

raw = raw.drop_duplicates().reset_index(drop=True)
print("After dropping duplicate rows:", raw.shape)

train_raw, test_raw = train_test_split(raw, test_size=0.2, random_state=SEED,
                                       stratify=raw[TARGET])

# keep the test rows in their original format so the app sees exactly what a user would upload
test_raw.to_csv("test_data.csv", index=False)

train_df = clean(train_raw)
test_df = clean(test_raw)
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")
print(f"Purchase rate - train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

# ---------------- models ----------------
# Same preprocessing for every model: scale numeric columns, one-hot the categorical ones.
# Scaling matters for LR, KNN and SVM; it does no harm to the tree or Naive Bayes.
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    # depth / k picked with a quick 5-fold CV on the training set (see README)
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, gamma="scale"),
}

os.makedirs(MODEL_DIR, exist_ok=True)
rows = []
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    scores = evaluate(y_test, y_pred, positive_scores(pipe, X_test))
    rows.append({"Model": name, **scores})

    file_name = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join(MODEL_DIR, file_name))
    print(f"{name:24s} done -> {file_name}")

results = pd.DataFrame(rows).round(4)
results.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)

print("\nTest set results:")
print(results.to_string(index=False))
