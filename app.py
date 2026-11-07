"""
Streamlit app for ML Assignment 2.

Upload a test CSV (same columns as the UCI Online Shoppers dataset, including
the Revenue column), pick one of the five saved models and see how it does.
The models are only loaded from model/ and used for prediction, never retrained here.
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from preprocessing import FEATURES, TARGET, clean

st.set_page_config(page_title="Shopper Purchase Predictor", page_icon="🛒", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_models():
    return {name: joblib.load(f"model/{fname}") for name, fname in MODEL_FILES.items()}


@st.cache_data
def read_csv(file):
    return pd.read_csv(file)


def positive_scores(model, X):
    # SVC has no predict_proba here, so fall back to its decision function for AUC
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def get_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if y_true.nunique() == 2 else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


models = load_models()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload a test CSV", type="csv")
    use_sample = st.checkbox("No file? Use the bundled test_data.csv", value=False)
    with open("test_data.csv", "rb") as f:
        st.download_button("Download sample test_data.csv", f, file_name="test_data.csv")

    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))

    st.divider()
    st.caption("Models were trained offline with train_models.py on 80% of the data. "
               "Nothing is retrained in this app.")

# ---------------- header ----------------
st.title("🛒 Will this shopper buy something?")
st.write("Predicting whether an online shopping session ends in a purchase, using the "
         "**Online Shoppers Purchasing Intention** dataset from UCI.")

if uploaded is not None:
    raw = read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    raw = read_csv("test_data.csv")
    source = "test_data.csv (bundled)"
else:
    st.info("Upload a CSV from the sidebar, or tick the box to use the bundled test data.")
    st.write("The file needs these columns:")
    st.code(", ".join(FEATURES + [TARGET]))
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
if missing:
    st.error(f"These columns are missing from the uploaded file: {', '.join(missing)}")
    st.stop()

df = clean(raw)
bad_rows = df[TARGET].isna() | df["Weekend"].isna()
if bad_rows.any():
    st.warning(f"Dropping {bad_rows.sum()} rows where Revenue/Weekend could not be read as True/False.")
    df = df[~bad_rows]

X, y = df[FEATURES], df[TARGET].astype(int)
st.caption(f"Using **{source}**: {len(df)} sessions, "
           f"{int(y.sum())} purchases ({y.mean():.1%}).")

tab_single, tab_compare, tab_data = st.tabs(["Selected model", "Compare all models", "Data preview"])

# ---------------- selected model ----------------
with tab_single:
    model = models[model_name]
    y_pred = model.predict(X)
    m = get_metrics(y, y_pred, positive_scores(model, X))

    st.subheader(f"{model_name}: results on the uploaded data")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No purchase", "Purchase"],
                    yticklabels=["No purchase", "Purchase"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=[0, 1],
                                       target_names=["No purchase", "Purchase"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

        tn, fp, fn, tp = cm.ravel()
        st.write(f"Out of {tp + fn} actual buyers the model caught **{tp}** and missed **{fn}**. "
                 f"It also flagged **{fp}** non-buyers as buyers.")

# ---------------- comparison ----------------
with tab_compare:
    st.subheader("All five models on the same data")
    rows = []
    for name, mdl in models.items():
        pred = mdl.predict(X)
        rows.append({"Model": name, **get_metrics(y, pred, positive_scores(mdl, X))})
    table = pd.DataFrame(rows).set_index("Model")

    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
                 width="stretch")
    best = table["F1"].idxmax()
    st.write(f"Best F1 on this data: **{best}** ({table.loc[best, 'F1']:.3f}). "
             "Green cells mark the best value in each column.")
    st.bar_chart(table[["F1", "MCC"]])

# ---------------- data ----------------
with tab_data:
    st.subheader("First 20 rows")
    st.dataframe(raw.head(20), width="stretch")
    st.write("Class balance in this file:")
    st.bar_chart(y.map({0: "No purchase", 1: "Purchase"}).value_counts())
