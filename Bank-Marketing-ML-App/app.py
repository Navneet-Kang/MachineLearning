# app.py
# Streamlit app for ML Assignment 2 (Bank Marketing Dataset)
# - Loads dataset from GitHub repo (bank.csv) 
# - Loads trained models from model/*.pkl
# - Shows comparison table for all 6 models (metrics)
# - Allows model selection
# - Displays selected model metrics + confusion matrix + classification report

import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report



st.set_page_config(page_title="ML Assignment 2 - Model Comparison", layout="wide")
st.title(" Bank Marketing — Classification Model Comparison (Streamlit App)")


def read_csv_safely(file_obj_or_path):

    # Try common separators first for better reliability than autodetect
    for sep in [";", ",", "\t", "|"]:
        try:
            df_try = pd.read_csv(file_obj_or_path, sep=sep)
            # If it parsed into 1 column only, it's probably wrong separator
            if df_try.shape[1] > 1:
                return df_try
        except Exception:
            pass

    # Fallback: autodetect
    return pd.read_csv(file_obj_or_path, sep=None, engine="python")



DEFAULT_DATA_PATH = os.path.join("data", "bank.csv")
METRICS_PATH = os.path.join("model", "model_metrics.csv")

MODEL_FILE_MAP = {
    "Logistic Regression": os.path.join("model", "logistic_regression.pkl"),
    "Decision Tree": os.path.join("model", "decision_tree.pkl"),
    "KNN": os.path.join("model", "knn.pkl"),
    "Naive Bayes (Gaussian)": os.path.join("model", "naive_bayes_gaussian.pkl"),
    "Random Forest": os.path.join("model", "random_forest.pkl"),
    "XGBoost": os.path.join("model", "xgboost.pkl"),
}


st.subheader(" Model Comparison Table (All 6 Models)")

if not os.path.exists(METRICS_PATH):
    st.error(
        " model/model_metrics.csv not found.\n\n"
        "Fix: Run your training script locally to generate it, then push the `model/` folder to GitHub."
    )
    st.stop()

metrics_df = pd.read_csv(METRICS_PATH)

expected_cols = {"Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"}
missing = expected_cols - set(metrics_df.columns)
if missing:
    st.error(f" model_metrics.csv is missing these columns: {sorted(list(missing))}")
    st.stop()

st.dataframe(metrics_df, use_container_width=True)


st.subheader("Dataset (GitHub default + CSV upload)")

left, right = st.columns(2)

with left:
    use_default = st.checkbox("Use default dataset from GitHub repo (data/bank.csv)", value=True)
    st.caption("Default file path expected: data/bank.csv")

with right:
    uploaded_file = st.file_uploader("Or upload a CSV file (recommended: small test set)", type=["csv"])

df = None

if uploaded_file is not None:
    try:
        df = read_csv_safely(uploaded_file)
        st.success(" Uploaded dataset loaded successfully.")
    except Exception as e:
        st.error(f" Failed to read uploaded CSV: {e}")
        st.stop()
elif use_default:
    if not os.path.exists(DEFAULT_DATA_PATH):
        st.error(
            " Default dataset not found at data/bank.csv.\n\n"
            "Fix: Put bank.csv inside a folder named `data/` in your GitHub repo, then redeploy."
        )
        st.stop()
    try:
        df = read_csv_safely(DEFAULT_DATA_PATH)
        st.success(" Default dataset loaded from GitHub repo.")
    except Exception as e:
        st.error(f"Failed to read data/bank.csv: {e}")
        st.stop()
else:
    st.info("Upload a CSV or enable default dataset to continue.")
    st.stop()

st.write("Preview:")
st.dataframe(df.head(), use_container_width=True)


# ---------------------------
# Model selection
# ---------------------------
st.subheader(" Select Model")

# Only show models that exist in your repo (prevents deployment crashes)
available_models = []
for m in metrics_df["Model"].tolist():
    if m in MODEL_FILE_MAP and os.path.exists(MODEL_FILE_MAP[m]):
        available_models.append(m)

if not available_models:
    st.error(
        " No model .pkl files found in the `model/` folder.\n\n"
        "Fix: Make sure you have these files in GitHub:\n"
        "- model/logistic_regression.pkl\n"
        "- model/decision_tree.pkl\n"
        "- model/knn.pkl\n"
        "- model/naive_bayes_gaussian.pkl\n"
        "- model/random_forest.pkl\n"
        "- model/xgboost.pkl\n"
    )
    st.stop()

model_option = st.selectbox("Choose one trained model:", available_models)

# Load selected model
try:
    model = joblib.load(MODEL_FILE_MAP[model_option])
except Exception as e:
    st.error(f" Could not load model file: {MODEL_FILE_MAP[model_option]}\nError: {e}")
    st.stop()

# Show selected model metrics
st.subheader("Selected Model Metrics")
st.dataframe(metrics_df[metrics_df["Model"] == model_option], use_container_width=True)

st.subheader(" Confusion Matrix & Classification Report")

st.caption("Your dataset must include target column `y` with values `yes/no` to compute the confusion matrix.")

if st.button("Run Evaluation"):
    if "y" not in df.columns:
        st.error(" Target column `y` not found in dataset. Add column `y` (yes/no) to evaluate.")
        st.stop()

    # Prepare X and y
    y = df["y"].astype(str).str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        st.error(" Column `y` must contain only `yes` or `no` values.")
        st.stop()

    X = df.drop(columns=["y"])

    # Predict
    try:
        preds = model.predict(X)
    except Exception as e:
        st.error(
            " Prediction failed. This usually happens if the uploaded CSV columns don't match training columns.\n\n"
            f"Error: {e}"
        )
        st.stop()

    # Confusion matrix plot
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)

    # Classification report text
    st.text("Classification Report:")
    st.text(classification_report(y, preds, zero_division=0))
