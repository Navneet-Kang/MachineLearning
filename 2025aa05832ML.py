import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)



# XGBoost (install: pip install xgboost)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False



def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing: impute, one-hot encode categoricals, scale numerics."""
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor




def evaluate_model(model, X_test, y_test) -> dict:
    """Compute required metrics for binary classification."""
    y_pred = model.predict(X_test)

    # Get probability scores for AUC if possible
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # decision_function can be used for AUC
        scores = model.decision_function(X_test)
        # normalize to 0..1 (not required, but keeps it stable)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        y_proba = None

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
    }

    if y_proba is not None:
        metrics["AUC"] = float(roc_auc_score(y_test, y_proba))
    else:
        metrics["AUC"] = np.nan

    return metrics

def main():
    # ====== CHANGE THIS PATH IF NEEDED ======
    csv_path = "bank.csv"
    # =======================================

    df = pd.read_csv(uploaded_file, sep=None, engine='python')

    if "y" not in df.columns:
        raise ValueError("Target column 'y' not found. Please rename your target column to 'y'.")

    # Convert target to 0/1
    y = df["y"].astype(str).str.lower().map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("Target 'y' must contain only 'yes'/'no' values.")

    X = df.drop(columns=["y"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        # GaussianNB needs dense input; we'll handle via a special pipeline below
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss"
        )
    else:
        print("xgboost not installed. Install with: pip install xgboost")
        print("   Skipping XGBoost training.")

    os.makedirs("model", exist_ok=True)

    results = []

    for name, clf in models.items():
        if "Naive Bayes" in name:
            # GaussianNB expects dense arrays; convert sparse to dense using a small wrapper
            pipeline = Pipeline(steps=[
                ("prep", preprocessor),
                ("to_dense", FunctionTransformerDense()),
                ("model", clf)
            ])
        else:
            pipeline = Pipeline(steps=[
                ("prep", preprocessor),
                ("model", clf)
            ])

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)

        # Save model
        joblib.dump(pipeline, f"model/{safe_filename(name)}.pkl")
        print(f" Trained & saved: {name}")

    results_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ].sort_values(by="AUC", ascending=False)

    print("\n===== METRICS COMPARISON TABLE =====")
    print(results_df.to_string(index=False))

    # Print confusion matrix + report for best model
    best_model_name = results_df.iloc[0]["Model"]
    best_model = joblib.load(f"model/{safe_filename(best_model_name)}.pkl")

    y_pred_best = best_model.predict(X_test)
    print(f"\n===== BEST MODEL: {best_model_name} =====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_best))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, zero_division=0))

    results_df.to_csv("model/model_metrics.csv", index=False)
    print("\n Saved metrics table to: model/model_metrics.csv")




def safe_filename(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
    )


class FunctionTransformerDense:
    """Turns sparse matrix into dense for GaussianNB."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # If already dense, keep it
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.array(X)


if __name__ == "__main__":

    main()

