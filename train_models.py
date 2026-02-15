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
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

print("Loading dataset...")

df = pd.read_csv("data/bank.csv", sep=";")
# Target conversion
y = df["y"].map({"yes":1,"no":0})
X = df.drop("y", axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

os.makedirs("model", exist_ok=True)

results = []


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Converts sparse matrix to dense array (needed for GaussianNB)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.array(X)


for name, model in models.items():
    print("Training:", name)

    if name == "Naive Bayes (Gaussian)":
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Convert to dense ONLY if sparse
        if hasattr(X_train_processed, "toarray"):
            X_train_processed = X_train_processed.toarray()
            X_test_processed = X_test_processed.toarray()

        model.fit(X_train_processed, y_train)
        preds = model.predict(X_test_processed)
        proba = model.predict_proba(X_test_processed)[:, 1]

        # Save as a real pipeline (so Streamlit can load .predict on raw df)
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("to_dense", DenseTransformer()),
            ("model", model)
        ])

    else:
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "MCC": matthews_corrcoef(y_test, preds)
    })

    joblib.dump(pipeline, f"model/{name.lower().replace(' ','_').replace('(','').replace(')','')}.pkl")

pd.DataFrame(results).to_csv("model/model_metrics.csv", index=False)

print("Training complete. Models saved in /model folder.")
