from pathlib import Path

import pandas as pd
import skops.io as sio
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline import NUMERIC_FEATURES, build_baseline_pipeline

DATA_PATH = Path("data/churn_customers.csv")
MODEL_PATH = Path("artifacts/churn_pipeline.skops")
RANDOM_STATE = 42
CATEGORICAL_FEATURES = ["contract_type", "region", "autopay"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def build_full_pipeline() -> Pipeline:
    numeric_pipeline  = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical",categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor",preprocessor),
            ("classifier",LogisticRegression(max_iter=1000)),
        ]
    )

def recall_at_threshold(actual_values,probabilities,threshold:float) -> float:
    actual_labels = [int(value) for value in actual_values]
    predicted_labels = [int(value >= threshold) for value in probabilities]
    true_positives = sum(
        actual == 1 and predicted == 1 for actual, predicted in zip(actual_labels, predicted_labels)
    )
    positive_count = sum(actual_labels)
    return true_positives/positive_count

def main() -> None:
    customer_data = pd.read_csv(DATA_PATH)
    X = customer_data[MODEL_FEATURES]
    y = customer_data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    baseline = build_baseline_pipeline()
    baseline.fit(X_train[NUMERIC_FEATURES], y_train)
    baseline_probabilities = baseline.predict_proba(X_test[NUMERIC_FEATURES])[:,1]
    baseline_auc = roc_auc_score(y_test, baseline_probabilities)

    full_pipeline = build_full_pipeline()
    cross_validation = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    cv_scores = cross_val_score(
        full_pipeline,
        X_train,
        y_train,
        cv=cross_validation,
        scoring="roc_auc",
    )
    full_pipeline.fit(X_train,y_train)
    holdout_probabilities = full_pipeline.predict_proba(X_test)[:,1]
    holdout_auc = roc_auc_score(y_test, holdout_probabilities)

    recall_default = recall_at_threshold(
        y_test, holdout_probabilities, threshold=0.5
    )

    recall_retention = recall_at_threshold(
        y_test, holdout_probabilities, threshold=0.35
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(full_pipeline, MODEL_PATH)

    print(f"Baseline holdout ROC AUC: {baseline_auc:.3f}")
    print(
        "Full pipeline 5-fold ROC AUC: "
        f"{cv_scores.mean():.3f} +/- {cv_scores.std():.3f}"
    )
    print(f"Full pipeline holdout ROC AUC: {holdout_auc:.3f}")
    print(f"ROC AUC improvement: {holdout_auc - baseline_auc:+.3f}")
    print(f"Recall at threshold 0.50: {recall_default:.3f}")
    print(f"Recall at threshold 0.35: {recall_retention:.3f}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()