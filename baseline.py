from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/churn_customers.csv")
RANDOM_STATE = 42
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
]

IGNORED_FEATURES = [
    "contract_type","region","autopay"
]
 
def build_baseline_pipeline() -> Pipeline:
    return Pipeline(

        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
            ("classifier",LogisticRegression(max_iter=1000)),
        ]
        
    )

def main() -> None:
    customer_data = pd.read_csv(DATA_PATH)
    X = customer_data[NUMERIC_FEATURES]
    y = customer_data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    baseline = build_baseline_pipeline()
    baseline.fit(X_train,y_train)
    probabilities = baseline.predict_proba(X_test)[:,1]
    baseline_auc = roc_auc_score(y_test,probabilities)

    print(f"Numeric-only holdout ROC AUC: {baseline_auc:.3f}")
    print(f"Ignored categorical fields: {', '.join(IGNORED_FEATURES)}")
    print("Obstacle: the baseline cannot use contract, region, or autopay signal.")


if __name__ == "__main__":
    main()