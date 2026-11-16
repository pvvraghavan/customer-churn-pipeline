import mlflow
import pandas as pd
import skops.io as sio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from train_model import (
    DATA_PATH, MODEL_FEATURES, MODEL_PATH, RANDOM_STATE, recall_at_threshold
)

TRACKING_URL = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "customer-churn"
RISK_THRESHOLD = 0.35

def evaluate_saved_model() -> tuple[float,float]:

    customer_data = pd.read_csv(DATA_PATH)
    X = customer_data[MODEL_FEATURES]
    y = customer_data["churn"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y,random_state=RANDOM_STATE,
    )

    unknown_types = sio.get_untrusted_types(file=MODEL_PATH)
    model = sio.load(MODEL_PATH, trusted=unknown_types)
    probabilities = model.predict_proba(X_test)[:,1]

    holdout_auc = roc_auc_score(y_test, probabilities)
    recall = recall_at_threshold(
        y_test, probabilities, threshold=RISK_THRESHOLD,
    ) 
    return holdout_auc, recall

def main() -> None:
    holdout_auc, recall = evaluate_saved_model()
    mlflow.set_tracking_uri(TRACKING_URL)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="sklearn-pipeline") as run:
        mlflow.log_param("model","Logistic Regression")
        mlflow.log_param("risk_threshold", RISK_THRESHOLD)
        mlflow.log_metric("holdout_roc_auc", holdout_auc)
        mlflow.log_metric("recall_at_0_35", recall)
        mlflow.log_artifact(str(MODEL_PATH),artifact_path="model")

    print(f"Tracked run: {run.info.run_id}")
    print(f"Holdout ROC AUC: {holdout_auc:.3f}")
    print(f"Recall at 0.35: {recall:.3f}")

if __name__ == "__main__":
    main()