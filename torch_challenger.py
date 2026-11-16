from pathlib import Path

import mlflow
import pandas as pd
import skops.io as sio
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from train_model import (
    DATA_PATH, MODEL_FEATURES, RANDOM_STATE, build_full_pipeline,
)

MODEL_PATH = Path("artifacts/churn_net.pt")
PREPROCESSOR_PATH = Path("artifacts/torch_preprocessor.skops")
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "customer-churn"


class ChurnNet(nn.Module):
    # Map transformed features to one churn logit
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

def prepare_data():

        customer_data = pd.read_csv(DATA_PATH)
        X = customer_data[MODEL_FEATURES]
        y = customer_data["churn"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
        )
        preprocessor = build_full_pipeline().named_steps["preprocessor"]
        train_values = preprocessor.fit_transform(X_train)
        test_values = preprocessor.transform(X_test)
        if hasattr(train_values, "toarray"):
            train_values = train_values.toarray()
            test_values = test_values.toarray()

        train_features = torch.tensor(train_values, dtype=torch.float32)
        test_features = torch.tensor(test_values, dtype=torch.float32)
        train_targets = torch.tensor(
            y_train.to_numpy(), dtype = torch.float32
        ).unsqueeze(1)

        return train_features, test_features, train_targets, y_test, preprocessor

def train_network(features, targets):

    torch.manual_seed(RANDOM_STATE)
    model = ChurnNet(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.BCEWithLogitsLoss()

    for _ in range(100):
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_function(logits, targets)
        loss.backward()
        optimizer.step()

    return model, float(loss.item())

def evaluate_and_save(model, test_features, y_test, preprocessor):

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(test_features)
        ).squeeze(1).numpy()
    holdout_auc = roc_auc_score(y_test, probabilities)


    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    sio.dump(preprocessor, PREPROCESSOR_PATH)
    return holdout_auc

def main() -> None:
    # Prepare data and train the challenger
    train_x, test_x, train_y, y_test, preprocessor = prepare_data()
    model, final_loss = train_network(train_x, train_y)
    holdout_auc = evaluate_and_save(
        model, test_x, y_test, preprocessor
    )

    # Record the challenger beside the reference run
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="pytorch-challenger") as run:
        mlflow.log_param("hidden_units", 16)
        mlflow.log_param("epochs", 100)
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_metric("final_training_loss", final_loss)
        mlflow.log_metric("holdout_roc_auc", holdout_auc)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(PREPROCESSOR_PATH), artifact_path="model")

    print(f"Tracked run: {run.info.run_id}")
    print(f"PyTorch holdout ROC AUC: {holdout_auc:.3f}")


if __name__ == "__main__":
    main()