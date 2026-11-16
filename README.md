# Customer Churn Pipeline

An end-to-end, containerised churn-prediction system: train and compare two models (scikit-learn and PyTorch), track every run in MLflow, and serve the selected pipeline through a FastAPI service running in Docker.

> Built on a **synthetic dataset** (2,500 generated customers) as a hands-on MLOps project, following a guided project brief. The goal is the engineering workflow, not a real-world benchmark.

## What it does

- **Numeric-only baseline** (logistic regression) deliberately ignores contract, region and autopay, to expose the signal it misses.
- **Full pipeline**: `ColumnTransformer` (median impute + scale for numeric fields, most-frequent impute + one-hot for categorical fields) feeding logistic regression, evaluated with stratified 5-fold CV so preprocessing is fitted inside each fold (no leakage).
- **PyTorch challenger**: a small two-layer network (16 hidden units) trained on the same transformed features and holdout rows.
- **MLflow tracking**: parameters, metrics and artifacts for both runs in a local SQLite store, compared side by side.
- **Model persistence with skops**: the fitted pipeline is saved and reloaded after an explicit untrusted-type review, instead of unpickling blindly.
- **FastAPI service**: Pydantic-validated request/response, model loaded once via a lifespan handler, `POST /predict` returns churn probability and a risk label.
- **Docker image** containing only the serving path (no PyTorch, no MLflow data).
- **Batch risk ranking**: scores several customers at once, ranks by churn probability, and handles an unseen category (`region="central"`) without retraining, via `OneHotEncoder(handle_unknown="ignore")`.

## Results (seed 42, 80/20 stratified split)

| Model | Holdout ROC AUC |
|---|---|
| Numeric-only baseline (logistic regression) | 0.596 |
| Full pipeline, 5-fold CV on train set | 0.827 ± 0.015 |
| **Full pipeline, holdout** | **0.852** |
| PyTorch challenger, holdout | 0.849 |

Recall of actual churners on the holdout: **0.642** at threshold 0.50, **0.753** at threshold 0.35 (the retention threshold used by the API). The scikit-learn pipeline is served because it matches the neural network's accuracy on this small tabular dataset while keeping preprocessing and prediction in one reviewed artifact.

## Project structure

```
generate_data.py       deterministic synthetic customer data (seeded)
baseline.py            numeric-only baseline
train_model.py         full pipeline, CV, holdout eval, saves .skops artifact
predict.py             standalone batch scoring and risk ranking
track_experiments.py   logs the scikit-learn run to MLflow
torch_challenger.py    PyTorch model, logged to MLflow
app.py                 FastAPI service
Dockerfile             serving image
requirements.txt       pinned dependencies
```

## Run it

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python generate_data.py
python baseline.py
python train_model.py
python predict.py
python track_experiments.py
python torch_challenger.py
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000   # http://127.0.0.1:5000
```

Serve locally:

```powershell
uvicorn app:app --reload        # http://127.0.0.1:8000/docs
```

Or in Docker:

```powershell
docker build -t customer-churn-api:latest .
docker run -d --name customer-churn-api -p 127.0.0.1:8000:8000 customer-churn-api:latest
```

Example request:

```json
{
  "tenure_months": 5,
  "monthly_charges": 109.5,
  "support_tickets": 3,
  "contract_type": "month_to_month",
  "region": "north",
  "autopay": "no"
}
```

Response: `{"churn_probability": <0-1>, "risk_label": "high" | "standard"}`

## Limitations and next steps

- Synthetic data generated from a known logistic formula, so logistic regression is close to the true model; real churn data will be messier and imbalanced.
- Single holdout split and one seed; repeated CV or multiple seeds would give uncertainty on the scikit-learn vs PyTorch comparison.
- No probability calibration, no gradient-boosted baseline (e.g. LightGBM), no drift monitoring.
- Natural extensions: a `/predict/batch` endpoint, MLflow Model Registry, CI that builds the image and runs tests, `docker compose` for API + MLflow.

## Stack

Python · pandas · scikit-learn · PyTorch · MLflow · skops · FastAPI · Pydantic · Uvicorn · Docker
