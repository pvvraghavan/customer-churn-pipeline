from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import skops.io as sio
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("artifacts/churn_pipeline.skops")
RISK_THRESHOLD = 0.35
model_store = {}

class CustomerRequest(BaseModel):
    # Match the six raw fields accepted by the pipeline
    tenure_months: int
    monthly_charges: float | None = None
    support_tickets: int
    contract_type: str
    region: str | None = None
    autopay: str


class PredictionResponse(BaseModel):
    # Return the probability and operational decision
    churn_probability: float
    risk_label: str


@asynccontextmanager
async def lifespan(app: FastAPI):

    unknown_types = sio.get_untrusted_types(file=MODEL_PATH)
    model_store["pipeline"] = sio.load(MODEL_PATH, trusted=unknown_types)
    yield

    model_store.clear()

app = FastAPI(
    title="Customer Churn API",
    lifespan=lifespan,
)

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerRequest) -> PredictionResponse:
    customer_frame = pd.DataFrame([customer.model_dump()])
    model = model_store["pipeline"]

    probability = float(
        model.predict_proba(customer_frame)[0,1]
    )

    risk_label = (
        "high" if probability >= RISK_THRESHOLD else "standard"
    )

    return PredictionResponse(
        churn_probability=probability,
        risk_label=risk_label,
    )