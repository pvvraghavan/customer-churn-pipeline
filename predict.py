from pathlib import Path

import pandas as pd
import skops.io as sio

MODEL_PATH = Path("artifacts/churn_pipeline.skops")
RISK_THRESHOLD = 0.35
MODEL_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "contract_type",
    "region",
    "autopay",
]

def main() -> None:
    unknown_types = sio.get_untrusted_types(file=MODEL_PATH)
    print(f"Reviewed untrusted types: {unknown_types}")

    model = sio.load(MODEL_PATH, trusted=unknown_types)
    customer_ids = ["CUST-NEW-001", "CUST-NEW-002", "CUST-NEW-003"]
    customers = pd.DataFrame(
        [
            {
                "tenure_months": 5,
                "monthly_charges": 109.50,
                "support_tickets": 3,
                "contract_type": "month_to_month",
                "region": "north",
                "autopay": "no",
            },
            {
                "tenure_months": 48,
                "monthly_charges": 64.00,
                "support_tickets": 0,
                "contract_type": "two_year",
                "region": "west",
                "autopay": "yes",
            },
            {
                "tenure_months": 18,
                "monthly_charges": 82.75,
                "support_tickets": 2,
                "contract_type": "one_year",
                "region": "central",
                "autopay": "no",
            },
        ]
    )

    probabilities = model.predict_proba(customers[MODEL_FEATURES])[:,1]
    ranked_customers = customers
    ranked_customers["customer_id"] = customer_ids
    ranked_customers["churn_probability"] = probabilities
    ranked_customers["risk_label"] = [
        "high" if probability >= RISK_THRESHOLD else "standard"
        for probability in probabilities
    ]
    ranked_customers = ranked_customers.sort_values(
        by="churn_probability",
        ascending=False
    )
 
    print(
        ranked_customers[
            ["customer_id", "region", "churn_probability", "risk_label"]
        ]
    )


if __name__ == "__main__":
    main()