from math import exp
from pathlib import Path
from random import Random
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 2500
DATA_PATH = Path("data/churn_customers.csv")

 
def main() -> None:
    rng = Random(RANDOM_SEED)
    rows: list[dict[str,object]] = []

    contract_effects = {
        "month_to_month" : 2.2,
        "one_year" : -0.4,
        "two_year" : -0.9,
    }

    region_effects = {
        "north" : 0.7,
        "south" : 0.2,
        "east" : 0.0,
        "west" : -0.5,
    }

    for customer_number in range(1, N_CUSTOMERS+1):
        tenure_months = rng.randint(1,72)
        monthly_charges = round(max(20.0, min(130.0,rng.gauss(75.0,22.0))),2)
        support_tickets = min(6, int(rng.expovariate(0.9)))
        contract_type = rng.choices(
            ["month_to_month", "one_year", "two_year"],
            weights = [0.55,0.25,0.20],
            k=1,
        )[0]
        region = rng.choice(["north", "south", "east", "west"])
        autopay = rng.choice(["yes","no"])

        logit = (
            -2.5 
            + contract_effects[contract_type]
            + (1.3 if autopay == "no" else -0.2)
            + region_effects[region]
            + 0.45 * support_tickets
            + 0.006 * (monthly_charges - 75.0)
            - 0.015 * tenure_months
        )
        probability = 1.0 / (1.0 + exp(-logit))
        churn = int(rng.random() < probability)

        rows.append(
            {
                "customer_id" : f"CUST-{customer_number:04d}",
                "tenure_months" : tenure_months,
                "monthly_charges": (
                    None if rng.random() < 0.04 else monthly_charges
                ),
                "support_tickets" : support_tickets,
                "contract_type" : contract_type,
                "region" : None if rng.random() < 0.03 else region,
                "autopay" : autopay,
                "churn" : churn,
            }
        )

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    customer_data = pd.DataFrame(rows)
    customer_data.to_csv(DATA_PATH,index=False)

    print(f"Generated {len(customer_data):,} customer records")
    print(f"Saved dataset: {DATA_PATH}")
    print("Sample records:")
    print(*rows[:3], sep="\n")

if __name__ == "__main__":
    main()