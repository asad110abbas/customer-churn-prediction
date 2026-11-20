"""Synthetic subscription-business customers with a realistic, explainable churn signal."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["tenure_months", "monthly_charges", "support_tickets", "avg_session_minutes", "late_payments", "discount_pct"]
CATEGORICAL = ["contract", "payment_method", "plan", "region"]


def make_customers(n: int = 8000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Monthly", "Annual", "Two-year"], n, p=[0.55, 0.3, 0.15])
    plan = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.4, 0.4, 0.2])
    df = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(n)],
        "tenure_months": rng.integers(1, 72, n),
        "contract": contract,
        "plan": plan,
        "payment_method": rng.choice(["Card", "Bank transfer", "Wallet", "Cash"], n, p=[.4, .25, .2, .15]),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "monthly_charges": np.round(np.select([plan == "Basic", plan == "Standard"], [rng.normal(1500, 200, n),
                                     rng.normal(2800, 300, n)], rng.normal(4500, 400, n)), 0),
        "support_tickets": rng.poisson(1.2, n),
        "avg_session_minutes": np.round(rng.gamma(3, 12, n), 1),
        "late_payments": rng.poisson(0.5, n),
        "discount_pct": rng.choice([0, 0, 0, 5, 10, 20], n),
    })
    logit = (-1.2
             + 1.4 * (df.contract == "Monthly") - 0.8 * (df.contract == "Two-year")
             - 0.04 * df.tenure_months
             + 0.35 * df.support_tickets
             + 0.45 * df.late_payments
             - 0.02 * df.avg_session_minutes
             + 0.0002 * (df.monthly_charges - 2500)
             + 0.5 * (df.payment_method == "Cash")
             - 0.03 * df.discount_pct)
    p = 1 / (1 + np.exp(-logit))
    df["churned"] = (rng.random(n) < p).astype(int)
    return df
