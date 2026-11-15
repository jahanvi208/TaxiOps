"""Real month-to-month distribution comparisons; descriptive, not simulated drift."""
from __future__ import annotations

import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


def measure_drift(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """Report KS and Jensen-Shannon *distance* without making unsupported causal claims."""
    numeric = {}
    for key in ("trip_distance", "fare_amount"):
        result = ks_2samp(train[key].to_numpy(), validation[key].to_numpy())
        numeric[key] = {
            "train_mean": float(train[key].mean()),
            "validation_mean": float(validation[key].mean()),
            "ks_statistic": float(result.statistic),
            "ks_pvalue": float(result.pvalue),
        }
    categorical = {}
    for key in ("pickup_zone_id", "pickup_hour"):
        categories = sorted(set(train[key].unique()) | set(validation[key].unique()))
        a = train[key].value_counts(normalize=True).reindex(categories, fill_value=0)
        b = validation[key].value_counts(normalize=True).reindex(categories, fill_value=0)
        categorical[key] = {"jensen_shannon_distance": float(jensenshannon(a, b, base=2))}
    return {"numeric": numeric, "categorical": categorical,
            "description": "Observed Jan–Feb distributions; not synthetic drift"}
