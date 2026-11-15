"""Actual XGBoost fit/predict smoke test; synthetic fixture, NOT a TLC benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxiops.config import FEATURE_COLUMNS
from taxiops.metrics import rmse
from taxiops.model import make_pipeline


def test_real_xgboost_pipeline_beats_train_mean_on_synthetic_fixture():
    rng = np.random.default_rng(2026)
    count = 260
    distance = rng.uniform(0.3, 15, size=count)
    zone = rng.integers(1, 15, size=count)
    data = pd.DataFrame(
        {
            "trip_distance": distance,
            "passenger_count": rng.integers(1, 5, size=count),
            "pickup_zone_id": zone,
            "dropoff_zone_id": rng.integers(1, 15, size=count),
            "pickup_hour": rng.integers(0, 24, size=count),
            "pickup_weekday": rng.integers(0, 7, size=count),
            "pickup_month": rng.integers(1, 3, size=count),
        }
    )
    target = 4.5 + 2.4 * distance + rng.normal(0, 0.25, count)
    train_x, val_x = data.loc[:209, FEATURE_COLUMNS], data.loc[210:, FEATURE_COLUMNS]
    train_y, val_y = target[:210], target[210:]
    pipeline = make_pipeline(
        {"n_estimators": 45, "max_depth": 3, "learning_rate": 0.15}, seed=2026
    )
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(val_x)
    assert len(predictions) == len(val_y)
    assert np.isfinite(predictions).all()
    assert rmse(val_y, predictions) < rmse(val_y, np.full_like(val_y, train_y.mean()))
