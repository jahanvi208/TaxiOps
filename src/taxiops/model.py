"""Reproducible sparse preprocessing + XGBoost regression model."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from taxiops.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_pipeline(params: dict, seed: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        sparse_threshold=1.0,
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=2,
        random_state=seed,
        **params,
    )
    return Pipeline([("features", preprocessor), ("xgb", model)])
