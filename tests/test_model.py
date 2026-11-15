from __future__ import annotations

import pandas as pd

from taxiops.config import FEATURE_COLUMNS
from taxiops.model import make_pipeline


def test_pipeline_end_to_end_small_fixture():
    # In-memory TEST DATA only; never substituted for TLC training/validation data.
    sample = pd.DataFrame(
        [
            [1.0 + i * 0.2, 1, 161, 237, 9 + i % 3, i % 7, 1]
            for i in range(18)
        ],
        columns=FEATURE_COLUMNS,
    )
    target = pd.Series([8.0 + i for i in range(len(sample))])
    model = make_pipeline({"n_estimators": 5, "max_depth": 2})
    model.fit(sample, target)
    assert len(model.predict(sample.head(3))) == 3
