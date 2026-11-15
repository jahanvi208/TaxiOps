"""RMSE regression comparison utilities."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import root_mean_squared_error


def rmse(actual, predicted) -> float:
    return float(root_mean_squared_error(actual, predicted))


def regression_passes(candidate_rmse: float, champion_rmse: float, tolerance: float = 0.0) -> bool:
    """Allow a tolerance only if explicitly opted in (default: zero regression)."""
    if not np.isfinite([candidate_rmse, champion_rmse]).all():
        raise ValueError("RMSE must be finite")
    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative")
    return candidate_rmse <= champion_rmse * (1.0 + tolerance)
