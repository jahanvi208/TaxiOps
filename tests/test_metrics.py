import pytest

from taxiops.metrics import regression_passes, rmse


def test_rmse_is_zero_for_perfect_predictions():
    assert rmse([1, 2, 3], [1, 2, 3]) == pytest.approx(0)


def test_regression_gate_rejects_worse_model():
    assert not regression_passes(10.01, 10.0)


def test_regression_gate_accepts_better_model():
    assert regression_passes(9.99, 10.0)


def test_regression_gate_accepts_equal_model():
    assert regression_passes(10, 10)


def test_invalid_rmse_rejected():
    with pytest.raises(ValueError):
        regression_passes(float("nan"), 10)


def test_invalid_tolerance_rejected():
    with pytest.raises(ValueError):
        regression_passes(9, 10, -0.01)
