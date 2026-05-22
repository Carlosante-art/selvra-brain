"""Tests för Predictor-klasser."""

from __future__ import annotations

import pytest

from selvra_brain.prediction.predictor import (
    ConstantPredictor,
    LinearTrendPredictor,
    MovingAveragePredictor,
    Prediction,
    PredictionError,
    default_predictor,
)


# ─── PredictionError ──────────────────────────────────────────────


def test_error_normalized_magnitude_soft_cap() -> None:
    """x / (x+1) som soft cap: 0 → 0, 1 → 0.5, 100 → 0.99..."""
    err_small = PredictionError(
        source_name="s", predicted=0, observed=0,
        magnitude=0.0, signed=0.0, level=0,
    )
    err_mid = PredictionError(
        source_name="s", predicted=0, observed=1,
        magnitude=1.0, signed=1.0, level=0,
    )
    err_large = PredictionError(
        source_name="s", predicted=0, observed=99,
        magnitude=99.0, signed=99.0, level=0,
    )
    assert err_small.normalized_magnitude == 0.0
    assert err_mid.normalized_magnitude == 0.5
    assert err_large.normalized_magnitude == pytest.approx(0.99, abs=0.01)


def test_error_normalized_magnitude_bounded_0_1() -> None:
    err = PredictionError(
        source_name="s", predicted=0, observed=10000,
        magnitude=10000, signed=10000, level=0,
    )
    assert 0.0 <= err.normalized_magnitude <= 1.0


# ─── ConstantPredictor ───────────────────────────────────────────


def test_constant_predictor_always_returns_constant() -> None:
    p = ConstantPredictor("temp", constant=42.0)
    pred = p.predict()
    assert pred is not None
    assert pred.predicted_value == 42.0


def test_constant_predictor_observation_count_increments() -> None:
    p = ConstantPredictor("temp")
    for v in [1, 2, 3]:
        p.observe_and_predict(v)
    assert p.observation_count == 3


# ─── MovingAveragePredictor ──────────────────────────────────────


def test_moving_avg_cold_start_returns_none() -> None:
    """Innan första update → None (cold start)."""
    p = MovingAveragePredictor("temp", window=5)
    assert p.predict() is None


def test_moving_avg_predicts_after_observations() -> None:
    p = MovingAveragePredictor("temp", window=5)
    for v in [10.0, 12.0, 14.0]:
        p.update(v)
    pred = p.predict()
    assert pred is not None
    assert pred.predicted_value == pytest.approx(12.0)


def test_moving_avg_window_limits_history() -> None:
    """Bara senaste N värdena räknas i snittet."""
    p = MovingAveragePredictor("temp", window=3)
    for v in [100.0, 100.0, 100.0, 10.0, 10.0, 10.0]:
        p.update(v)
    pred = p.predict()
    # Senaste 3 = [10, 10, 10] → snitt 10
    assert pred.predicted_value == pytest.approx(10.0)


def test_moving_avg_confidence_grows_with_history() -> None:
    p = MovingAveragePredictor("temp", window=5)
    confidences: list[float] = []
    for v in [5.0, 5.0, 5.0, 5.0, 5.0]:
        p.update(v)
        pred = p.predict()
        if pred:
            confidences.append(pred.confidence)
    # Confidence ska stiga (history växer, stable källa)
    assert confidences[-1] > confidences[0]


def test_moving_avg_confidence_penalizes_instability() -> None:
    """Stable signal får högre confidence än instabil."""
    p_stable = MovingAveragePredictor("temp", window=5)
    p_unstable = MovingAveragePredictor("temp", window=5)
    for _ in range(5):
        p_stable.update(10.0)
    for v in [0.0, 100.0, 0.0, 100.0, 0.0]:
        p_unstable.update(v)
    assert p_stable.predict().confidence > p_unstable.predict().confidence


def test_moving_avg_observe_and_predict_returns_error() -> None:
    p = MovingAveragePredictor("temp", window=3)
    # Första observation — ingen prior, ingen error
    err1, pred1 = p.observe_and_predict(10.0)
    assert err1 is None
    assert pred1 is not None
    # Andra observation — prior fanns
    err2, pred2 = p.observe_and_predict(15.0)
    assert err2 is not None
    assert err2.magnitude == pytest.approx(5.0)  # predict 10 vs observe 15
    assert err2.signed == pytest.approx(5.0)


# ─── LinearTrendPredictor ────────────────────────────────────────


def test_linear_cold_start_returns_none() -> None:
    p = LinearTrendPredictor("temp")
    assert p.predict() is None


def test_linear_predicts_linear_trend_accurately() -> None:
    """Vid linjär trend ska prediction-error vara nära 0."""
    p = LinearTrendPredictor("temp")
    # Trend: 10, 12, 14, 16, ... (delta = 2)
    for v in [10.0, 12.0, 14.0]:
        p.update(v)
    pred = p.predict()
    assert pred is not None
    # Nästa förväntad: 14 + 2 = 16
    assert pred.predicted_value == pytest.approx(16.0)


def test_linear_returns_zero_delta_on_constant() -> None:
    p = LinearTrendPredictor("temp")
    for v in [5.0, 5.0, 5.0]:
        p.update(v)
    pred = p.predict()
    # Last delta = 0 → predict same value
    assert pred.predicted_value == pytest.approx(5.0)


def test_linear_handles_abrupt_change() -> None:
    p = LinearTrendPredictor("temp")
    p.observe_and_predict(10.0)
    p.observe_and_predict(11.0)
    p.observe_and_predict(12.0)
    # Nu trend (delta=1). Surprise:
    err, _ = p.observe_and_predict(50.0)
    assert err is not None
    # Predikterade 13, observerade 50 → magnitude 37
    assert err.magnitude == pytest.approx(37.0, abs=1.0)


# ─── default_predictor factory ───────────────────────────────────


def test_default_factory_level_0_is_linear() -> None:
    p = default_predictor("temp", level=0)
    assert isinstance(p, LinearTrendPredictor)


def test_default_factory_level_1_is_moving_avg() -> None:
    p = default_predictor("temp", level=1)
    assert isinstance(p, MovingAveragePredictor)


def test_default_factory_preserves_source_and_level() -> None:
    p = default_predictor("my_source", level=2)
    assert p.source_name == "my_source"
    assert p.level == 2
