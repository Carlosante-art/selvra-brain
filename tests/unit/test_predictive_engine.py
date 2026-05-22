"""Tests för HierarchicalPredictiveEngine — multi-level PP-1."""

from __future__ import annotations

import pytest

from selvra_brain.core.events import EventCategory, EventStore
from selvra_brain.prediction.engine import (
    HierarchicalPredictiveEngine,
    _confidence_to_enum,
    _error_valence,
)
from selvra_brain.core.epistemic import Confidence, Valence


# ─── _confidence_to_enum ─────────────────────────────────────────


def test_confidence_to_enum_buckets() -> None:
    assert _confidence_to_enum(0.9) == Confidence.HIGH
    assert _confidence_to_enum(0.5) == Confidence.MEDIUM
    assert _confidence_to_enum(0.2) == Confidence.LOW
    assert _confidence_to_enum(0.05) == Confidence.UNAVAILABLE


# ─── _error_valence ──────────────────────────────────────────────


def test_error_valence_small_is_neutral() -> None:
    assert _error_valence(0.05) == Valence.NEUTRAL


def test_error_valence_medium_is_negative() -> None:
    assert _error_valence(0.3) == Valence.NEGATIVE


def test_error_valence_large_is_negative_strong() -> None:
    assert _error_valence(2.0) == Valence.NEGATIVE_STRONG


# ─── Engine basic ────────────────────────────────────────────────


def test_engine_requires_at_least_one_level() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        HierarchicalPredictiveEngine(EventStore(), levels=0)


def test_first_observation_no_error() -> None:
    """Första observation har ingen prior prediction → ingen error-event."""
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=2)
    result = engine.observe(source="temp", value=20.0)
    assert result.level_0_error is None
    assert result.level_0_new_prediction is not None
    # Bara PERCEPTION + PREDICTION (för nästa)
    assert len(store.by_category(EventCategory.PERCEPTION)) == 1
    assert len(store.by_category(EventCategory.PREDICTION)) >= 1
    assert len(store.by_category(EventCategory.PREDICTION_ERROR)) == 0


def test_second_observation_generates_error() -> None:
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=1)
    engine.observe(source="temp", value=20.0)
    result = engine.observe(source="temp", value=25.0)
    assert result.level_0_error is not None
    assert result.level_0_error.magnitude > 0
    # Nu finns PREDICTION_ERROR-event
    assert len(store.by_category(EventCategory.PREDICTION_ERROR)) >= 1


def test_stable_signal_low_error() -> None:
    """Predikterbar input → låg error över tid."""
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=1)
    # Konstant 10.0 — LinearTrend lär sig delta=0 snabbt
    for _ in range(5):
        engine.observe(source="temp", value=10.0)
    # Senaste error ska vara liten (engine har lärt sig)
    errors = store.by_category(EventCategory.PREDICTION_ERROR)
    last_error = errors[-1]
    assert last_error.payload["magnitude"] < 1.0


def test_surprise_generates_large_error() -> None:
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=1)
    # Etablera stabil signal
    for _ in range(5):
        engine.observe(source="temp", value=10.0)
    # Surprise!
    result = engine.observe(source="temp", value=100.0)
    assert result.level_0_error is not None
    assert result.level_0_error.magnitude > 50
    # normalized_magnitude → nära 1
    assert result.level_0_error.normalized_magnitude > 0.9


def test_payload_includes_normalized_magnitude() -> None:
    """PREDICTION_ERROR-event har normalized_magnitude i payload —
    detta är vad visualization läser."""
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=1)
    engine.observe(source="temp", value=10.0)
    engine.observe(source="temp", value=50.0)

    errors = store.by_category(EventCategory.PREDICTION_ERROR)
    assert len(errors) >= 1
    payload = errors[-1].payload
    assert "normalized_magnitude" in payload
    assert "magnitude" in payload
    assert "signed" in payload
    assert "predicted" in payload
    assert "observed" in payload
    assert 0.0 <= payload["normalized_magnitude"] <= 1.0


def test_multi_level_engine_generates_l1_events() -> None:
    """Level 1 predictor ska reagera på Level 0 errors."""
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=2)
    # Behöver minst 3 observations för level-1 att se 2 errors
    for v in [10.0, 50.0, 100.0, 30.0]:
        engine.observe(source="temp", value=v)

    # Vi ska ha både l0 och l1 prediction-error-events
    errors = store.by_category(EventCategory.PREDICTION_ERROR)
    levels_seen = {e.payload["level"] for e in errors}
    assert 0 in levels_seen
    assert 1 in levels_seen


def test_separate_sources_have_independent_predictors() -> None:
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=1)

    engine.observe(source="temp", value=10.0)
    engine.observe(source="temp", value=10.0)
    engine.observe(source="humidity", value=80.0)

    # Predictorerna är separata
    temp_predictors = engine.predictors_for_source("temp")
    humidity_predictors = engine.predictors_for_source("humidity")
    assert 0 in temp_predictors
    assert 0 in humidity_predictors
    # De har sett olika antal observations
    assert temp_predictors[0].observation_count == 2
    assert humidity_predictors[0].observation_count == 1


def test_event_emission_order() -> None:
    """Per observation cycle: PERCEPTION → (PREDICTION_ERROR) → PREDICTION."""
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=1)
    engine.observe(source="temp", value=10.0)
    # Nu finns PERCEPTION + PREDICTION (men ingen error än)
    engine.observe(source="temp", value=20.0)

    all_events = store.all_events()
    # Hitta sista PERCEPTION och kolla att PREDICTION_ERROR kom efter
    perception_idx = None
    error_idx = None
    for i, e in enumerate(all_events):
        if e.category == EventCategory.PERCEPTION and e.payload.get("value") == 20.0:
            perception_idx = i
        if e.category == EventCategory.PREDICTION_ERROR and error_idx is None and perception_idx is not None:
            error_idx = i
    assert perception_idx is not None
    assert error_idx is not None
    assert error_idx > perception_idx
