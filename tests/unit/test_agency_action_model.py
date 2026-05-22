"""Tester för ActionEffectModel (AE-2)."""

from __future__ import annotations

import math

import pytest

from selvra_brain.agency import ActionEffectModel
from selvra_brain.core.events import EventCategory, EventStore


def test_predict_in_focus_returns_full_intensity():
    store = EventStore()
    model = ActionEffectModel(store=store)
    # Target på samma vinkel som source → in-focus
    pred = model.predict_effect(target_angle=0.0, source_positions={"a": 0.0})
    assert pred.predicted_intensities["a"] == pytest.approx(1.0)


def test_predict_in_periphery_returns_low_intensity():
    store = EventStore()
    model = ActionEffectModel(store=store)
    # Target på 0, source på π (motsatt sida) → periphery
    pred = model.predict_effect(target_angle=0.0, source_positions={"a": math.pi})
    assert pred.predicted_intensities["a"] <= 0.3


def test_observe_effect_returns_errors():
    store = EventStore()
    model = ActionEffectModel(store=store)
    model.predict_effect(target_angle=0.0, source_positions={"a": 0.0, "b": math.pi})
    # Faktiska intensities: a är i fokus (1.0), b är periphery (0.2)
    errors = model.observe_effect(actual_intensities={"a": 1.0, "b": 0.2})
    # Med initial guess matchar exakt → låga errors
    err_dict = {e.source: e.error_magnitude for e in errors}
    assert err_dict["a"] < 0.1
    assert err_dict["b"] < 0.1


def test_observe_without_prediction_returns_empty():
    store = EventStore()
    model = ActionEffectModel(store=store)
    errors = model.observe_effect(actual_intensities={"a": 1.0})
    assert errors == ()


def test_learning_adjusts_focus_width():
    """Om actual_intensity i fokus < 1, modellen sänker focus_width."""
    store = EventStore()
    model = ActionEffectModel(store=store, learning_rate=0.5)
    initial_width = model.estimated_focus_width
    for _ in range(10):
        model.predict_effect(target_angle=0.0, source_positions={"a": 0.0})
        # Actual intensity i centrum ska vara 1.0, men vi simulerar 0.7
        # → focus_width borde sjunka
        model.observe_effect(actual_intensities={"a": 0.7})
    assert model.estimated_focus_width < initial_width


def test_emits_prediction_and_effect_events():
    store = EventStore()
    model = ActionEffectModel(store=store)
    model.predict_effect(target_angle=0.0, source_positions={"a": 0.0})
    model.observe_effect(actual_intensities={"a": 1.0})
    pred_events = [
        e for e in store.by_category(EventCategory.PREDICTION)
        if e.event_type == "action_effect_prediction"
    ]
    effect_events = store.by_category(EventCategory.ACTION_EFFECT)
    assert len(pred_events) == 1
    assert len(effect_events) == 1


def test_last_error_magnitude_updates():
    store = EventStore()
    model = ActionEffectModel(store=store)
    assert model.last_error_magnitude == 0.0
    model.predict_effect(target_angle=0.0, source_positions={"a": 0.0})
    model.observe_effect(actual_intensities={"a": 0.5})  # significantly off
    assert model.last_error_magnitude > 0.1


def test_focus_width_bounded():
    """estimated_focus_width never escapes [π/12, π]."""
    store = EventStore()
    model = ActionEffectModel(store=store, learning_rate=10.0)
    for _ in range(50):
        model.predict_effect(target_angle=0.0, source_positions={"a": 0.0})
        model.observe_effect(actual_intensities={"a": 0.0})  # extreme push toward smaller
    assert model.estimated_focus_width >= math.pi / 12
    assert model.estimated_focus_width <= math.pi
