"""Tester för PerceptionModule (RPT-1 + RPT-2)."""

from __future__ import annotations

import math

import pytest

from selvra_brain.core.events import EventCategory, EventStore
from selvra_brain.perception.scene import PerceptionModule, SceneRepresentation
from selvra_brain.world.symbol_world import Observation


def _mk_obs(object_id: str, value: float, intensity: float = 1.0, angle: float = 0.0) -> Observation:
    return Observation(
        object_id=object_id,
        signal_value=value,
        signal_type="generic",
        position_angle=angle,
        distance=0.5,
        intensity=intensity,
    )


def test_process_returns_scene_with_features():
    store = EventStore()
    pm = PerceptionModule(store=store)
    scene = pm.process([_mk_obs("a", 1.0), _mk_obs("b", 2.0)], tick=1)
    assert isinstance(scene, SceneRepresentation)
    assert scene.tick == 1
    assert set(scene.object_features.keys()) == {"a", "b"}
    assert scene.object_features["a"].value == 1.0
    assert scene.object_features["b"].value == 2.0


def test_smoothing_blends_previous_value_rpt1():
    """RPT-1: recurrent processing — previous scene påverkar nuvarande."""
    store = EventStore()
    pm = PerceptionModule(store=store, smoothing_weight=0.5)
    pm.process([_mk_obs("a", 10.0)], tick=1)
    scene2 = pm.process([_mk_obs("a", 20.0)], tick=2)
    # 0.5 * 20 + 0.5 * 10 = 15
    assert scene2.object_features["a"].value_smoothed == pytest.approx(15.0)
    # delta = 20 - 10 = 10
    assert scene2.object_features["a"].delta_from_previous == pytest.approx(10.0)


def test_no_previous_means_smoothed_equals_current():
    store = EventStore()
    pm = PerceptionModule(store=store)
    scene = pm.process([_mk_obs("a", 7.0)], tick=1)
    assert scene.object_features["a"].value_smoothed == pytest.approx(7.0)
    assert scene.object_features["a"].delta_from_previous == 0.0


def test_salience_combines_magnitude_intensity_distance():
    """Salience = |value| * intensity * (1 - 0.3*distance)."""
    obs = Observation(
        object_id="x",
        signal_value=-4.0,
        signal_type="g",
        position_angle=0.0,
        distance=0.5,
        intensity=0.8,
    )
    store = EventStore()
    pm = PerceptionModule(store=store)
    scene = pm.process([obs], tick=1)
    expected = 4.0 * 0.8 * (1.0 - 0.3 * 0.5)
    assert scene.salience_distribution["x"] == pytest.approx(expected)


def test_most_salient_resolves():
    store = EventStore()
    pm = PerceptionModule(store=store)
    scene = pm.process(
        [_mk_obs("a", 1.0), _mk_obs("b", 10.0), _mk_obs("c", 3.0)],
        tick=1,
    )
    assert scene.most_salient_id == "b"


def test_recurrent_magnitude_tracks_total_change():
    store = EventStore()
    pm = PerceptionModule(store=store)
    pm.process([_mk_obs("a", 5.0), _mk_obs("b", 0.0)], tick=1)
    scene = pm.process([_mk_obs("a", 8.0), _mk_obs("b", 3.0)], tick=2)
    # |8-5| + |3-0| = 6
    assert scene.recurrent_magnitude == pytest.approx(6.0)


def test_process_emits_scene_integrated_event():
    store = EventStore()
    pm = PerceptionModule(store=store)
    pm.process([_mk_obs("a", 1.0)], tick=1)
    perception_events = store.by_category(EventCategory.PERCEPTION)
    assert any(e.event_type == "scene_integrated" for e in perception_events)


def test_scene_valence_neutral_for_zero_avg():
    store = EventStore()
    pm = PerceptionModule(store=store, valence_scale=0.1)
    scene = pm.process([_mk_obs("a", 0.0)], tick=1)
    assert scene.scene_valence.to_numeric() == 0.0


def test_reset_clears_previous():
    store = EventStore()
    pm = PerceptionModule(store=store)
    pm.process([_mk_obs("a", 5.0)], tick=1)
    assert pm.previous_scene is not None
    pm.reset()
    assert pm.previous_scene is None


def test_salient_objects_filter():
    store = EventStore()
    pm = PerceptionModule(store=store)
    scene = pm.process(
        [_mk_obs("low", 0.1), _mk_obs("high", 10.0), _mk_obs("mid", 1.0)],
        tick=1,
    )
    sal_objs = scene.salient_objects(threshold=0.5)
    assert "high" in sal_objs
    assert "low" not in sal_objs
