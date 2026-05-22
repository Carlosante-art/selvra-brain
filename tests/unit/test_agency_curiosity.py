"""Tester för CuriosityDriver (AE-1)."""

from __future__ import annotations

import math

import pytest

from selvra_brain.agency import CuriosityDriver
from selvra_brain.agency.types import ActionType, GoalType
from selvra_brain.core.epistemic import (
    Confidence,
    DataType,
    EpistemicTag,
    MemoryType,
    Mutability,
    Persistence,
    Valence,
)
from selvra_brain.core.events import BrainEvent, EventCategory, EventStore


def _add_prediction_error(
    store: EventStore,
    *,
    source: str,
    normalized_magnitude: float = 0.5,
    valence: Valence = Valence.NEGATIVE,
) -> None:
    store.append(
        BrainEvent(
            category=EventCategory.PREDICTION_ERROR,
            event_type="surprise",
            payload={"source": source, "normalized_magnitude": normalized_magnitude},
            tag=EpistemicTag(
                data_type=DataType.DERIVED,
                confidence=Confidence.MEDIUM,
                mutability=Mutability.IMMUTABLE,
                persistence=Persistence.TRANSIENT,
                memory_type=MemoryType.WORKING,
                valence=valence,
            ),
        )
    )


def test_idle_when_no_signals():
    store = EventStore()
    driver = CuriosityDriver(store=store)
    intent = driver.decide(source_positions={"a": 0.0}, reliability_per_source={"a": 1.0})
    # Allt fullt pålitligt + inga errors → drive nära 0
    assert intent.action == ActionType.IDLE
    assert intent.goal == GoalType.NONE


def test_low_reliability_triggers_reduce_uncertainty():
    store = EventStore()
    driver = CuriosityDriver(store=store)
    intent = driver.decide(
        source_positions={"a": 0.5, "b": 1.0},
        reliability_per_source={"a": 0.1, "b": 0.95},
    )
    assert intent.action == ActionType.LOOK_AT
    assert intent.target_object_id == "a"
    assert intent.goal == GoalType.REDUCE_UNCERTAINTY


def test_recent_surprise_triggers_investigate():
    store = EventStore()
    driver = CuriosityDriver(store=store)
    _add_prediction_error(store, source="b", normalized_magnitude=0.6)
    _add_prediction_error(store, source="b", normalized_magnitude=0.7)
    intent = driver.decide(
        source_positions={"a": 0.0, "b": 1.0},
        reliability_per_source={"a": 1.0, "b": 1.0},
    )
    assert intent.target_object_id == "b"
    assert intent.goal == GoalType.INVESTIGATE_SURPRISE


def test_explore_neglected_after_imbalance():
    # Lägre drive_threshold så att explore-only (utan surprise/uncertainty)
    # kan trigga action. I full demo kombineras alla drives — då räcker
    # default-tröskeln. Här isolerar vi explore.
    store = EventStore()
    driver = CuriosityDriver(store=store, drive_threshold=0.05)
    for _ in range(50):
        driver.record_observation("a")
    for _ in range(2):
        driver.record_observation("b")
    intent = driver.decide(
        source_positions={"a": 0.0, "b": 1.0},
        reliability_per_source={"a": 1.0, "b": 1.0},
    )
    assert intent.target_object_id == "b"


def test_emits_action_event():
    store = EventStore()
    driver = CuriosityDriver(store=store)
    driver.decide(source_positions={"a": 0.0}, reliability_per_source={"a": 0.2})
    action_events = store.by_category(EventCategory.ACTION)
    assert len(action_events) == 1
    assert action_events[0].payload["goal"] == GoalType.REDUCE_UNCERTAINTY.value


def test_target_angle_matches_position():
    store = EventStore()
    driver = CuriosityDriver(store=store)
    intent = driver.decide(
        source_positions={"a": math.pi / 4},
        reliability_per_source={"a": 0.1},
    )
    assert intent.target_angle == pytest.approx(math.pi / 4)


def test_drive_strength_bounded():
    """drive_strength alltid [0, 1]."""
    store = EventStore()
    driver = CuriosityDriver(store=store)
    for _ in range(50):
        _add_prediction_error(store, source="a", normalized_magnitude=0.99)
    intent = driver.decide(
        source_positions={"a": 0.0},
        reliability_per_source={"a": 0.0},
    )
    assert 0.0 <= intent.drive_strength <= 1.0


def test_reasoning_present_when_acting():
    store = EventStore()
    driver = CuriosityDriver(store=store)
    intent = driver.decide(
        source_positions={"a": 0.0},
        reliability_per_source={"a": 0.1},
    )
    assert intent.reasoning  # not empty
    assert any("reliability" in r for r in intent.reasoning)
