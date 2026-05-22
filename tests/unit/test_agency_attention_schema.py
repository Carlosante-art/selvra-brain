"""Tester för AttentionSchema (AST-1)."""

from __future__ import annotations

from selvra_brain.agency import AttentionSchema
from selvra_brain.agency.types import ActionIntent, ActionType, GoalType
from selvra_brain.core.events import EventCategory, EventStore


def _intent(target: str | None, goal: GoalType = GoalType.REDUCE_UNCERTAINTY) -> ActionIntent:
    return ActionIntent(
        action=ActionType.LOOK_AT if target else ActionType.IDLE,
        target_angle=0.0,
        target_object_id=target,
        goal=goal,
        drive_strength=0.5,
        reasoning=("test",),
    )


def test_first_update_sets_current_target():
    store = EventStore()
    schema = AttentionSchema(store=store)
    report = schema.update_from_intent(_intent("sun"))
    assert report.target_object_id == "sun"
    assert report.goal_type == GoalType.REDUCE_UNCERTAINTY


def test_duration_increments_when_target_stable():
    store = EventStore()
    schema = AttentionSchema(store=store)
    schema.update_from_intent(_intent("sun"))
    schema.update_from_intent(_intent("sun"))
    report = schema.update_from_intent(_intent("sun"))
    assert report.duration_ticks == 2  # 1st = 0, 2nd = 1, 3rd = 2


def test_duration_resets_on_transition():
    store = EventStore()
    schema = AttentionSchema(store=store)
    schema.update_from_intent(_intent("sun"))
    schema.update_from_intent(_intent("sun"))
    report = schema.update_from_intent(_intent("bird"))
    assert report.duration_ticks == 0
    assert report.target_object_id == "bird"


def test_transitions_counted_in_window():
    store = EventStore()
    schema = AttentionSchema(store=store, transition_window=10)
    schema.update_from_intent(_intent("a"))
    schema.update_from_intent(_intent("b"))
    schema.update_from_intent(_intent("c"))
    report = schema.update_from_intent(_intent("d"))
    # 4 transitions inom fönstret (a, b, c, d → 4 entries)
    assert report.transitions_recent == 4


def test_emits_metacognition_events():
    store = EventStore()
    schema = AttentionSchema(store=store)
    schema.update_from_intent(_intent("sun"))
    meta_events = [
        e for e in store.by_category(EventCategory.METACOGNITION)
        if e.event_type == "attention_schema_report"
    ]
    assert len(meta_events) == 1
    assert meta_events[0].payload["target_object_id"] == "sun"


def test_as_self_report_human_readable():
    store = EventStore()
    schema = AttentionSchema(store=store)
    report = schema.update_from_intent(_intent("bird", goal=GoalType.INVESTIGATE_SURPRISE))
    text = report.as_self_report()
    assert "bird" in text
    assert "investigate_surprise" in text


def test_idle_intent_drift():
    store = EventStore()
    schema = AttentionSchema(store=store)
    report = schema.update_from_intent(_intent(None, goal=GoalType.NONE))
    assert report.target_object_id is None
    assert "drift" in report.as_self_report().lower()


def test_workspace_broadcast_updates_schema():
    store = EventStore()
    schema = AttentionSchema(store=store)
    report = schema.update_from_workspace_broadcast(
        item_source_id="bird",
        item_priority=0.7,
    )
    assert report.target_object_id == "bird"
    assert report.goal_type == GoalType.INVESTIGATE_SURPRISE
