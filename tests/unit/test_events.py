"""Tests för event-sourcing core.

Verifierar:
- BrainEvent är frozen
- EventStore append-only + len/query
- Event-categorier täcker alla Butlin-arkitektur-faser
- Temporal-window-query (för RPT-2)
- to_dict producerar serialiserbar struktur
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

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


def _tag(valence: Valence = Valence.NEUTRAL) -> EpistemicTag:
    return EpistemicTag(
        data_type=DataType.OBSERVED,
        confidence=Confidence.HIGH,
        mutability=Mutability.IMMUTABLE,
        persistence=Persistence.STABLE,
        memory_type=MemoryType.EPISODIC,
        valence=valence,
    )


# ─── EventCategory täckning ────────────────────────────────────────────


def test_event_category_covers_all_architecture_phases() -> None:
    """Alla Butlin-arkitektur-faser har en kategori."""
    expected_categories = {
        "perception",  # RPT input
        "workspace_entry",  # GW broadcast
        "metacognition",  # HOT
        "prediction",  # PP
        "prediction_error",  # PP error-signal
        "action",  # AE output
        "action_effect",  # AE response
        "body_state",  # embodiment
        "valence_shift",  # affektiv förändring
    }
    actual = {c.value for c in EventCategory}
    assert expected_categories == actual


# ─── BrainEvent ────────────────────────────────────────────────────────


def test_event_construction() -> None:
    e = BrainEvent(
        category=EventCategory.PERCEPTION,
        event_type="visual_input",
        payload={"intensity": 0.5},
        tag=_tag(),
    )
    assert e.category == EventCategory.PERCEPTION
    assert e.event_type == "visual_input"


def test_event_is_frozen() -> None:
    e = BrainEvent(tag=_tag())
    with pytest.raises(Exception):
        e.event_type = "changed"  # type: ignore[misc]


def test_event_to_dict_serializable() -> None:
    e = BrainEvent(
        category=EventCategory.WORKSPACE_ENTRY,
        event_type="broadcast",
        payload={"content": "test"},
        tag=_tag(Valence.POSITIVE),
    )
    d = e.to_dict()
    assert d["category"] == "workspace_entry"
    assert d["tag"]["valence"] == "positive"
    # Source events default empty
    assert d["source_event_ids"] == []


def test_event_with_provenance() -> None:
    """source_event_ids ska bevaras — viktigt för temporal binding."""
    parent = BrainEvent(tag=_tag())
    child = BrainEvent(
        category=EventCategory.METACOGNITION,
        event_type="self_report",
        tag=_tag(),
        source_event_ids=(parent.id,),
    )
    assert parent.id in child.source_event_ids


# ─── EventStore ────────────────────────────────────────────────────────


def test_store_append_increments_length() -> None:
    store = EventStore()
    assert len(store) == 0
    store.append(BrainEvent(tag=_tag()))
    assert len(store) == 1


def test_store_append_returns_event_id() -> None:
    store = EventStore()
    event = BrainEvent(tag=_tag())
    returned_id = store.append(event)
    assert returned_id == event.id


def test_store_by_category() -> None:
    store = EventStore()
    store.append(BrainEvent(category=EventCategory.PERCEPTION, tag=_tag()))
    store.append(BrainEvent(category=EventCategory.METACOGNITION, tag=_tag()))
    store.append(BrainEvent(category=EventCategory.PERCEPTION, tag=_tag()))

    perception_events = store.by_category(EventCategory.PERCEPTION)
    assert len(perception_events) == 2
    metacog_events = store.by_category(EventCategory.METACOGNITION)
    assert len(metacog_events) == 1


def test_store_by_event_type() -> None:
    store = EventStore()
    store.append(BrainEvent(event_type="visual", tag=_tag()))
    store.append(BrainEvent(event_type="auditory", tag=_tag()))
    store.append(BrainEvent(event_type="visual", tag=_tag()))

    visual = store.by_event_type("visual")
    assert len(visual) == 2


def test_store_recent_returns_n_newest_first() -> None:
    store = EventStore()
    for i in range(5):
        store.append(BrainEvent(event_type=f"e{i}", tag=_tag()))

    recent = store.recent(n=3)
    assert len(recent) == 3
    # Nyast först → e4, e3, e2
    assert recent[0].event_type == "e4"
    assert recent[1].event_type == "e3"
    assert recent[2].event_type == "e2"


def test_store_in_time_window() -> None:
    """RPT-2 temporal binding kräver tidsfönster-query."""
    store = EventStore()
    now = datetime.now(tz=UTC)

    # Event 1 sek sedan
    old = BrainEvent(tag=_tag())
    object.__setattr__(old, "created_at", now - timedelta(seconds=10))
    store.append(old)

    # Event nyss
    new = BrainEvent(tag=_tag())
    store.append(new)

    # Fönster sista 2 sekunderna ska bara fånga den nya
    window_events = store.in_time_window(
        start=now - timedelta(seconds=2),
        end=now + timedelta(seconds=1),
    )
    assert len(window_events) == 1
    assert window_events[0].id == new.id


def test_store_all_events_returns_tuple() -> None:
    """all_events ska returnera tuple (read-only) — append-only-invariant."""
    store = EventStore()
    store.append(BrainEvent(tag=_tag()))
    events = store.all_events()
    assert isinstance(events, tuple)
    # Vi kan inte appenda till tuple
    with pytest.raises(AttributeError):
        events.append(BrainEvent(tag=_tag()))  # type: ignore[attr-defined]
