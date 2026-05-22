"""Tests för GlobalWorkspace + producers."""

from __future__ import annotations

import pytest

from selvra_brain.core.events import EventCategory, EventStore
from selvra_brain.prediction.predictor import PredictionError
from selvra_brain.workspace import (
    AttentionVector,
    BroadcastSignal,
    GlobalWorkspace,
    PredictionErrorProducer,
    WorkspaceItem,
    WorkspaceSource,
    prediction_error_to_workspace_item,
)


def _item(content="x", priority=0.5, valence=0.0, source=WorkspaceSource.PERCEPTION):
    return WorkspaceItem(
        content=content,
        source=source,
        priority=priority,
        valence=valence,
    )


# ─── AttentionVector ──────────────────────────────────────────────


def test_attention_magnitude() -> None:
    assert AttentionVector(0, 0).magnitude() == 0
    assert AttentionVector(3, 4).magnitude() == 5.0


def test_attention_normalized() -> None:
    v = AttentionVector(3, 4).normalized()
    assert abs(v.magnitude() - 1.0) < 1e-9


def test_attention_normalized_zero() -> None:
    v = AttentionVector(0, 0).normalized()
    assert v.x == 0 and v.y == 0


# ─── WorkspaceItem saliency ──────────────────────────────────────


def test_saliency_neutral_low_priority() -> None:
    item = _item(priority=0.1, valence=0.0)
    assert item.saliency() == pytest.approx(0.1)


def test_saliency_valence_boosts() -> None:
    item = _item(priority=0.3, valence=0.8)
    # 0.3 + 0.5 * 0.8 = 0.7
    assert item.saliency() == pytest.approx(0.7)


def test_saliency_clamped_at_one() -> None:
    item = _item(priority=1.0, valence=1.0)
    assert item.saliency() == 1.0


# ─── GlobalWorkspace basic ──────────────────────────────────────


def test_workspace_requires_positive_capacity() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        GlobalWorkspace(EventStore(), capacity=0)


def test_workspace_empty_initially() -> None:
    ws = GlobalWorkspace(EventStore())
    assert len(ws) == 0
    assert not ws.is_full()


def test_propose_below_threshold_rejected() -> None:
    ws = GlobalWorkspace(EventStore(), acceptance_threshold=0.5)
    item = _item(priority=0.1, valence=0.0)  # saliency 0.1
    assert ws.propose(item) is False
    assert len(ws) == 0


def test_propose_above_threshold_accepted() -> None:
    ws = GlobalWorkspace(EventStore(), acceptance_threshold=0.3)
    item = _item(priority=0.5, valence=0.2)
    assert ws.propose(item) is True
    assert len(ws) == 1


# ─── GW-2: capacity ─────────────────────────────────────────────


def test_workspace_capacity_overflow_replaces_lowest_saliency() -> None:
    ws = GlobalWorkspace(EventStore(), capacity=2, acceptance_threshold=0.0)
    high1 = _item(content="high1", priority=0.8)
    high2 = _item(content="high2", priority=0.7)
    low = _item(content="low", priority=0.3)
    ws.propose(high1)
    ws.propose(high2)
    # workspace full med {high1=0.8, high2=0.7}
    # Ny low (0.3) bör REJECTAS (lägre än alla i workspace)
    assert ws.propose(low) is False
    assert len(ws) == 2

    # Ny med högre saliency än lägsta ersätter
    very_high = _item(content="very_high", priority=0.95)
    assert ws.propose(very_high) is True
    contents = {i.content for i in ws.current_items()}
    assert "very_high" in contents
    assert "high2" not in contents  # ersatte lägsta (0.7)


# ─── GW-3: broadcast ───────────────────────────────────────────


def test_broadcast_to_subscribers() -> None:
    ws = GlobalWorkspace(EventStore(), acceptance_threshold=0.0)
    received: list[str] = []

    def listener(sig: BroadcastSignal) -> None:
        received.append(sig.item.content)

    ws.subscribe(listener)
    ws.propose(_item(content="hello"))
    ws.propose(_item(content="world"))
    assert received == ["hello", "world"]


def test_rejected_item_not_broadcast() -> None:
    ws = GlobalWorkspace(EventStore(), acceptance_threshold=0.9)
    received: list[str] = []
    ws.subscribe(lambda sig: received.append(sig.item.content))
    ws.propose(_item(content="too_low", priority=0.1))
    assert received == []


def test_unsubscribe() -> None:
    ws = GlobalWorkspace(EventStore(), acceptance_threshold=0.0)
    received: list[str] = []
    listener = lambda sig: received.append(sig.item.content)  # noqa: E731
    ws.subscribe(listener)
    ws.propose(_item(content="x"))
    ws.unsubscribe(listener)
    ws.propose(_item(content="y"))
    assert received == ["x"]


def test_subscriber_exception_does_not_break_others() -> None:
    """En subscribers fel ska inte stoppa övriga."""
    ws = GlobalWorkspace(EventStore(), acceptance_threshold=0.0)
    received: list[str] = []

    def failing(sig: BroadcastSignal) -> None:
        raise RuntimeError("boom")

    def working(sig: BroadcastSignal) -> None:
        received.append(sig.item.content)

    ws.subscribe(failing)
    ws.subscribe(working)
    ws.propose(_item(content="x"))
    assert received == ["x"]


# ─── Event-emission ─────────────────────────────────────────────


def test_accepted_item_emits_workspace_entry_event() -> None:
    store = EventStore()
    ws = GlobalWorkspace(store, acceptance_threshold=0.0)
    ws.propose(_item(content="something"))
    events = store.by_category(EventCategory.WORKSPACE_ENTRY)
    assert len(events) == 1
    assert events[0].payload["content"] == "something"


def test_event_payload_includes_attention() -> None:
    store = EventStore()
    ws = GlobalWorkspace(store, acceptance_threshold=0.0)
    item = WorkspaceItem(
        content="x",
        source=WorkspaceSource.PERCEPTION,
        priority=0.6,
        attention_vector=AttentionVector(x=0.5, y=-0.3),
    )
    ws.propose(item)
    event = store.by_category(EventCategory.WORKSPACE_ENTRY)[0]
    assert event.payload["attention_x"] == 0.5
    assert event.payload["attention_y"] == -0.3


# ─── stats() ────────────────────────────────────────────────────


def test_stats_tracks_proposed_accepted_rejected() -> None:
    ws = GlobalWorkspace(EventStore(), acceptance_threshold=0.5)
    ws.propose(_item(content="a", priority=0.7))  # accepted
    ws.propose(_item(content="b", priority=0.1))  # rejected
    ws.propose(_item(content="c", priority=0.9))  # accepted
    stats = ws.stats()
    assert stats["proposed"] == 3
    assert stats["accepted"] == 2
    assert stats["rejected"] == 1


# ─── Producers: PredictionError → WorkspaceItem ────────────────


def _err(magnitude=1.0, signed=1.0):
    return PredictionError(
        source_name="temp",
        predicted=10.0,
        observed=10.0 + signed,
        magnitude=magnitude,
        signed=signed,
        level=0,
    )


def test_small_error_does_not_become_workspace_item() -> None:
    """Errors under threshold → ingen workspace-candidate."""
    err = _err(magnitude=0.05, signed=0.05)
    item = prediction_error_to_workspace_item(err, surprise_threshold=0.3)
    assert item is None


def test_large_error_becomes_workspace_item() -> None:
    err = _err(magnitude=10.0, signed=10.0)
    item = prediction_error_to_workspace_item(err)
    assert item is not None
    assert item.source == WorkspaceSource.PREDICTION_ERROR
    assert "surprise" in item.content


def test_error_workspace_item_has_attention_vector() -> None:
    err = _err(magnitude=5.0, signed=5.0)
    item = prediction_error_to_workspace_item(err)
    # Positiv surprise → x > 0
    assert item.attention_vector.x > 0
    # Stor magnitude → y negativ (upp)
    assert item.attention_vector.y < 0


def test_error_workspace_item_negative_valence() -> None:
    """Surprise är affektivt laddat — negativ (Damasio)."""
    err = _err(magnitude=10.0)
    item = prediction_error_to_workspace_item(err)
    assert item.valence < 0


# ─── PredictionErrorProducer ───────────────────────────────────


def test_producer_stats() -> None:
    store = EventStore()
    ws = GlobalWorkspace(store, acceptance_threshold=0.3)
    producer = PredictionErrorProducer(ws, surprise_threshold=0.3)

    # Liten error — ska inte ens propose:as
    producer.handle_error(_err(magnitude=0.05))
    # Stor error — proposed + accepted
    producer.handle_error(_err(magnitude=10.0))

    assert producer.proposed_count == 1
    assert producer.accepted_count == 1
