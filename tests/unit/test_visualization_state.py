"""Tests för BrainVisualState + derivation från EventStore."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

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
from selvra_brain.visualization.state import (
    AttentionVector,
    BrainVisualState,
    derive_state_from_store,
    write_state,
    write_state_from_store,
)


def _tag(valence: Valence = Valence.NEUTRAL) -> EpistemicTag:
    return EpistemicTag(
        data_type=DataType.OBSERVED,
        confidence=Confidence.HIGH,
        mutability=Mutability.IMMUTABLE,
        persistence=Persistence.STABLE,
        memory_type=MemoryType.EPISODIC,
        valence=valence,
    )


# ─── AttentionVector ──────────────────────────────────────────────


def test_attention_default_is_inward() -> None:
    """Default attention är (0, 0) — inåtblick."""
    v = AttentionVector()
    assert v.x == 0.0
    assert v.y == 0.0


def test_attention_magnitude() -> None:
    assert AttentionVector(x=0, y=0).magnitude() == 0
    assert AttentionVector(x=1, y=0).magnitude() == 1.0
    assert abs(AttentionVector(x=0.6, y=0.8).magnitude() - 1.0) < 1e-9


# ─── BrainVisualState ────────────────────────────────────────────


def test_default_state() -> None:
    s = BrainVisualState()
    assert s.event_count == 0
    assert s.valence == 0.0
    assert s.arousal == 0.5
    assert s.cognitive_load == 0.0
    assert s.workspace_items == 0


def test_state_to_dict() -> None:
    s = BrainVisualState(
        event_count=5,
        valence=0.5,
        attention=AttentionVector(x=0.3, y=-0.2),
    )
    d = s.to_dict()
    assert d["event_count"] == 5
    assert d["valence"] == 0.5
    assert d["attention"]["x"] == 0.3
    assert d["attention"]["y"] == -0.2


def test_state_to_json_is_valid() -> None:
    s = BrainVisualState(event_count=3, valence=-0.5)
    parsed = json.loads(s.to_json())
    assert parsed["event_count"] == 3
    assert parsed["valence"] == -0.5


# ─── Derivation från EventStore ──────────────────────────────────


def test_derive_from_empty_store_returns_zero_state() -> None:
    store = EventStore()
    state = derive_state_from_store(store)
    assert state.event_count == 0
    assert state.valence == 0.0
    assert state.last_event_type is None


def test_derive_event_count_matches_store() -> None:
    store = EventStore()
    for i in range(7):
        store.append(BrainEvent(event_type=f"e{i}", tag=_tag()))
    state = derive_state_from_store(store)
    assert state.event_count == 7


def test_derive_valence_aggregates_positive() -> None:
    """Bara positiva events → positiv aggregat-valens."""
    store = EventStore()
    for _ in range(5):
        store.append(BrainEvent(tag=_tag(Valence.POSITIVE_STRONG)))
    state = derive_state_from_store(store)
    assert state.valence > 0.5


def test_derive_valence_aggregates_negative() -> None:
    store = EventStore()
    for _ in range(5):
        store.append(BrainEvent(tag=_tag(Valence.NEGATIVE_STRONG)))
    state = derive_state_from_store(store)
    assert state.valence < -0.5


def test_derive_valence_aggregates_mix() -> None:
    """Lika många positiva som negativa → nära neutral."""
    store = EventStore()
    for _ in range(5):
        store.append(BrainEvent(tag=_tag(Valence.POSITIVE_STRONG)))
        store.append(BrainEvent(tag=_tag(Valence.NEGATIVE_STRONG)))
    state = derive_state_from_store(store)
    assert abs(state.valence) < 0.1


def test_derive_valence_shift_flag() -> None:
    """Senaste event = VALENCE_SHIFT → flag är True."""
    store = EventStore()
    store.append(BrainEvent(category=EventCategory.PERCEPTION, tag=_tag()))
    store.append(
        BrainEvent(category=EventCategory.VALENCE_SHIFT, tag=_tag(Valence.POSITIVE))
    )
    state = derive_state_from_store(store)
    assert state.valence_just_shifted is True


def test_derive_workspace_items() -> None:
    """WORKSPACE_ENTRY-events räknas (senaste 10 sek)."""
    store = EventStore()
    for _ in range(3):
        store.append(
            BrainEvent(category=EventCategory.WORKSPACE_ENTRY, event_type="focus_a", tag=_tag())
        )
    state = derive_state_from_store(store)
    assert state.workspace_items == 3
    assert state.workspace_focus == "focus_a"


def test_derive_cognitive_load_from_prediction_errors() -> None:
    """Andel PREDICTION_ERROR-events bland senaste = cognitive_load."""
    store = EventStore()
    # 3 normala + 2 prediction-error → 2/5 = 0.4
    for _ in range(3):
        store.append(BrainEvent(category=EventCategory.PERCEPTION, tag=_tag()))
    for _ in range(2):
        store.append(BrainEvent(category=EventCategory.PREDICTION_ERROR, tag=_tag()))
    state = derive_state_from_store(store)
    assert abs(state.cognitive_load - 0.4) < 0.01


def test_derive_last_event_type() -> None:
    store = EventStore()
    store.append(BrainEvent(event_type="first", tag=_tag()))
    store.append(BrainEvent(event_type="second", tag=_tag()))
    state = derive_state_from_store(store)
    assert state.last_event_type == "second"


# ─── write_state ──────────────────────────────────────────────────


def test_write_state_creates_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        state = BrainVisualState(event_count=5, valence=0.5)
        write_state(state, path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["event_count"] == 5


def test_write_state_creates_parent_dirs() -> None:
    """write_state ska skapa parent-dirs om de saknas."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "deep" / "state.json"
        write_state(BrainVisualState(), path)
        assert path.exists()


def test_write_state_is_atomic() -> None:
    """Efter write ska INGEN .tmp-fil ligga kvar."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        write_state(BrainVisualState(), path)
        # Inga .tmp-filer
        assert not (path.with_suffix(".json.tmp")).exists()


def test_write_state_from_store_helper() -> None:
    """Convenience-funktion ska både derivera och skriva."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        store = EventStore()
        store.append(BrainEvent(event_type="hello", tag=_tag(Valence.POSITIVE)))
        state = write_state_from_store(store, path)
        # State returneras
        assert state.event_count == 1
        # Och skrivs till fil
        loaded = json.loads(path.read_text())
        assert loaded["event_count"] == 1
        assert loaded["last_event_type"] == "hello"
