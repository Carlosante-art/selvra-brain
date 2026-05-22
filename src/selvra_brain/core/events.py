"""Event-sourcing core för selvra-brain.

Förenklad version av Selvras events-modul:
- Inga tenant-id, inga RLS, single-agent fokus
- Append-only (Princip 9 från Selvra)
- Replay-kapabel för temporal-binding (RPT/GWT-relevant)
- In-memory först; senare persistens via t.ex. SQLite om vi behöver

Varje event har EpistemicTag (inkl Valence) — alla representationer i
brain har affektiv laddning per Damasio core consciousness.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from selvra_brain.core.epistemic import EpistemicTag


class EventCategory(StrEnum):
    """Top-level kategori för events i brain.

    Speglar olika faser i Butlin-arkitekturen — perception (RPT/GW
    input), workspace-broadcast, metacognitive-self-report,
    prediction-error, agency-action, embodiment-state.
    """

    PERCEPTION = "perception"  # RPT-input
    WORKSPACE_ENTRY = "workspace_entry"  # GW-broadcast
    METACOGNITION = "metacognition"  # HOT self-report
    PREDICTION = "prediction"  # PP forecast
    PREDICTION_ERROR = "prediction_error"  # PP error-signal
    ACTION = "action"  # AE-output
    ACTION_EFFECT = "action_effect"  # AE-environment-response
    BODY_STATE = "body_state"  # interocepting/propriocepting
    VALENCE_SHIFT = "valence_shift"  # affektiv förändring


@dataclass(frozen=True)
class BrainEvent:
    """En händelse i agentens representation.

    Frozen — events muteras inte. Att ändra något producerar ett nytt
    event. Detta bevarar temporal-binding-möjlighet.
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    category: EventCategory = EventCategory.PERCEPTION
    event_type: str = ""  # specifik typ inom kategori
    payload: dict[str, Any] = field(default_factory=dict)
    tag: EpistemicTag | None = None
    # Provenance — vilka tidigare events bidrog till denna?
    # För temporal binding (RPT-2) + metacognitive-self-report (HOT-2).
    source_event_ids: tuple[uuid.UUID, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "category": self.category.value,
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "tag": self.tag.to_dict() if self.tag else None,
            "source_event_ids": [str(i) for i in self.source_event_ids],
        }


class EventStore:
    """In-memory append-only event-store.

    Senare iterationer kan persistera till SQLite om vi behöver
    replay-över-omstart eller multi-agent-shared-memory. För Fas 0-1
    räcker in-memory.
    """

    def __init__(self) -> None:
        self._events: list[BrainEvent] = []

    def append(self, event: BrainEvent) -> uuid.UUID:
        """Append-only. Returnerar event-id."""
        self._events.append(event)
        return event.id

    def __len__(self) -> int:
        return len(self._events)

    def all_events(self) -> tuple[BrainEvent, ...]:
        """Snapshot av alla events. Read-only tuple."""
        return tuple(self._events)

    def by_category(self, category: EventCategory) -> tuple[BrainEvent, ...]:
        return tuple(e for e in self._events if e.category == category)

    def by_event_type(self, event_type: str) -> tuple[BrainEvent, ...]:
        return tuple(e for e in self._events if e.event_type == event_type)

    def recent(self, n: int = 10) -> tuple[BrainEvent, ...]:
        """Senaste N events (nyast först)."""
        return tuple(reversed(self._events[-n:]))

    def in_time_window(
        self,
        *,
        start: datetime,
        end: datetime,
    ) -> tuple[BrainEvent, ...]:
        """Events i fönster — för RPT-2 temporal binding."""
        return tuple(e for e in self._events if start <= e.created_at <= end)
