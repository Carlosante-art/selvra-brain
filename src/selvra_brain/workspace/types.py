"""Workspace types — items, attention, broadcast.

Per Butlin et al GW-1..4:

GW-1: Multipla specialiserade system kapabla att operera parallellt
GW-2: Begränsad kapacitet workspace, attention-flaskhals
GW-3: Global broadcast — workspace gör information tillgänglig för alla
GW-4: Tillstånds-beroende attention för att styra workspace
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class WorkspaceSource(StrEnum):
    """Vilken modul producerade detta candidate-item."""

    PERCEPTION = "perception"
    PREDICTION_ERROR = "prediction_error"
    MEMORY = "memory"
    VALENCE_SHIFT = "valence_shift"
    METACOGNITION = "metacognition"
    BODY = "body"
    AGENT = "agent"
    EXTERNAL = "external"


@dataclass(frozen=True)
class AttentionVector:
    """2D-riktning för attention. Mappas till blickriktning i visualization.

    (0, 0) = inåtblick / fokus på intern state.
    (1, 0) = höger, (-1, 0) = vänster, (0, -1) = upp, (0, 1) = ned.
    """

    x: float = 0.0
    y: float = 0.0

    def magnitude(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5

    def normalized(self) -> AttentionVector:
        m = self.magnitude()
        if m == 0:
            return AttentionVector()
        return AttentionVector(x=self.x / m, y=self.y / m)


@dataclass(frozen=True)
class WorkspaceItem:
    """Ett kandidat-item som vill in i workspace.

    Per GW-2: bara N items kan vara i workspace samtidigt. Attention-
    mekanismen väljer baserat på:
    - priority (modulens egen vikt)
    - valence_magnitude (affektivt laddat → mer relevant)
    - recency (nyare items vinner vid lika)

    Detta är inte hard rule — workspace-implementationen kan ha
    sin egen utvärdering (state-dependent attention per GW-4).
    """

    content: str  # Vad är detta? Textuell beskrivning för introspektion
    source: WorkspaceSource
    priority: float = 0.5  # 0-1, modulens egen bedömning
    valence: float = 0.0  # -1 to 1, affektiv laddning
    attention_vector: AttentionVector = field(default_factory=AttentionVector)
    payload: dict = field(default_factory=dict)  # source-specifik data
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def saliency(self) -> float:
        """Beräkna saliency-score för attention-bottleneck.

        Saliency = priority + 0.5 * |valence|. Klampat 0-1.
        Detta är default — GlobalWorkspace.attention_select kan
        override för state-dependent logic (GW-4).
        """
        s = self.priority + 0.5 * abs(self.valence)
        return max(0.0, min(1.0, s))


@dataclass(frozen=True)
class BroadcastSignal:
    """Vad alla subscribers får när en item kommer in i workspace.

    Detta är GW-3 — global broadcast. Varje modul i systemet får
    notifikation om att något har "nått medvetandet" (worken).
    """

    item: WorkspaceItem
    workspace_size: int  # hur många items i workspace just nu
    accepted_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
