"""AttentionSchema — AST-1 implementation.

AST-1 per Butlin et al. 2023: "A predictive model representing the
current state of attention".

Detta är inte själva attention-mekanismen (det är GW) — det är en
MODELL AV den. Agenten kan rapportera: "min attention är på X för
att Y, och den har varit där i Z ticks".

Implementeringen är minimal men strukturellt korrekt: attention-schema
abonnerar på workspace-broadcasts OCH action-events, och bygger en
löpande "self-report" av sin egen attention-process.

Notera att detta är distinkt från HOT-2 (metacognitive monitor) — HOT-2
mäter SÄKERHET om predictions. AST-1 mäter "vart riktar sig agenten,
och varför". Båda är higher-order, men över olika aspekter.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque

from selvra_brain.agency.types import ActionIntent, GoalType
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


@dataclass(frozen=True)
class AttentionReason:
    """Aktuell self-report om varför attention är där den är."""

    target_object_id: str | None
    target_angle: float
    goal_type: GoalType
    drive_strength: float
    duration_ticks: int  # hur många ticks attention har varit på denna källa
    transitions_recent: int  # hur många byten i senaste fönstret
    summary: str  # human-readable

    def as_self_report(self) -> str:
        """Selvras self-report om sin egen attention."""
        if self.target_object_id is None:
            return f"Min uppmärksamhet är drifting (ingen explicit goal)."
        return (
            f"Min uppmärksamhet är på {self.target_object_id} "
            f"({self.duration_ticks} ticks) "
            f"för att {self.goal_type.value} "
            f"(drive {self.drive_strength:.2f}). "
            f"{self.transitions_recent} skift i senaste fönstret."
        )


@dataclass
class AttentionSchema:
    """Modell av attention-process. Self-report-substrate.

    Hooks:
      - update_from_intent(intent: ActionIntent) — varje gång driver
        producerar en intent
      - update_from_workspace_broadcast(signal) — varje GW-broadcast

    Bygger:
      - current_target
      - duration_on_target (ticks sedan target ändrades)
      - transition_history (deque)
      - goal_history
    """

    store: EventStore
    transition_window: int = 30  # ticks att räkna transitions över
    current_target: str | None = field(default=None, init=False)
    current_goal: GoalType = field(default=GoalType.NONE, init=False)
    current_drive: float = field(default=0.0, init=False)
    current_angle: float = field(default=0.0, init=False)
    duration_on_target: int = field(default=0, init=False)
    _transitions: Deque[int] = field(default_factory=lambda: deque(maxlen=200), init=False)
    _tick: int = field(default=0, init=False)
    _report_count: int = field(default=0, init=False)

    def update_from_intent(self, intent: ActionIntent) -> AttentionReason:
        """Processa en ny ActionIntent. Returnera self-report."""
        self._tick += 1

        new_target = intent.target_object_id
        if new_target != self.current_target:
            self._transitions.append(self._tick)
            self.duration_on_target = 0
        else:
            self.duration_on_target += 1

        self.current_target = new_target
        self.current_goal = intent.goal
        self.current_drive = intent.drive_strength
        self.current_angle = intent.target_angle

        return self._compose_report(intent_reasoning=intent.reasoning)

    def update_from_workspace_broadcast(
        self,
        *,
        item_source_id: str | None,
        item_priority: float,
    ) -> AttentionReason:
        """När workspace broadcastar, attention skiftar implicit dit.

        Detta är "involuntary" attention — agenten styrde inte själv
        men en broadcast drog hennes attention. Schema noterar det.
        """
        self._tick += 1
        if item_source_id != self.current_target:
            self._transitions.append(self._tick)
            self.duration_on_target = 0
            self.current_target = item_source_id
            self.current_goal = GoalType.INVESTIGATE_SURPRISE  # broadcast = något hände
            self.current_drive = item_priority
        else:
            self.duration_on_target += 1

        return self._compose_report(intent_reasoning=())

    def _compose_report(self, intent_reasoning: tuple[str, ...]) -> AttentionReason:
        transitions_recent = sum(
            1 for t in self._transitions if t > self._tick - self.transition_window
        )

        if self.current_target is None:
            summary = "drifting (no explicit target)"
        else:
            summary = (
                f"on {self.current_target} for {self.duration_on_target} ticks, "
                f"goal {self.current_goal.value}, "
                f"drive {self.current_drive:.2f}"
            )

        report = AttentionReason(
            target_object_id=self.current_target,
            target_angle=self.current_angle,
            goal_type=self.current_goal,
            drive_strength=self.current_drive,
            duration_ticks=self.duration_on_target,
            transitions_recent=transitions_recent,
            summary=summary,
        )

        self._report_count += 1

        # Emit METACOGNITION-event med self-report (som AST-1 är higher-order
        # om attention, sätter vi det som metacognition snarare än separat
        # kategori — temat är "agent reports on own state")
        self.store.append(
            BrainEvent(
                category=EventCategory.METACOGNITION,
                event_type="attention_schema_report",
                payload={
                    "target_object_id": self.current_target,
                    "goal_type": self.current_goal.value,
                    "drive_strength": self.current_drive,
                    "duration_ticks": self.duration_on_target,
                    "transitions_recent": transitions_recent,
                    "summary": summary,
                    "reasoning_chain": list(intent_reasoning),
                },
                tag=EpistemicTag(
                    data_type=DataType.SELF_REPORTED,
                    confidence=Confidence.MEDIUM,
                    mutability=Mutability.IMMUTABLE,
                    persistence=Persistence.TRANSIENT,
                    memory_type=MemoryType.WORKING,
                    valence=Valence.NEUTRAL,
                ),
            )
        )

        return report

    def stats(self) -> dict:
        return {
            "report_count": self._report_count,
            "current_target": self.current_target,
            "current_goal": self.current_goal.value,
            "duration_on_target": self.duration_on_target,
            "transitions_total": len(self._transitions),
        }
