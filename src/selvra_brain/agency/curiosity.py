"""CuriosityDriver — AE-1 implementation.

AE-1 per Butlin et al. 2023: "Agency: learning from feedback and selecting
outputs so as to pursue goals, especially where this involves flexible
responsiveness to competing goals".

Per Fas 1: agenten har ett enkelt drivsystem med flera konkurrerande
sub-drives:
  1. REDUCE_UNCERTAINTY — låg reliability per source drar attention dit
  2. INVESTIGATE_SURPRISE — färska prediction-errors drar dit
  3. EXPLORE_NEGLECTED — sources med få observations drar dit
  4. APPROACH/AVOID VALENCE — positiv valens attraherar, negativ repellerar

Drives kombineras viktat. Vinnaren bestämmer ActionIntent.

Detta är inte RL — det är heuristisk drift-balansering. Senare faser kan
ersätta heuristiken med learned weights. Det är vad "learning + flexible"
i AE-1 syftar på — och vi gör substratet flexibelt nu, lär oss senare.

Notera: vi PRODUCERAR en intent, men ACTION-effekt-modellen kör i
parallell. Agentens look_at-handling drivs av intent + effekt-modellen
kompenserar för förväntade vs. faktiska skift.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from selvra_brain.agency.types import ActionIntent, ActionType, GoalType
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
class GoalSignal:
    """En aktuell drivkraft över en specifik källa.

    Drivers samlar GoalSignals och summerar per (goal_type, source).
    Den starkaste vinner och blir en ActionIntent.
    """

    source: str  # object_id
    goal_type: GoalType
    strength: float  # 0-1
    reason: str


@dataclass
class CuriosityDriver:
    """Heuristisk drift-balanseringsmotor.

    State som drivern läser:
    - per_source_reliability (från MetacognitiveMonitor)
    - recent prediction_error events
    - source observation counts
    - per-source recent valence

    Vikter är tunable. Default-värden är empiriskt rimliga för Fas 1
    men kan ändras per demo.
    """

    store: EventStore
    weight_reduce_uncertainty: float = 0.35
    weight_investigate_surprise: float = 0.30
    weight_explore_neglected: float = 0.15
    weight_valence: float = 0.20
    surprise_recency_window: int = 30  # senaste N events
    drive_threshold: float = 0.15  # under detta = IDLE (drift)

    _observation_counts: Counter[str] = field(default_factory=Counter, init=False)
    _last_action_target: str | None = field(default=None, init=False)
    _action_count: int = field(default=0, init=False)

    def record_observation(self, source: str) -> None:
        """Räkna observation per källa — för EXPLORE_NEGLECTED-mätningen."""
        self._observation_counts[source] += 1

    def decide(
        self,
        *,
        source_positions: dict[str, float],
        reliability_per_source: dict[str, float],
    ) -> ActionIntent:
        """Producera ActionIntent baserat på aktuellt state.

        Args:
            source_positions: object_id → position_angle
            reliability_per_source: object_id → reliability [0, 1]

        Returnerar:
            ActionIntent med action, target, goal, drive_strength, reasoning.
        """
        signals: list[GoalSignal] = []

        # 1. REDUCE_UNCERTAINTY: 1 - reliability per source
        for src in source_positions:
            r = reliability_per_source.get(src, 0.5)
            uncertainty = 1.0 - r
            if uncertainty > 0.1:
                signals.append(
                    GoalSignal(
                        source=src,
                        goal_type=GoalType.REDUCE_UNCERTAINTY,
                        strength=uncertainty * self.weight_reduce_uncertainty,
                        reason=f"reliability {r:.2f} → uncertainty {uncertainty:.2f}",
                    )
                )

        # 2. INVESTIGATE_SURPRISE: senaste prediction-errors
        recent_errors = self.store.recent(self.surprise_recency_window)
        per_source_recent_surprise: Counter[str] = Counter()
        for ev in recent_errors:
            if ev.category == EventCategory.PREDICTION_ERROR:
                src = ev.payload.get("source")
                mag = ev.payload.get("normalized_magnitude", 0.0)
                if src in source_positions and mag > 0.2:
                    per_source_recent_surprise[src] += 1
        for src, count in per_source_recent_surprise.items():
            # 1 spike = 0.4, 2 = 0.7, 3+ = 1.0
            strength = min(1.0, 0.4 * count)
            signals.append(
                GoalSignal(
                    source=src,
                    goal_type=GoalType.INVESTIGATE_SURPRISE,
                    strength=strength * self.weight_investigate_surprise,
                    reason=f"{count} recent surprises",
                )
            )

        # 3. EXPLORE_NEGLECTED: jämfört med medel-observation-count
        if self._observation_counts:
            total = sum(self._observation_counts.values())
            n = len(source_positions)
            expected = total / max(n, 1)
            for src in source_positions:
                count = self._observation_counts.get(src, 0)
                if count < expected * 0.7 and expected > 5:
                    deficit = (expected - count) / max(expected, 1.0)
                    signals.append(
                        GoalSignal(
                            source=src,
                            goal_type=GoalType.EXPLORE_NEGLECTED,
                            strength=deficit * self.weight_explore_neglected,
                            reason=f"obs_count {count} vs expected {expected:.0f}",
                        )
                    )

        # 4. VALENCE: per source från recent valence-shifts
        per_source_valence: dict[str, float] = {}
        for ev in recent_errors:
            if ev.category == EventCategory.PREDICTION_ERROR:
                src = ev.payload.get("source")
                if src in source_positions and ev.tag:
                    v = ev.tag.valence.to_numeric()
                    per_source_valence[src] = (
                        0.7 * per_source_valence.get(src, 0.0) + 0.3 * v
                    )
        for src, v in per_source_valence.items():
            if abs(v) > 0.2:
                if v > 0:
                    signals.append(
                        GoalSignal(
                            source=src,
                            goal_type=GoalType.APPROACH_VALENCE,
                            strength=abs(v) * self.weight_valence,
                            reason=f"positive recent valence {v:.2f}",
                        )
                    )
                else:
                    # Negativ valens kan båda dra (investigate) och stöta
                    # bort — i Fas 1 låter vi den dra med lägre vikt
                    signals.append(
                        GoalSignal(
                            source=src,
                            goal_type=GoalType.AVOID_VALENCE,
                            strength=abs(v) * self.weight_valence * 0.5,
                            reason=f"negative recent valence {v:.2f}",
                        )
                    )

        # Aggregera per source (summera strengths)
        per_source: dict[str, float] = {}
        per_source_dominant: dict[str, GoalSignal] = {}
        for sig in signals:
            per_source[sig.source] = per_source.get(sig.source, 0.0) + sig.strength
            if (
                sig.source not in per_source_dominant
                or sig.strength > per_source_dominant[sig.source].strength
            ):
                per_source_dominant[sig.source] = sig

        if not per_source:
            # Inga drives — IDLE
            intent = ActionIntent(
                action=ActionType.IDLE,
                goal=GoalType.NONE,
                drive_strength=0.0,
                reasoning=("no active drives",),
            )
        else:
            best_source = max(per_source, key=per_source.get)
            best_strength = per_source[best_source]
            if best_strength < self.drive_threshold:
                intent = ActionIntent(
                    action=ActionType.IDLE,
                    goal=GoalType.NONE,
                    drive_strength=best_strength,
                    reasoning=(f"strongest drive {best_strength:.2f} under threshold",),
                )
            else:
                dominant = per_source_dominant[best_source]
                # Sub-reasons: alla signals för vinnande källa
                source_signals = [s for s in signals if s.source == best_source]
                reasons = tuple(
                    f"{s.goal_type.value}: {s.reason} (={s.strength:.2f})"
                    for s in source_signals
                )
                intent = ActionIntent(
                    action=ActionType.LOOK_AT,
                    target_angle=source_positions[best_source],
                    target_object_id=best_source,
                    goal=dominant.goal_type,
                    drive_strength=min(1.0, best_strength),
                    reasoning=reasons,
                )

        # Emit ACTION-event så hela kedjan är spårbar
        self._action_count += 1
        valence_for_action = self._intent_valence(intent)
        self.store.append(
            BrainEvent(
                category=EventCategory.ACTION,
                event_type=f"action_{intent.action.value}",
                payload={
                    "action": intent.action.value,
                    "target_angle": intent.target_angle,
                    "target_object_id": intent.target_object_id,
                    "goal": intent.goal.value,
                    "drive_strength": intent.drive_strength,
                    "reasoning_count": len(intent.reasoning),
                },
                tag=EpistemicTag(
                    data_type=DataType.SELF_REPORTED,
                    confidence=Confidence.MEDIUM,
                    mutability=Mutability.IMMUTABLE,
                    persistence=Persistence.SHORT_TERM,
                    memory_type=MemoryType.PROCEDURAL,
                    valence=valence_for_action,
                ),
            )
        )

        if intent.target_object_id is not None:
            self._last_action_target = intent.target_object_id
        return intent

    @staticmethod
    def _intent_valence(intent: ActionIntent) -> Valence:
        """Affektiv signatur per goal-typ.

        REDUCE_UNCERTAINTY = mild positiv (curiosity är belönande)
        INVESTIGATE_SURPRISE = neutral (ambivalent — surprise är negativt
            men investigation positivt)
        EXPLORE_NEGLECTED = mild positiv
        APPROACH_VALENCE = positiv
        AVOID_VALENCE = negativ
        IDLE = neutral
        """
        mapping = {
            GoalType.REDUCE_UNCERTAINTY: Valence.POSITIVE,
            GoalType.INVESTIGATE_SURPRISE: Valence.NEUTRAL,
            GoalType.EXPLORE_NEGLECTED: Valence.POSITIVE,
            GoalType.APPROACH_VALENCE: Valence.POSITIVE_STRONG,
            GoalType.AVOID_VALENCE: Valence.NEGATIVE,
            GoalType.NONE: Valence.NEUTRAL,
        }
        return mapping[intent.goal]

    def stats(self) -> dict:
        return {
            "action_count": self._action_count,
            "observation_counts": dict(self._observation_counts),
            "last_action_target": self._last_action_target,
        }
