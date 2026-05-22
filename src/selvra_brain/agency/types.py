"""Bas-typer för agency-modulen."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ActionType(StrEnum):
    """Vilken typ av action agenten kan utföra.

    Fas 1: bara LOOK_AT (modulerar perception). Senare faser kan
    addera MOVE_TO, UTTER, REST etc.
    """

    LOOK_AT = "look_at"  # rikta attention/focus mot en angle
    IDLE = "idle"  # ingen explicit action — drift mot defaultblick


class GoalType(StrEnum):
    """Varför agenten valde denna action — meta-info för AST-1.

    Detta är inte alla möjliga goals i världen — det är de KÄLLOR av
    drive som CuriosityDriver kan identifiera i Fas 1.
    """

    REDUCE_UNCERTAINTY = "reduce_uncertainty"  # låg reliability drar
    INVESTIGATE_SURPRISE = "investigate_surprise"  # hög prediction-error drar
    EXPLORE_NEGLECTED = "explore_neglected"  # information-gap drar
    APPROACH_VALENCE = "approach_valence"  # positiv valens drar
    AVOID_VALENCE = "avoid_valence"  # negativ valens skjuter
    NONE = "none"  # ingen tydlig drivkraft — idle drift


@dataclass(frozen=True)
class ActionIntent:
    """Agentens beslut om nästa action.

    Inkluderar BÅDE actionen själv OCH goal-motivationen, eftersom
    AST-1 (attention-schema) behöver veta varför attention är där.
    Detta gör self-report möjligt.
    """

    action: ActionType
    target_angle: float = 0.0  # för LOOK_AT
    target_object_id: str | None = None  # om agenten själv "vet" namn
    goal: GoalType = GoalType.NONE
    drive_strength: float = 0.0  # 0-1, hur stark drivkraften är
    reasoning: tuple[str, ...] = field(default_factory=tuple)  # human-readable
