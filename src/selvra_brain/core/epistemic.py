"""Epistemic taxonomy — kategorisering av representations-statusen.

Baserad på Selvras `types/epistemics.py`, utvidgad med Valence
(affektiv laddning) per Damasios core consciousness-position.

Varje representation i selvra-brain har ett EpistemicTag som spårar:
- data_type: hur kom representationen till
- confidence: hur säker är systemet på den
- mutability: hur ändras den över tid
- persistence: hur stabil är den
- memory_type: vilken minnes-kategori
- valence: AFFEKTIV LADDNING (positiv/neutral/negativ) — alltid
  närvarande, aldrig "neutral fact" som default

Sista punkten är arkitektoniskt skiljd från Selvra. Per Damasio:
en medveten hjärna registrerar inte värdeneutrala facts. Allt är
laddat — även "neutralt" är en aktiv klassificering, inte frånvaro.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DataType(StrEnum):
    """Hur kom representationen till."""

    OBSERVED = "observed"  # från sensor
    DERIVED = "derived"  # från beräkning över andra representationer
    PREDICTED = "predicted"  # från predictive-coding-modulen
    SELF_REPORTED = "self_reported"  # från agentens eget metacognitiva system


class Confidence(StrEnum):
    """Systemets säkerhet om representationen."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNAVAILABLE = "unavailable"


class Mutability(StrEnum):
    """Hur kan representationen ändras."""

    IMMUTABLE = "immutable"  # bevarat append-only
    SYSTEM_MUTABLE = "system_mutable"  # interna processer kan revidera
    ENVIRONMENT_MUTABLE = "environment_mutable"  # extern input ändrar


class Persistence(StrEnum):
    """Stabilitet över tid."""

    TRANSIENT = "transient"  # workspace-only, försvinner snabbt
    SHORT_TERM = "short_term"  # några minuter till timmar
    STABLE = "stable"  # längre, bevaras
    CORE = "core"  # del av agent-identitet, ändras sällan


class MemoryType(StrEnum):
    """Minnes-kategorisering."""

    PROCEDURAL = "procedural"  # hur-att-göra
    EPISODIC = "episodic"  # specifika händelser med kontext
    SEMANTIC = "semantic"  # generell kunskap
    WORKING = "working"  # aktiv i workspace nu


# ─── Affektiv valens (utvidgning bortom Butlin) ───────────────────────


class Valence(StrEnum):
    """Affektiv laddning per Damasios core consciousness.

    Inte en feature ovanpå representationen — en grundläggande dimension.
    Varje representation som hamnar i workspace är affektivt klassificerad.
    "NEUTRAL" är ett aktivt tillstånd, inte default-frånvaro.

    Nivåerna är ordinala — POSITIVE_STRONG > POSITIVE > NEUTRAL >
    NEGATIVE > NEGATIVE_STRONG. Detta gör matematik möjlig (skala till
    [-1, 1] för viktning i attention-mekanismer).
    """

    POSITIVE_STRONG = "positive_strong"  # 1.0
    POSITIVE = "positive"  # 0.5
    NEUTRAL = "neutral"  # 0.0 — AKTIV klassificering, inte default
    NEGATIVE = "negative"  # -0.5
    NEGATIVE_STRONG = "negative_strong"  # -1.0

    def to_numeric(self) -> float:
        """Skala till [-1, 1] för matematisk användning."""
        mapping = {
            "positive_strong": 1.0,
            "positive": 0.5,
            "neutral": 0.0,
            "negative": -0.5,
            "negative_strong": -1.0,
        }
        return mapping[self.value]

    @classmethod
    def from_numeric(cls, x: float) -> "Valence":
        """Inverse av to_numeric. Diskretiserar kontinuerligt värde."""
        if x >= 0.75:
            return cls.POSITIVE_STRONG
        if x >= 0.25:
            return cls.POSITIVE
        if x >= -0.25:
            return cls.NEUTRAL
        if x >= -0.75:
            return cls.NEGATIVE
        return cls.NEGATIVE_STRONG


# ─── Tag-struktur (sammanslagen kategorisering) ───────────────────────


@dataclass(frozen=True)
class EpistemicTag:
    """Klassificering av en representation.

    Bär alla dimensioner som beskriver representations-statusen.
    Frozen — en representations epistemiska tag ändras inte; en ny
    representation skapas med ny tag om något förändras.
    """

    data_type: DataType
    confidence: Confidence
    mutability: Mutability
    persistence: Persistence
    memory_type: MemoryType
    valence: Valence  # OBLIGATORISK — ingen default till "neutral"

    def to_dict(self) -> dict[str, str]:
        return {
            "data_type": self.data_type.value,
            "confidence": self.confidence.value,
            "mutability": self.mutability.value,
            "persistence": self.persistence.value,
            "memory_type": self.memory_type.value,
            "valence": self.valence.value,
        }
