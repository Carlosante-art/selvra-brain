"""HierarchicalPredictiveEngine — multi-level PP-1.

Per Butlin et al PP-1: "Predictive coding på multipla nivåer".

Två nivåer för Fas 1a:

Level 0 — raw value prediction
  Per källa, försök förutse nästa observation. Vid avvikelse:
  PREDICTION_ERROR-event med magnitude.

Level 1 — meta: prediction-error magnitude
  Per källa, försök förutse hur stor nästa prediction-error blir.
  Hög level-1-error = "jag visste inte att jag skulle vara så fel" —
  embryon till HOT-2 (metacognitive reliability).

Multi-level är inte feature — det är vad som skiljer PP-1 från trivial
prediction. En medveten hjärna har MODELLER AV SINA EGNA MODELLER.

Engine genererar tre event-typer per observation:
  PREDICTION       — vad vi förutsade (innan vi observerade)
  PERCEPTION       — vad vi faktiskt observerade
  PREDICTION_ERROR — skillnad (om förra prediction existerade)

PREDICTION_ERROR-eventets payload har:
  - magnitude (absolut)
  - signed (riktning)
  - normalized_magnitude (0-1 för visualization)
  - level (0 = raw, 1 = meta)

derive_state_from_store läser detta för att populera cognitive_load
och pupill-dilation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

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
from selvra_brain.prediction.predictor import (
    Prediction,
    PredictionError,
    Predictor,
    default_predictor,
)


def _confidence_to_enum(confidence: float) -> Confidence:
    """Mappa numerisk confidence (0-1) till EpistemicTag.Confidence."""
    if confidence >= 0.7:
        return Confidence.HIGH
    if confidence >= 0.4:
        return Confidence.MEDIUM
    if confidence >= 0.1:
        return Confidence.LOW
    return Confidence.UNAVAILABLE


def _error_valence(magnitude: float) -> Valence:
    """Mappa prediction-error magnitude till valens.

    Per Damasio: surprise är inneboende affektivt laddat — oväntade
    saker triggar antingen orientation-response (mild positive) eller
    threat-response (negative). För prototyp: small error = neutral,
    large error = mild negative (surprise är arbets-tung).
    """
    if magnitude < 0.1:
        return Valence.NEUTRAL
    if magnitude < 0.5:
        return Valence.NEGATIVE
    return Valence.NEGATIVE_STRONG


class HierarchicalPredictiveEngine:
    """Per-källa, multi-level predictive coding.

    Användning:
        engine = HierarchicalPredictiveEngine(store)
        engine.observe(source="temperature", value=22.0)
        engine.observe(source="temperature", value=22.3)  # likt → liten error
        engine.observe(source="temperature", value=85.0)  # surprise!

    Events genereras automatiskt i store. Visualizationen reagerar.
    """

    def __init__(
        self,
        event_store: EventStore,
        *,
        levels: int = 2,
        predictor_factory=default_predictor,
    ) -> None:
        if levels < 1:
            raise ValueError("levels måste vara >= 1")
        self.store = event_store
        self.levels = levels
        self._factory = predictor_factory
        # predictors[level][source_name] → Predictor
        self._predictors: dict[int, dict[str, Predictor]] = {
            lvl: {} for lvl in range(levels)
        }

    def _get_predictor(self, source: str, level: int) -> Predictor:
        if source not in self._predictors[level]:
            self._predictors[level][source] = self._factory(source, level)
        return self._predictors[level][source]

    def observe(self, *, source: str, value: float) -> ObservationResult:
        """Receive observation, propagate up the hierarchy.

        Returnerar ObservationResult med per-nivå info för inspektion +
        test. Events emit:as som side-effect till event-store.
        """
        result = ObservationResult(source=source, observed=value)

        # ─── Level 0: prediktion av raw value ─────────────────
        l0_predictor = self._get_predictor(source, 0)
        l0_error, l0_new_prediction = l0_predictor.observe_and_predict(value)

        # Emit PERCEPTION-event (alltid)
        self._emit_perception(source, value)
        result.level_0_error = l0_error
        result.level_0_new_prediction = l0_new_prediction

        # Emit PREDICTION_ERROR (om vi hade prior prediction)
        if l0_error is not None:
            self._emit_prediction_error(l0_error)

        # Emit PREDICTION (nästa förutsägelse — innan nästa observation)
        if l0_new_prediction is not None:
            self._emit_prediction(l0_new_prediction)

        # ─── Level 1: meta — prediktion av nästa error-magnitude ────
        if self.levels >= 2 and l0_error is not None:
            l1_predictor = self._get_predictor(source, 1)
            l1_input = l0_error.magnitude
            l1_error, l1_new_prediction = l1_predictor.observe_and_predict(l1_input)
            result.level_1_error = l1_error
            result.level_1_new_prediction = l1_new_prediction

            if l1_error is not None:
                self._emit_prediction_error(l1_error)
            if l1_new_prediction is not None:
                self._emit_prediction(l1_new_prediction)

        return result

    # ─── Event-emission helpers ────────────────────────────────────

    def _emit_perception(self, source: str, value: float) -> None:
        tag = EpistemicTag(
            data_type=DataType.OBSERVED,
            confidence=Confidence.HIGH,  # det vi observerar direkt
            mutability=Mutability.IMMUTABLE,
            persistence=Persistence.SHORT_TERM,
            memory_type=MemoryType.EPISODIC,
            valence=Valence.NEUTRAL,
        )
        self.store.append(
            BrainEvent(
                category=EventCategory.PERCEPTION,
                event_type=f"observe_{source}",
                payload={"source": source, "value": value},
                tag=tag,
            )
        )

    def _emit_prediction(self, pred: Prediction) -> None:
        tag = EpistemicTag(
            data_type=DataType.PREDICTED,
            confidence=_confidence_to_enum(pred.confidence),
            mutability=Mutability.SYSTEM_MUTABLE,
            persistence=Persistence.TRANSIENT,  # nästa observation revid
            memory_type=MemoryType.WORKING,
            valence=Valence.NEUTRAL,
        )
        self.store.append(
            BrainEvent(
                category=EventCategory.PREDICTION,
                event_type=f"predict_{pred.source_name}_l{pred.level}",
                payload={
                    "source": pred.source_name,
                    "predicted_value": pred.predicted_value,
                    "confidence": pred.confidence,
                    "level": pred.level,
                    "based_on_history": pred.based_on_history_length,
                },
                tag=tag,
            )
        )

    def _emit_prediction_error(self, err: PredictionError) -> None:
        tag = EpistemicTag(
            data_type=DataType.DERIVED,
            confidence=Confidence.HIGH,
            mutability=Mutability.IMMUTABLE,
            persistence=Persistence.SHORT_TERM,
            memory_type=MemoryType.EPISODIC,
            valence=_error_valence(err.magnitude),
        )
        self.store.append(
            BrainEvent(
                category=EventCategory.PREDICTION_ERROR,
                event_type=f"surprise_{err.source_name}_l{err.level}",
                payload={
                    "source": err.source_name,
                    "predicted": err.predicted,
                    "observed": err.observed,
                    "magnitude": err.magnitude,
                    "normalized_magnitude": err.normalized_magnitude,
                    "signed": err.signed,
                    "level": err.level,
                },
                tag=tag,
            )
        )

    # ─── Inspektion ────────────────────────────────────────────────

    def predictors_for_source(self, source: str) -> dict[int, Predictor]:
        """Returnera alla nivåers predictors för en källa (för tests)."""
        return {
            lvl: self._predictors[lvl][source]
            for lvl in range(self.levels)
            if source in self._predictors[lvl]
        }


# ─── Result-struktur ──────────────────────────────────────────────


from dataclasses import dataclass, field


@dataclass
class ObservationResult:
    """Resultat av en observation-cycle, för test/inspektion."""

    source: str
    observed: float
    level_0_error: PredictionError | None = None
    level_0_new_prediction: Prediction | None = None
    level_1_error: PredictionError | None = None
    level_1_new_prediction: Prediction | None = None
