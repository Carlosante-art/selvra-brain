"""MetacognitiveMonitor — HOT-2 reliability-skattning.

Per Butlin et al HOT-2: "Metakognitiv övervakning som skiljer pålitliga
från opålitliga representationer".

Pipeline:
  PredictiveEngine genererar L0 + L1 prediction-errors
       ↓
  L1-error förutsäger storleken på L0-error
       ↓
  Aktuell L1-prediction → "förväntad osäkerhet"
       ↓
  Reliability = inversion av förväntad osäkerhet
       ↓
  Hög reliability → systemet säger "jag är pålitlig nu"
  Låg reliability → systemet säger "jag är osäker just nu"

Visualization: aura-opacitet och aura-stabilitet drivs av reliability.

Vad detta INTE är:
- Inte phenomenologisk metakognition. Vi mäter prediction-error-statistik.
- Inte ett bevis på medvetenhet. HOT-2 är en indikator, inte ett claim.
- Inte över-tolkbar. När reliability = 0.7 betyder det att L1-predictorn
  förutsade liten error. Inget mer. Inget mindre.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
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
from selvra_brain.prediction.engine import HierarchicalPredictiveEngine


@dataclass(frozen=True)
class ReliabilityAssessment:
    """En metakognitiv self-report per källa.

    expected_error: vad L1-predictorn predikterade som nästa L0-error
    actual_error: vad faktiska L0-error blev
    reliability: 0-1, hur väl-kalibrerad var L1
    """

    source_name: str
    expected_error: float
    actual_error: float
    reliability: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    @classmethod
    def from_errors(
        cls, source_name: str, expected: float, actual: float
    ) -> ReliabilityAssessment:
        """Beräkna reliability från (expected, actual) error pair.

        Reliability = 1 / (1 + |expected - actual|)
        Perfekt kalibrering (expected == actual) → 1.0
        Stora avvikelser → låg reliability

        Detta är mjuk monotont mappad — gradvis sjunker, ingen cliff.
        """
        miss = abs(expected - actual)
        return cls(
            source_name=source_name,
            expected_error=expected,
            actual_error=actual,
            reliability=1.0 / (1.0 + miss),
        )


class MetacognitiveMonitor:
    """HOT-2 reliability-tracker.

    Lyssnar på prediction-errors från en HierarchicalPredictiveEngine
    och beräknar reliability per källa över tid. Global reliability =
    aggregat över alla källor (vägd av observation-count).

    Användning:

        engine = HierarchicalPredictiveEngine(store, levels=2)
        monitor = MetacognitiveMonitor(engine, store)

        # Efter varje observation:
        result = engine.observe(source="temp", value=20.0)
        monitor.update_from_result(result)

        # Inspektera:
        rel = monitor.global_reliability()  # 0-1
    """

    def __init__(
        self,
        engine: HierarchicalPredictiveEngine,
        event_store: EventStore,
        *,
        window: int = 20,
        emit_events: bool = True,
    ) -> None:
        self.engine = engine
        self.store = event_store
        self.window = window
        self.emit_events = emit_events
        # Per-source rolling history av reliability-assessments
        self._history: dict[str, deque[ReliabilityAssessment]] = {}

    def update_from_result(self, observation_result: Any) -> ReliabilityAssessment | None:
        """Bedöm reliability baserat på ObservationResult från engine.

        Vi behöver att BÅDE level-0 (faktisk error) OCH level-1
        (vad förutsade vi att error skulle bli) finns. Detta händer
        bara från ~3:e observation och framåt.

        Returnerar None om data är otillräcklig än.
        """
        l0_err = getattr(observation_result, "level_0_error", None)
        l1_err = getattr(observation_result, "level_1_error", None)
        l1_new = getattr(observation_result, "level_1_new_prediction", None)

        if l0_err is None:
            return None

        # Vi vill jämföra: vad förutsade L1 INNAN denna observation
        # som nästa L0-error? Det fanns prior om l1_err finns (l1_err
        # beräknades från prior L1-prediction vs faktisk L0-error).
        if l1_err is None:
            # Vi har bara L0-error än, ingen L1-prior att jämföra mot.
            # Reliability är då okänd — vi behöver mer data.
            return None

        # l1_err.predicted = vad L1 förutsade som nästa L0-magnitude
        # l1_err.observed = faktisk L0-magnitude
        assessment = ReliabilityAssessment.from_errors(
            source_name=l0_err.source_name,
            expected=l1_err.predicted,
            actual=l1_err.observed,
        )

        # Lagra i history
        if assessment.source_name not in self._history:
            self._history[assessment.source_name] = deque(maxlen=self.window)
        self._history[assessment.source_name].append(assessment)

        if self.emit_events:
            self._emit_metacognition_event(assessment)

        return assessment

    def reliability_for_source(self, source_name: str) -> float | None:
        """Aktuell reliability för en specifik källa.

        Returnerar None om vi inte har data än.
        Annars: snitt över rolling window.
        """
        history = self._history.get(source_name)
        if not history:
            return None
        return sum(a.reliability for a in history) / len(history)

    def global_reliability(self) -> float:
        """Aggregat över alla källor.

        Vägt snitt av per-source reliability, viktat av observation-count.
        Default 0.5 om vi inte har data än (matchar BrainVisualState
        default).
        """
        if not self._history:
            return 0.5
        total_weight = 0
        weighted_sum = 0.0
        for src, history in self._history.items():
            if not history:
                continue
            weight = len(history)
            avg = sum(a.reliability for a in history) / weight
            weighted_sum += avg * weight
            total_weight += weight
        if total_weight == 0:
            return 0.5
        return weighted_sum / total_weight

    def stats(self) -> dict[str, Any]:
        return {
            "sources_tracked": len(self._history),
            "global_reliability": self.global_reliability(),
            "per_source": {
                src: {
                    "reliability": self.reliability_for_source(src),
                    "history_size": len(hist),
                }
                for src, hist in self._history.items()
            },
        }

    # ─── Event-emission ──────────────────────────────────────────

    def _emit_metacognition_event(self, assessment: ReliabilityAssessment) -> None:
        # Reliability → confidence i tag. Hög reliability → HIGH.
        if assessment.reliability >= 0.7:
            conf = Confidence.HIGH
        elif assessment.reliability >= 0.4:
            conf = Confidence.MEDIUM
        else:
            conf = Confidence.LOW

        tag = EpistemicTag(
            data_type=DataType.SELF_REPORTED,
            confidence=conf,
            mutability=Mutability.SYSTEM_MUTABLE,
            persistence=Persistence.TRANSIENT,
            memory_type=MemoryType.WORKING,
            valence=Valence.NEUTRAL,
        )

        self.store.append(
            BrainEvent(
                category=EventCategory.METACOGNITION,
                event_type=f"reliability_{assessment.source_name}",
                payload={
                    "source": assessment.source_name,
                    "expected_error": assessment.expected_error,
                    "actual_error": assessment.actual_error,
                    "reliability": assessment.reliability,
                    "global_reliability": self.global_reliability(),
                },
                tag=tag,
            )
        )
