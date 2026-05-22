"""Predictor abstractions för PP-1.

Per Butlin et al PP-1: predictive coding på multipla nivåer. Detta är
substratet — abstrakt Predictor-bas + tre konkreta baselines:

- ConstantPredictor: alltid samma värde (trivial baseline)
- MovingAveragePredictor: snitt över senaste N
- LinearTrendPredictor: linjär extrapolation från senaste delta

Varje Predictor har INTERNAL state — den ackumulerar history och
uppdaterar sin internal model genom update(observed). Den är inte
stateless function; det är medvetet — en medvetenhets-relevant
prediktor MÅSTE ha minne.

Multi-level-hierarki sker i engine.py — predictors här är atomic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Prediction:
    """En förutsägelse om vad nästa observation kommer vara.

    Inkluderar confidence — högre confidence = predikatorn tycker den
    har bra modell av källan. Confidence är input till HOT-2 (meta-
    cognitive reliability) senare.
    """

    source_name: str
    predicted_value: float
    confidence: float  # 0-1
    level: int  # 0 = raw value, 1 = meta (predict error), ...
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    based_on_history_length: int = 0


@dataclass(frozen=True)
class PredictionError:
    """Skillnad mellan vad vi förutsade och vad vi observerade.

    magnitude är absolut skillnad. signed är signerad (negativ = vi
    förutsade för högt; positiv = vi förutsade för lågt). Båda är
    användbara: magnitude driver cognitive_load + pupill-dilation;
    signed kan driva ögonbryns-riktning (upp = positiv surprise).
    """

    source_name: str
    predicted: float
    observed: float
    magnitude: float
    signed: float
    level: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    @property
    def normalized_magnitude(self) -> float:
        """Normaliserad till [0, 1] med soft cap.

        Magnitude är obegränsad. För visualization vill vi [0, 1].
        Vi använder x / (x + 1) som soft cap: 0 → 0, 1 → 0.5, ∞ → 1.
        Detta ger smooth dilation utan hard clipping.
        """
        x = self.magnitude
        return x / (x + 1.0) if x >= 0 else 0.0


# ─── Abstract base ────────────────────────────────────────────────


class Predictor(ABC):
    """Abstrakt predictor för en specifik (källa, nivå)-kombination.

    Subklasser implementerar predict() + update(). Engine.py
    orchestrerar dem över multipla källor och nivåer.
    """

    def __init__(self, source_name: str, level: int = 0) -> None:
        self.source_name = source_name
        self.level = level
        self._last_prediction: Prediction | None = None
        self._observation_count: int = 0

    @abstractmethod
    def predict(self) -> Prediction | None:
        """Generera prediktion baserat på nuvarande internal state.

        Returnerar None om predictor inte har tillräcklig data ännu
        (cold-start). Annars Prediction med confidence.
        """

    @abstractmethod
    def update(self, observed: float) -> None:
        """Uppdatera internal state med ny observation."""

    def observe_and_predict(self, observed: float) -> tuple[PredictionError | None, Prediction | None]:
        """Kombi: jämför observed mot senaste prediction, uppdatera, nästa prediction.

        Detta är typiska use-case för engine.py. Returnerar:
        - (error, new_prediction) — vid varm start
        - (None, None) — vid kall start, första observation
        - (None, new_prediction) — efter första update, vi har prediction
          för nästa men inget error att jämföra med
        """
        error: PredictionError | None = None
        if self._last_prediction is not None:
            magnitude = abs(self._last_prediction.predicted_value - observed)
            signed = observed - self._last_prediction.predicted_value
            error = PredictionError(
                source_name=self.source_name,
                predicted=self._last_prediction.predicted_value,
                observed=observed,
                magnitude=magnitude,
                signed=signed,
                level=self.level,
            )

        self.update(observed)
        self._observation_count += 1

        new_prediction = self.predict()
        self._last_prediction = new_prediction
        return error, new_prediction

    @property
    def observation_count(self) -> int:
        return self._observation_count


# ─── Konkreta predictors ─────────────────────────────────────────


class ConstantPredictor(Predictor):
    """Trivial baseline: alltid förutse samma konstant.

    Nytta: sanity-check + tester. Inte tänkt att vara meningsfull
    i drift.
    """

    def __init__(
        self,
        source_name: str,
        constant: float = 0.0,
        level: int = 0,
    ) -> None:
        super().__init__(source_name, level)
        self.constant = constant

    def predict(self) -> Prediction | None:
        return Prediction(
            source_name=self.source_name,
            predicted_value=self.constant,
            confidence=0.5,
            level=self.level,
            based_on_history_length=self._observation_count,
        )

    def update(self, observed: float) -> None:  # noqa: ARG002
        # Constant predictor lär aldrig
        pass


class MovingAveragePredictor(Predictor):
    """Prediktion = medelvärde över senaste N observationer.

    Confidence växer med antal observationer (cold-start → 0, varm → 1).
    Standardavvikelse i historiken minskar confidence (instabil källa
    → osäker prediktor).
    """

    def __init__(
        self,
        source_name: str,
        window: int = 5,
        level: int = 0,
    ) -> None:
        super().__init__(source_name, level)
        self.window = window
        self._history: deque[float] = deque(maxlen=window)

    def predict(self) -> Prediction | None:
        if not self._history:
            return None
        avg = sum(self._history) / len(self._history)
        # Confidence: ramp upp med history-storlek, ner med spridning
        history_fill = len(self._history) / self.window
        if len(self._history) >= 2:
            mean = avg
            variance = sum((x - mean) ** 2 for x in self._history) / len(self._history)
            stdev = variance**0.5
            # Soft penalty på spridning: 1 / (1 + stdev) som faktor
            stability = 1.0 / (1.0 + stdev)
        else:
            stability = 0.5
        confidence = max(0.0, min(1.0, history_fill * stability))
        return Prediction(
            source_name=self.source_name,
            predicted_value=avg,
            confidence=confidence,
            level=self.level,
            based_on_history_length=len(self._history),
        )

    def update(self, observed: float) -> None:
        self._history.append(observed)


class LinearTrendPredictor(Predictor):
    """Prediktion = senaste värde + senaste delta (linjär extrapolation).

    Reagerar på trender. Vid stadigt stigande/sjunkande input får den
    låg prediction-error. Vid abrupt skift får den hög error.
    """

    def __init__(self, source_name: str, level: int = 0) -> None:
        super().__init__(source_name, level)
        self._last_observed: float | None = None
        self._last_delta: float = 0.0
        self._error_history: deque[float] = deque(maxlen=10)

    def predict(self) -> Prediction | None:
        if self._last_observed is None:
            return None
        next_value = self._last_observed + self._last_delta
        # Confidence baseras på senaste prediction-error-history
        if self._error_history:
            avg_error = sum(self._error_history) / len(self._error_history)
            # Soft mapping: små errors → hög confidence
            confidence = max(0.0, min(1.0, 1.0 / (1.0 + avg_error)))
        else:
            confidence = 0.5
        return Prediction(
            source_name=self.source_name,
            predicted_value=next_value,
            confidence=confidence,
            level=self.level,
            based_on_history_length=self._observation_count,
        )

    def update(self, observed: float) -> None:
        if self._last_observed is not None:
            new_delta = observed - self._last_observed
            # Lägg till senaste prediction-error till history
            if self._last_prediction is not None:
                err = abs(self._last_prediction.predicted_value - observed)
                self._error_history.append(err)
            self._last_delta = new_delta
        self._last_observed = observed


# ─── Factory hjälpare ────────────────────────────────────────────


def default_predictor(source_name: str, level: int = 0) -> Predictor:
    """Returnera default-predictor per nivå.

    Level 0 (raw value) → LinearTrend (bra på most signals)
    Level 1+ (meta) → MovingAverage (mer stabil för derivat-signals)
    """
    if level == 0:
        return LinearTrendPredictor(source_name, level=level)
    return MovingAveragePredictor(source_name, window=8, level=level)
