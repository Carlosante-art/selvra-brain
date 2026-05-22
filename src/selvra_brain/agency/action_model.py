"""ActionEffectModel — AE-2 implementation.

AE-2 per Butlin et al. 2023: "Embodiment: modeling output–input
contingencies, including with respect to the body".

Konkret: agenten har en INTERN MODELL av hur hennes look_at(angle)
påverkar input. Innan action: prediktion av nästa observation-intensity
per källa. Efter action: faktisk intensity. Diff = action-effect-error.

Detta är PP-1 applied to actions — predictive coding över sin egen
embodiment. Skiljt från PP-1 över signaler (som är predictive coding
över världen).

Den interna modellen för Fas 1 är minimal men strukturellt korrekt:
- För varje källa, given (current_focus, intended_focus): predict intensity
- Baserat på en lärd "focus_width" som agenten själv estimerar
- Update via gradient: faktisk intensity vs predicted → justera estimate

Detta är ingen sofistikerad inverse-model — det är en första-iteration.
Men det är embodiment: agenten har en modell AV sin egen kropp + dess
relation till världen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
class ActionEffectPrediction:
    """Prediktion: efter look_at(target), vad blir intensity per källa?"""

    target_angle: float
    predicted_intensities: dict[str, float]  # object_id → predicted intensity
    estimated_focus_width: float


@dataclass(frozen=True)
class ActionEffectError:
    """Diff mellan predicted och actual efter en action."""

    source: str
    predicted_intensity: float
    actual_intensity: float
    error_magnitude: float


@dataclass
class ActionEffectModel:
    """Modell av hur look_at påverkar perception.

    Internal state:
      - estimated_focus_width: agentens nuvarande tro om sitt fokus-fält
        (initial 60°, justeras via observation)
      - per_source_calibration: per-source minimum intensity (mätt över tid)

    Per tick:
      1. predict_effect(target_angle) → ActionEffectPrediction
      2. (agenten utför look_at)
      3. (världen producerar nya observations)
      4. observe_effect(actual_intensities) → updates + ActionEffectError per källa
    """

    store: EventStore
    learning_rate: float = 0.05
    estimated_focus_width: float = math.pi / 3  # 60° initial guess
    _periphery_intensity_low: float = 0.2  # initial guess
    _last_prediction: ActionEffectPrediction | None = field(default=None, init=False)
    _last_source_positions: dict[str, float] = field(default_factory=dict, init=False)
    _prediction_count: int = field(default=0, init=False)
    _effect_count: int = field(default=0, init=False)
    _last_error_magnitude: float = field(default=0.0, init=False)

    def predict_effect(
        self,
        *,
        target_angle: float,
        source_positions: dict[str, float],
    ) -> ActionEffectPrediction:
        """Prediktera intensity per källa efter look_at(target_angle).

        Modellen: intensity = 1.0 om |angle_diff| <= focus_width/2,
        annars linjär fade till _periphery_intensity_low.

        Detta är agentens INTERN modell — den behöver inte matcha
        världens implementation perfekt. Det är hela poängen med
        prediction-error: skillnaden mäter modellens kalibrering.
        """
        half = self.estimated_focus_width / 2
        predictions: dict[str, float] = {}
        for src, pos in source_positions.items():
            diff = self._angular_diff(pos, target_angle)
            if abs(diff) <= half:
                intensity = 1.0
            else:
                far_factor = (abs(diff) - half) / (math.pi - half)
                far_factor = max(0.0, min(1.0, far_factor))
                intensity = max(
                    self._periphery_intensity_low,
                    1.0 - (1.0 - self._periphery_intensity_low) * far_factor,
                )
            predictions[src] = intensity

        prediction = ActionEffectPrediction(
            target_angle=target_angle,
            predicted_intensities=predictions,
            estimated_focus_width=self.estimated_focus_width,
        )

        self._last_prediction = prediction
        self._last_source_positions = dict(source_positions)
        self._prediction_count += 1

        # Emit PREDICTION-event tag:ad som SELF_REPORTED (om sin egen action)
        self.store.append(
            BrainEvent(
                category=EventCategory.PREDICTION,
                event_type="action_effect_prediction",
                payload={
                    "target_angle": target_angle,
                    "predicted_intensities": predictions,
                    "estimated_focus_width": self.estimated_focus_width,
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
        return prediction

    def observe_effect(
        self,
        *,
        actual_intensities: dict[str, float],
    ) -> tuple[ActionEffectError, ...]:
        """Jämför actual mot last prediction. Update model. Emit events.

        Returns: en ActionEffectError per source som hade en prediction.
        """
        if self._last_prediction is None:
            return ()

        errors: list[ActionEffectError] = []
        total_error = 0.0
        for src, actual in actual_intensities.items():
            predicted = self._last_prediction.predicted_intensities.get(src)
            if predicted is None:
                continue
            mag = abs(actual - predicted)
            errors.append(
                ActionEffectError(
                    source=src,
                    predicted_intensity=predicted,
                    actual_intensity=actual,
                    error_magnitude=mag,
                )
            )
            total_error += mag

        # Learning: justera focus_width
        # Om actual generellt > predicted för periphery → fokus är bredare
        # Om actual generellt < predicted för centrum → fokus är smalare
        # Enkel heuristik: jämför periphery-intensity-snitt
        if errors:
            self._update_estimates(errors)
            avg_error = total_error / len(errors)
        else:
            avg_error = 0.0

        self._last_error_magnitude = avg_error
        self._effect_count += 1

        # Emit ACTION_EFFECT med samlat error
        valence = (
            Valence.NEUTRAL if avg_error < 0.15
            else Valence.NEGATIVE if avg_error < 0.35
            else Valence.NEGATIVE_STRONG
        )
        self.store.append(
            BrainEvent(
                category=EventCategory.ACTION_EFFECT,
                event_type="action_effect_observed",
                payload={
                    "avg_error_magnitude": avg_error,
                    "max_error_magnitude": max((e.error_magnitude for e in errors), default=0.0),
                    "source_count": len(errors),
                    "estimated_focus_width_after_update": self.estimated_focus_width,
                    "periphery_intensity_low_after_update": self._periphery_intensity_low,
                },
                tag=EpistemicTag(
                    data_type=DataType.DERIVED,
                    confidence=Confidence.MEDIUM,
                    mutability=Mutability.IMMUTABLE,
                    persistence=Persistence.TRANSIENT,
                    memory_type=MemoryType.WORKING,
                    valence=valence,
                ),
            )
        )

        return tuple(errors)

    def _update_estimates(self, errors: list[ActionEffectError]) -> None:
        """Gradient-update av focus_width + periphery_intensity_low.

        Enkel: räkna ut residual för "in focus" (actual nära 1.0) och
        "peripheri" (actual << 1.0) separat. Justera estimat åt rätt håll.
        """
        in_focus = [e for e in errors if e.predicted_intensity > 0.85]
        in_periphery = [e for e in errors if e.predicted_intensity < 0.5]

        if in_focus:
            # Om actual i fokus < predicted → focus_width är overestimated
            avg_actual_focus = sum(e.actual_intensity for e in in_focus) / len(in_focus)
            error = avg_actual_focus - 1.0  # negative om actual < 1
            self.estimated_focus_width = max(
                math.pi / 12,  # min 15°
                min(
                    math.pi,  # max 180°
                    self.estimated_focus_width + self.learning_rate * error * math.pi / 2,
                ),
            )

        if in_periphery:
            avg_actual_periphery = sum(
                e.actual_intensity for e in in_periphery
            ) / len(in_periphery)
            avg_predicted_periphery = sum(
                e.predicted_intensity for e in in_periphery
            ) / len(in_periphery)
            error = avg_actual_periphery - avg_predicted_periphery
            self._periphery_intensity_low = max(
                0.05,
                min(0.6, self._periphery_intensity_low + self.learning_rate * error),
            )

    @staticmethod
    def _angular_diff(a: float, b: float) -> float:
        """Signed angular diff i [-π, π]."""
        d = (a - b) % (2 * math.pi)
        if d > math.pi:
            d -= 2 * math.pi
        return d

    @property
    def last_error_magnitude(self) -> float:
        return self._last_error_magnitude

    def stats(self) -> dict:
        return {
            "prediction_count": self._prediction_count,
            "effect_count": self._effect_count,
            "estimated_focus_width": self.estimated_focus_width,
            "periphery_intensity_low": self._periphery_intensity_low,
            "last_error_magnitude": self._last_error_magnitude,
        }
