"""PerceptionModule — RPT-1 + RPT-2 implementation.

RPT-1 (algorithmic recurrence): processeringen har återkoppling. Föregående
scene-tillstånd påverkar hur nuvarande input integreras (residual + smoothing).
Detta är inte feed-forward — det är recurrent.

RPT-2 (organiserad integrerad representation): flera samtidiga Observations
binds till EN SceneRepresentation. Per-objekt-features, scene-level valence,
salience-distribution. Detta är substratet för temporal binding i Butlin §3.

Position i arkitekturen:
    SymbolWorld → Observation[] → PerceptionModule.process() → Scene
                                         ↓
                                  PERCEPTION-event (DERIVED)
                                  HierarchicalPredictiveEngine (per source)

Notera att PP-1 fortfarande får rå signal per source — Scene är en
parallell, integrerad representation. Båda finns. Detta speglar att
hjärnan har både moduläre signaler OCH bound representations.
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
from selvra_brain.world.symbol_world import Observation


@dataclass(frozen=True)
class ObjectFeatures:
    """Per-objekt features i scenen."""

    value: float
    value_smoothed: float  # RPT-1: blandat med föregående scene
    intensity: float
    position_angle: float
    distance: float
    delta_from_previous: float  # förändring sen föregående tick


@dataclass(frozen=True)
class SceneRepresentation:
    """Integrerad perceptuell representation (RPT-2).

    Detta är inte en lista av observations — det är ETT objekt som
    binder samtidiga observations till en scen-helhet. Per Butlin §3
    Recurrent Processing Theory: "integrated perceptual representations
    encompassing multiple modalities or feature-types".

    Frozen för att bevara temporal-binding-möjlighet (en scene-state
    kan refereras från senare events utan att ha förändrats).
    """

    tick: int
    object_features: dict[str, ObjectFeatures]
    scene_valence: Valence
    salience_distribution: dict[str, float]  # object_id -> salience [0, ∞)
    most_salient_id: str | None
    most_salient_score: float
    recurrent_magnitude: float  # total |delta| över alla objects

    def salient_objects(self, threshold: float = 0.5) -> tuple[str, ...]:
        """Objekt med salience > threshold, sorterade fallande."""
        items = [(k, v) for k, v in self.salience_distribution.items() if v > threshold]
        items.sort(key=lambda kv: kv[1], reverse=True)
        return tuple(k for k, _ in items)


@dataclass
class PerceptionModule:
    """RPT-implementation. Tar Observations från world, returnerar Scene.

    Recurrent — håller `_previous_scene` mellan calls. Smoothing-vikt
    `smoothing_weight` styr hur mycket previous bidrar (0.0 = ren feed-forward,
    1.0 = bara historisk smoothing). Default 0.7 = current dominant men
    previous synlig.

    Emittar PERCEPTION-event per process()-call med kategorin
    "scene_integrated" så att downstream (workspace, prediction, monitor)
    kan se scene-level signaler.
    """

    store: EventStore
    smoothing_weight: float = 0.7
    valence_scale: float = 0.05  # hur mycket scene-value mappar till valens
    _previous_scene: SceneRepresentation | None = field(default=None, init=False)

    def process(self, observations: list[Observation], tick: int) -> SceneRepresentation:
        """RPT-1 + RPT-2 i ett steg.

        1. Per objekt: feature-extraktion + smoothing mot previous (RPT-1).
        2. Salience-distribution: |value| * intensity per objekt.
        3. Scene-valence: viktad medel-aktivering.
        4. Recurrent-magnitude: total förändring (drift-mått).
        5. Emit PERCEPTION-event (DERIVED, IMMUTABLE).
        """
        features: dict[str, ObjectFeatures] = {}
        salience: dict[str, float] = {}
        total_signed_value = 0.0
        total_weight = 0.0
        recurrent_mag = 0.0

        prev_features = (
            self._previous_scene.object_features if self._previous_scene else {}
        )

        for obs in observations:
            prev = prev_features.get(obs.object_id)
            prev_value = prev.value if prev is not None else obs.signal_value
            delta = obs.signal_value - prev_value
            recurrent_mag += abs(delta)

            smoothed = (
                self.smoothing_weight * obs.signal_value
                + (1.0 - self.smoothing_weight) * prev_value
            )

            features[obs.object_id] = ObjectFeatures(
                value=obs.signal_value,
                value_smoothed=smoothed,
                intensity=obs.intensity,
                position_angle=obs.position_angle,
                distance=obs.distance,
                delta_from_previous=delta,
            )

            # Salience: signalmagnitud * intensity. Distance dämpar.
            distance_attenuation = 1.0 - 0.3 * obs.distance
            sal = abs(obs.signal_value) * obs.intensity * distance_attenuation
            salience[obs.object_id] = sal

            total_signed_value += obs.signal_value * obs.intensity
            total_weight += obs.intensity

        # Scene-level valence: viktad medel skalad och clamped
        if total_weight > 0:
            avg_value = total_signed_value / total_weight
        else:
            avg_value = 0.0
        scaled = max(-1.0, min(1.0, avg_value * self.valence_scale))
        scene_valence = Valence.from_numeric(scaled)

        if salience:
            most_id, most_score = max(salience.items(), key=lambda kv: kv[1])
        else:
            most_id, most_score = None, 0.0

        scene = SceneRepresentation(
            tick=tick,
            object_features=features,
            scene_valence=scene_valence,
            salience_distribution=salience,
            most_salient_id=most_id,
            most_salient_score=most_score,
            recurrent_magnitude=recurrent_mag,
        )

        self.store.append(
            BrainEvent(
                category=EventCategory.PERCEPTION,
                event_type="scene_integrated",
                payload={
                    "tick": tick,
                    "object_count": len(observations),
                    "scene_valence": scaled,
                    "max_salience": most_score,
                    "max_salience_source": most_id,
                    "recurrent_magnitude": recurrent_mag,
                    "smoothing_weight": self.smoothing_weight,
                },
                tag=EpistemicTag(
                    data_type=DataType.DERIVED,
                    confidence=Confidence.MEDIUM,
                    mutability=Mutability.IMMUTABLE,
                    persistence=Persistence.TRANSIENT,
                    memory_type=MemoryType.WORKING,
                    valence=scene_valence,
                ),
            )
        )

        self._previous_scene = scene
        return scene

    @property
    def previous_scene(self) -> SceneRepresentation | None:
        return self._previous_scene

    def reset(self) -> None:
        """Glöm previous — för testing eller "väck om"."""
        self._previous_scene = None
