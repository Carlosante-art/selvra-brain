"""SymbolWorld — minimal symbol-baserad värld för agent-observation.

Detta är inte spel-engine eller fysik-simulering. Det är en strukturerad
informations-källa som agenten observerar och predikterar.

Värld-struktur:
- En cirkulär arena med N "objekt" på positioner runt agenten
- Varje objekt har egenskaper: signal_strength (kontinuerlig),
  signal_type (kategorisk), stability (predikterbarhet)
- Agenten kan "look_at(direction)" för att fokusera observation
- Världen tickar — varje tick uppdaterar objekt-state enligt deras
  inneboende dynamik (vissa stabila, vissa oscillerande, vissa
  random)

Detta är substrat för:
- PP-1: agent predikterar nästa observation per objekt
- GW: olika objekt blir candidates för attention
- AE-2 (senare): look_at är agentens action som påverkar input
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import StrEnum


class SignalDynamic(StrEnum):
    """Hur ett objekt beter sig över tid."""

    STABLE = "stable"        # konstant signal_strength
    OSCILLATING = "oscillating"  # sinus över tid
    NOISY = "noisy"          # konstant + small noise
    BURSTING = "bursting"    # mestadels lugn, sporadiska spikes
    DRIFTING = "drifting"    # långsam linjär förändring


@dataclass
class WorldObject:
    """Ett objekt i världen. Har position, dynamik, aktuell state."""

    object_id: str
    position_angle: float  # radianer från agentens origo (0 = höger, π/2 = upp)
    distance: float = 1.0   # 0-1, hur långt från agenten
    signal_type: str = "generic"
    dynamic: SignalDynamic = SignalDynamic.STABLE
    base_value: float = 0.0
    amplitude: float = 1.0
    period_ticks: int = 20  # för OSCILLATING
    noise_std: float = 0.05  # för NOISY
    burst_probability: float = 0.05  # för BURSTING
    drift_rate: float = 0.0  # för DRIFTING (per tick)
    _internal_phase: float = 0.0  # för OSCILLATING

    def tick(self, current_tick: int) -> float:
        """Producera signal-värde för denna tick."""
        if self.dynamic == SignalDynamic.STABLE:
            return self.base_value

        if self.dynamic == SignalDynamic.OSCILLATING:
            phase = 2 * math.pi * current_tick / self.period_ticks
            return self.base_value + self.amplitude * math.sin(phase + self._internal_phase)

        if self.dynamic == SignalDynamic.NOISY:
            return self.base_value + random.gauss(0, self.noise_std)

        if self.dynamic == SignalDynamic.BURSTING:
            if random.random() < self.burst_probability:
                spike = self.amplitude * random.choice([-1, 1])
                return self.base_value + spike
            return self.base_value

        if self.dynamic == SignalDynamic.DRIFTING:
            return self.base_value + self.drift_rate * current_tick

        return self.base_value


@dataclass
class Observation:
    """Vad agenten "ser" från ett objekt vid en tick."""

    object_id: str
    signal_value: float
    signal_type: str
    position_angle: float
    distance: float
    intensity: float = 1.0  # hur tydligt sett (beror på agent-fokus)


class SymbolWorld:
    """Symbol-värld som ticks och presenterar observationer till agenten.

    Användning:
        world = SymbolWorld()
        world.add_object(WorldObject(
            object_id="sun",
            position_angle=math.pi/2,  # ovanför agent
            dynamic=SignalDynamic.OSCILLATING,
        ))
        for tick in range(100):
            world.tick()
            observations = world.observe_all()
    """

    def __init__(self, *, seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)
        self._objects: dict[str, WorldObject] = {}
        self._current_tick: int = 0

    def add_object(self, obj: WorldObject) -> None:
        if obj.object_id in self._objects:
            raise ValueError(f"Object {obj.object_id} already exists")
        self._objects[obj.object_id] = obj

    def remove_object(self, object_id: str) -> None:
        if object_id in self._objects:
            del self._objects[object_id]

    def tick(self) -> None:
        """Avancera världen en tick. Objekt-tillstånd uppdateras
        internt; observationer hämtas via observe_all()."""
        self._current_tick += 1

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def objects(self) -> tuple[WorldObject, ...]:
        return tuple(self._objects.values())

    def observe_all(self) -> list[Observation]:
        """Returnera observationer från alla objekt vid aktuell tick."""
        observations = []
        for obj in self._objects.values():
            value = obj.tick(self._current_tick)
            observations.append(
                Observation(
                    object_id=obj.object_id,
                    signal_value=value,
                    signal_type=obj.signal_type,
                    position_angle=obj.position_angle,
                    distance=obj.distance,
                    intensity=1.0,
                )
            )
        return observations

    def observe_with_focus(
        self,
        *,
        focus_angle: float,
        focus_width: float = math.pi / 3,  # 60°
    ) -> list[Observation]:
        """Observation MED agent-attention. Objekt utanför fokus-cone
        får lägre intensity (men är fortfarande synliga — periphery).

        Detta är substrat för AE-2: agentens look_at(direction) påverkar
        VAD den ser tydligast.
        """
        observations = []
        half_width = focus_width / 2
        for obj in self._objects.values():
            value = obj.tick(self._current_tick)
            # Angular skillnad till fokus
            diff = self._angular_diff(obj.position_angle, focus_angle)
            if abs(diff) <= half_width:
                intensity = 1.0
            else:
                # Linjär fade till 0.2 utanför fokus (periphery)
                far_factor = (abs(diff) - half_width) / (math.pi - half_width)
                intensity = max(0.2, 1.0 - 0.8 * far_factor)
            observations.append(
                Observation(
                    object_id=obj.object_id,
                    signal_value=value,
                    signal_type=obj.signal_type,
                    position_angle=obj.position_angle,
                    distance=obj.distance,
                    intensity=intensity,
                )
            )
        return observations

    @staticmethod
    def _angular_diff(a: float, b: float) -> float:
        """Skillnad mellan två vinklar, signed, i [-π, π]."""
        d = (a - b) % (2 * math.pi)
        if d > math.pi:
            d -= 2 * math.pi
        return d


def make_default_world(*, seed: int | None = 42) -> SymbolWorld:
    """Skapar en världen med 4 standard-objekt för testning + demo.

    - sun: OSCILLATING (period 30 ticks)
    - clock: STABLE
    - bird: BURSTING (sporadiska spikes)
    - sea: NOISY (konstant + brus)
    """
    world = SymbolWorld(seed=seed)
    world.add_object(WorldObject(
        object_id="sun",
        position_angle=math.pi / 2,  # ovanför
        distance=0.8,
        signal_type="warmth",
        dynamic=SignalDynamic.OSCILLATING,
        base_value=20.0,
        amplitude=8.0,
        period_ticks=30,
    ))
    world.add_object(WorldObject(
        object_id="clock",
        position_angle=0,  # höger
        distance=0.3,
        signal_type="rhythm",
        dynamic=SignalDynamic.STABLE,
        base_value=1.0,
    ))
    world.add_object(WorldObject(
        object_id="bird",
        position_angle=-math.pi / 4,  # höger-ner
        distance=0.6,
        signal_type="movement",
        dynamic=SignalDynamic.BURSTING,
        base_value=0.0,
        amplitude=5.0,
        burst_probability=0.08,
    ))
    world.add_object(WorldObject(
        object_id="sea",
        position_angle=math.pi,  # vänster (motsatt höger)
        distance=0.9,
        signal_type="presence",
        dynamic=SignalDynamic.NOISY,
        base_value=5.0,
        noise_std=0.8,
    ))
    return world
