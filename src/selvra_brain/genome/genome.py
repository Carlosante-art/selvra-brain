"""Genome — frozen konfigurations-DNA för en agent.

Designprinciper:
- IMMUTABLE under agentens livstid (nature). Mutationer producerar nya
  Genome-objekt, ändrar aldrig existerande.
- Lagras som separat artefakt, INTE som event i EventStore. agent_born
  refererar till genome_id, men själva Genome bor utanför event-streamen.
- 16 parametrar fördelade över modulerna: perception (RPT), prediction
  (PP), workspace (GW), metacognition (HOT-2), curiosity (AE-1),
  action_model (AE-2), attention_schema (AST-1).
- Diskreta parametrar (ints för capacity/levels/windows) muteras
  separat från kontinuerliga (gaussisk perturbation).

Rekombination:
- Default: per-parameter uniform val mellan föräldrarna (oberoende
  arv per gen).
- Alternativ: crossover-punkter (en konsekutiv blocking-arv).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field, fields, replace
from typing import Any


class GenomeValidationError(ValueError):
    """Raised när en Genome-parameter ligger utanför sin tillåtna range."""


@dataclass(frozen=True)
class GenomeRange:
    """Tillåtna ranges för en genome-parameter.

    Används både för validering OCH för mutation (sigma skalas mot
    range-bredd) OCH för slumpmässig sampling.
    """

    min_value: float
    max_value: float
    is_integer: bool = False

    def clamp(self, value: float) -> float:
        return max(self.min_value, min(self.max_value, value))

    def sample(self, rng: random.Random) -> float:
        if self.is_integer:
            return rng.randint(int(self.min_value), int(self.max_value))
        return rng.uniform(self.min_value, self.max_value)


# ─── 16 parametrar (Selvras default-genome ── 2026-05-22) ─────────
#
# Default-värden matchar nuvarande config i moduler-DEFAULTS så att
# build_brain_from_genome(Genome.default()) reproducerar nuvarande
# beteende exakt.

GENOME_SCHEMA: dict[str, GenomeRange] = {
    # ─── perception (RPT) ─────────────────────────────────────
    "smoothing_weight": GenomeRange(0.0, 1.0),
    "perception_valence_scale": GenomeRange(0.001, 0.5),
    # ─── prediction (PP) ──────────────────────────────────────
    "pred_levels": GenomeRange(1, 4, is_integer=True),
    # ─── workspace (GW) ───────────────────────────────────────
    "ws_capacity": GenomeRange(1, 12, is_integer=True),
    "ws_acceptance_threshold": GenomeRange(0.0, 1.0),
    # ─── producer (workspace coupling) ─────────────────────────
    "producer_surprise_threshold": GenomeRange(0.0, 1.0),
    # ─── metacognition (HOT-2) ────────────────────────────────
    "metacog_window": GenomeRange(5, 50, is_integer=True),
    # ─── curiosity (AE-1) ─────────────────────────────────────
    "weight_reduce_uncertainty": GenomeRange(0.0, 1.0),
    "weight_investigate_surprise": GenomeRange(0.0, 1.0),
    "weight_explore_neglected": GenomeRange(0.0, 1.0),
    "weight_valence": GenomeRange(0.0, 1.0),
    "surprise_recency_window": GenomeRange(10, 100, is_integer=True),
    "drive_threshold": GenomeRange(0.0, 1.0),
    # ─── action_model (AE-2) ──────────────────────────────────
    "action_learning_rate": GenomeRange(0.0, 0.3),
    "initial_focus_width_radians": GenomeRange(0.1, math.pi),
    "initial_periphery_intensity": GenomeRange(0.0, 0.6),
    # ─── attention_schema (AST-1) ─────────────────────────────
    "attention_transition_window": GenomeRange(5, 100, is_integer=True),
}


@dataclass(frozen=True)
class Genome:
    """Immutabel konfigurations-DNA för en agent.

    Skapa via:
        Genome.default()        — Selvras "wild-type" konfiguration
        Genome.random(rng)      — slumpmässig inom GENOME_SCHEMA-ranges
        Genome(**params)        — explicit per-parameter
        a.recombine(b, rng)     — rekombinera två genomer (sexuell)
        g.mutate(rate, rng)     — perturbera (mutation)

    En agent som lever bär samma Genome hela sin livstid. Förändring
    av Genome producerar en ny Genome-instans — befintliga objekt
    muteras aldrig.
    """

    # perception
    smoothing_weight: float = 0.7
    perception_valence_scale: float = 0.05
    # prediction
    pred_levels: int = 2
    # workspace
    ws_capacity: int = 5
    ws_acceptance_threshold: float = 0.25
    # producer
    producer_surprise_threshold: float = 0.25
    # metacognition
    metacog_window: int = 15
    # curiosity
    weight_reduce_uncertainty: float = 0.35
    weight_investigate_surprise: float = 0.30
    weight_explore_neglected: float = 0.15
    weight_valence: float = 0.20
    surprise_recency_window: int = 30
    drive_threshold: float = 0.15
    # action_model
    action_learning_rate: float = 0.05
    initial_focus_width_radians: float = math.pi / 3
    initial_periphery_intensity: float = 0.2
    # attention_schema
    attention_transition_window: int = 30

    def __post_init__(self) -> None:
        self.validate()

    # ─── Validering ──────────────────────────────────────────────────
    def validate(self) -> None:
        """Verifiera att alla parametrar är inom GENOME_SCHEMA-ranges."""
        for name, range_def in GENOME_SCHEMA.items():
            value = getattr(self, name)
            if range_def.is_integer and not isinstance(value, int):
                raise GenomeValidationError(
                    f"{name} must be int, got {type(value).__name__}={value!r}"
                )
            if not (range_def.min_value <= value <= range_def.max_value):
                raise GenomeValidationError(
                    f"{name}={value} outside range "
                    f"[{range_def.min_value}, {range_def.max_value}]"
                )

    # ─── Serialisering ───────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self, *, pretty: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=2 if pretty else None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Genome":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        # Säkerställ att integer-fält faktiskt är ints (JSON kan ge float)
        for name, range_def in GENOME_SCHEMA.items():
            if range_def.is_integer and name in filtered:
                filtered[name] = int(filtered[name])
        return cls(**filtered)

    @classmethod
    def from_json(cls, payload: str) -> "Genome":
        return cls.from_dict(json.loads(payload))

    # ─── Identitet ───────────────────────────────────────────────────
    @property
    def genome_id(self) -> str:
        """Deterministic hash av parametrar. Två identiska Genome har
        samma genome_id. Användas för referens i agent_born-event."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # ─── Constructors ────────────────────────────────────────────────
    @classmethod
    def default(cls) -> "Genome":
        """Selvras default 'wild-type' Genome — matchar modul-DEFAULTS."""
        return cls()

    @classmethod
    def random(cls, rng: random.Random | None = None) -> "Genome":
        """Helt slumpmässigt Genome inom alla parameter-ranges."""
        rng = rng or random.Random()
        params: dict[str, Any] = {}
        for name, range_def in GENOME_SCHEMA.items():
            params[name] = range_def.sample(rng)
        return cls(**params)

    # ─── Rekombination (sexuell reproduction) ───────────────────────
    def recombine(
        self,
        other: "Genome",
        *,
        rng: random.Random | None = None,
        use_crossover: bool = False,
    ) -> "Genome":
        """Producera en barn-Genome från två föräldra-Genome.

        Två lägen:
          - Default (use_crossover=False): per-parameter uniform val.
            Varje parameter ärvs självständigt från någon förälder med
            50/50 sannolikhet. Detta är "free recombination" — som om
            varje gen ligger på egen kromosom.
          - use_crossover=True: en slumpmässig crossover-punkt mellan
            1 och N-1. Alla parametrar före punkten ärvs från self,
            alla efter från other. Approximerar enkel-kromosom-genetik.

        Mutation appliceras INTE här — det är separat steg.
        """
        rng = rng or random.Random()
        param_names = [f.name for f in fields(self)]
        child_params: dict[str, Any] = {}

        if use_crossover:
            n = len(param_names)
            point = rng.randint(1, n - 1)
            for i, name in enumerate(param_names):
                child_params[name] = (
                    getattr(self, name) if i < point else getattr(other, name)
                )
        else:
            for name in param_names:
                child_params[name] = (
                    getattr(self, name)
                    if rng.random() < 0.5
                    else getattr(other, name)
                )

        return Genome(**child_params)

    # ─── Mutation ───────────────────────────────────────────────────
    def mutate(
        self,
        *,
        mutation_rate: float = 0.1,
        sigma_scale: float = 0.1,
        rng: random.Random | None = None,
    ) -> "Genome":
        """Producera en perturberad Genome.

        Args:
            mutation_rate: sannolikhet per parameter att muteras.
            sigma_scale: standardavvikelse för kontinuerlig mutation,
                skalad mot range-bredd. 0.1 → sigma = 10% av (max-min).
            rng: random.Random-instans för determinism.

        Kontinuerliga parametrar: gaussisk perturbation, sedan clamp.
        Diskreta parametrar (ints): ±1 till ±3 med uniform val, clamp.
        """
        rng = rng or random.Random()
        new_params: dict[str, Any] = self.to_dict()

        for name, range_def in GENOME_SCHEMA.items():
            if rng.random() >= mutation_rate:
                continue
            current = new_params[name]
            if range_def.is_integer:
                shift = rng.randint(-3, 3)
                if shift == 0:
                    shift = 1 if rng.random() < 0.5 else -1
                new_value = int(range_def.clamp(current + shift))
            else:
                width = range_def.max_value - range_def.min_value
                sigma = width * sigma_scale
                new_value = range_def.clamp(rng.gauss(current, sigma))
            new_params[name] = new_value

        return Genome(**new_params)

    # ─── Inspektion ─────────────────────────────────────────────────
    def distance_to(self, other: "Genome") -> float:
        """Normaliserat avstånd mellan två genomer i parameter-rymden.

        För varje parameter: (|self - other| / range_width). Snittet
        över alla parametrar är [0, 1]. Nyttigt för att se hur "släkt"
        två genomer är.
        """
        total = 0.0
        for name, range_def in GENOME_SCHEMA.items():
            width = range_def.max_value - range_def.min_value
            if width == 0:
                continue
            diff = abs(getattr(self, name) - getattr(other, name))
            total += diff / width
        return total / len(GENOME_SCHEMA)

    def diff(self, other: "Genome") -> dict[str, tuple[Any, Any]]:
        """Returnera dict över parametrar som skiljer mellan två genomer."""
        out: dict[str, tuple[Any, Any]] = {}
        for f in fields(self):
            a = getattr(self, f.name)
            b = getattr(other, f.name)
            if a != b:
                out[f.name] = (a, b)
        return out

    # ─── Convenience replace (immutability-friendly) ────────────────
    def with_updates(self, **overrides: Any) -> "Genome":
        """Skapa en ny Genome med specifika fält ändrade."""
        return replace(self, **overrides)
