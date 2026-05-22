"""Tester för Genome (Fas 1h)."""

from __future__ import annotations

import math
import random

import pytest

from selvra_brain.core.events import EventStore
from selvra_brain.genome import (
    GENOME_SCHEMA,
    Genome,
    GenomeValidationError,
    birth_agent,
    build_brain_from_genome,
)


# ─── Immutability ─────────────────────────────────────────────────


def test_genome_is_frozen():
    g = Genome.default()
    with pytest.raises(Exception):
        g.smoothing_weight = 0.5  # type: ignore[misc]


def test_with_updates_creates_new_instance():
    g = Genome.default()
    g2 = g.with_updates(smoothing_weight=0.5)
    assert g.smoothing_weight == 0.7
    assert g2.smoothing_weight == 0.5
    assert g is not g2


# ─── Validation ──────────────────────────────────────────────────


def test_default_passes_validation():
    Genome.default()  # ska inte raisea


def test_out_of_range_fails():
    with pytest.raises(GenomeValidationError):
        Genome(smoothing_weight=1.5)


def test_negative_capacity_fails():
    with pytest.raises(GenomeValidationError):
        Genome(ws_capacity=0)


def test_non_integer_for_int_field_fails():
    with pytest.raises(GenomeValidationError):
        Genome(ws_capacity=4.5)  # type: ignore[arg-type]


# ─── Serialisering ───────────────────────────────────────────────


def test_serialize_roundtrip_preserves_genome_id():
    g = Genome.default()
    j = g.to_json()
    g2 = Genome.from_json(j)
    assert g.genome_id == g2.genome_id
    assert g == g2


def test_from_dict_coerces_floats_to_ints_for_integer_fields():
    g = Genome.default()
    d = g.to_dict()
    d["ws_capacity"] = 5.0  # JSON kan ge float
    g2 = Genome.from_dict(d)
    assert g2.ws_capacity == 5
    assert isinstance(g2.ws_capacity, int)


def test_from_dict_ignores_unknown_fields():
    g = Genome.default()
    d = g.to_dict()
    d["unknown_field"] = "ignore me"
    g2 = Genome.from_dict(d)
    assert g == g2


# ─── Genome ID ───────────────────────────────────────────────────


def test_genome_id_is_deterministic():
    a = Genome.default()
    b = Genome.default()
    assert a.genome_id == b.genome_id


def test_genome_id_differs_on_change():
    a = Genome.default()
    b = a.with_updates(smoothing_weight=0.5)
    assert a.genome_id != b.genome_id


def test_genome_id_is_short_hash():
    g = Genome.default()
    assert len(g.genome_id) == 16


# ─── Random + recombine ──────────────────────────────────────────


def test_random_produces_valid_genome():
    rng = random.Random(42)
    g = Genome.random(rng)
    g.validate()  # ska inte raisea


def test_random_with_seed_is_deterministic():
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    a = Genome.random(rng_a)
    b = Genome.random(rng_b)
    assert a == b


def test_recombine_uniform_per_parameter():
    """Varje parameter ska komma från en av föräldrarna."""
    p1 = Genome.default()
    p2 = Genome.random(random.Random(7))
    child = p1.recombine(p2, rng=random.Random(42))
    for name in [f for f in GENOME_SCHEMA]:
        val = getattr(child, name)
        assert val == getattr(p1, name) or val == getattr(p2, name)


def test_recombine_seed_deterministic():
    p1 = Genome.default()
    p2 = Genome.random(random.Random(100))
    c1 = p1.recombine(p2, rng=random.Random(42))
    c2 = p1.recombine(p2, rng=random.Random(42))
    assert c1 == c2


def test_recombine_crossover_preserves_blocks():
    """Crossover-läge: barnet ärver konsekutiva block från föräldrar."""
    p1 = Genome.default()
    p2 = Genome.random(random.Random(50))
    child = p1.recombine(p2, rng=random.Random(42), use_crossover=True)
    # Varje param ska fortfarande matcha någon förälder
    from dataclasses import fields
    for f in fields(Genome):
        val = getattr(child, f.name)
        assert val == getattr(p1, f.name) or val == getattr(p2, f.name)


# ─── Mutation ────────────────────────────────────────────────────


def test_mutate_returns_new_genome():
    g = Genome.default()
    m = g.mutate(mutation_rate=1.0, rng=random.Random(42))
    assert m is not g


def test_mutate_zero_rate_preserves():
    g = Genome.default()
    m = g.mutate(mutation_rate=0.0, rng=random.Random(42))
    assert m == g


def test_mutate_preserves_int_types():
    g = Genome.default()
    m = g.mutate(mutation_rate=1.0, rng=random.Random(42))
    for name, range_def in GENOME_SCHEMA.items():
        if range_def.is_integer:
            assert isinstance(getattr(m, name), int), f"{name} should be int"


def test_mutate_stays_in_range():
    g = Genome.default()
    for seed in range(20):
        m = g.mutate(mutation_rate=1.0, sigma_scale=1.0, rng=random.Random(seed))
        m.validate()  # ska inte raisea


def test_mutate_deterministic_with_seed():
    g = Genome.default()
    m1 = g.mutate(mutation_rate=0.5, rng=random.Random(42))
    m2 = g.mutate(mutation_rate=0.5, rng=random.Random(42))
    assert m1 == m2


# ─── distance + diff ────────────────────────────────────────────


def test_distance_to_self_is_zero():
    g = Genome.default()
    assert g.distance_to(g) == 0.0


def test_distance_to_extreme_is_bounded():
    g1 = Genome.default()
    g2 = Genome.random(random.Random(99))
    d = g1.distance_to(g2)
    assert 0.0 <= d <= 1.0


def test_diff_empty_for_identical():
    g = Genome.default()
    assert g.diff(g) == {}


def test_diff_lists_changed_fields():
    g = Genome.default()
    g2 = g.with_updates(smoothing_weight=0.5, ws_capacity=8)
    d = g.diff(g2)
    assert set(d.keys()) == {"smoothing_weight", "ws_capacity"}


# ─── Builder ─────────────────────────────────────────────────────


def test_builder_applies_genome_to_workspace_capacity():
    g = Genome.default().with_updates(ws_capacity=10)
    brain = build_brain_from_genome(g)
    # Workspace har capacity-attribute (verifierad via init)
    assert brain.workspace.capacity == 10


def test_builder_applies_genome_to_perception_smoothing():
    g = Genome.default().with_updates(smoothing_weight=0.3)
    brain = build_brain_from_genome(g)
    assert brain.perception.smoothing_weight == 0.3


def test_builder_applies_genome_to_curiosity_weights():
    g = Genome.default().with_updates(weight_investigate_surprise=0.9)
    brain = build_brain_from_genome(g)
    assert brain.curiosity.weight_investigate_surprise == 0.9


def test_builder_applies_genome_to_action_model_focus_width():
    g = Genome.default().with_updates(initial_focus_width_radians=math.pi / 6)
    brain = build_brain_from_genome(g)
    assert brain.action_model.estimated_focus_width == pytest.approx(math.pi / 6)


def test_builder_applies_genome_to_periphery_intensity():
    g = Genome.default().with_updates(initial_periphery_intensity=0.4)
    brain = build_brain_from_genome(g)
    assert brain.action_model._periphery_intensity_low == pytest.approx(0.4)


def test_builder_reuses_existing_store():
    g = Genome.default()
    store = EventStore()
    brain = build_brain_from_genome(g, store=store)
    assert brain.store is store


# ─── Birth event ─────────────────────────────────────────────────


def test_birth_emits_agent_born_event():
    g = Genome.default()
    brain = build_brain_from_genome(g)
    ev = birth_agent(store=brain.store, genome=g, name="selvra")
    assert ev.event_type == "agent_born"
    assert ev.payload["name"] == "selvra"
    assert ev.payload["genome_id"] == g.genome_id


def test_birth_event_is_in_store():
    g = Genome.default()
    brain = build_brain_from_genome(g)
    initial_size = len(brain.store)
    birth_agent(store=brain.store, genome=g)
    assert len(brain.store) == initial_size + 1


def test_birth_event_contains_full_genome():
    g = Genome.default()
    brain = build_brain_from_genome(g)
    ev = birth_agent(store=brain.store, genome=g)
    assert ev.payload["genome"] == g.to_dict()
