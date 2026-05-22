"""Tests för SymbolWorld + Agent."""

from __future__ import annotations

import math

import pytest

from selvra_brain.world import (
    Agent,
    Observation,
    SignalDynamic,
    SymbolWorld,
    WorldObject,
    make_default_world,
)


# ─── WorldObject dynamics ──────────────────────────────────────


def test_stable_object_constant_value() -> None:
    obj = WorldObject(
        object_id="x",
        position_angle=0,
        dynamic=SignalDynamic.STABLE,
        base_value=42.0,
    )
    assert obj.tick(0) == 42.0
    assert obj.tick(100) == 42.0


def test_oscillating_object_varies_smoothly() -> None:
    obj = WorldObject(
        object_id="x",
        position_angle=0,
        dynamic=SignalDynamic.OSCILLATING,
        base_value=10.0,
        amplitude=5.0,
        period_ticks=20,
    )
    # Tick 0 → sin(0) = 0 → base = 10
    assert obj.tick(0) == pytest.approx(10.0)
    # Tick 5 (kvart av period) → sin(π/2) = 1 → base + amp = 15
    assert obj.tick(5) == pytest.approx(15.0)
    # Tick 15 (3 kvartar) → sin(3π/2) = -1 → base - amp = 5
    assert obj.tick(15) == pytest.approx(5.0)


def test_drifting_object_linear_trend() -> None:
    obj = WorldObject(
        object_id="x",
        position_angle=0,
        dynamic=SignalDynamic.DRIFTING,
        base_value=0.0,
        drift_rate=0.1,
    )
    assert obj.tick(0) == 0.0
    assert obj.tick(10) == pytest.approx(1.0)
    assert obj.tick(100) == pytest.approx(10.0)


# ─── SymbolWorld ───────────────────────────────────────────────


def test_world_starts_at_tick_zero() -> None:
    world = SymbolWorld()
    assert world.current_tick == 0


def test_world_tick_advances() -> None:
    world = SymbolWorld()
    world.tick()
    world.tick()
    assert world.current_tick == 2


def test_add_object() -> None:
    world = SymbolWorld()
    obj = WorldObject(object_id="x", position_angle=0)
    world.add_object(obj)
    assert len(world.objects) == 1


def test_cannot_add_duplicate_object() -> None:
    world = SymbolWorld()
    world.add_object(WorldObject(object_id="x", position_angle=0))
    with pytest.raises(ValueError, match="already exists"):
        world.add_object(WorldObject(object_id="x", position_angle=1))


def test_observe_all() -> None:
    world = SymbolWorld()
    world.add_object(WorldObject(object_id="a", position_angle=0, base_value=1.0))
    world.add_object(WorldObject(object_id="b", position_angle=math.pi, base_value=2.0))
    obs = world.observe_all()
    assert len(obs) == 2
    by_id = {o.object_id: o for o in obs}
    assert by_id["a"].signal_value == 1.0
    assert by_id["b"].signal_value == 2.0


def test_observe_with_focus_intensity_in_cone() -> None:
    world = SymbolWorld()
    # Objekt direkt framför
    world.add_object(WorldObject(object_id="front", position_angle=0))
    obs = world.observe_with_focus(focus_angle=0, focus_width=math.pi / 3)
    by_id = {o.object_id: o for o in obs}
    # I fokus → intensity 1.0
    assert by_id["front"].intensity == 1.0


def test_observe_with_focus_periphery_lower_intensity() -> None:
    world = SymbolWorld()
    world.add_object(WorldObject(object_id="front", position_angle=0))
    world.add_object(WorldObject(object_id="behind", position_angle=math.pi))
    obs = world.observe_with_focus(focus_angle=0, focus_width=math.pi / 3)
    by_id = {o.object_id: o for o in obs}
    # Bakom är i periferin
    assert by_id["behind"].intensity < 1.0
    assert by_id["behind"].intensity >= 0.2  # min floor


# ─── make_default_world ────────────────────────────────────────


def test_default_world_has_four_objects() -> None:
    world = make_default_world(seed=42)
    assert len(world.objects) == 4
    ids = {obj.object_id for obj in world.objects}
    assert ids == {"sun", "clock", "bird", "sea"}


def test_default_world_observable() -> None:
    world = make_default_world(seed=42)
    obs = world.observe_all()
    assert len(obs) == 4


# ─── Agent ─────────────────────────────────────────────────────


def test_agent_default_focus_zero() -> None:
    world = SymbolWorld()
    agent = Agent(name="selvra", world=world)
    assert agent.focus_angle == 0.0


def test_agent_look_at_changes_focus() -> None:
    world = SymbolWorld()
    agent = Agent(name="selvra", world=world)
    agent.look_at(math.pi / 2)
    assert agent.focus_angle == math.pi / 2


def test_agent_look_at_object_by_id() -> None:
    world = make_default_world(seed=42)
    agent = Agent(name="selvra", world=world)
    assert agent.look_at_object("sun") is True
    assert agent.focus_angle == math.pi / 2  # sun is positioned ovanför


def test_agent_look_at_unknown_object_returns_false() -> None:
    world = make_default_world(seed=42)
    agent = Agent(name="selvra", world=world)
    assert agent.look_at_object("nonexistent") is False


def test_agent_attention_vector_for_default_focus() -> None:
    world = SymbolWorld()
    agent = Agent(name="selvra", world=world)
    x, y = agent.attention_vector()
    # focus_angle=0 → cos=1, sin=0 → (1, 0) (höger)
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)


def test_agent_attention_vector_for_up() -> None:
    world = SymbolWorld()
    agent = Agent(name="selvra", world=world)
    agent.look_at(math.pi / 2)
    x, y = agent.attention_vector()
    # angle=π/2 → cos=0, sin=1, men y=-sin → -1 (upp på SVG)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(-1.0)


def test_agent_observe_uses_focus() -> None:
    world = make_default_world(seed=42)
    agent = Agent(name="selvra", world=world)
    agent.look_at_object("sun")
    obs = agent.observe()
    # Sun ska vara i fokus → intensity 1.0
    by_id = {o.object_id: o for o in obs}
    assert by_id["sun"].intensity == 1.0
