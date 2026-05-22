"""Demo: två olika Genome ger olika beteenden i samma värld.

Detta är substratet för Fas 2 (Adam) och Fas 3 (barn). Här visar vi
bara att två agenter med olika konfig får olika observable behavior
över samma värld + samma seed.

Vad du ser:
  - "Selvra" (default Genome): balanserad explorer-default
  - "Adam-prototype" (annan Genome): högre surprise-tolerance, bredare
    focus, snabbare lärning på action-model

Båda kör samma SymbolWorld med seed=42. Skillnaden i behavior kommer
ENDAST från Genome.

Kör:
    python3 -m examples.genome_demo --duration 30
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selvra_brain.agency.types import ActionType
from selvra_brain.genome import (
    Genome,
    birth_agent,
    build_brain_from_genome,
)
from selvra_brain.world import Agent, make_default_world


def run_agent(genome: Genome, name: str, duration_seconds: int, world_seed: int) -> dict:
    """Kör en agent med en specifik Genome och returnera summary."""
    brain = build_brain_from_genome(genome)
    birth_agent(store=brain.store, genome=genome, name=name)

    world = make_default_world(seed=world_seed)
    agent = Agent(name=name, world=world)
    positions = {o.object_id: o.position_angle for o in world.objects}

    start = time.time()
    tick = 0

    while time.time() - start < duration_seconds:
        world.tick()
        tick += 1
        observations = agent.observe()
        agent.tick_idle_drift()
        brain.perception.process(observations, tick=tick)
        for obs in observations:
            brain.curiosity.record_observation(obs.object_id)
            value = obs.signal_value * obs.intensity
            result = brain.prediction.observe(source=obs.object_id, value=value)
            if result.level_0_error is not None:
                brain.producer.handle_error(result.level_0_error)
            brain.monitor.update_from_result(result)
        actual = {obs.object_id: obs.intensity for obs in observations}
        brain.action_model.observe_effect(actual_intensities=actual)
        rel = {
            s: i["reliability"]
            for s, i in brain.monitor.stats()["per_source"].items()
            if i["reliability"] is not None
        }
        intent = brain.curiosity.decide(
            source_positions=positions,
            reliability_per_source=rel,
        )
        if intent.action == ActionType.LOOK_AT:
            brain.action_model.predict_effect(
                target_angle=intent.target_angle,
                source_positions=positions,
            )
            agent.look_at(intent.target_angle)
        brain.attention_schema.update_from_intent(intent)
        time.sleep(0.01)  # snabb tick

    schema_stats = brain.attention_schema.stats()
    action_stats = brain.action_model.stats()
    curiosity_stats = brain.curiosity.stats()

    return {
        "name": name,
        "genome_id": genome.genome_id,
        "ticks": tick,
        "events": len(brain.store),
        "workspace": brain.workspace.stats(),
        "reliability": brain.monitor.global_reliability(),
        "action_count": curiosity_stats["action_count"],
        "transitions": schema_stats["transitions_total"],
        "duration_on_target": schema_stats["duration_on_target"],
        "obs_counts": curiosity_stats["observation_counts"],
        "action_model": action_stats,
        "current_goal": schema_stats["current_goal"],
        "current_target": schema_stats["current_target"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Two Genomes, same world.")
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Selvra: default
    selvra_genome = Genome.default()

    # Adam-prototype: olika prioriteringar
    adam_genome = Genome.default().with_updates(
        smoothing_weight=0.4,  # mindre recurrent — mer reaktiv
        ws_capacity=7,  # bredare uppmärksamhet
        ws_acceptance_threshold=0.4,  # mer selektiv
        weight_reduce_uncertainty=0.20,  # mindre angelägen att kalibrera
        weight_investigate_surprise=0.45,  # mer surprise-attraherad
        weight_explore_neglected=0.20,  # mer utforskande
        weight_valence=0.15,
        action_learning_rate=0.10,  # snabbare embodiment-lärning
        initial_focus_width_radians=math.pi / 2,  # bredare initial fokus
        attention_transition_window=15,  # snabbare memory på skift
    )

    print(f"Selvra Genome  : {selvra_genome.genome_id}")
    print(f"Adam   Genome  : {adam_genome.genome_id}")
    print(f"Genome-distance: {selvra_genome.distance_to(adam_genome):.3f}")
    print(f"Diff fields    : {list(selvra_genome.diff(adam_genome).keys())}")
    print()

    print(f"Running each for {args.duration}s with world seed={args.seed}…")
    print()

    selvra_summary = run_agent(
        selvra_genome, "selvra", args.duration, world_seed=args.seed
    )
    adam_summary = run_agent(adam_genome, "adam", args.duration, world_seed=args.seed)

    print()
    print("─── Selvra ───────────────────────────────────")
    for k, v in selvra_summary.items():
        print(f"  {k}: {v}")
    print()
    print("─── Adam ─────────────────────────────────────")
    for k, v in adam_summary.items():
        print(f"  {k}: {v}")
    print()
    print("─── Skillnader (samma värld, olika Genome) ──")
    print(
        f"  workspace accepted   : selvra={selvra_summary['workspace']['accepted']} "
        f"adam={adam_summary['workspace']['accepted']}"
    )
    print(
        f"  reliability         : selvra={selvra_summary['reliability']:.3f} "
        f"adam={adam_summary['reliability']:.3f}"
    )
    print(
        f"  attention transitions: selvra={selvra_summary['transitions']} "
        f"adam={adam_summary['transitions']}"
    )
    print(
        f"  obs balance         : selvra={selvra_summary['obs_counts']} "
        f"adam={adam_summary['obs_counts']}"
    )


if __name__ == "__main__":
    main()
