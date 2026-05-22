"""Curious-world demo — AE-1 + AE-2 + AST-1 + RPT i drift.

Selvra lever i SymbolWorld med autonoma actions. Hennes look_at-rörelser
drivs INTE av workspace-broadcasts (som i alive_world) utan av hennes
egen CuriosityDriver. Action-effect-modellen predikterar effekt av
look_at innan handling, mäter felet efter.

Vad du ska SE i ansiktet:
  - Body-sway när drive är stark (hon "vill" titta någonstans)
  - Pupiller skiftar aktivt mellan objekt
  - Telemetry visar nuvarande goal + target + action-error
  - Goal-rad förklarar varför attention är där
  - Pupil-dilation vid surprise (PP-1)
  - Aura-stabilisering över tid (HOT-2)

Kör:
    python3 -m examples.curious_world --duration 120 --open

Loop per tick:
  1. world.tick()
  2. agent observe (med focus från senaste action)
  3. perception_module.process() → SceneRepresentation (RPT-2)
  4. per source: engine.observe → PP-1 + PE → workspace + monitor
  5. curiosity.decide() → ActionIntent
  6. action_model.predict_effect(intent.target) → predicted intensities
  7. agent.look_at(intent.target_angle)
  8. attention_schema.update_from_intent(intent)
  9. write_state
  (next tick:)
  10. action_model.observe_effect(actual_intensities)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selvra_brain.agency import (
    ActionEffectModel,
    ActionType,
    AttentionSchema,
    CuriosityDriver,
)
from selvra_brain.core.events import EventStore
from selvra_brain.metacognition import MetacognitiveMonitor
from selvra_brain.perception import PerceptionModule
from selvra_brain.prediction import HierarchicalPredictiveEngine
from selvra_brain.visualization.state import write_state_from_store
from selvra_brain.workspace import GlobalWorkspace, PredictionErrorProducer
from selvra_brain.world import Agent, make_default_world


REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = REPO_ROOT / "viz"
STATE_PATH = VIZ_DIR / "state.json"
INDEX_HTML = VIZ_DIR / "index.html"


def run_demo(duration_seconds: int = 120, tick_interval: float = 0.4) -> None:
    # ─── Setup ─────────────────────────────────────────────────────
    store = EventStore()
    perception = PerceptionModule(store=store)
    engine = HierarchicalPredictiveEngine(store, levels=2)
    workspace = GlobalWorkspace(store, capacity=4, acceptance_threshold=0.25)
    producer = PredictionErrorProducer(workspace, surprise_threshold=0.25)
    monitor = MetacognitiveMonitor(engine, store, window=15)
    curiosity = CuriosityDriver(store=store)
    action_model = ActionEffectModel(store=store)
    attention_schema = AttentionSchema(store=store)

    world = make_default_world(seed=42)
    agent = Agent(name="selvra", world=world)
    source_positions = {obj.object_id: obj.position_angle for obj in world.objects}

    write_state_from_store(store, STATE_PATH)

    print("Selvra wakes — autonomous curiosity-driven action…")
    print(f"  State file:  {STATE_PATH}")
    print(f"  Viz file:    file://{INDEX_HTML.resolve()}")
    print(f"  Duration:    {duration_seconds}s")
    print()
    print("Indicators active in this run:")
    print("  PP-1   prediction-error → pupiller")
    print("  GW-1..4 workspace bottleneck + broadcast")
    print("  HOT-2  metacognitive reliability → aura")
    print("  RPT-1  recurrent perception (scene smoothing)")
    print("  RPT-2  scene-integration (multi-object binding)")
    print("  AE-1   curiosity-driven action-selection")
    print("  AE-2   action-effect prediction + error")
    print("  AST-1  attention-schema self-report")
    print()
    print("Objects in world:")
    for obj in world.objects:
        print(
            f"  - {obj.object_id:8s}  angle={obj.position_angle:+5.2f}  dyn={obj.dynamic.value}"
        )
    print()

    start = time.time()
    tick = 0

    while True:
        t = time.time() - start
        if t > duration_seconds:
            break

        # 1. World tick
        world.tick()
        tick += 1

        # 2. Agent observe (med current focus)
        observations = agent.observe()
        agent.tick_idle_drift()

        # 3. RPT: integrera observations till scene
        perception.process(observations, tick=tick)

        # 4. Per source: PP + workspace + monitor
        for obs in observations:
            curiosity.record_observation(obs.object_id)
            effective_value = obs.signal_value * obs.intensity
            result = engine.observe(source=obs.object_id, value=effective_value)
            if result.level_0_error is not None:
                producer.handle_error(result.level_0_error)
            monitor.update_from_result(result)

        # 4b. AE-2: observe-effect (faktiska intensities mot förra
        # predikterade) — körs FÖRE nästa action så att model uppdateras
        actual_intensities = {obs.object_id: obs.intensity for obs in observations}
        action_model.observe_effect(actual_intensities=actual_intensities)

        # 5. AE-1: curiosity decides next intent
        reliability_per_source = {
            src: info["reliability"]
            for src, info in monitor.stats()["per_source"].items()
            if info["reliability"] is not None
        }
        intent = curiosity.decide(
            source_positions=source_positions,
            reliability_per_source=reliability_per_source,
        )

        # 6. AE-2: predict effect av denna action
        if intent.action == ActionType.LOOK_AT:
            action_model.predict_effect(
                target_angle=intent.target_angle,
                source_positions=source_positions,
            )
            # 7. Execute action
            agent.look_at(intent.target_angle)

        # 8. AST-1: schema noterar
        attention_schema.update_from_intent(intent)

        # 9. Write state
        write_state_from_store(store, STATE_PATH)

        # Periodisk inspektion
        if tick % 20 == 0:
            ws_stats = workspace.stats()
            curiosity_stats = curiosity.stats()
            action_stats = action_model.stats()
            schema_stats = attention_schema.stats()
            print(
                f"[t={t:5.1f}s tick={tick:3d}] "
                f"events={len(store):4d} "
                f"ws={ws_stats['accepted']}/{ws_stats['proposed']} "
                f"rel={monitor.global_reliability():.2f} "
                f"actions={curiosity_stats['action_count']} "
                f"goal={schema_stats['current_goal']:25s} "
                f"target={str(schema_stats['current_target']):8s} "
                f"act_err={action_stats['last_error_magnitude']:.2f}"
            )

        time.sleep(tick_interval)

    print()
    print("Demo done.")
    print(f"  Total events:   {len(store)}")
    print(f"  Workspace:      {workspace.stats()}")
    print(f"  Reliability:    {monitor.global_reliability():.3f}")
    print(f"  Action count:   {curiosity.stats()['action_count']}")
    print(f"  AE-2 estimates: focus_width={action_model.estimated_focus_width:.3f} "
          f"periphery={action_model._periphery_intensity_low:.3f}")
    print()
    print("Per-source observation counts:")
    for src, count in curiosity.stats()["observation_counts"].items():
        print(f"  {src:8s}: {count}")
    print()
    print("Final AST-1 self-report:")
    schema_stats = attention_schema.stats()
    print(f"  current_target={schema_stats['current_target']}")
    print(f"  current_goal={schema_stats['current_goal']}")
    print(f"  duration_on_target={schema_stats['duration_on_target']} ticks")
    print(f"  total transitions={schema_stats['transitions_total']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AE-1 + AE-2 + AST-1 + RPT demo")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--tick", type=float, default=0.4)
    parser.add_argument("--open", action="store_true", help="Öppna viz i browser")
    parser.add_argument("--port", type=int, default=8765, help="HTTP-server port")
    args = parser.parse_args()

    if args.open:
        from examples._browser import open_viz_in_browser

        open_viz_in_browser(VIZ_DIR, port=args.port)
        time.sleep(1.5)

    run_demo(args.duration, tick_interval=args.tick)


if __name__ == "__main__":
    main()
