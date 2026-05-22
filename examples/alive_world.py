"""Integrated demo: Selvra lever i SymbolWorld.

Detta är Fas 1-integration. Alla 4 implementerade moduler i samma loop:

  PP-1 (HierarchicalPredictiveEngine)  predikterar varje objekts signal
  GW (GlobalWorkspace)                 surprise > threshold → workspace
  HOT-2 (MetacognitiveMonitor)         L1-error → reliability
  World (SymbolWorld + Agent)          minimal embodiment

Loop per tick:
  1. world.tick()
  2. agent observerar (med focus-modulering från senaste broadcast)
  3. engine.observe() per objekt → events
  4. producer skickar prediction-errors > threshold till workspace
  5. workspace accepterar/avvisar, broadcastar accepterade
  6. monitor uppdaterar reliability
  7. state.json skrivs → ansiktet uppdateras

Vad du ska SE i ansiktet:
  - Pupiller dilateras vid surprise (bird's bursts, world-stochasticity)
  - Blick skiftar mot källan av senaste surprise
  - Aura stabiliseras när engine konvergerar (HOT-2 reliability stiger)
  - Andning fortsätter alltid

Kör:
    python -m examples.alive_world --duration 90 --open
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selvra_brain.core.events import EventStore
from selvra_brain.metacognition import MetacognitiveMonitor
from selvra_brain.prediction import HierarchicalPredictiveEngine
from selvra_brain.visualization.state import write_state_from_store
from selvra_brain.workspace import (
    GlobalWorkspace,
    PredictionErrorProducer,
)
from selvra_brain.world import Agent, make_default_world


REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = REPO_ROOT / "viz"
STATE_PATH = VIZ_DIR / "state.json"
INDEX_HTML = VIZ_DIR / "index.html"


def run_demo(duration_seconds: int = 90, tick_interval: float = 0.4) -> None:
    # ─── Setup ───────────────────────────────────────────────────
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=2)
    workspace = GlobalWorkspace(store, capacity=4, acceptance_threshold=0.25)
    producer = PredictionErrorProducer(workspace, surprise_threshold=0.25)
    monitor = MetacognitiveMonitor(engine, store, window=15)

    world = make_default_world(seed=42)
    agent = Agent(name="selvra", world=world)

    # Broadcast subscriber: när workspace tar in ett item drar agent
    # blicken mot källan om den känner till objektets position
    object_id_to_angle = {obj.object_id: obj.position_angle for obj in world.objects}

    def on_broadcast(signal) -> None:
        # Workspace-broadcast → om payload har predictor_source som matchar
        # ett world-object, dra blicken dit
        src = signal.item.payload.get("predictor_source")
        if src in object_id_to_angle:
            agent.look_at(object_id_to_angle[src])

    workspace.subscribe(on_broadcast)

    write_state_from_store(store, STATE_PATH)

    print("Selvra wakes in her world…")
    print(f"  State file:  {STATE_PATH}")
    print(f"  Viz file:    file://{INDEX_HTML.resolve()}")
    print(f"  Duration:    {duration_seconds}s")
    print()
    print("Objects in world:")
    for obj in world.objects:
        print(f"  - {obj.object_id:8s}  angle={obj.position_angle:5.2f}  dyn={obj.dynamic.value}")
    print()

    start = time.time()
    tick = 0

    while True:
        t = time.time() - start
        if t > duration_seconds:
            break

        # ─── World tick ──────────────────────────────────────
        world.tick()
        tick += 1

        # ─── Agent observerar (med focus) ────────────────────
        observations = agent.observe()
        agent.tick_idle_drift()

        # ─── Per observation: engine.observe ─────────────────
        for obs in observations:
            # Dämpa signal-värdet med intensity (periphery = vagare)
            effective_value = obs.signal_value * obs.intensity
            result = engine.observe(source=obs.object_id, value=effective_value)

            # Skicka prediction-error till workspace om över threshold
            if result.level_0_error is not None:
                producer.handle_error(result.level_0_error)

            # Uppdatera metacognitive monitor
            monitor.update_from_result(result)

        # ─── Skriv state för visualization ───────────────────
        write_state_from_store(store, STATE_PATH)

        # ─── Periodisk inspektion ────────────────────────────
        if tick % 15 == 0:
            ws_stats = workspace.stats()
            print(
                f"[t={t:5.1f}s tick={tick:3d}] "
                f"events={len(store):4d}  "
                f"ws_accepted={ws_stats['accepted']:3d}/{ws_stats['proposed']:3d}  "
                f"reliability={monitor.global_reliability():.2f}  "
                f"focus_angle={agent.focus_angle:5.2f}"
            )

        time.sleep(tick_interval)

    print()
    print(f"Demo done. Final stats:")
    print(f"  Total events: {len(store)}")
    print(f"  Workspace: {workspace.stats()}")
    print(f"  Reliability: global={monitor.global_reliability():.3f}")
    print(f"  Per-source reliability:")
    for src, info in monitor.stats()["per_source"].items():
        if info["reliability"] is not None:
            print(f"    {src:8s}: {info['reliability']:.3f}  (history {info['history_size']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated demo: Selvra i SymbolWorld")
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--tick", type=float, default=0.4)
    parser.add_argument("--open", action="store_true", help="Öppna viz i browser")
    args = parser.parse_args()

    if args.open:
        url = f"file://{INDEX_HTML.resolve()}"
        print(f"Opening {url}")
        try:
            webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001
            print(f"  could not open: {exc}")
        time.sleep(1.5)

    run_demo(args.duration, tick_interval=args.tick)


if __name__ == "__main__":
    main()
