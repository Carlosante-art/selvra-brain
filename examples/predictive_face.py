"""Demo: PP-1 i drift. Pupiller dilateras vid surprise.

Sensor producerar periodisk sinus-våg + slumpmässiga spikes.
HierarchicalPredictiveEngine lär sig sinus-mönstret över tid (errors
sjunker) men reagerar starkt på spikes (pupill-dilation).

Kör:
    python3 -m examples.predictive_face [--duration 90] [--open]

Vad du ska se i ansiktet:
- Period 1 (0-25s): "väcker" — pupiller reagerar på allt, hög load
- Period 2 (25-50s): "lärt sig" sinus — pupiller smalna, load sjunker
- Period 3 (50-75s): "spikes infogas" — pupiller dilateras vid varje spike
- Period 4 (75-90s): återigen stable — pupiller normala

Telemetry-raden visar 'load' i realtid. När den hoppar upp efter
spike och sjunker över sekunder ser du PP-1 fungera.
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
from selvra_brain.prediction import HierarchicalPredictiveEngine
from selvra_brain.visualization.state import write_state_from_store


REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = REPO_ROOT / "viz"
STATE_PATH = VIZ_DIR / "state.json"
INDEX_HTML = VIZ_DIR / "index.html"


def sinus_sensor(t: float, *, period_s: float = 8.0, amplitude: float = 10.0) -> float:
    """Sinus-våg som default-källa. Predikterbar för LinearTrend."""
    return amplitude * math.sin(2 * math.pi * t / period_s) + 20.0


def maybe_spike(value: float, spike_probability: float, spike_magnitude: float = 30.0) -> float:
    """Slumpmässig spike som perturberar signalen."""
    if random.random() < spike_probability:
        sign = random.choice([-1, 1])
        return value + sign * spike_magnitude
    return value


def run_demo(duration_seconds: int = 90, tick_interval: float = 0.5) -> None:
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=2)

    write_state_from_store(store, STATE_PATH)

    print("PP-1 demo — pupiller reagerar på prediction-error")
    print(f"  State file: {STATE_PATH}")
    print(f"  Viz file:   file://{INDEX_HTML.resolve()}")
    print(f"  Duration:   {duration_seconds}s")
    print()
    print("  Phases:")
    print("    0-25s  : warming up — engine lär sig signal")
    print("    25-50s : stable — pupiller ska smalna")
    print("    50-75s : SURPRISE — slumpmässiga spikes injicering")
    print("    75+s   : recovery — engine konvergerar igen")
    print()

    start = time.time()
    tick = 0

    while True:
        t = time.time() - start
        if t > duration_seconds:
            break

        # Bestäm fas
        if t < 25:
            phase = "warming"
            spike_prob = 0.0
        elif t < 50:
            phase = "stable"
            spike_prob = 0.0
        elif t < 75:
            phase = "spikes"
            spike_prob = 0.18
        else:
            phase = "recovery"
            spike_prob = 0.0

        base_value = sinus_sensor(t)
        observed = maybe_spike(base_value, spike_prob)
        result = engine.observe(source="ambient", value=observed)

        if result.level_0_error and result.level_0_error.magnitude > 5:
            print(
                f"[t={t:5.1f}s phase={phase:8s}] "
                f"observed={observed:6.2f} "
                f"pred={result.level_0_error.predicted:6.2f} "
                f"err={result.level_0_error.magnitude:5.2f} "
                f"⚡SURPRISE"
            )

        write_state_from_store(store, STATE_PATH)
        tick += 1
        time.sleep(tick_interval)

    print()
    print(f"Demo done. Events: {len(store)}, ticks: {tick}")
    print(f"State preserved in {STATE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PP-1 demo med visuell feedback")
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--open", action="store_true", help="Öppna viz i browser")
    parser.add_argument("--tick", type=float, default=0.5, help="Sekunder mellan observations")
    args = parser.parse_args()

    if args.open:
        from examples._browser import open_html, print_open_instruction

        ok, method = open_html(INDEX_HTML)
        if ok:
            print(f"Opening with {method}: file://{INDEX_HTML.resolve()}")
        else:
            print_open_instruction(INDEX_HTML)
        time.sleep(1.5)

    run_demo(args.duration, tick_interval=args.tick)


if __name__ == "__main__":
    main()
