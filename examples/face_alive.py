"""Demo: visa Selvra-ansiktet "leva" genom syntetiska events.

Kör:
    python3 -m examples.face_alive [--duration 60]

Vad det gör:
1. Skapar en EventStore
2. Loopar och skapar diverse events över tid (perception, valence-shifts,
   workspace-entries, prediction-errors)
3. För varje event: uppdaterar viz/state.json
4. Öppnar viz/index.html i browsern (eller printar URL)

Vad du ser:
- Andning som alltid pågår
- Slumpmässig blink
- Munnen rör sig vid valence-shifts
- Pupiller dilateras vid prediction-errors
- Aura växer med event_count

Detta är Fas 0-demo. Bara core (events + valens) är implementerat.
Senare faser (PP-1, GW, HOT-2, etc) kommer adda flera visuella signaler.
"""

from __future__ import annotations

import argparse
import math
import random
import time
import webbrowser
from pathlib import Path

# Add src to path för att köra utan pip install
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
from selvra_brain.visualization.state import write_state_from_store


REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = REPO_ROOT / "viz"
STATE_PATH = VIZ_DIR / "state.json"
INDEX_HTML = VIZ_DIR / "index.html"


def make_tag(valence: Valence, data_type: DataType = DataType.OBSERVED) -> EpistemicTag:
    return EpistemicTag(
        data_type=data_type,
        confidence=Confidence.MEDIUM,
        mutability=Mutability.SYSTEM_MUTABLE,
        persistence=Persistence.SHORT_TERM,
        memory_type=MemoryType.EPISODIC,
        valence=valence,
    )


# ─── Syntetiska event-mönster ────────────────────────────────────


def perception_event(intensity: float) -> BrainEvent:
    """Sensorisk input. Valens baserad på intensity (hög = activated)."""
    valence = Valence.from_numeric(intensity * 0.4)  # mild activation
    return BrainEvent(
        category=EventCategory.PERCEPTION,
        event_type=f"sensory_input_{int(intensity * 10)}",
        payload={"intensity": intensity},
        tag=make_tag(valence),
    )


def workspace_focus(focus_name: str, valence: Valence) -> BrainEvent:
    return BrainEvent(
        category=EventCategory.WORKSPACE_ENTRY,
        event_type=focus_name,
        payload={"item": focus_name},
        tag=make_tag(valence),
    )


def prediction_error(magnitude: float) -> BrainEvent:
    """Surprise — verklighet matchade inte prediction."""
    valence = Valence.from_numeric(-magnitude * 0.5)  # negative bias
    return BrainEvent(
        category=EventCategory.PREDICTION_ERROR,
        event_type="surprise",
        payload={"magnitude": magnitude},
        tag=make_tag(valence, data_type=DataType.DERIVED),
    )


def valence_shift(new_valence: Valence) -> BrainEvent:
    return BrainEvent(
        category=EventCategory.VALENCE_SHIFT,
        event_type=f"shift_to_{new_valence.value}",
        payload={},
        tag=make_tag(new_valence),
    )


def body_signal(arousal: float) -> BrainEvent:
    valence = Valence.NEUTRAL
    if arousal > 0.7:
        valence = Valence.NEGATIVE  # hög arousal är ofta stress
    return BrainEvent(
        category=EventCategory.BODY_STATE,
        event_type="interoception",
        payload={"arousal": arousal},
        tag=make_tag(valence),
    )


# ─── Demo-loop ────────────────────────────────────────────────────


def run_demo(duration_seconds: int = 60) -> None:
    store = EventStore()

    # Skapa state-fil direkt så HTML har något att läsa även innan
    # första event
    write_state_from_store(store, STATE_PATH)

    print(f"Selvra is waking up... (duration {duration_seconds}s)")
    print(f"  State file:  {STATE_PATH}")
    print(f"  Viz file:    file://{INDEX_HTML.resolve()}")
    print(f"  Open in browser to see her face.")
    print()

    start = time.time()
    tick = 0
    last_phase_change = start
    phase = "perception"  # "perception" | "workspace" | "surprise" | "rest"

    while True:
        elapsed = time.time() - start
        if elapsed > duration_seconds:
            break

        tick += 1

        # Phasen byts var ~8 sekunder för en levande dynamik
        if time.time() - last_phase_change > 8:
            phases = ["perception", "workspace", "surprise", "rest"]
            phase = random.choice(phases)
            last_phase_change = time.time()
            print(f"[t={elapsed:5.1f}s tick={tick:3d}] phase → {phase}")

        # Generera events baserat på fas
        if phase == "perception":
            intensity = 0.3 + 0.5 * math.sin(tick * 0.3)
            store.append(perception_event(abs(intensity)))
        elif phase == "workspace":
            focus = random.choice(["light_source", "movement", "memory_trace", "self_signal"])
            v = random.choice([Valence.POSITIVE, Valence.NEUTRAL, Valence.POSITIVE])
            store.append(workspace_focus(focus, v))
        elif phase == "surprise":
            mag = 0.3 + random.random() * 0.7
            store.append(prediction_error(mag))
            # ibland body-reaktion på surprise
            if random.random() < 0.5:
                store.append(body_signal(arousal=0.6 + random.random() * 0.3))
        elif phase == "rest":
            # Lugnt — bara enstaka body-events
            if random.random() < 0.3:
                store.append(body_signal(arousal=0.2 + random.random() * 0.2))

        # Lägg in valence-shift ibland för dramatik
        if random.random() < 0.08:
            new_v = random.choice(list(Valence))
            store.append(valence_shift(new_v))

        # Skriv state
        write_state_from_store(store, STATE_PATH)

        # Tempo
        time.sleep(0.6)

    print()
    print(f"Demo complete. Total events: {len(store)}.")
    print(f"State preserved in {STATE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo: Selvra-ansiktet med syntetiska events."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Demo-tid i sekunder (default 60)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Försök öppna viz/index.html i browsern automatiskt",
    )
    args = parser.parse_args()

    if args.open:
        url = f"file://{INDEX_HTML.resolve()}"
        print(f"Opening {url}...")
        try:
            webbrowser.open(url)
        except Exception as exc:
            print(f"  (could not open: {exc})")
        time.sleep(1.5)  # ge browsern tid att öppna

    run_demo(args.duration)


if __name__ == "__main__":
    main()
