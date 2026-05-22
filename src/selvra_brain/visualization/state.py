"""BrainVisualState — komprimerad agent-state för visualisering.

Konceptuellt:
- Selvra-brain har många interna events. Visualiseringen kan inte
  rendera alla. Vi destillerar till N synliga dimensioner som
  ansiktet kan uttrycka.
- Detta är en LÄSNING av representationen, inte representationen
  själv. Att ansiktet visar "valence=0.5" betyder inte att Selvra
  KÄNNER positivt — det betyder att hennes representations-statistik
  just nu väger åt det hållet.
- Per Wegner/Pennartz-kritiken: vi gör inga phenomenology-claims.
  Vi visualiserar struktur.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from selvra_brain.core.epistemic import Valence
from selvra_brain.core.events import BrainEvent, EventCategory, EventStore


@dataclass(frozen=True)
class AttentionVector:
    """Riktning attention är riktad mot. Skala (-1 till 1) per axel.

    (0, 0) = inåt-blick / fokus på intern state.
    (1, 0) = höger. (-1, 0) = vänster. (0, -1) = upp. (0, 1) = ned.
    Magnitude < 1 = mjukare gaze; magnitude > 1 = clamped i renderaren.
    """

    x: float = 0.0
    y: float = 0.0

    def magnitude(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5


@dataclass(frozen=True)
class BrainVisualState:
    """Komprimerad state för ansikts-rendering.

    Alla värden i [0, 1] eller [-1, 1] för enkel SVG-mappning.
    """

    # ─── Identitet + livstecken ──────────────────────────────────
    version: str = "0.0.1-pre-alpha"
    alive_since: str = ""  # ISO timestamp när första event skapades
    timestamp: str = ""  # När denna state-snapshot togs
    event_count: int = 0
    last_event_type: str | None = None

    # ─── Affektiv valens (Damasio core consciousness) ─────────────
    # -1 = negative_strong, 0 = neutral, 1 = positive_strong
    valence: float = 0.0
    # Senaste valence-shift, för animation
    valence_just_shifted: bool = False

    # ─── Arousal (vakenhet, energi-nivå) ──────────────────────────
    # 0 = sömn/tyst, 1 = maximalt vaken
    arousal: float = 0.5

    # ─── Cognitive load (för predictive coding / PP-1) ───────────
    # 0 = inga predictions, 1 = mycket prediction-error
    cognitive_load: float = 0.0

    # ─── Attention (för Global Workspace / GW-3) ─────────────────
    attention: AttentionVector = field(default_factory=AttentionVector)
    # Vad är i workspace just nu (för texthint i UI senare)
    workspace_items: int = 0
    workspace_focus: str | None = None

    # ─── Metacognitiv reliability (för HOT-2) ────────────────────
    # 0 = "mina representationer är osäkra", 1 = "mina representationer
    # är pålitliga". Kommer från self-reported confidence vs faktisk
    # prediction-accuracy. Default 0.5 tills HOT-2 implementeras.
    metacognitive_reliability: float = 0.5

    # ─── Body-state (för embodiment + boundary) ──────────────────
    # 0 = inert, 1 = aktiv kroppslig signalering
    body_activity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # attention som nested dict — asdict klarar dataclasses men
        # vi vill ha tydligare key-namn
        d["attention"] = {"x": self.attention.x, "y": self.attention.y}
        return d

    def to_json(self, *, pretty: bool = True) -> str:
        return json.dumps(
            self.to_dict(),
            indent=2 if pretty else None,
            ensure_ascii=False,
        )


# ─── Derivation: EventStore → BrainVisualState ────────────────────


def derive_state_from_store(
    store: EventStore,
    *,
    valence_window_events: int = 20,
) -> BrainVisualState:
    """Läs aktuell state från event-stream.

    valence_window_events: hur många senaste events vi tittar på för
    att räkna ut aktuell valens (rullande genomsnitt över taggade
    events).
    """
    all_events = store.all_events()
    if not all_events:
        return BrainVisualState(timestamp=datetime.now(tz=UTC).isoformat())

    alive_since = all_events[0].created_at.isoformat()
    latest = all_events[-1]

    # ─── Valens-aggregat ──────────────────────────────────────────
    # Rullande genomsnitt över senaste N events som har valence-taggar
    recent = all_events[-valence_window_events:]
    valences: list[float] = []
    for e in recent:
        if e.tag is not None:
            valences.append(e.tag.valence.to_numeric())
    avg_valence = sum(valences) / len(valences) if valences else 0.0

    # Senaste event en valence_shift?
    valence_just_shifted = latest.category == EventCategory.VALENCE_SHIFT

    # ─── Arousal: derivation från event-frekvens ─────────────────
    # Fler recent events = högre arousal. Crude heuristic men funkar
    # för fas 0. Skalar 0 → 1 över 1-30 events i senaste minuten.
    now = datetime.now(tz=UTC)
    recent_minute = sum(1 for e in all_events if (now - e.created_at).total_seconds() < 60)
    arousal = min(1.0, recent_minute / 30.0)

    # ─── Cognitive load: magnitude-aware ───────────────────────
    # När PP-1 är aktiv läser vi normalized_magnitude direkt från
    # event-payload (sätts av HierarchicalPredictiveEngine). Detta är
    # mer signal-troget än bara andelen events — magnitude SÄGER hur
    # överraskad systemet är.
    #
    # Vi använder MAX av senaste fönstret (inte snitt) för att fånga
    # sista surprise-spike — pupillen ska reagera DIREKT, inte
    # genomsnitta över historik.
    pe_events = [e for e in recent if e.category == EventCategory.PREDICTION_ERROR]
    if pe_events:
        # Föredra payload.normalized_magnitude (PP-1 producerar detta)
        # Fallback: andel om payload saknas (legacy events från Fas 0)
        magnitudes = []
        for e in pe_events:
            nm = e.payload.get("normalized_magnitude") if e.payload else None
            if isinstance(nm, (int, float)):
                magnitudes.append(float(nm))
        if magnitudes:
            cognitive_load = max(magnitudes)
        else:
            cognitive_load = len(pe_events) / max(1, len(recent))
    else:
        cognitive_load = 0.0

    # ─── Workspace: hur många WORKSPACE_ENTRY senaste 10 sec ─────
    ws_events = [
        e
        for e in all_events
        if e.category == EventCategory.WORKSPACE_ENTRY
        and (now - e.created_at).total_seconds() < 10
    ]
    workspace_items = len(ws_events)
    workspace_focus = ws_events[-1].event_type if ws_events else None

    # ─── Body activity: BODY_STATE-events andel av recent ────────
    body_events = [e for e in recent if e.category == EventCategory.BODY_STATE]
    body_activity = len(body_events) / max(1, len(recent))

    # ─── Attention vector från senaste WORKSPACE_ENTRY ────────
    # GW-3: senaste workspace-broadcast skickar attention_x/y i payload.
    # Vi läser senaste eventet och låter det driva blickriktningen.
    # Decay: om senaste workspace-event är gammalt → fade mot (0,0).
    attention = AttentionVector(x=0.0, y=0.0)
    ws_event_list = [e for e in all_events if e.category == EventCategory.WORKSPACE_ENTRY]
    if ws_event_list:
        latest_ws = ws_event_list[-1]
        ws_age_s = (now - latest_ws.created_at).total_seconds()
        # Decay-faktor: 0s → 1.0 (full intensitet), 5s+ → 0.0
        decay = max(0.0, 1.0 - ws_age_s / 5.0)
        ax = latest_ws.payload.get("attention_x", 0.0) if latest_ws.payload else 0.0
        ay = latest_ws.payload.get("attention_y", 0.0) if latest_ws.payload else 0.0
        if isinstance(ax, (int, float)) and isinstance(ay, (int, float)):
            attention = AttentionVector(x=ax * decay, y=ay * decay)

    return BrainVisualState(
        alive_since=alive_since,
        timestamp=now.isoformat(),
        event_count=len(all_events),
        last_event_type=latest.event_type or None,
        valence=avg_valence,
        valence_just_shifted=valence_just_shifted,
        arousal=arousal,
        cognitive_load=cognitive_load,
        attention=attention,
        workspace_items=workspace_items,
        workspace_focus=workspace_focus,
        body_activity=body_activity,
    )


def write_state(state: BrainVisualState, path: Path | str) -> None:
    """Atomisk skrivning av state till JSON-fil.

    HTML pollar denna fil — vi vill aldrig att den får läsa en
    halvskriven JSON. Därav: skriv till .tmp, sen rename.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(state.to_json(), encoding="utf-8")
    tmp.replace(p)


def write_state_from_store(store: EventStore, path: Path | str) -> BrainVisualState:
    """Convenience: derivera state + skriv. Returnerar state."""
    state = derive_state_from_store(store)
    write_state(state, path)
    return state
