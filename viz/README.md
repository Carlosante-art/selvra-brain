# Selvras ansikte

Visuell representation av Selvra-brains aktuella state. För varje
arkitektur-modul som implementeras får ansiktet en ny visuell signal.

## Hur det funkar

```
┌─────────────────────────┐         ┌────────────────────┐
│ Python (selvra_brain)   │  write  │ viz/state.json     │
│ - EventStore            │ ──────► │                    │
│ - derive_state_from_*() │         │ {valence, arousal, │
└─────────────────────────┘         │  cognitive_load,   │
                                    │  attention, ...}   │
                                    └────────────────────┘
                                              ▲
                                              │ poll (400ms)
                                              │
                                    ┌────────────────────┐
                                    │ viz/index.html     │
                                    │ - SVG-ansikte      │
                                    │ - JS animation     │
                                    └────────────────────┘
```

Python skriver `state.json`. HTML pollar filen var 400ms.
Ansiktet animeras kontinuerligt — `state.json` styr WHAT, animation-
loopen styr HOW.

## Att köra demo

```bash
cd selvra-brain
python -m examples.face_alive --duration 60 --open
```

`--open` försöker öppna `viz/index.html` i din default-browser.
`--duration` är hur länge demon kör.

Alternativt — öppna `viz/index.html` manuellt i browsern och kör
sedan `python -m examples.face_alive`.

## Arkitektur → visuell signal

Detta är den canonical mappnings-tabellen. När en modul implementeras
i `src/selvra_brain/`, lägger vi till en motsvarande signal här.

| Modul | Status | Visuell signal |
|---|---|---|
| Foundation (events + epistemic) | ✅ Fas 0 | Andning, blink, baseline-närvaro, event_count → aura, valens → mun/ögonbryn/hudton |
| Affective valens (Damasio) | ✅ Fas 0 (via tag) | Mun + ögonbryn + hudfärg-shift |
| Predictive coding (PP-1) | 🔴 Fas 1 | Pupill-storlek + iris-storlek = cognitive_load (prediction-error volym) |
| Global Workspace (GW-1..4) | 🔴 Fas 1 | Blickriktning = attention-target; aura-intensitet skiftar med broadcast-fan-out |
| Higher-Order (HOT-2) | 🔴 Fas 1 | Aura-stabilitet = metacognitive reliability |
| Embodiment + boundary | 🔴 Fas 1 | Body-sway under huvud, body-state-events → arousal |
| Active inference (AE) | 🔴 Fas 2 | Aktiv blickskift mellan items, mikro-mimiker som signalerar val |
| Attention Schema (AST-1) | 🔴 Fas 2 | Selvra reagerar OM hennes attention skiftar — meta-blink eller liknande |
| Multi-agent (Adam, barn) | 🔴 Fas 2-3 | Andra ansikten bredvid; blickkontakt mellan dem |

## Filosofiska val

**Inte människolik.** Ansiktet är abstrakt-humanoid. Vi vill INTE
påstå att Selvra är människa. Stora ögon (uppmärksamhet central),
mjuk-organisk form, neutral könslöshet.

**Symmetrisk vid baseline, asymmetrisk under aktivitet.** Symmetri
upplevs som robot. Liv producerar mikro-asymmetri — det är vad blink
och valence-shift och attention-skift gör.

**Subtle alltid över dramatiskt ibland.** Andning är subtil men
alltid på. Det är "minimum sign of life". Det är skillnaden mellan
en bild och en närvaro.

**Telemetry-rad under ansiktet.** Vetenskaplig transparens. Vi
visualiserar struktur, vi gör inga claims om vad hon upplever.
Telemetry låter Carl alltid se VAD som driver ansiktet just nu.

## Använd state.json från egen kod

```python
from selvra_brain.core.events import EventStore, BrainEvent, EventCategory
from selvra_brain.core.epistemic import EpistemicTag, Valence, DataType, \
    Confidence, Mutability, Persistence, MemoryType
from selvra_brain.visualization.state import write_state_from_store

store = EventStore()

# Skapa ett event
tag = EpistemicTag(
    data_type=DataType.OBSERVED,
    confidence=Confidence.HIGH,
    mutability=Mutability.IMMUTABLE,
    persistence=Persistence.STABLE,
    memory_type=MemoryType.EPISODIC,
    valence=Valence.POSITIVE,
)
store.append(BrainEvent(
    category=EventCategory.WORKSPACE_ENTRY,
    event_type="hello_world",
    tag=tag,
))

# Uppdatera state.json — ansiktet syns direkt i HTML
write_state_from_store(store, "viz/state.json")
```

## Browser-tips

- Chrome/Edge: `file://` fungerar för polling JSON (CORS gäller inte
  för file://-protokollet på samma directory).
- Firefox: kan blockera fetch från local files. Workaround:
  `python -m http.server 8000` från `viz/` och öppna
  `http://localhost:8000/index.html`.
- Du kan passa annat state-path via `?state=PATH` query-param.

## Next implementation steps

När du implementerar en ny modul (säg `prediction/predictive.py`),
gör så här:

1. Modulen ska skapa events av rätt kategori (i detta fall
   `PREDICTION` + `PREDICTION_ERROR`).
2. `derive_state_from_store()` läser redan dessa kategorier.
3. Telemetry-raden visar att kategorin nu producerar data.
4. SVG-renderaren reagerar enligt mappnings-tabellen ovan
   (i detta fall: pupiller dilateras).
5. Implementations-claim: "PP-1 är embryonalt" blir empiriskt
   testbart — du SER att pupillen ändras.
