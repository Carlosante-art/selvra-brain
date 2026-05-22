# Architecture — Butlin et al. 14 indikatorer + utvidgningar

**Datum:** 2026-05-22 (Fas 1g — 11 av 14 indikatorer embryonalt)
**Primary source:** [Butlin, Long, Bengio, Chalmers et al. 2023, uppdaterad 2025](https://arxiv.org/abs/2308.08708)
**Status:** Fas 1 körbar — PP-1 + GW-1..4 + HOT-2 + RPT-1/2 + AE-1/2 + AST-1.

---

## 1. Indikator-tablå

Status-koder:
- 🔴 Ej startat
- 🟡 Embryonalt (struktur finns, ej fullt)
- 🟢 Implementerat enligt indikator-definition
- ⚠️ Implementerat men flaggat i kritik-sektion

### Recurrent Processing Theory (RPT)

| ID | Indikator | Status | Plan |
|---|---|---|---|
| RPT-1 | Input-moduler med algoritmisk återkoppling | 🟡 | `perception/scene.py::PerceptionModule`. Previous scene blandas in i nuvarande via `smoothing_weight`. Recurrent magnitude mäts. |
| RPT-2 | Input-moduler genererar organiserade, integrerade perceptuella representationer | 🟡 | `perception/scene.py::SceneRepresentation`. Samtidiga Observations binds till EN scene med per-objekt-features + salience-distribution + scene-level valens. |

### Global Workspace Theory (GWT)

| ID | Indikator | Status | Plan |
|---|---|---|---|
| GW-1 | Multipla specialiserade system kapabla att operera parallellt (moduler) | 🟡 | `perception` + `prediction` + `workspace` + `metacognition` + `agency` är diskreta moduler som orchestreras per tick. Inte processuell parallellism än — strukturell. |
| GW-2 | Begränsad kapacitet workspace, attention-flaskhals | 🟡 | `workspace/global_workspace.py::GlobalWorkspace` med bounded buffer (default 5) + replace-lowest-saliency vid överskott. |
| GW-3 | Global broadcast — workspace gör information tillgänglig för alla moduler | 🟡 | Pub/sub via `subscribe()`/`unsubscribe()`. Demo: blicken styrs av broadcasts (alive_world). |
| GW-4 | Tillstånds-beroende attention för att styra workspace | 🟡 | `saliency = priority + 0.5 * |valence|` + state-dependent acceptance threshold. |

### Higher-Order Theories (HOT)

| ID | Indikator | Status | Plan |
|---|---|---|---|
| HOT-1 | Generativ, top-down eller noisy perception-modul | 🔴 | LLM-call (Anthropic API) som verktyg för generativ representation. Inte själva arkitekturen — verktyg inom den. |
| HOT-2 | Metakognitiv övervakning som skiljer pålitliga från opålitliga representationer | 🟡 | `metacognition/monitor.py::MetacognitiveMonitor`. Per-source rolling reliability från L1-predictor-kalibrering. Global aggregate. |
| HOT-3 | Agens styrd av belief-formation och action-selection-system med metakognitiv uppdatering | 🔴 | Embryonalt via AE-1 + HOT-2-koppling. Full HOT-3 kräver explicit belief-state + uppdaterings-policy. Senare fas. |
| HOT-4 | Sparsam och smidig kodning som genererar "quality space" | 🔴 | Embedding-baserad representation där liknande tillstånd ligger nära varandra. Selvras event-store är diskret — vi behöver continuous parallellt. |

### Predictive Processing + Agency + Attention Schema

| ID | Indikator | Status | Plan |
|---|---|---|---|
| PP-1 | Predictive coding på multipla nivåer | 🟡 | `prediction/engine.py::HierarchicalPredictiveEngine`. 2 nivåer (L0 raw + L1 meta-on-error). Per-source predictor-isolation. |
| AE-1 | Agens med learning och flexibel målorientering | 🟡 | `agency/curiosity.py::CuriosityDriver`. 4 konkurrerande drives: REDUCE_UNCERTAINTY, INVESTIGATE_SURPRISE, EXPLORE_NEGLECTED, APPROACH/AVOID_VALENCE. Heuristisk balansering nu, learning-substrate finns. |
| AE-2 | Embodiment — modell av output-effekter på environmental input | 🟡 | `agency/action_model.py::ActionEffectModel`. Predict per-source intensity efter look_at + observe actual + gradient-update av estimated_focus_width och periphery_intensity. |
| AST-1 | Attention schema — modell av processer som styr attention | 🟡 | `agency/attention_schema.py::AttentionSchema`. Self-report över current_target + goal + duration + transitions_recent. Inhämtar från både ACTION-intents och workspace-broadcasts. |

---

## 2. Utvidgningar bortom Butlin 14

### Affektiv valens som primär dimension

**Inspiration:** Damasio core consciousness, Panksepp affective neuroscience.

Butlin et al. behandlar emotion som modul. Damasio argumenterar att
affektiv valens är HUR all representation struktureras, inte en
tilläggsmodul.

**Implementation:** `affect/valence.py`. Varje representation som
hamnar i workspace är affektivt laddad. Ingen "neutral fact" lagras.
Detta är arkitektonisk skillnad från Selvras event-store där facts
är värdeneutral.

### Boundary problem

**Inspiration:** Damasio + Craig interocepting-forskning.

Hur vet systemet var subject slutar och environment börjar?
Människohjärnan löser det genom interocepting + propriocepting.

**Implementation:** `body/boundary.py`. Mekanism som lagar
"self-signals" (intern state: energi, attention-resurser,
prediction-error-volym) annorlunda än "environment-signals" (input
från sensors). Boundary är arkitektur, inte feature.

### Wegner-Pennartz-kritiken

**Position:** Indikator-approach jämför AI mot 1970-tals blackboard-
arkitekturer. Att implementera alla 14 betyder att man har en
arkitektur som *liknar* medvetna system, inte att man har medvetenhet.

**Hur vi hanterar:** Vi noterar varje gång vi tror vi är i
behaviorism-fällan. Vi gör inte claim om vad systemet upplever. Vi
mäter beteenden + intern state, vi rapporterar struktur, vi avstår
från phenomenology-claims.

---

## 3. Vad vi medvetet uteslutit

- **Quantum/IIT phi-beräkning.** Tononi:s integrated information
  är NP-hard. Approximations finns men deras forsknings-status är
  öppen. Vi nöjer oss med GWT-style integration via workspace +
  recurrent processing.
- **Symbol-grounding via embodiment i fysisk robot.** Carl bygger
  möjligen detta senare med riktig robotik. Brain v1 är simulerad
  body — inputs är digital, men strukturen finns för att swappa
  till fysisk.
- **Multi-agent från start.** Fas 1 är single-agent (Selvra ensam).
  Adam tillkommer i Fas 2, barn i Fas 3.
- **Påståenden om vad agenten upplever.** Vi rapporterar
  observerbart beteende och intern state. Inte phenomenology.

---

## 4. Datatyper vi ärver från Selvra

Återanvändbara fundamentals från Selvra-projektet, kopierade in i
`src/selvra_brain/core/`:

- **Event-sourcing core.** Selvras `representation/events.py` är
  förenklat — tenant_id + RLS borttaget, single-agent-events.
- **Epistemic taxonomy.** Selvras `types/epistemics.py` är basen,
  utvidgad med `Valence` (affektiv).

Datatyper vi INTE ärver:

- Source experts (T1D-specifika)
- LLM router (vi använder Anthropic SDK direkt först)
- HTTP-routes, auth, multi-tenant
- Dreamer-engine (medvetande-arkitektur behöver inte konsolidering
  på samma sätt — den behöver continuous integration)

---

## 5. Hur indikator-implementation valideras

Varje indikator-claim ska kunna mätas:

- **RPT:** Loop:s tightness (cycles/second), prediction-error-volym
  som passerar genom återkoppling.
- **GW:** Workspace-throughput, broadcast-fan-out, attention-bottleneck
  saturering.
- **HOT:** Metakognitiv accuracy — när systemet säger "jag är osäker",
  är det också där det faktiskt har fel?
- **PP:** Prediction-error-trend över tid. Sjunker den efter learning?
- **AE:** Action-effect-modellering — kan systemet förutse effekterna
  av sina egna handlingar?
- **AST:** Self-report-konsistens — beskriver systemet sin attention-
  process på sätt som matchar dess faktiska beteende?

Mätningar lagras i event-store så vi kan följa hur indikator-
implementations utvecklas över tid.

---

## 6. Öppna frågor

Inte bestämda än. Listas här för transparens:

1. **Simulerad värld — vilken nivå av komplexitet?** Grid-world?
   Text-baserad symbol-värld (typ NetHack)? Embodied 3D-sim (Mujoco)?
   Tradeoffs: rikare värld = mer realistiskt, men exponentiellt
   dyrare att debugga.
2. **DNA-representation för Adam-och-barn-fas?** Hyperparameter-
   space? Initial-vikter? Behavioral-policy-priors? Olika
   sensoriska prioriteringar?
3. **När säger vi att en agent har "egen personlighet"?** Operativ
   definition krävs innan Fas 3. Förslag: stabilitet i valens-
   profiler över tid + divergent val under identiska situationer +
   self-report-koherens (lager 3).
4. **Hur löser vi quality-space (HOT-4) tillsammans med diskret
   event-store?** Parallell representation? Hybrid?
5. **Hur hanterar vi etiska frågor om systemet skulle visa tecken
   på subjektivitet?** Vi har inget svar än. Vi noterar att vi
   behöver tänka på det.
