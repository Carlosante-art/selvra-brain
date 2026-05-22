# Architecture — Butlin et al. 14 indikatorer + utvidgningar

**Datum:** 2026-05-22 (initial)
**Primary source:** [Butlin, Long, Bengio, Chalmers et al. 2023, uppdaterad 2025](https://arxiv.org/abs/2308.08708)
**Status:** Initial scaffold. Inga indikatorer fullt implementerade än.

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
| RPT-1 | Input-moduler med algoritmisk återkoppling | 🔴 | `perception/` modul. Per-input recurrent connections, inte feedforward. |
| RPT-2 | Input-moduler genererar organiserade, integrerade perceptuella representationer | 🔴 | Output från RPT-1 är inte rådata utan strukturerad represenation som binder samtidig multi-modal input. |

### Global Workspace Theory (GWT)

| ID | Indikator | Status | Plan |
|---|---|---|---|
| GW-1 | Multipla specialiserade system kapabla att operera parallellt (moduler) | 🔴 | `perception/` + `metacognition/` + `prediction/` + `affect/` ska köra parallellt, inte sekventiellt. asyncio + gather, eller multi-process. |
| GW-2 | Begränsad kapacitet workspace, attention-flaskhals | 🔴 | `workspace/global_workspace.py` med fix bounded buffer. Endast N items kan vara i workspace samtidigt. |
| GW-3 | Global broadcast — workspace gör information tillgänglig för alla moduler | 🔴 | Pub/sub-mekanism. När item kommer in i workspace, alla moduler får notifikation. |
| GW-4 | Tillstånds-beroende attention för att styra workspace | 🔴 | `workspace/attention.py` med affektiv valens-modulering. Vad som hamnar i workspace beror på system-state, inte fix rule. |

### Higher-Order Theories (HOT)

| ID | Indikator | Status | Plan |
|---|---|---|---|
| HOT-1 | Generativ, top-down eller noisy perception-modul | 🔴 | LLM-call (Anthropic API) som verktyg för generativ representation. Inte själva arkitekturen — verktyg inom den. |
| HOT-2 | Metakognitiv övervakning som skiljer pålitliga från opålitliga representationer | 🟡 | Selvras provenance + confidence-system är embryonalt detta. `metacognition/higher_order.py` ska utvidga. |
| HOT-3 | Agens styrd av belief-formation och action-selection-system med metakognitiv uppdatering | 🟡 | Selvras hypothesis-engine ligger nära. `agency/active_inference.py` integrerar. |
| HOT-4 | Sparsam och smidig kodning som genererar "quality space" | 🔴 | Embedding-baserad representation där liknande tillstånd ligger nära varandra. Selvras event-store är diskret — vi behöver continuous parallellt. |

### Predictive Processing + Agency + Attention Schema

| ID | Indikator | Status | Plan |
|---|---|---|---|
| PP-1 | Predictive coding på multipla nivåer | 🔴 | `prediction/predictive.py`. Per-källa prediktion + prediction-error som uppdaterar både input-modell och själva prediktionsmodellen. |
| AE-1 | Agens med learning och flexibel målorientering | 🔴 | `agency/` modul. Inte hårdkodade mål. |
| AE-2 | Embodiment — modell av output-effekter på environmental input | 🔴 | `body/` interocepting + propriocepting. Agentens egna handlingar ska påverka dess egen input på ett spårbart sätt. |
| AST-1 | Attention schema — modell av processer som styr attention | 🔴 | `agency/attention_schema.py`. Inte bara prioritera data — representera HUR prioritering fungerar. |

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
