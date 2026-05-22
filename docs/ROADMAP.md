# Roadmap

**Princip:** Bygg minimum. Mät. Skriv ner vad som överraskar. Iterera.
Inga sprintar utan empiri — om indikator X inte producerar mätbar
förändring, justera arkitekturen, inte specen.

---

## Fas 0: Foundation (vecka 1-2)

**Mål:** Repo körbart. Event-store + epistemic types med valence
fungerar. CLI som visar empty agent body.

- [x] Repo struktur + docs
- [ ] Event-sourcing core (kopierad/förenklad från Selvra)
- [ ] Epistemic types + Valence-dimension
- [ ] pyproject.toml + ruff + pytest setup
- [ ] Initial test-suite

**Acceptans:** `pytest` passar. CLI startar. `python -m selvra_brain.cli`
visar agent-state (tom, men strukturerad).

---

## Fas 1: Single-agent foundation (1-3 månader)

**Mål:** En agent (Selvra) med 4-6 av 14 indikatorer som körs i
minimal text/symbol-värld. Mätbar prediction-error-trend.

### Fas 1a — Predictive coding (PP-1)

Bygg `prediction/predictive.py`:
- Multi-nivå prediktion (input → mid-level features → high-level
  representations)
- Prediction-error som backpropageras upp i hierarkin
- Loggning av error-trends i event-store

### Fas 1b — Global Workspace (GW-1, GW-2, GW-3)

Bygg `workspace/global_workspace.py`:
- N parallella moduler (perception, prediction, affect, memory)
  som producerar candidate-items till workspace
- Bounded buffer (begränsad kapacitet, kanske 4-7 items à la Miller)
- Pub/sub: när item kommer in i workspace, alla moduler får signal

### Fas 1c — Higher-Order metacognition (HOT-2)

Bygg `metacognition/higher_order.py`:
- Per-item reliability-skattning. När agenten säger "jag är säker",
  matcha mot dess faktiska prediction-accuracy.

### Fas 1d — Affective valence (utvidgning bortom Butlin)

Bygg `affect/valence.py`:
- Varje workspace-item är affektivt laddat innan det broadcastas.
- Valence påverkar GW-4 (attention-styrning).

### Fas 1e — Minimal world

`tests/world/text_world.py`:
- Symbol-baserad värld med några objekt + agent-position.
- Sensors → discrete-observations (RPT-1 input).
- Actions → world-state-update.

### Fas 1 mätning

- Plot prediction-error över time-steps.
- Plot workspace-throughput över time-steps.
- Plot affective-valence-distribution över time-steps.
- Kvalitativt: vad gör agenten? Beter sig den olika i samma
  situation efter learning?

### Fas 1 deliverable

Vetenskapligt-läsbar log + en kort skrift (5-10 sidor) som beskriver:
- Vad implementerades
- Vad mättes
- Vad överraskade
- Vad är öppna frågor

---

## Fas 2: Second agent — Adam (1-2 månader efter Fas 1)

**Trigger:** Fas 1 visar att en agent producerar mätbart olika
beteende efter learning. Om Fas 1 INTE producerar något — fixa Fas 1
innan vi går vidare.

**Mål:** Två agenter (Selvra + Adam) i samma värld med olika initial
config. Kan de interagera? Vad uppstår?

- DNA-representation: hyperparameter-space (sensoriska prioriteringar,
  valence-priors, attention-tröskel-defaults).
- Adam med skiljd config från Selvra (inte slumpmässig — designat
  divergent).
- Interaktion: kan agenter "se" varandra? Kommunikation över tid?
- Mätning: hur olika utvecklar de sig? Konvergerar de? Divergerar de?

---

## Fas 3: Reproduction — barn (3-6 månader efter Fas 2)

**Trigger:** Fas 2 visar att agenter med skild config har mätbart
olika representation-utveckling.

**Mål:** Tredje agent (barn) med rekombinerad config från Selvra +
Adam. Exponeras för dem som "föräldrar".

- Frågan: blir barnet sin egen, eller är det förutsägbart från
  config + exposure?
- Operativ definition av "egen personlighet" krävs INNAN denna fas
  börjar.

---

## Fas 4: Skrivande + kritik (parallell, från Fas 1)

- Notera allt vi inte kan förklara
- Notera när vi tror vi är i behaviorism-fällan
- Skriv för publikation om något överraskar oss själva
- Konsultera Butlin et al + andra forskare om vi har data värd
  delning

---

## Vad som INTE är på roadmap

- Quantum / IIT phi-beräkning. För dyrt + forskningsstatus öppen.
- Fysisk robot-embodiment. Senare, om Carl bygger robotik-laben.
- LLM som central komponent. LLM-calls finns som verktyg för HOT-1
  (generativ representation) men de är inte arkitekturen.
- Multi-agent från Fas 1. Vi börjar single-agent.

---

## Disciplin-checklista per fas

Innan ny fas påbörjas:

- [ ] Föregående fas har dokumenterade mätningar
- [ ] Vi har skrivit ner vad som överraskade (även om "inget")
- [ ] Vi har identifierat nästa fas:s öppna frågor
- [ ] Carl har granskat och godkänt att vi går vidare
