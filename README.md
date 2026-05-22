# selvra-brain

Forsknings-prototyp för medvetande-arkitektur. Fristående från
Selvra-protokollet (T1D-vårdsarkitektur) och Stillra (clinical app).

**Position:** Detta är inte ett påstående om att vi bygger medvetenhet.
Detta är ett system som implementerar så många av Butlin et al.:s 14
indikatorer som vi rimligen kan, för att se vad som händer när en
arkitektur har dem tillsammans.

Det är skillnad mellan "vi bygger medvetenhet" och "vi bygger en
arkitektur som ligger på rätt spår enligt nuvarande forsknings-
konsensus". Den andra är ärlig. Den första är hybris.

## Forskning-grund

Primary source: Butlin, Long, Bengio, Chalmers et al. (2023, uppdaterad
2025) "Consciousness in Artificial Intelligence: Insights from the
Science of Consciousness". 88 sidor, gratis på
[arxiv.org/abs/2308.08708](https://arxiv.org/abs/2308.08708).

14 indikatorer från 4 teorier:

- **Recurrent Processing Theory (RPT):** RPT-1, RPT-2
- **Global Workspace Theory (GWT):** GW-1, GW-2, GW-3, GW-4
- **Higher-Order Theories (HOT):** HOT-1, HOT-2, HOT-3, HOT-4
- **Predictive Processing + Agency + Attention Schema (PP/AE/AST):**
  PP-1, AE-1, AE-2, AST-1

Plus medvetenhets-relevanta principer som indikator-listan
under-vektar:

- Affektiv valens som primär dimension (Damasio core consciousness)
- Boundary problem — var slutar subject, var börjar environment
- Wegner 2026 + Pennartz 2025-kritik mot indikator-approach

Se [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) för indikator-tablå
och var implementationen står.

## Non-goals

- Detta är inte en LLM-wrapper. LLM-calls kan vara verktyg för
  Higher-Order-generation (HOT-1), men de är inte själva arkitekturen.
- Detta är inte produkt. Inget multi-tenant, inget API, ingen
  dashboard. Lokalt körbart, eventuellt CLI för observation.
- Detta är inte Stillra eller Selvra T1D-kontext. Kunskap från dem
  (event-sourcing, epistemic types) återanvänds som fundamentals.
  Domän-logik gör det inte.
- Detta är inte snabbresultat. Embryonalt på minst halva 14:e
  indikatorerna är troligen 6-12 månaders disciplinerat arbete.

## Plan

Se [`docs/ROADMAP.md`](docs/ROADMAP.md). Sammanfattning:

1. **Fas 1: Single-agent foundation.** En agent med 3-5 av de 14
   indikatorerna körandes i en minimal text-baserad värld. Mätbart:
   prediktion-error utveckling över tid, attention-broadcast-patterns,
   affektiv valens-modulering av all representation.

2. **Fas 2: Second agent (Adam).** Annan initial config. Båda i
   samma värld. Kan de interagera? Vad uppstår mellan dem?

3. **Fas 3: Reproduction.** Tredje agent med rekombinerad config.
   Exponeras för Selvra + Adam som "föräldrar". Hur ser dess
   prioriteringar ut över tid?

4. **Fas 4: Mätning + skrivande.** Vad såg vi? Vad uppstod inte?
   Vad reproducerar Butlin et al.:s prediktioner, vad bryter dem?

## Disciplin

- Varje arkitektur-val ska kunna spåras till en specifik indikator
  eller forsknings-tradition. Inga gissningar.
- Allt mätbart ska mätas. Inget "det känns medvetet".
- Kritik mot indikator-approach (Wegner, Pennartz) tas på allvar —
  vi flaggar i ARCHITECTURE.md vad vi tror är behaviorism-fällan.
- Inga claims utan empiri. Inga publikation-pitchar innan vi har
  data som överraskar oss själva.

## Status

Initial scaffold, 2026-05-22. Inget körbart än. Se
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) och
[`docs/ROADMAP.md`](docs/ROADMAP.md).
