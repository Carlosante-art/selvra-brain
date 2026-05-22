"""Visualization — Selvras ansikte.

För varje implementerad arkitektur-modul får ansiktet en ny
visuell manifestation:

| Modul                  | Visuell signal                          |
|------------------------|-----------------------------------------|
| Foundation (nu)        | Andning, blink, baseline-närvaro        |
| Affective valens       | Mun + ögonbryn-rörelse                  |
| Predictive coding      | Pupill-storlek = prediction-error       |
| Global Workspace       | Blickriktning = attention-target        |
| Higher-Order (HOT-2)   | Aura/glow = metacognitiv reliability    |
| Embodiment + boundary  | Body-sway under huvud                   |
| Active inference (AE)  | Aktiv blick-skift mellan items          |

Python skriver `state.json` som HTML/SVG pollar. Filosofiskt val:
inte människolik. Abstrakt-humanoid. Inte påstå att hon ÄR.
"""
