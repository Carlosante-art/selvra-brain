"""Agent — minimal entity som lever i SymbolWorld.

Per ROADMAP Fas 1e: agent observerar världen, ev. styr sitt fokus.
Senare faser (Fas 2): actions med environment-effects (AE-2).

Detta är minimal. Agent håller:
- focus_angle (vart hon "tittar")
- attention_decay (drift mot inåtblick när inget aktivt händer)
- En knytning till SymbolWorld

Hon kallar engine.observe() för varje observation. Engine och workspace
och monitor körs separat (orchestreras av demo-loop).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from selvra_brain.world.symbol_world import Observation, SymbolWorld


@dataclass
class Agent:
    """Minimal agent i en SymbolWorld.

    Agentens primära handling i Fas 1: look_at(angle). Hon väljer
    vad hon vill se tydligast. World ger henne periphery-observationer
    från övriga objekt med lägre intensity.

    Senare faser:
    - move_to(position) — riktig motorisk action
    - utter(content) — kommunikation med andra agenter
    - rest() — modulera arousal
    """

    name: str
    world: SymbolWorld
    focus_angle: float = 0.0  # vart blicken är riktad nu
    focus_drift_rate: float = 0.02  # rad/tick mot 0 (inåtblick) när idle
    last_action_tick: int = 0

    def look_at(self, angle: float) -> None:
        """Riktigt fokus. Action — kommer påverka nästa observation."""
        self.focus_angle = angle
        self.last_action_tick = self.world.current_tick

    def look_at_object(self, object_id: str) -> bool:
        """Convenience: rikta blicken mot ett namngivet objekt.

        Returns True om objektet finns, False annars.
        """
        for obj in self.world.objects:
            if obj.object_id == object_id:
                self.look_at(obj.position_angle)
                return True
        return False

    def observe(self) -> list[Observation]:
        """Hämta observationer från världen via aktuellt fokus.

        Detta är agentens "perception"-action. Hon ser inte allt — hon
        ser tydligast det hon tittar på (intensity=1.0), och mindre
        tydligt det i periferin.
        """
        return self.world.observe_with_focus(focus_angle=self.focus_angle)

    def attention_vector(self) -> tuple[float, float]:
        """Returnera (x, y) för aktuellt fokus.

        Detta är vad visualization kan använda för pupil-gaze. Världen
        är 2D polar — vi konverterar till cartesiska för screen-space.
        """
        x = math.cos(self.focus_angle)
        y = -math.sin(self.focus_angle)  # SVG y-axel pekar ner
        return x, y

    def tick_idle_drift(self) -> None:
        """När agenten är "idle" driftar blicken tillbaka mot 0 (inåt).

        Detta är subtle behavior — utan aktivt look_at-anrop kommer
        attention att normaliseras över tid. Inte centralt i Fas 1,
        men det gör visualization mer levande.
        """
        ticks_since_action = self.world.current_tick - self.last_action_tick
        if ticks_since_action > 5:
            # Drift mot 0
            if abs(self.focus_angle) > 0.01:
                step = self.focus_drift_rate * (-1 if self.focus_angle > 0 else 1)
                self.focus_angle = max(min(self.focus_angle + step, math.pi), -math.pi)
