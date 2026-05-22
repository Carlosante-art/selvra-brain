"""Perception modules — RPT-1 + RPT-2.

RPT-1: Input-moduler med algoritmisk återkoppling (recurrent processing).
RPT-2: Integrerade perceptuella representationer (scenes över flera samtidiga
       inputs).
"""

from selvra_brain.perception.scene import (
    ObjectFeatures,
    PerceptionModule,
    SceneRepresentation,
)

__all__ = ["ObjectFeatures", "PerceptionModule", "SceneRepresentation"]
