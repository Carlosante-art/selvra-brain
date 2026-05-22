"""Higher-Order Theory modules.

Implementerar:
- HOT-2: Metakognitiv övervakning (reliability)

HOT-1, HOT-3, HOT-4 är ej implementerade i Fas 1.

HOT-2-implementationen läser PredictiveEngine.L1-predictor (som
predikterar L0-error magnitude) och jämför mot faktisk L0-error.
Bra L1-kalibrering = hög reliability = stark aura i visualization.
"""

from selvra_brain.metacognition.monitor import (
    MetacognitiveMonitor,
    ReliabilityAssessment,
)

__all__ = ["MetacognitiveMonitor", "ReliabilityAssessment"]
