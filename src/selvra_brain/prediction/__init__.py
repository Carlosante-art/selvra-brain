"""Predictive Processing modules.

Implementerar:
- PP-1: Predictive coding på multipla nivåer (Butlin et al)

Multi-level är inte feature — det är fundamental skillnad mellan
trivial prediction och PP-1. En medveten arkitektur har modeller av
sina egna modeller.

Engine genererar PREDICTION/PERCEPTION/PREDICTION_ERROR-events automatiskt.
Visualization läser PREDICTION_ERROR.payload.normalized_magnitude för
pupill-dilation.
"""

from selvra_brain.prediction.engine import (
    HierarchicalPredictiveEngine,
    ObservationResult,
)
from selvra_brain.prediction.predictor import (
    ConstantPredictor,
    LinearTrendPredictor,
    MovingAveragePredictor,
    Prediction,
    PredictionError,
    Predictor,
    default_predictor,
)

__all__ = [
    "ConstantPredictor",
    "HierarchicalPredictiveEngine",
    "LinearTrendPredictor",
    "MovingAveragePredictor",
    "ObservationResult",
    "Prediction",
    "PredictionError",
    "Predictor",
    "default_predictor",
]
