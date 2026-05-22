"""Agency-modulen — AE-1, AE-2, AST-1.

AE-1: Agency med learning + flexibel målorientering.
    CuriosityDriver väljer mål baserat på information-gap, surprise,
    och reliability. Inte hardcoded routes — det är emergent från
    inre state.

AE-2: Embodiment — modell av action-effekter på environmental input.
    ActionEffectModel predikterar hur look_at(angle) påverkar input-
    intensity-distribution och mäter prediction-error.

AST-1: Attention schema — modell av attention-process.
    AttentionSchema representerar VARFÖR attention är var den är.
    Self-report: "min uppmärksamhet är på X för att Y".

Per Butlin et al. 2023 §6 + §7.
"""

from selvra_brain.agency.action_model import ActionEffectModel, ActionEffectPrediction
from selvra_brain.agency.attention_schema import AttentionReason, AttentionSchema
from selvra_brain.agency.curiosity import CuriosityDriver, GoalSignal
from selvra_brain.agency.types import ActionIntent, ActionType, GoalType

__all__ = [
    "ActionEffectModel",
    "ActionEffectPrediction",
    "ActionIntent",
    "ActionType",
    "AttentionReason",
    "AttentionSchema",
    "CuriosityDriver",
    "GoalSignal",
    "GoalType",
]
