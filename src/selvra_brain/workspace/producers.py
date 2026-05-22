"""Producer-adapters: kopplar source-moduler till GlobalWorkspace.

Per GW-1: multipla specialiserade moduler producerar kandidater
parallellt. Detta är limet — predictor → workspace, perception →
workspace, etc.

Producers är tunna — de översätter modul-events till WorkspaceItems
och anropar workspace.propose(). De håller INTE workspace-state.
"""

from __future__ import annotations

from selvra_brain.prediction.predictor import PredictionError
from selvra_brain.workspace.global_workspace import GlobalWorkspace
from selvra_brain.workspace.types import AttentionVector, WorkspaceItem, WorkspaceSource


def prediction_error_to_workspace_item(
    error: PredictionError,
    *,
    surprise_threshold: float = 0.3,
) -> WorkspaceItem | None:
    """Mappa prediction-error till workspace-candidate.

    Bara errors med normalized_magnitude > threshold producerar
    candidates — vi vill inte att varje liten avvikelse stör
    workspace. Det är substrat för GW-2 attention-bottleneck.

    Returnerar None om error är för liten — då propose:as aldrig.
    """
    if error.normalized_magnitude < surprise_threshold:
        return None

    # Attention-vector: surprises drar blicken mot källan.
    # För Fas 1 har vi inga 2D-källor — vi använder error-signed
    # som horizontal axis (positiv surprise → höger, negativ → vänster)
    # och magnitude som vertical (stor surprise → upp = "uppmärksamhet").
    direction_x = max(-1.0, min(1.0, error.signed / max(error.magnitude, 1.0)))
    direction_y = -error.normalized_magnitude  # upp = mer surprise

    # Valence: stor surprise → negativ (orient-response per Damasio)
    valence = -error.normalized_magnitude * 0.7

    # Priority: hur "viktig" tycker prediktor att detta är?
    # Vi binder priority till normalized_magnitude — större surprise
    # → högre priority för workspace.
    priority = error.normalized_magnitude

    return WorkspaceItem(
        content=f"surprise:{error.source_name}(±{error.magnitude:.2f})",
        source=WorkspaceSource.PREDICTION_ERROR,
        priority=priority,
        valence=valence,
        attention_vector=AttentionVector(x=direction_x, y=direction_y),
        payload={
            "predictor_source": error.source_name,
            "magnitude": error.magnitude,
            "normalized_magnitude": error.normalized_magnitude,
            "signed": error.signed,
            "level": error.level,
            "predicted": error.predicted,
            "observed": error.observed,
        },
    )


class PredictionErrorProducer:
    """Producer som auto-propose:ar prediction-errors > threshold.

    Användning:

        producer = PredictionErrorProducer(workspace, surprise_threshold=0.3)
        # När engine.observe körs, anropa:
        result = engine.observe(source="temp", value=42.0)
        if result.level_0_error:
            producer.handle_error(result.level_0_error)
    """

    def __init__(
        self,
        workspace: GlobalWorkspace,
        *,
        surprise_threshold: float = 0.3,
    ) -> None:
        self.workspace = workspace
        self.surprise_threshold = surprise_threshold
        self.proposed_count = 0
        self.accepted_count = 0

    def handle_error(self, error: PredictionError) -> bool:
        """Konvertera + propose. Returnerar True om accepterad."""
        item = prediction_error_to_workspace_item(
            error, surprise_threshold=self.surprise_threshold
        )
        if item is None:
            return False
        self.proposed_count += 1
        accepted = self.workspace.propose(item)
        if accepted:
            self.accepted_count += 1
        return accepted
