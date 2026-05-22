"""Global Workspace Theory modules.

Per Butlin et al:
- GW-1: Multipla specialiserade system kapabla att operera parallellt
- GW-2: Begränsad kapacitet workspace
- GW-3: Global broadcast
- GW-4: Tillstånds-beroende attention

Användning:

    from selvra_brain.workspace import GlobalWorkspace, WorkspaceItem, WorkspaceSource

    ws = GlobalWorkspace(event_store, capacity=5)
    ws.subscribe(lambda signal: print(signal.item.content))
    ws.propose(WorkspaceItem(
        content="bright_object",
        source=WorkspaceSource.PERCEPTION,
        priority=0.7,
        valence=0.3,
    ))
"""

from selvra_brain.workspace.global_workspace import (
    BroadcastSubscriber,
    GlobalWorkspace,
)
from selvra_brain.workspace.producers import (
    PredictionErrorProducer,
    prediction_error_to_workspace_item,
)
from selvra_brain.workspace.types import (
    AttentionVector,
    BroadcastSignal,
    WorkspaceItem,
    WorkspaceSource,
)

__all__ = [
    "AttentionVector",
    "BroadcastSignal",
    "BroadcastSubscriber",
    "GlobalWorkspace",
    "PredictionErrorProducer",
    "WorkspaceItem",
    "WorkspaceSource",
    "prediction_error_to_workspace_item",
]
