"""GlobalWorkspace — bounded buffer + attention + broadcast.

Per Butlin et al:
- GW-2: bounded capacity (default 5 items — Miller's 7±2)
- GW-3: global broadcast till alla subscribers
- GW-4: state-dependent attention

Användning:

    workspace = GlobalWorkspace(event_store, capacity=5)

    # Subscriber registreras
    def on_broadcast(signal):
        print(f"Got broadcast: {signal.item.content}")
    workspace.subscribe(on_broadcast)

    # Producer skickar candidate
    item = WorkspaceItem(content="bright_light", source=WorkspaceSource.PERCEPTION,
                          priority=0.7, valence=0.3)
    workspace.propose(item)
    # Om saliency > threshold OCH workspace har plats → accepted, broadcasted

Workspace har INTERN state — items kan ackumuleras tills capacity.
Vid capacity-overflow ersätts lägsta-saliency-item av nya. Detta är
attention-bottleneck per GW-2.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from selvra_brain.core.epistemic import (
    Confidence,
    DataType,
    EpistemicTag,
    MemoryType,
    Mutability,
    Persistence,
    Valence,
)
from selvra_brain.core.events import BrainEvent, EventCategory, EventStore
from selvra_brain.workspace.types import (
    AttentionVector,
    BroadcastSignal,
    WorkspaceItem,
    WorkspaceSource,
)

logger = logging.getLogger(__name__)


BroadcastSubscriber = Callable[[BroadcastSignal], None]


class GlobalWorkspace:
    """Bounded buffer + saliency-baserad attention + pub/sub broadcast.

    Per GW-2: max `capacity` items samtidigt. Vid overflow ersätts
    item med lägst saliency. Detta är hard bottleneck — det är vad
    GW säger gör medvetandet "begränsat".
    """

    def __init__(
        self,
        event_store: EventStore,
        *,
        capacity: int = 5,
        acceptance_threshold: float = 0.3,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity måste vara >= 1")
        self.store = event_store
        self.capacity = capacity
        self.acceptance_threshold = acceptance_threshold
        self._items: list[WorkspaceItem] = []
        self._subscribers: list[BroadcastSubscriber] = []
        self._total_proposed: int = 0
        self._total_accepted: int = 0
        self._total_rejected: int = 0

    # ─── Pub/sub (GW-3) ─────────────────────────────────────────

    def subscribe(self, callback: BroadcastSubscriber) -> None:
        """Registrera broadcast-mottagare. Alla items som accepteras
        i workspace skickas till alla subscribers."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: BroadcastSubscriber) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _broadcast(self, item: WorkspaceItem) -> None:
        signal = BroadcastSignal(item=item, workspace_size=len(self._items))
        for sub in self._subscribers:
            try:
                sub(signal)
            except Exception as exc:  # noqa: BLE001
                # En subscribers fel ska inte stoppa de andra
                logger.warning("broadcast_subscriber_failed: %s", exc)

    # ─── Attention-selection (GW-4) ────────────────────────────

    def _attention_select(self, candidate: WorkspaceItem) -> bool:
        """State-dependent attention: ska denna candidate få plats?

        Default heuristic:
        1. Saliency under threshold → reject
        2. Annars: alltid acceptera. Capacity-management görs i propose().

        Override denna metod för custom state-dependent logic
        (t.ex. valence-context, current task, etc).
        """
        return candidate.saliency() >= self.acceptance_threshold

    def _replace_lowest_saliency(self, new_item: WorkspaceItem) -> WorkspaceItem | None:
        """När capacity nådd: ersätt lägst-saliency item om ny är högre.

        Returnerar det ersatta itemet (om något) — nyttigt för audit.
        """
        if not self._items:
            return None
        lowest = min(self._items, key=lambda i: i.saliency())
        if new_item.saliency() > lowest.saliency():
            self._items.remove(lowest)
            self._items.append(new_item)
            return lowest
        return None

    # ─── Main entry: propose ──────────────────────────────────

    def propose(self, item: WorkspaceItem) -> bool:
        """Producer-modul vill ha sitt item in i workspace.

        Returnerar True om item accepteras (och broadcastas), False
        om den avvisas. Avvisas vid:
        - saliency < threshold
        - capacity full OCH item:s saliency < befintliga
        """
        self._total_proposed += 1

        if not self._attention_select(item):
            self._total_rejected += 1
            return False

        if len(self._items) < self.capacity:
            self._items.append(item)
        else:
            replaced = self._replace_lowest_saliency(item)
            if replaced is None:
                # Ny item var inte högre än lägsta — reject
                self._total_rejected += 1
                return False

        self._total_accepted += 1
        self._emit_workspace_entry(item)
        self._broadcast(item)
        return True

    # ─── Inspection ────────────────────────────────────────────

    def current_items(self) -> tuple[WorkspaceItem, ...]:
        """Read-only snapshot av items i workspace."""
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def is_full(self) -> bool:
        return len(self._items) >= self.capacity

    def stats(self) -> dict[str, int]:
        return {
            "proposed": self._total_proposed,
            "accepted": self._total_accepted,
            "rejected": self._total_rejected,
            "current_size": len(self._items),
            "capacity": self.capacity,
        }

    def clear(self) -> None:
        """Töm workspace. Används mellan epoker eller vid testning."""
        self._items.clear()

    # ─── Event-emission ────────────────────────────────────────

    def _emit_workspace_entry(self, item: WorkspaceItem) -> None:
        """Emit WORKSPACE_ENTRY-event så event-store och visualization vet."""
        # Mappa item-valence (float) till EpistemicTag.Valence enum
        if item.valence >= 0.75:
            v_enum = Valence.POSITIVE_STRONG
        elif item.valence >= 0.25:
            v_enum = Valence.POSITIVE
        elif item.valence <= -0.75:
            v_enum = Valence.NEGATIVE_STRONG
        elif item.valence <= -0.25:
            v_enum = Valence.NEGATIVE
        else:
            v_enum = Valence.NEUTRAL

        tag = EpistemicTag(
            data_type=DataType.DERIVED,
            confidence=Confidence.HIGH,
            mutability=Mutability.SYSTEM_MUTABLE,
            persistence=Persistence.SHORT_TERM,
            memory_type=MemoryType.WORKING,
            valence=v_enum,
        )

        self.store.append(
            BrainEvent(
                category=EventCategory.WORKSPACE_ENTRY,
                event_type=f"workspace_{item.source.value}",
                payload={
                    "content": item.content,
                    "source": item.source.value,
                    "priority": item.priority,
                    "valence": item.valence,
                    "saliency": item.saliency(),
                    "attention_x": item.attention_vector.x,
                    "attention_y": item.attention_vector.y,
                    "workspace_size": len(self._items),
                    **item.payload,
                },
                tag=tag,
            )
        )
