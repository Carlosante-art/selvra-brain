"""birth_agent — emittar agent_born-event som refererar till Genome-id.

Per architectural decision 2026-05-22 (Fas 1h): Genome är inte ett event
i EventStore. Det är separat artefakt. Men det bör finnas en referens
i event-strömmen så att man från en agents biografi kan följa tillbaka
till hennes konfigurations-DNA.

agent_born är ett event där event_type kodar tidpunkt för agentens start,
och payload innehåller genome_id + name + initial conditions. Det är
typed som BODY_STATE eftersom det är ett kroppsligt-existens-event (denna
agent börjar nu vara i världen).
"""

from __future__ import annotations

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
from selvra_brain.genome.genome import Genome


def birth_agent(
    *,
    store: EventStore,
    genome: Genome,
    name: str = "selvra",
    initial_valence: Valence = Valence.NEUTRAL,
) -> BrainEvent:
    """Emit ett agent_born-event som markerar agentens start-i-världen.

    Args:
        store: EventStore som ska få event:et.
        genome: agentens (immutabla) konfigurations-DNA.
        name: agentens namn.
        initial_valence: affektiv baseline vid födsel. NEUTRAL är vad
            Damasio kallar "homeostatic null point".

    Returns:
        BrainEvent som har lagts till i store.
    """
    event = BrainEvent(
        category=EventCategory.BODY_STATE,
        event_type="agent_born",
        payload={
            "name": name,
            "genome_id": genome.genome_id,
            "genome": genome.to_dict(),
        },
        tag=EpistemicTag(
            data_type=DataType.OBSERVED,
            confidence=Confidence.HIGH,
            mutability=Mutability.IMMUTABLE,
            persistence=Persistence.CORE,
            memory_type=MemoryType.EPISODIC,
            valence=initial_valence,
        ),
    )
    store.append(event)
    return event
