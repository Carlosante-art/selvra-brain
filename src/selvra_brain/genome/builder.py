"""build_brain_from_genome — factory som instantierar moduler från Genome.

Detta är arkitektoniska Single Source of Truth: när man bygger en agent
ska Genome vara enda källan till parametrarna. Direkt-instansiering av
moduler (utan Genome) är fortfarande tillåten för existerande tester och
demos, men nya kod-paths bör gå via builder.

BrainComponents bundlar alla moduler så att caller har en samlad
referens. Visualization, demos, och multi-agent-orchestration kan
returnera/passera BrainComponents istället för 7 separata variabler.
"""

from __future__ import annotations

from dataclasses import dataclass

from selvra_brain.agency import (
    ActionEffectModel,
    AttentionSchema,
    CuriosityDriver,
)
from selvra_brain.core.events import EventStore
from selvra_brain.genome.genome import Genome
from selvra_brain.metacognition import MetacognitiveMonitor
from selvra_brain.perception import PerceptionModule
from selvra_brain.prediction import HierarchicalPredictiveEngine
from selvra_brain.workspace import GlobalWorkspace, PredictionErrorProducer


@dataclass(frozen=True)
class BrainComponents:
    """Samlad referens till alla moduler en agent har.

    Frozen så att man inte oavsiktligt swappar ut en modul mid-life
    (vilket skulle bryta nature/nurture-separationen — moduler
    konfigureras vid födsel, beteende varierar via EventStore).
    """

    genome: Genome
    store: EventStore
    perception: PerceptionModule
    prediction: HierarchicalPredictiveEngine
    workspace: GlobalWorkspace
    producer: PredictionErrorProducer
    monitor: MetacognitiveMonitor
    curiosity: CuriosityDriver
    action_model: ActionEffectModel
    attention_schema: AttentionSchema


def build_brain_from_genome(
    genome: Genome,
    *,
    store: EventStore | None = None,
) -> BrainComponents:
    """Instantiera alla moduler från en Genome.

    Args:
        genome: konfigurations-DNA. Immutable.
        store: existerande EventStore eller None (skapar ny). Tillåter
            multi-agent-setup där flera agenter delar event-store.

    Returns:
        BrainComponents med alla moduler initialiserade enligt Genome.
    """
    if store is None:
        store = EventStore()

    perception = PerceptionModule(
        store=store,
        smoothing_weight=genome.smoothing_weight,
        valence_scale=genome.perception_valence_scale,
    )
    prediction = HierarchicalPredictiveEngine(
        store,
        levels=genome.pred_levels,
    )
    workspace = GlobalWorkspace(
        store,
        capacity=genome.ws_capacity,
        acceptance_threshold=genome.ws_acceptance_threshold,
    )
    producer = PredictionErrorProducer(
        workspace,
        surprise_threshold=genome.producer_surprise_threshold,
    )
    monitor = MetacognitiveMonitor(
        prediction,
        store,
        window=genome.metacog_window,
    )
    curiosity = CuriosityDriver(
        store=store,
        weight_reduce_uncertainty=genome.weight_reduce_uncertainty,
        weight_investigate_surprise=genome.weight_investigate_surprise,
        weight_explore_neglected=genome.weight_explore_neglected,
        weight_valence=genome.weight_valence,
        surprise_recency_window=genome.surprise_recency_window,
        drive_threshold=genome.drive_threshold,
    )
    action_model = ActionEffectModel(
        store=store,
        learning_rate=genome.action_learning_rate,
        estimated_focus_width=genome.initial_focus_width_radians,
    )
    # ActionEffectModel:s initial_periphery_intensity är private field
    # — vi sätter den explicit post-construction. Acceptabelt för
    # genome-init eftersom det är födsels-konfiguration, inte runtime-mutation.
    object.__setattr__(
        action_model,
        "_periphery_intensity_low",
        genome.initial_periphery_intensity,
    )
    attention_schema = AttentionSchema(
        store=store,
        transition_window=genome.attention_transition_window,
    )

    return BrainComponents(
        genome=genome,
        store=store,
        perception=perception,
        prediction=prediction,
        workspace=workspace,
        producer=producer,
        monitor=monitor,
        curiosity=curiosity,
        action_model=action_model,
        attention_schema=attention_schema,
    )
