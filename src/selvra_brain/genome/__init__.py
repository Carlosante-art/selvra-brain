"""Genome — agentens immutabla konfigurations-DNA.

Per architectural decision 2026-05-22 (Fas 1h): det här är nature-lagret.
Genome är frozen för en agents livstid. Vad som varierar över tid är
EventStore (epigenetik/biografi).

Två agenter med IDENTISK Genome men olika EventStores blir olika
"personligheter" över tid. Det är hela poängen med nature/nurture-
separationen i denna arkitektur.

Genome används för att:
- Konstruera alla moduler (build_brain_from_genome)
- Rekombinera till barn-Genome (Genome.recombine)
- Mutera (Genome.mutate)
- Persistera (to_dict/from_dict)
"""

from selvra_brain.genome.birth import birth_agent
from selvra_brain.genome.builder import BrainComponents, build_brain_from_genome
from selvra_brain.genome.genome import (
    Genome,
    GenomeRange,
    GenomeValidationError,
    GENOME_SCHEMA,
)

__all__ = [
    "BrainComponents",
    "GENOME_SCHEMA",
    "Genome",
    "GenomeRange",
    "GenomeValidationError",
    "birth_agent",
    "build_brain_from_genome",
]
