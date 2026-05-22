"""Tests för epistemic taxonomy.

Verifierar:
- Alla enum-värden existerar
- Valence har korrekt numeric-mapping (-1 till 1)
- Valence.from_numeric är inverse till to_numeric
- EpistemicTag är frozen
- EpistemicTag.to_dict producerar serialiserbar struktur
- Valence är obligatorisk (ingen default — per Damasio-arkitektur)
"""

from __future__ import annotations

import pytest

from selvra_brain.core.epistemic import (
    Confidence,
    DataType,
    EpistemicTag,
    MemoryType,
    Mutability,
    Persistence,
    Valence,
)


# ─── Enums exists ──────────────────────────────────────────────────────


def test_data_type_values() -> None:
    assert DataType.OBSERVED.value == "observed"
    assert DataType.DERIVED.value == "derived"
    assert DataType.PREDICTED.value == "predicted"
    assert DataType.SELF_REPORTED.value == "self_reported"


def test_confidence_has_four_levels() -> None:
    assert {c.value for c in Confidence} == {"high", "medium", "low", "unavailable"}


def test_memory_type_includes_working() -> None:
    """WORKING är central för GW-2 workspace — får inte saknas."""
    assert MemoryType.WORKING in MemoryType


# ─── Valence ───────────────────────────────────────────────────────────


def test_valence_five_levels() -> None:
    assert {v.value for v in Valence} == {
        "positive_strong",
        "positive",
        "neutral",
        "negative",
        "negative_strong",
    }


def test_valence_to_numeric_endpoints() -> None:
    assert Valence.POSITIVE_STRONG.to_numeric() == 1.0
    assert Valence.NEGATIVE_STRONG.to_numeric() == -1.0


def test_valence_to_numeric_midpoint() -> None:
    assert Valence.NEUTRAL.to_numeric() == 0.0


def test_valence_to_numeric_intermediates() -> None:
    assert Valence.POSITIVE.to_numeric() == 0.5
    assert Valence.NEGATIVE.to_numeric() == -0.5


def test_valence_from_numeric_endpoints() -> None:
    assert Valence.from_numeric(1.0) == Valence.POSITIVE_STRONG
    assert Valence.from_numeric(-1.0) == Valence.NEGATIVE_STRONG


def test_valence_from_numeric_zero_is_neutral() -> None:
    assert Valence.from_numeric(0.0) == Valence.NEUTRAL


def test_valence_from_numeric_clamps_out_of_range() -> None:
    assert Valence.from_numeric(2.0) == Valence.POSITIVE_STRONG
    assert Valence.from_numeric(-2.0) == Valence.NEGATIVE_STRONG


def test_valence_from_to_roundtrip_at_canonical_points() -> None:
    for v in Valence:
        assert Valence.from_numeric(v.to_numeric()) == v


# ─── EpistemicTag ──────────────────────────────────────────────────────


def test_tag_requires_valence_no_default() -> None:
    """Per Damasio: valence är obligatorisk, ingen default till neutral."""
    with pytest.raises(TypeError):
        EpistemicTag(  # type: ignore[call-arg]
            data_type=DataType.OBSERVED,
            confidence=Confidence.HIGH,
            mutability=Mutability.IMMUTABLE,
            persistence=Persistence.STABLE,
            memory_type=MemoryType.EPISODIC,
            # valence saknas avsiktligt
        )


def test_tag_construction() -> None:
    tag = EpistemicTag(
        data_type=DataType.OBSERVED,
        confidence=Confidence.HIGH,
        mutability=Mutability.IMMUTABLE,
        persistence=Persistence.STABLE,
        memory_type=MemoryType.EPISODIC,
        valence=Valence.NEUTRAL,
    )
    assert tag.valence == Valence.NEUTRAL


def test_tag_is_frozen() -> None:
    tag = EpistemicTag(
        data_type=DataType.OBSERVED,
        confidence=Confidence.HIGH,
        mutability=Mutability.IMMUTABLE,
        persistence=Persistence.STABLE,
        memory_type=MemoryType.EPISODIC,
        valence=Valence.NEUTRAL,
    )
    with pytest.raises(Exception):
        tag.valence = Valence.POSITIVE  # type: ignore[misc]


def test_tag_to_dict_includes_valence() -> None:
    tag = EpistemicTag(
        data_type=DataType.OBSERVED,
        confidence=Confidence.HIGH,
        mutability=Mutability.IMMUTABLE,
        persistence=Persistence.STABLE,
        memory_type=MemoryType.EPISODIC,
        valence=Valence.POSITIVE,
    )
    d = tag.to_dict()
    assert d["valence"] == "positive"
    assert "data_type" in d
    assert "confidence" in d
