"""Tests för MetacognitiveMonitor (HOT-2)."""

from __future__ import annotations

import pytest

from selvra_brain.core.events import EventCategory, EventStore
from selvra_brain.metacognition import MetacognitiveMonitor, ReliabilityAssessment
from selvra_brain.prediction.engine import HierarchicalPredictiveEngine


# ─── ReliabilityAssessment.from_errors ─────────────────────────


def test_perfect_calibration_reliability_one() -> None:
    a = ReliabilityAssessment.from_errors("temp", expected=2.0, actual=2.0)
    assert a.reliability == 1.0


def test_large_miss_low_reliability() -> None:
    a = ReliabilityAssessment.from_errors("temp", expected=1.0, actual=10.0)
    # miss = 9 → reliability = 1/10 = 0.1
    assert a.reliability == pytest.approx(0.1)


def test_reliability_in_unit_interval() -> None:
    a = ReliabilityAssessment.from_errors("temp", expected=0, actual=1000)
    assert 0.0 <= a.reliability <= 1.0


# ─── MetacognitiveMonitor ──────────────────────────────────────


def test_monitor_initial_global_reliability_is_half() -> None:
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=2)
    monitor = MetacognitiveMonitor(engine, store)
    assert monitor.global_reliability() == 0.5


def test_monitor_needs_l1_data() -> None:
    """Reliability kan inte bedömas innan L1-predictor har historia."""
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=2)
    monitor = MetacognitiveMonitor(engine, store)

    # Bara första observation — ingen L0-error än
    result = engine.observe(source="temp", value=20.0)
    assert monitor.update_from_result(result) is None

    # Andra observation — L0-error finns, men L1-error finns inte
    # (L1-predictor har bara sett ETT värde)
    result = engine.observe(source="temp", value=21.0)
    assert monitor.update_from_result(result) is None


def test_monitor_works_after_warmup() -> None:
    """Efter ~3 observations har vi L1-error som kan bedöma reliability."""
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=2)
    monitor = MetacognitiveMonitor(engine, store)

    for v in [10.0, 11.0, 12.0, 13.0]:
        result = engine.observe(source="temp", value=v)
        monitor.update_from_result(result)

    # Vid 4:e observation har vi tillräcklig data
    assessment = monitor.reliability_for_source("temp")
    assert assessment is not None
    assert 0.0 <= assessment <= 1.0


def test_monitor_stable_signal_high_reliability() -> None:
    """När signal är predikterbar lär engine sig → låg L0-error →
    L1 förutser låg L0-error → hög reliability."""
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=2)
    monitor = MetacognitiveMonitor(engine, store)

    # Stable signal: konstant
    for _ in range(10):
        result = engine.observe(source="temp", value=20.0)
        monitor.update_from_result(result)

    rel = monitor.reliability_for_source("temp")
    # När signalen är konstant blir L0-error nära 0,
    # L1 lär sig förutse låg error → bra kalibrering
    assert rel is not None
    assert rel > 0.4  # not perfect men decent


def test_monitor_event_emission() -> None:
    """METACOGNITION-events ska emit:as när reliability beräknas."""
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=2)
    monitor = MetacognitiveMonitor(engine, store)

    for v in [10.0, 11.0, 12.0, 13.0]:
        result = engine.observe(source="temp", value=v)
        monitor.update_from_result(result)

    meta_events = store.by_category(EventCategory.METACOGNITION)
    assert len(meta_events) >= 1
    # Event har global_reliability i payload (för visualization)
    payload = meta_events[-1].payload
    assert "reliability" in payload
    assert "global_reliability" in payload


def test_monitor_disable_event_emission() -> None:
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=2)
    monitor = MetacognitiveMonitor(engine, store, emit_events=False)

    for v in [10.0, 11.0, 12.0, 13.0]:
        result = engine.observe(source="temp", value=v)
        monitor.update_from_result(result)

    meta_events = store.by_category(EventCategory.METACOGNITION)
    assert len(meta_events) == 0


def test_monitor_stats() -> None:
    store = EventStore()
    engine = HierarchicalPredictiveEngine(store, levels=2)
    monitor = MetacognitiveMonitor(engine, store)

    for v in [10.0, 11.0, 12.0]:
        result = engine.observe(source="temp", value=v)
        monitor.update_from_result(result)

    stats = monitor.stats()
    assert "sources_tracked" in stats
    assert "global_reliability" in stats
    assert "per_source" in stats
