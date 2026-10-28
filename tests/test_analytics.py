"""Automated unit tests for TraceRoot AI analytical and data transformation functions.

These tests verify pure functions without requiring a running Streamlit server.
"""

import pandas as pd
import pytest

from app import (
    ANOMALY_COLUMNS,
    LOG_LEVELS,
    calculate_statistical_cutoff,
    calculate_summary,
    count_by_severity,
    count_errors_per_service,
    detect_error_anomalies,
    filter_logs,
    get_error_records,
    load_sample_logs,
    summarize_error_messages,
    summarize_errors_by_service,
)


@pytest.fixture
def sample_logs() -> pd.DataFrame:
    """Fixture providing the standard 24 simulated log records."""
    return load_sample_logs()


@pytest.fixture
def empty_logs() -> pd.DataFrame:
    """Fixture providing an empty DataFrame with the expected log schema."""
    return pd.DataFrame(columns=["timestamp", "level", "service", "message"])


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------
def test_load_sample_logs_structure(sample_logs: pd.DataFrame):
    """Verify that sample logs contain exactly 24 records with proper schema."""
    assert len(sample_logs) == 24
    assert list(sample_logs.columns) == ["timestamp", "level", "service", "message"]
    assert pd.api.types.is_datetime64_any_dtype(sample_logs["timestamp"])
    # Verify chronological ordering
    assert sample_logs["timestamp"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Summary calculation tests
# ---------------------------------------------------------------------------
def test_calculate_summary_sample_data(sample_logs: pd.DataFrame):
    """Verify headline counts on the baseline sample dataset."""
    summary = calculate_summary(sample_logs)
    assert summary["total"] == 24
    assert summary["errors"] == 7
    assert summary["warnings"] == 7
    assert summary["services"] == 6


def test_calculate_summary_empty(empty_logs: pd.DataFrame):
    """Verify headline counts on an empty log dataset."""
    summary = calculate_summary(empty_logs)
    assert summary["total"] == 0
    assert summary["errors"] == 0
    assert summary["warnings"] == 0
    assert summary["services"] == 0


# ---------------------------------------------------------------------------
# Filtering tests
# ---------------------------------------------------------------------------
def test_filter_logs_all(sample_logs: pd.DataFrame):
    """Filtering with all levels and all services should preserve all records."""
    all_levels = LOG_LEVELS
    all_services = sorted(sample_logs["service"].unique())
    filtered = filter_logs(sample_logs, all_levels, all_services)
    assert len(filtered) == 24


def test_filter_logs_specific_subset(sample_logs: pd.DataFrame):
    """Filter by ERROR level and a specific service."""
    filtered = filter_logs(sample_logs, ["ERROR"], ["database-proxy"])
    assert len(filtered) == 2
    assert (filtered["level"] == "ERROR").all()
    assert (filtered["service"] == "database-proxy").all()


def test_filter_logs_empty_criteria(sample_logs: pd.DataFrame):
    """Clearing selections should return an empty DataFrame without errors."""
    filtered_no_levels = filter_logs(sample_logs, [], ["database-proxy"])
    assert filtered_no_levels.empty

    filtered_no_services = filter_logs(sample_logs, ["ERROR"], [])
    assert filtered_no_services.empty

    filtered_none = filter_logs(sample_logs, [], [])
    assert filtered_none.empty


# ---------------------------------------------------------------------------
# Severity counting tests
# ---------------------------------------------------------------------------
def test_count_by_severity_sample_data(sample_logs: pd.DataFrame):
    """Verify severity distribution counts on the sample data."""
    counts = count_by_severity(sample_logs)
    assert counts["INFO"] == 10
    assert counts["WARNING"] == 7
    assert counts["ERROR"] == 7
    assert list(counts.index) == LOG_LEVELS


def test_count_by_severity_empty(empty_logs: pd.DataFrame):
    """Empty logs should still produce a Series with all levels indexed at 0."""
    counts = count_by_severity(empty_logs)
    assert counts["INFO"] == 0
    assert counts["WARNING"] == 0
    assert counts["ERROR"] == 0
    assert list(counts.index) == LOG_LEVELS


# ---------------------------------------------------------------------------
# Error investigation helpers tests
# ---------------------------------------------------------------------------
def test_get_error_records(sample_logs: pd.DataFrame):
    """Verify get_error_records filters strictly for ERROR level rows."""
    errors = get_error_records(sample_logs)
    assert len(errors) == 7
    assert (errors["level"] == "ERROR").all()


def test_summarize_errors_by_service(sample_logs: pd.DataFrame):
    """Verify errors are grouped and sorted by count descending."""
    errors = get_error_records(sample_logs)
    summary = summarize_errors_by_service(errors)
    assert len(summary) == 4  # 4 services produced errors
    # Top services have 2 errors each
    assert summary.iloc[0]["error_count"] == 2
    assert set(summary["service"]) == {
        "api-gateway",
        "auth-service",
        "database-proxy",
        "payment-service",
    }


def test_summarize_error_messages(sample_logs: pd.DataFrame):
    """Verify identical error messages are grouped and repeat counts tracked."""
    errors = get_error_records(sample_logs)
    message_summary = summarize_error_messages(errors)
    # Database timeout and HTTP 500 repeat twice
    repeated = message_summary[message_summary["occurrences"] > 1]
    assert len(repeated) == 3
    for _, row in repeated.iterrows():
        assert row["occurrences"] == 2
        assert row["first_seen"] <= row["last_seen"]


# ---------------------------------------------------------------------------
# Anomaly detection tests
# ---------------------------------------------------------------------------
def test_count_errors_per_service(sample_logs: pd.DataFrame):
    """Verify error counts include 0 for services with only INFO/WARNING records."""
    counts = count_errors_per_service(sample_logs)
    assert len(counts) == 6
    service_map = dict(zip(counts["service"], counts["error_count"]))
    assert service_map["order-service"] == 0
    assert service_map["deployment-service"] == 0
    assert service_map["auth-service"] == 1
    assert service_map["api-gateway"] == 2
    assert service_map["database-proxy"] == 2
    assert service_map["payment-service"] == 2


def test_detect_error_anomalies_baseline_sample_data(sample_logs: pd.DataFrame):
    """Baseline sample data has max 2 errors per service, so no anomalies trigger."""
    counts = count_errors_per_service(sample_logs)
    anomalies = detect_error_anomalies(counts)
    assert anomalies.empty


def test_detect_error_anomalies_fixed_threshold_rule():
    """Service with >= 3 errors triggers Fixed threshold rule."""
    test_counts = pd.DataFrame(
        [
            {"service": "svc-a", "error_count": 3},
            {"service": "svc-b", "error_count": 3},
        ]
    )
    anomalies = detect_error_anomalies(test_counts)
    assert len(anomalies) == 2
    assert (anomalies["priority"] == "Medium").all()
    assert (anomalies["rules_triggered"] == "Fixed threshold").all()


def test_detect_error_anomalies_statistical_rule():
    """Service with 2 errors exceeding mean + 1 std triggers Above-average rule."""
    # 3 services: one has 2 errors, two have 0 errors
    # mean = 2/3 = 0.67, std = 0.94, cutoff = 1.61
    # svc-a has 2 errors: 2 >= 2 and 2 > 1.61 -> triggers above-average rule only
    test_counts = pd.DataFrame(
        [
            {"service": "svc-a", "error_count": 2},
            {"service": "svc-b", "error_count": 0},
            {"service": "svc-c", "error_count": 0},
        ]
    )
    anomalies = detect_error_anomalies(test_counts)
    assert len(anomalies) == 1
    assert anomalies.iloc[0]["service"] == "svc-a"
    assert anomalies.iloc[0]["rules_triggered"] == "Above-average rule"
    assert anomalies.iloc[0]["priority"] == "Medium"


def test_detect_error_anomalies_high_priority_both_rules():
    """Service triggering both Fixed threshold and Above-average rule gets High priority."""
    # 4 services: svc-a has 6 errors, others have 0
    # 6 >= 3 triggers Fixed threshold
    # 6 > mean + 1 std triggers Above-average rule
    test_counts = pd.DataFrame(
        [
            {"service": "svc-a", "error_count": 6},
            {"service": "svc-b", "error_count": 0},
            {"service": "svc-c", "error_count": 0},
            {"service": "svc-d", "error_count": 0},
        ]
    )
    anomalies = detect_error_anomalies(test_counts)
    assert len(anomalies) == 1
    top = anomalies.iloc[0]
    assert top["service"] == "svc-a"
    assert top["priority"] == "High"
    assert "Fixed threshold" in top["rules_triggered"]
    assert "Above-average rule" in top["rules_triggered"]


def test_detect_error_anomalies_empty():
    """Empty error counts should return an empty DataFrame with ANOMALY_COLUMNS."""
    empty_counts = pd.DataFrame(columns=["service", "error_count"])
    anomalies = detect_error_anomalies(empty_counts)
    assert anomalies.empty
    assert list(anomalies.columns) == ANOMALY_COLUMNS


def test_calculate_statistical_cutoff_insufficient_services():
    """When fewer than 3 services exist, cutoff calculation returns None."""
    small_counts = pd.DataFrame(
        [
            {"service": "svc-a", "error_count": 5},
            {"service": "svc-b", "error_count": 2},
        ]
    )
    assert calculate_statistical_cutoff(small_counts) is None

