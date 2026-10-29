"""Automated unit tests for TraceRoot AI analytical, simulation, and spike detection functions.

These tests verify pure functions without requiring a running Streamlit server.
"""

import pandas as pd
import pytest

from app import (
    ANOMALY_COLUMNS,
    LOG_LEVELS,
    SERVICES,
    SPIKE_COLUMNS,
    append_and_cap_logs,
    calculate_statistical_cutoff,
    calculate_summary,
    count_by_severity,
    count_errors_per_service,
    detect_error_anomalies,
    detect_rolling_error_spikes,
    filter_logs,
    generate_error_spike,
    generate_simulated_logs,
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
# Data loading tests (Milestone 1)
# ---------------------------------------------------------------------------
def test_load_sample_logs_structure(sample_logs: pd.DataFrame):
    """Verify that sample logs contain exactly 24 records with proper schema."""
    assert len(sample_logs) == 24
    assert list(sample_logs.columns) == ["timestamp", "level", "service", "message"]
    assert pd.api.types.is_datetime64_any_dtype(sample_logs["timestamp"])
    # Verify chronological ordering
    assert sample_logs["timestamp"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Summary calculation tests (Milestone 1)
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
# Filtering tests (Milestone 1)
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
# Severity counting tests (Milestone 1)
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
# Error investigation helpers tests (Milestone 1)
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
    # Database timeout, HTTP 500, and payment failed repeat twice
    repeated = message_summary[message_summary["occurrences"] > 1]
    assert len(repeated) == 3
    for _, row in repeated.iterrows():
        assert row["occurrences"] == 2
        assert row["first_seen"] <= row["last_seen"]


# ---------------------------------------------------------------------------
# Cohort anomaly detection tests (Milestone 1)
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


# ===========================================================================
# Milestone 2: Dynamic Simulation & Rolling Spike Detection Tests
# ===========================================================================
def test_generate_simulated_logs_structure():
    """Verify generated logs adhere to schema, valid levels, and valid services."""
    base_time = pd.Timestamp("2026-10-27 10:00:00")
    new_logs = generate_simulated_logs(base_time=base_time, count=8, seed=42)

    assert len(new_logs) == 8
    assert list(new_logs.columns) == ["timestamp", "level", "service", "message"]
    assert pd.api.types.is_datetime64_any_dtype(new_logs["timestamp"])
    assert new_logs["timestamp"].is_monotonic_increasing
    assert (new_logs["timestamp"] > base_time).all()
    assert set(new_logs["level"]).issubset(set(LOG_LEVELS))
    assert set(new_logs["service"]).issubset(set(SERVICES))


def test_generate_error_spike_structure():
    """Verify error spike generator produces only ERROR records for the specified service."""
    base_time = pd.Timestamp("2026-10-27 10:00:00")
    target_service = "payment-service"
    spike_logs = generate_error_spike(
        base_time=base_time, service=target_service, count=5
    )

    assert len(spike_logs) == 5
    assert (spike_logs["level"] == "ERROR").all()
    assert (spike_logs["service"] == target_service).all()
    assert spike_logs["timestamp"].is_monotonic_increasing
    assert (spike_logs["timestamp"] > base_time).all()


def test_append_and_cap_logs():
    """Verify appending preserves sorting and enforces maximum capacity."""
    base_logs = load_sample_logs()  # 24 records
    new_logs = generate_simulated_logs(base_time=base_logs["timestamp"].max(), count=6, seed=1)

    # Normal append without exceeding max
    combined = append_and_cap_logs(base_logs, new_logs, max_records=500)
    assert len(combined) == 30
    assert combined["timestamp"].is_monotonic_increasing

    # Test capacity cap
    capped = append_and_cap_logs(base_logs, new_logs, max_records=20)
    assert len(capped) == 20
    # Should retain the latest 20 records
    assert capped.iloc[-1]["timestamp"] == new_logs.iloc[-1]["timestamp"]


def test_append_and_cap_logs_empty():
    """Verify append_and_cap_logs handles empty inputs safely."""
    base_logs = load_sample_logs()
    empty_df = pd.DataFrame(columns=["timestamp", "level", "service", "message"])

    assert len(append_and_cap_logs(base_logs, empty_df)) == 24
    assert len(append_and_cap_logs(empty_df, base_logs)) == 24
    assert append_and_cap_logs(empty_df, empty_df).empty


def test_metrics_change_when_new_records_added(sample_logs: pd.DataFrame):
    """Verify summary metrics react dynamically to newly appended records."""
    initial_summary = calculate_summary(sample_logs)
    assert initial_summary["total"] == 24
    assert initial_summary["errors"] == 7

    # Add a spike of 4 errors on a new or existing service
    spike = generate_error_spike(
        base_time=sample_logs["timestamp"].max(), service="auth-service", count=4
    )
    updated_logs = append_and_cap_logs(sample_logs, spike)
    updated_summary = calculate_summary(updated_logs)

    assert updated_summary["total"] == 28
    assert updated_summary["errors"] == 11
    assert updated_summary["warnings"] == initial_summary["warnings"]


def test_detect_rolling_error_spikes_baseline_no_alert(sample_logs: pd.DataFrame):
    """In baseline 10-minute window (09:38 to 09:48), max errors for any service is 1 (< 3)."""
    # Max timestamp is 09:48:01. In the last 10 minutes, only 1 error exists.
    spikes = detect_rolling_error_spikes(
        sample_logs, window_minutes=10, error_threshold=3
    )
    assert spikes.empty
    assert list(spikes.columns) == SPIKE_COLUMNS


def test_detect_rolling_error_spikes_trigger_high_severity(sample_logs: pd.DataFrame):
    """Injecting 3 errors within the window triggers a High severity spike alert."""
    latest_ts = sample_logs["timestamp"].max()
    spike = generate_error_spike(base_time=latest_ts, service="order-service", count=3)
    combined = append_and_cap_logs(sample_logs, spike)

    spikes = detect_rolling_error_spikes(
        combined, window_minutes=10, error_threshold=3
    )
    assert len(spikes) == 1
    record = spikes.iloc[0]
    assert record["service"] == "order-service"
    assert record["error_count"] == 3
    assert record["threshold"] == 3
    assert record["severity"] == "High"
    assert "order-service" in record["explanation"]


def test_detect_rolling_error_spikes_trigger_critical_severity(sample_logs: pd.DataFrame):
    """Injecting >= 2x threshold errors triggers a Critical severity alert."""
    latest_ts = sample_logs["timestamp"].max()
    # 6 errors with threshold 3 -> Critical
    spike = generate_error_spike(base_time=latest_ts, service="payment-service", count=6)
    combined = append_and_cap_logs(sample_logs, spike)

    spikes = detect_rolling_error_spikes(
        combined, window_minutes=10, error_threshold=3
    )
    assert len(spikes) >= 1
    payment_spike = spikes[spikes["service"] == "payment-service"].iloc[0]
    # May include previous baseline error if within 10 min window
    assert payment_spike["error_count"] >= 6
    assert payment_spike["severity"] == "Critical"


def test_detect_rolling_error_spikes_time_window_boundary():
    """Errors outside the rolling window should not trigger a spike."""
    ref_time = pd.Timestamp("2026-10-27 12:00:00")
    # 5 errors 30 minutes ago
    old_errors = pd.DataFrame(
        [
            {
                "timestamp": ref_time - pd.Timedelta(minutes=30),
                "level": "ERROR",
                "service": "database-proxy",
                "message": "Old timeout",
            }
            for _ in range(5)
        ]
    )
    # 1 error 2 minutes ago
    recent_error = pd.DataFrame(
        [
            {
                "timestamp": ref_time - pd.Timedelta(minutes=2),
                "level": "ERROR",
                "service": "database-proxy",
                "message": "Recent timeout",
            }
        ]
    )
    logs = pd.concat([old_errors, recent_error], ignore_index=True)

    # 10-minute window from ref_time: only recent_error is inside (count 1 < threshold 3)
    spikes = detect_rolling_error_spikes(
        logs, window_minutes=10, error_threshold=3, reference_time=ref_time
    )
    assert spikes.empty

    # 45-minute window from ref_time: all 6 errors are inside (count 6 >= threshold 3)
    wide_spikes = detect_rolling_error_spikes(
        logs, window_minutes=45, error_threshold=3, reference_time=ref_time
    )
    assert len(wide_spikes) == 1
    assert wide_spikes.iloc[0]["error_count"] == 6


def test_detect_rolling_error_spikes_multiple_services():
    """Multiple services spiking simultaneously are reported independently."""
    ref_time = pd.Timestamp("2026-10-27 12:00:00")
    logs = pd.DataFrame(
        [
            {
                "timestamp": ref_time - pd.Timedelta(minutes=1),
                "level": "ERROR",
                "service": "api-gateway",
                "message": "500",
            }
            for _ in range(4)
        ]
        + [
            {
                "timestamp": ref_time - pd.Timedelta(minutes=2),
                "level": "ERROR",
                "service": "auth-service",
                "message": "Auth fail",
            }
            for _ in range(7)
        ]
    )

    spikes = detect_rolling_error_spikes(
        logs, window_minutes=10, error_threshold=3, reference_time=ref_time
    )
    assert len(spikes) == 2
    # Sorted by error_count descending
    assert spikes.iloc[0]["service"] == "auth-service"
    assert spikes.iloc[0]["error_count"] == 7
    assert spikes.iloc[0]["severity"] == "Critical"

    assert spikes.iloc[1]["service"] == "api-gateway"
    assert spikes.iloc[1]["error_count"] == 4
    assert spikes.iloc[1]["severity"] == "High"


def test_detect_rolling_error_spikes_empty_and_zero_errors(empty_logs: pd.DataFrame):
    """Empty dataset and logs with zero ERRORs return empty SPIKE_COLUMNS DataFrame."""
    assert detect_rolling_error_spikes(empty_logs).empty
    assert list(detect_rolling_error_spikes(empty_logs).columns) == SPIKE_COLUMNS

    info_only = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-10-27 12:00:00"),
                "level": "INFO",
                "service": "api-gateway",
                "message": "200 OK",
            }
        ]
    )
    assert detect_rolling_error_spikes(info_only).empty
