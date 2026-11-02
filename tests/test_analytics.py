"""Automated unit tests for TraceRoot AI analytical, simulation, and spike detection functions.

These tests verify pure functions without requiring a running Streamlit server.
"""

import hashlib
from unittest import mock
import pandas as pd
import pytest
import streamlit as st

from app import (
    ANOMALY_COLUMNS,
    LOG_LEVELS,
    MAX_UPLOAD_RECORDS,
    MAX_UPLOAD_SIZE_BYTES,
    SERVICES,
    SPIKE_COLUMNS,
    append_and_cap_logs,
    calculate_statistical_cutoff,
    calculate_summary,
    compute_uploaded_file_hash,
    count_by_severity,
    count_errors_per_service,
    detect_error_anomalies,
    detect_rolling_error_spikes,
    filter_logs,
    generate_error_spike,
    generate_simulated_logs,
    get_error_records,
    load_sample_logs,
    parse_jsonl_text,
    parse_log_text,
    parse_uploaded_log_file,
    reset_uploaded_file_state,
    should_reparse_uploaded_file,
    show_data_source_controls,
    show_error_spike_section,
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


# ===========================================================================
# Milestone 3: Real Log-File Ingestion & Parsing Tests
# ===========================================================================
def test_parse_log_text_valid():
    """Verify standard plain text .log parsing with valid records."""
    raw_log = """
    2026-10-29 10:15:30 ERROR payment-service Database connection timeout
    2026-10-29 10:15:45 INFO api-gateway GET /api/orders returned 200 OK in 45ms
    2026-10-29 10:16:10 WARNING auth-service Slow token verification took 1200ms
    """
    df, summary = parse_log_text(raw_log)

    assert len(df) == 3
    assert summary["valid_records"] == 3
    assert summary["rejected_records"] == 0
    assert summary["blank_lines"] == 2
    assert list(df.columns) == ["timestamp", "level", "service", "message"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].is_monotonic_increasing
    assert list(df["level"]) == ["ERROR", "INFO", "WARNING"]
    assert list(df["service"]) == ["payment-service", "api-gateway", "auth-service"]


def test_parse_jsonl_text_valid():
    """Verify JSON Lines parsing with primary and alternative key aliases."""
    raw_jsonl = """
    {"timestamp": "2026-10-29 10:15:30", "level": "ERROR", "service": "payment-service", "message": "Database connection timeout"}
    {"time": "2026-10-29 10:15:45", "severity": "INFO", "app": "api-gateway", "msg": "GET /api/orders returned 200 OK"}
    {"ts": "2026-10-29 10:16:00", "log_level": "WARN", "service_name": "auth-service", "message": "Slow token verification"}
    """
    df, summary = parse_jsonl_text(raw_jsonl)

    assert len(df) == 3
    assert summary["valid_records"] == 3
    assert summary["rejected_records"] == 0
    # "WARN" should be normalized to "WARNING"
    assert list(df["level"]) == ["ERROR", "INFO", "WARNING"]
    assert list(df["service"]) == ["payment-service", "api-gateway", "auth-service"]


def test_parse_log_text_invalid_timestamp():
    """Verify text log parser rejects unparseable or out-of-range timestamps."""
    raw_log = "2026-99-99 99:99:99 ERROR payment-service Database timeout"
    df, summary = parse_log_text(raw_log)

    assert df.empty
    assert summary["valid_records"] == 0
    assert summary["rejected_records"] == 1
    assert "Invalid timestamp" in summary["errors"][0]["reason"]


def test_parse_jsonl_text_invalid_timestamp():
    """Verify JSONL parser rejects invalid timestamp strings."""
    raw_jsonl = '{"timestamp": "not-a-real-date", "level": "ERROR", "service": "payment-service", "message": "fail"}'
    df, summary = parse_jsonl_text(raw_jsonl)

    assert df.empty
    assert summary["valid_records"] == 0
    assert summary["rejected_records"] == 1
    assert "Invalid timestamp" in summary["errors"][0]["reason"]


def test_parse_log_text_invalid_severity():
    """Verify text log parser rejects unsupported severity levels (e.g. DEBUG)."""
    raw_log = "2026-10-29 10:15:30 DEBUG payment-service Debug trace event"
    df, summary = parse_log_text(raw_log)

    assert df.empty
    assert summary["valid_records"] == 0
    assert summary["rejected_records"] == 1
    assert "Unsupported severity" in summary["errors"][0]["reason"]


def test_parse_jsonl_text_invalid_severity():
    """Verify JSONL parser rejects unsupported severity levels (e.g. CRITICAL)."""
    raw_jsonl = '{"timestamp": "2026-10-29 10:15:30", "level": "CRITICAL", "service": "payment-service", "message": "fail"}'
    df, summary = parse_jsonl_text(raw_jsonl)

    assert df.empty
    assert summary["valid_records"] == 0
    assert summary["rejected_records"] == 1
    assert "Unsupported severity" in summary["errors"][0]["reason"]


def test_parse_jsonl_text_malformed_json():
    """Verify JSONL parser handles invalid JSON syntax gracefully."""
    raw_jsonl = """
    {"timestamp": "2026-10-29 10:15:30", "level": "INFO", "service": "api", "message": "ok"}
    {this is broken json, not closed
    """
    df, summary = parse_jsonl_text(raw_jsonl)

    assert len(df) == 1
    assert summary["valid_records"] == 1
    assert summary["rejected_records"] == 1
    assert "Malformed JSON" in summary["errors"][0]["reason"]


def test_parse_empty_and_blank_lines():
    """Verify blank lines and empty strings are ignored gracefully without errors."""
    empty_text = "   \n\n\t  \n"
    df_log, summary_log = parse_log_text(empty_text)
    assert df_log.empty
    assert summary_log["valid_records"] == 0
    assert summary_log["rejected_records"] == 0

    df_jsonl, summary_jsonl = parse_jsonl_text(empty_text)
    assert df_jsonl.empty
    assert summary_jsonl["valid_records"] == 0
    assert summary_jsonl["rejected_records"] == 0


def test_parse_jsonl_missing_required_fields():
    """Verify JSONL rejects objects missing timestamp, level, service, or message."""
    # Missing timestamp
    _, s1 = parse_jsonl_text('{"level": "INFO", "service": "api", "message": "ok"}')
    assert s1["rejected_records"] == 1
    assert "timestamp" in s1["errors"][0]["reason"]

    # Missing level
    _, s2 = parse_jsonl_text('{"timestamp": "2026-10-29 10:15:30", "service": "api", "message": "ok"}')
    assert s2["rejected_records"] == 1
    assert "level" in s2["errors"][0]["reason"]

    # Missing service
    _, s3 = parse_jsonl_text('{"timestamp": "2026-10-29 10:15:30", "level": "INFO", "message": "ok"}')
    assert s3["rejected_records"] == 1
    assert "service" in s3["errors"][0]["reason"]

    # Missing message
    _, s4 = parse_jsonl_text('{"timestamp": "2026-10-29 10:15:30", "level": "INFO", "service": "api"}')
    assert s4["rejected_records"] == 1
    assert "message" in s4["errors"][0]["reason"]


def test_parse_log_text_mixed_records():
    """Verify mixed input imports valid records while recording rejection details."""
    mixed_content = """
    2026-10-29 10:15:30 INFO api-gateway Healthcheck passed
    Random unformatted text line that should fail
    2026-10-29 10:15:40 ERROR auth-service Invalid token
    2026-99-99 99:99:99 WARNING payment-service Bad time
    2026-10-29 10:15:50 FATAL database-proxy Bad severity
    2026-10-29 10:16:00 ERROR database-proxy Connection failed
    """
    df, summary = parse_log_text(mixed_content)

    assert len(df) == 3
    assert summary["valid_records"] == 3
    assert summary["rejected_records"] == 3
    assert set(df["service"]) == {"api-gateway", "auth-service", "database-proxy"}


def test_uploaded_records_flow_into_existing_analytics():
    """Verify uploaded records flow seamlessly into all existing dashboard analytics."""
    raw_log = """
    2026-10-29 10:00:00 INFO gateway Request 1
    2026-10-29 10:01:00 WARNING gateway High latency
    2026-10-29 10:02:00 ERROR payments Payment timed out
    2026-10-29 10:02:10 ERROR payments Payment timed out
    2026-10-29 10:02:20 ERROR payments Payment timed out
    2026-10-29 10:02:30 ERROR payments Payment timed out
    """
    df, _ = parse_log_text(raw_log)

    # 1. Summary metrics
    summary = calculate_summary(df)
    assert summary["total"] == 6
    assert summary["errors"] == 4
    assert summary["warnings"] == 1
    assert summary["services"] == 2

    # 2. Severity counting
    severity = count_by_severity(df)
    assert severity["ERROR"] == 4
    assert severity["WARNING"] == 1
    assert severity["INFO"] == 1

    # 3. Filtering
    filtered = filter_logs(df, ["ERROR"], ["payments"])
    assert len(filtered) == 4

    # 4. Error investigation
    err_summary = summarize_errors_by_service(get_error_records(df))
    assert err_summary.iloc[0]["service"] == "payments"
    assert err_summary.iloc[0]["error_count"] == 4

    # 5. Rolling error spike detection
    spikes = detect_rolling_error_spikes(df, window_minutes=10, error_threshold=3)
    assert len(spikes) == 1
    assert spikes.iloc[0]["service"] == "payments"
    assert spikes.iloc[0]["error_count"] == 4
    assert spikes.iloc[0]["severity"] == "High"


def test_no_mixing_between_simulated_and_uploaded_datasets():
    """Verify simulated baseline records and uploaded records maintain total isolation."""
    baseline_df = load_sample_logs()
    uploaded_text = "2026-10-29 12:00:00 ERROR custom-service Custom failure occurred"
    uploaded_df, _ = parse_log_text(uploaded_text)

    assert len(baseline_df) == 24
    assert len(uploaded_df) == 1
    assert "custom-service" not in set(baseline_df["service"])
    assert "payment-service" not in set(uploaded_df["service"])


def test_parse_uploaded_log_file_auto_detection():
    """Verify format auto-detection by filename extension and content probing."""
    json_str = '{"timestamp": "2026-10-29 10:00:00", "level": "INFO", "service": "api", "message": "ok"}'
    log_str = "2026-10-29 10:00:00 INFO api ok"

    # Auto-detect via extension
    df1, s1 = parse_uploaded_log_file(json_str, "sample.jsonl")
    assert s1["file_type"] == "jsonl"
    assert len(df1) == 1

    df2, s2 = parse_uploaded_log_file(log_str, "sample.log")
    assert s2["file_type"] == "text/log"
    assert len(df2) == 1

    # Auto-detect via content when extension is absent
    df3, s3 = parse_uploaded_log_file(json_str, "unnamed_data")
    assert s3["file_type"] == "jsonl"
    assert len(df3) == 1


def test_parse_log_text_capping_limit():
    """Verify input exceeding MAX_UPLOAD_RECORDS is safely capped without memory runaway."""
    # Generate 5005 log lines
    lines = [
        f"2026-10-29 10:00:{i%60:02d} INFO api Event {i}"
        for i in range(5005)
    ]
    df, summary = parse_log_text("\n".join(lines))
    assert len(df) == MAX_UPLOAD_RECORDS  # 5000
    assert summary["capped"] is True
    assert summary["valid_records"] == MAX_UPLOAD_RECORDS


# ---------------------------------------------------------------------------
# Milestone 5 - Task 5.1: Same-Filename File Re-Upload Hardening Tests
# ---------------------------------------------------------------------------
def test_compute_uploaded_file_hash_deterministic():
    """Verify compute_uploaded_file_hash produces deterministic SHA-256 for bytes and text."""
    raw_bytes = b"2026-10-29 10:00:00 INFO api User login\n"
    expected_hex = hashlib.sha256(raw_bytes).hexdigest()

    # From bytes
    assert compute_uploaded_file_hash(raw_bytes) == expected_hex
    # From string
    assert compute_uploaded_file_hash(raw_bytes.decode("utf-8")) == expected_hex
    # Minor content change produces different hash
    modified_bytes = b"2026-10-29 10:00:01 INFO api User login\n"
    assert compute_uploaded_file_hash(modified_bytes) != expected_hex

    # Invalid type raises TypeError
    with pytest.raises(TypeError, match="Expected bytes or str"):
        compute_uploaded_file_hash(12345)


def test_reparse_needed_on_first_upload(sample_logs: pd.DataFrame):
    """Verify first upload triggers parsing when no previously processed dataset exists."""
    assert should_reparse_uploaded_file(
        uploaded_df=None,
        last_filename=None,
        current_filename="production.log",
        last_hash=None,
        current_hash="hash_v1",
    ) is True

    # Even if last_filename happened to be set, None uploaded_df must re-parse
    assert should_reparse_uploaded_file(
        uploaded_df=None,
        last_filename="production.log",
        current_filename="production.log",
        last_hash="hash_v1",
        current_hash="hash_v1",
    ) is True


def test_reparse_triggered_on_same_filename_modified_content(sample_logs: pd.DataFrame):
    """Verify same filename with modified content hash triggers re-parsing."""
    hash_v1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    hash_v2 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    assert should_reparse_uploaded_file(
        uploaded_df=sample_logs,
        last_filename="service.log",
        current_filename="service.log",
        last_hash=hash_v1,
        current_hash=hash_v2,
    ) is True


def test_reparse_skipped_on_same_filename_identical_content(sample_logs: pd.DataFrame):
    """Verify same filename with identical content does not trigger unnecessary re-parsing."""
    hash_v1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    assert should_reparse_uploaded_file(
        uploaded_df=sample_logs,
        last_filename="service.log",
        current_filename="service.log",
        last_hash=hash_v1,
        current_hash=hash_v1,
    ) is False


def test_reparse_triggered_on_different_filename(sample_logs: pd.DataFrame):
    """Verify changing the filename triggers parsing even if content hash is identical or different."""
    hash_v1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    hash_v2 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    # Different filename, same hash
    assert should_reparse_uploaded_file(
        uploaded_df=sample_logs,
        last_filename="old_service.log",
        current_filename="new_service.log",
        last_hash=hash_v1,
        current_hash=hash_v1,
    ) is True

    # Different filename, different hash
    assert should_reparse_uploaded_file(
        uploaded_df=sample_logs,
        last_filename="old_service.log",
        current_filename="new_service.log",
        last_hash=hash_v1,
        current_hash=hash_v2,
    ) is True


def test_clearing_uploaded_file_resets_metadata_and_enables_reparsing(sample_logs: pd.DataFrame):
    """Verify reset_uploaded_file_state clears all 4 upload session state keys."""
    mock_state = {
        "uploaded_logs_df": sample_logs,
        "uploaded_file_summary": {"valid_records": 24},
        "uploaded_file_name": "app.log",
        "uploaded_file_hash": "hash_xyz",
    }

    reset_uploaded_file_state(mock_state)

    assert mock_state["uploaded_logs_df"] is None
    assert mock_state["uploaded_file_summary"] is None
    assert mock_state["uploaded_file_name"] is None
    assert mock_state["uploaded_file_hash"] is None

    # After clearing, re-uploading the exact same file must reparse
    assert should_reparse_uploaded_file(
        uploaded_df=mock_state["uploaded_logs_df"],
        last_filename=mock_state["uploaded_file_name"],
        current_filename="app.log",
        last_hash=mock_state["uploaded_file_hash"],
        current_hash="hash_xyz",
    ) is True


def test_invalid_upload_does_not_mark_hash_as_processed():
    """Verify that an invalid or failed upload does not record its hash as successfully processed."""
    mock_session_state = {
        "data_source_mode": "uploaded",
        "uploaded_logs_df": None,
        "uploaded_file_summary": None,
        "uploaded_file_name": None,
        "uploaded_file_hash": None,
    }

    # Simulate an uploaded file object whose size exceeds the limit
    oversized_file = mock.MagicMock()
    oversized_file.name = "huge.log"
    oversized_file.size = MAX_UPLOAD_SIZE_BYTES + 1024
    oversized_file.getvalue.return_value = b"some oversized content"

    with mock.patch.object(st, "session_state", mock_session_state, create=True), \
         mock.patch("streamlit.sidebar.radio", return_value="Uploaded Log File"), \
         mock.patch("streamlit.sidebar.file_uploader", return_value=oversized_file), \
         mock.patch("streamlit.sidebar.error") as mock_error, \
         mock.patch("streamlit.sidebar.markdown"), \
         mock.patch("streamlit.sidebar.subheader"), \
         mock.patch("streamlit.sidebar.header"):
        mode = show_data_source_controls()
        assert mode == "uploaded"
        mock_error.assert_called_once()
        assert "File too large!" in mock_error.call_args[0][0]
        # Hash must NOT be set to the oversized file's hash
        assert mock_session_state["uploaded_file_hash"] is None
        assert mock_session_state["uploaded_logs_df"] is None


def test_show_data_source_controls_same_filename_reupload_flow():
    """Verify integrated flow: same filename with modified content triggers re-parsing."""
    content_v1 = b"2026-10-29 10:00:00 INFO api Request 1\n"
    content_v2 = (
        b"2026-10-29 10:00:00 INFO api Request 1\n"
        b"2026-10-29 10:01:00 ERROR api Request 2 failed\n"
    )

    mock_session_state = {
        "data_source_mode": "uploaded",
        "uploaded_logs_df": None,
        "uploaded_file_summary": None,
        "uploaded_file_name": None,
        "uploaded_file_hash": None,
    }

    file_v1 = mock.MagicMock()
    file_v1.name = "service.log"
    file_v1.size = len(content_v1)
    file_v1.getvalue.return_value = content_v1

    # 1. First upload of service.log
    with mock.patch.object(st, "session_state", mock_session_state, create=True), \
         mock.patch("streamlit.sidebar.radio", return_value="Uploaded Log File"), \
         mock.patch("streamlit.sidebar.file_uploader", return_value=file_v1), \
         mock.patch("streamlit.sidebar.markdown"), \
         mock.patch("streamlit.sidebar.subheader"), \
         mock.patch("streamlit.sidebar.header"), \
         mock.patch("streamlit.sidebar.button", return_value=False):
        show_data_source_controls()

    assert mock_session_state["uploaded_file_name"] == "service.log"
    assert mock_session_state["uploaded_logs_df"] is not None
    assert len(mock_session_state["uploaded_logs_df"]) == 1
    expected_hash_v1 = hashlib.sha256(content_v1).hexdigest()
    assert mock_session_state["uploaded_file_hash"] == expected_hash_v1

    # 2. Second run: exact same file content & name -> does not re-parse
    with mock.patch.object(st, "session_state", mock_session_state, create=True), \
         mock.patch("streamlit.sidebar.radio", return_value="Uploaded Log File"), \
         mock.patch("streamlit.sidebar.file_uploader", return_value=file_v1), \
         mock.patch("streamlit.sidebar.markdown"), \
         mock.patch("streamlit.sidebar.subheader"), \
         mock.patch("streamlit.sidebar.header"), \
         mock.patch("streamlit.sidebar.button", return_value=False), \
         mock.patch("app.parse_uploaded_log_file") as mock_parse:
        show_data_source_controls()
        mock_parse.assert_not_called()

    # 3. Third run: same filename "service.log" with updated content_v2 -> triggers re-parsing
    file_v2 = mock.MagicMock()
    file_v2.name = "service.log"
    file_v2.size = len(content_v2)
    file_v2.getvalue.return_value = content_v2

    with mock.patch.object(st, "session_state", mock_session_state, create=True), \
         mock.patch("streamlit.sidebar.radio", return_value="Uploaded Log File"), \
         mock.patch("streamlit.sidebar.file_uploader", return_value=file_v2), \
         mock.patch("streamlit.sidebar.markdown"), \
         mock.patch("streamlit.sidebar.subheader"), \
         mock.patch("streamlit.sidebar.header"), \
         mock.patch("streamlit.sidebar.button", return_value=False):
        show_data_source_controls()

    assert mock_session_state["uploaded_file_name"] == "service.log"
    assert len(mock_session_state["uploaded_logs_df"]) == 2
    expected_hash_v2 = hashlib.sha256(content_v2).hexdigest()
    assert mock_session_state["uploaded_file_hash"] == expected_hash_v2
    assert expected_hash_v2 != expected_hash_v1

    # 4. Clear button clicked -> resets state
    with mock.patch.object(st, "session_state", mock_session_state, create=True), \
         mock.patch("streamlit.sidebar.radio", return_value="Uploaded Log File"), \
         mock.patch("streamlit.sidebar.file_uploader", return_value=None), \
         mock.patch("streamlit.sidebar.markdown"), \
         mock.patch("streamlit.sidebar.subheader"), \
         mock.patch("streamlit.sidebar.header"), \
         mock.patch("streamlit.sidebar.button", return_value=True), \
         mock.patch("streamlit.rerun") as mock_rerun:
        show_data_source_controls()
        mock_rerun.assert_called_once()

    assert mock_session_state["uploaded_logs_df"] is None
    assert mock_session_state["uploaded_file_name"] is None
    assert mock_session_state["uploaded_file_hash"] is None
    assert mock_session_state["uploaded_file_summary"] is None


def test_show_data_source_controls_parsing_exception_resets_hash():
    """Verify that if parse_uploaded_log_file raises an unexpected exception, state is cleared."""
    mock_session_state = {
        "data_source_mode": "uploaded",
        "uploaded_logs_df": None,
        "uploaded_file_summary": None,
        "uploaded_file_name": None,
        "uploaded_file_hash": "stale_hash",
    }

    bad_file = mock.MagicMock()
    bad_file.name = "corrupt.log"
    bad_file.size = 100
    bad_file.getvalue.return_value = b"corrupt binary bytes"

    with mock.patch.object(st, "session_state", mock_session_state, create=True), \
         mock.patch("streamlit.sidebar.radio", return_value="Uploaded Log File"), \
         mock.patch("streamlit.sidebar.file_uploader", return_value=bad_file), \
         mock.patch("streamlit.sidebar.markdown"), \
         mock.patch("streamlit.sidebar.subheader"), \
         mock.patch("streamlit.sidebar.header"), \
         mock.patch("streamlit.sidebar.button", return_value=False), \
         mock.patch("streamlit.sidebar.error") as mock_error, \
         mock.patch("app.parse_uploaded_log_file", side_effect=RuntimeError("Parsing crashed")):
        show_data_source_controls()
        mock_error.assert_called_once()
        assert "Parsing crashed" in mock_error.call_args[0][0]
        # All 4 keys must be reset to None atomically
        assert mock_session_state["uploaded_logs_df"] is None
        assert mock_session_state["uploaded_file_summary"] is None
        assert mock_session_state["uploaded_file_name"] is None
        assert mock_session_state["uploaded_file_hash"] is None


def test_oversized_replacement_upload_clears_all_state_and_dataset(sample_logs: pd.DataFrame):
    """Verify that an oversized replacement upload resets all 4 session keys and clears old data (TR-AUD-05)."""
    # Active valid dataset from prior upload
    mock_session_state = {
        "data_source_mode": "uploaded",
        "uploaded_logs_df": sample_logs.copy(),
        "uploaded_file_summary": {"valid_records": 24, "total_lines": 24, "rejected_records": 0, "blank_lines": 0},
        "uploaded_file_name": "previous_valid.log",
        "uploaded_file_hash": "previous_valid_hash_12345",
    }

    # User attempts to replace with an oversized file
    oversized_file = mock.MagicMock()
    oversized_file.name = "oversized_replacement.log"
    oversized_file.size = MAX_UPLOAD_SIZE_BYTES + 4096
    oversized_file.getvalue.return_value = b"oversized content"

    with mock.patch.object(st, "session_state", mock_session_state, create=True), \
         mock.patch("streamlit.sidebar.radio", return_value="Uploaded Log File"), \
         mock.patch("streamlit.sidebar.file_uploader", return_value=oversized_file), \
         mock.patch("streamlit.sidebar.markdown"), \
         mock.patch("streamlit.sidebar.subheader"), \
         mock.patch("streamlit.sidebar.header"), \
         mock.patch("streamlit.sidebar.button", return_value=False), \
         mock.patch("streamlit.sidebar.error") as mock_error:
        mode = show_data_source_controls()
        assert mode == "uploaded"
        mock_error.assert_called_once()
        assert "File too large!" in mock_error.call_args[0][0]

    # All four session keys must be atomically cleared (no stale dataset or split-brain metadata)
    assert mock_session_state["uploaded_logs_df"] is None
    assert mock_session_state["uploaded_file_summary"] is None
    assert mock_session_state["uploaded_file_name"] is None
    assert mock_session_state["uploaded_file_hash"] is None


def test_parsing_exception_replacement_clears_all_state_and_dataset(sample_logs: pd.DataFrame):
    """Verify that a replacement upload encountering a parse exception resets all 4 session keys (TR-AUD-05)."""
    # Active valid dataset from prior upload
    mock_session_state = {
        "data_source_mode": "uploaded",
        "uploaded_logs_df": sample_logs.copy(),
        "uploaded_file_summary": {"valid_records": 24},
        "uploaded_file_name": "previous_valid.log",
        "uploaded_file_hash": "previous_valid_hash_12345",
    }

    # User attempts to upload corrupt file
    corrupt_content = b"corrupt binary bytes that crash parser"
    bad_file = mock.MagicMock()
    bad_file.name = "corrupt_replacement.log"
    bad_file.size = len(corrupt_content)
    bad_file.getvalue.return_value = corrupt_content

    with mock.patch.object(st, "session_state", mock_session_state, create=True), \
         mock.patch("streamlit.sidebar.radio", return_value="Uploaded Log File"), \
         mock.patch("streamlit.sidebar.file_uploader", return_value=bad_file), \
         mock.patch("streamlit.sidebar.markdown"), \
         mock.patch("streamlit.sidebar.subheader"), \
         mock.patch("streamlit.sidebar.header"), \
         mock.patch("streamlit.sidebar.button", return_value=False), \
         mock.patch("streamlit.sidebar.error") as mock_error, \
         mock.patch("app.parse_uploaded_log_file", side_effect=RuntimeError("Decoder crash on corrupted tokens")):
        mode = show_data_source_controls()
        assert mode == "uploaded"
        mock_error.assert_called_once()
        assert "Decoder crash on corrupted tokens" in mock_error.call_args[0][0]

    # All four session keys must be reset to None
    assert mock_session_state["uploaded_logs_df"] is None
    assert mock_session_state["uploaded_file_summary"] is None
    assert mock_session_state["uploaded_file_name"] is None
    assert mock_session_state["uploaded_file_hash"] is None


def test_subsequent_valid_upload_recovers_after_failed_replacement(sample_logs: pd.DataFrame):
    """Verify that a subsequent valid upload works cleanly after an upload failure (TR-AUD-05)."""
    # Start with active valid upload
    mock_session_state = {
        "data_source_mode": "uploaded",
        "uploaded_logs_df": sample_logs.copy(),
        "uploaded_file_summary": {"valid_records": 24},
        "uploaded_file_name": "initial.log",
        "uploaded_file_hash": "initial_hash",
    }

    # Step 1: Failed replacement upload (oversized)
    bad_file = mock.MagicMock()
    bad_file.name = "oversized.log"
    bad_file.size = MAX_UPLOAD_SIZE_BYTES + 100
    bad_file.getvalue.return_value = b"oversized"

    with mock.patch.object(st, "session_state", mock_session_state, create=True), \
         mock.patch("streamlit.sidebar.radio", return_value="Uploaded Log File"), \
         mock.patch("streamlit.sidebar.file_uploader", return_value=bad_file), \
         mock.patch("streamlit.sidebar.markdown"), \
         mock.patch("streamlit.sidebar.subheader"), \
         mock.patch("streamlit.sidebar.header"), \
         mock.patch("streamlit.sidebar.button", return_value=False), \
         mock.patch("streamlit.sidebar.error"):
        show_data_source_controls()

    # Confirmed cleared
    assert mock_session_state["uploaded_logs_df"] is None

    # Step 2: User fixes the issue and uploads a new valid file
    valid_content = b"2026-10-29 10:00:00 INFO api Healthy recovery event\n"
    good_file = mock.MagicMock()
    good_file.name = "recovered.log"
    good_file.size = len(valid_content)
    good_file.getvalue.return_value = valid_content

    with mock.patch.object(st, "session_state", mock_session_state, create=True), \
         mock.patch("streamlit.sidebar.radio", return_value="Uploaded Log File"), \
         mock.patch("streamlit.sidebar.file_uploader", return_value=good_file), \
         mock.patch("streamlit.sidebar.markdown"), \
         mock.patch("streamlit.sidebar.subheader"), \
         mock.patch("streamlit.sidebar.header"), \
         mock.patch("streamlit.sidebar.button", return_value=False):
        show_data_source_controls()

    # Successfully ingested
    assert mock_session_state["uploaded_file_name"] == "recovered.log"
    assert mock_session_state["uploaded_logs_df"] is not None
    assert len(mock_session_state["uploaded_logs_df"]) == 1
    assert mock_session_state["uploaded_file_summary"]["valid_records"] == 1
    expected_hash = hashlib.sha256(valid_content).hexdigest()
    assert mock_session_state["uploaded_file_hash"] == expected_hash


# ---------------------------------------------------------------------------
# Milestone 5 - Task 5.2.2: Operational Spike Detection & Filter Alignment Tests
# ---------------------------------------------------------------------------
def test_show_error_spike_section_detects_spike_when_error_deselected_in_filters():
    """Verify operational spike detection alerts on active_logs even when ERROR is filtered out (TR-E2E-02)."""
    base_time = pd.Timestamp("2026-10-29 12:00:00")
    # 5 rapid ERROR records for payment-service within 5 minutes
    spike_records = [
        {"timestamp": base_time + pd.Timedelta(minutes=i), "level": "ERROR", "service": "payment-service", "message": f"Timeout {i}"}
        for i in range(5)
    ]
    # Plus normal INFO records
    info_records = [
        {"timestamp": base_time + pd.Timedelta(minutes=i), "level": "INFO", "service": "auth-service", "message": "Login ok"}
        for i in range(3)
    ]
    active_logs = pd.DataFrame(spike_records + info_records).sort_values("timestamp").reset_index(drop=True)

    # Operator deselects ERROR in visual sidebar filters: filtered_logs contains only INFO
    filtered_logs = filter_logs(active_logs, selected_levels=["INFO"], selected_services=["auth-service", "payment-service"])
    assert (filtered_logs["level"] == "ERROR").sum() == 0

    # Operational spike detection receives active_logs
    with mock.patch("streamlit.header"), \
         mock.patch("streamlit.caption"), \
         mock.patch("streamlit.expander"), \
         mock.patch("streamlit.markdown"), \
         mock.patch("streamlit.dataframe") as mock_df, \
         mock.patch("streamlit.error") as mock_error, \
         mock.patch("streamlit.success") as mock_success:
        show_error_spike_section(active_logs, window_minutes=10, error_threshold=3)

        # Operational alert must trigger
        mock_error.assert_called_once()
        assert "1 Service Error Spike(s) Detected!" in mock_error.call_args[0][0]
        # Success banner must NOT be displayed
        mock_success.assert_not_called()
        # Dataframe rendered in alert must contain payment-service
        rendered_df = mock_df.call_args[0][0]
        assert len(rendered_df) == 1
        assert rendered_df.iloc[0]["service"] == "payment-service"
        assert rendered_df.iloc[0]["error_count"] == 5


def test_show_error_spike_section_empty_dataset_renders_info_no_healthy_claim(empty_logs: pd.DataFrame):
    """Verify empty dataset displays informational message and does not claim services are healthy."""
    with mock.patch("streamlit.header"), \
         mock.patch("streamlit.caption"), \
         mock.patch("streamlit.expander"), \
         mock.patch("streamlit.markdown"), \
         mock.patch("streamlit.info") as mock_info, \
         mock.patch("streamlit.success") as mock_success, \
         mock.patch("streamlit.error") as mock_error:
        show_error_spike_section(empty_logs, window_minutes=10, error_threshold=3)

        mock_info.assert_called_once()
        assert "No log records available" in mock_info.call_args[0][0]
        # Must not display success or make false healthy claims
        mock_success.assert_not_called()
        mock_error.assert_not_called()


def test_show_error_spike_section_non_empty_no_spikes_accurate_messaging(sample_logs: pd.DataFrame):
    """Verify non-empty dataset with zero spikes shows success without claiming normal health."""
    with mock.patch("streamlit.header"), \
         mock.patch("streamlit.caption"), \
         mock.patch("streamlit.expander"), \
         mock.patch("streamlit.markdown"), \
         mock.patch("streamlit.info") as mock_info, \
         mock.patch("streamlit.success") as mock_success, \
         mock.patch("streamlit.error") as mock_error:
        show_error_spike_section(sample_logs, window_minutes=10, error_threshold=3)

        mock_info.assert_not_called()
        mock_error.assert_not_called()
        mock_success.assert_called_once()
        success_msg = mock_success.call_args[0][0]
        assert "No error spikes detected" in success_msg
        assert "active log dataset" in success_msg
        # Must not claim all services are healthy or operating normally
        assert "All services operating within expected rates" not in success_msg
        assert "operating normally" not in success_msg.lower()
        assert "healthy" not in success_msg.lower()


