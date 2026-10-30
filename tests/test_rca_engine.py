"""
Unit tests for TraceRoot AI Rule-Based Root Cause Analysis (RCA) Engine.
(Milestone 4 - Layer 1: Deterministic Offline Correlation & Diagnosis)

All tests are pure, deterministic, and execute completely offline without external APIs.
"""

import pandas as pd
import pytest

from rca_engine import (
    calculate_confidence_score,
    correlate_temporal_events,
    detect_known_failure_patterns,
    identify_patient_zero,
    run_rule_based_rca,
)


@pytest.fixture
def db_exhaustion_logs() -> pd.DataFrame:
    """Fixture providing a classic 3-tier Database Exhaustion Cascade incident."""
    records = [
        ("2026-10-27 10:00:00", "ERROR", "database-proxy", "Database connection timeout after 30 seconds"),
        ("2026-10-27 10:00:15", "ERROR", "payment-service", "Payment failed for order #5002: database connection timeout"),
        ("2026-10-27 10:00:30", "ERROR", "api-gateway", "HTTP 504 Gateway Timeout: upstream service unreachable"),
        ("2026-10-27 10:00:45", "ERROR", "database-proxy", "Connection pool exhausted (200/200 connections active)"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def deployment_regression_logs() -> pd.DataFrame:
    """Fixture providing a deployment event followed by an error surge within 5 minutes."""
    records = [
        ("2026-10-27 11:00:00", "INFO", "deployment-service", "Deployment completed: payment-service v2.4.1"),
        ("2026-10-27 11:01:30", "ERROR", "payment-service", "Payment transaction declined: processor unreachable (503)"),
        ("2026-10-27 11:02:15", "ERROR", "order-service", "Order creation failed: database constraint violation"),
        ("2026-10-27 11:03:00", "ERROR", "payment-service", "Payment failed for order #7001: connection timeout"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def auth_cascade_logs() -> pd.DataFrame:
    """Fixture providing an Authentication Cascade incident."""
    records = [
        ("2026-10-27 12:00:00", "ERROR", "auth-service", "OAuth provider timeout after 15 seconds"),
        ("2026-10-27 12:00:10", "ERROR", "api-gateway", "Authentication failure: token validation service unreachable"),
        ("2026-10-27 12:00:25", "ERROR", "order-service", "Unauthorized request: token validation service unreachable"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Test 1: Database Exhaustion Cascade
# ---------------------------------------------------------------------------
def test_database_exhaustion_cascade(db_exhaustion_logs: pd.DataFrame):
    """Verify that a 3-tier database timeout sequence is matched with correct root cause candidate."""
    patterns = detect_known_failure_patterns(db_exhaustion_logs)
    db_pattern = next((p for p in patterns if p["pattern_name"] == "Database Exhaustion Cascade"), None)

    assert db_pattern is not None
    assert db_pattern["matched"] is True
    assert db_pattern["suspected_root_cause_service"] == "database-proxy"
    assert db_pattern["temporal_order_valid"] is True
    assert set(db_pattern["affected_services"]) == {"database-proxy", "payment-service", "api-gateway"}
    assert len(db_pattern["evidence"]) >= 3

    # Run full RCA
    result = run_rule_based_rca(db_exhaustion_logs)
    assert result["status"] == "completed"
    assert len(result["ranked_causes"]) > 0

    top_cause = result["ranked_causes"][0]
    assert top_cause["candidate_service"] == "database-proxy"
    assert top_cause["confidence_score"] >= 75.0
    assert "Database Exhaustion Cascade" in top_cause["possible_root_cause"]
    assert top_cause["classification"] == "Hypothesis (Rule-Based Heuristic)"


# ---------------------------------------------------------------------------
# Test 2: Deployment Regression
# ---------------------------------------------------------------------------
def test_deployment_regression(deployment_regression_logs: pd.DataFrame):
    """Verify that an error surge following a deployment event within 5 mins is detected."""
    patterns = detect_known_failure_patterns(
        deployment_regression_logs[deployment_regression_logs["level"] == "ERROR"],
        all_logs_df=deployment_regression_logs,
    )
    deploy_pattern = next((p for p in patterns if p["pattern_name"] == "Deployment Regression"), None)

    assert deploy_pattern is not None
    assert deploy_pattern["matched"] is True
    assert deploy_pattern["suspected_root_cause_service"] == "deployment-service"
    assert deploy_pattern["temporal_order_valid"] is True
    assert "deployment-service" in deploy_pattern["affected_services"]
    assert len(deploy_pattern["evidence"]) >= 2


# ---------------------------------------------------------------------------
# Test 3: Authentication Cascade
# ---------------------------------------------------------------------------
def test_authentication_cascade(auth_cascade_logs: pd.DataFrame):
    """Verify that an auth token verification failure propagating upstream is matched."""
    patterns = detect_known_failure_patterns(auth_cascade_logs)
    auth_pattern = next((p for p in patterns if p["pattern_name"] == "Authentication Cascade"), None)

    assert auth_pattern is not None
    assert auth_pattern["matched"] is True
    assert auth_pattern["suspected_root_cause_service"] == "auth-service"
    assert auth_pattern["temporal_order_valid"] is True
    assert "auth-service" in auth_pattern["affected_services"]


# ---------------------------------------------------------------------------
# Test 4: Temporal Event Correlation
# ---------------------------------------------------------------------------
def test_temporal_event_correlation():
    """Verify that error events within window_seconds are grouped into a single incident cluster."""
    records = [
        ("2026-10-27 10:00:00", "ERROR", "payment-service", "Payment failed"),
        ("2026-10-27 10:00:45", "ERROR", "api-gateway", "HTTP 500 error"),
        ("2026-10-27 10:01:30", "ERROR", "payment-service", "Retry limit exceeded"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    incidents = correlate_temporal_events(df, window_seconds=120)

    assert len(incidents) == 1
    assert incidents[0]["incident_id"] == 1
    assert incidents[0]["error_count"] == 3
    assert set(incidents[0]["services"]) == {"payment-service", "api-gateway"}
    assert incidents[0]["duration_seconds"] == 90.0


# ---------------------------------------------------------------------------
# Test 5: Events Separated by Hours Are Not Correlated
# ---------------------------------------------------------------------------
def test_events_separated_by_hours_not_correlated():
    """Verify that errors separated by hours form distinct, separate incidents."""
    records = [
        ("2026-10-27 08:00:00", "ERROR", "api-gateway", "HTTP 500 morning glitch"),
        ("2026-10-27 13:00:00", "ERROR", "payment-service", "Payment timeout afternoon"),
        ("2026-10-27 20:00:00", "ERROR", "auth-service", "Token expired night"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    incidents = correlate_temporal_events(df, window_seconds=120)

    assert len(incidents) == 3
    assert [inc["error_count"] for inc in incidents] == [1, 1, 1]


# ---------------------------------------------------------------------------
# Test 6: Patient-Zero Candidate Identification
# ---------------------------------------------------------------------------
def test_patient_zero_candidate_identification():
    """Verify patient-zero identification selects the earliest error and notes ambiguities."""
    records = [
        ("2026-10-27 10:05:00", "ERROR", "order-service", "Order failed"),
        ("2026-10-27 10:00:00", "ERROR", "database-proxy", "Initial database connection pool drop"),
        ("2026-10-27 10:01:00", "ERROR", "payment-service", "Downstream payment timeout"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    pz = identify_patient_zero(df)

    assert pz["candidate_service"] == "database-proxy"
    assert pz["first_seen"] == pd.Timestamp("2026-10-27 10:00:00")
    assert pz["is_ambiguous"] is False
    assert "database-proxy" in pz["explanation"]
    assert "Caution" in pz["uncertainty_warning"]


def test_patient_zero_ambiguity_when_multiple_services_fail_simultaneously():
    """Verify ambiguity is flagged when different services log errors within 1s of incident start."""
    records = [
        ("2026-10-27 10:00:00", "ERROR", "service-a", "Service A crashed"),
        ("2026-10-27 10:00:00", "ERROR", "service-b", "Service B crashed"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    pz = identify_patient_zero(df)

    assert pz["is_ambiguous"] is True
    assert "Timestamp Ambiguity" in pz["uncertainty_warning"]


# ---------------------------------------------------------------------------
# Test 7: Confidence Score Boundaries
# ---------------------------------------------------------------------------
def test_confidence_score_boundaries():
    """Verify confidence scores stay strictly bounded between 5.0 and 95.0."""
    # Test maximum possible evidence
    max_evidence = {
        "is_patient_zero": True,
        "pattern_matched": True,
        "pattern_name": "Database Exhaustion Cascade",
        "affected_services_count": 5,
        "error_count_for_candidate": 10,
        "total_incident_errors": 25,
        "is_ambiguous": False,
        "missing_evidence_count": 0,
        "has_repeated_messages": True,
    }
    high_score = calculate_confidence_score("database-proxy", max_evidence)
    assert 5.0 <= high_score["score"] <= 95.0
    assert high_score["confidence_level"] == "High"
    assert high_score["classification"] == "Hypothesis (Rule-Based Heuristic)"

    # Test minimal evidence with deductions
    min_evidence = {
        "is_patient_zero": False,
        "pattern_matched": False,
        "pattern_name": None,
        "affected_services_count": 1,
        "error_count_for_candidate": 1,
        "total_incident_errors": 1,
        "is_ambiguous": True,
        "missing_evidence_count": 4,
        "has_repeated_messages": False,
    }
    low_score = calculate_confidence_score("unknown-service", min_evidence)
    assert 5.0 <= low_score["score"] <= 40.0
    assert low_score["confidence_level"] == "Low"


# ---------------------------------------------------------------------------
# Test 8: Low-Confidence Unrelated Errors
# ---------------------------------------------------------------------------
def test_low_confidence_unrelated_errors():
    """Verify an isolated, unpatterned error receives low confidence and explicit hypothesis tag."""
    records = [
        ("2026-10-27 10:00:00", "ERROR", "isolated-service", "Random transient socket reset"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    result = run_rule_based_rca(df)

    assert result["status"] == "completed"
    assert len(result["ranked_causes"]) == 1
    cause = result["ranked_causes"][0]
    assert cause["confidence_score"] < 45.0
    assert cause["confidence_level"] == "Low"
    assert cause["classification"] == "Hypothesis (Rule-Based Heuristic)"


# ---------------------------------------------------------------------------
# Test 9: Zero-Error Dataset
# ---------------------------------------------------------------------------
def test_zero_error_dataset():
    """Verify dataset with only INFO and WARNING records returns graceful empty RCA findings."""
    records = [
        ("2026-10-27 10:00:00", "INFO", "api-gateway", "GET /api/products 200 OK"),
        ("2026-10-27 10:01:00", "WARNING", "database-proxy", "Slow query took 2100 ms"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    result = run_rule_based_rca(df)

    assert result["status"] == "completed"
    assert result["total_errors"] == 0
    assert result["incident_count"] == 0
    assert result["ranked_causes"] == []
    assert any("No ERROR-level records" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Test 10: Empty DataFrame
# ---------------------------------------------------------------------------
def test_empty_dataframe():
    """Verify completely empty DataFrames are handled gracefully across all RCA functions."""
    empty_df = pd.DataFrame()

    incidents = correlate_temporal_events(empty_df)
    assert incidents == []

    patterns = detect_known_failure_patterns(empty_df)
    assert len(patterns) == 3
    assert all(not p["matched"] for p in patterns)

    pz = identify_patient_zero(empty_df)
    assert pz["candidate_service"] is None

    result = run_rule_based_rca(empty_df)
    assert result["status"] == "completed"
    assert result["total_errors"] == 0


# ---------------------------------------------------------------------------
# Test 11: Missing Required Columns
# ---------------------------------------------------------------------------
def test_missing_required_columns():
    """Verify DataFrames missing required columns return invalid_input status with warnings."""
    bad_df = pd.DataFrame({"random_col": [1, 2, 3]})
    result = run_rule_based_rca(bad_df)

    assert result["status"] == "invalid_input"
    assert result["total_errors"] == 0
    assert any("Missing required log columns" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Test 12: Malformed Timestamps
# ---------------------------------------------------------------------------
def test_malformed_timestamps():
    """Verify that unparseable timestamps are coerced and filtered with warnings, without crashing."""
    records = [
        ("CORRUPT_TIMESTAMP_STRING", "ERROR", "payment-service", "Payment failed"),
        ("2026-10-27 10:05:00", "ERROR", "order-service", "Order failed"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    result = run_rule_based_rca(df)

    assert result["status"] == "completed"
    assert result["total_errors"] == 1
    assert any("Ignored 1 record(s) with invalid" in w for w in result["warnings"])
    assert result["ranked_causes"][0]["candidate_service"] == "order-service"


# ---------------------------------------------------------------------------
# Test 13: Multiple Independent Incidents
# ---------------------------------------------------------------------------
def test_multiple_independent_incidents():
    """Verify that multiple incident clusters separated by time are tracked accurately."""
    records = [
        # Incident 1 (morning)
        ("2026-10-27 09:00:00", "ERROR", "database-proxy", "Connection timeout"),
        ("2026-10-27 09:00:30", "ERROR", "payment-service", "Database error"),
        # Incident 2 (afternoon, 6 hours later)
        ("2026-10-27 15:00:00", "ERROR", "auth-service", "Token validation failure"),
        ("2026-10-27 15:00:45", "ERROR", "api-gateway", "401 Unauthorized downstream"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    result = run_rule_based_rca(df)

    assert result["incident_count"] == 2
    assert len(result["correlated_events"]) == 2
    assert result["correlated_events"][0]["incident_id"] == 1
    assert result["correlated_events"][1]["incident_id"] == 2


# ---------------------------------------------------------------------------
# Test 14: Missing Evidence Handling
# ---------------------------------------------------------------------------
def test_missing_evidence_handling():
    """Verify that patterns honestly report missing evidence when only partial signals exist."""
    # Only database-proxy error, without any downstream payment or gateway errors
    records = [
        ("2026-10-27 10:00:00", "ERROR", "database-proxy", "Database connection timeout after 30 seconds"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    patterns = detect_known_failure_patterns(df)
    db_pattern = next(p for p in patterns if p["pattern_name"] == "Database Exhaustion Cascade")

    assert db_pattern["matched"] is False
    assert len(db_pattern["missing_evidence"]) >= 1
    assert any("downstream service" in m.lower() for m in db_pattern["missing_evidence"])


# ---------------------------------------------------------------------------
# Test 15: Input DataFrame Remains Unchanged
# ---------------------------------------------------------------------------
def test_input_dataframe_remains_unchanged(db_exhaustion_logs: pd.DataFrame):
    """Verify that calling RCA functions causes zero mutation to caller's original DataFrame."""
    original_copy = db_exhaustion_logs.copy(deep=True)

    # Invoke all engine functions
    _ = correlate_temporal_events(db_exhaustion_logs)
    _ = detect_known_failure_patterns(db_exhaustion_logs)
    _ = identify_patient_zero(db_exhaustion_logs)
    _ = run_rule_based_rca(db_exhaustion_logs)

    # Verify identical structure and contents
    pd.testing.assert_frame_equal(db_exhaustion_logs, original_copy)

