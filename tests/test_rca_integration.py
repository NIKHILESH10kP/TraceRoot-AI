"""Unit tests for TraceRoot AI RCA integration helpers, caching logic, and audit remediations.
(Milestone 4 - Tasks 4.2 & 4.2.2: Dashboard RCA Integration & Caching Remediation)

Verifies:
1. Pure execution helper calls run_rule_based_rca safely without modifying inputs.
2. Handles empty and zero-error datasets predictably.
3. Deterministic dataset fingerprint accounts for content, timestamp, severity,
   service, row sequence, schema, and data source mode.
4. Reliable caching prevents redundant execution and strictly isolates modes.
5. Remediations:
   - Cache isolation: explicit cache priority, lazy session state init, ephemeral standalone cache.
   - Exception safety: unexpected engine exceptions return status='error' without leaking canaries.
   - Unambiguous canonical serialization: prevents delimiter collisions in unhashable object fallback.
"""

from unittest import mock
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import streamlit as st

from app import (
    _canonical_serialize_value,
    _safe_float,
    _safe_int,
    clear_rca_cache,
    compute_dataset_fingerprint,
    filter_logs,
    get_or_compute_rca,
    load_sample_logs,
    show_root_cause_analysis_section,
)
from rca_engine import run_rule_based_rca


@pytest.fixture
def sample_logs() -> pd.DataFrame:
    """Fixture providing the standard 24 simulated log records."""
    return load_sample_logs()


@pytest.fixture
def empty_logs() -> pd.DataFrame:
    """Fixture providing an empty DataFrame with standard schema."""
    return pd.DataFrame(columns=["timestamp", "level", "service", "message"])


@pytest.fixture
def isolated_cache() -> dict:
    """Fixture providing an isolated dictionary for cache testing."""
    return {}


# ---------------------------------------------------------------------------
# Test 1: RCA Execution on Baseline Sample Logs
# ---------------------------------------------------------------------------
def test_rca_execution_on_sample_logs(sample_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that get_or_compute_rca runs cleanly on sample logs and populates metadata."""
    result = get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)

    assert result["status"] == "completed"
    assert result["incident_count"] == 4
    assert result["total_errors"] == 7
    assert len(result["ranked_causes"]) > 0
    assert result["data_source_mode"] == "simulated"
    assert result["fingerprint"] == compute_dataset_fingerprint(sample_logs, mode="simulated")

    # Top candidate should be database-proxy with high confidence
    top_cause = result["ranked_causes"][0]
    assert top_cause["candidate_service"] == "database-proxy"
    assert top_cause["confidence_score"] >= 85.0
    assert "Database Exhaustion Cascade" in top_cause["possible_root_cause"]


# ---------------------------------------------------------------------------
# Test 2: RCA Execution on Empty DataFrame
# ---------------------------------------------------------------------------
def test_rca_execution_on_empty_dataframe(empty_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that an empty DataFrame is handled safely without raising exceptions."""
    result = get_or_compute_rca(empty_logs, mode="simulated", cache=isolated_cache)

    assert result["status"] == "completed"
    assert result["incident_count"] == 0
    assert result["total_errors"] == 0
    assert result["ranked_causes"] == []
    assert result["correlated_events"] == []
    assert any("empty" in w.lower() for w in result["warnings"])
    assert result["fingerprint"] == compute_dataset_fingerprint(empty_logs, mode="simulated")


# ---------------------------------------------------------------------------
# Test 3: RCA Execution on Dataset with No ERROR Records
# ---------------------------------------------------------------------------
def test_rca_execution_on_no_error_records(isolated_cache: dict):
    """Verify that a dataset containing only INFO and WARNING records completes safely."""
    records = [
        ("2026-10-27 09:00:00", "INFO", "api-gateway", "GET /api/products 200 OK"),
        ("2026-10-27 09:01:00", "WARNING", "database-proxy", "Connection pool at 80%"),
        ("2026-10-27 09:02:00", "INFO", "payment-service", "Payment processed"),
    ]
    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    result = get_or_compute_rca(df, mode="simulated", cache=isolated_cache)

    assert result["status"] == "completed"
    assert result["incident_count"] == 0
    assert result["total_errors"] == 0
    assert result["ranked_causes"] == []
    assert any("No ERROR-level records" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Test 4: Cache Reuse for Identical Datasets
# ---------------------------------------------------------------------------
def test_rca_cache_reuse_for_identical_datasets(sample_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that identical datasets in the same mode hit the cache without recomputing."""
    with mock.patch("app.run_rule_based_rca", wraps=run_rule_based_rca) as mock_engine:
        res1 = get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 1

        # Second call with identical data and mode
        res2 = get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 1  # Recomputed 0 additional times
        assert res1 == res2


# ---------------------------------------------------------------------------
# Test 5: Cache Invalidation when a Message Changes
# ---------------------------------------------------------------------------
def test_rca_cache_invalidation_when_message_changes(sample_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that altering a log message alters the fingerprint and invalidates cache."""
    with mock.patch("app.run_rule_based_rca", wraps=run_rule_based_rca) as mock_engine:
        get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 1

        modified_df = sample_logs.copy()
        modified_df.loc[0, "message"] = "GET /api/products returned 500 error modified message"

        assert compute_dataset_fingerprint(sample_logs) != compute_dataset_fingerprint(modified_df)

        get_or_compute_rca(modified_df, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 2


# ---------------------------------------------------------------------------
# Test 6: Cache Invalidation when Severity Changes
# ---------------------------------------------------------------------------
def test_rca_cache_invalidation_when_severity_changes(sample_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that altering a log severity level invalidates the cache."""
    with mock.patch("app.run_rule_based_rca", wraps=run_rule_based_rca) as mock_engine:
        get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 1

        modified_df = sample_logs.copy()
        err_idx = modified_df[modified_df["level"] == "ERROR"].index[0]
        modified_df.loc[err_idx, "level"] = "WARNING"

        assert compute_dataset_fingerprint(sample_logs) != compute_dataset_fingerprint(modified_df)

        get_or_compute_rca(modified_df, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 2


# ---------------------------------------------------------------------------
# Test 7: Cache Invalidation when Timestamp Changes
# ---------------------------------------------------------------------------
def test_rca_cache_invalidation_when_timestamp_changes(sample_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that shifting an event timestamp invalidates the cache."""
    with mock.patch("app.run_rule_based_rca", wraps=run_rule_based_rca) as mock_engine:
        get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 1

        modified_df = sample_logs.copy()
        modified_df.loc[0, "timestamp"] += pd.Timedelta(seconds=1)

        assert compute_dataset_fingerprint(sample_logs) != compute_dataset_fingerprint(modified_df)

        get_or_compute_rca(modified_df, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 2


# ---------------------------------------------------------------------------
# Test 8: Cache Invalidation when Service Name Changes
# ---------------------------------------------------------------------------
def test_rca_cache_invalidation_when_service_changes(sample_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that changing a service name invalidates the cache."""
    with mock.patch("app.run_rule_based_rca", wraps=run_rule_based_rca) as mock_engine:
        get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 1

        modified_df = sample_logs.copy()
        modified_df.loc[0, "service"] = "custom-mock-gateway"

        assert compute_dataset_fingerprint(sample_logs) != compute_dataset_fingerprint(modified_df)

        get_or_compute_rca(modified_df, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 2


# ---------------------------------------------------------------------------
# Test 9: Cache Invalidation when Row Order Changes
# ---------------------------------------------------------------------------
def test_rca_cache_invalidation_when_row_order_changes(sample_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that swapping row sequence invalidates the cache."""
    with mock.patch("app.run_rule_based_rca", wraps=run_rule_based_rca) as mock_engine:
        get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 1

        swapped_df = sample_logs.copy()
        r0 = swapped_df.iloc[0].copy()
        swapped_df.iloc[0] = swapped_df.iloc[1]
        swapped_df.iloc[1] = r0

        assert compute_dataset_fingerprint(sample_logs) != compute_dataset_fingerprint(swapped_df)

        get_or_compute_rca(swapped_df, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 2


# ---------------------------------------------------------------------------
# Test 10: Cache Separation Between Simulated and Uploaded Modes
# ---------------------------------------------------------------------------
def test_rca_cache_separation_between_simulated_and_uploaded_modes(
    sample_logs: pd.DataFrame, isolated_cache: dict
):
    """Verify that simulated and uploaded modes maintain distinct fingerprints and cache entries."""
    fp_sim = compute_dataset_fingerprint(sample_logs, mode="simulated")
    fp_up = compute_dataset_fingerprint(sample_logs, mode="uploaded")

    assert fp_sim != fp_up

    with mock.patch("app.run_rule_based_rca", wraps=run_rule_based_rca) as mock_engine:
        res_sim = get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)
        assert mock_engine.call_count == 1
        assert res_sim["data_source_mode"] == "simulated"

        res_up = get_or_compute_rca(sample_logs, mode="uploaded", cache=isolated_cache)
        assert mock_engine.call_count == 2
        assert res_up["data_source_mode"] == "uploaded"

        assert len(isolated_cache) == 2
        assert fp_sim in isolated_cache
        assert fp_up in isolated_cache


# ---------------------------------------------------------------------------
# Test 11: Input DataFrame Remains Completely Unchanged
# ---------------------------------------------------------------------------
def test_rca_input_dataframe_remains_unchanged(sample_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that calling get_or_compute_rca never mutates caller DataFrame."""
    df_before = sample_logs.copy(deep=True)

    get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)

    pdt.assert_frame_equal(sample_logs, df_before)


# ---------------------------------------------------------------------------
# Test 12: Invalid Input Handling
# ---------------------------------------------------------------------------
def test_rca_invalid_input_handling(isolated_cache: dict):
    """Verify that non-DataFrames or DataFrames missing required columns fail safely."""
    for invalid_val in [None, 123, "invalid_str", [1, 2, 3], {"timestamp": "2026-10-27"}]:
        result = get_or_compute_rca(invalid_val, mode="simulated", cache=isolated_cache)
        assert result["status"] == "invalid_input"
        assert result["total_errors"] == 0
        assert result["incident_count"] == 0
        assert result["ranked_causes"] == []
        assert any("not a valid pandas DataFrame" in w for w in result["warnings"])

    bad_df = pd.DataFrame({"unrelated_col": [1, 2, 3]})
    bad_result = get_or_compute_rca(bad_df, mode="simulated", cache=isolated_cache)
    assert bad_result["status"] == "invalid_input"
    assert any("Missing required log columns" in w for w in bad_result["warnings"])


# ---------------------------------------------------------------------------
# Test 13: Cache Returns Defensive Copy (Mutation Defense)
# ---------------------------------------------------------------------------
def test_rca_cache_returns_defensive_copy(sample_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that mutating the returned dictionary does not mutate the cached item."""
    res1 = get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)
    original_causes_count = len(res1["ranked_causes"])

    res1["ranked_causes"].clear()
    res1["status"] = "mutated_status"

    res2 = get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)
    assert res2["status"] == "completed"
    assert len(res2["ranked_causes"]) == original_causes_count


# ---------------------------------------------------------------------------
# Test 14: Clear Cache Functionality
# ---------------------------------------------------------------------------
def test_clear_rca_cache(sample_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that clear_rca_cache clears caller cache."""
    get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)
    assert len(isolated_cache) == 1

    clear_rca_cache(cache=isolated_cache)
    assert len(isolated_cache) == 0


# ---------------------------------------------------------------------------
# Test 15: Schema Sensitivity in Fingerprint
# ---------------------------------------------------------------------------
def test_compute_dataset_fingerprint_schema_sensitivity():
    """Verify that changing column names or column ordering changes the fingerprint."""
    df1 = pd.DataFrame([{"a": 1, "b": "x"}])
    df2 = pd.DataFrame([{"b": "x", "a": 1}])
    df3 = pd.DataFrame([{"c": 1, "d": "x"}])

    fp1 = compute_dataset_fingerprint(df1)
    fp2 = compute_dataset_fingerprint(df2)
    fp3 = compute_dataset_fingerprint(df3)

    assert fp1 != fp2
    assert fp1 != fp3


# ---------------------------------------------------------------------------
# Test 16: Fingerprint Fallback on Unhashable Cell Objects
# ---------------------------------------------------------------------------
def test_compute_dataset_fingerprint_unhashable_type_fallback():
    """Verify that a DataFrame containing unhashable objects (like lists) does not raise an exception."""
    df_with_list = pd.DataFrame([{"service": "svc", "details": [1, 2, 3]}])
    fp = compute_dataset_fingerprint(df_with_list)

    assert isinstance(fp, str)
    assert len(fp) == 64


# ===========================================================================
# Task 4.2.2 Remediation Tests (TR-INT-01, TR-INT-02, TR-INT-03)
# ===========================================================================

# ---------------------------------------------------------------------------
# Remediation 1: Explicit Cache Takes Priority Over Streamlit Session State
# ---------------------------------------------------------------------------
def test_explicit_cache_takes_priority_over_session_state(sample_logs: pd.DataFrame):
    """Verify that providing an explicit dictionary cache takes precedence over Streamlit session state."""
    explicit_cache = {}
    mock_session_state = {"rca_cache": {}}

    with mock.patch("streamlit.runtime.exists", return_value=True), \
         mock.patch.object(st, "session_state", mock_session_state, create=True):
        res = get_or_compute_rca(sample_logs, mode="simulated", cache=explicit_cache)

        assert res["status"] == "completed"
        # Explicit cache must be populated
        assert len(explicit_cache) == 1
        assert res["fingerprint"] in explicit_cache
        # Session state must NOT be touched when explicit cache is provided
        assert len(mock_session_state["rca_cache"]) == 0


# ---------------------------------------------------------------------------
# Remediation 2: Streamlit Session Cache is Lazily Initialized
# ---------------------------------------------------------------------------
def test_streamlit_session_cache_lazily_initialized(sample_logs: pd.DataFrame):
    """Verify that in an active Streamlit runtime, session cache is created lazily if missing."""
    mock_session_state = {}  # "rca_cache" not present

    with mock.patch("streamlit.runtime.exists", return_value=True), \
         mock.patch.object(st, "session_state", mock_session_state, create=True):
        res = get_or_compute_rca(sample_logs, mode="simulated", cache=None)

        assert res["status"] == "completed"
        assert "rca_cache" in mock_session_state
        assert len(mock_session_state["rca_cache"]) == 1
        assert res["fingerprint"] in mock_session_state["rca_cache"]


# ---------------------------------------------------------------------------
# Remediation 3: Standalone Execution Does Not Use Shared Global Cache
# ---------------------------------------------------------------------------
def test_standalone_execution_does_not_use_shared_global_cache(sample_logs: pd.DataFrame):
    """Verify that standalone execution (outside Streamlit) does not retain a persistent global cache."""
    with mock.patch("streamlit.runtime.exists", return_value=False), \
         mock.patch("app.run_rule_based_rca", wraps=run_rule_based_rca) as mock_engine:
        # First call without cache
        res1 = get_or_compute_rca(sample_logs, mode="simulated", cache=None)
        assert mock_engine.call_count == 1
        assert res1["status"] == "completed"

        # Second call without cache: must recompute because each call uses an ephemeral local cache
        res2 = get_or_compute_rca(sample_logs, mode="simulated", cache=None)
        assert mock_engine.call_count == 2
        assert res2["status"] == "completed"


# ---------------------------------------------------------------------------
# Remediation 4: Engine Exceptions Return status="error"
# ---------------------------------------------------------------------------
def test_engine_exception_returns_error_status(sample_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that unexpected engine runtime exceptions return status='error' with empty collections."""
    with mock.patch("app.run_rule_based_rca", side_effect=RuntimeError("Unexpected clustering crash")):
        res = get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)

        assert res["status"] == "error"
        assert res["total_errors"] == 0
        assert res["incident_count"] == 0
        assert res["ranked_causes"] == []
        assert res["correlated_events"] == []
        assert res["patterns_detected"] == []
        assert res["warnings"] == ["RCA analysis could not be completed due to an internal processing error."]
        assert res["data_source_mode"] == "simulated"
        assert len(res["fingerprint"]) > 0


# ---------------------------------------------------------------------------
# Remediation 5: Exception Messages Containing Canary Strings Never Leak
# ---------------------------------------------------------------------------
def test_exception_canary_leakage_prevented(sample_logs: pd.DataFrame, isolated_cache: dict):
    """Verify that exception strings, tracebacks, and sensitive canaries never leak into outputs."""
    canary = "SENSITIVE_CANARY_JWT_TOKEN_ABC123_XYZ"
    filepath_canary = "C:\\secret\\internal\\database_config.yaml"

    with mock.patch(
        "app.run_rule_based_rca",
        side_effect=ValueError(f"Failed opening {filepath_canary} with key {canary}"),
    ):
        res = get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)

        serialized_result = str(res)
        assert canary not in serialized_result
        assert filepath_canary not in serialized_result
        assert "ValueError" not in serialized_result
        assert "Failed opening" not in serialized_result
        assert res["warnings"] == ["RCA analysis could not be completed due to an internal processing error."]


# ---------------------------------------------------------------------------
# Remediation 6: Error Results are Distinguishable from Zero-Incident Results
# ---------------------------------------------------------------------------
def test_error_results_distinguishable_from_zero_incident_results(
    sample_logs: pd.DataFrame, empty_logs: pd.DataFrame, isolated_cache: dict
):
    """Verify that an error result is distinct from a successful zero-incident result."""
    # Successful zero-incident result
    zero_inc_res = get_or_compute_rca(empty_logs, mode="simulated", cache=isolated_cache)
    assert zero_inc_res["status"] == "completed"
    assert zero_inc_res["incident_count"] == 0

    # Error result from engine exception
    with mock.patch("app.run_rule_based_rca", side_effect=Exception("Fatal failure")):
        error_res = get_or_compute_rca(sample_logs, mode="simulated", cache=isolated_cache)
        assert error_res["status"] == "error"
        assert error_res["incident_count"] == 0

    # Must be clearly distinguishable by status
    assert zero_inc_res["status"] != error_res["status"]


# ---------------------------------------------------------------------------
# Remediation 7: Special Characters in Unhashable Fallback Do Not Cause Collisions
# ---------------------------------------------------------------------------
def test_unhashable_fallback_special_chars_no_collision():
    """Verify that delimiters (| ; : newline quotes) in unhashable values do not collide."""
    # Case 1: Pipe character in lists
    df1 = pd.DataFrame([{"col1": ["a|b"], "col2": ["c"]}])
    df2 = pd.DataFrame([{"col1": ["a"], "col2": ["b|c"]}])
    assert compute_dataset_fingerprint(df1) != compute_dataset_fingerprint(df2)

    # Case 2: Semicolon delimiter in nested strings
    df3 = pd.DataFrame([{"col1": ["a;b;"], "col2": ["c"]}])
    df4 = pd.DataFrame([{"col1": ["a"], "col2": [";b;c"]}])
    assert compute_dataset_fingerprint(df3) != compute_dataset_fingerprint(df4)

    # Case 3: Colon delimiter in dict keys/values
    df5 = pd.DataFrame([{"data": [{"k:1": "v"}]}])
    df6 = pd.DataFrame([{"data": [{"k": "1:v"}]}])
    assert compute_dataset_fingerprint(df5) != compute_dataset_fingerprint(df6)

    # Case 4: Newline character
    df7 = pd.DataFrame([{"data": ["line1\nline2"]}])
    df8 = pd.DataFrame([{"data": ["line1", "line2"]}])
    assert compute_dataset_fingerprint(df7) != compute_dataset_fingerprint(df8)


# ---------------------------------------------------------------------------
# Remediation 8: Repeated Equivalent Unhashable Datasets Produce Same Fingerprint
# ---------------------------------------------------------------------------
def test_unhashable_fallback_deterministic_reproducibility():
    """Verify that dictionaries with different key insertion orders produce identical fingerprints."""
    df_a = pd.DataFrame([{"service": "svc", "details": [{"z": 10, "a": 1, "m": 5}]}])
    df_b = pd.DataFrame([{"service": "svc", "details": [{"a": 1, "m": 5, "z": 10}]}])

    fp_a = compute_dataset_fingerprint(df_a)
    fp_b = compute_dataset_fingerprint(df_b)
    assert fp_a == fp_b


# ---------------------------------------------------------------------------
# Remediation 9: Cache Clearing Works with Revised Architecture
# ---------------------------------------------------------------------------
def test_clear_rca_cache_revised_architecture(sample_logs: pd.DataFrame):
    """Verify that clear_rca_cache clears caller dictionary and session state."""
    caller_cache = {}
    mock_session_state = {"rca_cache": {"key": "cached_val"}}

    with mock.patch("streamlit.runtime.exists", return_value=True), \
         mock.patch.object(st, "session_state", mock_session_state, create=True):
        get_or_compute_rca(sample_logs, mode="simulated", cache=caller_cache)
        assert len(caller_cache) == 1

        clear_rca_cache(cache=caller_cache)
        assert len(caller_cache) == 0
        assert len(mock_session_state["rca_cache"]) == 0


# ---------------------------------------------------------------------------
# Remediation 10: Non-Deterministic Object Raises Controlled Error
# ---------------------------------------------------------------------------
def test_nondeterministic_object_raises_controlled_error():
    """Verify that custom unhandled objects cannot produce unreliable fingerprints."""
    df_with_custom = pd.DataFrame([{"service": "svc", "custom_obj": [object()]}])

    with pytest.raises(ValueError, match="cannot be deterministically serialized"):
        compute_dataset_fingerprint(df_with_custom)

    # Calling get_or_compute_rca on such a dataset fails safely without crashing
    res = get_or_compute_rca(df_with_custom, mode="simulated")
    assert res["status"] == "error"
    assert res["warnings"] == ["RCA analysis could not be completed due to an internal processing error."]


# ===========================================================================
# Task 4.3 Tests: RCA Dashboard UI Integration & Final Remediations
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 4.3 Test 1: UI Renders Successfully for Dataset with Incidents
# ---------------------------------------------------------------------------
def test_show_root_cause_analysis_section_with_incidents(sample_logs: pd.DataFrame):
    """Verify that show_root_cause_analysis_section renders metrics, clusters, and hypotheses."""
    col_mock = mock.MagicMock()
    with mock.patch.object(st, "columns", side_effect=lambda spec, **kwargs: [col_mock] * (spec if isinstance(spec, int) else len(spec))), \
         mock.patch.object(st, "markdown") as mock_markdown, \
         mock.patch.object(st, "progress") as mock_progress:
        show_root_cause_analysis_section(sample_logs, mode="simulated")

        # Metric calls must include Analysis Status, Errors Analyzed, Incident Clusters, etc.
        metric_labels = [call.args[0] for call in col_mock.metric.call_args_list if call.args]
        assert "Analysis Status" in metric_labels
        assert "Errors Analyzed" in metric_labels
        assert "Incident Clusters" in metric_labels
        assert "Ranked Hypotheses" in metric_labels

        # Progress bar must be drawn for candidate confidence score
        assert mock_progress.called


# ---------------------------------------------------------------------------
# Task 4.3 Test 2: Correct Handling of Zero Incidents in UI
# ---------------------------------------------------------------------------
def test_show_root_cause_analysis_section_zero_incidents(empty_logs: pd.DataFrame):
    """Verify that a zero-error dataset displays the clean green success notification."""
    with mock.patch.object(st, "success") as mock_success, \
         mock.patch.object(st, "metric") as mock_metric:
        show_root_cause_analysis_section(empty_logs, mode="simulated")

        assert mock_success.called
        success_msg = mock_success.call_args[0][0]
        assert "No Failure Incidents Detected" in success_msg
        assert "No multi-service failure cascades or correlated incident clusters were detected in the active log dataset." in success_msg
        assert "operating normally" not in success_msg.lower()


# ---------------------------------------------------------------------------
# Task 4.3 Test 3: Correct Handling of Invalid Input in UI
# ---------------------------------------------------------------------------
def test_show_root_cause_analysis_section_invalid_input():
    """Verify that an invalid DataFrame missing required schema columns triggers warning alert."""
    bad_df = pd.DataFrame({"random_col": [1, 2, 3]})
    with mock.patch.object(st, "warning") as mock_warning:
        show_root_cause_analysis_section(bad_df, mode="simulated")

        assert mock_warning.called
        warning_msg = mock_warning.call_args[0][0]
        assert "Invalid Log Dataset" in warning_msg


# ---------------------------------------------------------------------------
# Task 4.3 Test 4: Correct Handling of Engine Errors in UI
# ---------------------------------------------------------------------------
def test_show_root_cause_analysis_section_engine_error(sample_logs: pd.DataFrame):
    """Verify that status='error' displays an error alert and never shows zero incidents success."""
    with mock.patch("app.run_rule_based_rca", side_effect=RuntimeError("Clustering crash")), \
         mock.patch.object(st, "error") as mock_error, \
         mock.patch.object(st, "success") as mock_success:
        show_root_cause_analysis_section(sample_logs, mode="simulated")

        assert mock_error.called
        error_msg = mock_error.call_args[0][0]
        assert "Root Cause Analysis Error" in error_msg
        assert not mock_success.called


# ---------------------------------------------------------------------------
# Task 4.3 Test 5: RCA Evaluates Complete Active Dataset, Not Filtered Data
# ---------------------------------------------------------------------------
def test_rca_evaluates_active_dataset_not_filtered_data(sample_logs: pd.DataFrame):
    """Verify that RCA evaluates active_logs rather than a filtered subset, capturing cross-service cascades."""
    # Suppose user filtered sidebar to only payment-service
    filtered_subset = filter_logs(sample_logs, selected_levels=["ERROR"], selected_services=["payment-service"])

    # Evaluating filtered_subset directly would miss patient zero (database-proxy)
    res_filtered = get_or_compute_rca(filtered_subset, mode="simulated")
    top_filtered_service = res_filtered["ranked_causes"][0]["candidate_service"]
    assert top_filtered_service == "payment-service"

    # Evaluating active_logs captures the full cascade with database-proxy as primary suspect
    res_active = get_or_compute_rca(sample_logs, mode="simulated")
    top_active_service = res_active["ranked_causes"][0]["candidate_service"]
    assert top_active_service == "database-proxy"
    assert res_active["ranked_causes"][0]["confidence_score"] >= 85.0


# ---------------------------------------------------------------------------
# Task 4.3 Test 6: Simulated and Uploaded Datasets Remain Separate in UI
# ---------------------------------------------------------------------------
def test_simulated_and_uploaded_datasets_remain_separate_in_ui(sample_logs: pd.DataFrame):
    """Verify that UI calls pass mode accurately and keep dataset caches isolated."""
    with mock.patch("app.get_or_compute_rca", wraps=get_or_compute_rca) as spy_get_rca:
        show_root_cause_analysis_section(sample_logs, mode="simulated")
        assert spy_get_rca.call_args[1]["mode"] == "simulated"

        show_root_cause_analysis_section(sample_logs, mode="uploaded")
        assert spy_get_rca.call_args[1]["mode"] == "uploaded"


# ---------------------------------------------------------------------------
# Task 4.3 Test 7: UI Section Does Not Mutate Active Logs DataFrame
# ---------------------------------------------------------------------------
def test_show_root_cause_analysis_section_does_not_mutate_active_logs(sample_logs: pd.DataFrame):
    """Verify that running show_root_cause_analysis_section leaves active_logs completely untouched."""
    logs_before = sample_logs.copy(deep=True)

    show_root_cause_analysis_section(sample_logs, mode="simulated")

    pdt.assert_frame_equal(sample_logs, logs_before)


# ---------------------------------------------------------------------------
# Task 4.3 Test 8: TR-REV-01 Heterogeneous Dict Key Tie-Breaking
# ---------------------------------------------------------------------------
def test_remediation_tr_rev_01_heterogeneous_dict_keys():
    """Verify that dicts with heterogeneous keys (e.g. 1 and '1') produce identical fingerprints regardless of insertion order."""
    d1 = {1: "int_key", "1": "str_key"}
    d2 = {"1": "str_key", 1: "int_key"}

    # Canonical bytes must be identical
    b1 = _canonical_serialize_value(d1)
    b2 = _canonical_serialize_value(d2)
    assert b1 == b2

    df1 = pd.DataFrame([{"details": [d1]}])
    df2 = pd.DataFrame([{"details": [d2]}])
    assert compute_dataset_fingerprint(df1) == compute_dataset_fingerprint(df2)


# ---------------------------------------------------------------------------
# Task 4.3 Test 9: TR-REV-02 NumPy Scalar Support in Canonical Serializer
# ---------------------------------------------------------------------------
def test_remediation_tr_rev_02_numpy_scalars():
    """Verify that np.integer, np.bool_, and np.floating scalars serialize cleanly without errors."""
    b_int = _canonical_serialize_value(np.int64(42))
    assert b_int == b"I:42;"

    b_bool = _canonical_serialize_value(np.bool_(True))
    assert b_bool == b"B:1;"

    b_float = _canonical_serialize_value(np.float64(3.14))
    assert b"F:3.14" in b_float

    # Test nested inside unhashable container
    df_with_numpy = pd.DataFrame([{"details": [np.int64(10), np.bool_(False), np.float32(2.5)]}])
    fp = compute_dataset_fingerprint(df_with_numpy)
    assert isinstance(fp, str)
    assert len(fp) == 64


# ---------------------------------------------------------------------------
# Task 4.3 Test 10: TR-REV-03 Bounded Recursion Depth Check
# ---------------------------------------------------------------------------
def test_remediation_tr_rev_03_bounded_recursion_depth():
    """Verify that cyclic or deeply nested containers fail with a controlled ValueError and do not crash."""
    # Cyclic list
    cyclic = []
    cyclic.append(cyclic)

    with pytest.raises(ValueError, match="Maximum nesting depth exceeded or cyclic container detected"):
        _canonical_serialize_value(cyclic)

    # Calling get_or_compute_rca handles cyclic container safely as status='error'
    df_cyclic = pd.DataFrame([{"data": [cyclic]}])
    res = get_or_compute_rca(df_cyclic, mode="simulated")
    assert res["status"] == "error"
    assert res["warnings"] == ["RCA analysis could not be completed due to an internal processing error."]


# ---------------------------------------------------------------------------
# Task 4.3.2 Test 1: Zero-incident message does not claim services are healthy
# ---------------------------------------------------------------------------
def test_zero_incident_message_does_not_claim_services_healthy(empty_logs: pd.DataFrame):
    """Verify that zero-incident display does not claim services are healthy or operating normally."""
    with mock.patch.object(st, "success") as mock_success, \
         mock.patch.object(st, "metric"):
        show_root_cause_analysis_section(empty_logs, mode="simulated")

        assert mock_success.called
        msg = mock_success.call_args[0][0]
        assert "No Failure Incidents Detected" in msg
        assert "No multi-service failure cascades or correlated incident clusters were detected in the active log dataset." in msg
        assert "operating normally" not in msg.lower()
        assert "healthy" not in msg.lower()

    # Also test when there is an isolated error but zero incident clusters
    isolated_error_res = {
        "status": "completed",
        "total_errors": 1,
        "incident_count": 0,
        "ranked_causes": [],
        "correlated_events": [],
        "patterns_detected": [],
        "warnings": ["Single isolated error."],
    }
    with mock.patch("app.get_or_compute_rca", return_value=isolated_error_res), \
         mock.patch.object(st, "success") as mock_success, \
         mock.patch.object(st, "metric"):
        show_root_cause_analysis_section(empty_logs, mode="simulated")

        assert mock_success.called
        msg2 = mock_success.call_args[0][0]
        assert "operating normally" not in msg2.lower()
        assert "healthy" not in msg2.lower()


# ---------------------------------------------------------------------------
# Task 4.3.2 Test 2: Explicit None duration values do not crash rendering
# ---------------------------------------------------------------------------
def test_show_root_cause_analysis_section_explicit_none_duration(sample_logs: pd.DataFrame):
    """Verify that explicit None duration and error_count values fallback safely without TypeError."""
    mock_rca_res = {
        "status": "completed",
        "total_errors": 2,
        "incident_count": 1,
        "correlated_events": [{
            "incident_id": 1,
            "start_time": "2026-10-27 09:00:00",
            "end_time": "2026-10-27 09:01:00",
            "duration_seconds": None,  # Explicit None
            "error_count": None,      # Explicit None
            "services": None,         # Explicit None
            "records": [],
        }],
        "ranked_causes": [{
            "candidate_service": "database-proxy",
            "possible_root_cause": "Database connection timeout",
            "confidence_score": 75.0,
            "confidence_level": "High",
            "confidence_breakdown": {},
            "evidence_supporting": [],
            "evidence_against": [],
            "recommended_verification_steps": [],
        }],
        "patterns_detected": [],
        "warnings": [],
    }

    col_mock = mock.MagicMock()
    with mock.patch("app.get_or_compute_rca", return_value=mock_rca_res), \
         mock.patch.object(st, "columns", side_effect=lambda spec, **kwargs: [col_mock] * (spec if isinstance(spec, int) else len(spec))), \
         mock.patch.object(st, "markdown"), \
         mock.patch.object(st, "progress"):
        # Must execute without throwing TypeError
        show_root_cause_analysis_section(sample_logs, mode="simulated")

        # Caption calls on col_mock should format Duration with 0.0s fallback
        caption_calls = [c.args[0] for c in col_mock.caption.call_args_list if c.args]
        assert any("0.0s" in str(txt) for txt in caption_calls)


# ---------------------------------------------------------------------------
# Task 4.3.2 Test 3: Explicit None confidence values do not crash rendering
# ---------------------------------------------------------------------------
def test_show_root_cause_analysis_section_explicit_none_confidence(sample_logs: pd.DataFrame):
    """Verify that explicit None confidence score and breakdown points fallback safely without TypeError."""
    mock_rca_res = {
        "status": "completed",
        "total_errors": 2,
        "incident_count": 1,
        "correlated_events": [{
            "incident_id": 1,
            "start_time": "2026-10-27 09:00:00",
            "end_time": "2026-10-27 09:01:00",
            "duration_seconds": 15.2,
            "error_count": 2,
            "services": ["database-proxy"],
            "records": [],
        }],
        "ranked_causes": [{
            "candidate_service": "database-proxy",
            "possible_root_cause": "Database connection timeout",
            "confidence_score": None,  # Explicit None
            "confidence_level": None,  # Explicit None
            "confidence_breakdown": {
                "temporal_priority": {"points": None, "max_points": None, "reason": "No priority"},
                "isolated_num": None,
            },
            "evidence_supporting": None,
            "evidence_against": None,
            "recommended_verification_steps": None,
        }],
        "patterns_detected": [],
        "warnings": [],
    }

    col_mock = mock.MagicMock()
    with mock.patch("app.get_or_compute_rca", return_value=mock_rca_res), \
         mock.patch.object(st, "columns", side_effect=lambda spec, **kwargs: [col_mock] * (spec if isinstance(spec, int) else len(spec))), \
         mock.patch.object(st, "markdown") as mock_md, \
         mock.patch.object(st, "progress") as mock_prog:
        # Must execute without throwing TypeError
        show_root_cause_analysis_section(sample_logs, mode="simulated")

        # Progress bar should receive 0.0
        assert mock_prog.called
        assert mock_prog.call_args[0][0] == 0.0

        # Markdown header should format 0.0% Confidence (Low)
        md_calls = [c.args[0] for c in mock_md.call_args_list if c.args]
        assert any("0.0% Confidence (Low)" in str(txt) for txt in md_calls)


# ---------------------------------------------------------------------------
# Task 4.3.2 Test 4: Existing states (completed, invalid_input, error) remain intact
# ---------------------------------------------------------------------------
def test_existing_states_completed_invalid_and_error_remain_correct(sample_logs: pd.DataFrame):
    """Verify that completed, invalid_input, and error states continue to be routed accurately."""
    # 1. Invalid input state
    bad_df = pd.DataFrame({"unrelated": [1, 2]})
    with mock.patch.object(st, "warning") as mock_warn:
        show_root_cause_analysis_section(bad_df, mode="simulated")
        assert mock_warn.called
        assert "Invalid Log Dataset" in mock_warn.call_args[0][0]

    # 2. Error state
    with mock.patch("app.run_rule_based_rca", side_effect=RuntimeError("Internal fault")), \
         mock.patch.object(st, "error") as mock_err, \
         mock.patch.object(st, "success") as mock_succ:
        show_root_cause_analysis_section(sample_logs, mode="simulated")
        assert mock_err.called
        assert "Root Cause Analysis Error" in mock_err.call_args[0][0]
        assert not mock_succ.called

    # 3. Completed state
    col_mock = mock.MagicMock()
    with mock.patch.object(st, "columns", side_effect=lambda spec, **kwargs: [col_mock] * (spec if isinstance(spec, int) else len(spec))), \
         mock.patch.object(st, "markdown"), \
         mock.patch.object(st, "progress"):
        show_root_cause_analysis_section(sample_logs, mode="simulated")
        metric_labels = [call.args[0] for call in col_mock.metric.call_args_list if call.args]
        assert "Analysis Status" in metric_labels
        assert "Errors Analyzed" in metric_labels


# ---------------------------------------------------------------------------
# Task 4.3.4 Test 1: NaN, +inf, -inf confidence scores fall back safely to 0.0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -float("inf")])
def test_nan_and_inf_confidence_score_falls_back_safely(sample_logs: pd.DataFrame, bad_score: float):
    """Verify that non-finite confidence scores do not crash st.progress or cause StreamlitValueOutOfRangeError."""
    mock_payload = {
        "status": "completed",
        "total_errors": 2,
        "incident_count": 1,
        "correlated_events": [{
            "incident_id": 1,
            "start_time": "2026-10-27 09:00:00",
            "end_time": "2026-10-27 09:01:00",
            "duration_seconds": 10.0,
            "error_count": 2,
            "services": ["api-gateway"],
            "records": [],
        }],
        "ranked_causes": [{
            "candidate_service": "api-gateway",
            "possible_root_cause": "Timeout cascade",
            "confidence_score": bad_score,
            "confidence_level": "Low",
            "confidence_breakdown": {},
            "evidence_supporting": [],
            "evidence_against": [],
            "recommended_verification_steps": [],
        }],
        "patterns_detected": [],
        "warnings": [],
    }

    col_mock = mock.MagicMock()
    with mock.patch("app.get_or_compute_rca", return_value=mock_payload), \
         mock.patch.object(st, "columns", side_effect=lambda spec, **kwargs: [col_mock] * (spec if isinstance(spec, int) else len(spec))), \
         mock.patch.object(st, "markdown") as mock_md, \
         mock.patch.object(st, "progress") as mock_prog:
        show_root_cause_analysis_section(sample_logs, mode="simulated")

        # Progress bar must receive finite float 0.0 without throwing RangeError
        assert mock_prog.called
        assert mock_prog.call_args[0][0] == 0.0

        # Markdown header must render 0.0% Confidence (Low)
        md_calls = [c.args[0] for c in mock_md.call_args_list if c.args]
        assert any("0.0% Confidence (Low)" in str(txt) for txt in md_calls)


# ---------------------------------------------------------------------------
# Task 4.3.4 Test 2: NaN, +inf, -inf duration fall back safely to 0.0s
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_dur", [float("nan"), float("inf"), -float("inf")])
def test_nan_and_inf_duration_falls_back_safely(sample_logs: pd.DataFrame, bad_dur: float):
    """Verify that non-finite durations do not produce 'nans' or 'infs' captions."""
    mock_payload = {
        "status": "completed",
        "total_errors": 2,
        "incident_count": 1,
        "correlated_events": [{
            "incident_id": 1,
            "start_time": "2026-10-27 09:00:00",
            "end_time": "2026-10-27 09:01:00",
            "duration_seconds": bad_dur,
            "error_count": 2,
            "services": ["api-gateway"],
            "records": [],
        }],
        "ranked_causes": [{
            "candidate_service": "api-gateway",
            "possible_root_cause": "Timeout",
            "confidence_score": 50.0,
            "confidence_level": "Medium",
            "confidence_breakdown": {},
            "evidence_supporting": [],
            "evidence_against": [],
            "recommended_verification_steps": [],
        }],
        "patterns_detected": [],
        "warnings": [],
    }

    col_mock = mock.MagicMock()
    with mock.patch("app.get_or_compute_rca", return_value=mock_payload), \
         mock.patch.object(st, "columns", side_effect=lambda spec, **kwargs: [col_mock] * (spec if isinstance(spec, int) else len(spec))), \
         mock.patch.object(st, "markdown"), \
         mock.patch.object(st, "progress"):
        show_root_cause_analysis_section(sample_logs, mode="simulated")

        caption_calls = [c.args[0] for c in col_mock.caption.call_args_list if c.args]
        assert any("`0.0s`" in str(txt) for txt in caption_calls)
        assert not any("nan" in str(txt).lower() for txt in caption_calls)
        assert not any("inf" in str(txt).lower() for txt in caption_calls)


# ---------------------------------------------------------------------------
# Task 4.3.4 Test 3: Positive and negative infinity in integer counters
# ---------------------------------------------------------------------------
def test_pos_and_neg_inf_in_integer_counters_falls_back_safely(sample_logs: pd.DataFrame):
    """Verify that float('inf') and -float('inf') do not raise unhandled OverflowError in int() conversions."""
    # Test _safe_int helper directly
    assert _safe_int(float("inf")) == 0
    assert _safe_int(-float("inf")) == 0
    assert _safe_int(float("nan")) == 0

    mock_payload = {
        "status": "completed",
        "total_errors": float("inf"),
        "incident_count": -float("inf"),
        "correlated_events": [{
            "incident_id": 1,
            "start_time": "2026-10-27 09:00:00",
            "end_time": "2026-10-27 09:01:00",
            "duration_seconds": 10.0,
            "error_count": float("inf"),
            "services": ["api-gateway"],
            "records": [],
        }],
        "ranked_causes": [],
        "patterns_detected": [],
        "warnings": [],
    }

    col_mock = mock.MagicMock()
    with mock.patch("app.get_or_compute_rca", return_value=mock_payload), \
         mock.patch.object(st, "columns", side_effect=lambda spec, **kwargs: [col_mock] * (spec if isinstance(spec, int) else len(spec))), \
         mock.patch.object(st, "markdown"), \
         mock.patch.object(st, "success") as mock_succ:
        # Must execute cleanly without OverflowError
        show_root_cause_analysis_section(sample_logs, mode="simulated")

        # total_errors and incident_count fall back to 0, so zero incidents success is shown
        assert mock_succ.called


# ---------------------------------------------------------------------------
# Task 4.3.4 Test 4: NaN and infinity in breakdown points and maximum points
# ---------------------------------------------------------------------------
def test_nan_and_inf_in_breakdown_points_and_max_points(sample_logs: pd.DataFrame):
    """Verify that non-finite points and max_points fall back cleanly without formatting 'nan' or 'inf'."""
    # Test _safe_float helper directly
    assert _safe_float(float("nan"), default=0.0) == 0.0
    assert _safe_float(float("inf"), default=0.0) == 0.0
    assert _safe_float(float("nan"), default=None) is None
    assert _safe_float(float("inf"), default=None) is None

    mock_payload = {
        "status": "completed",
        "total_errors": 2,
        "incident_count": 1,
        "correlated_events": [{
            "incident_id": 1,
            "start_time": "2026-10-27 09:00:00",
            "end_time": "2026-10-27 09:01:00",
            "duration_seconds": 5.0,
            "error_count": 2,
            "services": ["api-gateway"],
            "records": [],
        }],
        "ranked_causes": [{
            "candidate_service": "api-gateway",
            "possible_root_cause": "Timeout",
            "confidence_score": 60.0,
            "confidence_level": "Medium",
            "confidence_breakdown": {
                "pattern_match": {"points": float("nan"), "max_points": float("nan"), "reason": "No pattern"},
                "cascade_impact": {"points": float("inf"), "max_points": float("inf"), "reason": "Wide cascade"},
                "scalar_nan": float("nan"),
                "scalar_inf": float("inf"),
            },
            "evidence_supporting": [],
            "evidence_against": [],
            "recommended_verification_steps": [],
        }],
        "patterns_detected": [],
        "warnings": [],
    }

    col_mock = mock.MagicMock()
    with mock.patch("app.get_or_compute_rca", return_value=mock_payload), \
         mock.patch.object(st, "columns", side_effect=lambda spec, **kwargs: [col_mock] * (spec if isinstance(spec, int) else len(spec))), \
         mock.patch.object(st, "markdown"), \
         mock.patch.object(st, "progress"):
        show_root_cause_analysis_section(sample_logs, mode="simulated")

        metric_calls = [c.args for c in col_mock.metric.call_args_list if len(c.args) >= 2]
        metric_dict = {c[0]: c[1] for c in metric_calls}

        # Dict with NaN points and max_points: points falls back to 0.0, max_points omitted (no '/ nan')
        assert metric_dict.get("Pattern Match") == "+0.0 pts"
        # Dict with Inf points and max_points: points falls back to 0.0, max_points omitted (no '/ inf')
        assert metric_dict.get("Cascade Impact") == "+0.0 pts"


# ---------------------------------------------------------------------------
# Task 4.3.4 Test 5: Boolean values in numeric fields do not become valid counts
# ---------------------------------------------------------------------------
def test_boolean_values_in_numeric_fields_rejected(sample_logs: pd.DataFrame):
    """Verify that True/False values in integer or float positions do not evaluate to 1 or 1.0."""
    assert _safe_int(True) == 0
    assert _safe_int(False) == 0
    assert _safe_int(np.bool_(True)) == 0
    assert _safe_float(True) == 0.0
    assert _safe_float(False) == 0.0
    assert _safe_float(np.bool_(True)) == 0.0
    assert _safe_float(True, default=None) is None

    mock_payload = {
        "status": "completed",
        "total_errors": True,       # Must NOT become 1
        "incident_count": False,    # Must NOT become 1
        "correlated_events": [{
            "incident_id": 1,
            "start_time": "2026-10-27 09:00:00",
            "end_time": "2026-10-27 09:01:00",
            "duration_seconds": True,   # Must NOT become 1.0s
            "error_count": True,        # Must NOT become 1
            "services": ["api-gateway"],
            "records": [],
        }],
        "ranked_causes": [{
            "candidate_service": "api-gateway",
            "possible_root_cause": "Timeout",
            "confidence_score": True,   # Must NOT become 1.0%
            "confidence_level": "Low",
            "confidence_breakdown": {
                "check": {"points": True, "max_points": False},
                "scalar_bool": True,
            },
            "evidence_supporting": [],
            "evidence_against": [],
            "recommended_verification_steps": [],
        }],
        "patterns_detected": [],
        "warnings": [],
    }

    col_mock = mock.MagicMock()
    with mock.patch("app.get_or_compute_rca", return_value=mock_payload), \
         mock.patch.object(st, "columns", side_effect=lambda spec, **kwargs: [col_mock] * (spec if isinstance(spec, int) else len(spec))), \
         mock.patch.object(st, "markdown"), \
         mock.patch.object(st, "success") as mock_succ:
        show_root_cause_analysis_section(sample_logs, mode="simulated")

        # Because total_errors and incident_count safely parse as 0, zero-incident success triggers
        assert mock_succ.called
        metric_calls = [c.args for c in col_mock.metric.call_args_list if len(c.args) >= 2]
        metric_dict = {c[0]: c[1] for c in metric_calls}
        assert metric_dict.get("Errors Analyzed") == 0
        assert metric_dict.get("Incident Clusters") == 0

