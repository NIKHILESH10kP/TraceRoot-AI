"""
End-to-End Test Suite for TraceRoot AI using Streamlit AppTest.

This test suite executes the real application execution flow in an isolated
headless test environment, validating:
- Test A: Cold startup and baseline telemetry metrics rendering.
- Test B: Simulated dataset baseline and dynamic event generation controls.
- Test C: Uploaded mode empty state and prevention of simulated data leakage.
- Test D: Operational error-spike alert detection independence from visual display filters.
- Test E: Dataset mode isolation between simulated and uploaded session states.
- Test F: Empty dataset status messaging vs non-empty no-spike messaging accuracy.
- Test G: RCA dashboard integration evaluating the full active dataset despite visual filters.
"""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
DEFAULT_APPTEST_TIMEOUT = 15.0


def create_app() -> AppTest:
    """Instantiate and run an isolated AppTest instance for app.py."""
    at = AppTest.from_file(APP_PATH, default_timeout=DEFAULT_APPTEST_TIMEOUT)
    return at.run()


def _get_metric_map(at: AppTest) -> Dict[str, str]:
    """Extract metric label -> value mappings from the rendered AppTest tree."""
    return {m.label: str(m.value) for m in at.metric}


# ===========================================================================
# Test A: Cold Startup
# ===========================================================================
def test_cold_startup():
    """Verify that TraceRoot AI starts cleanly without preloaded state."""
    at = create_app()

    # 1. Assert no Streamlit exceptions occurred during startup
    assert not at.exception, f"Unexpected exception on cold startup: {at.exception}"

    # 2. Assert application title is rendered
    assert len(at.title) >= 1, "Expected at least one Title element"
    assert at.title[0].value == "TraceRoot AI", f"Unexpected title: {at.title[0].value}"

    # 3. Assert baseline headline summary metrics are rendered
    metrics = _get_metric_map(at)
    assert metrics.get("Total log records") == "24", f"Unexpected total records: {metrics}"
    assert metrics.get("ERROR records") == "7", f"Unexpected ERROR records: {metrics}"
    assert metrics.get("WARNING records") == "7", f"Unexpected WARNING records: {metrics}"
    assert metrics.get("Unique services") == "6", f"Unexpected Unique services: {metrics}"

    # 4. Assert default data source mode is simulated
    radio = at.sidebar.radio[0]
    assert radio.value == "Simulated Demonstration Logs"


# ===========================================================================
# Test B: Simulated Dataset Controls
# ===========================================================================
def test_simulated_dataset_controls():
    """Verify simulated mode baseline count and simulation button interactions."""
    at = create_app()
    assert not at.exception

    # Verify baseline count is 24
    metrics_initial = _get_metric_map(at)
    assert metrics_initial.get("Total log records") == "24"

    # Click "+ Generate 5 Events" button
    gen_btn = next((b for b in at.sidebar.button if "Generate 5 Events" in b.label), None)
    assert gen_btn is not None, "Generate 5 Events button not found in sidebar"
    gen_btn.click().run()

    assert not at.exception, f"Exception after generating events: {at.exception}"
    metrics_after_gen = _get_metric_map(at)
    assert metrics_after_gen.get("Total log records") == "29", (
        f"Expected 29 records after generating 5 events, got {metrics_after_gen.get('Total log records')}"
    )

    # Click "Reset Baseline" button
    reset_btn = next((b for b in at.sidebar.button if "Reset Baseline" in b.label), None)
    assert reset_btn is not None, "Reset Baseline button not found in sidebar"
    reset_btn.click().run()

    assert not at.exception, f"Exception after resetting baseline: {at.exception}"
    metrics_after_reset = _get_metric_map(at)
    assert metrics_after_reset.get("Total log records") == "24", (
        f"Expected 24 records after reset, got {metrics_after_reset.get('Total log records')}"
    )


# ===========================================================================
# Test C: Uploaded Mode Empty State
# ===========================================================================
def test_uploaded_mode_empty_state():
    """Verify uploaded mode empty state without a file and assert no simulated data leakage."""
    at = create_app()
    assert not at.exception

    # Switch radio to Uploaded Log File
    at.sidebar.radio[0].set_value("Uploaded Log File").run()
    assert not at.exception, f"Exception when switching to uploaded mode: {at.exception}"

    # Verify informational empty state banner
    info_texts = [info.value for info in at.info]
    assert any("No file uploaded yet" in text for text in info_texts), (
        f"Expected 'No file uploaded yet' banner in at.info: {info_texts}"
    )
    assert any("No log records available" in text for text in info_texts), (
        f"Expected 'No log records available' banner in at.info: {info_texts}"
    )

    # Assert simulated records do not leak into uploaded mode
    metrics = _get_metric_map(at)
    assert metrics.get("Total log records") == "0", (
        f"Simulated records leaked into uploaded mode: {metrics.get('Total log records')}"
    )
    assert metrics.get("ERROR records") == "0"
    assert metrics.get("WARNING records") == "0"
    assert metrics.get("Unique services") == "0"

    # Assert no false success spike alert is rendered
    spike_success = [s.value for s in at.success if "error spikes" in s.value.lower()]
    assert len(spike_success) == 0, f"False positive spike success alert found on empty dataset: {spike_success}"


# ===========================================================================
# Test D: Operational Alert Independent of Visual Filters
# ===========================================================================
def test_operational_alert_independent_of_visual_filters():
    """Verify that operational error-spike detection evaluates active_logs despite filter exclusion."""
    at = create_app()
    assert not at.exception

    # 1. Trigger the payment-service error spike via UI button
    spike_btn = next((b for b in at.sidebar.button if "Trigger Spike" in b.label), None)
    assert spike_btn is not None, "Trigger Spike button not found in sidebar"
    spike_btn.click().run()
    assert not at.exception, f"Exception after triggering spike: {at.exception}"

    # Verify error spike banner appears
    error_alerts_before = [e.value for e in at.error]
    assert any("Service Error Spike(s) Detected" in e for e in error_alerts_before), (
        f"Expected spike alert in at.error: {error_alerts_before}"
    )

    # 2. Exclude 'ERROR' from the Log level multiselect visual filter
    level_ms = next((ms for ms in at.sidebar.multiselect if ms.label == "Log level"), None)
    assert level_ms is not None, "Log level multiselect not found in sidebar"
    level_ms.unselect("ERROR").run()
    assert not at.exception, f"Exception after unselecting ERROR filter: {at.exception}"

    # 3. Confirm visible filtered log table excludes ERROR entries
    log_table = next(
        (df for df in at.dataframe if "level" in df.value.columns and "timestamp" in df.value.columns),
        None,
    )
    assert log_table is not None, "Filtered log records table not found in rendered output"
    assert len(log_table.value) > 0, "Filtered log table should contain INFO and WARNING rows"
    assert "ERROR" not in log_table.value["level"].values, (
        f"Filtered log table still contains ERROR entries: {log_table.value['level'].unique()}"
    )

    # 4. Confirm operational error-spike alert STILL appears
    error_alerts_after = [e.value for e in at.error]
    assert any("Service Error Spike(s) Detected" in e for e in error_alerts_after), (
        f"Operational spike alert was suppressed by visual filter! at.error: {error_alerts_after}"
    )


# ===========================================================================
# Test E: Dataset Mode Isolation
# ===========================================================================
def test_dataset_mode_isolation():
    """Verify dataset isolation between simulated logs and uploaded logs across mode switches."""
    at = create_app()
    assert not at.exception

    # Generate 5 additional simulated events so count is 29
    gen_btn = next((b for b in at.sidebar.button if "Generate 5 Events" in b.label), None)
    assert gen_btn is not None
    gen_btn.click().run()
    assert len(at.session_state["logs_df"]) == 29

    # Switch to uploaded mode
    at.sidebar.radio[0].set_value("Uploaded Log File").run()
    assert not at.exception
    metrics_uploaded = _get_metric_map(at)
    assert metrics_uploaded.get("Total log records") == "0"

    # Inject sample uploaded DataFrame into session state
    custom_uploaded_df = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2026-11-01 10:00:00"),
            "level": "INFO",
            "service": "auth-service",
            "message": "User session established",
        },
        {
            "timestamp": pd.Timestamp("2026-11-01 10:01:00"),
            "level": "INFO",
            "service": "api-gateway",
            "message": "GET /health 200 OK",
        },
    ])
    at.session_state["uploaded_logs_df"] = custom_uploaded_df
    at.session_state["uploaded_file_name"] = "custom_test.log"
    at.run()

    # Verify uploaded mode reflects the 2 uploaded records
    metrics_custom = _get_metric_map(at)
    assert metrics_custom.get("Total log records") == "2"

    # Switch back to simulated mode
    at.session_state["data_source_mode"] = "simulated"
    at.sidebar.radio[0].set_value("Simulated Demonstration Logs").run()
    assert not at.exception

    # Verify simulated dataset remained intact with 29 records and was not overwritten
    metrics_restored = _get_metric_map(at)
    assert metrics_restored.get("Total log records") == "29"
    assert len(at.session_state["logs_df"]) == 29
    assert at.session_state["uploaded_logs_df"] is not None
    assert len(at.session_state["uploaded_logs_df"]) == 2


# ===========================================================================
# Test F: Empty Dataset Messaging vs No-Spike Messaging
# ===========================================================================
def test_empty_dataset_messaging():
    """Verify empty datasets show informational messages and non-empty datasets show accurate no-spike wording."""
    # Subcase 1: Empty active dataset (uploaded mode with no file)
    at_empty = create_app()
    at_empty.sidebar.radio[0].set_value("Uploaded Log File").run()
    assert not at_empty.exception

    empty_infos = [info.value for info in at_empty.info]
    assert any("No log records available" in text for text in empty_infos), (
        f"Expected informational empty banner: {empty_infos}"
    )

    # Empty active dataset must NOT claim 'No error spikes detected' or 'operating normally'
    for s in at_empty.success:
        assert "no error spikes detected" not in s.value.lower(), (
            f"Empty dataset incorrectly rendered no-spike success alert: {s.value}"
        )
        assert "expected rates" not in s.value.lower()
        assert "operating normally" not in s.value.lower()

    # Subcase 2: Non-empty dataset without spikes (baseline sample logs)
    at_baseline = create_app()
    assert not at_baseline.exception

    success_messages = [s.value for s in at_baseline.success]
    assert any("No error spikes detected" in text for text in success_messages), (
        f"Expected accurate 'No error spikes detected' banner: {success_messages}"
    )

    # Defensive copy check: Alert must not claim the whole system is healthy/operating normally
    for text in success_messages:
        assert "operating normally" not in text.lower(), (
            f"Banner overclaimed system health: {text}"
        )
        assert "all services are operating" not in text.lower(), (
            f"Banner overclaimed service normalcy: {text}"
        )


# ===========================================================================
# Test G: RCA Integration Under Active vs Filtered Evaluation
# ===========================================================================
def test_rca_integration_active_dataset_evaluation():
    """Verify RCA components render and evaluate active dataset independent of sidebar visual filters."""
    at = create_app()
    assert not at.exception

    # 1. Assert baseline RCA components are rendered
    initial_metrics = _get_metric_map(at)
    assert initial_metrics.get("Analysis Status") == "Completed"
    assert initial_metrics.get("Errors Analyzed") == "7"
    assert int(initial_metrics.get("Incident Clusters", 0)) >= 1
    assert int(initial_metrics.get("Ranked Hypotheses", 0)) >= 1
    assert int(initial_metrics.get("Matched Patterns", 0)) >= 1

    # 2. Deselect 'ERROR' in sidebar visual filter
    level_ms = next((ms for ms in at.sidebar.multiselect if ms.label == "Log level"), None)
    assert level_ms is not None
    level_ms.unselect("ERROR").run()
    assert not at.exception, f"Exception after deselecting ERROR: {at.exception}"

    # 3. Verify RCA still evaluates all 7 errors from the active dataset
    filtered_metrics = _get_metric_map(at)
    assert filtered_metrics.get("Analysis Status") == "Completed"
    assert filtered_metrics.get("Errors Analyzed") == "7", (
        f"RCA errors analyzed changed when visual filter changed: {filtered_metrics.get('Errors Analyzed')}"
    )
    assert filtered_metrics.get("Incident Clusters") == initial_metrics.get("Incident Clusters")
    assert filtered_metrics.get("Ranked Hypotheses") == initial_metrics.get("Ranked Hypotheses")
    assert filtered_metrics.get("Matched Patterns") == initial_metrics.get("Matched Patterns")

    # 4. Verify that triage error investigation section reflects visual filter (0 errors)
    assert "Total errors in selection" not in filtered_metrics, (
        "Total errors in selection metric should be absent when ERROR is deselected"
    )

