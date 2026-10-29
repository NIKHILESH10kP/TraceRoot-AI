"""
TraceRoot AI - Log Monitoring & Error Spike Detection Dashboard
(Milestone 1 Baseline + Milestone 2 Dynamic Event Simulation & Rolling Spike Detection)

This dashboard displays SIMULATED log data only. It does not read real
application logs, and nothing here is live monitoring of production systems.

Run with:
    python -m streamlit run app.py
"""

import random
import time
import pandas as pd
import streamlit as st

# The three log levels used throughout the app, in order of increasing severity.
LOG_LEVELS = ["INFO", "WARNING", "ERROR"]

# Height (in pixels) of the severity chart. Keeps the chart compact.
SEVERITY_CHART_HEIGHT = 220

# --- Anomaly detection settings (Milestone 1 cohort statistics) ---
ERROR_COUNT_THRESHOLD = 3
STD_DEVIATIONS_ABOVE_AVERAGE = 1
MIN_SERVICES_FOR_STATISTICS = 3
MIN_ERRORS_FOR_STATISTICAL_RULE = 2

ANOMALY_COLUMNS = ["service", "error_count", "rules_triggered", "priority", "explanation"]

# --- Error Spike Detection settings (Milestone 2 rolling time window) ---
DEFAULT_SPIKE_WINDOW_MINUTES = 10
DEFAULT_SPIKE_ERROR_THRESHOLD = 3
MAX_STORED_LOGS = 500

SPIKE_COLUMNS = [
    "service",
    "error_count",
    "threshold",
    "window_minutes",
    "severity",
    "window_start",
    "window_end",
    "explanation",
]

# Monitored microservices
SERVICES = [
    "api-gateway",
    "auth-service",
    "order-service",
    "payment-service",
    "database-proxy",
    "deployment-service",
]

# Realistic simulated log message templates by service and level
MESSAGE_TEMPLATES = {
    "api-gateway": {
        "INFO": [
            "GET /api/products returned 200 OK in 85 ms",
            "GET /api/orders returned 200 OK in 110 ms",
            "POST /api/cart/items returned 201 Created in 64 ms",
            "GET /api/users/profile returned 200 OK in 45 ms",
        ],
        "WARNING": [
            "Slow response: GET /api/orders took 2450 ms",
            "Slow response: POST /api/payments took 3100 ms",
            "Rate limit threshold reached (80%) for IP 192.168.1.45",
        ],
        "ERROR": [
            "HTTP 500 Internal Server Error on POST /api/payments",
            "HTTP 504 Gateway Timeout: upstream service unreachable",
            "HTTP 502 Bad Gateway: connection refused by upstream",
        ],
    },
    "auth-service": {
        "INFO": [
            "User login successful for user_id=1042",
            "User login successful for user_id=3310",
            "OAuth token refreshed successfully for user_id=2087",
            "Session token validated successfully",
        ],
        "WARNING": [
            "Failed login attempt for user_id=2087 (invalid password)",
            "Account locked temporarily for user_id=4091 after 3 failed attempts",
            "Slow token verification: validation took 1250 ms",
        ],
        "ERROR": [
            "Authentication failure: token validation service unreachable",
            "OAuth provider timeout after 15 seconds",
            "Cryptographic signature verification failed for token header",
        ],
    },
    "order-service": {
        "INFO": [
            "Order #5001 created successfully",
            "Order #5003 created successfully",
            "Order #5005 created successfully",
            "Inventory reservation confirmed for order #5006",
        ],
        "WARNING": [
            "Order #5004 processing delayed: waiting on payment-service",
            "Inventory stock below reorder threshold for SKU-902",
            "Order notification delivery retry 1 of 3",
        ],
        "ERROR": [
            "Order creation failed: database constraint violation",
            "Order processing aborted: inventory reservation failed",
            "Order state machine error: invalid transition pending -> fulfilled",
        ],
    },
    "payment-service": {
        "INFO": [
            "Payment processed successfully for order #5001",
            "Payment processed successfully for order #5005",
            "Refund processed successfully for order #4990",
            "Payment pre-authorization completed in 140 ms",
        ],
        "WARNING": [
            "Retrying payment for order #5002 (attempt 2 of 3)",
            "Payment gateway latency elevated: transaction took 4200 ms",
            "Card processor warning: address verification mismatch",
        ],
        "ERROR": [
            "Payment failed for order #5002: database connection timeout",
            "Payment gateway connection timeout after 30 seconds",
            "Payment transaction declined: processor unreachable (503)",
        ],
    },
    "database-proxy": {
        "INFO": [
            "Connection pool usage returned to normal (62/200)",
            "Read replica query completed in 18 ms",
            "Periodic connection health check passed (200 connections healthy)",
            "Transaction committed successfully in 12 ms",
        ],
        "WARNING": [
            "Connection pool usage at 85% (170/200)",
            "Slow query detected: SELECT * FROM transactions took 2100 ms",
            "Connection wait queue depth: 15 threads waiting",
        ],
        "ERROR": [
            "Database connection timeout after 30 seconds",
            "Connection pool exhausted (200/200 connections active)",
            "Primary database node failover detected; replica promotion in progress",
        ],
    },
    "deployment-service": {
        "INFO": [
            "Deployment started: payment-service v2.4.1",
            "Deployment completed: payment-service v2.4.1",
            "Configuration reload completed across cluster",
            "Canary deployment initiated for order-service v2.2",
        ],
        "WARNING": [
            "Deployment canary warning: error rate slightly elevated (1.2%)",
            "Rolling update delayed: waiting for draining connections",
        ],
        "ERROR": [
            "Deployment failed: healthcheck timed out after 180 seconds",
            "Automated rollback triggered for order-service v2.2",
        ],
    },
}


# ---------------------------------------------------------------------------
# Data Baseline & Dynamic Generation
# ---------------------------------------------------------------------------
def load_sample_logs() -> pd.DataFrame:
    """Return a DataFrame of the baseline simulated log records (24 original records)."""
    sample_records = [
        ("2026-10-27 09:00:02", "INFO", "api-gateway", "GET /api/products returned 200 OK in 85 ms"),
        ("2026-10-27 09:03:41", "INFO", "auth-service", "User login successful for user_id=1042"),
        ("2026-10-27 09:07:15", "INFO", "order-service", "Order #5001 created successfully"),
        ("2026-10-27 09:12:30", "WARNING", "auth-service", "Failed login attempt for user_id=2087 (invalid password)"),
        ("2026-10-27 09:14:05", "WARNING", "auth-service", "Failed login attempt for user_id=2087 (invalid password)"),
        ("2026-10-27 09:15:00", "INFO", "deployment-service", "Deployment started: payment-service v2.4.1"),
        ("2026-10-27 09:16:42", "INFO", "deployment-service", "Deployment completed: payment-service v2.4.1"),
        ("2026-10-27 09:18:10", "INFO", "payment-service", "Payment processed successfully for order #5001"),
        ("2026-10-27 09:21:55", "WARNING", "api-gateway", "Slow response: GET /api/orders took 2300 ms"),
        ("2026-10-27 09:24:18", "WARNING", "database-proxy", "Connection pool usage at 85% (170/200)"),
        ("2026-10-27 09:26:03", "ERROR", "database-proxy", "Database connection timeout after 30 seconds"),
        ("2026-10-27 09:26:09", "ERROR", "payment-service", "Payment failed for order #5002: database connection timeout"),
        ("2026-10-27 09:27:40", "ERROR", "api-gateway", "HTTP 500 Internal Server Error on POST /api/payments"),
        ("2026-10-27 09:29:12", "WARNING", "api-gateway", "Slow response: POST /api/payments took 4800 ms"),
        ("2026-10-27 09:31:25", "ERROR", "auth-service", "Authentication failure: token validation service unreachable"),
        ("2026-10-27 09:33:50", "INFO", "order-service", "Order #5003 created successfully"),
        ("2026-10-27 09:35:02", "ERROR", "database-proxy", "Database connection timeout after 30 seconds"),
        ("2026-10-27 09:36:47", "ERROR", "api-gateway", "HTTP 500 Internal Server Error on POST /api/payments"),
        ("2026-10-27 09:38:20", "WARNING", "payment-service", "Retrying payment for order #5002 (attempt 2 of 3)"),
        ("2026-10-27 09:40:05", "ERROR", "payment-service", "Payment failed for order #5002: database connection timeout"),
        ("2026-10-27 09:42:33", "INFO", "auth-service", "User login successful for user_id=3310"),
        ("2026-10-27 09:45:00", "WARNING", "order-service", "Order #5004 processing delayed: waiting on payment-service"),
        ("2026-10-27 09:47:16", "INFO", "database-proxy", "Connection pool usage returned to normal (62/200)"),
        ("2026-10-27 09:48:01", "INFO", "api-gateway", "GET /api/products returned 200 OK in 91 ms"),
    ]
    logs = pd.DataFrame(
        sample_records, columns=["timestamp", "level", "service", "message"]
    )
    logs["timestamp"] = pd.to_datetime(logs["timestamp"])
    return logs.sort_values("timestamp").reset_index(drop=True)


def generate_simulated_logs(
    base_time: pd.Timestamp = None,
    count: int = 5,
    error_probability: float = 0.15,
    warning_probability: float = 0.25,
    seed: int = None,
) -> pd.DataFrame:
    """Generate a batch of new simulated log events with advancing timestamps."""
    if seed is not None:
        random.seed(seed)

    if base_time is None:
        base_time = pd.Timestamp.now()
    else:
        base_time = pd.to_datetime(base_time)

    records = []
    current_time = base_time

    for _ in range(count):
        # Step forward in time by 15 to 45 seconds for each event
        current_time += pd.Timedelta(seconds=random.randint(15, 45))
        service = random.choice(SERVICES)

        rand_val = random.random()
        if rand_val < error_probability:
            level = "ERROR"
        elif rand_val < error_probability + warning_probability:
            level = "WARNING"
        else:
            level = "INFO"

        templates = MESSAGE_TEMPLATES.get(service, {}).get(level, [f"{service} generic event"])
        message = random.choice(templates)
        records.append({
            "timestamp": current_time,
            "level": level,
            "service": service,
            "message": message,
        })

    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def generate_error_spike(
    base_time: pd.Timestamp = None,
    service: str = "payment-service",
    count: int = 5,
    message: str = None,
) -> pd.DataFrame:
    """Deliberately produce a concentrated cluster of ERROR events for a target service."""
    if base_time is None:
        base_time = pd.Timestamp.now()
    else:
        base_time = pd.to_datetime(base_time)

    if message is None:
        default_messages = MESSAGE_TEMPLATES.get(service, {}).get("ERROR", ["Service error occurred"])
        message = default_messages[0]

    records = []
    current_time = base_time

    for i in range(count):
        # Cluster errors closely together (5 to 15 seconds apart)
        current_time += pd.Timedelta(seconds=random.randint(5, 15))
        records.append({
            "timestamp": current_time,
            "level": "ERROR",
            "service": service,
            "message": message,
        })

    df = pd.DataFrame(records, columns=["timestamp", "level", "service", "message"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def append_and_cap_logs(
    existing_logs: pd.DataFrame,
    new_logs: pd.DataFrame,
    max_records: int = MAX_STORED_LOGS,
) -> pd.DataFrame:
    """Append new log records to existing records and cap at max_records to prevent memory growth."""
    if existing_logs.empty:
        combined = new_logs.copy()
    elif new_logs.empty:
        combined = existing_logs.copy()
    else:
        combined = pd.concat([existing_logs, new_logs], ignore_index=True)

    if combined.empty:
        return pd.DataFrame(columns=["timestamp", "level", "service", "message"])

    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    if len(combined) > max_records:
        combined = combined.iloc[-max_records:].reset_index(drop=True)

    return combined


# ---------------------------------------------------------------------------
# Data processing & analytics (pure functions, easily testable)
# ---------------------------------------------------------------------------
def filter_logs(logs: pd.DataFrame, selected_levels: list, selected_services: list) -> pd.DataFrame:
    """Keep only rows whose level AND service were selected in the sidebar."""
    if logs.empty:
        return logs.copy()
    level_matches = logs["level"].isin(selected_levels)
    service_matches = logs["service"].isin(selected_services)
    return logs[level_matches & service_matches].reset_index(drop=True)


def calculate_summary(logs: pd.DataFrame) -> dict:
    """Count the headline numbers shown at the top of the dashboard."""
    return {
        "total": len(logs),
        "errors": int((logs["level"] == "ERROR").sum()),
        "warnings": int((logs["level"] == "WARNING").sum()),
        "services": int(logs["service"].nunique()),
    }


def count_by_severity(logs: pd.DataFrame) -> pd.Series:
    """Count records per log level, always listing all three levels (even if 0)."""
    return logs["level"].value_counts().reindex(LOG_LEVELS, fill_value=0)


def get_error_records(logs: pd.DataFrame) -> pd.DataFrame:
    """Return only the ERROR rows."""
    if logs.empty:
        return logs.copy()
    return logs[logs["level"] == "ERROR"].reset_index(drop=True)


def summarize_errors_by_service(errors: pd.DataFrame) -> pd.DataFrame:
    """Count how many errors each service produced (most errors first)."""
    if errors.empty:
        return pd.DataFrame(columns=["service", "error_count"])
    counts = errors.groupby("service").size().reset_index(name="error_count")
    return counts.sort_values("error_count", ascending=False).reset_index(drop=True)


def summarize_error_messages(errors: pd.DataFrame) -> pd.DataFrame:
    """Group identical error messages and show how often each one occurred."""
    if errors.empty:
        return pd.DataFrame(columns=["service", "message", "occurrences", "first_seen", "last_seen"])
    grouped = (
        errors.groupby(["service", "message"])
        .agg(
            occurrences=("message", "size"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
        .reset_index()
    )
    return grouped.sort_values(
        ["occurrences", "last_seen"], ascending=[False, False]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Milestone 1: Cohort Anomaly Detection (Statistical Baseline)
# ---------------------------------------------------------------------------
def count_errors_per_service(logs: pd.DataFrame) -> pd.DataFrame:
    """Count ERROR records for every service in the logs."""
    if logs.empty:
        return pd.DataFrame(columns=["service", "error_count"])
    is_error = logs["level"] == "ERROR"
    counts = is_error.groupby(logs["service"]).sum().astype(int)
    return counts.rename("error_count").reset_index()


def calculate_statistical_cutoff(error_counts: pd.DataFrame):
    """Work out the statistical cutoff across all services."""
    if len(error_counts) < MIN_SERVICES_FOR_STATISTICS:
        return None
    average = float(error_counts["error_count"].mean())
    spread = float(error_counts["error_count"].std(ddof=0))
    return {
        "average": average,
        "spread": spread,
        "cutoff": average + STD_DEVIATIONS_ABOVE_AVERAGE * spread,
    }


def detect_error_anomalies(error_counts: pd.DataFrame) -> pd.DataFrame:
    """Flag services with unusually high cumulative error counts across the cohort."""
    statistics = calculate_statistical_cutoff(error_counts)
    flagged_rows = []

    for _, row in error_counts.iterrows():
        service = row["service"]
        error_count = int(row["error_count"])
        rules_triggered = []
        reasons = []

        if error_count >= ERROR_COUNT_THRESHOLD:
            rules_triggered.append("Fixed threshold")
            reasons.append(
                f"it logged {error_count} errors, which is at or above the fixed limit of "
                f"{ERROR_COUNT_THRESHOLD}"
            )

        if (
            statistics is not None
            and error_count >= MIN_ERRORS_FOR_STATISTICAL_RULE
            and error_count > statistics["cutoff"]
        ):
            rules_triggered.append("Above-average rule")
            reasons.append(
                f"its {error_count} errors are above the cutoff of {statistics['cutoff']:.1f} "
                f"(the average is {statistics['average']:.1f} errors per service, plus "
                f"{STD_DEVIATIONS_ABOVE_AVERAGE} standard deviation)"
            )

        if not rules_triggered:
            continue

        priority = "High" if len(rules_triggered) == 2 else "Medium"
        explanation = (
            f"{service} was flagged because " + " and ".join(reasons) + ". "
            "It is worth reviewing, but this does not prove it is the root cause."
        )
        flagged_rows.append(
            {
                "service": service,
                "error_count": error_count,
                "rules_triggered": " + ".join(rules_triggered),
                "priority": priority,
                "explanation": explanation,
            }
        )

    anomalies = pd.DataFrame(flagged_rows, columns=ANOMALY_COLUMNS)
    if anomalies.empty:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)
    return anomalies.sort_values(
        ["error_count", "service"], ascending=[False, True]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Milestone 2: Rolling Time-Window Error Spike Detection
# ---------------------------------------------------------------------------
def detect_rolling_error_spikes(
    logs: pd.DataFrame,
    window_minutes: int = DEFAULT_SPIKE_WINDOW_MINUTES,
    error_threshold: int = DEFAULT_SPIKE_ERROR_THRESHOLD,
    reference_time: pd.Timestamp = None,
) -> pd.DataFrame:
    """Detect services experiencing a surge in ERROR logs within a rolling time window.

    Parameters
    ----------
    logs : pd.DataFrame
        Log records containing at least 'timestamp', 'level', and 'service'.
    window_minutes : int
        Duration of the rolling window looking back from reference_time.
    error_threshold : int
        Minimum number of errors within the window to flag as a spike.
    reference_time : pd.Timestamp, optional
        The end timestamp of the evaluation window. Defaults to logs['timestamp'].max().

    Returns
    -------
    pd.DataFrame
        Detected spikes matching SPIKE_COLUMNS, sorted by error_count descending.
    """
    if logs.empty or "level" not in logs.columns or "timestamp" not in logs.columns:
        return pd.DataFrame(columns=SPIKE_COLUMNS)

    if (logs["level"] == "ERROR").sum() == 0:
        return pd.DataFrame(columns=SPIKE_COLUMNS)

    if reference_time is None:
        reference_time = logs["timestamp"].max()
    else:
        reference_time = pd.to_datetime(reference_time)

    window_start = reference_time - pd.Timedelta(minutes=window_minutes)

    in_window = (logs["timestamp"] >= window_start) & (logs["timestamp"] <= reference_time)
    window_logs = logs[in_window]

    errors_in_window = window_logs[window_logs["level"] == "ERROR"]
    if errors_in_window.empty:
        return pd.DataFrame(columns=SPIKE_COLUMNS)

    counts = errors_in_window.groupby("service").size().reset_index(name="error_count")
    spiked = counts[counts["error_count"] >= error_threshold].copy()

    if spiked.empty:
        return pd.DataFrame(columns=SPIKE_COLUMNS)

    rows = []
    for _, row in spiked.iterrows():
        service = row["service"]
        count = int(row["error_count"])
        # Severe surge (at least double the threshold) marked as Critical
        severity = "Critical" if count >= error_threshold * 2 else "High"
        explanation = (
            f"Service '{service}' experienced {count} errors within the last "
            f"{window_minutes} minutes (threshold: {error_threshold}). "
            "This suggests an unusual error spike (anomaly), though it is not "
            "definitive proof of the underlying root cause."
        )
        rows.append({
            "service": service,
            "error_count": count,
            "threshold": error_threshold,
            "window_minutes": window_minutes,
            "severity": severity,
            "window_start": window_start,
            "window_end": reference_time,
            "explanation": explanation,
        })

    result = pd.DataFrame(rows, columns=SPIKE_COLUMNS)
    return result.sort_values(["error_count", "service"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dashboard UI Rendering Sections
# ---------------------------------------------------------------------------
def show_header(latest_time: pd.Timestamp = None, total_records: int = 0) -> None:
    """Draw the application header and simulation notices."""
    st.title("TraceRoot AI")
    st.subheader("Intelligent Application Failure Detection & Root Cause Analysis System")
    st.warning(
        "⚠️ **Simulated Demonstration Environment.** All logs below are simulated records "
        "generated in-memory. This dashboard does NOT connect to a live production cluster or external database."
    )
    if latest_time is not None:
        st.caption(
            f"📅 **Latest Event Time:** `{latest_time.strftime('%Y-%m-%d %H:%M:%S')}` | "
            f"📊 **Cumulative Session Log Records:** `{total_records}`"
        )


def show_summary_metrics(logs: pd.DataFrame, is_filtered: bool = False) -> None:
    """Draw top headline summary metrics."""
    st.header("Summary Metrics")
    if is_filtered:
        st.caption("Reflecting current sidebar filter selections.")
    else:
        st.caption("Reflecting all active simulated logs.")

    summary = calculate_summary(logs)
    col_total, col_errors, col_warnings, col_services = st.columns(4)
    col_total.metric("Total log records", summary["total"])
    col_errors.metric("ERROR records", summary["errors"])
    col_warnings.metric("WARNING records", summary["warnings"])
    col_services.metric("Unique services", summary["services"])


def show_simulation_controls(current_logs: pd.DataFrame):
    """Draw simulation controls in the sidebar and return the auto-stream toggle state."""
    st.sidebar.markdown("---")
    st.sidebar.header("🎮 Log Simulation Controls")
    st.sidebar.caption("Generate dynamic simulated events over time (Milestone 2).")

    col_gen, col_reset = st.sidebar.columns(2)
    with col_gen:
        if st.sidebar.button("➕ Generate 5 Events", help="Generate 5 new simulated log events"):
            latest_time = current_logs["timestamp"].max() if not current_logs.empty else pd.Timestamp.now()
            new_batch = generate_simulated_logs(base_time=latest_time, count=5)
            updated = append_and_cap_logs(current_logs, new_batch)
            st.session_state["logs_df"] = updated
            st.rerun()

    with col_reset:
        if st.sidebar.button("🔄 Reset Baseline", help="Reset simulation back to original 24 records"):
            st.session_state["logs_df"] = load_sample_logs()
            st.rerun()

    st.sidebar.subheader("Inject Error Spike Scenario")
    spike_service = st.sidebar.selectbox("Target Service", options=SERVICES, index=3)  # payment-service
    if st.sidebar.button(f"⚡ Trigger Spike ({spike_service})", help=f"Inject 5 rapid ERROR events for {spike_service}"):
        latest_time = current_logs["timestamp"].max() if not current_logs.empty else pd.Timestamp.now()
        spike_batch = generate_error_spike(base_time=latest_time, service=spike_service, count=5)
        updated = append_and_cap_logs(current_logs, spike_batch)
        st.session_state["logs_df"] = updated
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Spike Detection Parameters")
    window_minutes = st.sidebar.slider(
        "Rolling Window (Minutes)", min_value=1, max_value=60, value=DEFAULT_SPIKE_WINDOW_MINUTES, step=1
    )
    error_threshold = st.sidebar.slider(
        "Spike Error Threshold", min_value=1, max_value=15, value=DEFAULT_SPIKE_ERROR_THRESHOLD, step=1
    )

    auto_stream = st.sidebar.toggle("⏱️ Auto-Stream Events (every 4s)", value=False)
    if auto_stream:
        st.sidebar.info("🔴 Streaming active. Click toggle to pause.")

    return window_minutes, error_threshold, auto_stream


def show_sidebar_filters(logs: pd.DataFrame):
    """Draw filter controls in the sidebar and return user selections."""
    st.sidebar.header("Log Filters")
    selected_levels = st.sidebar.multiselect(
        "Log level", options=LOG_LEVELS, default=LOG_LEVELS
    )
    all_services = sorted(logs["service"].unique()) if not logs.empty else SERVICES
    selected_services = st.sidebar.multiselect(
        "Service name", options=all_services, default=all_services
    )
    return selected_levels, selected_services


def show_error_spike_section(
    filtered_logs: pd.DataFrame,
    window_minutes: int = DEFAULT_SPIKE_WINDOW_MINUTES,
    error_threshold: int = DEFAULT_SPIKE_ERROR_THRESHOLD,
) -> None:
    """Draw the real-time rolling error spike detection section."""
    st.header("⚡ Real-Time Error Spike Detection")
    st.caption("Monitors the velocity of ERROR events within a rolling time window (Milestone 2).")

    with st.expander("How does rolling spike detection work?"):
        st.markdown(
            f"Unlike static counting, rolling spike detection evaluates errors within the last "
            f"**{window_minutes} minutes** of the event stream. When a service logs **{error_threshold} or more errors** "
            "in this window, a surge alert is raised.\n\n"
            "- **Severity: Critical** if error count is >= 2x threshold.\n"
            "- **Severity: High** if error count meets the threshold.\n\n"
            "⚠️ *An error spike flags an active anomaly for investigation, but is not definitive proof of root cause.*"
        )

    spikes = detect_rolling_error_spikes(
        filtered_logs, window_minutes=window_minutes, error_threshold=error_threshold
    )

    if spikes.empty:
        st.success(
            f"🟢 **No error spikes detected** in the last {window_minutes} minutes "
            f"(Threshold: {error_threshold} errors). All services operating within expected rates."
        )
    else:
        st.error(f"🚨 **{len(spikes)} Service Error Spike(s) Detected!**")
        for _, spike in spikes.iterrows():
            severity_badge = "🔴 Critical" if spike["severity"] == "Critical" else "🟠 High"
            st.markdown(
                f"**{severity_badge} Spike:** `{spike['service']}` logged **{spike['error_count']} errors** "
                f"in the last {spike['window_minutes']} minutes (Threshold: {spike['threshold']})."
            )
            st.caption(f"{spike['explanation']}")

        st.dataframe(spikes, hide_index=True)


def show_severity_chart(severity_counts: pd.Series) -> None:
    """Draw a compact horizontal bar chart so severity labels stay horizontal."""
    try:
        st.bar_chart(severity_counts, horizontal=True, height=SEVERITY_CHART_HEIGHT)
    except TypeError:
        # Older Streamlit versions fallback
        st.bar_chart(severity_counts, height=SEVERITY_CHART_HEIGHT)


def show_log_table_and_chart(filtered_logs: pd.DataFrame) -> None:
    """Draw the log records explorer and severity chart side by side."""
    st.header("Log Records")
    st.caption(f"Showing {len(filtered_logs)} record(s) matching active filters.")

    if filtered_logs.empty:
        st.info("No log records match the current filters. Select more levels or services in the sidebar.")
        return

    table_column, chart_column = st.columns([3, 2])
    with table_column:
        st.dataframe(filtered_logs, hide_index=True)
    with chart_column:
        st.subheader("Records by Severity")
        show_severity_chart(count_by_severity(filtered_logs))


def show_error_investigation(filtered_logs: pd.DataFrame) -> None:
    """Draw the error investigation triage section."""
    st.header("Error Investigation")
    errors = get_error_records(filtered_logs)

    if errors.empty:
        st.info(
            "There are no ERROR records in the current selection. "
            "Include ERROR level in the sidebar filter to investigate."
        )
        return

    errors_by_service = summarize_errors_by_service(errors)
    error_messages = summarize_error_messages(errors)

    st.metric("Total errors in selection", len(errors))

    st.subheader("Services Producing Errors")
    st.dataframe(errors_by_service, hide_index=True)

    st.subheader("Error Messages & Repeat Frequencies")
    st.dataframe(error_messages, hide_index=True)

    st.subheader("Investigation Guidance & Priority")
    show_investigation_guidance(errors_by_service, error_messages)


def show_investigation_guidance(errors_by_service: pd.DataFrame, error_messages: pd.DataFrame) -> None:
    """Explain in plain language how to prioritize the errors."""
    st.markdown(
        "- **Repeated errors** (the same message appearing more than once) are "
        "usually worth checking first, because they suggest a recurring issue.\n"
        "- **Services with the most errors** are good starting points for investigation.\n"
        "- **One-off errors** may be transient network blips, so they have lower priority."
    )

    repeated = error_messages[error_messages["occurrences"] > 1]
    if not errors_by_service.empty:
        top_service = errors_by_service.iloc[0]
        st.write(
            f"`{top_service['service']}` produced the most errors "
            f"({top_service['error_count']}) in this selection."
        )

    if repeated.empty:
        st.write("In this selection, no error message repeats.")
    else:
        st.write(f"In this selection, {len(repeated)} error message(s) repeat:")
        for _, row in repeated.iterrows():
            st.write(f"- `{row['service']}`: \"{row['message']}\" ({row['occurrences']} times)")

    st.caption(
        "Note: Counting rules help triage where to look first. Errors in a downstream service "
        "can be symptoms of an upstream failure."
    )


def show_anomaly_detection(filtered_logs: pd.DataFrame) -> None:
    """Show services with unusually high error counts across the cohort (Milestone 1)."""
    st.header("Cohort Anomaly Detection")
    st.caption("Baseline statistical and threshold rules across monitored services (Milestone 1).")

    with st.expander("How does cohort anomaly detection work?"):
        st.markdown(
            "Errors are counted per service across the selection. "
            "A service is flagged when it triggers at least one rule:\n"
            f"1. **Fixed threshold:** >= {ERROR_COUNT_THRESHOLD} cumulative errors.\n"
            f"2. **Above-average rule:** >= {MIN_ERRORS_FOR_STATISTICAL_RULE} errors "
            f"and > average + {STD_DEVIATIONS_ABOVE_AVERAGE} std deviation across services "
            f"(requires >= {MIN_SERVICES_FOR_STATISTICS} services).\n\n"
            "**Priority:** High if both rules trigger, Medium if one does."
        )

    if filtered_logs.empty:
        st.info("No anomalies detected in the current selection (no log records).")
        return

    error_counts = count_errors_per_service(filtered_logs)
    anomalies = detect_error_anomalies(error_counts)

    if anomalies.empty:
        st.info("No cohort anomalies detected in the current selection.")
    else:
        st.write(f"{len(anomalies)} service(s) flagged by cohort rules:")
        for _, anomaly in anomalies.iterrows():
            st.warning(
                f"**`{anomaly['service']}`** - {anomaly['error_count']} errors\n\n"
                f"**Rule triggered:** {anomaly['rules_triggered']}\n\n"
                f"**Priority:** {anomaly['priority']}\n\n"
                f"{anomaly['explanation']}"
            )

    with st.expander("Show cohort error counts used for statistical baseline"):
        statistics = calculate_statistical_cutoff(error_counts)
        if statistics is None:
            st.caption(
                f"Fewer than {MIN_SERVICES_FOR_STATISTICS} services selected; statistical rule skipped."
            )
        else:
            st.caption(
                f"Average errors/service: {statistics['average']:.2f} | "
                f"Cutoff: {statistics['cutoff']:.2f} | Fixed threshold: {ERROR_COUNT_THRESHOLD}"
            )
        st.dataframe(error_counts, hide_index=True)


# ---------------------------------------------------------------------------
# Main Application Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="TraceRoot AI", page_icon="🔍", layout="wide")

    # Initialize session state with baseline sample logs if not present
    if "logs_df" not in st.session_state:
        st.session_state["logs_df"] = load_sample_logs()

    all_logs = st.session_state["logs_df"]
    latest_time = all_logs["timestamp"].max() if not all_logs.empty else None

    # Header
    show_header(latest_time=latest_time, total_records=len(all_logs))

    # Sidebar: Simulation Controls & Spike Detection Settings
    window_minutes, error_threshold, auto_stream = show_simulation_controls(all_logs)

    # Sidebar: Filters
    selected_levels, selected_services = show_sidebar_filters(all_logs)
    filtered_logs = filter_logs(all_logs, selected_levels, selected_services)

    # Summary Metrics (reflecting active filtered selection)
    is_filtered = len(filtered_logs) != len(all_logs)
    show_summary_metrics(filtered_logs, is_filtered=is_filtered)

    # Milestone 2: Real-Time Rolling Error Spike Detection
    show_error_spike_section(filtered_logs, window_minutes=window_minutes, error_threshold=error_threshold)

    # Core Log Visualizer & Table
    show_log_table_and_chart(filtered_logs)

    # Error Investigation
    show_error_investigation(filtered_logs)

    # Milestone 1: Cohort Anomaly Detection
    show_anomaly_detection(filtered_logs)

    # Auto-streaming handler: sleeps and reruns safely if toggle is active
    if auto_stream:
        time.sleep(4)
        latest_ts = all_logs["timestamp"].max() if not all_logs.empty else pd.Timestamp.now()
        new_batch = generate_simulated_logs(base_time=latest_ts, count=2)
        st.session_state["logs_df"] = append_and_cap_logs(all_logs, new_batch)
        st.rerun()


if __name__ == "__main__":
    main()
