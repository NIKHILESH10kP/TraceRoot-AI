"""
TraceRoot AI - Log Monitoring Dashboard with basic anomaly detection
(Milestone 1 dashboard + first version of Milestone 2 error-spike detection)

This dashboard displays SIMULATED log data only. It does not read real
application logs, and nothing here is live monitoring.

Run with:
    python -m streamlit run app.py
"""

import pandas as pd
import streamlit as st

# The three log levels used throughout the app, in order of increasing severity.
LOG_LEVELS = ["INFO", "WARNING", "ERROR"]

# Height (in pixels) of the severity chart. Keeps the chart compact.
SEVERITY_CHART_HEIGHT = 220

# --- Anomaly detection settings (simple, beginner-friendly rules) ---
# Rule 1 (fixed threshold): a service with this many errors (or more) is flagged.
ERROR_COUNT_THRESHOLD = 3
# Rule 2 (above-average): flag a service whose error count is more than this many
# standard deviations above the average error count of all services in the selection.
STD_DEVIATIONS_ABOVE_AVERAGE = 1
# Rule 2 needs enough services to make an "average" meaningful.
MIN_SERVICES_FOR_STATISTICS = 3
# Rule 2 ignores services with fewer errors than this (one stray error is not a spike).
MIN_ERRORS_FOR_STATISTICAL_RULE = 2

ANOMALY_COLUMNS = ["service", "error_count", "rules_triggered", "priority", "explanation"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_sample_logs() -> pd.DataFrame:
    """Return a DataFrame of simulated log records (fixed, hard-coded data)."""
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


# ---------------------------------------------------------------------------
# Data processing (plain functions, easy to read and test)
# ---------------------------------------------------------------------------
def filter_logs(logs: pd.DataFrame, selected_levels: list, selected_services: list) -> pd.DataFrame:
    """Keep only rows whose level AND service were selected in the sidebar."""
    level_matches = logs["level"].isin(selected_levels)
    service_matches = logs["service"].isin(selected_services)
    return logs[level_matches & service_matches]


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
    return logs[logs["level"] == "ERROR"]


def summarize_errors_by_service(errors: pd.DataFrame) -> pd.DataFrame:
    """Count how many errors each service produced (most errors first)."""
    counts = errors.groupby("service").size().reset_index(name="error_count")
    return counts.sort_values("error_count", ascending=False).reset_index(drop=True)


def summarize_error_messages(errors: pd.DataFrame) -> pd.DataFrame:
    """Group identical error messages and show how often each one occurred."""
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
# Anomaly detection (basic, deterministic rules on simulated data)
# ---------------------------------------------------------------------------
def count_errors_per_service(logs: pd.DataFrame) -> pd.DataFrame:
    """Count ERROR records for every service in the logs.

    Services that appear in the logs but have no errors get a count of 0, so
    the "average" used by the detection rule reflects all services.
    """
    is_error = logs["level"] == "ERROR"
    counts = is_error.groupby(logs["service"]).sum().astype(int)
    return counts.rename("error_count").reset_index()


def calculate_statistical_cutoff(error_counts: pd.DataFrame):
    """Work out the cutoff used by the above-average rule.

    cutoff = average error count + (STD_DEVIATIONS_ABOVE_AVERAGE x standard deviation)

    Returns None when there are too few services for the statistics to mean much.
    """
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
    """Flag services with unusually high error counts.

    Two simple rules are checked for every service:
      1. Fixed threshold: error_count >= ERROR_COUNT_THRESHOLD
      2. Above-average:   error_count >= MIN_ERRORS_FOR_STATISTICAL_RULE and
                          error_count > average + STD_DEVIATIONS_ABOVE_AVERAGE x std deviation

    Priority: "High" if both rules trigger, "Medium" if only one does.
    The same input always gives the same output (no randomness).
    """
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
    return anomalies.sort_values(
        ["error_count", "service"], ascending=[False, True]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dashboard sections (each function draws one part of the page)
# ---------------------------------------------------------------------------
def show_header() -> None:
    st.title("TraceRoot AI")
    st.subheader("Intelligent Application Failure Detection and Root Cause Analysis System")
    st.warning(
        "**Simulated demonstration data.** The logs below are hard-coded sample "
        "records, not live monitoring of a real application."
    )


def show_summary_metrics(logs: pd.DataFrame) -> None:
    st.header("Summary")
    summary = calculate_summary(logs)
    col_total, col_errors, col_warnings, col_services = st.columns(4)
    col_total.metric("Total log records", summary["total"])
    col_errors.metric("ERROR records", summary["errors"])
    col_warnings.metric("WARNING records", summary["warnings"])
    col_services.metric("Unique services", summary["services"])


def show_sidebar_filters(logs: pd.DataFrame):
    """Draw the filters in the sidebar and return what the user selected."""
    st.sidebar.header("Filters")
    selected_levels = st.sidebar.multiselect(
        "Log level", options=LOG_LEVELS, default=LOG_LEVELS
    )
    all_services = sorted(logs["service"].unique())
    selected_services = st.sidebar.multiselect(
        "Service name", options=all_services, default=all_services
    )
    return selected_levels, selected_services


def show_severity_chart(severity_counts: pd.Series) -> None:
    """Draw a compact horizontal bar chart so severity labels stay horizontal."""
    try:
        st.bar_chart(severity_counts, horizontal=True, height=SEVERITY_CHART_HEIGHT)
    except TypeError:
        # Older Streamlit versions do not support horizontal bars.
        st.bar_chart(severity_counts, height=SEVERITY_CHART_HEIGHT)


def show_log_table_and_chart(filtered_logs: pd.DataFrame) -> None:
    st.header("Log records")
    st.caption(f"Showing {len(filtered_logs)} record(s) that match the current filters.")

    if filtered_logs.empty:
        st.info(
            "No log records match the current filters. "
            "Try selecting more log levels or services in the sidebar."
        )
        return

    # Table on the left, compact chart on the right, so the chart does not
    # push the rest of the page far down.
    table_column, chart_column = st.columns([3, 2])
    with table_column:
        st.dataframe(filtered_logs, hide_index=True)
    with chart_column:
        st.subheader("Log records by severity")
        show_severity_chart(count_by_severity(filtered_logs))


def show_error_investigation(filtered_logs: pd.DataFrame) -> None:
    st.header("Error investigation")
    errors = get_error_records(filtered_logs)

    if errors.empty:
        st.info(
            "There are no ERROR records in the current selection, so there is "
            "nothing to investigate. Include the ERROR level in the sidebar "
            "filter, or choose different services."
        )
        return

    errors_by_service = summarize_errors_by_service(errors)
    error_messages = summarize_error_messages(errors)

    st.metric("Total errors", len(errors))

    st.subheader("Services producing errors")
    st.dataframe(errors_by_service, hide_index=True)

    st.subheader("Error messages")
    st.dataframe(error_messages, hide_index=True)

    st.subheader("Which errors deserve investigation first?")
    show_investigation_guidance(errors_by_service, error_messages)


def show_investigation_guidance(errors_by_service: pd.DataFrame, error_messages: pd.DataFrame) -> None:
    """Explain, in plain language, how to prioritise the errors."""
    st.markdown(
        "- **Repeated errors** (the same message appearing more than once) are "
        "usually worth checking first, because they suggest a problem that keeps happening.\n"
        "- **Services with the most errors** are good places to start looking.\n"
        "- **A single, one-off error** may be a temporary glitch, so it is lower priority "
        "unless it affects something important."
    )

    repeated = error_messages[error_messages["occurrences"] > 1]
    top_service = errors_by_service.iloc[0]

    if repeated.empty:
        st.write("In this selection, no error message repeats.")
    else:
        st.write(f"In this selection, {len(repeated)} error message(s) repeat:")
        for _, row in repeated.iterrows():
            st.write(f"- `{row['service']}`: \"{row['message']}\" ({row['occurrences']} times)")

    st.write(
        f"`{top_service['service']}` produced the most errors "
        f"({top_service['error_count']}) in this selection."
    )
    st.caption(
        "This is a simple counting rule to help you decide where to look first. "
        "It does not identify a root cause; errors in one service can be "
        "symptoms of a problem somewhere else."
    )


def show_anomaly_detection(filtered_logs: pd.DataFrame) -> None:
    """Show services with unusually high error counts (uses the active filters)."""
    st.header("Anomaly Detection")
    st.caption(
        "Basic anomaly detection on simulated data. This is a learning-level "
        "demonstration, not production-grade monitoring."
    )

    with st.expander("How does the detection work?"):
        st.markdown(
            "Errors are counted for each service in the current selection. "
            "A service is flagged when it triggers at least one of these rules:\n"
            f"1. **Fixed threshold:** it has {ERROR_COUNT_THRESHOLD} or more errors.\n"
            f"2. **Above-average rule:** it has at least {MIN_ERRORS_FOR_STATISTICAL_RULE} errors "
            f"and more than the average error count plus {STD_DEVIATIONS_ABOVE_AVERAGE} standard "
            f"deviation across services (needs at least {MIN_SERVICES_FOR_STATISTICS} services).\n\n"
            "**Priority:** High if both rules trigger, Medium if one does.\n\n"
            "A flagged service is a lead to investigate. It does not prove a root cause, "
            "because a service can show errors as a symptom of a problem elsewhere."
        )

    if filtered_logs.empty:
        st.info(
            "No anomalies detected in the current selection. "
            "The selection contains no log records."
        )
        return

    error_counts = count_errors_per_service(filtered_logs)
    anomalies = detect_error_anomalies(error_counts)

    if anomalies.empty:
        st.info("No anomalies detected in the current selection.")
    else:
        st.write(f"{len(anomalies)} service(s) flagged in the current selection:")
        for _, anomaly in anomalies.iterrows():
            st.warning(
                f"**`{anomaly['service']}`** - {anomaly['error_count']} errors\n\n"
                f"**Rule triggered:** {anomaly['rules_triggered']}\n\n"
                f"**Priority:** {anomaly['priority']}\n\n"
                f"{anomaly['explanation']}"
            )

    with st.expander("Show the error counts used for detection"):
        statistics = calculate_statistical_cutoff(error_counts)
        if statistics is None:
            st.caption(
                f"Fewer than {MIN_SERVICES_FOR_STATISTICS} services are selected, so the "
                "above-average rule is skipped. Only the fixed threshold applies."
            )
        else:
            st.caption(
                f"Average errors per service: {statistics['average']:.2f}. "
                f"Above-average cutoff: {statistics['cutoff']:.2f}. "
                f"Fixed threshold: {ERROR_COUNT_THRESHOLD}."
            )
        st.dataframe(error_counts, hide_index=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="TraceRoot AI", page_icon="🔍", layout="wide")

    all_logs = load_sample_logs()

    show_header()
    show_summary_metrics(all_logs)

    selected_levels, selected_services = show_sidebar_filters(all_logs)
    filtered_logs = filter_logs(all_logs, selected_levels, selected_services)

    show_log_table_and_chart(filtered_logs)
    show_error_investigation(filtered_logs)
    show_anomaly_detection(filtered_logs)


if __name__ == "__main__":
    main()
