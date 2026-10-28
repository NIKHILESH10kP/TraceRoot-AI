# TraceRoot AI

**Intelligent Application Failure Detection and Root Cause Analysis System**

TraceRoot AI is an application monitoring and failure investigation platform designed to analyze application logs, detect abnormal error patterns, correlate evidence across services, rank probable root causes, and recommend corrective actions.

> [!NOTE]
> **Prototype Status:** TraceRoot AI is currently in **Milestone 1** (initial prototype). All log data shown in the dashboard is **simulated demonstration data**. The platform does not yet connect to live applications, stream real-time logs, or execute autonomous AI root cause investigations.

---

## Current Features (Milestone 1)

- **Simulated Log Stream:** 24 realistic sample log records simulating request completions, authentication failures, database timeouts, HTTP 500 errors, and deployment events across 6 distinct microservices.
- **Headline Summary Metrics:** Instant overview of total log count (24), ERROR count (7), WARNING count (7), and unique active services (6).
- **Interactive Multi-Attribute Filters:** Filter the log view dynamically by log level (`INFO`, `WARNING`, `ERROR`) and service name via the sidebar.
- **Log Records Explorer:** Tabular inspection of filtered records with timestamps, log levels, service identifiers, and message bodies.
- **Severity Distribution Visualization:** Compact horizontal bar chart illustrating the proportion of INFO, WARNING, and ERROR records.
- **Error Investigation Section:**
  - Aggregated error counts grouped by producing service.
  - Repeated error message tracking with occurrence frequency and first/last seen timestamps.
  - Plain-language triage guidance to assist operators in prioritizing investigations.
- **Rule-Based Anomaly Detection:**
  - **Fixed Threshold Rule:** Flags any service logging 3 or more errors.
  - **Statistical Outlier Rule:** Flags services with 2 or more errors whose error count exceeds the cohort average by more than 1 standard deviation (active when 3 or more services are selected).
  - Assigns `High` or `Medium` priority with clear explanatory summaries.
- **Defensive Error Handling:** Graceful fallback messaging and empty-state handling when filter selections return zero records.

---

## Technology Stack & Compatibility

- **Python Runtime:** Python **3.12** recommended (compatible with Python 3.11 – 3.14)
- **Frontend & Dashboard:** Streamlit (`>=1.40.0, <2.0.0`)
- **Data Analysis & Processing:** Pandas (`>=2.2.0, <4.0.0`)
- **Testing Framework:** Pytest (`>=8.0.0, <9.0.0`)

---

## Local Setup Guide

### 1. Prerequisites

Ensure Python 3.12 (or a compatible 3.11+ version) is installed on your system.

Clone the repository and navigate into the project root:

```bash
cd Traceroot-AI
```

### 2. Create and Activate a Virtual Environment

#### On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*Note: If PowerShell restricts script execution, run once for your user session:*
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

#### On macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

When activated, `(.venv)` will appear at the beginning of your shell prompt.

### 3. Install Dependencies

#### For running the application:

```bash
pip install -r requirements.txt
```

#### For development and running tests:

```bash
pip install -r requirements-dev.txt
```

### 4. Run the Dashboard

```bash
streamlit run app.py
```

Streamlit will launch a local web server (typically at `http://localhost:8501`) and automatically open the dashboard in your default browser. Press `Ctrl+C` in your terminal to shut down the server.

---

## Running Automated Tests

TraceRoot AI includes an automated unit test suite covering pure analytical functions, summary counts, filtering logic, and anomaly detection rules.

Run the tests using `pytest`:

```bash
pytest
```

To run with verbose output:

```bash
pytest -v
```

---

## Current Architecture & Limitations

- **Simulated In-Memory Data:** All 24 log records are hard-coded in Python tuple structures. Refreshing the dashboard resets the view to the initial state.
- **Deterministic Heuristics:** Anomaly detection relies on simple counting thresholds and standard deviation cutoffs rather than machine learning or temporal spike analysis.
- **No External Services:** No database, external API, authentication layer, or cloud log stream integrations are connected in this milestone.

---

## Planned Future Milestones

1. **Milestone 1:** Basic log monitoring dashboard with simulated data *(Current version)*
2. **Milestone 2:** Time-series error spike detection and temporal anomaly correlation
3. **Milestone 3:** Evidence-based AI root cause analysis using LLM reasoning and pattern clustering
4. **Milestone 4:** Incident timeline construction, severity scoring, and mitigation recommendations
5. **Milestone 5:** Database integration (PostgreSQL / SQLite), Docker containerization, and CI/CD pipelines
6. **Milestone 6:** Real-time log ingestion via OpenTelemetry, WebSockets, or Fluentd
7. **Milestone 7:** Automated DevOps remediation hooks and incident ticketing workflows
