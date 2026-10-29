# TraceRoot AI

**Intelligent Application Failure Detection and Root Cause Analysis System**

TraceRoot AI is an application monitoring and failure investigation platform designed to analyze application logs, detect abnormal error patterns, correlate evidence across services, rank probable root causes, and recommend corrective actions.

> [!NOTE]
> **Prototype Status:** TraceRoot AI is currently in **Milestone 3** (Dual Data Sources: Dynamic Event Simulation + Real Log-File Ingestion). The platform supports both in-memory synthetic demonstration streams and user-uploaded `.log` / `.jsonl` files. It does not yet connect to live streaming infrastructure or external production databases.

---

## Current Features

### 1. Real Log-File Ingestion (Milestone 3)
- **Supported File Formats:** Ingest real application logs via `.log`, `.txt`, and `.jsonl` (JSON Lines) files.
- **Strict Format Validation:**
  - **Text Logs (`.log` / `.txt`):** Format: `<timestamp> <level> <service> <message>`. Example:  
    `2026-10-29 10:15:30 ERROR payment-service Database connection timeout`
  - **JSON Lines (`.jsonl`):** One JSON object per line with required fields `timestamp`, `level`, `service`, and `message` (supporting common key aliases like `time`/`ts`, `severity`, `app`/`service_name`, and `msg`).
- **Comprehensive Ingestion Report:** Surfaces total lines processed, valid imported records, rejected lines with line numbers and exact failure reasons, and blank lines ignored.
- **Data Source Isolation:** Explicit toggle between *Simulated Demonstration Logs* and *Uploaded Log File*. Uploaded records are never mixed with synthetic data.
- **Safety & Performance Guardrails:** 5 MB file size limit, 5,000 record maximum import cap, in-memory session processing (no disk persistence or code execution).

### 2. Dynamic Log Simulation (Milestone 2)
- **Interactive Batch Generation:** Add batches of 5 realistic, timestamped events across 6 core microservices (`api-gateway`, `auth-service`, `order-service`, `payment-service`, `database-proxy`, `deployment-service`).
- **Simulate Error Spike Scenario:** Inject concentrated error bursts (5 rapid errors in under 90 seconds) for any selected microservice to simulate active outages.
- **Controlled Auto-Streaming:** Toggle continuous simulated event streaming (every 4 seconds) with controlled rate-limiting and one-click pause.
- **Session State Persistence & Bounded Storage:** Retains simulated logs across filter adjustments in Streamlit session state, with a safe 500-record cap.
- **One-Click Baseline Reset:** Instantly revert to the original 24-record baseline.

### 3. Real-Time Rolling Error-Spike Detection (Milestone 2)
- **Temporal Velocity Analysis:** Evaluates error rates within a configurable rolling window (e.g., last 10 minutes) rather than static historical totals. Works seamlessly on both simulated and uploaded logs.
- **Dynamic Alerting:** Alerts trigger when error count meets or exceeds the threshold:
  - **High:** Error count reaches threshold within window.
  - **Critical:** Error count is at least double ($\ge 2\times$) the threshold.
- **Configurable Parameters:** Adjust the rolling window (1–60 minutes) and error threshold (1–15 errors) dynamically in the sidebar.

### 4. Log Analytics & Visualization (Milestone 1 Baseline)
- **Summary Metrics:** Dynamic counters for Total Records, ERRORs, WARNINGs, and Unique Services, with latest event timestamp.
- **Multi-Attribute Filters:** Filter records simultaneously by severity (`INFO`, `WARNING`, `ERROR`) and service.
- **Side-by-Side Explorer:** Synchronized tabular log explorer and horizontal severity distribution chart.
- **Error Investigation Triage:** Detailed breakdowns of producing services, repeat message frequencies, and triage guidance.
- **Cohort Anomaly Detection:** Statistical $\mu + 1\sigma$ and fixed-threshold rules across historical service totals.

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

TraceRoot AI includes an automated unit test suite with **43 passing tests** covering analytical functions, dynamic event generation, log parsing (.log and .jsonl), input validation, dataset isolation, and rolling spike detection.

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

- **Session-Local In-Memory Data:** Uploaded files and simulated records are processed in memory during the browser session. Closing the browser tab resets the simulation to the 24-record baseline and clears uploaded files (no disk persistence).
- **Rule-Based Heuristics:** Spike detection relies on rolling time windows and count thresholds rather than AI / machine learning models.
- **No External Services:** No database, external API, authentication layer, or cloud log stream integrations are connected in this milestone.

---

## Planned Future Milestones

1. **Milestone 1:** Basic log monitoring dashboard with simulated data *(Completed)*
2. **Milestone 2:** Dynamic simulated log generation and rolling time-window error-spike detection *(Completed)*
3. **Milestone 3:** Real log-file ingestion (.log and .jsonl) with validation and triage *(Completed)*
4. **Milestone 4:** Evidence-based AI root cause analysis using LLM reasoning and pattern clustering
5. **Milestone 5:** Incident timeline construction, severity scoring, and mitigation recommendations
6. **Milestone 6:** Database integration (PostgreSQL / SQLite), Docker containerization, and CI/CD pipelines
7. **Milestone 7:** Real-time log streaming via OpenTelemetry, WebSockets, or Fluentd
