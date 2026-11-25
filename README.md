# TraceRoot AI

**Intelligent Application Failure Detection and Root Cause Analysis System**

TraceRoot AI is an application monitoring and failure investigation platform designed to analyze application logs, detect abnormal error patterns, correlate evidence across distributed microservices, rank probable root causes using transparent deterministic heuristics, and recommend corrective actions.

> [!NOTE]
> **Prototype Status:** TraceRoot AI is in **Milestone 5** (Dual Data Sources, Deterministic Root Cause Analysis, Telemetry Privacy Sanitization, and Headless E2E Testing). The platform operates completely in-memory and 100% offline, with zero external cloud or LLM dependencies.

---

## Core Capabilities & Features

### 1. Real Log-File Ingestion (Milestone 3 & Hardening 5.1)
- **Supported File Formats:** Ingest real application logs via plain text (`.log`, `.txt`) and JSON Lines (`.jsonl`) files.
- **Strict Format Validation:**
  - **Text Logs (`.log` / `.txt`):** Format: `<timestamp> <level> <service> <message>`. Example:  
    `2026-10-29 10:15:30 ERROR payment-service Database connection timeout`
  - **JSON Lines (`.jsonl`):** One JSON object per line with required fields `timestamp`, `level`, `service`, and `message` (supporting common key aliases like `time`/`ts`, `severity`, `app`/`service_name`, and `msg`).
- **Comprehensive Ingestion Report:** Surfaces total lines processed, valid imported records, rejected lines with line numbers and exact failure reasons, and blank lines ignored.
- **SHA-256 Mutation Detection & Atomic State Management:** Computes SHA-256 content hashes to detect file modifications under identical filenames; atomically resets all session-state values upon replacement failure or file clearing, preventing stale data leakage.
- **Safety & Performance Guardrails:** 5 MB file size limit, 5,000 record maximum import cap, in-memory session processing (no disk persistence or code execution).

### 2. Dynamic Log Simulation (Milestone 2)
- **Interactive Batch Generation:** Add batches of 5 realistic, timestamped events across 6 core microservices (`api-gateway`, `auth-service`, `order-service`, `payment-service`, `database-proxy`, `deployment-service`).
- **Simulate Error Spike Scenario:** Inject concentrated error bursts (5 rapid errors in under 90 seconds) for any selected microservice to simulate active outages.
- **Controlled Auto-Streaming:** Toggle continuous simulated event streaming (every 4 seconds) with controlled rate-limiting and one-click pause.
- **Session State Persistence & Bounded Storage:** Retains simulated logs across filter adjustments in Streamlit session state, with a safe 500-record cap.
- **One-Click Baseline Reset:** Instantly revert to the original 24-record baseline.

### 3. Real-Time Rolling Error-Spike Detection (Milestones 2 & 5.2)
- **Temporal Velocity Analysis:** Evaluates error rates within a configurable rolling window (e.g., last 15 minutes) rather than static historical totals.
- **Filter-Independent Operational Monitoring:** Spike detection evaluates the complete active dataset (`active_logs`), ensuring operational alerts remain visible even when visual sidebar filters exclude `ERROR` records.
- **Dynamic Alerting:** Alerts trigger when error count meets or exceeds the threshold:
  - **High:** Error count reaches threshold within window.
  - **Critical:** Error count is at least double ($\ge 2\times$) the threshold.
- **Defensive Copy & Empty Dataset Handling:** Displays clear informational guidance when no records exist, without false "all services operating normally" claims.

### 4. Deterministic Rule-Based Root Cause Analysis (Milestone 4)
- **Temporal Event Correlation:** Groups related error events occurring across distinct microservices within close temporal proximity (< 120s sliding window) into cascading failure incident clusters.
- **Patient-Zero Chronology:** Evaluates log timestamps to identify the earliest failure in an incident cluster, granting priority score weighting (+25 pts).
- **Architectural Failure Pattern Matching:** Codified rules for common distributed system failure topologies:
  - *Database Exhaustion Cascade:* Connection pool exhaustion or query timeouts in `database-proxy` propagating downstream to `payment-service` and `order-service`, resulting in `api-gateway` 5xx errors (+30 pts).
  - *Deployment Regression:* Software deployment or configuration rollout immediately preceding an error surge within a 5-minute operational window (+30 pts).
  - *Authentication Cascade:* Token validation failures in `auth-service` causing upstream gateway rejections (+30 pts).
- **Transparent Confidence Scoring:** Evaluates hypotheses against a bounded scoring model ([0.0, 100.0]%) combining temporal precedence, pattern matching, blast radius breadth, and message repetition, with penalties for missing causal evidence.
- **Clear Hypothesis Framing:** All root-cause candidates are presented transparently as algorithmic hypotheses with verification checklists, debug recommendations, and observability limitation notices (100% offline, zero LLM hallucination risk).
- **Order-Sensitive RCA Caching:** Content-and-order sensitive SHA-256 dataset fingerprinting with strict cache separation between simulated and uploaded data modes.

### 5. Telemetry Privacy & Data Sanitization (Milestone 5)
- **Deep Redaction Engine:** High-performance regex sanitization covering Authorization headers (Bearer, Basic), JWT tokens, API keys, AWS access keys, passwords, credit card numbers, email addresses, and IPv4/IPv6 addresses.
- **Structural Schema Allowlisting:** Recursive sanitization for complex nested structures with strict schema allowlists, recursion depth protection, and a 64 KB serialized payload cap.
- **Numeric Boundary Guards:** Safe numeric normalization rejecting `NaN`, `Infinity`, `-Infinity`, and malformed types.

### 6. Log Analytics & Visualization (Milestone 1 Baseline)
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
- **E2E UI Testing:** Streamlit AppTest (`streamlit.testing.v1.AppTest`)

---

## Local Setup Guide

### 1. Prerequisites

Ensure Python 3.12 (or compatible 3.11+ version) is installed on your system.

Clone the repository and navigate into the project root:

```bash
git clone https://github.com/NIKHILESH10kP/TraceRoot-AI.git
cd TraceRoot-AI
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

TraceRoot AI includes a comprehensive automated test suite with **214 passing tests** across 5 distinct test suites:

- **`tests/test_analytics.py` (50 tests):** Ingestion, parsing, filtering, SHA-256 re-upload detection, atomic state resets, and rolling spike velocity.
- **`tests/test_privacy_sanitizer.py` (80 tests):** PII redaction, token stripping, schema allowlisting, and recursion depth boundaries.
- **`tests/test_rca_engine.py` (20 tests):** Temporal event clustering, patient-zero chronology, pattern matching, confidence scoring, and malformed input handling.
- **`tests/test_rca_integration.py` (57 tests):** RCA caching, session-state isolation, UI rendering fallbacks, and NaN/Inf robustness.
- **`tests/test_e2e_apptest.py` (7 tests):** Streamlit AppTest headless end-to-end integration flows (startup, simulation, upload empty state, filter decoupling, and RCA active evaluation).

Run the full test suite using `pytest`:

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
- **Deterministic Heuristics:** Root cause analysis relies on temporal proximity, patient-zero ordering, and architectural topology rules rather than opaque AI / cloud LLM models.
- **100% Offline & Private:** Zero telemetry phone-home calls, external APIs, cloud databases, or third-party tracking services are connected.

---

## Milestones Roadmap

1. **Milestone 1:** Basic log monitoring dashboard with simulated data *(Completed)*
2. **Milestone 2:** Dynamic simulated log generation and rolling time-window error-spike detection *(Completed)*
3. **Milestone 3:** Real log-file ingestion (.log and .jsonl) with validation and triage *(Completed)*
4. **Milestone 4:** Deterministic rule-based root cause analysis engine, event clustering, and pattern detection *(Completed)*
5. **Milestone 5:** Telemetry privacy sanitization, upload hardening, filter decoupling, and AppTest E2E suite *(Completed)*
6. **Milestone 6:** Database integration (PostgreSQL / SQLite), Docker containerization, and CI/CD pipelines *(Planned)*
7. **Milestone 7:** Real-time log streaming via OpenTelemetry, WebSockets, or Fluentd *(Planned)*
