"""
TraceRoot AI - Rule-Based Root Cause Analysis (RCA) Engine
(Milestone 4 - Layer 1: Deterministic Offline Correlation & Diagnosis)

This module provides deterministic, evidence-based log correlation, pattern detection,
patient-zero identification, and transparent confidence scoring for distributed systems.

All functions are pure, deterministic, and operate 100% offline without external APIs.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


# Expected log schema columns
REQUIRED_COLUMNS = {"timestamp", "level", "service", "message"}


def _validate_and_prepare_logs(logs_df: Any) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Validate DataFrame structure and return a safe copy with normalized timestamps.

    Parameters
    ----------
    logs_df : Any
        Input DataFrame to validate.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], List[str]]
        A clean copy of the DataFrame (or None if invalid) and a list of validation warnings.
    """
    warnings: List[str] = []

    if logs_df is None or not isinstance(logs_df, pd.DataFrame):
        warnings.append("Input is not a valid pandas DataFrame.")
        return None, warnings

    if logs_df.empty:
        warnings.append("Input DataFrame is empty.")
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS)), warnings

    missing_cols = REQUIRED_COLUMNS - set(logs_df.columns)
    if missing_cols:
        warnings.append(f"Missing required log columns: {sorted(list(missing_cols))}")
        return None, warnings

    # Make an explicit copy to guarantee no mutation of the caller's DataFrame
    clean_df = logs_df.copy()

    # Parse and normalize timestamps to timezone-naive datetime
    try:
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
        # Strip timezone if present to ensure uniform comparisons
        if hasattr(clean_df["timestamp"].dt, "tz") and clean_df["timestamp"].dt.tz is not None:
            clean_df["timestamp"] = clean_df["timestamp"].dt.tz_localize(None)
    except Exception as err:
        warnings.append(f"Error normalizing timestamps: {err}")

    # Check for NaT timestamps
    invalid_ts_count = int(clean_df["timestamp"].isna().sum())
    if invalid_ts_count > 0:
        warnings.append(f"Ignored {invalid_ts_count} record(s) with invalid or unparseable timestamps.")
        clean_df = clean_df.dropna(subset=["timestamp"]).reset_index(drop=True)

    if clean_df.empty:
        warnings.append("No valid records remaining after timestamp validation.")
        return clean_df, warnings

    # Ensure level and service are strings
    clean_df["level"] = clean_df["level"].astype(str).str.upper()
    clean_df["service"] = clean_df["service"].astype(str)
    clean_df["message"] = clean_df["message"].astype(str)

    # Sort chronologically
    clean_df = clean_df.sort_values("timestamp").reset_index(drop=True)
    return clean_df, warnings


def correlate_temporal_events(
    errors_df: pd.DataFrame,
    window_seconds: int = 120,
) -> List[Dict[str, Any]]:
    """Group error events occurring close together into discrete correlated incident clusters.

    Events are grouped where subsequent errors occur within `window_seconds` of the preceding
    error (sliding burst window). This separates incidents separated by hours while clustering
    rapid cascading failures across services.

    Parameters
    ----------
    errors_df : pd.DataFrame
        DataFrame of error records containing 'timestamp', 'level', 'service', and 'message'.
    window_seconds : int, optional
        Maximum gap in seconds between consecutive events to consider them part of the same
        incident cluster. Default is 120 seconds.

    Returns
    -------
    List[Dict[str, Any]]
        List of incident cluster dictionaries, each containing incident metadata,
        involved services, and supporting error records.
    """
    clean_df, _ = _validate_and_prepare_logs(errors_df)
    if clean_df is None or clean_df.empty:
        return []

    # Filter strictly to ERROR records if level is present
    if "level" in clean_df.columns:
        errs = clean_df[clean_df["level"] == "ERROR"].copy()
    else:
        errs = clean_df.copy()

    if errs.empty:
        return []

    errs = errs.sort_values("timestamp").reset_index(drop=True)

    clusters: List[List[Dict[str, Any]]] = []
    current_cluster: List[Dict[str, Any]] = []
    last_timestamp: Optional[pd.Timestamp] = None

    for _, row in errs.iterrows():
        ts = row["timestamp"]
        record = {
            "timestamp": ts,
            "level": row["level"],
            "service": row["service"],
            "message": row["message"],
        }

        if last_timestamp is None:
            current_cluster.append(record)
            last_timestamp = ts
        else:
            delta = (ts - last_timestamp).total_seconds()
            if delta <= window_seconds:
                current_cluster.append(record)
                last_timestamp = ts
            else:
                clusters.append(current_cluster)
                current_cluster = [record]
                last_timestamp = ts

    if current_cluster:
        clusters.append(current_cluster)

    # Transform clusters into structured incident dictionaries
    incidents: List[Dict[str, Any]] = []
    for idx, cluster in enumerate(clusters, start=1):
        start_time = cluster[0]["timestamp"]
        end_time = cluster[-1]["timestamp"]
        services_involved = sorted(list({r["service"] for r in cluster}))
        duration = float((end_time - start_time).total_seconds())

        incidents.append({
            "incident_id": idx,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "error_count": len(cluster),
            "services": services_involved,
            "records": cluster,
        })

    return incidents


def detect_known_failure_patterns(
    errors_df: pd.DataFrame,
    all_logs_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Detect established multi-service failure patterns using log content and temporal correlation.

    Evaluates:
    1. Database Exhaustion Cascade:
       database-proxy connection timeout -> payment-service failure -> api-gateway 500/504
    2. Deployment Regression:
       deployment event followed by an error surge within 5 minutes (300 seconds)
    3. Authentication Cascade:
       auth-service token failure followed by related upstream gateway/service failures

    Parameters
    ----------
    errors_df : pd.DataFrame
        DataFrame of error records.
    all_logs_df : Optional[pd.DataFrame]
        Optional complete logs DataFrame (including INFO and WARNING levels) to capture
        preceding deployment lifecycle events.

    Returns
    -------
    List[Dict[str, Any]]
        List of pattern detection results with matched status, evidence, missing evidence,
        and suspected root cause services.
    """
    clean_errors, _ = _validate_and_prepare_logs(errors_df)
    clean_all, _ = _validate_and_prepare_logs(all_logs_df) if all_logs_df is not None else (clean_errors, [])

    patterns: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Pattern 1: Database Exhaustion Cascade
    # -------------------------------------------------------------------------
    db_pattern: Dict[str, Any] = {
        "pattern_name": "Database Exhaustion Cascade",
        "matched": False,
        "suspected_root_cause_service": "database-proxy",
        "description": (
            "Database proxy connection pool exhaustion or query timeout propagating downstream "
            "to payment/order services, resulting in API gateway HTTP 500/504 responses."
        ),
        "evidence": [],
        "missing_evidence": [],
        "affected_services": [],
        "temporal_order_valid": False,
    }

    if clean_errors is not None and not clean_errors.empty:
        # Search for database-proxy errors
        db_errors = clean_errors[
            (clean_errors["service"].str.contains(r"database|db", case=False)) &
            (clean_errors["message"].str.contains(r"connection timeout|pool|exhausted|database error", case=False))
        ]

        # Search for downstream service errors (payment-service / order-service)
        downstream_errors = clean_errors[
            (clean_errors["service"].str.contains(r"payment|order", case=False)) &
            (clean_errors["message"].str.contains(r"database|timeout|connection", case=False))
        ]

        # Search for API gateway errors (500, 502, 504)
        gateway_errors = clean_errors[
            (clean_errors["service"].str.contains(r"gateway|api", case=False)) &
            (clean_errors["message"].str.contains(r"500|502|504|Internal Server Error|Gateway Timeout", case=False))
        ]

        if not db_errors.empty:
            earliest_db = db_errors["timestamp"].min()
            db_pattern["evidence"].append(
                f"Database proxy error observed at {earliest_db}: '{db_errors.iloc[0]['message']}'"
            )
        else:
            db_pattern["missing_evidence"].append("No database-proxy connection or pool errors found in logs.")

        if not downstream_errors.empty:
            earliest_downstream = downstream_errors["timestamp"].min()
            db_pattern["evidence"].append(
                f"Downstream service '{downstream_errors.iloc[0]['service']}' failed at {earliest_downstream}: "
                f"'{downstream_errors.iloc[0]['message']}'"
            )
        else:
            db_pattern["missing_evidence"].append(
                "No downstream service (payment/order) database failures found."
            )

        if not gateway_errors.empty:
            earliest_gw = gateway_errors["timestamp"].min()
            db_pattern["evidence"].append(
                f"API gateway failure at {earliest_gw}: '{gateway_errors.iloc[0]['message']}'"
            )
        else:
            db_pattern["missing_evidence"].append("No upstream API gateway 500/504 errors found.")

        # Temporal validation: database-proxy error must precede or coincide with downstream errors
        if not db_errors.empty and (not downstream_errors.empty or not gateway_errors.empty):
            earliest_db_ts = db_errors["timestamp"].min()
            other_ts = []
            if not downstream_errors.empty:
                other_ts.append(downstream_errors["timestamp"].min())
            if not gateway_errors.empty:
                other_ts.append(gateway_errors["timestamp"].min())

            min_downstream_ts = min(other_ts)
            # Allow db error to be within 60s window of downstream
            if (earliest_db_ts <= min_downstream_ts) or ((earliest_db_ts - min_downstream_ts).total_seconds() <= 15):
                db_pattern["temporal_order_valid"] = True
                db_pattern["matched"] = True
                services = ["database-proxy"]
                if not downstream_errors.empty:
                    services.extend(downstream_errors["service"].unique().tolist())
                if not gateway_errors.empty:
                    services.extend(gateway_errors["service"].unique().tolist())
                db_pattern["affected_services"] = sorted(list(set(services)))
            else:
                db_pattern["missing_evidence"].append(
                    f"Temporal sequence mismatch: downstream failures ({min_downstream_ts}) occurred significantly "
                    f"before database-proxy failure ({earliest_db_ts})."
                )
    else:
        db_pattern["missing_evidence"].append("No error records available for pattern evaluation.")

    patterns.append(db_pattern)

    # -------------------------------------------------------------------------
    # Pattern 2: Deployment Regression
    # -------------------------------------------------------------------------
    deploy_pattern: Dict[str, Any] = {
        "pattern_name": "Deployment Regression",
        "matched": False,
        "suspected_root_cause_service": "deployment-service",
        "description": (
            "A software deployment or configuration rollout immediately preceded an error surge "
            "within a 5-minute (300-second) operational window."
        ),
        "evidence": [],
        "missing_evidence": [],
        "affected_services": [],
        "temporal_order_valid": False,
    }

    eval_logs = clean_all if clean_all is not None and not clean_all.empty else clean_errors
    if eval_logs is not None and not eval_logs.empty and clean_errors is not None and not clean_errors.empty:
        # Identify deployment events
        deploy_records = eval_logs[
            (eval_logs["service"].str.contains(r"deployment", case=False)) |
            (eval_logs["message"].str.contains(r"deployment (?:started|completed)|deployed|rollout|canary", case=False))
        ]

        if not deploy_records.empty:
            deploy_ev = deploy_records.iloc[-1]
            deploy_ts = deploy_ev["timestamp"]
            deploy_msg = deploy_ev["message"]
            deploy_pattern["evidence"].append(
                f"Deployment event detected at {deploy_ts}: '{deploy_msg}' ({deploy_ev['service']})"
            )

            # Look for subsequent application errors occurring within 300 seconds (5 mins) after deployment
            subsequent_errors = clean_errors[
                (clean_errors["timestamp"] >= deploy_ts) &
                (clean_errors["timestamp"] <= deploy_ts + pd.Timedelta(seconds=300)) &
                (clean_errors["service"] != deploy_ev["service"])
            ]

            if len(subsequent_errors) >= 2:
                deploy_pattern["matched"] = True
                deploy_pattern["temporal_order_valid"] = True
                deploy_pattern["evidence"].append(
                    f"{len(subsequent_errors)} error(s) logged within 5 minutes of deployment across services: "
                    f"{sorted(subsequent_errors['service'].unique().tolist())}"
                )
                affected = [deploy_ev["service"]] + subsequent_errors["service"].unique().tolist()
                deploy_pattern["affected_services"] = sorted(list(set(affected)))
            elif len(subsequent_errors) == 1:
                deploy_pattern["missing_evidence"].append(
                    "Only a single isolated error occurred within 5 minutes of deployment; insufficient evidence of regression."
                )
            else:
                deploy_pattern["missing_evidence"].append(
                    "No application errors occurred within 5 minutes after the deployment event."
                )
        else:
            deploy_pattern["missing_evidence"].append(
                "No deployment lifecycle events found in evaluated logs."
            )
    else:
        deploy_pattern["missing_evidence"].append("Insufficient log records to evaluate deployment events.")

    patterns.append(deploy_pattern)

    # -------------------------------------------------------------------------
    # Pattern 3: Authentication Cascade
    # -------------------------------------------------------------------------
    auth_pattern: Dict[str, Any] = {
        "pattern_name": "Authentication Cascade",
        "matched": False,
        "suspected_root_cause_service": "auth-service",
        "description": (
            "Authentication or token verification failures in auth-service cascading upstream into "
            "gateway rejections or dependent service auth errors."
        ),
        "evidence": [],
        "missing_evidence": [],
        "affected_services": [],
        "temporal_order_valid": False,
    }

    if clean_errors is not None and not clean_errors.empty:
        auth_errors = clean_errors[
            (clean_errors["service"].str.contains(r"auth", case=False)) &
            (clean_errors["message"].str.contains(r"token|auth|unreachable|signature|oauth", case=False))
        ]

        upstream_auth_errors = clean_errors[
            (~clean_errors["service"].str.contains(r"auth", case=False)) &
            (clean_errors["message"].str.contains(r"auth|token|unauthorized|401|403|token validation", case=False))
        ]

        if not auth_errors.empty:
            earliest_auth = auth_errors["timestamp"].min()
            auth_pattern["evidence"].append(
                f"Auth service error observed at {earliest_auth}: '{auth_errors.iloc[0]['message']}'"
            )
        else:
            auth_pattern["missing_evidence"].append("No auth-service authentication errors found.")

        if not upstream_auth_errors.empty:
            earliest_upstream = upstream_auth_errors["timestamp"].min()
            auth_pattern["evidence"].append(
                f"Dependent service '{upstream_auth_errors.iloc[0]['service']}' logged auth-related failure at {earliest_upstream}"
            )
        else:
            auth_pattern["missing_evidence"].append("No upstream/dependent services logged cascading auth errors.")

        if not auth_errors.empty and not upstream_auth_errors.empty:
            earliest_auth_ts = auth_errors["timestamp"].min()
            earliest_upstream_ts = upstream_auth_errors["timestamp"].min()
            # Auth error should precede or coincide with upstream failure
            if (earliest_auth_ts <= earliest_upstream_ts) or ((earliest_auth_ts - earliest_upstream_ts).total_seconds() <= 15):
                auth_pattern["matched"] = True
                auth_pattern["temporal_order_valid"] = True
                services = auth_errors["service"].unique().tolist() + upstream_auth_errors["service"].unique().tolist()
                auth_pattern["affected_services"] = sorted(list(set(services)))
            else:
                auth_pattern["missing_evidence"].append(
                    "Temporal mismatch: upstream failures occurred before auth-service reported errors."
                )
    else:
        auth_pattern["missing_evidence"].append("No error records available to evaluate authentication cascade.")

    patterns.append(auth_pattern)

    return patterns


def identify_patient_zero(errors_df: pd.DataFrame) -> Dict[str, Any]:
    """Identify the earliest plausible triggering event in an error dataset.

    Note: In distributed systems, earliest timestamp is a temporal candidate, but does not
    prove causality due to potential logging delays, asynchronous queues, or clock skew.

    Parameters
    ----------
    errors_df : pd.DataFrame
        DataFrame of error records.

    Returns
    -------
    Dict[str, Any]
        Dictionary describing the patient-zero candidate, supporting timestamp, ambiguity status,
        and explicit uncertainty explanation.
    """
    clean_errors, warnings = _validate_and_prepare_logs(errors_df)
    if clean_errors is None or clean_errors.empty:
        return {
            "candidate_service": None,
            "first_seen": None,
            "message": None,
            "explanation": "No valid error records provided to identify an initial failure event.",
            "is_ambiguous": False,
            "uncertainty_warning": "Insufficient error data.",
            "supporting_record": None,
        }

    # Filter to ERROR records
    if "level" in clean_errors.columns:
        errs = clean_errors[clean_errors["level"] == "ERROR"].copy()
    else:
        errs = clean_errors.copy()

    if errs.empty:
        return {
            "candidate_service": None,
            "first_seen": None,
            "message": None,
            "explanation": "Dataset contains no ERROR-level logs.",
            "is_ambiguous": False,
            "uncertainty_warning": "No errors found.",
            "supporting_record": None,
        }

    errs = errs.sort_values("timestamp").reset_index(drop=True)
    min_ts = errs["timestamp"].min()

    # Check for timestamp ambiguity (multiple services failing within 1 second of earliest timestamp)
    window_boundary = min_ts + pd.Timedelta(seconds=1)
    earliest_batch = errs[errs["timestamp"] <= window_boundary]
    unique_services_in_batch = earliest_batch["service"].unique().tolist()

    first_row = errs.iloc[0]
    candidate_service = str(first_row["service"])
    first_seen = first_row["timestamp"]
    first_message = str(first_row["message"])

    is_ambiguous = len(unique_services_in_batch) > 1
    if is_ambiguous:
        uncertainty_warning = (
            f"Timestamp Ambiguity: {len(unique_services_in_batch)} services "
            f"({', '.join(unique_services_in_batch)}) logged initial errors within 1 second of {min_ts}. "
            "Network latency, buffering, or clock skew may have reordered log arrival."
        )
    else:
        uncertainty_warning = (
            "Caution: Earliest recorded timestamp indicates temporal precedence, but does not definitively prove "
            "causality. An unlogged upstream dependency failure or network partition may be the true root cause."
        )

    explanation = (
        f"Service '{candidate_service}' logged the earliest error at {first_seen}: \"{first_message}\". "
        "In distributed architectures, upstream component failures often precede downstream cascading symptoms."
    )

    return {
        "candidate_service": candidate_service,
        "first_seen": first_seen,
        "message": first_message,
        "explanation": explanation,
        "is_ambiguous": is_ambiguous,
        "uncertainty_warning": uncertainty_warning,
        "supporting_record": {
            "timestamp": first_seen,
            "service": candidate_service,
            "message": first_message,
            "level": str(first_row.get("level", "ERROR")),
        },
    }


def calculate_confidence_score(
    candidate: str,
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculate an explainable confidence score from 0.0 to 100.0 for a suspected root cause.

    Weights:
    - Temporal precedence (patient zero / early failure): up to 25.0 points
    - Known failure pattern match: up to 30.0 points
    - Cross-service cascade impact: up to 20.0 points
    - Error recurrence & volume: up to 15.0 points
    - Deductions for missing evidence or ambiguity: -5.0 to -25.0 points

    Never artificially forces confidence to 80% or higher without strong corroboration.
    Low evidence strictly yields low confidence.

    Parameters
    ----------
    candidate : str
        The candidate microservice identifier.
    evidence : Dict[str, Any]
        Evidence dictionary containing:
        - is_patient_zero (bool)
        - pattern_matched (bool)
        - pattern_name (Optional[str])
        - affected_services_count (int)
        - error_count_for_candidate (int)
        - total_incident_errors (int)
        - is_ambiguous (bool)
        - missing_evidence_count (int)
        - has_repeated_messages (bool)

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'score' (float 0-100), 'confidence_level', 'breakdown', and classification.
    """
    is_patient_zero = bool(evidence.get("is_patient_zero", False))
    pattern_matched = bool(evidence.get("pattern_matched", False))
    pattern_name = evidence.get("pattern_name")
    affected_services_count = int(evidence.get("affected_services_count", 1))
    error_count = int(evidence.get("error_count_for_candidate", 1))
    total_errors = int(evidence.get("total_incident_errors", error_count))
    is_ambiguous = bool(evidence.get("is_ambiguous", False))
    missing_evidence_count = int(evidence.get("missing_evidence_count", 0))
    has_repeated_messages = bool(evidence.get("has_repeated_messages", False))

    breakdown: Dict[str, Any] = {}

    # 1. Temporal Precedence (max 25.0 pts)
    if is_patient_zero and not is_ambiguous:
        temp_pts = 25.0
        temp_reason = "Service logged the earliest error with no immediate timestamp ambiguity."
    elif is_patient_zero and is_ambiguous:
        temp_pts = 15.0
        temp_reason = "Service logged an earliest error, but multiple services logged errors within 1 second."
    else:
        temp_pts = 5.0
        temp_reason = "Service experienced errors during the incident, but was not the first to log a failure."
    breakdown["temporal_precedence"] = {
        "points": temp_pts,
        "max_points": 25.0,
        "reason": temp_reason,
    }

    # 2. Known Failure Pattern Match (max 30.0 pts)
    if pattern_matched:
        pat_pts = 30.0
        pat_reason = f"Confirmed match with established failure pattern: '{pattern_name}'."
    else:
        pat_pts = 0.0
        pat_reason = "No established structural failure pattern matched for this service."
    breakdown["pattern_match"] = {
        "points": pat_pts,
        "max_points": 30.0,
        "reason": pat_reason,
    }

    # 3. Cross-Service Cascade Impact (max 20.0 pts)
    if affected_services_count >= 3:
        impact_pts = 20.0
        impact_reason = f"Correlated with cascading failures across {affected_services_count} distinct microservices."
    elif affected_services_count == 2:
        impact_pts = 12.0
        impact_reason = "Correlated with failures across 2 microservices."
    else:
        impact_pts = 4.0
        impact_reason = "Failures isolated solely to this service; no downstream propagation observed."
    breakdown["cross_service_impact"] = {
        "points": impact_pts,
        "max_points": 20.0,
        "reason": impact_reason,
    }

    # 4. Error Recurrence & Volume (max 15.0 pts)
    if has_repeated_messages and error_count >= 2:
        rec_pts = 15.0
        rec_reason = f"Repeated error patterns logged ({error_count} occurrences) indicating sustained failure."
    elif error_count >= 2:
        rec_pts = 10.0
        rec_reason = f"Multiple errors logged ({error_count} events), but messages did not strictly repeat."
    else:
        rec_pts = 3.0
        rec_reason = "Single isolated error event observed for this service."
    breakdown["recurrence_and_volume"] = {
        "points": rec_pts,
        "max_points": 15.0,
        "reason": rec_reason,
    }

    # 5. Penalties and Deductions
    deductions = 0.0
    deduction_reasons = []

    if missing_evidence_count > 0:
        ded_pts = min(missing_evidence_count * 5.0, 15.0)
        deductions += ded_pts
        deduction_reasons.append(f"Missing {missing_evidence_count} corroborating piece(s) of log evidence (-{ded_pts} pts).")

    if is_ambiguous:
        deductions += 5.0
        deduction_reasons.append("Timestamp clustering ambiguity (-5 pts).")

    if total_errors < 2:
        deductions += 10.0
        deduction_reasons.append("Very low error volume across incident dataset (-10 pts).")

    breakdown["penalties"] = {
        "points": -deductions,
        "reasons": deduction_reasons if deduction_reasons else ["No penalties applied."],
    }

    raw_score = (temp_pts + pat_pts + impact_pts + rec_pts) - deductions
    # Bounded strictly between 5.0 and 95.0 (heuristic correlation is never 100% or 0%)
    final_score = float(round(max(5.0, min(95.0, raw_score)), 1))

    if final_score >= 75.0:
        confidence_level = "High"
    elif final_score >= 45.0:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    return {
        "score": final_score,
        "confidence_level": confidence_level,
        "breakdown": breakdown,
        "classification": "Hypothesis (Rule-Based Heuristic)",
    }


def run_rule_based_rca(logs_df: Any) -> Dict[str, Any]:
    """Execute complete deterministic rule-based Root Cause Analysis on an application log dataset.

    Orchestrates:
    1. Log validation and safe preparation (no input mutation).
    2. Extraction and chronological sorting of ERROR events.
    3. Temporal clustering into distinct incident windows.
    4. Multi-service pattern matching (Database Exhaustion, Deployment Regression, Auth Cascade).
    5. Patient-zero candidate identification per incident.
    6. Candidate ranking with transparent confidence scoring and verification checklists.

    Parameters
    ----------
    logs_df : Any
        Input DataFrame of application logs.

    Returns
    -------
    Dict[str, Any]
        Structured findings dictionary:
        {
            "status": "completed" | "invalid_input",
            "total_errors": int,
            "incident_count": int,
            "ranked_causes": List[Dict[str, Any]],
            "correlated_events": List[Dict[str, Any]],
            "patterns_detected": List[Dict[str, Any]],
            "warnings": List[str]
        }
    """
    clean_df, warnings = _validate_and_prepare_logs(logs_df)

    if clean_df is None:
        return {
            "status": "invalid_input",
            "total_errors": 0,
            "incident_count": 0,
            "ranked_causes": [],
            "correlated_events": [],
            "patterns_detected": [],
            "warnings": warnings,
        }

    # Isolate ERROR records
    errors_df = clean_df[clean_df["level"] == "ERROR"].copy() if "level" in clean_df.columns else clean_df.copy()

    if errors_df.empty:
        warnings.append("No ERROR-level records detected in the active log dataset.")
        return {
            "status": "completed",
            "total_errors": 0,
            "incident_count": 0,
            "ranked_causes": [],
            "correlated_events": [],
            "patterns_detected": [],
            "warnings": warnings,
        }

    total_errors = len(errors_df)

    # 1. Temporal Event Correlation
    incidents = correlate_temporal_events(errors_df, window_seconds=120)

    # 2. Known Failure Pattern Detection
    patterns = detect_known_failure_patterns(errors_df, all_logs_df=clean_df)

    # 3. Patient Zero Identification across overall error stream
    patient_zero_info = identify_patient_zero(errors_df)

    # Map matched patterns by suspected service
    matched_patterns = [p for p in patterns if p.get("matched", False)]
    pattern_by_service: Dict[str, Dict[str, Any]] = {
        p["suspected_root_cause_service"]: p for p in matched_patterns
    }

    # Aggregate candidate microservices
    # Candidates originate from:
    # - Matched failure pattern root causes
    # - Patient zero service
    # - Services producing repeated error messages
    candidate_services = set()
    if patient_zero_info.get("candidate_service"):
        candidate_services.add(patient_zero_info["candidate_service"])
    for svc in pattern_by_service.keys():
        candidate_services.add(svc)
    for incident in incidents:
        for svc in incident["services"]:
            candidate_services.add(svc)

    ranked_causes: List[Dict[str, Any]] = []

    for candidate in sorted(list(candidate_services)):
        candidate_errs = errors_df[errors_df["service"] == candidate]
        if candidate_errs.empty:
            continue

        cand_error_count = len(candidate_errs)
        first_cand_ts = candidate_errs["timestamp"].min()

        # Check message repetition
        message_counts = candidate_errs["message"].value_counts()
        has_repeated = bool((message_counts > 1).any())
        top_message = message_counts.index[0]

        is_pz = (candidate == patient_zero_info.get("candidate_service"))
        matched_pat = pattern_by_service.get(candidate)
        has_pattern = matched_pat is not None

        # Determine affected services across all incidents where candidate appears
        related_services = set()
        for inc in incidents:
            if candidate in inc["services"]:
                related_services.update(inc["services"])
        if not related_services:
            related_services.add(candidate)

        # Count missing evidence
        missing_count = len(matched_pat.get("missing_evidence", [])) if matched_pat else 1

        evidence_payload = {
            "is_patient_zero": is_pz,
            "pattern_matched": has_pattern,
            "pattern_name": matched_pat.get("pattern_name") if matched_pat else None,
            "affected_services_count": len(related_services),
            "error_count_for_candidate": cand_error_count,
            "total_incident_errors": total_errors,
            "is_ambiguous": patient_zero_info.get("is_ambiguous", False) if is_pz else False,
            "missing_evidence_count": missing_count,
            "has_repeated_messages": has_repeated,
        }

        confidence = calculate_confidence_score(candidate, evidence_payload)

        # Compile supporting evidence
        evidence_supporting = [
            f"Logged {cand_error_count} error record(s) during the evaluated period.",
            f"First observed failure at {first_cand_ts}: \"{top_message}\".",
        ]
        if is_pz:
            evidence_supporting.append(
                f"Identified as 'Patient Zero' (earliest error timestamp: {first_cand_ts})."
            )
        if has_pattern:
            evidence_supporting.extend(matched_pat.get("evidence", []))
        if has_repeated:
            evidence_supporting.append(
                f"Error message repeats {message_counts.iloc[0]} times, indicating persistent failure state."
            )
        if len(related_services) > 1:
            evidence_supporting.append(
                f"Failures correlated temporally with errors in: {sorted(list(related_services - {candidate}))}."
            )

        # Compile evidence against / uncertainties
        evidence_against = [
            "Correlation does not establish absolute causation; downstream logging delays may mask the true root cause.",
            "External dependencies (hardware, cloud hypervisor, third-party APIs) are not represented in application logs.",
        ]
        if not is_pz:
            evidence_against.append(
                f"Not the earliest service to log an error (earliest service was '{patient_zero_info.get('candidate_service')}')."
            )
        if not has_pattern:
            evidence_against.append("No recognized architectural failure pattern was matched for this service.")
        if matched_pat and matched_pat.get("missing_evidence"):
            evidence_against.extend(matched_pat["missing_evidence"])

        # Prescribe specific troubleshooting steps
        troubleshooting = _generate_troubleshooting_steps(candidate, top_message, matched_pat)

        # Define possible root cause summary
        possible_cause = _generate_cause_summary(candidate, top_message, matched_pat)

        ranked_causes.append({
            "candidate_service": candidate,
            "possible_root_cause": possible_cause,
            "confidence_score": confidence["score"],
            "confidence_level": confidence["confidence_level"],
            "confidence_breakdown": confidence["breakdown"],
            "evidence_supporting": evidence_supporting,
            "evidence_against": evidence_against,
            "affected_services": sorted(list(related_services)),
            "first_observed_timestamp": str(first_cand_ts),
            "recommended_verification_steps": troubleshooting,
            "classification": "Hypothesis (Rule-Based Heuristic)",
        })

    # Sort candidates by confidence score descending, then by error count
    ranked_causes.sort(key=lambda x: (x["confidence_score"], len(x["evidence_supporting"])), reverse=True)

    return {
        "status": "completed",
        "total_errors": total_errors,
        "incident_count": len(incidents),
        "ranked_causes": ranked_causes,
        "correlated_events": incidents,
        "patterns_detected": patterns,
        "warnings": warnings,
    }


def _generate_cause_summary(
    service: str,
    top_message: str,
    pattern: Optional[Dict[str, Any]],
) -> str:
    """Generate a clear, human-readable description of the suspected root cause."""
    if pattern and pattern.get("matched"):
        return f"{pattern['pattern_name']}: {pattern['description']}"

    lower_msg = top_message.lower()
    if "database" in lower_msg or "connection pool" in lower_msg or "timeout" in lower_msg:
        return f"Database connection pool exhaustion or query latency within {service}."
    elif "token" in lower_msg or "auth" in lower_msg:
        return f"Authentication service provider timeout or token verification failure in {service}."
    elif "deployment" in lower_msg or "rollback" in lower_msg:
        return f"Recent deployment rollout or configuration regression initiated by {service}."
    else:
        return f"Localized application failure or exception surge within {service} ({top_message})."


def _generate_troubleshooting_steps(
    service: str,
    top_message: str,
    pattern: Optional[Dict[str, Any]],
) -> List[str]:
    """Generate actionable, context-aware troubleshooting recommendations for SREs."""
    steps = []

    if pattern and pattern.get("pattern_name") == "Database Exhaustion Cascade":
        steps.extend([
            f"Inspect active connection count and pool limits on {service} and primary database instance.",
            "Review database slow-query logs for long-running unindexed transactions during the incident window.",
            "Verify whether read-replica lag or primary failover triggered connection throttling.",
        ])
    elif pattern and pattern.get("pattern_name") == "Deployment Regression":
        steps.extend([
            f"Review git commit diff and configuration changes deployed by {service}.",
            "Initiate immediate canary rollback or toggle associated feature flags.",
            "Check service pod healthchecks and container restart counts in the cluster.",
        ])
    elif pattern and pattern.get("pattern_name") == "Authentication Cascade":
        steps.extend([
            f"Inspect identity provider (IdP) / OAuth server reachability from {service}.",
            "Verify token signing certificate validity and rotation timestamps.",
            "Check authorization cache hit rates and Redis/memory health on auth-service.",
        ])
    else:
        steps.extend([
            f"Inspect detailed stack traces and debug logs for {service} around incident onset.",
            f"Check CPU, memory, and thread utilization metrics for {service} pods.",
            "Verify upstream network connectivity and gateway routing health to this service.",
        ])

    return steps

