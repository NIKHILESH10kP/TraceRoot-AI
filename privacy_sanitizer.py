"""
TraceRoot AI - Privacy & Data Sanitization Module
(Milestone 4 - Task 3A.1: Pre-AI Evidence Sanitization & Redaction Engine)

This module provides privacy sanitization and credential redaction for structured
Root Cause Analysis (RCA) findings before any external AI processing.

Guarantees:
- Completely offline (standard library regex and copy only).
- Deep copy of all structures (never mutates original RCA results).
- Redacts PII, network addresses, authorization credentials, and secrets.
- Preserves technical incident context (services, failure patterns, metrics, safe timestamps).
- Supports universal URI schemes, snake_case credentials, and collision-safe dictionary key sanitization.
"""

import copy
from datetime import date, datetime
import json
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ===========================================================================
# Regex Redaction Patterns
# ===========================================================================

# 1. Email addresses (e.g. user@domain.com, john.doe+alert@sub.company.org)
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# 2. IPv4 Addresses (with optional port, e.g. 192.168.1.1, 10.0.0.5:8080)
# Strictly matches valid 4-octet ranges (0-255) to avoid version numbers like v2.4.1
IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(?::\d{1,5})?\b"
)

# 3. IPv6 Addresses (e.g. 2001:0db8:85a3::8a2e:0370:7334, fe80::1, ::1)
# Requires either '::' or at least 3 colons to prevent false positives on timestamps like 10:15:30
IPV6_PATTERN = re.compile(
    r"(?:::1\b)|"
    r"(?:\b[0-9a-fA-F]{1,4}(?:::[0-9a-fA-F]{1,4}|::[0-9a-fA-F]{0,4}))\b|"
    r"(?:\b(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}\b)"
)

# 4. Universal URL & Connection-String Embedded Credentials (RFC 3986 URI format)
# Supports any protocol scheme (amqp://, mongodb+srv://, kafka://, postgresql://, redis://, https://, etc.)
# Preserves the scheme and host while redacting username and password.
URL_CREDENTIALS_PATTERN = re.compile(
    r"((?:[a-zA-Z][a-zA-Z0-9+.-]*:)?//)"
    r"([^:\s/@]+):([^@\s/]+)@"
)

# 5. Generic Authorization Headers & Tokens
# Matches Authorization: <token>, authorization=<token>, Authorization: Bearer <token>, Authorization: Basic <token>
AUTH_HEADER_PATTERN = re.compile(
    r"(?i)\b((?:proxy[_-]?)?authorization\s*[:=]\s*)(['\"]?)(?:(bearer\s+)|(basic\s+))?([^\s,'\"}&;]+)(\2)"
)

# 6. Standalone Bearer & Basic Auth headers (without Authorization: prefix)
BEARER_TOKEN_PATTERN = re.compile(
    r"(?i)\b(bearer\s+)[A-Za-z0-9\-._~+/=]{8,}\b"
)
BASIC_AUTH_PATTERN = re.compile(
    r"(?i)\b(basic\s+)[A-Za-z0-9+/=]{8,}\b"
)

# 7. JSON Web Tokens (JWT: 3 base64 segments separated by dots)
JWT_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
)

# 8. Well-known API Key and Secret Patterns
# Google AI Studio, OpenAI, AWS Access Key, GitHub token, Slack token
KNOWN_API_KEYS_PATTERN = re.compile(
    r"\b(?:"
    r"AIza[0-9A-Za-z\-_]{35}|"                       # Google API Key
    r"sk-(?:live-|test-|ant-)?[A-Za-z0-9_-]{20,}|"   # OpenAI / Anthropic Key
    r"AKIA[0-9A-Z]{16}|"                             # AWS Access Key ID
    r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}|"    # GitHub Token
    r"xox[baprs]-[A-Za-z0-9-]{10,}"                 # Slack Token
    r")\b"
)

# 9. Key-Value Secret Patterns (supports snake_case prefixes, e.g. api_token, db_password, jwt_secret)
# Supports quoted and unquoted keys, e.g. {"password": "secret"}, {'token': 'val'}, access_key=...
# Detects crypto and auth keys (secret_key, access_key, signing_key, encryption_key, auth_key, master_key)
# Stops at query parameter and header separators (& and ;) to prevent swallowing subsequent parameters.
KV_SECRET_PATTERN = re.compile(
    r"(?i)(['\"]?)\b("
    r"(?:[a-z0-9_-]+[_-])?(?:pass(?:word)?|passwd|pwd|secret|api[_-]?key|apikey|private[_-]?key|credential(?:s)?|token)|"
    r"(?:[a-z0-9_-]+[_-])?(?:secret|access|signing|encryption|auth|master)[_-]?key(?:[_-]?id)?|"
    r"client[_-]?secret"
    r")\1\s*([:=]\s*)(?:(['\"])(.*?)\4|([^\s,'\"}&;]+))"
)

# 10. Payment Card Numbers (16-digit sequences or 4x4 groups)
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[ -]){3}\d{4}\b|\b\d{16}\b"
)

# Sensitive dictionary key substrings to check case-insensitively
SENSITIVE_KEY_SUBSTRINGS = (
    "password", "passwd", "pwd",
    "secret",
    "token",
    "api_key", "apikey",
    "private_key",
    "credential",
    "authorization",
    "access_key", "accesskey",
    "signing_key", "signingkey",
    "encryption_key", "encryptionkey",
    "auth_key", "authkey",
    "master_key", "masterkey",
)

# Non-sensitive metrics keys that might contain "token" (e.g. usage counters)
EXCLUDED_NON_SENSITIVE_KEYS = {
    "token_count", "total_tokens", "max_tokens", "prompt_tokens", "completion_tokens",
}


# ===========================================================================
# Pure Sanitization Functions
# ===========================================================================

def _redact_auth_header(match: re.Match) -> str:
    """Helper to redact Authorization headers while preserving scheme."""
    prefix = match.group(1)      # e.g. "Authorization: "
    quote_open = match.group(2) or ""
    is_bearer = match.group(3)   # "Bearer " if present
    is_basic = match.group(4)    # "Basic " if present
    quote_close = match.group(6) or quote_open

    if is_bearer:
        scheme = is_bearer
    elif is_basic:
        scheme = is_basic
    else:
        scheme = ""

    return f"{prefix}{quote_open}{scheme}[REDACTED_TOKEN]{quote_close}"


def _redact_kv_secret(match: re.Match) -> str:
    """Helper to redact key-value secret fields while preserving quotes and delimiters."""
    quote_key = match.group(1) or ""
    key_name = match.group(2)
    delim = match.group(3)
    quote_val = match.group(4) or ""
    return f"{quote_key}{key_name}{quote_key}{delim}{quote_val}[REDACTED_SECRET]{quote_val}"


def sanitize_text(text: str) -> str:
    """Sanitize a text string by redacting sensitive data patterns.

    Detects and masks:
    - Universal URL credentials (amqp, mongodb+srv, kafka, postgresql, redis, https)
    - Generic and scheme-specific Authorization headers (Bearer, Basic, raw token)
    - Snake_case and kebab-case secrets (api_token=..., db_password=..., jwt_secret=..., admin_password=...)
    - Quoted JSON/dict key-value secrets ({"password": "..."}, {'token': '...'})
    - Cryptographic and auth keys (secret_key, access_key, signing_key, encryption_key, auth_key, master_key)
    - Query parameters stopping at '&' and ';' delimiters
    - Email addresses -> [REDACTED_EMAIL]
    - IPv4 and IPv6 addresses -> [REDACTED_IP]
    - JWT tokens -> [REDACTED_JWT_TOKEN]
    - Well-known API keys -> [REDACTED_API_KEY]
    - Payment card numbers -> [REDACTED_CARD]

    Preserves:
    - Microservice names (e.g. payment-service, database-proxy)
    - Semantic error descriptions (e.g. timeout, connection pool exhausted, HTTP 500)
    - Safe timestamps (e.g. 2026-10-27 10:15:30)
    - Version strings (e.g. v2.4.1)
    - Unrelated URL query parameters (e.g. &user_id=100)
    - Non-sensitive token counters (e.g. token_count, total_tokens, prompt_tokens)

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        Sanitized string with sensitive tokens masked.
    """
    if not isinstance(text, str) or not text:
        return text

    sanitized = text

    # 1. Universal URL credentials (preserves scheme & host, redacts user:pass)
    sanitized = URL_CREDENTIALS_PATTERN.sub(
        r"\g<1>[REDACTED_USER]:[REDACTED_PASSWORD]@", sanitized
    )

    # 2. Authorization headers (Authorization: secret, Authorization: Bearer secret, etc.)
    sanitized = AUTH_HEADER_PATTERN.sub(_redact_auth_header, sanitized)

    # 3. JWT Tokens (3 base64 segments)
    sanitized = JWT_PATTERN.sub("[REDACTED_JWT_TOKEN]", sanitized)

    # 4. Known API Keys (Google, OpenAI, AWS, GitHub, Slack)
    sanitized = KNOWN_API_KEYS_PATTERN.sub("[REDACTED_API_KEY]", sanitized)

    # 5. Standalone Authorization headers (Bearer / Basic without Authorization: prefix)
    sanitized = BEARER_TOKEN_PATTERN.sub(r"\g<1>[REDACTED_TOKEN]", sanitized)
    sanitized = BASIC_AUTH_PATTERN.sub(r"\g<1>[REDACTED_TOKEN]", sanitized)

    # 6. Key-value secrets with snake_case support, quoted JSON keys, and query param bounds
    sanitized = KV_SECRET_PATTERN.sub(_redact_kv_secret, sanitized)

    # 7. Email addresses
    sanitized = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)

    # 8. IPv4 addresses
    sanitized = IPV4_PATTERN.sub("[REDACTED_IP]", sanitized)

    # 9. IPv6 addresses
    sanitized = IPV6_PATTERN.sub("[REDACTED_IP]", sanitized)

    # 10. Credit card numbers
    sanitized = CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", sanitized)

    return sanitized


def is_sensitive_key(key: str) -> bool:
    """Check if a dictionary key name indicates sensitive or secret data.

    Detects names such as api_token, db_password, jwt_secret, client_secret, private_key,
    access_key, signing_key, encryption_key, auth_key, master_key.
    Supports snake_case, kebab-case, and case-insensitive matching.
    Excludes non-sensitive counter metrics like token_count.
    """
    if not isinstance(key, str):
        return False
    lower_key = key.lower().strip()
    if lower_key in EXCLUDED_NON_SENSITIVE_KEYS:
        return False
    normalized = lower_key.replace("-", "_")
    return any(term in lower_key or term in normalized for term in SENSITIVE_KEY_SUBSTRINGS)


def sanitize_data_structure(data: Any) -> Tuple[Any, Set[str]]:
    """Recursively traverse and sanitize an arbitrary Python data structure.

    Protects:
    - Dict values (and sanitizes dict keys with collision avoidance)
    - Lists and tuples
    - Sets and frozensets (safely converted without in-place mutation)

    Parameters
    ----------
    data : Any
        Dict, list, set, primitive, or string to sanitize.

    Returns
    -------
    Tuple[Any, Set[str]]
        A tuple of (sanitized_data, detected_categories_set).
    """
    detected_categories: Set[str] = set()

    if isinstance(data, str):
        sanitized = sanitize_text(data)
        if sanitized != data:
            if "[REDACTED_EMAIL]" in sanitized:
                detected_categories.add("Email Address")
            if "[REDACTED_IP]" in sanitized:
                detected_categories.add("IP Address")
            if "[REDACTED_TOKEN]" in sanitized or "[REDACTED_JWT_TOKEN]" in sanitized:
                detected_categories.add("Authorization Token")
            if "[REDACTED_API_KEY]" in sanitized:
                detected_categories.add("API Key")
            if "[REDACTED_SECRET]" in sanitized:
                detected_categories.add("Password / Secret")
            if "[REDACTED_CARD]" in sanitized:
                detected_categories.add("Payment Card")
            if "[REDACTED_USER]" in sanitized:
                detected_categories.add("URL Credentials")
        return sanitized, detected_categories

    elif isinstance(data, dict):
        sanitized_dict: Dict[Any, Any] = {}
        for k, v in data.items():
            k_str = str(k)
            # 1. Sanitize the key itself
            sanitized_k = sanitize_text(k_str)
            if sanitized_k != k_str:
                detected_categories.add("Sensitive Dictionary Key")

            # 2. Check if key name indicates secret data
            if is_sensitive_key(k_str):
                sanitized_v = "[REDACTED_SECRET]"
                detected_categories.add("Sensitive Dictionary Key")
            else:
                sanitized_v, sub_categories = sanitize_data_structure(v)
                detected_categories.update(sub_categories)

            # 3. Collision Prevention: avoid silently overwriting data if two keys sanitize to identical string
            if sanitized_k not in sanitized_dict:
                sanitized_dict[sanitized_k] = sanitized_v
            else:
                collision_idx = 2
                disambiguated_k = f"{sanitized_k}_{collision_idx}"
                while disambiguated_k in sanitized_dict:
                    collision_idx += 1
                    disambiguated_k = f"{sanitized_k}_{collision_idx}"
                sanitized_dict[disambiguated_k] = sanitized_v

        return sanitized_dict, detected_categories

    elif isinstance(data, list):
        sanitized_list = []
        for item in data:
            sanitized_item, sub_categories = sanitize_data_structure(item)
            sanitized_list.append(sanitized_item)
            detected_categories.update(sub_categories)
        return sanitized_list, detected_categories

    elif isinstance(data, tuple):
        sanitized_items = []
        for item in data:
            sanitized_item, sub_categories = sanitize_data_structure(item)
            sanitized_items.append(sanitized_item)
            detected_categories.update(sub_categories)
        return tuple(sanitized_items), detected_categories

    elif isinstance(data, (datetime, date)):
        # Converts Python datetime, date, and Pandas Timestamp to ISO 8601 strings
        # Avoids Pandas dependency by relying on standard library datetime/date inheritance
        if str(data) == "NaT":
            return None, detected_categories
        return data.isoformat(), detected_categories

    elif isinstance(data, (set, frozenset)):
        sanitized_set = set()
        for item in data:
            sanitized_item, sub_categories = sanitize_data_structure(item)
            sanitized_set.add(sanitized_item)
            detected_categories.update(sub_categories)
        return (sanitized_set if isinstance(data, set) else frozenset(sanitized_set)), detected_categories

    else:
        # Ints, floats, booleans, None, and timestamps pass through safely
        return data, detected_categories


# ===========================================================================
# Strict Schema Allowlist & Data Minimization Constants
# ===========================================================================

MAX_PAYLOAD_SIZE_BYTES = 64 * 1024  # 64 KB hard cap
MAX_RANKED_CAUSES = 5
MAX_INCIDENTS = 5
MAX_PATTERNS = 5
MAX_WARNINGS = 10
MAX_RECORDS_PER_INCIDENT_CAP = 10
DEFAULT_MAX_RECORDS_PER_INCIDENT = 5
MAX_EVIDENCE_ITEMS = 10
MAX_SERVICES_COUNT = 20
MAX_TEXT_LENGTH = 500
MAX_IDENTIFIER_LENGTH = 64
MAX_TIMESTAMP_LENGTH = 35
MAX_NESTING_DEPTH = 7

ALLOWED_TOP_LEVEL_KEYS = {
    "status",
    "validation_passed",
    "total_errors",
    "incident_count",
    "ranked_causes",
    "correlated_events",
    "patterns_detected",
    "warnings",
    "sanitization_info",
}

ALLOWED_CAUSE_KEYS = {
    "candidate_service",
    "possible_root_cause",
    "confidence_score",
    "confidence_level",
    "confidence_breakdown",
    "evidence_supporting",
    "evidence_against",
    "affected_services",
    "first_observed_timestamp",
    "recommended_verification_steps",
    "patient_zero_candidate",
    "classification",
}

ALLOWED_INCIDENT_KEYS = {
    "incident_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "primary_service",
    "services",
    "error_count",
}

# Note: Raw records, stack traces, and messages are completely excluded from AI payloads.
# ALLOWED_RECORD_KEYS is retained for internal reference only.
ALLOWED_RECORD_KEYS = {
    "timestamp",
    "service",
    "level",
    "message",
}

ALLOWED_PATTERN_KEYS = {
    "pattern_name",
    "matched",
    "description",
    "evidence",
}

# Strict Service Identifier regex: Starts with alphanumeric, followed by up to 63 chars of [A-Za-z0-9_.-]
SERVICE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

# Strict Incident Identifier regex: Starts with alphanumeric, followed by up to 31 chars of [A-Za-z0-9_.-]
INCIDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")


def is_valid_service_identifier(service_name: Any) -> bool:
    """Validate service identifier format against a strict alphanumeric allowlist.

    Allowed format:
    - Starts with an alphanumeric character [A-Za-z0-9]
    - Followed by up to 63 characters consisting of [A-Za-z0-9_.-]
    - Total length 1 to 64 characters
    - Rejects whitespace, newlines, control characters, quotes, and prompt injection
    """
    if not isinstance(service_name, str) or isinstance(service_name, bool):
        return False
    if len(service_name) < 1 or len(service_name) > MAX_IDENTIFIER_LENGTH:
        return False
    return bool(SERVICE_IDENTIFIER_PATTERN.fullmatch(service_name))


def is_valid_incident_id(val: Any) -> Tuple[bool, str]:
    """Validate and normalize an incident identifier.

    Accepts:
    - Non-negative integers (e.g. 1, 2, 42) -> normalized to str (e.g. "1")
    - Alphanumeric strings matching ^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$ (e.g. "INC-101", "inc_01")

    Rejects:
    - Booleans (True, False)
    - Floats, None, lists, dicts, unexpected objects
    - Strings with whitespace, newlines, tabs, quotes, control characters
    - Prompt injection phrases (e.g. 'SYSTEM: OVERRIDE INSTRUCTIONS')
    - Overlong strings (>32 chars) or empty strings

    Returns:
    - Tuple[bool, str]: (is_valid, normalized_incident_id_str)
    """
    if val is None or isinstance(val, bool):
        return False, ""

    if isinstance(val, int):
        if val < 0 or val > 999_999_999:
            return False, ""
        return True, str(val)

    if isinstance(val, str):
        if not val or len(val) > 32 or len(val) < 1:
            return False, ""
        if not INCIDENT_ID_PATTERN.fullmatch(val):
            return False, ""
        return True, val

    return False, ""


def validate_and_normalize_timestamp(val: Any) -> Tuple[bool, str]:
    """Validate and normalize a timestamp to standard ISO-8601 format.

    Accepts:
    - Valid ISO-8601 strings (e.g., '2026-10-27T10:15:30', '2026-10-27 10:15:30',
      '2026-10-27T10:15:30Z', '2026-10-27T10:15:30.123456+00:00', '2026-10-27')
    - Python datetime or date objects (including pandas Timestamps)

    Rejects:
    - Arbitrary text or prompt injection attempts
    - Malformed dates (e.g. '2026-02-31', 'invalid-date')
    - Control characters, tabs, newlines, or strings exceeding 35 characters
    - Leading or trailing whitespace
    - Non-string / non-datetime types (bool, int, float, list, dict, None)
    - NaT (Not-a-Time) objects or strings

    Returns:
    - Tuple[bool, str]: (is_valid, normalized_iso_string)
    """
    if val is None or isinstance(val, bool):
        return False, ""

    # Check for pandas/numpy NaT or string "NaT"
    str_val = str(val)
    if str_val == "NaT" or "NaT" in str_val:
        return False, ""

    if isinstance(val, datetime):
        try:
            return True, val.isoformat()
        except Exception:
            return False, ""

    if isinstance(val, date):
        try:
            return True, val.isoformat()
        except Exception:
            return False, ""

    if isinstance(val, str):
        if not val or len(val) > MAX_TIMESTAMP_LENGTH or len(val) < 8:
            return False, ""
        if val != val.strip():
            return False, ""
        if any(ord(c) < 32 or ord(c) == 127 for c in val):
            return False, ""

        try:
            dt = datetime.fromisoformat(val)
            return True, dt.isoformat()
        except Exception:
            return False, ""

    return False, ""


def _check_nesting_depth(data: Any, current_depth: int = 1, visited: Optional[Set[int]] = None) -> bool:
    """Verify that structure depth does not exceed MAX_NESTING_DEPTH and has no cycles."""
    if visited is None:
        visited = set()

    if current_depth > MAX_NESTING_DEPTH:
        return False

    obj_id = id(data)
    if isinstance(data, (dict, list, tuple, set, frozenset)):
        if obj_id in visited:
            return False  # Circular reference detected
        visited.add(obj_id)

    if isinstance(data, dict):
        for k, v in data.items():
            if not _check_nesting_depth(v, current_depth + 1, visited):
                return False
    elif isinstance(data, (list, tuple, set, frozenset)):
        for item in data:
            if not _check_nesting_depth(item, current_depth + 1, visited):
                return False

    if isinstance(data, (dict, list, tuple, set, frozenset)):
        visited.remove(obj_id)

    return True


def _make_rejection_payload(reason: str, original_warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    """Construct a standardized, fail-closed rejection payload.

    Security guarantees:
    - Never echoes unredacted raw input; applies sanitize_text() to reason and warnings.
    - Limits string lengths to 200 characters and warnings count to MAX_WARNINGS.
    - Explicitly sets is_sanitized=False and eligible_for_transmission=False.
    - Prefers fixed error codes/messages instead of interpolating arbitrary input.
    """
    clean_reason = sanitize_text(str(reason))[:200]
    warnings: List[str] = []
    if original_warnings and isinstance(original_warnings, list):
        for w in original_warnings[:MAX_WARNINGS]:
            if isinstance(w, str):
                warnings.append(sanitize_text(w)[:200])
            else:
                warnings.append(sanitize_text(str(w))[:200])

    formatted_msg = f"Payload validation failed: {clean_reason}"
    if len(warnings) < MAX_WARNINGS:
        warnings.append(formatted_msg)
    elif warnings:
        warnings[-1] = formatted_msg

    return {
        "status": "rejected",
        "validation_passed": False,
        "rejection_reason": clean_reason,
        "total_errors": 0,
        "incident_count": 0,
        "ranked_causes": [],
        "correlated_events": [],
        "patterns_detected": [],
        "warnings": warnings,
        "sanitization_info": {
            "is_sanitized": False,
            "eligible_for_transmission": False,
            "redacted_categories": [],
            "status": "validation_rejected",
        },
    }


def _clean_str(val: Any, max_len: int, categories: Set[str]) -> str:
    """Convert to string, truncate to max_len, sanitize, and record categories."""
    if val is None:
        return ""
    if isinstance(val, (datetime, date)):
        str_val = str(val) if str(val) == "NaT" else val.isoformat()
    else:
        str_val = str(val)

    if len(str_val) > max_len:
        str_val = str_val[:max_len]

    sanitized, cats = sanitize_data_structure(str_val)
    categories.update(cats)
    return sanitized


def sanitize_rca_result(rca_result: Any) -> Dict[str, Any]:
    """Sanitize complete structured findings returned by `run_rule_based_rca()`.

    Guarantees:
    - Never modifies the input `rca_result` (uses deep copy).
    - Returns a safe, fully sanitized dictionary structure.
    - Handles None, empty dicts, or malformed inputs without exceptions.
    - Invalid inputs fail closed with is_sanitized=False.
    - Adds `sanitization_info` metadata describing what categories were redacted.

    Parameters
    ----------
    rca_result : Any
        The dictionary returned by `rca_engine.run_rule_based_rca()`.

    Returns
    -------
    Dict[str, Any]
        Sanitized copy of the RCA findings dictionary with metadata.
    """
    if rca_result is None or not isinstance(rca_result, dict):
        return {
            "status": "invalid_input",
            "total_errors": 0,
            "incident_count": 0,
            "ranked_causes": [],
            "correlated_events": [],
            "patterns_detected": [],
            "warnings": ["Input to privacy sanitizer was not a valid dictionary."],
            "sanitization_info": {
                "is_sanitized": False,
                "eligible_for_transmission": False,
                "redacted_categories": [],
                "status": "rejected_invalid_input",
            },
        }

    # Deep copy ensures zero mutation of caller's data
    try:
        data_copy = copy.deepcopy(rca_result)
    except Exception:
        return {
            "status": "invalid_input",
            "total_errors": 0,
            "incident_count": 0,
            "ranked_causes": [],
            "correlated_events": [],
            "patterns_detected": [],
            "warnings": ["Input structure could not be safely duplicated."],
            "sanitization_info": {
                "is_sanitized": False,
                "eligible_for_transmission": False,
                "redacted_categories": [],
                "status": "rejected_invalid_input",
            },
        }

    sanitized_data, categories = sanitize_data_structure(data_copy)

    # Attach sanitization audit metadata
    sanitized_data["sanitization_info"] = {
        "is_sanitized": True,
        "eligible_for_transmission": sanitized_data.get("status") == "completed",
        "redacted_categories": sorted(list(categories)),
        "redactions_applied": len(categories) > 0,
        "status": "sanitized",
    }

    return sanitized_data


def prepare_ai_payload(
    rca_result: Any,
    max_records_per_incident: int = 5,
) -> Dict[str, Any]:
    """Prepare a compact, sanitized summary payload tailored for external AI prompts.

    Guarantees:
    - Strict schema allowlisting: only explicitly approved keys and types survive.
    - Completely excludes raw log records, messages, stack traces, and headers.
    - Preserves high-level incident metadata (timestamps, duration, services, error count).
    - Drops raw DataFrames, uploaded file contents, session state, and unknown fields.
    - Deep copy of input ensures zero mutation of caller's RCA result; guarded against exceptions.
    - Enforces recursion depth and bounds to prevent denial-of-service.
    - Sanitizes all string fields through the credential and PII redaction engine.
    - Strictly validates all numeric fields against NaN, Infinity, and type violations (RFC 8259 JSON).
    - Never reflects untrusted user input into rejection reasons or warnings.
    - Enforces strict list limits and a 64 KB total serialized payload size limit.
    - Fails closed on invalid or malformed inputs (never marks invalid inputs as sanitized).

    Parameters
    ----------
    rca_result : Any
        Structured RCA result dictionary.
    max_records_per_incident : int, optional
        Deprecated; raw records are excluded from external transmission payloads.

    Returns
    -------
    Dict[str, Any]
        Compact, privacy-safe, allowlisted dictionary ready for AI prompt conversion.
    """
    # 1. Type validation: Input must be a dictionary
    if rca_result is None or not isinstance(rca_result, dict):
        return _make_rejection_payload("Input must be a valid dictionary")

    # 2. Immutability & Defensive Copying: Deep copy before any processing
    try:
        input_copy = copy.deepcopy(rca_result)
    except Exception:
        return _make_rejection_payload("Input structure could not be duplicated")

    orig_warnings = input_copy.get("warnings") if isinstance(input_copy.get("warnings"), list) else None

    # 3. Recursion & depth defense
    if not _check_nesting_depth(input_copy):
        return _make_rejection_payload(
            "Input structure exceeds maximum nesting depth or contains cyclical references",
            orig_warnings,
        )

    # 4. Strict status check: Must be 'completed' (fixed safe message, no input reflection)
    status = input_copy.get("status")
    if status != "completed":
        return _make_rejection_payload("Field 'status' must be 'completed'", orig_warnings)

    # 5. Numerical integrity
    total_errors = input_copy.get("total_errors")
    incident_count = input_copy.get("incident_count")
    if not isinstance(total_errors, int) or isinstance(total_errors, bool) or total_errors < 0:
        return _make_rejection_payload("Field 'total_errors' must be a non-negative integer", orig_warnings)
    if not isinstance(incident_count, int) or isinstance(incident_count, bool) or incident_count < 0:
        return _make_rejection_payload("Field 'incident_count' must be a non-negative integer", orig_warnings)

    categories: Set[str] = set()

    # 6. Filter and construct 'ranked_causes' (allowlist & finite numeric enforcement)
    raw_causes = input_copy.get("ranked_causes", [])
    if not isinstance(raw_causes, list):
        return _make_rejection_payload("Field 'ranked_causes' must be a list", orig_warnings)

    for cause in raw_causes:
        if isinstance(cause, dict):
            raw_cand = cause.get("candidate_service")
            if not is_valid_service_identifier(raw_cand):
                return _make_rejection_payload("Field 'candidate_service' contains an invalid service identifier", orig_warnings)

            raw_affected = cause.get("affected_services", [])
            if raw_affected is not None and not isinstance(raw_affected, list):
                return _make_rejection_payload("Field 'affected_services' must be a list", orig_warnings)
            if isinstance(raw_affected, list):
                for svc in raw_affected:
                    if not is_valid_service_identifier(svc):
                        return _make_rejection_payload("Field 'affected_services' contains an invalid service identifier", orig_warnings)

            if "first_observed_timestamp" in cause:
                raw_first_ts = cause["first_observed_timestamp"]
                is_valid_ts, _ = validate_and_normalize_timestamp(raw_first_ts)
                if not is_valid_ts:
                    return _make_rejection_payload("Field 'first_observed_timestamp' contains an invalid timestamp", orig_warnings)

    clean_causes: List[Dict[str, Any]] = []
    for cause in raw_causes[:MAX_RANKED_CAUSES]:
        if not isinstance(cause, dict):
            continue

        cand_svc = _clean_str(cause.get("candidate_service", ""), MAX_IDENTIFIER_LENGTH, categories)
        possible_cause = _clean_str(cause.get("possible_root_cause", ""), MAX_TEXT_LENGTH, categories)

        if "confidence_score" in cause:
            raw_score = cause["confidence_score"]
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)) or math.isnan(raw_score) or math.isinf(raw_score):
                return _make_rejection_payload("Field 'confidence_score' must be a finite number between 0 and 100", orig_warnings)
            conf_score = float(raw_score)
            if conf_score < 0.0 or conf_score > 100.0:
                return _make_rejection_payload("Field 'confidence_score' must be between 0 and 100", orig_warnings)
        else:
            conf_score = 0.0

        conf_level = _clean_str(cause.get("confidence_level", "Unknown"), 32, categories)

        raw_breakdown = cause.get("confidence_breakdown", {})
        clean_breakdown: Dict[str, Any] = {}
        if isinstance(raw_breakdown, dict):
            for bk, bv in raw_breakdown.items():
                if isinstance(bk, str):
                    if isinstance(bv, bool) or (isinstance(bv, (int, float)) and (math.isnan(bv) or math.isinf(bv))):
                        return _make_rejection_payload("Values in 'confidence_breakdown' must be finite numbers", orig_warnings)
                    if isinstance(bv, (int, float)):
                        clean_breakdown[_clean_str(bk, 32, categories)] = float(bv)
                    elif isinstance(bv, dict):
                        clean_sub: Dict[str, Any] = {}
                        for sub_k, sub_v in bv.items():
                            if isinstance(sub_k, str):
                                if isinstance(sub_v, bool) or (isinstance(sub_v, (int, float)) and (math.isnan(sub_v) or math.isinf(sub_v))):
                                    return _make_rejection_payload("Values in 'confidence_breakdown' must be finite numbers", orig_warnings)
                                if isinstance(sub_v, (int, float)):
                                    clean_sub[_clean_str(sub_k, 32, categories)] = float(sub_v)
                                elif isinstance(sub_v, str):
                                    clean_sub[_clean_str(sub_k, 32, categories)] = _clean_str(sub_v, 100, categories)
                        clean_breakdown[_clean_str(bk, 32, categories)] = clean_sub
                    elif isinstance(bv, str):
                        clean_breakdown[_clean_str(bk, 32, categories)] = _clean_str(bv, 100, categories)

        raw_ev_supp = cause.get("evidence_supporting", [])
        clean_ev_supp: List[str] = []
        if isinstance(raw_ev_supp, list):
            for item in raw_ev_supp[:MAX_EVIDENCE_ITEMS]:
                clean_ev_supp.append(_clean_str(item, MAX_TEXT_LENGTH, categories))

        raw_ev_against = cause.get("evidence_against", [])
        clean_ev_against: List[str] = []
        if isinstance(raw_ev_against, list):
            for item in raw_ev_against[:MAX_EVIDENCE_ITEMS]:
                clean_ev_against.append(_clean_str(item, MAX_TEXT_LENGTH, categories))

        raw_affected = cause.get("affected_services", [])
        clean_affected: List[str] = []
        if isinstance(raw_affected, list):
            for svc in raw_affected[:MAX_SERVICES_COUNT]:
                clean_affected.append(_clean_str(svc, MAX_IDENTIFIER_LENGTH, categories))

        raw_steps = cause.get("recommended_verification_steps", [])
        clean_steps: List[str] = []
        if isinstance(raw_steps, list):
            for step in raw_steps[:MAX_EVIDENCE_ITEMS]:
                clean_steps.append(_clean_str(step, MAX_TEXT_LENGTH, categories))

        clean_first_ts = ""
        if "first_observed_timestamp" in cause:
            _, norm_first_ts = validate_and_normalize_timestamp(cause["first_observed_timestamp"])
            clean_first_ts = norm_first_ts

        clean_causes.append({
            "candidate_service": cand_svc,
            "possible_root_cause": possible_cause,
            "confidence_score": conf_score,
            "confidence_level": conf_level,
            "confidence_breakdown": clean_breakdown,
            "evidence_supporting": clean_ev_supp,
            "evidence_against": clean_ev_against,
            "affected_services": clean_affected,
            "first_observed_timestamp": clean_first_ts,
            "recommended_verification_steps": clean_steps,
            "patient_zero_candidate": bool(cause.get("patient_zero_candidate", False)),
            "classification": _clean_str(cause.get("classification", "Hypothesis"), 64, categories),
        })

    # 7. Filter and construct 'correlated_events' (raw records completely omitted)
    raw_events = input_copy.get("correlated_events", [])
    if not isinstance(raw_events, list):
        return _make_rejection_payload("Field 'correlated_events' must be a list", orig_warnings)

    for event in raw_events:
        if isinstance(event, dict):
            # Validate incident_id
            raw_inc_id = event.get("incident_id")
            is_valid_id, _ = is_valid_incident_id(raw_inc_id)
            if not is_valid_id:
                return _make_rejection_payload("Field 'incident_id' contains an invalid incident identifier", orig_warnings)

            raw_svcs = event.get("services", [])
            if raw_svcs is not None and not isinstance(raw_svcs, list):
                return _make_rejection_payload("Field 'services' must be a list", orig_warnings)
            if isinstance(raw_svcs, list):
                for s in raw_svcs:
                    if not is_valid_service_identifier(s):
                        return _make_rejection_payload("Field 'services' contains an invalid service identifier", orig_warnings)

            has_start = "start_time" in event
            norm_start_ts = ""
            if has_start:
                raw_start = event["start_time"]
                is_valid_ts, norm_start_ts = validate_and_normalize_timestamp(raw_start)
                if not is_valid_ts:
                    return _make_rejection_payload("Field 'start_time' contains an invalid timestamp", orig_warnings)

            has_end = "end_time" in event
            norm_end_ts = ""
            if has_end:
                raw_end = event["end_time"]
                is_valid_ts, norm_end_ts = validate_and_normalize_timestamp(raw_end)
                if not is_valid_ts:
                    return _make_rejection_payload("Field 'end_time' contains an invalid timestamp", orig_warnings)

            # Temporal consistency validation: end_time must not precede start_time
            if has_start and has_end and norm_start_ts and norm_end_ts:
                try:
                    dt_start = datetime.fromisoformat(norm_start_ts)
                    dt_end = datetime.fromisoformat(norm_end_ts)
                    if (dt_start.tzinfo is None) != (dt_end.tzinfo is None):
                        return _make_rejection_payload("Field 'end_time' must not precede start_time", orig_warnings)
                    if dt_end < dt_start:
                        return _make_rejection_payload("Field 'end_time' must not precede start_time", orig_warnings)
                except Exception:
                    return _make_rejection_payload("Field 'end_time' must not precede start_time", orig_warnings)

    clean_events: List[Dict[str, Any]] = []
    for event in raw_events[:MAX_INCIDENTS]:
        if not isinstance(event, dict):
            continue

        raw_svcs = event.get("services", [])
        clean_svcs: List[str] = []
        if isinstance(raw_svcs, list):
            for s in raw_svcs[:MAX_SERVICES_COUNT]:
                clean_svcs.append(_clean_str(s, MAX_IDENTIFIER_LENGTH, categories))

        dur_raw = event.get("duration_seconds", 0.0)
        if isinstance(dur_raw, bool) or not isinstance(dur_raw, (int, float)) or math.isnan(dur_raw) or math.isinf(dur_raw):
            return _make_rejection_payload("Field 'duration_seconds' must be a finite number", orig_warnings)
        dur = float(dur_raw)
        if dur < 0.0:
            return _make_rejection_payload("Field 'duration_seconds' must be non-negative", orig_warnings)

        raw_err_cnt = event.get("error_count")
        if raw_err_cnt is not None:
            if isinstance(raw_err_cnt, bool) or not isinstance(raw_err_cnt, int) or raw_err_cnt < 0:
                return _make_rejection_payload("Field 'error_count' must be a non-negative integer", orig_warnings)
            clean_err_count = raw_err_cnt
        else:
            raw_records = event.get("records")
            clean_err_count = len(raw_records) if isinstance(raw_records, list) else 0

        clean_start_ts = ""
        if "start_time" in event:
            _, norm_start_ts = validate_and_normalize_timestamp(event["start_time"])
            clean_start_ts = norm_start_ts

        clean_end_ts = ""
        if "end_time" in event:
            _, norm_end_ts = validate_and_normalize_timestamp(event["end_time"])
            clean_end_ts = norm_end_ts

        _, clean_inc_id = is_valid_incident_id(event.get("incident_id"))

        clean_ev = {
            "incident_id": clean_inc_id,
            "start_time": clean_start_ts,
            "end_time": clean_end_ts,
            "duration_seconds": dur,
            "services": clean_svcs,
            "error_count": clean_err_count,
        }
        clean_events.append(clean_ev)

    # 8. Filter and construct 'patterns_detected' (allowlist enforcement)
    raw_patterns = input_copy.get("patterns_detected", [])
    clean_patterns: List[Dict[str, Any]] = []
    if isinstance(raw_patterns, list):
        for pat in raw_patterns[:MAX_PATTERNS]:
            if isinstance(pat, dict):
                raw_p_ev = pat.get("evidence", [])
                clean_p_ev: List[str] = []
                if isinstance(raw_p_ev, list):
                    for pev in raw_p_ev[:MAX_EVIDENCE_ITEMS]:
                        clean_p_ev.append(_clean_str(pev, MAX_TEXT_LENGTH, categories))
                clean_patterns.append({
                    "pattern_name": _clean_str(pat.get("pattern_name", ""), MAX_IDENTIFIER_LENGTH, categories),
                    "matched": bool(pat.get("matched", False)),
                    "description": _clean_str(pat.get("description", ""), MAX_TEXT_LENGTH, categories),
                    "evidence": clean_p_ev,
                })
    elif raw_patterns is not None:
        return _make_rejection_payload("Field 'patterns_detected' must be a list", orig_warnings)

    # 9. Filter and construct 'warnings' (allowlist enforcement)
    raw_warnings = input_copy.get("warnings", [])
    clean_warnings: List[str] = []
    if isinstance(raw_warnings, list):
        for w in raw_warnings[:MAX_WARNINGS]:
            clean_warnings.append(_clean_str(w, 200, categories))
    elif raw_warnings is not None:
        return _make_rejection_payload("Field 'warnings' must be a list", orig_warnings)

    # 10. Assemble the restricted, allowlisted dictionary
    allowed_payload = {
        "status": "completed",
        "validation_passed": True,
        "total_errors": total_errors,
        "incident_count": incident_count,
        "ranked_causes": clean_causes,
        "correlated_events": clean_events,
        "patterns_detected": clean_patterns,
        "warnings": clean_warnings,
        "sanitization_info": {
            "is_sanitized": True,
            "eligible_for_transmission": True,
            "redacted_categories": sorted(list(categories)),
            "redactions_applied": len(categories) > 0,
            "status": "validated_and_sanitized",
        },
    }

    # 11. Size limit and JSON compliance enforcement (RFC 8259, no NaN/Infinity)
    try:
        serialized = json.dumps(allowed_payload, allow_nan=False)
        if len(serialized.encode("utf-8")) > MAX_PAYLOAD_SIZE_BYTES:
            return _make_rejection_payload("Payload serialized size exceeds limit", orig_warnings)
    except Exception:
        return _make_rejection_payload("Payload JSON serialization failed", orig_warnings)

    return allowed_payload
