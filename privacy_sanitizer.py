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


def sanitize_rca_result(rca_result: Any) -> Dict[str, Any]:
    """Sanitize complete structured findings returned by `run_rule_based_rca()`.

    Guarantees:
    - Never modifies the input `rca_result` (uses deep copy).
    - Returns a safe, fully sanitized dictionary structure.
    - Handles None, empty dicts, or malformed inputs without exceptions.
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
                "is_sanitized": True,
                "redacted_categories": [],
                "status": "skipped_invalid_input",
            },
        }

    # Deep copy ensures zero mutation of caller's data
    data_copy = copy.deepcopy(rca_result)

    sanitized_data, categories = sanitize_data_structure(data_copy)

    # Attach sanitization audit metadata
    sanitized_data["sanitization_info"] = {
        "is_sanitized": True,
        "redacted_categories": sorted(list(categories)),
        "redactions_applied": len(categories) > 0,
    }

    return sanitized_data


def prepare_ai_payload(
    rca_result: Any,
    max_records_per_incident: int = 5,
) -> Dict[str, Any]:
    """Prepare a compact, sanitized summary payload tailored for external AI prompts.

    Guarantees that every string in the payload has been passed through the sanitization
    pipeline. Also caps the number of individual log records per incident to prevent
    prompt token bloat and stay comfortably within free-tier rate and token limits.

    Parameters
    ----------
    rca_result : Any
        Structured RCA result dictionary.
    max_records_per_incident : int, optional
        Maximum individual log records to include per incident cluster (default: 5).

    Returns
    -------
    Dict[str, Any]
        Compact, privacy-safe dictionary ready to be converted into an LLM prompt.
    """
    sanitized_rca = sanitize_rca_result(rca_result)

    # Compact the correlated_events records to prevent prompt bloat
    if "correlated_events" in sanitized_rca and isinstance(sanitized_rca["correlated_events"], list):
        for incident in sanitized_rca["correlated_events"]:
            if isinstance(incident, dict) and "records" in incident:
                records = incident["records"]
                if isinstance(records, list):
                    if len(records) > max_records_per_incident:
                        incident["records"] = records[:max_records_per_incident]
                        incident["records_truncated"] = True
                        incident["total_records_in_incident"] = len(records)

    return sanitized_rca
