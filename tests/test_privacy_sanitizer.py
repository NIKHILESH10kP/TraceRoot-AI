"""
Unit tests for TraceRoot AI Privacy & Data Sanitization Module.
(Milestone 4 - Task 3A: Pre-AI Evidence Sanitization & Redaction)

All tests use synthetic test data only. No live APIs or external calls are made.
"""

import copy
from datetime import date, datetime
import json
import pandas as pd
import pytest

from privacy_sanitizer import (
    is_sensitive_key,
    prepare_ai_payload,
    sanitize_data_structure,
    sanitize_rca_result,
    sanitize_text,
)
from rca_engine import run_rule_based_rca


# ===========================================================================
# 1. Text Sanitizer Unit Tests
# ===========================================================================

def test_sanitize_email_addresses():
    """Verify various email address formats are redacted while surrounding text is kept."""
    raw = "Alert dispatched to oncall@company.com and dev.team+urgent@ops.internal.org."
    sanitized = sanitize_text(raw)

    assert "[REDACTED_EMAIL]" in sanitized
    assert "oncall@company.com" not in sanitized
    assert "dev.team+urgent@ops.internal.org" not in sanitized
    assert "Alert dispatched to [REDACTED_EMAIL] and [REDACTED_EMAIL]." == sanitized


def test_sanitize_ipv4_addresses_and_ports():
    """Verify IPv4 addresses with or without ports are redacted without affecting versions."""
    raw = "Connection refused by host 192.168.1.100:8080 and peer 10.0.0.15 for version v2.4.1."
    sanitized = sanitize_text(raw)

    assert "[REDACTED_IP]" in sanitized
    assert "192.168.1.100:8080" not in sanitized
    assert "10.0.0.15" not in sanitized
    # Technical version string must remain intact
    assert "v2.4.1" in sanitized
    assert "Connection refused by host [REDACTED_IP] and peer [REDACTED_IP] for version v2.4.1." == sanitized


def test_sanitize_ipv6_addresses_without_breaking_timestamps():
    """Verify IPv6 addresses are redacted while standard HH:MM:SS timestamps are preserved."""
    raw = "Error logged at 10:15:30 from host 2001:0db8:85a3:0000:0000:8a2e:0370:7334 and loopback ::1."
    sanitized = sanitize_text(raw)

    assert "[REDACTED_IP]" in sanitized
    assert "2001:0db8:85a3:0000:0000:8a2e:0370:7334" not in sanitized
    assert "::1" not in sanitized
    # Standard 3-part timestamp must NOT be redacted
    assert "10:15:30" in sanitized


def test_sanitize_bearer_and_basic_auth():
    """Verify authorization headers and tokens are masked."""
    raw_bearer = "Authorization header: Bearer abc123xyz789SecretTokenValue"
    sanitized_bearer = sanitize_text(raw_bearer)
    assert "[REDACTED_TOKEN]" in sanitized_bearer
    assert "abc123xyz789SecretTokenValue" not in sanitized_bearer
    assert "Bearer [REDACTED_TOKEN]" in sanitized_bearer

    raw_basic = "Authorization: Basic dXNlcjpwYXNzd29yZDEyMw=="
    sanitized_basic = sanitize_text(raw_basic)
    assert "[REDACTED_TOKEN]" in sanitized_basic
    assert "dXNlcjpwYXNzd29yZDEyMw==" not in sanitized_basic


def test_sanitize_url_embedded_credentials():
    """Verify connection strings with usernames and passwords are scrubbed."""
    raw = "Database connection failed for postgresql://admin_user:SuperSecretPassword123@10.0.0.1:5432/orders_db"
    sanitized = sanitize_text(raw)

    assert "SuperSecretPassword123" not in sanitized
    assert "admin_user" not in sanitized
    assert "[REDACTED_USER]:[REDACTED_PASSWORD]@" in sanitized
    assert "[REDACTED_IP]" in sanitized
    assert "orders_db" in sanitized


def test_sanitize_jwt_tokens():
    """Verify JSON Web Tokens (3 base64 segments) are masked."""
    fake_jwt = "Token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c expired"
    sanitized = sanitize_text(fake_jwt)

    assert "[REDACTED_JWT_TOKEN]" in sanitized
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitized


def test_sanitize_known_api_keys():
    """Verify well-known API key signatures (Google, OpenAI, AWS, GitHub) are detected."""
    google_key = "AIzaSyD-1234567890123456789012345678901"
    openai_key = "sk-live-1234567890abcdef1234567890"
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    github_token = "ghp_1234567890abcdef1234567890abcdef1234"

    raw = f"Failed API sync with Google={google_key}, OpenAI={openai_key}, AWS={aws_key}, GitHub={github_token}"
    sanitized = sanitize_text(raw)

    assert "[REDACTED_API_KEY]" in sanitized
    assert google_key not in sanitized
    assert openai_key not in sanitized
    assert aws_key not in sanitized
    assert github_token not in sanitized


def test_sanitize_key_value_secrets():
    """Verify key-value secret fields (password, secret, token, api_key) are masked."""
    raw = (
        'Failed request with password="MySecretPassword123" and '
        "api_key=sk-fakeSecretVal123 and auth_token='my-secret-auth-token-456'"
    )
    sanitized = sanitize_text(raw)

    assert "MySecretPassword123" not in sanitized
    assert "my-secret-auth-token-456" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized


def test_sanitize_credit_card_numbers():
    """Verify payment card sequences are masked."""
    raw = "Transaction failed for card 4111 2222 3333 4444 on gateway"
    sanitized = sanitize_text(raw)

    assert "[REDACTED_CARD]" in sanitized
    assert "4111 2222 3333 4444" not in sanitized
    assert "Transaction failed for card [REDACTED_CARD] on gateway" == sanitized


# ===========================================================================
# 2. Technical Evidence Preservation (No False Positives)
# ===========================================================================

def test_preserves_technical_evidence_and_diagnostics():
    """Verify valid technical diagnostics, metrics, error codes, and versions are not altered."""
    clean_technical_log = (
        "2026-10-27 10:15:30 ERROR database-proxy: Database connection timeout after 30 seconds. "
        "Connection pool usage at 100% (200/200 connections active). "
        "HTTP 504 Gateway Timeout on POST /api/payments in payment-service v2.4.1 taking 4800 ms."
    )
    sanitized = sanitize_text(clean_technical_log)

    # Everything in this log is pure technical diagnostic context and must be preserved
    assert sanitized == clean_technical_log
    assert "database-proxy" in sanitized
    assert "payment-service" in sanitized
    assert "v2.4.1" in sanitized
    assert "HTTP 504 Gateway Timeout" in sanitized
    assert "200/200" in sanitized
    assert "4800 ms" in sanitized


# ===========================================================================
# 3. Input Immutability & Deep Copy Verification
# ===========================================================================

def test_original_rca_result_remains_unchanged():
    """Verify sanitize_rca_result never mutates any field of the caller's input dictionary."""
    original_rca = {
        "status": "completed",
        "total_errors": 1,
        "ranked_causes": [
            {
                "candidate_service": "payment-service",
                "possible_root_cause": "Database connection failure with password=MySecret123 at 192.168.1.50",
                "confidence_score": 80.0,
                "evidence_supporting": [
                    "Host 10.0.0.1:8080 timeout with token=abc123secret",
                ],
                "affected_services": ["payment-service", "database-proxy"],
            }
        ],
        "correlated_events": [
            {
                "incident_id": 1,
                "services": ["payment-service"],
                "records": [
                    {
                        "message": "User user@domain.com auth failed with Bearer secretBearerVal123",
                        "service": "payment-service",
                    }
                ],
            }
        ],
        "warnings": ["Warning from 172.16.0.1"],
    }

    # Deep copy for strict comparison
    original_snapshot = copy.deepcopy(original_rca)

    # Perform sanitization
    sanitized_rca = sanitize_rca_result(original_rca)

    # 1. Original input must be strictly identical before and after
    assert original_rca == original_snapshot

    # 2. Sanitized output must have sensitive tokens masked
    top_cause = sanitized_rca["ranked_causes"][0]
    assert "[REDACTED_SECRET]" in top_cause["possible_root_cause"]
    assert "[REDACTED_IP]" in top_cause["possible_root_cause"]
    assert "MySecret123" not in top_cause["possible_root_cause"]
    assert "192.168.1.50" not in top_cause["possible_root_cause"]

    # 3. Supporting evidence list must be sanitized
    assert "[REDACTED_IP]" in top_cause["evidence_supporting"][0]
    assert "[REDACTED_SECRET]" in top_cause["evidence_supporting"][0]

    # 4. Correlated records must be sanitized
    rec_msg = sanitized_rca["correlated_events"][0]["records"][0]["message"]
    assert "[REDACTED_EMAIL]" in rec_msg
    assert "[REDACTED_TOKEN]" in rec_msg
    assert "user@domain.com" not in rec_msg

    # 5. Sanitization info metadata must be present
    assert sanitized_rca["sanitization_info"]["is_sanitized"] is True
    assert sanitized_rca["sanitization_info"]["redactions_applied"] is True
    assert len(sanitized_rca["sanitization_info"]["redacted_categories"]) > 0


# ===========================================================================
# 4. Edge Cases: Malformed, Empty, and Unexpected Inputs
# ===========================================================================

def test_handles_none_input_safely():
    """Verify None input returns a safe default structure without raising exceptions."""
    result = sanitize_rca_result(None)
    assert result["status"] == "invalid_input"
    assert result["sanitization_info"]["is_sanitized"] is True
    assert result["ranked_causes"] == []


def test_handles_non_dict_input_safely():
    """Verify non-dict types (string, list, int) return a safe structure."""
    result = sanitize_rca_result(["not", "a", "dict"])
    assert result["status"] == "invalid_input"
    assert result["sanitization_info"]["is_sanitized"] is True


def test_handles_empty_dict():
    """Verify empty dict input returns safely with sanitization metadata."""
    result = sanitize_rca_result({})
    assert result["sanitization_info"]["is_sanitized"] is True
    assert result["sanitization_info"]["redactions_applied"] is False


def test_sensitive_dictionary_key_detection():
    """Verify dictionary keys indicating secrets have their values redacted wholesale."""
    assert is_sensitive_key("password") is True
    assert is_sensitive_key("api_key") is True
    assert is_sensitive_key("auth_token") is True
    assert is_sensitive_key("client_secret") is True
    assert is_sensitive_key("service_name") is False
    assert is_sensitive_key("timestamp") is False

    test_payload = {
        "service": "auth-service",
        "api_key": "raw_sensitive_key_value",
        "password": "plain_text_password",
    }
    sanitized, categories = sanitize_data_structure(test_payload)
    assert sanitized["service"] == "auth-service"
    assert sanitized["api_key"] == "[REDACTED_SECRET]"
    assert sanitized["password"] == "[REDACTED_SECRET]"
    assert "Sensitive Dictionary Key" in categories


# ===========================================================================
# 5. End-to-End Integration with `rca_engine.run_rule_based_rca()`
# ===========================================================================

def test_full_pipeline_with_rca_engine():
    """Verify real output from run_rule_based_rca() with sensitive logs is sanitized cleanly."""
    # Synthetic logs simulating real microservice errors containing embedded credentials and IPs
    raw_logs = [
        (
            "2026-10-27 10:00:00",
            "ERROR",
            "database-proxy",
            "Database connection timeout after 30 seconds connecting to postgresql://dbadmin:SuperSecretPass999@10.240.0.12:5432/core",
        ),
        (
            "2026-10-27 10:00:15",
            "ERROR",
            "payment-service",
            "Payment failed for order #5002: database connection timeout from client 192.168.10.45:9000 with token=abcTokenSecret789",
        ),
        (
            "2026-10-27 10:00:30",
            "ERROR",
            "api-gateway",
            "HTTP 504 Gateway Timeout contacting oncall-admin@example.org with Bearer mySecretTokenPayload12345",
        ),
    ]
    df = pd.DataFrame(raw_logs, columns=["timestamp", "level", "service", "message"])

    # 1. Run Layer 1 Rule-Based RCA
    rca_result = run_rule_based_rca(df)
    assert rca_result["status"] == "completed"
    assert len(rca_result["ranked_causes"]) > 0

    # 2. Sanitize RCA result
    sanitized_rca = sanitize_rca_result(rca_result)

    # 3. Assert zero leakage of sensitive data
    sanitized_str = str(sanitized_rca)
    assert "SuperSecretPass999" not in sanitized_str
    assert "10.240.0.12" not in sanitized_str
    assert "192.168.10.45" not in sanitized_str
    assert "abcTokenSecret789" not in sanitized_str
    assert "oncall-admin@example.org" not in sanitized_str
    assert "mySecretTokenPayload12345" not in sanitized_str

    # 4. Assert diagnostic evidence is preserved
    assert "database-proxy" in sanitized_str
    assert "payment-service" in sanitized_str
    assert "api-gateway" in sanitized_str
    assert "Database connection timeout" in sanitized_str
    assert "HTTP 504 Gateway Timeout" in sanitized_str
    assert sanitized_rca["sanitization_info"]["is_sanitized"] is True
    assert sanitized_rca["sanitization_info"]["redactions_applied"] is True


# ===========================================================================
# 6. AI Prompt Payload Preparation
# ===========================================================================

def test_prepare_ai_payload_record_capping():
    """Verify prepare_ai_payload caps raw records per incident to prevent prompt bloat."""
    incident_records = [
        {"timestamp": f"2026-10-27 10:00:{i:02d}", "service": "payment-service", "message": f"Error event #{i}"}
        for i in range(20)
    ]
    fake_rca = {
        "status": "completed",
        "total_errors": 20,
        "incident_count": 1,
        "ranked_causes": [],
        "correlated_events": [
            {
                "incident_id": 1,
                "services": ["payment-service"],
                "records": incident_records,
            }
        ],
    }

    ai_payload = prepare_ai_payload(fake_rca, max_records_per_incident=5)

    incident = ai_payload["correlated_events"][0]
    assert len(incident["records"]) == 5
    assert incident["records_truncated"] is True
    assert incident["total_records_in_incident"] == 20



# ===========================================================================
# 7. Task 3A.1 Security Audit Regression Tests
# ===========================================================================

def test_snake_case_credential_detection():
    """Verify snake-case secret keys (api_token, db_password, etc.) are detected and redacted."""
    raw = (
        "Service config: api_token=mySecretToken123 and db_password=myPassword456 and "
        "jwt_secret=myJwtSecret789 and admin_password=myAdminPass123 and "
        "service_access_token=myAccessToken777 and client_secret=myClientSecret888 "
        "with technical context user_id=100 and service_name=order-service and status_code=500"
    )
    sanitized = sanitize_text(raw)

    # All secrets must be redacted
    assert "mySecretToken123" not in sanitized
    assert "myPassword456" not in sanitized
    assert "myJwtSecret789" not in sanitized
    assert "myAdminPass123" not in sanitized
    assert "myAccessToken777" not in sanitized
    assert "myClientSecret888" not in sanitized

    # Redaction markers must be in place
    assert "api_token=[REDACTED_SECRET]" in sanitized
    assert "db_password=[REDACTED_SECRET]" in sanitized
    assert "jwt_secret=[REDACTED_SECRET]" in sanitized
    assert "admin_password=[REDACTED_SECRET]" in sanitized
    assert "service_access_token=[REDACTED_SECRET]" in sanitized
    assert "client_secret=[REDACTED_SECRET]" in sanitized

    # Technical identifiers must be preserved
    assert "user_id=100" in sanitized
    assert "service_name=order-service" in sanitized
    assert "status_code=500" in sanitized


def test_snake_case_case_insensitive_matching():
    """Verify snake_case secret keys match regardless of casing."""
    raw = "Init: API_TOKEN=secretVal1 and Db_Password='secretVal2' and JWT_SECRET=\"secretVal3\""
    sanitized = sanitize_text(raw)

    assert "secretVal1" not in sanitized
    assert "secretVal2" not in sanitized
    assert "secretVal3" not in sanitized
    assert "API_TOKEN=[REDACTED_SECRET]" in sanitized
    assert "Db_Password='[REDACTED_SECRET]'" in sanitized
    assert "JWT_SECRET=\"[REDACTED_SECRET]\"" in sanitized


def test_generic_authorization_headers():
    """Verify Authorization headers with or without Bearer/Basic are cleanly redacted."""
    cases = [
        ("Authorization: secretTokenRaw123", "Authorization: [REDACTED_TOKEN]"),
        ("authorization=secretTokenRaw123", "authorization=[REDACTED_TOKEN]"),
        ("Authorization: Bearer secretTokenBearer456", "Authorization: Bearer [REDACTED_TOKEN]"),
        ("Authorization: Basic dXNlcjpwYXNz", "Authorization: Basic [REDACTED_TOKEN]"),
        ('authorization="Bearer secretInQuotes789"', 'authorization="Bearer [REDACTED_TOKEN]"'),
        ("Proxy-Authorization: proxySecretToken", "Proxy-Authorization: [REDACTED_TOKEN]"),
    ]

    for raw, expected in cases:
        sanitized = sanitize_text(raw)
        assert sanitized == expected
        assert "secretToken" not in sanitized
        assert "proxySecretToken" not in sanitized


def test_universal_connection_string_credentials():
    """Verify credentials in universal URI schemes (amqp, mongodb+srv, kafka, etc.) are scrubbed."""
    uri_cases = [
        ("amqp://guest:secretBrokerPass@10.0.0.1:5672/vhost", "secretBrokerPass"),
        ("mongodb+srv://admin:passCluster123@cluster0.abc.mongodb.net/db", "passCluster123"),
        ("kafka://client:passKafka456@broker1.internal:9092", "passKafka456"),
        ("postgresql://postgres:secretPg789@db.internal:5432/orders", "secretPg789"),
        ("redis://default:redispwd999@redis.internal:6379/0", "redispwd999"),
        ("https://apiuser:tokenpwd321@api.service.com/v1", "tokenpwd321"),
        ("jdbc:postgresql://dbuser:dbpwd555@localhost:5432/core", "dbpwd555"),
    ]

    for raw_uri, secret in uri_cases:
        sanitized = sanitize_text(raw_uri)
        assert secret not in sanitized
        assert "[REDACTED_USER]:[REDACTED_PASSWORD]@" in sanitized


def test_dictionary_key_protection_and_collision_avoidance():
    """Verify sensitive dictionary keys are masked and key collisions are resolved safely."""
    # Check is_sensitive_key
    assert is_sensitive_key("api_token") is True
    assert is_sensitive_key("db_password") is True
    assert is_sensitive_key("jwt_secret") is True
    assert is_sensitive_key("client_secret") is True
    assert is_sensitive_key("private_key") is True
    assert is_sensitive_key("token_count") is False

    raw_dict = {
        "192.168.1.1": "server_alpha",
        "192.168.1.2": "server_beta",
        "admin@example.org": "admin_role",
        "billing@example.org": "billing_role",
        "api_token": "myRawSecretToken1",
        "db_password": "myRawDbPassword2",
        "jwt_secret": "myRawJwtSecret3",
        "service": "payment-service",
        "token_count": 42,
    }

    sanitized_dict, categories = sanitize_data_structure(raw_dict)

    # 1. Sensitive values must be masked
    assert sanitized_dict["api_token"] == "[REDACTED_SECRET]"
    assert sanitized_dict["db_password"] == "[REDACTED_SECRET]"
    assert sanitized_dict["jwt_secret"] == "[REDACTED_SECRET]"

    # 2. Non-sensitive key token_count preserved
    assert sanitized_dict["token_count"] == 42
    assert sanitized_dict["service"] == "payment-service"

    # 3. Keys containing sensitive data are sanitized
    assert "[REDACTED_IP]" in sanitized_dict
    assert "[REDACTED_IP]_2" in sanitized_dict
    assert "[REDACTED_EMAIL]" in sanitized_dict
    assert "[REDACTED_EMAIL]_2" in sanitized_dict

    # 4. Values preserved across collision resolution (no silent overwrite)
    assert sanitized_dict["[REDACTED_IP]"] == "server_alpha"
    assert sanitized_dict["[REDACTED_IP]_2"] == "server_beta"
    assert sanitized_dict["[REDACTED_EMAIL]"] == "admin_role"
    assert sanitized_dict["[REDACTED_EMAIL]_2"] == "billing_role"
    assert len(sanitized_dict) == len(raw_dict)


def test_url_query_parameter_boundary_handling():
    """Verify secret matching stops at query parameter separators (& and ;) preserving subsequent params."""
    raw = "GET /api/v1/auth?token=mySecretToken123&user_id=100&role=viewer;session_mode=active"
    sanitized = sanitize_text(raw)

    assert "mySecretToken123" not in sanitized
    assert "token=[REDACTED_SECRET]" in sanitized
    # Subsequent parameters must be preserved
    assert "&user_id=100" in sanitized
    assert "&role=viewer" in sanitized
    assert ";session_mode=active" in sanitized


def test_set_collection_handling():
    """Verify set collections are sanitized safely without mutating the original set."""
    original_set = {
        "User email: oncall@example.com",
        "Connecting from 192.168.1.55",
        "Normal technical message",
    }
    original_snapshot = set(original_set)

    sanitized_set, categories = sanitize_data_structure(original_set)

    # 1. Original set is unmutated
    assert original_set == original_snapshot

    # 2. Returned collection is a set with sanitized values
    assert isinstance(sanitized_set, set)
    sanitized_str = " ".join(sanitized_set)
    assert "oncall@example.com" not in sanitized_str
    assert "192.168.1.55" not in sanitized_str
    assert "[REDACTED_EMAIL]" in sanitized_str
    assert "[REDACTED_IP]" in sanitized_str
    assert "Normal technical message" in sanitized_set
    assert "Email Address" in categories
    assert "IP Address" in categories


# ===========================================================================
# 8. Task 3A.3 Remediation Regression Tests
# ===========================================================================

def test_quoted_json_credential_keys():
    """Verify double-quoted JSON credential keys and their values are sanitized."""
    cases = [
        ('{"password": "SecretPassword123"}', '{"password": "[REDACTED_SECRET]"}'),
        ('{"token": "MySecretToken456"}', '{"token": "[REDACTED_SECRET]"}'),
        ('{"db_password": "database-secret"}', '{"db_password": "[REDACTED_SECRET]"}'),
        ('{"api_token": "api-secret-789"}', '{"api_token": "[REDACTED_SECRET]"}'),
    ]
    for raw, expected in cases:
        sanitized = sanitize_text(raw)
        assert sanitized == expected, f"Failed: {raw} -> {sanitized}"
        assert "[REDACTED_SECRET]" in sanitized


def test_single_quoted_dict_representations():
    """Verify single-quoted Python dict representations are sanitized cleanly."""
    cases = [
        ("{'api_token': 'secret-value'}", "{'api_token': '[REDACTED_SECRET]'}"),
        ("{'password': 'secretPassword'}", "{'password': '[REDACTED_SECRET]'}"),
        ("{'token': 'mySecretToken'}", "{'token': '[REDACTED_SECRET]'}"),
    ]
    for raw, expected in cases:
        sanitized = sanitize_text(raw)
        assert sanitized == expected, f"Failed: {raw} -> {sanitized}"


def test_adjacent_non_sensitive_json_fields_preserved():
    """Verify adjacent non-sensitive JSON fields and token metrics are untouched."""
    raw_payload = (
        '{"status": "error", "password": "SuperSecretVal1", "service": "auth-service", '
        '"attempt": 3, "total_tokens": 1500, "prompt_tokens": 200, "token_count": 42}'
    )
    sanitized = sanitize_text(raw_payload)

    assert "SuperSecretVal1" not in sanitized
    assert '"password": "[REDACTED_SECRET]"' in sanitized
    assert '"status": "error"' in sanitized
    assert '"service": "auth-service"' in sanitized
    assert '"attempt": 3' in sanitized
    assert '"total_tokens": 1500' in sanitized
    assert '"prompt_tokens": 200' in sanitized
    assert '"token_count": 42' in sanitized


def test_cryptographic_and_auth_key_variants():
    """Verify all 6 cryptographic and authentication key variants are redacted."""
    crypto_keys = [
        ("secret_key", "secVal1"),
        ("access_key", "accVal2"),
        ("signing_key", "sigVal3"),
        ("encryption_key", "encVal4"),
        ("auth_key", "autVal5"),
        ("master_key", "masVal6"),
    ]

    for key_name, secret_val in crypto_keys:
        # Standard snake_case
        raw1 = f"{key_name}={secret_val}"
        san1 = sanitize_text(raw1)
        assert secret_val not in san1, f"Failed for {raw1}"
        assert f"{key_name}=[REDACTED_SECRET]" == san1

        # Kebab-case
        kebab_name = key_name.replace("_", "-")
        raw2 = f"{kebab_name}: {secret_val}"
        san2 = sanitize_text(raw2)
        assert secret_val not in san2, f"Failed for {raw2}"
        assert f"{kebab_name}: [REDACTED_SECRET]" == san2

        # Quoted JSON
        raw3 = f'{{"{key_name}": "{secret_val}"}}'
        san3 = sanitize_text(raw3)
        assert secret_val not in san3, f"Failed for {raw3}"
        assert f'{{"{key_name}": "[REDACTED_SECRET]"}}' == san3

        # Mixed-case
        upper_name = key_name.upper()
        raw4 = f"{upper_name}='{secret_val}'"
        san4 = sanitize_text(raw4)
        assert secret_val not in san4, f"Failed for {raw4}"
        assert f"{upper_name}='[REDACTED_SECRET]'" == san4


def test_datetime_and_pandas_timestamp_serialization():
    """Verify datetime, date, and Pandas Timestamp are serialized to ISO 8601 strings."""
    test_struct = {
        "py_datetime": datetime(2026, 10, 31, 10, 15, 30),
        "py_date": date(2026, 10, 31),
        "pd_timestamp": pd.Timestamp("2026-10-31 10:15:30"),
        "list_timestamps": [
            pd.Timestamp("2026-10-31 11:00:00"),
            datetime(2026, 10, 31, 12, 0, 0),
        ],
        "tuple_timestamps": (pd.Timestamp("2026-10-31 13:00:00"),),
    }

    sanitized, _ = sanitize_data_structure(test_struct)

    assert sanitized["py_datetime"] == "2026-10-31T10:15:30"
    assert sanitized["py_date"] == "2026-10-31"
    assert sanitized["pd_timestamp"] == "2026-10-31T10:15:30"
    assert sanitized["list_timestamps"] == ["2026-10-31T11:00:00", "2026-10-31T12:00:00"]
    assert sanitized["tuple_timestamps"] == ("2026-10-31T13:00:00",)


def test_json_dumps_on_prepared_payload():
    """Verify prepare_ai_payload produces output natively serializable by json.dumps."""
    df = pd.DataFrame([
        {
            "timestamp": "2026-10-27 10:15:30",
            "service": "database-proxy",
            "level": "ERROR",
            "message": "Connection pool exhausted. password='secretPass'",
        },
        {
            "timestamp": "2026-10-27 10:15:35",
            "service": "database-proxy",
            "level": "ERROR",
            "message": "Connection pool exhausted. password='secretPass'",
        },
        {
            "timestamp": "2026-10-27 10:15:40",
            "service": "payment-service",
            "level": "ERROR",
            "message": "Database query timeout after 30s. user_token=myTokenVal",
        },
        {
            "timestamp": "2026-10-27 10:15:45",
            "service": "payment-service",
            "level": "ERROR",
            "message": "Database query timeout after 30s. user_token=myTokenVal",
        },
    ])

    rca_result = run_rule_based_rca(df)
    payload = prepare_ai_payload(rca_result)

    # Standard json.dumps must succeed without TypeError or custom encoder
    serialized = json.dumps(payload)
    assert isinstance(serialized, str)
    assert len(serialized) > 0
    assert "secretPass" not in serialized
    assert "myTokenVal" not in serialized
    assert "[REDACTED_SECRET]" in serialized

    # Verify deserialization returns identical structure
    roundtrip = json.loads(serialized)
    assert roundtrip["status"] == "completed"
    assert roundtrip["sanitization_info"]["is_sanitized"] is True


def test_prepare_ai_payload_malformed_records():
    """Verify prepare_ai_payload handles non-list records defensively without TypeError."""
    malformed_rca = {
        "status": "completed",
        "total_errors": 4,
        "incident_count": 4,
        "correlated_events": [
            {"incident_id": "inc_int", "records": 42},
            {"incident_id": "inc_str", "records": "malformed_records_string"},
            {"incident_id": "inc_none", "records": None},
            {"incident_id": "inc_dict", "records": {"error": "not a list"}},
            {"incident_id": "inc_valid", "records": [1, 2, 3, 4, 5, 6, 7]},
        ],
        "findings": [],
        "ranked_causes": [],
        "patterns_detected": [],
        "warnings": [],
    }

    # Must execute safely without raising TypeError
    payload = prepare_ai_payload(malformed_rca, max_records_per_incident=3)

    events = payload["correlated_events"]
    assert events[0]["records"] == 42
    assert "records_truncated" not in events[0]
    assert events[1]["records"] == "malformed_records_string"
    assert events[2]["records"] is None
    assert events[3]["records"] == {"error": "not a list"}
    # Valid list gets capped properly
    assert events[4]["records"] == [1, 2, 3]
    assert events[4]["records_truncated"] is True
    assert events[4]["total_records_in_incident"] == 7


def test_dictionary_crypto_key_protection():
    """Verify dictionary keys matching cryptographic/auth keys are detected and redacted."""
    crypto_keys = [
        "secret_key", "Secret-Key",
        "access_key", "Access_Key",
        "signing_key", "signing-key",
        "encryption_key", "ENCRYPTION_KEY",
        "auth_key", "Auth-Key",
        "master_key", "MASTER_KEY",
    ]

    for k in crypto_keys:
        assert is_sensitive_key(k) is True, f"Failed for {k}"

    raw_dict = {
        "secret_key": "rawSecret1",
        "access-key": "rawSecret2",
        "signing_key": "rawSecret3",
        "token_count": 42,
        "prompt_tokens": 150,
    }
    sanitized, _ = sanitize_data_structure(raw_dict)
    assert sanitized["secret_key"] == "[REDACTED_SECRET]"
    assert sanitized["access-key"] == "[REDACTED_SECRET]"
    assert sanitized["signing_key"] == "[REDACTED_SECRET]"
    assert sanitized["token_count"] == 42
    assert sanitized["prompt_tokens"] == 150
