"""
PII redaction for log output.

Deterministic, regex-based. Runs on log text BEFORE it leaves the pod
(i.e. before it reaches the Claude API or Telegram). This is the only place
log content is held by our own code, so it is the only place redaction can be
enforced — Bash `kubectl logs` output goes straight to the model and is blocked
elsewhere precisely because it cannot be sanitized.

Covered types (allowlist — we redact only what we recognize):
  email, international phone numbers (any country), KZ IIN/BIN (12 digits),
  payment cards (Luhn-checked), JWT, Bearer tokens, AWS access keys, and the
  values of name-bearing fields (first/last/middle name, surname, full name,
  fio — latin and cyrillic keys).

Limitation: regex cannot catch free-floating personal names with no field label.
For names we redact the VALUE of known keys (JSON `"firstName": "..."` or
`first_name=...`), which is reliable; bare names in prose may pass through.
"""
import re

# ── Token / structured-secret patterns ──────────────────────────────────────────

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# JWT: three base64url segments separated by dots, starting with the standard header.
_JWT = re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")

# Bearer / Authorization tokens.
_BEARER = re.compile(r"\b[Bb]earer\s+[A-Za-z0-9._\-]+")

# AWS access key id.
_AWS_KEY = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")

# Name-bearing keys (latin + cyrillic). Used in both JSON and key=value forms.
_NAME_KEYS = (
    r"first[_\-]?name|last[_\-]?name|middle[_\-]?name|full[_\-]?name|sur[_\-]?name|"
    r"patronymic|fio|имя|фамилия|отчество|фио"
)
# JSON: "firstName": "value"
_NAME_JSON = re.compile(rf'("(?:{_NAME_KEYS})"\s*:\s*")([^"]*)(")', re.IGNORECASE)
# key=value / key: value (unquoted), value is a single token.
_NAME_KV = re.compile(
    rf'\b((?:{_NAME_KEYS})\s*[=:]\s*)("?)([^\s",;}}]+)(\2)', re.IGNORECASE
)

# ── Numeric PII (validated in code to avoid false positives) ─────────────────────

# International phone (any country): + or 00 prefix, then a run of digits/separators.
# The upper bound (40) just caps backtracking; the digit-count check in code decides
# whether the span is actually a phone. Wide enough to swallow a full grouped number.
_PHONE_INTL = re.compile(r"(?<![\w+])(?:\+|00)[\s\-.]?\d[\d\s\-.()]{4,40}\d(?!\w)")
# National RU/KZ style: leading 8, 11 digits total.
_PHONE_NATIONAL = re.compile(
    r"(?<!\d)8[\s\-.]?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{2}[\s\-.]?\d{2}(?!\d)"
)

# KZ IIN/BIN — exactly 12 digits standing alone.
_IIN = re.compile(r"(?<!\d)\d{12}(?!\d)")

# Candidate payment card: 13–19 digits with optional space/dash separators.
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?<=\d)")


def _digit_count(s: str) -> int:
    return sum(c.isdigit() for c in s)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_intl_phones(text: str) -> str:
    def _sub(m: re.Match) -> str:
        # The match contains only digits/separators (no letters). Redact the whole
        # span once it holds at least a phone's worth of digits. No upper bound:
        # two space-separated numbers may merge into one token, but nothing leaks
        # (a capped match could otherwise leave a trailing number fragment behind).
        return "[REDACTED_PHONE]" if _digit_count(m.group()) >= 7 else m.group()
    return _PHONE_INTL.sub(_sub, text)


def _redact_cards(text: str) -> str:
    def _sub(m: re.Match) -> str:
        raw = m.group()
        digits = re.sub(r"[ \-]", "", raw)
        if not (13 <= len(digits) <= 19):
            return raw
        # Contiguous (separator-less) runs are treated as cards only at canonical
        # PAN lengths, so 13-digit epoch-ms timestamps aren't mistaken for cards.
        has_sep = " " in raw or "-" in raw
        if not has_sep and len(digits) not in (15, 16, 19):
            return raw
        return "[REDACTED_CARD]" if _luhn_ok(digits) else raw
    return _CARD_CANDIDATE.sub(_sub, text)


def redact(text: str) -> str:
    """Return `text` with recognized PII replaced by [REDACTED_*] tokens.

    Order matters: high-specificity / longer patterns run first so they claim
    their spans before broader numeric patterns (phones & cards before the bare
    12-digit IIN match).
    """
    if not text:
        return text
    # Tokens & structured secrets first.
    text = _JWT.sub("[REDACTED_JWT]", text)
    text = _BEARER.sub("Bearer [REDACTED_TOKEN]", text)
    text = _AWS_KEY.sub("[REDACTED_AWS_KEY]", text)
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    # Named fields (value only).
    text = _NAME_JSON.sub(lambda m: m.group(1) + "[REDACTED_NAME]" + m.group(3), text)
    text = _NAME_KV.sub(lambda m: m.group(1) + m.group(2) + "[REDACTED_NAME]" + m.group(4), text)
    # Phones (claim + / 00 / 8 numbers before generic numeric matches).
    text = _redact_intl_phones(text)
    text = _PHONE_NATIONAL.sub("[REDACTED_PHONE]", text)
    # Cards (Luhn-validated), then bare IIN/BIN.
    text = _redact_cards(text)
    text = _IIN.sub("[REDACTED_IIN]", text)
    return text
