"""Shared contract constants for the direct OpenAI-compatible adapter (#431 S1).

Keeping the structured-output mode list and the generation-param whitelist in
one dependency-free module prevents the config validator, the backend and the
GUI from silently drifting apart.
"""

from __future__ import annotations

import re

STRUCTURED_OUTPUT_MODE_ORDER: tuple[str, ...] = (
    "strict_json_schema",
    "json_object",
    "prompt_only_json",
)
STRUCTURED_OUTPUT_MODES = frozenset(STRUCTURED_OUTPUT_MODE_ORDER)
DEFAULT_STRUCTURED_OUTPUT_MODE = "prompt_only_json"

GENERATION_PARAM_KEYS = frozenset({
    "temperature",
    "max_output_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
    "timeout",
})

# Keys that map 1:1 into the Chat Completions JSON body.  ``max_output_tokens``
# is renamed to ``max_tokens`` and ``timeout`` is transport-only, so neither may
# be copied into the request body as an unknown provider parameter.
REQUEST_BODY_PARAM_KEYS = frozenset({
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
})

SENSITIVE_HEADER_NAME_MARKERS = (
    "authorization",
    "apikey",
    "api-key",
    "token",
    "secret",
    "password",
    "cookie",
)
SENSITIVE_HEADER_VALUE_MARKERS = (
    "bearer ",
    "api_key=",
    "apikey=",
    "access_token=",
    "token=",
    "secret=",
    "password=",
)
_SK_KEY_PATTERN = re.compile(r"(?:^|[\s,;=:])sk-[a-z0-9_-]{8,}")


def is_sensitive_header(name: object, value: object) -> bool:
    """Return whether a header name/value pair looks credential-bearing.

    Shared by config validation and the backend so an unvalidated/tolerated
    config path cannot smuggle a credential into ``extra_headers`` or override
    the ``Authorization`` header derived from ``credential_ref``.
    """

    normalized_name = re.sub(
        r"[^a-z0-9]",
        "",
        str(name or "").casefold(),
    )
    if any(
        re.sub(r"[^a-z0-9]", "", marker) in normalized_name
        for marker in SENSITIVE_HEADER_NAME_MARKERS
    ):
        return True
    lowered_value = str(value or "").casefold()
    if any(marker in lowered_value for marker in SENSITIVE_HEADER_VALUE_MARKERS):
        return True
    return bool(_SK_KEY_PATTERN.search(lowered_value))
