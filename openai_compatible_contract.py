"""Shared contract constants for the direct OpenAI-compatible adapter (#431 S1).

Keeping the structured-output mode list and the generation-param whitelist in
one dependency-free module prevents the config validator, the backend and the
GUI from silently drifting apart.
"""

from __future__ import annotations

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
