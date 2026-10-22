"""Shared GUI parsing for provider-neutral synchronous output diagnostics."""
from __future__ import annotations

import re


_TOKEN_SUMMARY_RE = re.compile(
    r"^Sync output tokens:\s*"
    r"completion=(\d+|unknown)\s+"
    r"reasoning=(\d+|unknown)\s+"
    r"text=(\d+|unknown)\s*$",
    re.MULTILINE,
)


def collect_sync_usage_facts(output: str) -> list[str]:
    """Return compact token and output-budget facts from stable CLI lines."""
    facts: list[str] = []
    token_match = _TOKEN_SUMMARY_RE.search(output)
    if token_match:
        completion, reasoning, text_output = token_match.groups()
        facts.append(
            "输出 Token："
            f"completion {completion} / reasoning {reasoning} / 正文 {text_output}"
        )
    warning_match = re.search(
        r"^Reasoning budget warnings:\s*(\d+)\s*$",
        output,
        re.MULTILINE,
    )
    if warning_match and int(warning_match.group(1)) > 0:
        facts.append(f"Reasoning 预算告警：{warning_match.group(1)} 次")
    truncated_match = re.search(
        r"^Truncated sync responses:\s*(\d+)\s*$",
        output,
        re.MULTILINE,
    )
    if truncated_match and int(truncated_match.group(1)) > 0:
        facts.append(f"输出截断：{truncated_match.group(1)} 次")
    return facts
