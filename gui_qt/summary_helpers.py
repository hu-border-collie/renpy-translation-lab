"""Shared helpers for GUI summary fact rendering."""
from __future__ import annotations

from .user_copy import format_notice_fact


def append_unique_fact(facts: list[str], fact: str) -> None:
    text = fact.strip()
    if text and text not in facts:
        facts.append(text)


def extend_facts_with_notices(facts: list[str], notices: list[str]) -> list[str]:
    merged = list(facts)
    for notice in notices:
        if isinstance(notice, str) and notice.strip():
            append_unique_fact(merged, format_notice_fact(notice))
    return merged