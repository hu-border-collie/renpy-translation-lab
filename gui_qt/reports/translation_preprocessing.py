"""Saved preprocessing choices for the translation page; no readiness gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foundation.project_asset_paths import (
    resolve_configured_glossary_value,
    resolve_glossary_path,
    resolve_macro_setting_path,
)
from foundation.project_context_settings import resolve_batch_context_flags
from gui_qt.reports.bootstrap_report import coerce_bool
from gui_qt.user_copy import TRANSLATION_PREPROCESSING_COPY as COPY


@dataclass(frozen=True)
class TranslationPreprocessingSummary:
    """Configuration and asset presence, without claiming material readiness."""

    text: str
    details: str


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    return value if isinstance(value, Mapping) else {}


def _asset_status(path: str) -> str:
    try:
        found = bool(path) and Path(path).is_file()
    except (OSError, ValueError):
        return COPY["asset_unknown"]
    return COPY["asset_found"] if found else COPY["asset_missing"]


def summarize_translation_preprocessing(
    config: dict[str, Any] | None,
    *,
    game_root: str,
    strategy: str,
) -> TranslationPreprocessingSummary:
    """Read saved execution settings and project overrides using core readers.

    Only the small project override file and asset file presence are inspected.
    This never reads game scripts, validates stores, or initializes providers;
    an enabled switch is not evidence that retrieval is ready or has matched.
    """
    if not game_root:
        return TranslationPreprocessingSummary(COPY["no_project"], COPY["note"])
    if not isinstance(config, dict) or strategy not in {"gemini_batch", "sync"}:
        return TranslationPreprocessingSummary(COPY["unavailable"], COPY["note"])

    batch = strategy == "gemini_batch"
    scope = _section(config, "batch" if batch else "sync")
    flags = resolve_batch_context_flags(config, game_root)
    enabled = {
        "rag": (
            flags["rag_enabled"] if batch
            else coerce_bool(_section(scope, "rag").get("enabled"))
        ),
        "source_index": flags[
            "source_index_enabled" if batch else "sync_source_index_enabled"
        ],
        "story_memory": coerce_bool(_section(scope, "story_memory").get("enabled")),
        "project_analysis_brief": flags[
            "project_analysis_inject_enabled" if batch else "sync_project_analysis_inject_enabled"
        ],
    }
    unsupported = (
        {"story_memory", "project_analysis_brief"}
        if config.get("engine") == "rpgmaker_mv" else set()
    )
    labels = COPY["features"]
    on = [labels[key] for key, value in enabled.items() if value and key not in unsupported]
    off = [labels[key] for key, value in enabled.items() if not value and key not in unsupported]
    lines = [COPY["enabled"].format(items="、".join(on) or COPY["none"])]
    if off:
        lines.append(COPY["disabled"].format(items="、".join(off)))
    if unsupported:
        items = [
            labels[key] + (COPY["unsupported_enabled"] if value else "")
            for key, value in enabled.items() if key in unsupported
        ]
        lines.append(COPY["unsupported"].format(items="、".join(items)))

    glossary_path = resolve_glossary_path(
        resolve_configured_glossary_value(config), game_root=game_root,
    )
    macro_file = scope.get("macro_setting_file")
    # Sync has a default file; Batch only loads a file when configured.
    macro_path = (
        resolve_macro_setting_path(macro_file, game_root=game_root)
        if macro_file or not batch else ""
    )
    inline_macro = batch and any(
        isinstance(value, str) and bool(value.strip())
        for value in (scope.get("macro_setting"), config.get("batch_macro_setting"))
    )
    macro_status = COPY["inline_macro"] if inline_macro else (
        _asset_status(macro_path) if macro_path else COPY["not_configured"]
    )
    lines.append(COPY["assets"].format(glossary=_asset_status(glossary_path), macro=macro_status))
    details = [
        COPY["note"],
        COPY["glossary_path"].format(path=glossary_path),
        COPY["macro_path"].format(path=macro_path or COPY["not_configured"]),
    ]
    if batch:
        details.append(COPY["analysis_build"].format(
            state=COPY["on"] if flags["project_analysis_enabled"] else COPY["off"],
        ))
    return TranslationPreprocessingSummary("\n".join(lines), "\n".join(details))
