# -*- coding: utf-8 -*-
"""Single source of truth for Gemini model IDs used by GUI and CLI.

Built-in lists live here. Users can extend them without code changes via
``translator_config.json``:

.. code-block:: json

    {
      "model_catalog": {
        "gemini": ["gemini-experimental-foo"],
        "gemini_embedding": ["gemini-embedding-experimental"]
      }
    }

Resolved UI/CLI lists = builtins (order preserved) + catalog extras + any
currently selected model IDs that are not already present.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

# Default selection for sync/batch when config omits model.
DEFAULT_GEMINI_TRANSLATION_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# GUI display order: newer / primary options first.
BUILTIN_GEMINI_TRANSLATION_MODELS: tuple[str, ...] = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)

BUILTIN_GEMINI_EMBEDDING_MODELS: tuple[str, ...] = (
    "gemini-embedding-2",
    "gemini-embedding-001",
)

# Config keys under translator_config["model_catalog"].
CATALOG_SECTION = "model_catalog"
CATALOG_GEMINI_KEY = "gemini"
CATALOG_EMBEDDING_KEY = "gemini_embedding"


def normalize_model_names(values: Any) -> list[str]:
    """Coerce config/list/string values into a de-duplicated model name list."""
    if values is None:
        return []
    if isinstance(values, str):
        items: Iterable[Any] = [values]
    elif isinstance(values, (list, tuple, set)):
        items = values
    else:
        return []

    seen: set[str] = set()
    result: list[str] = []
    for value in items:
        name = str(value).strip() if value is not None else ""
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def merge_model_lists(*lists: Sequence[str] | None) -> list[str]:
    """Merge model lists left-to-right; first occurrence wins."""
    seen: set[str] = set()
    result: list[str] = []
    for group in lists:
        if not group:
            continue
        for name in group:
            cleaned = str(name).strip() if name is not None else ""
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
    return result


def default_model_rotation_list() -> list[str]:
    """CLI rotation/fallback list: default model first, then remaining builtins."""
    return merge_model_lists(
        [DEFAULT_GEMINI_TRANSLATION_MODEL],
        BUILTIN_GEMINI_TRANSLATION_MODELS,
    )


def read_model_catalog_section(translator_config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(translator_config, Mapping):
        return {}
    section = translator_config.get(CATALOG_SECTION)
    return dict(section) if isinstance(section, Mapping) else {}


def catalog_extra_models(
    translator_config: Mapping[str, Any] | None,
    *,
    kind: str = "translation",
) -> list[str]:
    section = read_model_catalog_section(translator_config)
    key = CATALOG_EMBEDDING_KEY if kind == "embedding" else CATALOG_GEMINI_KEY
    return normalize_model_names(section.get(key))


def resolve_gemini_translation_models(
    translator_config: Mapping[str, Any] | None = None,
    *,
    extra_selected: Sequence[str] | None = None,
) -> list[str]:
    """Builtin translation models + config extras + currently selected IDs."""
    return merge_model_lists(
        BUILTIN_GEMINI_TRANSLATION_MODELS,
        catalog_extra_models(translator_config, kind="translation"),
        normalize_model_names(extra_selected),
    )


def resolve_gemini_embedding_models(
    translator_config: Mapping[str, Any] | None = None,
    *,
    extra_selected: Sequence[str] | None = None,
) -> list[str]:
    """Builtin embedding models + config extras + currently selected IDs."""
    return merge_model_lists(
        BUILTIN_GEMINI_EMBEDDING_MODELS,
        catalog_extra_models(translator_config, kind="embedding"),
        normalize_model_names(extra_selected),
    )


def extras_beyond_builtins(
    models: Sequence[str] | None,
    *,
    kind: str = "translation",
) -> list[str]:
    """Return names not in the built-in catalog (for persisting user extensions)."""
    builtins = (
        set(BUILTIN_GEMINI_EMBEDDING_MODELS)
        if kind == "embedding"
        else set(BUILTIN_GEMINI_TRANSLATION_MODELS)
    )
    result: list[str] = []
    seen: set[str] = set()
    for name in normalize_model_names(models):
        if name in builtins or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def write_model_catalog_extras(
    translator_config: dict[str, Any],
    *,
    translation_models: Sequence[str] | None = None,
    embedding_models: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Persist non-builtin model names under ``model_catalog``.

    Empty extra lists remove the corresponding key; an empty catalog section is
    removed from the config.
    """
    section = read_model_catalog_section(translator_config)

    if translation_models is not None:
        extras = extras_beyond_builtins(translation_models, kind="translation")
        if extras:
            section[CATALOG_GEMINI_KEY] = extras
        else:
            section.pop(CATALOG_GEMINI_KEY, None)

    if embedding_models is not None:
        extras = extras_beyond_builtins(embedding_models, kind="embedding")
        if extras:
            section[CATALOG_EMBEDDING_KEY] = extras
        else:
            section.pop(CATALOG_EMBEDDING_KEY, None)

    if section:
        translator_config[CATALOG_SECTION] = section
    else:
        translator_config.pop(CATALOG_SECTION, None)
    return translator_config
