"""Declarative column model for the workspace project list table.

Design basis (EUI column sizing + Carbon resource-list structure):

* Predefined status columns size to the longest known label (header never
  truncated either).
* Dynamic long text (path) is the single flex column and is always last so
  Qt ``StretchLastSection`` works without middle-Stretch resize fights.
* Interactive column widths are persisted by stable column id, not index.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from games_registry import (
    LAYOUT_STATUS_LABELS,
    PLAY_STATUSES,
    TRANSLATION_STATUSES,
)

ColumnRole = Literal["identity", "enum", "short", "status", "path"]


class _RegistryRowLike(Protocol):
    name: str
    path: str
    version: str
    layout_status: str
    play_status: str
    translation_status: str

# New preference key (id → px). Legacy key still read for one release.
REGISTRY_PREF_TABLE_COLUMN_WIDTHS = "table_column_widths"
REGISTRY_PREF_TABLE_COLUMN_WIDTHS_LEGACY = "table_column_width_fractions"

_CELL_PAD_PX = 28
_MIN_ABSOLUTE_PX = 56

# Old column order (index → id) for migrating index-based width maps:
# 项目 | 路径 | 版本 | 目录状态 | 游玩 | 翻译
_LEGACY_INDEX_TO_ID: dict[int, str] = {
    0: "name",
    1: "path",  # flex — discarded when persisting
    2: "version",
    3: "layout",
    4: "play",
    5: "translation",
}


@dataclass(frozen=True)
class RegistryTableColumn:
    """One table column preset (EUI-style sizing metadata)."""

    id: str
    title: str
    role: ColumnRole
    default_width: int
    max_width: int | None = None
    flex: bool = False
    enum_samples: tuple[str, ...] = ()
    min_width_hint: int = _MIN_ABSOLUTE_PX


def _layout_status_samples() -> tuple[str, ...]:
    labels = tuple(LAYOUT_STATUS_LABELS.values())
    return labels + ("待确认",)


def _play_status_samples() -> tuple[str, ...]:
    return tuple(sorted(PLAY_STATUSES))


def _translation_status_samples() -> tuple[str, ...]:
    # Closed set only; free-form annotations may still ellipsis (tooltip shows full).
    return tuple(sorted(TRANSLATION_STATUSES))


# Scannable left cluster + one flex path last (Carbon resource list / EUI flex).
REGISTRY_TABLE_COLUMN_DEFS: tuple[RegistryTableColumn, ...] = (
    RegistryTableColumn(
        id="name",
        title="项目",
        role="identity",
        default_width=160,
        max_width=280,
        min_width_hint=88,
    ),
    RegistryTableColumn(
        id="layout",
        title="目录状态",
        role="enum",
        default_width=124,
        enum_samples=_layout_status_samples(),
        min_width_hint=88,
    ),
    RegistryTableColumn(
        id="version",
        title="版本",
        role="short",
        default_width=100,
        max_width=160,
        min_width_hint=72,
    ),
    RegistryTableColumn(
        id="play",
        title="游玩",
        role="enum",
        default_width=88,
        enum_samples=_play_status_samples(),
        min_width_hint=72,
    ),
    RegistryTableColumn(
        id="translation",
        title="翻译",
        role="status",
        default_width=120,
        max_width=220,
        enum_samples=_translation_status_samples(),
        min_width_hint=72,
    ),
    RegistryTableColumn(
        id="path",
        title="路径",
        role="path",
        default_width=240,
        flex=True,
        min_width_hint=96,
    ),
)

# Backward-compatible header tuple (titles only).
REGISTRY_TABLE_COLUMNS: tuple[str, ...] = tuple(
    column.title for column in REGISTRY_TABLE_COLUMN_DEFS
)

REGISTRY_TABLE_PATH_COLUMN: int = next(
    index
    for index, column in enumerate(REGISTRY_TABLE_COLUMN_DEFS)
    if column.flex
)

# Interactive (non-flex) defaults keyed by index — for callers that still want index maps.
REGISTRY_TABLE_DEFAULT_WIDTHS: dict[int, int] = {
    index: column.default_width
    for index, column in enumerate(REGISTRY_TABLE_COLUMN_DEFS)
    if not column.flex
}


def column_headers() -> list[str]:
    return list(REGISTRY_TABLE_COLUMNS)


def flex_column_index() -> int:
    return REGISTRY_TABLE_PATH_COLUMN


def interactive_column_indexes() -> list[int]:
    return [
        index
        for index, column in enumerate(REGISTRY_TABLE_COLUMN_DEFS)
        if not column.flex
    ]


def interactive_column_ids() -> list[str]:
    return [column.id for column in REGISTRY_TABLE_COLUMN_DEFS if not column.flex]


def column_by_id(column_id: str) -> RegistryTableColumn | None:
    for column in REGISTRY_TABLE_COLUMN_DEFS:
        if column.id == column_id:
            return column
    return None


def column_at(index: int) -> RegistryTableColumn | None:
    if 0 <= index < len(REGISTRY_TABLE_COLUMN_DEFS):
        return REGISTRY_TABLE_COLUMN_DEFS[index]
    return None


def default_width_map() -> dict[str, int]:
    return {
        column.id: column.default_width
        for column in REGISTRY_TABLE_COLUMN_DEFS
        if not column.flex
    }


def row_cell_values(row: _RegistryRowLike) -> list[str]:
    """Cell strings in current column order."""
    values_by_id = {
        "name": row.name,
        "layout": row.layout_status,
        "version": row.version,
        "play": row.play_status,
        "translation": row.translation_status,
        "path": row.path,
    }
    return [values_by_id[column.id] for column in REGISTRY_TABLE_COLUMN_DEFS]


def min_width_for_column(
    column: RegistryTableColumn,
    advance: Callable[[str], int],
    *,
    pad: int = _CELL_PAD_PX,
) -> int:
    """EUI-style min: header fully readable; enum columns fit longest sample."""
    candidates = [column.title, *column.enum_samples]
    text_px = max((advance(text) for text in candidates), default=0)
    return max(column.min_width_hint, _MIN_ABSOLUTE_PX, text_px + pad)


def clamp_width(
    column: RegistryTableColumn,
    width: int,
    *,
    min_width: int,
) -> int:
    value = max(min_width, int(width))
    if column.max_width is not None and not column.flex:
        # User drag may exceed max; fit-to-content respects max. Drag keeps min only.
        pass
    return value


def clamp_width_for_fit(
    column: RegistryTableColumn,
    width: int,
    *,
    min_width: int,
) -> int:
    """Cap fit-to-content so one long cell does not crush the flex path."""
    value = max(min_width, int(width))
    cap = column.max_width if column.max_width is not None else 360
    return min(value, cap)


def migrate_stored_widths(raw: Any) -> dict[str, int]:
    """Normalize preference payload to id → px (interactive columns only)."""
    defaults = default_width_map()
    if raw is None:
        return dict(defaults)

    loaded: dict[str, int] = {}

    if isinstance(raw, Mapping):
        keys = [str(key) for key in raw.keys()]
        # New shape: column ids.
        if any(key in defaults for key in keys):
            for key, value in raw.items():
                column_id = str(key)
                if column_id not in defaults:
                    continue
                try:
                    width = int(value)
                except (TypeError, ValueError):
                    continue
                if width > 0:
                    loaded[column_id] = width
        else:
            # Legacy: index → px (string or int keys).
            for key, value in raw.items():
                try:
                    index = int(key)
                    width = int(value)
                except (TypeError, ValueError):
                    continue
                column_id = _LEGACY_INDEX_TO_ID.get(index)
                if column_id is None or column_id not in defaults or width <= 0:
                    continue
                loaded[column_id] = width
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        # Legacy relative fractions for old 6-column order.
        if len(raw) == len(_LEGACY_INDEX_TO_ID):
            for index, frac in enumerate(raw):
                column_id = _LEGACY_INDEX_TO_ID.get(index)
                if column_id is None or column_id not in defaults:
                    continue
                try:
                    width = int(960 * float(frac))
                except (TypeError, ValueError):
                    continue
                if width > 0:
                    loaded[column_id] = width

    return {
        column_id: loaded.get(column_id, default)
        for column_id, default in defaults.items()
    }


def widths_for_persist(widths: Mapping[str, int]) -> dict[str, int]:
    """Drop flex/unknown ids; keep interactive ids only."""
    defaults = default_width_map()
    payload: dict[str, int] = {}
    for column_id in defaults:
        value = widths.get(column_id)
        if value is None:
            continue
        try:
            width = int(value)
        except (TypeError, ValueError):
            continue
        if width > 0:
            payload[column_id] = width
    return payload
