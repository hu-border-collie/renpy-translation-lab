"""Pure helpers for split-status table layout (no Qt dependency)."""
from __future__ import annotations

from typing import Literal

SPLIT_TABLE_GROUP_WIDTH = 6
SPLIT_ACTION_COLUMN_OFFSET = 5
SPLIT_ACTION_BUTTON_LABEL = "\u9009\u62e9"
SPLIT_ACTION_BUTTON_MIN_WIDTH = 72
SPLIT_ACTION_BUTTON_HEIGHT = 28
SPLIT_ACTION_CELL_MARGIN_H = 8
SPLIT_ACTION_CELL_MARGIN_V = 6

ButtonVisualState = Literal["normal", "hover", "pressed"]


def is_split_action_column(column: int) -> bool:
    return column % SPLIT_TABLE_GROUP_WIDTH == SPLIT_ACTION_COLUMN_OFFSET


def split_action_button_rect(
    cell_width: float,
    cell_height: float,
) -> tuple[float, float, float, float]:
    available_width = max(0.0, cell_width - (2 * SPLIT_ACTION_CELL_MARGIN_H))
    available_height = max(0.0, cell_height - (2 * SPLIT_ACTION_CELL_MARGIN_V))
    button_width = min(float(SPLIT_ACTION_BUTTON_MIN_WIDTH), available_width)
    button_height = min(float(SPLIT_ACTION_BUTTON_HEIGHT), available_height)
    left = SPLIT_ACTION_CELL_MARGIN_H + max(0.0, (available_width - button_width) / 2.0)
    top = SPLIT_ACTION_CELL_MARGIN_V + max(0.0, (available_height - button_height) / 2.0)
    return left, top, button_width, button_height


def split_action_item_payload(
    *,
    selectable: bool,
    manifest_path: str,
    part_label: str,
) -> dict[str, str] | None:
    if not selectable:
        return None
    return {
        "manifest_path": manifest_path,
        "part_label": part_label,
    }


def split_action_button_colors(*, dark: bool, state: ButtonVisualState) -> tuple[str, str, str]:
    if dark:
        palette = {
            "normal": ("#1e293b", "#475569", "#cbd5e1"),
            "hover": ("#334155", "#64748b", "#f8fafc"),
            "pressed": ("#0f172a", "#64748b", "#f8fafc"),
        }
    else:
        palette = {
            "normal": ("#ffffff", "#cbd5e1", "#475569"),
            "hover": ("#f8fafc", "#94a3b8", "#0f172a"),
            "pressed": ("#f1f5f9", "#94a3b8", "#0f172a"),
        }
    return palette[state]