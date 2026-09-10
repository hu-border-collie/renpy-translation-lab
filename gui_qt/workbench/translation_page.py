"""Unified translation target selector (#348 P3).

One navigation entry owns the translation workflow. Both execution pages host
this selector, which presents the primary ModelProfile and the
ExecutionStrategy it supports; switching the strategy switches the visible
execution page. The module is presentation only: it never resolves credentials
or decides writeback safety.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..user_copy import TRANSLATION_TARGET_COPY
from ..widget_helpers import NoWheelComboBox

_STRATEGY_ORDER = ("sync", "gemini_batch")


def strategy_label(strategy: str) -> str:
    return TRANSLATION_TARGET_COPY["strategy_labels"].get(strategy, strategy)


class TranslationTargetSection(QFrame):
    """Profile/strategy selector shared by both translation executions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("translation_target_section")
        self._on_select: Callable[[str, str], None] | None = None
        self._entries: dict[str, dict[str, Any]] = {}
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title = QLabel(TRANSLATION_TARGET_COPY["section_title"])
        title.setObjectName("translation_target_title")
        layout.addWidget(title)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        row_layout.addWidget(QLabel(TRANSLATION_TARGET_COPY["profile_label"]))
        self.profile_combo = NoWheelComboBox()
        self.profile_combo.setObjectName("translation_profile_combo")
        self.profile_combo.setMinimumWidth(200)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        row_layout.addWidget(self.profile_combo, 1)
        row_layout.addWidget(QLabel(TRANSLATION_TARGET_COPY["strategy_label"]))
        self.strategy_combo = NoWheelComboBox()
        self.strategy_combo.setObjectName("translation_strategy_combo")
        self.strategy_combo.setMinimumWidth(140)
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        row_layout.addWidget(self.strategy_combo, 1)
        layout.addWidget(row)

        self.hint_label = QLabel("")
        self.hint_label.setObjectName("translation_target_hint")
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        self.set_enabled(False)

    # -- external API ---------------------------------------------------

    def set_select_callback(self, callback: Callable[[str, str], None] | None) -> None:
        self._on_select = callback

    def set_target_choices(self, payload: Mapping[str, Any] | None) -> None:
        """Render selector data from ``model_routing_reader`` (or a hint)."""
        payload = dict(payload or {})
        entries = [
            dict(entry)
            for entry in (payload.get("profiles") or [])
            if isinstance(entry, Mapping)
        ]
        self._entries = {
            str(entry.get("id") or ""): entry for entry in entries
        }
        hint = str(payload.get("hint") or "")
        selected_profile = str(payload.get("selected_profile_id") or "")
        selected_strategy = str(payload.get("selected_strategy") or "")

        self._updating = True
        try:
            self.profile_combo.clear()
            for entry in entries:
                profile_id = str(entry.get("id") or "")
                if not profile_id:
                    continue
                label = str(entry.get("label") or profile_id)
                model = str(entry.get("model") or "")
                self.profile_combo.addItem(f"{label}（{model}）", profile_id)
            profile_index = self._index_for(self.profile_combo, selected_profile)
            if profile_index >= 0:
                self.profile_combo.setCurrentIndex(profile_index)
            self._rebuild_strategies(selected_strategy)
        finally:
            self._updating = False

        enabled = bool(entries) and not payload.get("disabled", False)
        self.set_enabled(enabled)
        if not enabled:
            self.hint_label.setText(hint or TRANSLATION_TARGET_COPY["empty_profiles"])
        else:
            self._refresh_hint(hint)

    def set_enabled(self, enabled: bool) -> None:
        self.profile_combo.setEnabled(bool(enabled))
        self.strategy_combo.setEnabled(bool(enabled))

    def current_selection(self) -> tuple[str, str]:
        return (
            str(self.profile_combo.currentData() or ""),
            str(self.strategy_combo.currentData() or ""),
        )

    # -- internals ------------------------------------------------------

    @staticmethod
    def _index_for(combo: QComboBox, value: str) -> int:
        for index in range(combo.count()):
            if str(combo.itemData(index) or "") == value:
                return index
        return -1

    def _entry(self, profile_id: str) -> dict[str, Any]:
        return dict(self._entries.get(profile_id) or {})

    def _rebuild_strategies(self, preferred: str = "") -> None:
        profile_id = str(self.profile_combo.currentData() or "")
        entry = self._entry(profile_id)
        supported = [str(item) for item in entry.get("strategies") or ()]
        unsupported = dict(entry.get("unsupported") or {})
        self.strategy_combo.clear()
        for strategy in _STRATEGY_ORDER:
            is_supported = strategy in supported
            label = strategy_label(strategy)
            if not is_supported:
                reason = str(
                    unsupported.get(strategy)
                    or TRANSLATION_TARGET_COPY["unsupported_reasons"].get(
                        strategy,
                        "",
                    )
                )
                label = (
                    f"{label}（不可用：{reason}）"
                    if reason
                    else f"{label}（不可用）"
                )
            self.strategy_combo.addItem(label, strategy)
            if not is_supported:
                item = self.strategy_combo.model().item(self.strategy_combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
        index = self._index_for(self.strategy_combo, preferred)
        if index < 0:
            for candidate in _STRATEGY_ORDER:
                index = self._index_for(self.strategy_combo, candidate)
                if index >= 0 and candidate in supported:
                    break
        if index >= 0:
            # A disabled entry stays selected so the hint can explain the
            # missing capability; the user can pick a supported strategy.
            self.strategy_combo.setCurrentIndex(index)

    def _refresh_hint(self, prefix: str = "") -> None:
        profile_id, strategy = self.current_selection()
        entry = self._entry(profile_id)
        model = str(entry.get("model") or "")
        supported = entry.get("strategies") or ()
        lines: list[str] = []
        if prefix:
            lines.append(prefix)
        if model and strategy:
            resolved = TRANSLATION_TARGET_COPY["resolved"].format(
                model=model,
                strategy=strategy_label(strategy),
            )
            if strategy not in supported:
                reason = str(
                    (entry.get("unsupported") or {}).get(strategy) or "缺少能力"
                )
                reason = TRANSLATION_TARGET_COPY["unsupported_reasons"].get(
                    reason,
                    reason,
                )
                resolved = (
                    f"{resolved}（"
                    + TRANSLATION_TARGET_COPY["unsupported_hint"].format(
                        strategy=strategy_label(strategy),
                        reason=reason,
                    )
                    + "）"
                )
            lines.append(resolved)
        hint = "；".join(lines)
        self.hint_label.setText(hint)
        self.hint_label.setToolTip(hint)

    def _on_profile_changed(self, _index: int) -> None:
        if self._updating:
            return
        preferred = str(self.strategy_combo.currentData() or "")
        self._updating = True
        try:
            self._rebuild_strategies(preferred)
        finally:
            self._updating = False
        self._emit_selection()

    def _on_strategy_changed(self, _index: int) -> None:
        if self._updating:
            return
        if self.strategy_combo.isEnabled():
            self._emit_selection()
        else:
            self._refresh_hint()

    def _emit_selection(self) -> None:
        profile_id, strategy = self.current_selection()
        if not profile_id or not strategy:
            return
        self._refresh_hint()
        supported = self._entry(profile_id).get("strategies") or ()
        if strategy not in supported:
            return
        if self._on_select is not None:
            self._on_select(profile_id, strategy)
