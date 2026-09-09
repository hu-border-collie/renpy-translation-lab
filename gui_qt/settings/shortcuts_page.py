"""Independent Shortcuts Settings page (#202 Phase D).

Read-only catalog of global GUI keyboard shortcuts. No translator_config
fields and no local worker. Navigation rows come from the host shell IA.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QGroupBox, QLabel, QScrollArea, QWidget

from .page_chrome import build_settings_scroll_page, settings_form
from .page_contract import SettingsIssue, SettingsPageActions
from .registry import SETTINGS_PAGE_SPEC_OBJECTS

SHORTCUTS_PAGE_KEY = "shortcuts"
_SHORTCUTS_SPEC = next(
    spec for spec in SETTINGS_PAGE_SPEC_OBJECTS if spec.key == SHORTCUTS_PAGE_KEY
)
SHORTCUTS_NAV_LABEL = _SHORTCUTS_SPEC.nav_label
SHORTCUTS_CONFIG_KEYS = _SHORTCUTS_SPEC.config_keys
SHORTCUTS_IMMEDIATE_ACTION_IDS = _SHORTCUTS_SPEC.immediate_action_ids

ShortcutRow = tuple[str, str]
ShortcutGroup = tuple[str, list[ShortcutRow]]


def shortcut_catalog(nav_labels: Sequence[str] | None = None) -> list[ShortcutGroup]:
    """Human-readable shortcut groups for Settings · 快捷键.

    Mirrors the current shell IA (sidebar routes + page-header diagnostics),
    not the legacy hidden main-tab indices.
    """
    nav_rows: list[ShortcutRow] = []
    for index, label in enumerate(nav_labels or (), start=1):
        if index > 9:
            break
        nav_rows.append((f"Ctrl+{index}", f"打开「{label}」"))
    nav_rows.append(("Ctrl+0", "打开「诊断与运行日志」"))
    return [
        (
            "任务",
            [
                ("Ctrl+D", "环境检查（运行中变为停止检查）"),
                (
                    "Ctrl+T",
                    "开始当前任务（翻译 / 生成模板 / 提取等，取决于当前页）",
                ),
                (
                    "Ctrl+K",
                    "停止当前任务（含环境检查、准备工作目录、生成模板、预建）",
                ),
            ],
        ),
        ("导航", nav_rows),
        (
            "日志与设置",
            [
                ("Ctrl+L", "打开「诊断与运行日志」"),
                ("Ctrl+Shift+L", "清空诊断日志输出"),
                ("Ctrl+S", "保存设置（仅在「设置」页有效）"),
            ],
        ),
    ]


class ShortcutsSettingsPage(QObject):
    """Read-only Settings page listing current global keyboard shortcuts."""

    page_key = SHORTCUTS_PAGE_KEY
    nav_label = SHORTCUTS_NAV_LABEL
    config_keys = SHORTCUTS_CONFIG_KEYS
    immediate_action_ids = SHORTCUTS_IMMEDIATE_ACTION_IDS

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        actions: SettingsPageActions | None = None,
        nav_labels: Sequence[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self._actions = actions or SettingsPageActions()
        self._nav_labels = tuple(nav_labels or ())
        self.widget, self.body = self._build_widgets()

    def set_action_callbacks(self, actions: SettingsPageActions) -> None:
        self._actions = actions

    def load(self, snapshot: Mapping[str, object], *, restore: bool = False) -> None:
        return None

    def collect(self) -> dict[str, object]:
        return {}

    def validate(self) -> Sequence[SettingsIssue]:
        return []

    def reset(self) -> None:
        return None

    def focus_issue(self, issue: SettingsIssue) -> bool:
        return False

    def set_task_running(self, running: bool) -> None:
        return None

    def _build_widgets(self) -> tuple[QScrollArea, QWidget]:
        page, body, layout = build_settings_scroll_page("settings_shortcuts")
        intro = QLabel(
            "以下为图形工作台当前生效的全局快捷键。"
            "任务类快捷键会在对应按钮禁用时同步关闭；"
            "导航类快捷键在任务运行中会遵循锁定规则（不可切换到其它任务页）。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("config_hint_label")
        layout.addWidget(intro)

        for group_title, rows in shortcut_catalog(self._nav_labels):
            box = QGroupBox(group_title)
            form = settings_form(box)
            for key, meaning in rows:
                value = QLabel(meaning)
                value.setWordWrap(True)
                value.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                )
                form.addRow(f"{key}：", value)
            layout.addWidget(box)

        layout.addStretch(1)
        return page, body
