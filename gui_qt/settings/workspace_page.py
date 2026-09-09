"""Independent Workspace Settings page (#202 Phase D).

Owns 项目列表 chrome and embeds GamesRegistryPanel. Refresh/import workers
stay on the panel. Project switch and workspace persist stay on the host.
No translator_config fields.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

from ..games_registry_panel import GamesRegistryPanel
from .page_chrome import build_settings_scroll_page
from .page_contract import SettingsIssue, SettingsPageActions
from .registry import SETTINGS_PAGE_SPEC_OBJECTS

WORKSPACE_PAGE_KEY = "workspace"
_WORKSPACE_SPEC = next(
    spec for spec in SETTINGS_PAGE_SPEC_OBJECTS if spec.key == WORKSPACE_PAGE_KEY
)
WORKSPACE_NAV_LABEL = _WORKSPACE_SPEC.nav_label
WORKSPACE_CONFIG_KEYS = _WORKSPACE_SPEC.config_keys
WORKSPACE_IMMEDIATE_ACTION_IDS = _WORKSPACE_SPEC.immediate_action_ids

WORKSPACE_WIDGET_ATTRS: tuple[str, ...] = ("_games_registry_panel",)

SwitchProjectHandler = Callable[[str], bool]
DoctorReportProvider = Callable[[], dict | None]
WorkspaceChangedHandler = Callable[[Path], None]


class WorkspaceSettingsPage(QObject):
    """Settings page for the workspace project list."""

    page_key = WORKSPACE_PAGE_KEY
    nav_label = WORKSPACE_NAV_LABEL
    config_keys = WORKSPACE_CONFIG_KEYS
    immediate_action_ids = WORKSPACE_IMMEDIATE_ACTION_IDS

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        actions: SettingsPageActions | None = None,
        workspace_root: Path | None = None,
        current_game_root: Path | None = None,
        get_doctor_report: DoctorReportProvider | None = None,
        on_switch_project: SwitchProjectHandler | None = None,
        on_workspace_changed: WorkspaceChangedHandler | None = None,
    ) -> None:
        super().__init__(parent)
        self._actions = actions or SettingsPageActions()
        self._workspace_root = workspace_root
        self._current_game_root = current_game_root
        self._get_doctor_report = get_doctor_report
        self._on_switch_project = on_switch_project
        self._on_workspace_changed = on_workspace_changed
        self.widget, self.body = self._build_widgets()

    def set_action_callbacks(self, actions: SettingsPageActions) -> None:
        self._actions = actions

    def attach_widget_aliases(self, host_obj: object) -> None:
        target = getattr(host_obj, "__dict__", None)
        if not isinstance(target, dict):
            return
        for name in WORKSPACE_WIDGET_ATTRS:
            target[name] = getattr(self, name)

    def load(self, snapshot: Mapping[str, object], *, restore: bool = False) -> None:
        return None

    def collect(self) -> dict[str, object]:
        return {}

    def validate(self) -> Sequence[SettingsIssue]:
        return []

    def reset(self) -> None:
        return None

    def focus_issue(self, issue: SettingsIssue) -> bool:
        if issue.field_key in {
            "switch_project",
            "refresh_registry",
            "import_projects",
            "workspace",
        }:
            self._games_registry_panel.setFocus()
            return True
        return False

    def set_task_running(self, running: bool) -> None:
        self._games_registry_panel.set_host_task_running(running)

    def _build_widgets(self) -> tuple[QScrollArea, QWidget]:
        page, body, layout = build_settings_scroll_page("settings_workspace")
        # Fill the settings viewport so the project table can grow; the table
        # itself scrolls instead of crushing into a short strip.
        body.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        content = page.widget()
        if content is not None:
            content.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )

        hint = QLabel(
            "工作区默认未设置，不会自动使用工具目录的上一级。"
            "先「创建 / 接入工作区…」预览并初始化/接入总表，再扫描、导入或切换项目。"
            "「切换到此项目」会写入当前 game_root 并留在本页；术语表 / 准备流程等到「项目」分区调整。"
            "总览表与详情可拖拽分隔；扫描新项目、导入与 GAMES.md 等操作在「维护」中展开。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        self._games_registry_panel = GamesRegistryPanel(
            None,
            workspace_root=self._workspace_root,
            current_game_root=self._current_game_root,
            get_doctor_report=self._get_doctor_report,
            on_switch_project=self._on_switch_project,
            on_workspace_changed=self._on_workspace_changed,
        )
        self._games_registry_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self._games_registry_panel, 1)
        return page, body
