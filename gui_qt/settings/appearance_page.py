"""Independent Appearance Settings page (#202 Phase D).

Owns the ``theme`` combo. Theme preview is persist=False via a host callback.
Font download/install stays on the host (FontInstallWorker + confirmation).
Persistence still goes through MainWindow's single save transaction.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QWidget,
)

from ..theme_helpers import (
    THEME_DARK,
    THEME_LIGHT,
    THEME_SYSTEM,
    normalize_theme_preference,
)
from ..widget_helpers import NoWheelComboBox
from .page_chrome import build_settings_scroll_page, settings_form
from .page_contract import SettingsIssue, SettingsPageActions
from .registry import SETTINGS_PAGE_SPEC_OBJECTS

APPEARANCE_PAGE_KEY = "appearance"
_APPEARANCE_SPEC = next(
    spec for spec in SETTINGS_PAGE_SPEC_OBJECTS if spec.key == APPEARANCE_PAGE_KEY
)
APPEARANCE_NAV_LABEL = _APPEARANCE_SPEC.nav_label
APPEARANCE_CONFIG_KEYS = _APPEARANCE_SPEC.config_keys
APPEARANCE_IMMEDIATE_ACTION_IDS = _APPEARANCE_SPEC.immediate_action_ids

APPEARANCE_WIDGET_ATTRS: tuple[str, ...] = (
    "theme_combo",
    "font_install_status_label",
    "download_fonts_btn",
    "font_install_progress",
)

ThemePreviewCallback = Callable[[str], None]


class AppearanceSettingsPage(QObject):
    """Settings page for GUI theme preview and recommended-font install chrome."""

    page_key = APPEARANCE_PAGE_KEY
    nav_label = APPEARANCE_NAV_LABEL
    config_keys = APPEARANCE_CONFIG_KEYS
    immediate_action_ids = APPEARANCE_IMMEDIATE_ACTION_IDS

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        actions: SettingsPageActions | None = None,
        on_theme_preview: ThemePreviewCallback | None = None,
        on_download_fonts: Callable[[], None] | None = None,
        is_font_install_running: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self._actions = actions or SettingsPageActions()
        self._on_theme_preview = on_theme_preview
        self._on_download_fonts = on_download_fonts
        self._is_font_install_running = is_font_install_running
        self._loading = False
        self._task_running = False
        self._baseline: dict[str, object] = {}
        self.widget, self.body = self._build_widgets()
        self._baseline = dict(self.collect())

    def set_action_callbacks(self, actions: SettingsPageActions) -> None:
        self._actions = actions

    def attach_widget_aliases(self, host_obj: object) -> None:
        target = getattr(host_obj, "__dict__", None)
        if not isinstance(target, dict):
            return
        for name in APPEARANCE_WIDGET_ATTRS:
            target[name] = getattr(self, name)

    def load(self, snapshot: Mapping[str, object], *, restore: bool = False) -> None:
        previous = self._loading
        self._loading = True
        try:
            if "theme" in snapshot:
                self._set_theme_value(snapshot.get("theme"))
            if not restore:
                self._baseline = dict(self.collect())
        finally:
            self._loading = previous

    def collect(self) -> dict[str, object]:
        preference = self.theme_combo.currentData()
        if not isinstance(preference, str):
            preference = THEME_SYSTEM
        return {"theme": normalize_theme_preference(preference)}

    def validate(self) -> Sequence[SettingsIssue]:
        return []

    def reset(self) -> None:
        if not self._baseline:
            return
        self.load(self._baseline)
        # load() blocks combo signals, so re-apply the baseline preview on
        # the host (QSS / _theme_preference).
        self._emit_theme_preview(self.theme_combo.currentIndex())

    def focus_issue(self, issue: SettingsIssue) -> bool:
        if issue.field_key == "theme":
            self.theme_combo.setFocus()
            return True
        return False

    def set_task_running(self, running: bool) -> None:
        self._task_running = bool(running)
        idle = not self._task_running
        self.theme_combo.setEnabled(idle)
        font_busy = bool(
            self._is_font_install_running and self._is_font_install_running()
        )
        self.download_fonts_btn.setEnabled(idle or font_busy)

    def _set_theme_value(self, value: object) -> None:
        theme = normalize_theme_preference(value)
        idx = self.theme_combo.findData(theme)
        if idx >= 0:
            previous = self.theme_combo.blockSignals(True)
            try:
                self.theme_combo.setCurrentIndex(idx)
            finally:
                self.theme_combo.blockSignals(previous)

    def _emit_theme_preview(self, _index: int) -> None:
        if self._loading:
            return
        preference = self.theme_combo.currentData()
        if not isinstance(preference, str):
            return
        if callable(self._on_theme_preview):
            self._on_theme_preview(preference)

    def _emit_download_fonts(self) -> None:
        if callable(self._on_download_fonts):
            self._on_download_fonts()

    def _build_widgets(self) -> tuple[QScrollArea, QWidget]:
        page, body, layout = build_settings_scroll_page("settings_appearance")

        appearance_box = QGroupBox("外观")
        appearance_layout = settings_form(appearance_box)
        self.theme_combo = NoWheelComboBox()
        self.theme_combo.setObjectName("theme_combo")
        self.theme_combo.addItem("跟随系统", THEME_SYSTEM)
        self.theme_combo.addItem("浅色", THEME_LIGHT)
        self.theme_combo.addItem("深色", THEME_DARK)
        self.theme_combo.currentIndexChanged.connect(self._emit_theme_preview)
        appearance_layout.addRow("主题：", self.theme_combo)
        hint = QLabel("切换主题会立即预览；点击保存设置后才会写入 translator_config.json。")
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        appearance_layout.addRow("", hint)
        layout.addWidget(appearance_box)

        fonts_box = QGroupBox("推荐字体")
        fonts_layout = settings_form(fonts_box)
        self.font_install_status_label = QLabel()
        self.font_install_status_label.setWordWrap(True)
        self.font_install_status_label.setObjectName("config_hint_label")
        fonts_layout.addRow("安装状态：", self.font_install_status_label)

        font_actions = QWidget()
        font_actions_layout = QHBoxLayout(font_actions)
        font_actions_layout.setContentsMargins(0, 0, 0, 0)
        font_actions_layout.setSpacing(8)
        self.download_fonts_btn = QPushButton("下载推荐字体")
        self.download_fonts_btn.setObjectName("secondary_btn")
        self.download_fonts_btn.setToolTip(
            "从华为和霞鹜文楷官方来源下载固定版本字体，并执行 SHA-256 校验。"
            "下载中点击可停止。"
        )
        self.download_fonts_btn.clicked.connect(self._emit_download_fonts)
        font_actions_layout.addWidget(self.download_fonts_btn)
        font_actions_layout.addStretch(1)
        fonts_layout.addRow("", font_actions)

        self.font_install_progress = QProgressBar()
        self.font_install_progress.setObjectName("font_install_progress")
        self.font_install_progress.setRange(0, 0)
        self.font_install_progress.setFormat("正在后台下载并校验推荐字体…")
        self.font_install_progress.setVisible(False)
        fonts_layout.addRow(self.font_install_progress)
        layout.addWidget(fonts_box)
        layout.addStretch(1)
        return page, body
