"""Independent API Keys Settings page (#202 Phase D).

Owns Gemini and LiteLLM key chrome. Dialogs, keyring writes, and status
refresh stay on the host. No translator_config fields.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import (
    QCompleter,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
)

from ..widget_helpers import NoWheelComboBox, add_editable_combo_popup_action
from .page_chrome import build_settings_scroll_page, settings_group
from .page_contract import SettingsIssue, SettingsPageActions
from .registry import SETTINGS_PAGE_SPEC_OBJECTS

API_KEYS_PAGE_KEY = "api_keys"
_API_KEYS_SPEC = next(
    spec for spec in SETTINGS_PAGE_SPEC_OBJECTS if spec.key == API_KEYS_PAGE_KEY
)
API_KEYS_NAV_LABEL = _API_KEYS_SPEC.nav_label
API_KEYS_CONFIG_KEYS = _API_KEYS_SPEC.config_keys
API_KEYS_IMMEDIATE_ACTION_IDS = _API_KEYS_SPEC.immediate_action_ids

API_KEYS_WIDGET_ATTRS: tuple[str, ...] = (
    "api_status_label",
    "api_btn",
    "litellm_keys_provider_combo",
    "litellm_keys_manage_btn",
    "litellm_keys_status_label",
)


class ApiKeysSettingsPage(QObject):
    """Settings page for Gemini and LiteLLM credential chrome."""

    page_key = API_KEYS_PAGE_KEY
    nav_label = API_KEYS_NAV_LABEL
    config_keys = API_KEYS_CONFIG_KEYS
    immediate_action_ids = API_KEYS_IMMEDIATE_ACTION_IDS

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        actions: SettingsPageActions | None = None,
        on_manage_gemini_keys: Callable[[], None] | None = None,
        on_manage_litellm_keys: Callable[[], None] | None = None,
        on_provider_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._actions = actions or SettingsPageActions()
        self._on_manage_gemini_keys = on_manage_gemini_keys
        self._on_manage_litellm_keys = on_manage_litellm_keys
        self._on_provider_changed = on_provider_changed
        self._task_running = False
        self.widget, self.body = self._build_widgets()

    def set_action_callbacks(self, actions: SettingsPageActions) -> None:
        self._actions = actions

    def attach_widget_aliases(self, host_obj: object) -> None:
        target = getattr(host_obj, "__dict__", None)
        if not isinstance(target, dict):
            return
        for name in API_KEYS_WIDGET_ATTRS:
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
        if issue.field_key in {"manage_gemini_keys", "api_keys"}:
            self.api_btn.setFocus()
            return True
        if issue.field_key in {"manage_litellm_keys", "litellm_provider"}:
            self.litellm_keys_provider_combo.setFocus()
            return True
        return False

    def set_task_running(self, running: bool) -> None:
        self._task_running = bool(running)
        idle = not self._task_running
        self.api_btn.setEnabled(idle)
        self.litellm_keys_manage_btn.setEnabled(idle)
        self.litellm_keys_provider_combo.setEnabled(idle)

    def _emit_manage_gemini(self) -> None:
        if callable(self._on_manage_gemini_keys):
            self._on_manage_gemini_keys()

    def _emit_manage_litellm(self) -> None:
        if callable(self._on_manage_litellm_keys):
            self._on_manage_litellm_keys()

    def _emit_provider_changed(self) -> None:
        if callable(self._on_provider_changed):
            self._on_provider_changed()

    def _configure_provider_combo(self, combo: NoWheelComboBox) -> None:
        combo.setEditable(True)
        combo.setInsertPolicy(combo.InsertPolicy.NoInsert)
        add_editable_combo_popup_action(combo)
        completer = combo.completer()
        if completer is not None:
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        line_edit = combo.lineEdit()
        if line_edit is not None:
            line_edit.editingFinished.connect(self._emit_provider_changed)

    def _build_widgets(self) -> tuple[QScrollArea, QWidget]:
        page, body, layout = build_settings_scroll_page("settings_api_keys")

        api_box, api_layout = settings_group("Gemini API Key")
        api_hint = QLabel(
            "Gemini 密钥保存在本地 api_keys.json 中，不会上传或代理；"
            "也可通过环境变量配置。可添加多把 Key，供同步/批量轮换使用。"
        )
        api_hint.setWordWrap(True)
        api_hint.setObjectName("config_hint_label")
        api_layout.addWidget(api_hint)

        self.api_status_label = QLabel()
        self.api_status_label.setWordWrap(True)
        self.api_status_label.setObjectName("api_status_label")
        api_layout.addWidget(self.api_status_label)

        api_actions = QHBoxLayout()
        self.api_btn = QPushButton("管理 Gemini API Key")
        self.api_btn.setObjectName("api_btn")
        self.api_btn.clicked.connect(self._emit_manage_gemini)
        api_actions.addWidget(self.api_btn)
        api_actions.addStretch()
        api_layout.addLayout(api_actions)
        layout.addWidget(api_box)

        litellm_box, litellm_layout = settings_group("LiteLLM Provider 密钥")
        litellm_hint = QLabel(
            "LiteLLM 各供应商密钥保存在操作系统凭据管理器中，与 Gemini 完全分离，"
            "不会写入 api_keys.json 或 translator_config.json。\n"
            "列表包含常用供应商，以及你在 LiteLLM 页联网加载过的供应商；"
            "也可直接输入任意 Provider id（与 LiteLLM 页一致）。"
            "每个 Provider 可保存多把 Key，并指定「当前使用」的那一把。"
        )
        litellm_hint.setWordWrap(True)
        litellm_hint.setObjectName("config_hint_label")
        litellm_layout.addWidget(litellm_hint)

        provider_row = QWidget()
        provider_layout = QHBoxLayout(provider_row)
        provider_layout.setContentsMargins(0, 0, 0, 0)
        provider_layout.setSpacing(8)
        self.litellm_keys_provider_combo = NoWheelComboBox()
        self.litellm_keys_provider_combo.setObjectName("litellm_keys_provider_combo")
        self.litellm_keys_provider_combo.setMinimumWidth(180)
        self._configure_provider_combo(self.litellm_keys_provider_combo)
        self.litellm_keys_provider_combo.currentIndexChanged.connect(
            lambda _index: self._emit_provider_changed()
        )
        provider_layout.addWidget(self.litellm_keys_provider_combo, 1)
        self.litellm_keys_manage_btn = QPushButton("管理 Provider Key")
        self.litellm_keys_manage_btn.setObjectName("api_btn")
        self.litellm_keys_manage_btn.clicked.connect(self._emit_manage_litellm)
        provider_layout.addWidget(self.litellm_keys_manage_btn)
        litellm_layout.addWidget(provider_row)

        self.litellm_keys_status_label = QLabel()
        self.litellm_keys_status_label.setWordWrap(True)
        self.litellm_keys_status_label.setObjectName("api_status_label")
        litellm_layout.addWidget(self.litellm_keys_status_label)

        layout.addWidget(litellm_box)
        layout.addStretch(1)
        return page, body
