"""Independent Model Profiles Settings page (#348 P3).

Owns the versioned ``model_routing`` section and renders the unified provider /
profile / strategy / stage-route surface. All mutations go through the Qt-free
``model_profiles_editor`` core; the page never resolves or stores credentials,
and the host keeps the single save transaction and file write.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import model_profiles_editor as editor
from ..user_copy import MODEL_PROFILES_PAGE_COPY
from ..widget_helpers import NoWheelComboBox
from .page_chrome import build_settings_scroll_page, settings_group
from .page_contract import SettingsIssue, SettingsPageActions
from .registry import SETTINGS_PAGE_SPEC_OBJECTS

PROFILES_PAGE_KEY = "profiles"
_PROFILES_SPEC = next(
    spec for spec in SETTINGS_PAGE_SPEC_OBJECTS if spec.key == PROFILES_PAGE_KEY
)
PROFILES_NAV_LABEL = _PROFILES_SPEC.nav_label
PROFILES_CONFIG_KEYS = _PROFILES_SPEC.config_keys
PROFILES_IMMEDIATE_ACTION_IDS = _PROFILES_SPEC.immediate_action_ids

_CAPABILITY_LABELS = {
    "sync_generation": "同步生成",
    "reasoning_request": "Reasoning 请求",
    "reasoning_response": "Reasoning 返回",
    "usage_stats": "Usage 统计",
    "remote_batch": "远程 Batch",
    "embedding": "Embedding",
}

_FIELD_WIDGETS = {
    "model_routing": "profiles_list",
}

_CONTEXT_LABELS = {
    "context_limit_tokens": "上下文上限 tokens",
    "context_budget_tokens": "上下文预算 tokens",
}


class ProfilesSettingsPage(QObject):
    """Settings page for Model Profiles, providers and stage routes."""

    page_key = PROFILES_PAGE_KEY
    nav_label = PROFILES_NAV_LABEL
    config_keys = PROFILES_CONFIG_KEYS
    immediate_action_ids = PROFILES_IMMEDIATE_ACTION_IDS

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        actions: SettingsPageActions | None = None,
    ) -> None:
        super().__init__(parent)
        self._actions = actions or SettingsPageActions()
        self._loading = False
        self._task_running = False
        self._section: dict[str, Any] | None = None
        self._baseline: dict[str, Any] | None = None
        self._remove_requested = False
        self._selected_profile_id = ""
        self._selected_provider_id = ""
        self._capability_combos: dict[str, NoWheelComboBox] = {}
        self._context_edits: dict[str, QLineEdit] = {}
        self._route_widgets: dict[str, dict[str, Any]] = {}
        self.widget, self.body = self._build_widgets()
        self._refresh_all()

    def set_action_callbacks(self, actions: SettingsPageActions) -> None:
        self._actions = actions

    def attach_widget_aliases(self, host_obj: object) -> None:
        target = getattr(host_obj, "__dict__", None)
        if not isinstance(target, dict):
            return
        for name in (
            "profiles_list",
            "profiles_add_btn",
            "profiles_copy_btn",
            "profiles_delete_btn",
            "profiles_diagnose_btn",
            "providers_list",
            "provider_add_btn",
            "provider_delete_btn",
            "profiles_probe_btn",
        ):
            target[name] = getattr(self, name)

    # -- #202 page contract ---------------------------------------------

    def load(self, snapshot: Mapping[str, object], *, restore: bool = False) -> None:
        raw = snapshot.get("model_routing") if isinstance(snapshot, Mapping) else None
        self._section = copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else None
        self._remove_requested = False
        if not restore:
            self._baseline = copy.deepcopy(self._section)
        self._refresh_all()

    def collect(self) -> dict[str, object]:
        if self._remove_requested:
            # Explicit removal: save_apply drops the key so the user can fall
            # back to legacy config or re-create a valid section.
            return {"model_routing": None}
        if self._section is None:
            return {}
        return {"model_routing": copy.deepcopy(self._section)}

    def validate(self) -> Sequence[SettingsIssue]:
        if self._section is None:
            return []
        issues = editor.section_issues(self._section)
        if not issues:
            return []
        messages = [f"{issue['code']}（{issue['path']}）" for issue in issues[:4]]
        message = "；".join(messages)
        if len(issues) > 4:
            message += f"；另有 {len(issues) - 4} 项"
        return [
            SettingsIssue(
                page_key=self.page_key,
                field_key="model_routing",
                message=f"model_routing 校验失败：{message}",
            )
        ]

    def reset(self) -> None:
        self._remove_requested = False
        self._section = copy.deepcopy(self._baseline)
        self._refresh_all()

    def focus_issue(self, issue: SettingsIssue) -> bool:
        attr = _FIELD_WIDGETS.get(issue.field_key)
        widget = getattr(self, attr, None) if attr else None
        if widget is not None and hasattr(widget, "setFocus"):
            widget.setFocus()
            return True
        return False

    def set_task_running(self, running: bool) -> None:
        self._task_running = bool(running)
        self._refresh_enabled()

    # -- widget construction --------------------------------------------

    def _build_widgets(self) -> tuple[QScrollArea, QWidget]:
        page, body, layout = build_settings_scroll_page("settings_profiles")

        self.hint_label = QLabel(MODEL_PROFILES_PAGE_COPY["hint"])
        self.hint_label.setWordWrap(True)
        self.hint_label.setObjectName("config_hint_label")
        layout.addWidget(self.hint_label)

        self.create_btn = QPushButton(MODEL_PROFILES_PAGE_COPY["create_button"])
        self.create_btn.setObjectName("profiles_create_btn")
        self.create_btn.clicked.connect(self._on_create_section)
        self.remove_btn = QPushButton(MODEL_PROFILES_PAGE_COPY["remove_button"])
        self.remove_btn.setObjectName("profiles_remove_btn")
        self.remove_btn.setToolTip(MODEL_PROFILES_PAGE_COPY["remove_tooltip"])
        self.remove_btn.clicked.connect(self._on_remove_section)
        layout.addWidget(self.create_btn)
        layout.addWidget(self.remove_btn)

        self.profiles_group, profiles_layout = settings_group(
            MODEL_PROFILES_PAGE_COPY["profiles_group"]
        )
        self.profiles_list = QListWidget()
        self.profiles_list.setObjectName("profiles_list")
        self.profiles_list.currentRowChanged.connect(self._on_profile_row_changed)
        profiles_layout.addWidget(self.profiles_list)
        profiles_buttons = QHBoxLayout()
        self.profiles_add_btn = QPushButton(MODEL_PROFILES_PAGE_COPY["add_profile"])
        self.profiles_add_btn.clicked.connect(self._on_add_profile)
        self.profiles_copy_btn = QPushButton(MODEL_PROFILES_PAGE_COPY["copy_profile"])
        self.profiles_copy_btn.clicked.connect(self._on_copy_profile)
        self.profiles_delete_btn = QPushButton(MODEL_PROFILES_PAGE_COPY["delete_profile"])
        self.profiles_delete_btn.clicked.connect(self._on_delete_profile)
        self.profiles_diagnose_btn = QPushButton(MODEL_PROFILES_PAGE_COPY["diagnose"])
        self.profiles_diagnose_btn.clicked.connect(self._run_diagnostics)
        self.profiles_probe_btn = QPushButton(
            MODEL_PROFILES_PAGE_COPY["probe_button"]
        )
        self.profiles_probe_btn.setObjectName("profiles_probe_btn")
        self.profiles_probe_btn.setToolTip(MODEL_PROFILES_PAGE_COPY["probe_tooltip"])
        self.profiles_probe_btn.clicked.connect(self._on_probe_profile)
        for button in (
            self.profiles_add_btn,
            self.profiles_copy_btn,
            self.profiles_delete_btn,
            self.profiles_diagnose_btn,
            self.profiles_probe_btn,
        ):
            profiles_buttons.addWidget(button)
        profiles_buttons.addStretch(1)
        profiles_layout.addLayout(profiles_buttons)
        self.profile_editor_group, self._profile_form = settings_group(
            MODEL_PROFILES_PAGE_COPY["providers_group"] + " · Profile"
        )
        self._build_profile_editor(self._profile_form)
        layout.addWidget(self.profiles_group)
        layout.addWidget(self.profile_editor_group)

        self.providers_group, providers_layout = settings_group(
            MODEL_PROFILES_PAGE_COPY["providers_group"]
        )
        self.providers_list = QListWidget()
        self.providers_list.setObjectName("providers_list")
        self.providers_list.currentRowChanged.connect(self._on_provider_row_changed)
        providers_layout.addWidget(self.providers_list)
        provider_buttons = QHBoxLayout()
        self.provider_add_btn = QPushButton(MODEL_PROFILES_PAGE_COPY["add_provider"])
        self.provider_add_btn.clicked.connect(self._on_add_provider)
        self.provider_delete_btn = QPushButton(MODEL_PROFILES_PAGE_COPY["delete_provider"])
        self.provider_delete_btn.clicked.connect(self._on_delete_provider)
        provider_buttons.addWidget(self.provider_add_btn)
        provider_buttons.addWidget(self.provider_delete_btn)
        provider_buttons.addStretch(1)
        providers_layout.addLayout(provider_buttons)
        self.provider_editor_group, self._provider_form = settings_group(
            MODEL_PROFILES_PAGE_COPY["providers_group"] + " · Provider"
        )
        self._build_provider_editor(self._provider_form)
        layout.addWidget(self.providers_group)
        layout.addWidget(self.provider_editor_group)

        self.defaults_group, defaults_layout = settings_group(
            MODEL_PROFILES_PAGE_COPY["defaults_group"]
        )
        self.default_profile_combo = NoWheelComboBox()
        self.default_profile_combo.setObjectName("profiles_default_profile_combo")
        self.default_profile_combo.currentIndexChanged.connect(self._on_defaults_changed)
        self.default_strategy_combo = NoWheelComboBox()
        self.default_strategy_combo.setObjectName("profiles_default_strategy_combo")
        self.default_strategy_combo.currentIndexChanged.connect(self._on_defaults_changed)
        defaults_layout.addWidget(QLabel(MODEL_PROFILES_PAGE_COPY["default_profile_label"]))
        defaults_layout.addWidget(self.default_profile_combo)
        defaults_layout.addWidget(QLabel(MODEL_PROFILES_PAGE_COPY["default_strategy_label"]))
        defaults_layout.addWidget(self.default_strategy_combo)
        layout.addWidget(self.defaults_group)

        self.routes_group, routes_layout = settings_group(
            MODEL_PROFILES_PAGE_COPY["routes_group"]
        )
        self._build_routes_editor(routes_layout)
        layout.addWidget(self.routes_group)

        self.diagnostics_group, diagnostics_layout = settings_group(
            MODEL_PROFILES_PAGE_COPY["diagnostics_group"]
        )
        self.diagnostics_label = QLabel(MODEL_PROFILES_PAGE_COPY["saved_hint"])
        self.diagnostics_label.setObjectName("profiles_diagnostics_label")
        self.diagnostics_label.setWordWrap(True)
        diagnostics_layout.addWidget(self.diagnostics_label)
        layout.addWidget(self.diagnostics_group)

        return page, body

    def _build_profile_editor(self, layout: QVBoxLayout) -> None:
        self.profile_label_edit = QLineEdit()
        self.profile_label_edit.setObjectName("profiles_profile_label_edit")
        self.profile_label_edit.editingFinished.connect(self._on_profile_fields_changed)
        layout.addWidget(QLabel(MODEL_PROFILES_PAGE_COPY["profile_label"]))
        layout.addWidget(self.profile_label_edit)

        self.profile_provider_combo = NoWheelComboBox()
        self.profile_provider_combo.setObjectName("profiles_profile_provider_combo")
        self.profile_provider_combo.currentIndexChanged.connect(
            self._on_profile_fields_changed
        )
        layout.addWidget(QLabel(MODEL_PROFILES_PAGE_COPY["provider_label"]))
        layout.addWidget(self.profile_provider_combo)

        self.profile_model_edit = QLineEdit()
        self.profile_model_edit.setObjectName("profiles_profile_model_edit")
        self.profile_model_edit.editingFinished.connect(self._on_profile_fields_changed)
        layout.addWidget(QLabel(MODEL_PROFILES_PAGE_COPY["model_label"]))
        layout.addWidget(self.profile_model_edit)

        self.profile_models_edit = QLineEdit()
        self.profile_models_edit.setObjectName("profiles_profile_models_edit")
        self.profile_models_edit.setPlaceholderText("逗号分隔；留空只使用主模型")
        self.profile_models_edit.editingFinished.connect(self._on_profile_fields_changed)
        layout.addWidget(QLabel(MODEL_PROFILES_PAGE_COPY["models_label"]))
        layout.addWidget(self.profile_models_edit)

        self.profile_embedding_combo = NoWheelComboBox()
        self.profile_embedding_combo.setObjectName("profiles_profile_embedding_combo")
        self.profile_embedding_combo.currentIndexChanged.connect(
            self._on_profile_fields_changed
        )
        layout.addWidget(QLabel(MODEL_PROFILES_PAGE_COPY["embedding_label"]))
        layout.addWidget(self.profile_embedding_combo)

        capabilities = QGroupBox(MODEL_PROFILES_PAGE_COPY["capabilities_group"])
        capabilities_layout = QVBoxLayout(capabilities)
        for key, label in _CAPABILITY_LABELS.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            combo = NoWheelComboBox()
            combo.setObjectName(f"profiles_capability_{key}")
            combo.addItem(MODEL_PROFILES_PAGE_COPY["inherit_value"], None)
            combo.addItem(MODEL_PROFILES_PAGE_COPY["force_on"], True)
            combo.addItem(MODEL_PROFILES_PAGE_COPY["force_off"], False)
            combo.currentIndexChanged.connect(self._on_profile_capabilities_changed)
            row.addWidget(combo, 1)
            capabilities_layout.addLayout(row)
            self._capability_combos[key] = combo
        for key, label in (
            ("context_limit_tokens", "上下文上限 tokens"),
            ("context_budget_tokens", "上下文预算 tokens"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            edit = QLineEdit()
            edit.setObjectName(f"profiles_{key}")
            edit.setPlaceholderText("留空跟随适配器")
            edit.editingFinished.connect(self._on_profile_capabilities_changed)
            row.addWidget(edit, 1)
            capabilities_layout.addLayout(row)
            self._context_edits[key] = edit
        risk = QLabel(MODEL_PROFILES_PAGE_COPY["capability_risk"])
        risk.setWordWrap(True)
        risk.setObjectName("config_hint_label")
        capabilities_layout.addWidget(risk)
        layout.addWidget(capabilities)

    def _build_provider_editor(self, layout: QVBoxLayout) -> None:
        fields = (
            ("provider_label_edit", "label", QLineEdit),
            ("provider_upstream_edit", "upstream", QLineEdit),
            ("provider_base_url_edit", "base_url", QLineEdit),
            ("provider_models_url_edit", "models_url", QLineEdit),
            ("provider_credential_name_edit", "credential_name", QLineEdit),
            ("provider_credential_env_edit", "credential_env", QLineEdit),
        )
        for attr, key, _cls in fields:
            edit = QLineEdit()
            edit.setObjectName(f"profiles_{attr}")
            edit.editingFinished.connect(self._on_provider_fields_changed)
            setattr(self, attr, edit)
            layout.addWidget(QLabel(MODEL_PROFILES_PAGE_COPY["provider_fields"][key]))
            layout.addWidget(edit)

        self.provider_adapter_combo = NoWheelComboBox()
        self.provider_adapter_combo.setObjectName("profiles_provider_adapter_combo")
        for adapter in editor.ADAPTERS:
            self.provider_adapter_combo.addItem(adapter, adapter)
        self.provider_adapter_combo.currentIndexChanged.connect(
            self._on_provider_fields_changed
        )
        layout.addWidget(QLabel(MODEL_PROFILES_PAGE_COPY["provider_fields"]["adapter"]))
        layout.addWidget(self.provider_adapter_combo)

        self.provider_credential_kind_combo = NoWheelComboBox()
        self.provider_credential_kind_combo.setObjectName(
            "profiles_provider_credential_kind_combo"
        )
        for kind in editor.CREDENTIAL_KINDS:
            label = MODEL_PROFILES_PAGE_COPY["credential_kind_labels"].get(kind, kind)
            self.provider_credential_kind_combo.addItem(label, kind)
        self.provider_credential_kind_combo.currentIndexChanged.connect(
            self._on_provider_fields_changed
        )
        layout.addWidget(
            QLabel(MODEL_PROFILES_PAGE_COPY["provider_fields"]["credential_kind"])
        )
        layout.addWidget(self.provider_credential_kind_combo)

    def _build_routes_editor(self, layout: QVBoxLayout) -> None:
        for stage in editor.STAGE_ORDER:
            row = QHBoxLayout()
            checkbox = QCheckBox(
                MODEL_PROFILES_PAGE_COPY["route_stage_labels"].get(stage, stage)
            )
            checkbox.setObjectName(f"profiles_route_{stage}_override")
            profile_combo = NoWheelComboBox()
            profile_combo.setObjectName(f"profiles_route_{stage}_profile")
            strategy_combo = NoWheelComboBox()
            strategy_combo.setObjectName(f"profiles_route_{stage}_strategy")
            for combo in (profile_combo, strategy_combo):
                combo.currentIndexChanged.connect(
                    lambda _index, stage=stage: self._on_route_changed(stage)
                )
            checkbox.toggled.connect(lambda _checked, stage=stage: self._on_route_changed(stage))
            row.addWidget(checkbox)
            row.addWidget(profile_combo, 2)
            row.addWidget(strategy_combo, 1)
            layout.addLayout(row)
            self._route_widgets[stage] = {
                "override": checkbox,
                "profile": profile_combo,
                "strategy": strategy_combo,
            }

    # -- state refresh ---------------------------------------------------

    def _section_or_none(self) -> dict[str, Any] | None:
        return self._section

    def _refresh_all(self) -> None:
        self._refresh_profiles_list()
        self._refresh_providers_list()
        self._refresh_defaults()
        self._refresh_routes()
        self._refresh_enabled()
        if self._section is None:
            self.hint_label.setText(MODEL_PROFILES_PAGE_COPY["legacy_hint"])
        else:
            issues = editor.section_issues(self._section)
            if issues:
                self.hint_label.setText(
                    MODEL_PROFILES_PAGE_COPY["hint"]
                    + " "
                    + f"当前有 {len(issues)} 项校验问题，保存会被阻止。"
                )
            else:
                self.hint_label.setText(MODEL_PROFILES_PAGE_COPY["hint"])
        self._populate_profile_editor()
        self._populate_provider_editor()

    def _refresh_enabled(self) -> None:
        editable = self._section is not None and not self._task_running
        for widget in (
            self.profiles_list,
            self.profiles_add_btn,
            self.profiles_copy_btn,
            self.profiles_delete_btn,
            self.providers_list,
            self.provider_add_btn,
            self.provider_delete_btn,
            self.profile_editor_group,
            self.provider_editor_group,
            self.defaults_group,
            self.routes_group,
        ):
            widget.setEnabled(editable)
        self.create_btn.setEnabled(not self._task_running and self._section is None)
        self.remove_btn.setEnabled(not self._task_running and self._section is not None)
        self.profiles_diagnose_btn.setEnabled(self._section is not None)
        self.profiles_probe_btn.setEnabled(
            editable and bool(self._selected_profile_id)
        )

    def _refresh_profiles_list(self) -> None:
        self._loading = True
        try:
            self.profiles_list.clear()
            for profile in editor.editor_view(self._section)["profiles"]:
                label = f"{profile['label']}（{profile['model'] or '未设置模型'}）"
                item = QListWidgetItem(label)
                item.setData(256, profile["id"])
                self.profiles_list.addItem(item)
            self._selected_profile_id = self._select_by_id(
                self.profiles_list,
                self._selected_profile_id,
            )
        finally:
            self._loading = False

    def _refresh_providers_list(self) -> None:
        self._loading = True
        try:
            self.providers_list.clear()
            for provider in editor.editor_view(self._section)["providers"]:
                item = QListWidgetItem(
                    f"{provider['label']}（{provider['adapter']}）"
                )
                item.setData(256, provider["id"])
                self.providers_list.addItem(item)
            self._selected_provider_id = self._select_by_id(
                self.providers_list,
                self._selected_provider_id,
            )
        finally:
            self._loading = False

    @staticmethod
    def _select_by_id(list_widget: QListWidget, wanted: str) -> str:
        if not wanted:
            if list_widget.count():
                list_widget.setCurrentRow(0)
                return str(list_widget.item(0).data(256) or "")
            list_widget.setCurrentRow(-1)
            return ""
        for row in range(list_widget.count()):
            if str(list_widget.item(row).data(256) or "") == wanted:
                list_widget.setCurrentRow(row)
                return wanted
        if list_widget.count():
            list_widget.setCurrentRow(0)
            return str(list_widget.item(0).data(256) or "")
        list_widget.setCurrentRow(-1)
        return ""

    def _refresh_defaults(self) -> None:
        if self._section is None:
            return
        view = editor.editor_view(self._section)
        defaults = view["defaults"] if isinstance(view["defaults"], Mapping) else {}
        self._loading = True
        try:
            self.default_profile_combo.clear()
            for profile in view["profiles"]:
                if profile["purpose"] == "embedding":
                    continue
                self.default_profile_combo.addItem(profile["label"], profile["id"])
            self._set_combo_data(
                self.default_profile_combo,
                str(defaults.get("primary_profile_id") or ""),
            )
            default_profile_id = str(self.default_profile_combo.currentData() or "")
            wanted_strategy = str(defaults.get("execution_strategy") or "")
            supported = list(
                editor.strategy_choices(self._section).get(default_profile_id, ())
            )
            if wanted_strategy and wanted_strategy not in supported and supported:
                # An adapter/capability edit can invalidate the stored default;
                # repair it in memory so the visible value is executable.
                try:
                    self._section = editor.set_defaults(
                        self._section,
                        primary_profile_id=default_profile_id,
                        execution_strategy=supported[0],
                    )
                except editor.ModelProfilesEditorError:
                    pass
                else:
                    wanted_strategy = supported[0]
            self._refresh_strategy_combo(
                self.default_strategy_combo,
                default_profile_id,
                wanted=wanted_strategy or (supported[0] if supported else ""),
            )
        finally:
            self._loading = False

    def _refresh_strategy_combo(
        self,
        combo: NoWheelComboBox,
        profile_id: str,
        *,
        wanted: str = "",
    ) -> None:
        choices = editor.strategy_choices(self._section)
        supported = list(choices.get(profile_id, ()))
        combo.clear()
        for strategy in editor.STRATEGY_ORDER:
            label = {
                "sync": "同步",
                "gemini_batch": "Gemini Batch",
            }.get(strategy, strategy)
            if strategy not in supported:
                label = f"{label}（不可用）"
            combo.addItem(label, strategy)
            if strategy not in supported:
                item = combo.model().item(combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
        index = combo.findData(wanted) if wanted and wanted in supported else -1
        if index < 0 and supported:
            index = combo.findData(supported[0])
        if index < 0:
            index = 0 if combo.count() else -1
        combo.setCurrentIndex(index)

    @staticmethod
    def _set_combo_data(combo: NoWheelComboBox, value: str) -> None:
        index = combo.findData(value) if value else -1
        combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))

    def _populate_profile_editor(self) -> None:
        if self._section is None:
            self.profile_label_edit.clear()
            return
        view = editor.editor_view(self._section)
        profile = next(
            (
                item
                for item in view["profiles"]
                if item["id"] == self._selected_profile_id
            ),
            None,
        )
        self._loading = True
        try:
            if profile is None:
                self.profile_label_edit.clear()
                self.profile_model_edit.clear()
                self.profile_models_edit.clear()
                return
            self.profile_label_edit.setText(profile["label"])
            self.profile_provider_combo.clear()
            for provider in view["providers"]:
                self.profile_provider_combo.addItem(
                    provider["label"],
                    provider["id"],
                )
            self._set_combo_data(self.profile_provider_combo, profile["provider_id"])
            self.profile_model_edit.setText(profile["model"])
            self.profile_models_edit.setText(
                "，".join(profile.get("rotation_extras") or ())
            )
            self.profile_embedding_combo.clear()
            self.profile_embedding_combo.addItem("（不绑定）", "")
            for item in view["profiles"]:
                if item["purpose"] == "embedding":
                    self.profile_embedding_combo.addItem(item["label"], item["id"])
            self._set_combo_data(
                self.profile_embedding_combo,
                profile["embedding_profile_id"],
            )
            overrides = dict(profile["capability_overrides"])
            for key, combo in self._capability_combos.items():
                value = overrides.get(key)
                index = combo.findData(value) if value is not None else 0
                combo.setCurrentIndex(index if index >= 0 else 0)
            for key, edit in self._context_edits.items():
                value = overrides.get(key)
                edit.setText("" if value is None else str(value))
        finally:
            self._loading = False

    def _populate_provider_editor(self) -> None:
        if self._section is None:
            return
        view = editor.editor_view(self._section)
        provider = next(
            (
                item
                for item in view["providers"]
                if item["id"] == self._selected_provider_id
            ),
            None,
        )
        self._loading = True
        try:
            if provider is None:
                for edit in self._provider_edits().values():
                    edit.clear()
                return
            self.provider_label_edit.setText(provider["label"])
            self._set_combo_data(self.provider_adapter_combo, provider["adapter"])
            self.provider_upstream_edit.setText(provider["provider"])
            self.provider_base_url_edit.setText(provider["base_url"])
            self.provider_models_url_edit.setText(provider["models_url"])
            credential = provider["credential_ref"]
            self._set_combo_data(
                self.provider_credential_kind_combo,
                credential["kind"],
            )
            self.provider_credential_name_edit.setText(credential["name"])
            self.provider_credential_env_edit.setText(credential["env_name"])
        finally:
            self._loading = False

    def _provider_edits(self) -> dict[str, QLineEdit]:
        return {
            "label": self.provider_label_edit,
            "upstream": self.provider_upstream_edit,
            "base_url": self.provider_base_url_edit,
            "models_url": self.provider_models_url_edit,
            "credential_name": self.provider_credential_name_edit,
            "credential_env": self.provider_credential_env_edit,
        }

    def _refresh_routes(self) -> None:
        if self._section is None:
            return
        view = editor.editor_view(self._section)
        routes = {row["stage"]: row for row in view["routes"]}
        self._loading = True
        try:
            for stage, widgets in self._route_widgets.items():
                route = routes.get(stage, {})
                widgets["override"].setChecked(bool(route.get("explicit")))
                profile_combo = widgets["profile"]
                profile_combo.clear()
                for profile in view["profiles"]:
                    if profile["purpose"] == "embedding":
                        continue
                    profile_combo.addItem(profile["label"], profile["id"])
                self._set_combo_data(
                    profile_combo,
                    str(route.get("profile_id") or ""),
                )
                route_profile_id = str(profile_combo.currentData() or "")
                wanted_route_strategy = str(route.get("strategy") or "")
                supported = list(
                    editor.strategy_choices(self._section).get(route_profile_id, ())
                )
                if (
                    route.get("explicit")
                    and wanted_route_strategy
                    and wanted_route_strategy not in supported
                    and supported
                ):
                    try:
                        self._section = editor.set_route(
                            self._section,
                            stage,
                            enabled=True,
                            profile_id=route_profile_id,
                            strategy=supported[0],
                        )
                    except editor.ModelProfilesEditorError:
                        pass
                    else:
                        wanted_route_strategy = supported[0]
                self._refresh_strategy_combo(
                    widgets["strategy"],
                    route_profile_id,
                    wanted=wanted_route_strategy
                    or (supported[0] if supported else ""),
                )
                editable = bool(route.get("explicit"))
                profile_combo.setEnabled(editable)
                widgets["strategy"].setEnabled(editable)
        finally:
            self._loading = False

    # -- user actions ----------------------------------------------------

    def _on_create_section(self) -> None:
        from gemini_model_catalog import DEFAULT_GEMINI_TRANSLATION_MODEL

        section = editor.empty_section()
        section = editor.add_provider(
            section,
            label="Google Gemini",
            adapter="gemini",
            provider="gemini",
            credential_kind="api_keys_json",
            credential_name="api_keys",
            credential_env_name="GEMINI_API_KEY",
        )
        provider_id = editor.provider_ids(section)[0]
        section = editor.add_profile(
            section,
            label="Gemini Main",
            provider_id=provider_id,
            model=DEFAULT_GEMINI_TRANSLATION_MODEL,
        )
        profile_id = editor.profile_ids(section)[0]
        # Store the default explicitly so the created section is valid and the
        # visible selector matches what will be saved.
        section = editor.set_defaults(
            section,
            primary_profile_id=profile_id,
            execution_strategy="sync",
        )
        self._remove_requested = False
        self._section = section
        self._selected_profile_id = profile_id
        self._selected_provider_id = provider_id
        self._refresh_all()

    def _on_remove_section(self) -> None:
        if self._section is None:
            return
        self._remove_requested = True
        self._section = None
        self._selected_profile_id = ""
        self._selected_provider_id = ""
        self._refresh_all()
        self._show_status(MODEL_PROFILES_PAGE_COPY["remove_pending"])

    def _on_add_profile(self) -> None:
        if self._section is None:
            return
        providers = editor.provider_ids(self._section)
        if not providers:
            # Keep the existing profiles/routes/defaults/unknown fields and add
            # a usable default connection instead of rebuilding the section.
            try:
                self._section = editor.add_provider(
                    self._section,
                    label="Google Gemini",
                    adapter="gemini",
                    provider="gemini",
                    credential_kind="api_keys_json",
                    credential_name="api_keys",
                    credential_env_name="GEMINI_API_KEY",
                )
            except editor.ModelProfilesEditorError as exc:
                self._show_error(exc)
                return
            providers = editor.provider_ids(self._section)
        try:
            self._section = editor.add_profile(
                self._section,
                label="新 ModelProfile",
                provider_id=providers[0],
                model="",
            )
        except editor.ModelProfilesEditorError as exc:
            self._show_error(exc)
            return
        profile_ids = editor.profile_ids(self._section)
        self._selected_profile_id = profile_ids[-1]
        self._refresh_all()

    def _on_copy_profile(self) -> None:
        if self._section is None or not self._selected_profile_id:
            return
        try:
            self._section = editor.copy_profile(
                self._section,
                self._selected_profile_id,
            )
        except editor.ModelProfilesEditorError as exc:
            self._show_error(exc)
            return
        profile_ids = editor.profile_ids(self._section)
        self._selected_profile_id = profile_ids[-1]
        self._refresh_all()

    def _on_delete_profile(self) -> None:
        if self._section is None or not self._selected_profile_id:
            return
        try:
            self._section = editor.delete_profile(
                self._section,
                self._selected_profile_id,
            )
        except editor.ModelProfilesEditorError as exc:
            if exc.code == "PROFILE_IN_USE":
                self._show_status(MODEL_PROFILES_PAGE_COPY["delete_profile_blocked"])
            else:
                self._show_error(exc)
            return
        self._selected_profile_id = ""
        self._refresh_all()

    def _on_add_provider(self) -> None:
        self._section = self._section or editor.empty_section()
        try:
            self._section = editor.add_provider(
                self._section,
                label="新 Provider",
                adapter="litellm",
                provider="",
            )
        except editor.ModelProfilesEditorError as exc:
            self._show_error(exc)
            return
        provider_ids = editor.provider_ids(self._section)
        self._selected_provider_id = provider_ids[-1]
        self._refresh_all()

    def _on_delete_provider(self) -> None:
        if self._section is None or not self._selected_provider_id:
            return
        try:
            self._section = editor.delete_provider(
                self._section,
                self._selected_provider_id,
            )
        except editor.ModelProfilesEditorError as exc:
            if exc.code == "PROVIDER_IN_USE":
                self._show_status(MODEL_PROFILES_PAGE_COPY["delete_provider_blocked"])
            else:
                self._show_error(exc)
            return
        self._selected_provider_id = ""
        self._refresh_all()

    def _on_probe_profile(self) -> None:
        if self._section is None or not self._selected_profile_id:
            self._show_status(MODEL_PROFILES_PAGE_COPY["no_selection"])
            return
        if self._actions.run_immediate is None:
            self._show_status(MODEL_PROFILES_PAGE_COPY["probe_unavailable"])
            return
        self._actions.run_immediate(
            "probe_profile",
            {"profile_id": self._selected_profile_id},
        )

    def set_probe_running(self, running: bool) -> None:
        """Reflect the host's probe worker without owning the task lock."""
        if running:
            self.profiles_probe_btn.setText(
                MODEL_PROFILES_PAGE_COPY["probe_running"]
            )
        else:
            self.profiles_probe_btn.setText(
                MODEL_PROFILES_PAGE_COPY["probe_button"]
            )

    def set_probe_report(self, report: Mapping[str, Any] | None) -> None:
        """Render a credential-free per-capability probe report."""
        data = dict(report or {})
        if not data:
            return
        lines = [
            f"能力探测：{data.get('profile_id') or '(unknown)'} "
            f"（{data.get('adapter') or '?'}）状态 {data.get('status') or 'unknown'}，"
            f"请求数 {data.get('requests', 0)}"
        ]
        for item in data.get("capabilities") or ():
            if not isinstance(item, Mapping):
                continue
            detail = str(item.get("detail") or "")
            suffix = f"（{detail}）" if detail else ""
            lines.append(
                f"- {item.get('name')}：{item.get('status')}{suffix}"
            )
        self.diagnostics_label.setText("\n".join(lines))

    def set_probe_error(self, code: str) -> None:
        self.diagnostics_label.setText(
            MODEL_PROFILES_PAGE_COPY["probe_failed"].format(code=code or "UNKNOWN")
        )

    def _run_diagnostics(self) -> None:
        if self._section is None:
            return
        view = editor.editor_view(self._section)
        lines: list[str] = []
        for profile in view["profiles"]:
            if profile["purpose"] == "embedding":
                continue
            capabilities = profile["capabilities"]
            strategies = ", ".join(profile["strategies"]) or "无"
            source = capabilities.get("structured_output_source") or "未知"
            references = ", ".join(profile["referenced_by"]) or "无"
            lines.append(
                f"{profile['label']}（{profile['adapter']}）："
                f"可用策略 {strategies}；结构化输出来源 {source}；引用 {references}"
            )
            sources = capabilities.get("sources") or {}
            for key in editor.CAPABILITY_FLAG_KEYS:
                if key not in capabilities:
                    continue
                label = _CAPABILITY_LABELS.get(key, key)
                state = "支持" if capabilities[key] else "不支持"
                lines.append(
                    f"  {label}：{state}（来源 {sources.get(key) or '未知'}）"
                )
        for issue in editor.section_issues(self._section):
            lines.append(f"[{issue['code']}] {issue['path']}")
        self.diagnostics_label.setText("\n".join(lines) or "没有可显示的诊断项。")

    # -- write-through handlers ------------------------------------------

    def _on_profile_row_changed(self, _row: int) -> None:
        if self._loading:
            return
        item = self.profiles_list.currentItem()
        self._selected_profile_id = str(item.data(256) or "") if item else ""
        self._populate_profile_editor()

    def _on_provider_row_changed(self, _row: int) -> None:
        if self._loading:
            return
        item = self.providers_list.currentItem()
        self._selected_provider_id = str(item.data(256) or "") if item else ""
        self._populate_provider_editor()

    def _on_profile_fields_changed(self, *_args: object) -> None:
        if self._loading or self._section is None or not self._selected_profile_id:
            return
        models = [
            part.strip()
            for part in self.profile_models_edit.text().replace("，", ",").split(",")
            if part.strip()
        ]
        try:
            self._section = editor.update_profile(
                self._section,
                self._selected_profile_id,
                label=self.profile_label_edit.text(),
                provider_id=str(self.profile_provider_combo.currentData() or ""),
                model=self.profile_model_edit.text(),
                models=models,
                embedding_profile_id=str(
                    self.profile_embedding_combo.currentData() or ""
                ),
            )
        except editor.ModelProfilesEditorError as exc:
            self._show_error(exc)
            return
        self._refresh_profiles_list()
        self._refresh_defaults()

    def _on_profile_capabilities_changed(self, *_args: object) -> None:
        if self._loading or self._section is None or not self._selected_profile_id:
            return
        overrides: dict[str, Any] = {}
        for key, combo in self._capability_combos.items():
            value = combo.currentData()
            if value is not None:
                overrides[key] = bool(value)
        invalid_fields: list[str] = []
        for key, edit in self._context_edits.items():
            text = edit.text().strip()
            if not text:
                continue
            try:
                overrides[key] = int(text)
            except ValueError:
                invalid_fields.append(_CONTEXT_LABELS.get(key, key))
        if invalid_fields:
            self._show_status(
                MODEL_PROFILES_PAGE_COPY["invalid_integer"].format(
                    fields="、".join(invalid_fields)
                )
            )
        try:
            self._section = editor.update_profile(
                self._section,
                self._selected_profile_id,
                capability_overrides=overrides,
            )
        except editor.ModelProfilesEditorError as exc:
            self._show_error(exc)
            return
        self._refresh_defaults()
        self._refresh_routes()

    def _on_provider_fields_changed(self, *_args: object) -> None:
        if self._loading or self._section is None or not self._selected_provider_id:
            return
        try:
            self._section = editor.update_provider(
                self._section,
                self._selected_provider_id,
                label=self.provider_label_edit.text(),
                adapter=str(self.provider_adapter_combo.currentData() or ""),
                provider=self.provider_upstream_edit.text(),
                base_url=self.provider_base_url_edit.text(),
                models_url=self.provider_models_url_edit.text(),
                credential_kind=str(
                    self.provider_credential_kind_combo.currentData() or "none"
                ),
                credential_name=self.provider_credential_name_edit.text(),
                credential_env_name=self.provider_credential_env_edit.text(),
            )
        except editor.ModelProfilesEditorError as exc:
            self._show_error(exc)
            return
        self._refresh_providers_list()
        self._refresh_profiles_list()
        self._refresh_defaults()
        self._refresh_routes()

    def _on_defaults_changed(self, *_args: object) -> None:
        if self._loading or self._section is None:
            return
        profile_id = str(self.default_profile_combo.currentData() or "")
        strategy = str(self.default_strategy_combo.currentData() or "")
        if not profile_id:
            return
        supported = editor.strategy_choices(self._section).get(profile_id, ())
        if strategy not in supported:
            # The strategy combo can still show the previous profile's value
            # while the profile combo is switching; fall back to a supported
            # strategy instead of rejecting and rolling the profile back.
            if not supported:
                self._show_status(
                    MODEL_PROFILES_PAGE_COPY["no_supported_strategy"]
                )
                self._refresh_defaults()
                return
            strategy = supported[0]
        try:
            self._section = editor.set_defaults(
                self._section,
                primary_profile_id=profile_id,
                execution_strategy=strategy,
            )
        except editor.ModelProfilesEditorError as exc:
            self._show_status(self._editor_error_message(exc))
            self._refresh_defaults()
            return
        # The profile changed, so the supported-strategy set must be rebuilt
        # before the user picks the default execution strategy.
        self._refresh_defaults()
        self._refresh_routes()

    def _on_route_changed(self, stage: str) -> None:
        if self._loading or self._section is None:
            return
        widgets = self._route_widgets[stage]
        enabled = widgets["override"].isChecked()
        profile_id = str(widgets["profile"].currentData() or "")
        strategy = str(widgets["strategy"].currentData() or "")
        if enabled:
            supported = editor.strategy_choices(self._section).get(profile_id, ())
            if strategy not in supported:
                # Stage-route profile switches hit the same stale-strategy
                # window as the defaults row; resolve instead of refusing.
                if not supported:
                    self._show_status(
                        MODEL_PROFILES_PAGE_COPY["no_supported_strategy"]
                    )
                    self._refresh_routes()
                    return
                strategy = supported[0]
        try:
            self._section = editor.set_route(
                self._section,
                stage,
                enabled=enabled,
                profile_id=profile_id,
                strategy=strategy,
            )
        except editor.ModelProfilesEditorError as exc:
            self._show_status(self._editor_error_message(exc))
        self._refresh_routes()

    # -- feedback --------------------------------------------------------

    def _show_status(self, message: str) -> None:
        if self._actions.show_status is not None:
            self._actions.show_status(message)
        else:
            self.diagnostics_label.setText(message)

    def _editor_error_message(self, exc: editor.ModelProfilesEditorError) -> str:
        """Map a structured editor refusal to actionable, localized copy."""
        reason_code = str((exc.details or {}).get("reason") or "")
        reason = MODEL_PROFILES_PAGE_COPY["reason_labels"].get(
            reason_code,
            reason_code,
        )
        template = MODEL_PROFILES_PAGE_COPY["error_messages"].get(exc.code)
        if template:
            return template.format(reason=reason) if "{reason}" in template else template
        return MODEL_PROFILES_PAGE_COPY["unknown_error"].format(reason=exc.code)

    def _show_error(self, exc: editor.ModelProfilesEditorError) -> None:
        self._show_status(self._editor_error_message(exc))
