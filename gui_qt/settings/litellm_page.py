"""Independent LiteLLM Settings page (#202 Phase C).

The page owns widgets, catalog/version/connection/warmup workers, and the
load/collect/validate/reset contract. Host callbacks cover dialogs, keyring,
install, logs, and cross-page gating so the page can be constructed without
``MainWindow``. Save still goes through the unique MainWindow transaction.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QCompleter,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from litellm_provider_config import (
    CustomLiteLLMProvider,
    catalog_source_label,
    custom_provider_from_mapping,
    custom_provider_registry,
    installed_litellm_version,
    litellm_install_probe,
    load_provider_api_key,
    load_provider_key_store,
    native_catalog_endpoint,
    provider_display_label,
    provider_from_model,
    resolve_provider_id,
    sort_provider_ids,
    version_key,
    ProviderCredentialStoreError,
)

from ..api_key_helpers import mask_api_key
from ..custom_provider_dialog import CustomLiteLLMProviderDialog
from ..litellm_catalog_cache import (
    CatalogSnapshot,
    LiteLLMCatalogCache,
    catalog_snapshot_warning,
)
from ..litellm_settings import provider_credential_status, read_sync_backend_models
from ..litellm_worker import (
    LiteLLMConnectionTestWorker,
    LiteLLMModelCatalogWorker,
    LiteLLMModuleWarmupWorker,
    LiteLLMProviderCatalogWorker,
    LiteLLMVersionWorker,
    is_cancelled_message,
)
from ..operation_identity import is_current_identity, litellm_connection_identity
from ..user_copy import (
    CUSTOM_LITELLM_PROVIDER_COPY,
    LITELLM_CACHE_COPY,
    LITELLM_CONNECTION_TEST_COPY,
)
from ..widget_helpers import (
    NoWheelComboBox,
    add_editable_combo_popup_action,
    message_box_information,
    message_box_question,
    message_box_warning,
)
from .page_contract import SettingsIssue, SettingsPageActions
from .registry import SETTINGS_PAGE_SPEC_OBJECTS

LITELLM_PAGE_KEY = "litellm"
_LITELLM_SPEC = next(spec for spec in SETTINGS_PAGE_SPEC_OBJECTS if spec.key == LITELLM_PAGE_KEY)
LITELLM_NAV_LABEL = _LITELLM_SPEC.nav_label
LITELLM_CONFIG_KEYS = _LITELLM_SPEC.config_keys
LITELLM_IMMEDIATE_ACTION_IDS = _LITELLM_SPEC.immediate_action_ids

LITELLM_MODEL_SELECTION_SAVE_DEBOUNCE_MS = 400

RETIRED_LITELLM_WARMUP_WORKERS: set[LiteLLMModuleWarmupWorker] = set()

LITELLM_WIDGET_ATTRS: tuple[str, ...] = (
    "sync_backend_combo",
    "litellm_provider_combo",
    "litellm_refresh_providers_btn",
    "litellm_clear_provider_btn",
    "litellm_provider_catalog_status_label",
    "litellm_model_combo",
    "litellm_refresh_models_btn",
    "litellm_catalog_status_label",
    "sync_backend_hint",
    "litellm_version_label",
    "litellm_check_version_btn",
    "install_litellm_btn",
    "litellm_install_progress",
    "custom_provider_table",
    "custom_provider_add_btn",
    "custom_provider_edit_btn",
    "custom_provider_delete_btn",
    "custom_provider_status_label",
    "litellm_credentials_box",
    "litellm_provider_label",
    "litellm_manage_keys_btn",
    "litellm_credential_status_label",
    "litellm_test_connection_btn",
    "litellm_connection_status_label",
)

LITELLM_FORWARDED_ATTRS: frozenset[str] = frozenset(
    {
        "_litellm_provider_catalog_worker",
        "_litellm_catalog_worker",
        "_litellm_version_worker",
        "_litellm_module_warmup_worker",
        "_litellm_connection_worker",
        "_litellm_latest_version",
        "_litellm_latest_compatible_version",
        "_litellm_latest_requires_python",
        "_updating_litellm_provider",
        "_applied_litellm_provider",
        "_pending_litellm_model_selection",
        "_litellm_saved_key_status",
        "_custom_litellm_providers",
        "_custom_litellm_providers_load_error",
        "_custom_litellm_providers_modified",
        "_litellm_cache",
    }
)

_FIELD_WIDGETS = {
    "sync_backend": "sync_backend_combo",
    "litellm_model": "litellm_model_combo",
    "custom_litellm_providers": "custom_provider_table",
}


@dataclass
class LiteLLMPageHost:
    """Host-owned callbacks. All optional so the page is independently constructible."""

    show_status: Callable[[str, int], None] | None = None
    append_log: Callable[[str], None] | None = None
    dialog_parent: Callable[[], QWidget | None] | None = None
    is_shutdown_requested: Callable[[], bool] | None = None
    is_loading_config: Callable[[], bool] | None = None
    is_install_running: Callable[[], bool] | None = None
    is_task_running: Callable[[], bool] | None = None
    load_translator_config: Callable[[], Mapping[str, object]] | None = None
    open_provider_keys: Callable[[str], bool] | None = None
    start_install: Callable[[], tuple[bool, str]] | None = None
    on_providers_changed: Callable[[str], None] | None = None
    on_backend_gating: Callable[[str], None] | None = None
    environment: Callable[[], Mapping[str, str]] | None = None
    load_api_key: Callable[[str], str] | None = None
    load_key_store: Callable[[str], object] | None = None
    message_information: Callable[[str, str], None] | None = None
    message_warning: Callable[[str, str], None] | None = None
    message_question: Callable[..., str] | None = None
    create_provider_catalog_worker: Callable[[], QThread] | None = None
    create_model_catalog_worker: Callable[[str, str, Mapping[str, CustomLiteLLMProvider]], QThread] | None = None
    create_version_worker: Callable[[], QThread] | None = None
    create_connection_worker: Callable[[str, str, Mapping[str, CustomLiteLLMProvider], str], QThread] | None = None
    create_warmup_worker: Callable[[], QThread] | None = None
    create_custom_provider_dialog: Callable[..., QDialog] | None = None


def _style_themed_surface(widget: QWidget) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


def _mapping_from_entry(entry: object) -> dict[str, object] | None:
    if isinstance(entry, Mapping):
        return dict(entry)
    if isinstance(entry, (list, tuple)) and all(
        isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in entry
    ):
        try:
            return dict(entry)
        except (TypeError, ValueError):
            return None
    return None


class LiteLLMSettingsPage(QObject):
    """Settings page for the LiteLLM backend, independently constructible."""

    page_key = LITELLM_PAGE_KEY
    nav_label = LITELLM_NAV_LABEL
    config_keys = LITELLM_CONFIG_KEYS
    immediate_action_ids = LITELLM_IMMEDIATE_ACTION_IDS

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        cache: LiteLLMCatalogCache | None = None,
        host: LiteLLMPageHost | None = None,
        actions: SettingsPageActions | None = None,
        start_warmup: bool = True,
    ) -> None:
        super().__init__(parent)
        self._host = host or LiteLLMPageHost()
        self._actions = actions or SettingsPageActions()
        self._litellm_cache = cache or LiteLLMCatalogCache()
        self._loading = False
        self._task_running = False
        self._baseline: dict[str, object] = {}
        self._retired_network_workers: set[QThread] = set()
        self._litellm_provider_catalog_worker: LiteLLMProviderCatalogWorker | None = None
        self._litellm_catalog_worker: LiteLLMModelCatalogWorker | None = None
        self._litellm_version_worker: LiteLLMVersionWorker | None = None
        self._litellm_module_warmup_worker: LiteLLMModuleWarmupWorker | None = None
        self._litellm_connection_worker: LiteLLMConnectionTestWorker | None = None
        self._litellm_latest_version = ""
        self._litellm_latest_compatible_version = ""
        self._litellm_latest_requires_python = ""
        self._updating_litellm_provider = False
        self._applied_litellm_provider = ""
        self._pending_litellm_model_selection: tuple[str, str] | None = None
        self._litellm_saved_key_status: dict[str, str] = {}
        self._custom_litellm_providers: dict[str, CustomLiteLLMProvider] = {}
        self._custom_litellm_providers_load_error = ""
        self._custom_litellm_providers_modified = False
        self._litellm_model_selection_save_timer = QTimer(self)
        self._litellm_model_selection_save_timer.setSingleShot(True)
        self._litellm_model_selection_save_timer.setInterval(
            LITELLM_MODEL_SELECTION_SAVE_DEBOUNCE_MS
        )
        self._litellm_model_selection_save_timer.timeout.connect(
            self._flush_litellm_model_selection_save
        )
        self.widget, self.body = self._build_widgets()
        if start_warmup:
            self._start_litellm_module_warmup()
        self._baseline = dict(self.collect())

    def set_action_callbacks(self, actions: SettingsPageActions) -> None:
        self._actions = actions

    def load(self, snapshot: Mapping[str, object], *, restore: bool = False) -> None:
        previous = self._loading
        self._loading = True
        try:
            if "sync_backend" in snapshot:
                backend = str(snapshot.get("sync_backend") or "gemini").strip().lower()
                if backend not in {"gemini", "litellm"}:
                    backend = "gemini"
                index = self.sync_backend_combo.findData(backend)
                if index >= 0:
                    self.sync_backend_combo.setCurrentIndex(index)
            if "custom_litellm_providers" in snapshot:
                self._apply_custom_providers_payload(
                    snapshot.get("custom_litellm_providers"),
                    restore=restore,
                )
            model = str(snapshot.get("litellm_model") or "").strip()
            if "litellm_model" in snapshot or "custom_litellm_providers" in snapshot:
                selected = provider_from_model(model) if model else (
                    self._current_litellm_provider()
                )
                self._populate_litellm_providers(
                    self._cached_litellm_provider_values(),
                    selected=selected,
                )
            if "litellm_model" in snapshot:
                self._restore_configured_litellm_model(model)
            self._on_sync_backend_changed(-1)
            self._baseline = dict(self.collect())
        finally:
            self._loading = previous

    def load_from_sync_config(
        self,
        sync_config: Mapping[str, object],
        *,
        recommended_gemini: str = "",
    ) -> None:
        backend = str(sync_config.get("backend") or "gemini").strip().lower()
        if backend not in {"gemini", "litellm"}:
            backend = "gemini"
        models = read_sync_backend_models(sync_config, backend, recommended_gemini)
        self.load(
            {
                "sync_backend": backend,
                "litellm_model": models.litellm_model,
                "custom_litellm_providers": sync_config.get(
                    "custom_litellm_providers"
                ),
            }
        )

    def collect(self) -> dict[str, object]:
        return {
            "sync_backend": self._selected_sync_backend(),
            "litellm_model": self._litellm_model_text(),
            "custom_litellm_providers": tuple(
                tuple(sorted(entry.items()))
                for entry in self._custom_provider_entries()
            ),
        }

    def validate(self) -> Sequence[SettingsIssue]:
        issues: list[SettingsIssue] = []
        if (
            self._custom_litellm_providers_modified
            and self._custom_litellm_providers_load_error
        ):
            issues.append(
                SettingsIssue(
                    self.page_key,
                    "custom_litellm_providers",
                    CUSTOM_LITELLM_PROVIDER_COPY["load_error_save_blocked"].format(
                        error=self._custom_litellm_providers_load_error
                    ),
                )
            )
        if self._selected_sync_backend() == "litellm" and not self._litellm_model_text():
            issues.append(
                SettingsIssue(
                    self.page_key,
                    "litellm_model",
                    "启用 LiteLLM 前，请填写带 provider 前缀的模型名称。",
                )
            )
        return issues

    def reset(self) -> None:
        self.request_shutdown()
        if self._baseline:
            self.load(self._baseline)

    def focus_issue(self, issue: SettingsIssue) -> bool:
        attr = _FIELD_WIDGETS.get(issue.field_key)
        widget = getattr(self, attr, None) if attr else None
        if widget is not None and hasattr(widget, "setFocus"):
            widget.setFocus()
            return True
        return False

    def set_task_running(self, running: bool) -> None:
        self._task_running = bool(running)
        self._on_sync_backend_changed(-1, notify_host=False)

    def request_shutdown(self) -> None:
        """Cancel page-owned network workers without waiting on the GUI thread."""
        self._cancel_litellm_model_selection_save()
        for attr in (
            "_litellm_provider_catalog_worker",
            "_litellm_catalog_worker",
            "_litellm_version_worker",
            "_litellm_connection_worker",
        ):
            worker = getattr(self, attr, None)
            if worker is None:
                continue
            self._retire_network_worker(worker)
            setattr(self, attr, None)
        self._detach_litellm_module_warmup()

    def has_active_network_workers(self) -> bool:
        for attr in (
            "_litellm_provider_catalog_worker",
            "_litellm_catalog_worker",
            "_litellm_version_worker",
            "_litellm_connection_worker",
        ):
            worker = getattr(self, attr, None)
            if worker is not None and getattr(worker, "isRunning", lambda: False)():
                return True
        return any(
            getattr(worker, "isRunning", lambda: False)()
            for worker in tuple(self._retired_network_workers)
        )

    def attach_widget_aliases(self, host_obj: object) -> None:
        """Copy widget attributes onto a host for lazy getattr compatibility."""
        target = getattr(host_obj, "__dict__", None)
        if not isinstance(target, dict):
            return
        for name in LITELLM_WIDGET_ATTRS:
            target[name] = getattr(self, name)

    def _apply_custom_providers_payload(
        self,
        raw: object,
        *,
        restore: bool = False,
    ) -> None:
        try:
            if raw is None:
                payload: object = []
            elif isinstance(raw, (list, tuple)):
                payload = [
                    converted
                    for entry in raw
                    if (converted := _mapping_from_entry(entry)) is not None
                ]
            else:
                payload = raw
            self._custom_litellm_providers = custom_provider_registry(
                payload,
                allow_import=False,
            )
            self._custom_litellm_providers_load_error = ""
        except ValueError as exc:
            self._custom_litellm_providers = {}
            self._custom_litellm_providers_load_error = str(exc)
            if restore:
                self._log(f"恢复自定义 Provider 快照失败，已忽略：{exc}")
            else:
                self._log(f"忽略无效的 custom_litellm_providers 配置：{exc}")
                self._show_status(
                    f"translator_config.json 中的 custom_litellm_providers "
                    f"配置无效，已忽略：{exc}",
                    8000,
                )
            self.custom_provider_status_label.setText(
                CUSTOM_LITELLM_PROVIDER_COPY["load_error_status"].format(error=exc)
            )
        self._custom_litellm_providers_modified = False
        self._refresh_custom_provider_table()

    def _retire_network_worker(self, worker: QThread) -> None:
        try:
            worker.completed.disconnect()
        except (RuntimeError, TypeError, AttributeError):
            pass
        request_cancel = getattr(worker, "request_cancel", None)
        if callable(request_cancel):
            request_cancel()
        worker.requestInterruption()
        self._retired_network_workers.add(worker)

        def _drop(current: QThread = worker) -> None:
            self._retired_network_workers.discard(current)
            current.deleteLater()

        try:
            worker.finished.connect(_drop)
        except RuntimeError:
            self._retired_network_workers.discard(worker)

    def _notify_providers_changed(self, selected: str = "") -> None:
        callback = self._host.on_providers_changed
        if callback is not None:
            callback(selected)

    def _notify_backend_gating(self) -> None:
        callback = self._host.on_backend_gating
        if callback is not None:
            callback(self._selected_sync_backend())

    def _show_status(self, message: str, timeout_ms: int = 5000) -> None:
        callback = self._host.show_status
        if callback is not None:
            callback(str(message), int(timeout_ms))

    def _log(self, message: str) -> None:
        callback = self._host.append_log
        if callback is not None:
            callback(str(message))

    def _dialog_parent(self) -> QWidget | None:
        callback = self._host.dialog_parent
        if callback is not None:
            return callback()
        return self.widget

    def _is_shutdown_requested(self) -> bool:
        callback = self._host.is_shutdown_requested
        return bool(callback()) if callback is not None else False

    def _is_loading_config(self) -> bool:
        if self._loading:
            return True
        callback = self._host.is_loading_config
        return bool(callback()) if callback is not None else False

    def _is_install_running(self) -> bool:
        callback = self._host.is_install_running
        return bool(callback()) if callback is not None else False

    def _is_global_task_running(self) -> bool:
        if self._task_running:
            return True
        callback = self._host.is_task_running
        return bool(callback()) if callback is not None else False

    def _environ(self) -> Mapping[str, str]:
        callback = self._host.environment
        if callback is not None:
            return callback()
        return os.environ

    def _load_api_key(self, provider: str) -> str:
        callback = self._host.load_api_key
        if callback is not None:
            return str(callback(provider) or "")
        try:
            return str(load_provider_api_key(provider) or "")
        except ProviderCredentialStoreError:
            return ""

    def _load_key_store(self, provider: str) -> object:
        callback = self._host.load_key_store
        if callback is not None:
            return callback(provider)
        return load_provider_key_store(provider)

    def _message_information(self, title: str, text: str) -> None:
        callback = self._host.message_information
        if callback is not None:
            callback(title, text)
            return
        message_box_information(self._dialog_parent(), title, text)

    def _message_warning(self, title: str, text: str) -> None:
        callback = self._host.message_warning
        if callback is not None:
            callback(title, text)
            return
        message_box_warning(self._dialog_parent(), title, text)

    def _message_question(self, title: str, text: str, **kwargs: Any) -> str:
        callback = self._host.message_question
        if callback is not None:
            return str(callback(title, text, **kwargs))
        return str(
            message_box_question(self._dialog_parent(), title, text, **kwargs)
        )

    def _load_translator_config(self) -> Mapping[str, object]:
        callback = self._host.load_translator_config
        if callback is None:
            return {}
        try:
            config = callback()
        except Exception:
            return {}
        return config if isinstance(config, Mapping) else {}

    def _make_version_worker(self) -> QThread:
        factory = self._host.create_version_worker
        if factory is not None:
            return factory()
        return LiteLLMVersionWorker(self)

    def _make_provider_catalog_worker(self) -> QThread:
        factory = self._host.create_provider_catalog_worker
        if factory is not None:
            return factory()
        return LiteLLMProviderCatalogWorker(self)

    def _make_warmup_worker(self) -> QThread:
        factory = self._host.create_warmup_worker
        if factory is not None:
            return factory()
        return LiteLLMModuleWarmupWorker(self)

    def _make_model_catalog_worker(self, provider: str, api_key: str) -> QThread:
        factory = self._host.create_model_catalog_worker
        if factory is not None:
            return factory(provider, api_key, dict(self._custom_litellm_providers))
        return LiteLLMModelCatalogWorker(
            provider,
            api_key=api_key,
            parent=self,
            custom_providers=self._custom_litellm_providers,
        )

    def _make_connection_worker(self, model: str, api_key: str) -> QThread:
        factory = self._host.create_connection_worker
        identity = self._litellm_connection_operation_identity()
        if factory is not None:
            return factory(
                model, api_key, dict(self._custom_litellm_providers), identity
            )
        return LiteLLMConnectionTestWorker(
            model,
            api_key,
            self,
            custom_providers=self._custom_litellm_providers,
            operation_identity=identity,
        )

    def _make_custom_provider_dialog(self, **kwargs: Any) -> QDialog:
        factory = self._host.create_custom_provider_dialog
        if factory is not None:
            return factory(**kwargs)
        return CustomLiteLLMProviderDialog(self._dialog_parent(), **kwargs)

    def _on_manage_litellm_keys(self) -> None:
        provider = self._current_litellm_provider()
        actions = self._actions
        if actions.run_immediate is not None:
            actions.run_immediate("manage_provider_keys", {"provider": provider})
            return
        callback = self._host.open_provider_keys
        if callback is not None:
            callback(provider)

    def _on_install_litellm(self) -> None:
        actions = self._actions
        if actions.run_immediate is not None:
            actions.run_immediate("install_litellm", {})
            return
        callback = self._host.start_install
        if callback is None:
            return
        started, message = callback()
        if not started:
            self._message_information("无法安装 LiteLLM", message)

    def _build_scroll_page(self, object_name: str) -> tuple[QScrollArea, QWidget, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        scroll.setObjectName(f"{object_name}_scroll")
        _style_themed_surface(scroll)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        viewport = scroll.viewport()
        viewport.setObjectName(f"{object_name}_viewport")
        _style_themed_surface(viewport)
        content = QWidget()
        content.setObjectName(f"{object_name}_content")
        _style_themed_surface(content)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        body = QWidget()
        body.setObjectName("settings_page_body")
        body.setProperty("settingsPage", object_name)
        body.setMinimumWidth(0)
        body.setMaximumWidth(16777215)
        body.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding,
        )
        _style_themed_surface(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(14)
        content_layout.addWidget(body, 1)
        scroll.setWidget(content)
        return scroll, body, layout

    def _settings_group(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 18, 14, 14)
        return group, layout

    def _settings_form(self, group: QGroupBox) -> QFormLayout:
        form = QFormLayout(group)
        form.setObjectName("settings_form")
        form.setContentsMargins(14, 18, 14, 14)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        return form

    def _configure_editable_model_combo(self, combo: NoWheelComboBox) -> None:
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        add_editable_combo_popup_action(combo)

    def _build_widgets(self) -> tuple[QScrollArea, QWidget]:
        page, body, layout = self._build_scroll_page("settings_litellm")

        backend_box = QGroupBox("LiteLLM 同步替代后端")
        backend_layout = self._settings_form(backend_box)

        self.sync_backend_combo = NoWheelComboBox()
        self.sync_backend_combo.addItem("Gemini 同步（推荐）", "gemini")
        self.sync_backend_combo.addItem("启用 LiteLLM 同步替代", "litellm")
        self.sync_backend_combo.currentIndexChanged.connect(self._on_sync_backend_changed)
        backend_layout.addRow("同步执行后端：", self.sync_backend_combo)

        self.litellm_provider_combo = NoWheelComboBox()
        self._configure_editable_model_combo(self.litellm_provider_combo)
        self.litellm_provider_combo.lineEdit().setPlaceholderText(
            "搜索或输入自定义 Provider"
        )
        provider_completer = self.litellm_provider_combo.completer()
        if provider_completer is not None:
            provider_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            provider_completer.setFilterMode(Qt.MatchFlag.MatchContains)
            provider_completer.setCompletionMode(
                QCompleter.CompletionMode.PopupCompletion
            )
        provider_row = QWidget()
        provider_layout = QHBoxLayout(provider_row)
        provider_layout.setContentsMargins(0, 0, 0, 0)
        provider_layout.setSpacing(8)
        provider_layout.addWidget(self.litellm_provider_combo, 1)
        self.litellm_refresh_providers_btn = QPushButton("联网加载供应商")
        self.litellm_refresh_providers_btn.setObjectName("secondary_btn")
        self.litellm_refresh_providers_btn.clicked.connect(
            self._on_refresh_litellm_providers
        )
        provider_layout.addWidget(self.litellm_refresh_providers_btn)
        self.litellm_clear_provider_btn = QPushButton("取消选择")
        self.litellm_clear_provider_btn.setObjectName("secondary_btn")
        self.litellm_clear_provider_btn.clicked.connect(self._on_clear_litellm_provider)
        provider_layout.addWidget(self.litellm_clear_provider_btn)
        backend_layout.addRow("Provider：", provider_row)
        self.litellm_provider_catalog_status_label = QLabel()
        self.litellm_provider_catalog_status_label.setWordWrap(True)
        self.litellm_provider_catalog_status_label.setObjectName("config_hint_label")
        backend_layout.addRow(self.litellm_provider_catalog_status_label)
        self._populate_litellm_providers(
            self._cached_litellm_provider_values(),
            selected=self._litellm_cache.selected_provider,
        )

        self.litellm_model_combo = NoWheelComboBox()
        self._configure_editable_model_combo(self.litellm_model_combo)
        self.litellm_model_combo.lineEdit().setPlaceholderText("尚未加载模型")
        self.litellm_model_combo.currentTextChanged.connect(self._on_litellm_model_changed)
        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(8)
        model_layout.addWidget(self.litellm_model_combo, 1)
        self.litellm_refresh_models_btn = QPushButton("联网加载模型")
        self.litellm_refresh_models_btn.setObjectName("secondary_btn")
        self.litellm_refresh_models_btn.clicked.connect(self._on_refresh_litellm_models)
        model_layout.addWidget(self.litellm_refresh_models_btn)
        backend_layout.addRow("LiteLLM 模型：", model_row)
        self.litellm_catalog_status_label = QLabel("模型目录：尚未加载。")
        self.litellm_catalog_status_label.setWordWrap(True)
        self.litellm_catalog_status_label.setObjectName("config_hint_label")
        backend_layout.addRow(self.litellm_catalog_status_label)

        self.sync_backend_hint = QLabel()
        self.sync_backend_hint.setWordWrap(True)
        self.sync_backend_hint.setObjectName("config_hint_label")
        backend_layout.addRow(self.sync_backend_hint)

        version_row = QWidget()
        version_layout = QHBoxLayout(version_row)
        version_layout.setContentsMargins(0, 0, 0, 0)
        version_layout.setSpacing(8)
        self.litellm_version_label = QLabel()
        self.litellm_version_label.setWordWrap(True)
        self.litellm_version_label.setMinimumWidth(0)
        self.litellm_version_label.setObjectName("config_hint_label")
        version_layout.addWidget(self.litellm_version_label, 1)
        self.litellm_check_version_btn = QPushButton("检查更新")
        self.litellm_check_version_btn.setObjectName("secondary_btn")
        self.litellm_check_version_btn.clicked.connect(self._on_check_litellm_version)
        version_layout.addWidget(self.litellm_check_version_btn)
        self.install_litellm_btn = QPushButton("安装 LiteLLM")
        self.install_litellm_btn.setObjectName("secondary_btn")
        self.install_litellm_btn.clicked.connect(self._on_install_litellm)
        self.install_litellm_btn.setVisible(False)
        version_layout.addWidget(self.install_litellm_btn)
        backend_layout.addRow("LiteLLM 版本：", version_row)
        self._refresh_litellm_version_label()

        self.litellm_install_progress = QProgressBar()
        self.litellm_install_progress.setObjectName("litellm_install_progress")
        self.litellm_install_progress.setTextVisible(True)
        self.litellm_install_progress.setFormat("正在后台安装 LiteLLM…")
        self.litellm_install_progress.setVisible(False)
        backend_layout.addRow(self.litellm_install_progress)
        layout.addWidget(backend_box)

        custom_box, custom_layout = self._settings_group("自定义 OpenAI 兼容 Provider")
        custom_hint = QLabel(
            "为 OpenAI 兼容但 LiteLLM 未内置的服务（OpenCode Go、中转站、本地 vLLM 等）"
            "注册通用 Provider。请求会改写为 openai/<模型> 并逐请求透传 API Base；"
            "模型列表走 GET {models_url}，密钥保存在系统凭据管理器（可加多把 Key）。"
            "id 同时用作模型前缀与密钥用户名，创建后不可修改。"
        )
        custom_hint.setWordWrap(True)
        custom_hint.setObjectName("config_hint_label")
        custom_layout.addWidget(custom_hint)

        self.custom_provider_table = QTableWidget(0, 4)
        self.custom_provider_table.setObjectName("custom_provider_table")
        self.custom_provider_table.setHorizontalHeaderLabels(
            ("Provider id", "显示名称", "API Base", "密钥环境变量")
        )
        self.custom_provider_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.custom_provider_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.custom_provider_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.custom_provider_table.verticalHeader().setVisible(False)
        self.custom_provider_table.setWordWrap(False)
        self.custom_provider_table.setTextElideMode(Qt.TextElideMode.ElideRight)
        header = self.custom_provider_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(60)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        self.custom_provider_table.setColumnWidth(0, 110)
        self.custom_provider_table.setColumnWidth(1, 110)
        self.custom_provider_table.setColumnWidth(3, 150)
        self.custom_provider_table.setMinimumHeight(120)
        self.custom_provider_table.itemSelectionChanged.connect(
            self._refresh_custom_provider_actions
        )
        custom_layout.addWidget(self.custom_provider_table)

        custom_actions = QHBoxLayout()
        self.custom_provider_add_btn = QPushButton("添加…")
        self.custom_provider_add_btn.setObjectName("secondary_btn")
        self.custom_provider_add_btn.clicked.connect(
            self._on_add_custom_litellm_provider
        )
        custom_actions.addWidget(self.custom_provider_add_btn)
        self.custom_provider_edit_btn = QPushButton("编辑…")
        self.custom_provider_edit_btn.setObjectName("secondary_btn")
        self.custom_provider_edit_btn.setEnabled(False)
        self.custom_provider_edit_btn.clicked.connect(
            self._on_edit_custom_litellm_provider
        )
        custom_actions.addWidget(self.custom_provider_edit_btn)
        self.custom_provider_delete_btn = QPushButton("删除")
        self.custom_provider_delete_btn.setObjectName("secondary_btn")
        self.custom_provider_delete_btn.setEnabled(False)
        self.custom_provider_delete_btn.clicked.connect(
            self._on_delete_custom_litellm_provider
        )
        custom_actions.addWidget(self.custom_provider_delete_btn)
        custom_actions.addStretch(1)
        custom_layout.addLayout(custom_actions)

        self.custom_provider_status_label = QLabel()
        self.custom_provider_status_label.setWordWrap(True)
        self.custom_provider_status_label.setObjectName("config_hint_label")
        custom_layout.addWidget(self.custom_provider_status_label)
        layout.addWidget(custom_box)

        credentials_box, credentials_layout = self._settings_group("Provider 凭据")
        self.litellm_credentials_box = credentials_box
        credentials_hint = QLabel(
            "密钥与 Gemini 配置完全分离，并保存到操作系统凭据管理器；"
            "不会写入 translator_config.json。也可继续使用 LiteLLM 约定的环境变量。\n"
            "DeepSeek / OpenAI / Anthropic / xAI 等：请先在「密钥」页或下方按钮中"
            "保存至少一把 API Key，再点「联网加载模型」。支持多 Key，仅显示脱敏后缀。"
        )
        credentials_hint.setWordWrap(True)
        credentials_hint.setObjectName("config_hint_label")
        credentials_layout.addWidget(credentials_hint)
        self.litellm_provider_label = QLabel()
        self.litellm_provider_label.setObjectName("litellm_provider_label")
        credentials_layout.addWidget(self.litellm_provider_label)
        self.litellm_credential_status_label = QLabel()
        self.litellm_credential_status_label.setWordWrap(True)
        self.litellm_credential_status_label.setObjectName("api_status_label")
        credentials_layout.addWidget(self.litellm_credential_status_label)
        credential_actions = QWidget()
        credential_actions_layout = QHBoxLayout(credential_actions)
        credential_actions_layout.setContentsMargins(0, 0, 0, 0)
        credential_actions_layout.setSpacing(8)
        self.litellm_manage_keys_btn = QPushButton("管理密钥…")
        self.litellm_manage_keys_btn.setObjectName("api_btn")
        self.litellm_manage_keys_btn.clicked.connect(self._on_manage_litellm_keys)
        credential_actions_layout.addWidget(self.litellm_manage_keys_btn)
        credential_actions_layout.addStretch(1)
        credentials_layout.addWidget(credential_actions)
        self.litellm_test_connection_btn = QPushButton("测试连接")
        self.litellm_test_connection_btn.clicked.connect(self._on_test_litellm_connection)
        credentials_layout.addWidget(self.litellm_test_connection_btn)
        self.litellm_connection_status_label = QLabel("尚未测试连接。")
        self.litellm_connection_status_label.setWordWrap(True)
        self.litellm_connection_status_label.setObjectName("config_hint_label")
        credentials_layout.addWidget(self.litellm_connection_status_label)
        layout.addWidget(credentials_box)

        self.litellm_provider_combo.currentIndexChanged.connect(
            self._on_litellm_provider_changed
        )
        self.litellm_provider_combo.lineEdit().editingFinished.connect(
            self._on_litellm_provider_changed
        )
        self._restore_litellm_cached_selection()
        self._refresh_litellm_catalog_status()
        self._refresh_litellm_credential_status()
        self._refresh_custom_provider_table()
        layout.addStretch(1)
        return page, body

    def _selected_sync_backend(self) -> str:
        # Use __dict__/settings_widget — plain getattr() would lazy-build every
        # config page via __getattr__ when the LiteLLM section is unvisited.
        combo = getattr(self, "sync_backend_combo", None)
        if combo is None:
            return "gemini"
        value = combo.currentData()
        return value if value in {"gemini", "litellm"} else "gemini"

    def _litellm_model_text(self) -> str:
        combo = getattr(self, "litellm_model_combo", None)
        current_text = getattr(combo, "currentText", None)
        model = str(current_text() or "").strip() if callable(current_text) else ""
        if not model or "/" in model:
            return model
        provider = self._litellm_provider_combo_value()
        return f"{provider}/{model}" if provider else model

    def _litellm_provider_combo_value(self) -> str:
        combo = getattr(self, "litellm_provider_combo", None)
        if combo is None:
            return ""
        index = combo.currentIndex()
        if index >= 0 and combo.currentText() == combo.itemText(index):
            data = str(combo.itemData(index) or "").strip().lower()
            if data:
                return data
        return resolve_provider_id(combo.currentText())

    def _current_litellm_provider(self) -> str:
        model_provider = provider_from_model(self._litellm_model_text())
        if model_provider:
            return model_provider
        return self._litellm_provider_combo_value()

    def _ensure_litellm_provider_item(self, provider: str) -> int:
        combo = getattr(self, "litellm_provider_combo", None)
        provider = str(provider or "").strip().lower()
        if combo is None or not provider:
            return -1
        index = combo.findData(provider)
        if index < 0:
            combo.addItem(
                provider_display_label(provider, self._custom_litellm_providers),
                provider,
            )
            index = combo.findData(provider)
        return index

    def _populate_litellm_providers(
        self,
        providers: tuple[str, ...],
        *,
        selected: str = "",
    ) -> None:
        combo = getattr(self, "litellm_provider_combo", None)
        if combo is None:
            return
        selected = str(selected or "").strip().lower()
        providers = tuple(
            dict.fromkeys((*providers, *self._custom_litellm_providers))
        )
        combo.blockSignals(True)
        combo.clear()
        for provider in sort_provider_ids(providers):
            provider = str(provider or "").strip().lower()
            if provider:
                combo.addItem(
                    provider_display_label(provider, self._custom_litellm_providers),
                    provider,
                )
        if selected:
            index = self._ensure_litellm_provider_item(selected)
            combo.setCurrentIndex(index)
        else:
            combo.setCurrentIndex(-1)
            if combo.isEditable():
                combo.lineEdit().clear()
        combo.blockSignals(False)
        self._applied_litellm_provider = selected

    @staticmethod
    def _litellm_snapshot_status(
        subject: str,
        snapshot: CatalogSnapshot,
        *,
        source_label: str,
    ) -> str:
        if not snapshot.values:
            return f"{subject}：尚未联网加载。"
        fetched = f"缓存时间：{snapshot.fetched_at}。" if snapshot.fetched_at else ""
        version = (
            f"LiteLLM {snapshot.litellm_version}。"
            if snapshot.litellm_version
            else ""
        )
        warning = catalog_snapshot_warning(
            snapshot,
            current_litellm_version=installed_litellm_version(),
        )
        return " ".join(
            part
            for part in (
                f"{subject}：已缓存 {len(snapshot.values)} 项。",
                source_label,
                fetched,
                version,
                warning,
            )
            if part
        )

    def _refresh_litellm_catalog_status(self) -> None:
        provider_label = getattr(self, "litellm_provider_catalog_status_label", None)
        if provider_label is not None:
            message = self._litellm_snapshot_status(
                "供应商目录",
                self._litellm_cache.providers,
                source_label="来源：LiteLLM 官方在线目录。",
            )
            if self._litellm_cache.load_error:
                message = f"{message} {self._litellm_cache.load_error}"
            provider_label.setText(message)
        model_label = getattr(self, "litellm_catalog_status_label", None)
        if model_label is not None:
            provider = self._litellm_provider_combo_value()
            snapshot = self._litellm_cache.models(provider)
            message = self._litellm_snapshot_status(
                "模型目录",
                snapshot,
                source_label=catalog_source_label(
                    snapshot.source,
                    self._custom_litellm_providers,
                ),
            )
            endpoint = native_catalog_endpoint(
                provider,
                self._custom_litellm_providers,
            )
            if endpoint is not None and endpoint.require_key:
                try:
                    has_key = bool(self._load_api_key(provider))
                except ProviderCredentialStoreError:
                    has_key = False
                if not has_key:
                    custom = self._custom_litellm_providers.get(provider)
                    if custom is not None and custom.api_key_env and self._environ().get(
                        custom.api_key_env
                    ):
                        has_key = True
                if not has_key:
                    custom = self._custom_litellm_providers.get(provider)
                    if custom is not None:
                        message = (
                            f"{message} 提示：{endpoint.label} 模型列表需先保存 API Key；"
                            "自定义 Provider 没有 LiteLLM 子集目录可回退。"
                        )
                    else:
                        message = (
                            f"{message} 提示：{endpoint.label} 官方列表需先保存 API Key；"
                            "未保存时只能尝试 LiteLLM 子集目录（可能依赖 GitHub 网络）。"
                        )
            model_label.setText(message)

    def _save_litellm_cache(self, action: Callable[[], None]) -> None:
        try:
            action()
        except OSError as exc:
            self._log(
                LITELLM_CACHE_COPY["save_failed_log"].format(error=exc)
            )
            self._show_status(
                LITELLM_CACHE_COPY["save_failed_status"],
                6000,
            )
        else:
            fallback_reason = getattr(self._litellm_cache, "fallback_reason", "")
            if fallback_reason:
                self._log(fallback_reason)
                self._show_status(
                    LITELLM_CACHE_COPY["save_status"],
                    6000,
                )

    def _schedule_litellm_model_selection_save(self, provider: str, model: str) -> None:
        provider = str(provider or "").strip().lower()
        model = str(model or "").strip()
        if not provider or not model:
            self._cancel_litellm_model_selection_save()
            return
        self._pending_litellm_model_selection = (provider, model)
        timer = getattr(self, "_litellm_model_selection_save_timer", None)
        if timer is None:
            self._flush_litellm_model_selection_save()
            return
        timer.start()

    def _cancel_litellm_model_selection_save(self) -> None:
        timer = getattr(self, "_litellm_model_selection_save_timer", None)
        if timer is not None:
            timer.stop()
        self._pending_litellm_model_selection = None

    def _flush_litellm_model_selection_save(self) -> None:
        timer = getattr(self, "_litellm_model_selection_save_timer", None)
        if timer is not None:
            timer.stop()
        pending = getattr(self, "_pending_litellm_model_selection", None)
        self._pending_litellm_model_selection = None
        if not pending:
            return
        provider, model = pending
        if provider and model:
            self._save_litellm_cache(
                lambda p=provider, m=model: self._litellm_cache.select_model(p, m)
            )

    def _restore_litellm_cached_selection(self) -> None:
        cache = self.__dict__.get("_litellm_cache")
        if cache is None:
            return
        provider = cache.selected_provider
        if not provider:
            self._applied_litellm_provider = ""
            self._set_litellm_models("", ())
            return
        index = self._ensure_litellm_provider_item(provider)
        combo = getattr(self, "litellm_provider_combo", None)
        if combo is not None:
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
        self._applied_litellm_provider = provider
        snapshot = cache.models(provider)
        self._set_litellm_models(
            provider,
            snapshot.values,
            selected=self._litellm_cache.selected_model(provider),
        )

    def _restore_configured_litellm_model(self, model: str) -> None:
        model = str(model or "").strip()
        if not model:
            self._restore_litellm_cached_selection()
            return
        provider = provider_from_model(model)
        if provider:
            index = self._ensure_litellm_provider_item(provider)
            combo = getattr(self, "litellm_provider_combo", None)
            if combo is not None:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
            self._applied_litellm_provider = provider
        cache = self.__dict__.get("_litellm_cache")
        snapshot = cache.models(provider) if cache is not None else CatalogSnapshot()
        self._set_litellm_models(provider, snapshot.values, selected=model)

    def _set_litellm_models(
        self,
        provider: str,
        models: tuple[str, ...],
        *,
        preserve_current: bool = False,
        selected: str = "",
    ) -> None:
        combo = getattr(self, "litellm_model_combo", None)
        if combo is None:
            return
        current = combo.currentText().strip()
        selected = current if preserve_current else str(selected or "").strip()
        values = tuple(
            dict.fromkeys(str(model).strip() for model in models if str(model).strip())
        )
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(list(values))
        if selected:
            combo.setEditText(selected)
        else:
            combo.setCurrentIndex(-1)
            if combo.isEditable():
                combo.lineEdit().clear()
        combo.blockSignals(False)
        self._on_litellm_model_changed(combo.currentText())

    def _litellm_saved_key_message(self, provider: str, *, force_reload: bool = False) -> str:
        """Human status for OS-stored provider keys (masked; never full secret)."""
        provider = str(provider or "").strip().lower()
        if not provider or provider == "ollama":
            return ""
        if not force_reload:
            cached = self._litellm_saved_key_status.get(provider)
            if cached:
                return cached
        try:
            store = self._load_key_store(provider)
        except ProviderCredentialStoreError as exc:
            # Transient failure: never cache it, or a stale error would persist
            # after the credential store recovers.
            self._litellm_saved_key_status.pop(provider, None)
            return str(exc)
        endpoint = native_catalog_endpoint(provider, self._custom_litellm_providers)
        needs_official_key = bool(endpoint is not None and endpoint.require_key)
        env_key_available = ""
        if not store.keys and needs_official_key:
            custom = self._custom_litellm_providers.get(provider)
            if custom is not None and custom.api_key_env and self._environ().get(
                custom.api_key_env
            ):
                env_key_available = custom.api_key_env
        if store.keys:
            masked = "、".join(mask_api_key(key) for key in store.keys)
            active = store.active_key()
            active_note = (
                f"当前使用：{mask_api_key(active)}。"
                if active and len(store.keys) > 1
                else ""
            )
            message = (
                f"系统凭据管理器中已保存 {len(store.keys)} 把密钥：{masked}。"
                f"{active_note}"
                "如同时存在环境变量，请求优先使用已保存的当前密钥。"
            )
        elif needs_official_key:
            label = (
                endpoint.label
                if endpoint is not None
                else provider_display_label(provider, self._custom_litellm_providers)
            )
            if env_key_available:
                message = (
                    "系统凭据管理器中尚未保存密钥；"
                    f"已检测到环境变量 {env_key_available}，将作为回退使用。"
                )
            else:
                message = (
                    f"系统凭据管理器中尚未保存密钥。"
                    f"加载 {label} 官方模型列表前请先保存 API Key。"
                )
        elif provider in self._custom_litellm_providers:
            custom = self._custom_litellm_providers[provider]
            message = (
                CUSTOM_LITELLM_PROVIDER_COPY["keyless_status"]
                if not custom.requires_key
                else "系统凭据管理器中尚未保存密钥。"
            )
        else:
            message = "系统凭据管理器中尚未保存密钥。"
        self._litellm_saved_key_status[provider] = message
        return message

    def _custom_provider_entries(self) -> list[dict[str, str]]:
        """Serialize the in-memory registry for config writes / dirty checks."""
        entries: list[dict[str, object]] = []
        registry = self.__dict__.get("_custom_litellm_providers") or {}
        for provider in registry.values():
            entry = {
                "id": provider.id,
                "label": provider.label,
                "base_url": provider.base_url,
                "models_url": provider.models_url,
            }
            if provider.api_key_env:
                entry["api_key_env"] = provider.api_key_env
            if not provider.requires_key:
                entry["requires_key"] = False
            entries.append(entry)
        return entries

    def _reserved_custom_provider_ids(self) -> frozenset[str]:
        from litellm_provider_config import reserved_litellm_provider_ids

        # Reserved set is based only on known LiteLLM prefixes and the current
        # registry. Historical catalog cache ids are intentionally excluded:
        # they are user-level state with no UI to clear, and including them
        # would block re-adding a deleted provider under the same id.
        # allow_import=False keeps this call off the ~10s synchronous litellm
        # import; the background warmup worker merges the installed provider
        # table into the reserved set as soon as it finishes.
        return frozenset(
            {
                *reserved_litellm_provider_ids(allow_import=False),
                *self.__dict__.get("_custom_litellm_providers", {}),
            }
        )

    def _refresh_custom_provider_table(self) -> None:
        table = getattr(self, "custom_provider_table", None)
        if table is None:
            return
        previous = self._selected_custom_provider()
        table.setRowCount(0)
        for provider in self._custom_litellm_providers.values():
            row = table.rowCount()
            table.insertRow(row)
            for column, text in enumerate(
                (
                    provider.id,
                    provider.label,
                    provider.base_url,
                    provider.api_key_env or "（未设置）",
                )
            ):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, provider.id)
                if len(text) > 20:
                    # Long API Base URLs and labels are elided in the cell;
                    # keep the full value readable on hover.
                    item.setToolTip(text)
                table.setItem(row, column, item)
            if previous and provider.id == previous:
                table.selectRow(row)
        self._refresh_custom_provider_actions()
        label = getattr(self, "custom_provider_status_label", None)
        if label is not None:
            load_error = self.__dict__.get("_custom_litellm_providers_load_error")
            if load_error:
                # Keep the load-error message visible instead of overwriting it
                # with the plain empty/count status on every refresh.
                label.setText(
                    CUSTOM_LITELLM_PROVIDER_COPY["load_error_status"].format(
                        error=load_error
                    )
                )
                return
            count = len(self._custom_litellm_providers)
            label.setText(
                CUSTOM_LITELLM_PROVIDER_COPY["table_count"].format(count=count)
                if count
                else CUSTOM_LITELLM_PROVIDER_COPY["table_empty"]
            )

    def _refresh_custom_provider_actions(self) -> None:
        idle = not self._is_global_task_running()
        has_selection = bool(self._selected_custom_provider())
        add_btn = getattr(self, "custom_provider_add_btn", None)
        if add_btn is not None:
            add_btn.setEnabled(idle)
        edit_btn = getattr(self, "custom_provider_edit_btn", None)
        delete_btn = getattr(self, "custom_provider_delete_btn", None)
        if edit_btn is not None:
            edit_btn.setEnabled(idle and has_selection)
        if delete_btn is not None:
            delete_btn.setEnabled(idle and has_selection)

    def _selected_custom_provider(self) -> str:
        table = getattr(self, "custom_provider_table", None)
        if table is None:
            return ""
        row = table.currentRow()
        if row < 0:
            return ""
        item = table.item(row, 0)
        return str(item.text()).strip().lower() if item is not None else ""

    def _after_custom_providers_changed(self) -> None:
        """Re-sync every LiteLLM surface that renders the provider registry."""
        self._refresh_custom_provider_table()
        current = self._current_litellm_provider()
        self._populate_litellm_providers(
            self._cached_litellm_provider_values(),
            selected=current,
        )
        self._notify_providers_changed(current)
        self._refresh_litellm_catalog_status()
        self._refresh_litellm_credential_status()
        self._on_sync_backend_changed(-1)

    def _cached_litellm_provider_values(self) -> tuple[str, ...]:
        """Provider ids from the user-level catalog cache, tolerant of missing
        or partially loaded snapshots (same contract as _reserved_custom_provider_ids)."""
        cache = self.__dict__.get("_litellm_cache")
        if cache is None:
            return ()
        providers = getattr(cache, "providers", None)
        values = getattr(providers, "values", ()) if providers is not None else ()
        return tuple(values) if values else ()

    def _on_add_custom_litellm_provider(self) -> None:
        dialog = self._make_custom_provider_dialog(
            reserved=self._reserved_custom_provider_ids(),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        entry = dialog.result_provider()
        provider = custom_provider_from_mapping(entry, allow_import=False)
        if provider.id in self._custom_litellm_providers:
            self._message_warning("重复的 Provider id",
                f"已存在 id 为 {provider.id} 的自定义 Provider，请先编辑或删除后再添加。",
            )
            return
        self._custom_litellm_providers[provider.id] = provider
        self._custom_litellm_providers_modified = True
        self._after_custom_providers_changed()
        self._show_status(
            f"已添加自定义 Provider：{provider.label}（{provider.id}）。",
            6000,
        )
        self._log(
            f"添加自定义 LiteLLM Provider：{provider.id} → {provider.base_url}"
        )

    def _on_edit_custom_litellm_provider(self) -> None:
        provider_id = self._selected_custom_provider()
        provider = self._custom_litellm_providers.get(provider_id)
        if provider is None:
            return
        dialog = self._make_custom_provider_dialog(
            provider={
                "id": provider.id,
                "label": provider.label,
                "base_url": provider.base_url,
                "models_url": provider.models_url,
                "api_key_env": provider.api_key_env,
                "requires_key": provider.requires_key,
            },
            # The edited id is already registered; exclude it from the reserved
            # set so accept does not report a self-conflict on a locked field.
            reserved=frozenset(
                {
                    *self._reserved_custom_provider_ids(),
                }
                - {provider.id}
            ),
            title="编辑自定义 Provider",
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        entry = dialog.result_provider()
        updated = custom_provider_from_mapping(entry, allow_import=False)
        self._custom_litellm_providers[provider.id] = updated
        self._custom_litellm_providers_modified = True
        self._after_custom_providers_changed()
        self._show_status(
            f"已更新自定义 Provider：{updated.label}（{updated.id}）。",
            6000,
        )
        self._log(
            f"更新自定义 LiteLLM Provider：{updated.id} → {updated.base_url}"
        )

    def _on_delete_custom_litellm_provider(self) -> None:
        provider_id = self._selected_custom_provider()
        provider = self._custom_litellm_providers.get(provider_id)
        if provider is None:
            return
        is_current = self._current_litellm_provider() == provider_id
        reply = self._message_question(CUSTOM_LITELLM_PROVIDER_COPY["delete_title"],
            (
                CUSTOM_LITELLM_PROVIDER_COPY["delete_confirm"].format(
                    label=provider.label,
                    id=provider.id,
                )
                + (
                    CUSTOM_LITELLM_PROVIDER_COPY["delete_current_note"]
                    if is_current
                    else ""
                )
            ),
            yes_text="删除",
            no_text="取消",
            default="no",
        )
        if reply != "yes":
            return
        del self._custom_litellm_providers[provider_id]
        self._custom_litellm_providers_modified = True
        # Also drop the user-level catalog artifacts (models snapshot and any
        # selection) so the deleted id cannot resurface from the persistent
        # cache in dropdowns or restores.
        self._save_litellm_cache(
            lambda p=provider_id: self._litellm_cache.remove_provider(p)
        )
        if is_current:
            # Drop the ghost selection: the model still references the deleted
            # id, which would fail on the next sync request after save.
            self._cancel_litellm_model_selection_save()
            self._set_litellm_models("", ())
            combo = getattr(self, "litellm_provider_combo", None)
            if combo is not None:
                combo.blockSignals(True)
                combo.setCurrentIndex(-1)
                if combo.isEditable():
                    combo.lineEdit().clear()
                combo.blockSignals(False)
            self._applied_litellm_provider = ""
        self._after_custom_providers_changed()
        self._show_status(
            f"已删除自定义 Provider：{provider.label}（{provider.id}）。",
            6000,
        )
        self._log(f"删除自定义 LiteLLM Provider：{provider.id}")

    def _refresh_litellm_credential_status(self) -> None:
        provider_label = getattr(self, "litellm_provider_label", None)
        status_label = getattr(self, "litellm_credential_status_label", None)
        if provider_label is None or status_label is None:
            return
        provider = self._current_litellm_provider()
        model = self._litellm_model_text()
        status = provider_credential_status(
            model or (f"{provider}/_" if provider else ""),
            self._environ(),
            self._custom_litellm_providers,
        )
        provider_label.setText(
            f"当前 Provider：{provider_display_label(provider, self._custom_litellm_providers)}"
            if provider
            else "当前 Provider：尚未选择"
        )
        credentials_box = getattr(self, "litellm_credentials_box", None)
        if credentials_box is not None:
            credentials_box.setTitle(
                f"Provider 凭据 — "
                f"{provider_display_label(provider, self._custom_litellm_providers)}"
                if provider
                else "Provider 凭据"
            )
        saved_message = self._litellm_saved_key_message(provider) if provider else ""
        status_label.setText(" ".join(part for part in (saved_message, status.message) if part))

    def _on_litellm_model_changed(self, _text: str) -> None:
        model = self._litellm_model_text()
        model_provider = provider_from_model(model)
        provider_combo = getattr(self, "litellm_provider_combo", None)
        updating = getattr(self, "_updating_litellm_provider", False)
        if model_provider and provider_combo is not None and not updating:
            previous_applied = self._applied_litellm_provider
            index = self._ensure_litellm_provider_item(model_provider)
            if index != provider_combo.currentIndex():
                provider_combo.blockSignals(True)
                provider_combo.setCurrentIndex(index)
                provider_combo.blockSignals(False)
            self._applied_litellm_provider = model_provider
            if not self._is_loading_config():
                self._save_litellm_cache(
                    lambda p=model_provider: self._litellm_cache.select_provider(p)
                )
            if model_provider != previous_applied:
                # Same side effects as provider combo switch: reload that
                # provider's cached catalog and re-gate credential controls,
                # while keeping the typed model as the selection.
                self._updating_litellm_provider = True
                try:
                    snapshot = self._litellm_cache.models(model_provider)
                    self._set_litellm_models(
                        model_provider,
                        snapshot.values,
                        selected=model,
                    )
                finally:
                    self._updating_litellm_provider = False
                self._refresh_litellm_catalog_status()
                # _set_litellm_models re-enters this handler for save + gating.
                return

        provider = model_provider or self._litellm_provider_combo_value()
        if provider and model and not self._is_loading_config():
            self._schedule_litellm_model_selection_save(provider, model)
        else:
            self._cancel_litellm_model_selection_save()
        # Re-run backend gating so “测试连接” tracks model text changes.
        self._on_sync_backend_changed(-1)

    def _on_clear_litellm_provider(self) -> None:
        combo = getattr(self, "litellm_provider_combo", None)
        if combo is None:
            return
        self._cancel_litellm_model_selection_save()
        combo.blockSignals(True)
        combo.setCurrentIndex(-1)
        if combo.isEditable():
            combo.lineEdit().clear()
        combo.blockSignals(False)
        self._on_litellm_provider_changed()

    def _on_litellm_provider_changed(self, _value: object = None) -> None:
        if self._updating_litellm_provider:
            return
        provider = self._litellm_provider_combo_value()
        if provider == self._applied_litellm_provider:
            return
        self._cancel_litellm_model_selection_save()
        self._applied_litellm_provider = provider
        if not self._is_loading_config():
            self._save_litellm_cache(
                lambda p=provider: self._litellm_cache.select_provider(p)
            )
        self._updating_litellm_provider = True
        try:
            snapshot = self._litellm_cache.models(provider)
            self._set_litellm_models(
                provider,
                snapshot.values,
                selected=self._litellm_cache.selected_model(provider),
            )
        finally:
            self._updating_litellm_provider = False
        self._refresh_litellm_catalog_status()
        self._on_sync_backend_changed(-1)

    def _refresh_litellm_version_label(self) -> None:
        label = getattr(self, "litellm_version_label", None)
        if label is None:
            return
        installed = installed_litellm_version()
        latest = str(getattr(self, "_litellm_latest_version", "") or "")
        compatible = str(
            getattr(self, "_litellm_latest_compatible_version", "") or ""
        )
        requires_python = str(
            getattr(self, "_litellm_latest_requires_python", "") or ""
        )
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if not installed:
            label.setText("尚未安装；可检查 PyPI 最新稳定版。")
        elif not latest:
            label.setText(f"本机 {installed}；尚未检查 PyPI。")
        elif compatible and version_key(compatible) < version_key(latest):
            requirement = f"（要求 Python {requires_python}）" if requires_python else ""
            state = (
                f"建议更新到 {compatible}。"
                if version_key(installed) < version_key(compatible)
                else "已是当前 Python 可用最新版。"
            )
            label.setText(
                f"本机 {installed}；PyPI 最新稳定版 {latest}{requirement}不支持当前 "
                f"Python {python_version}；\n兼容最新版 {compatible}，{state}"
            )
        elif compatible and version_key(installed) < version_key(compatible):
            label.setText(f"本机 {installed}；最新兼容稳定版 {compatible}，建议更新。")
        elif compatible:
            label.setText(f"本机 {installed}；已是最新兼容稳定版。")
        elif version_key(installed) < version_key(latest):
            label.setText(f"本机 {installed}；最新稳定版 {latest}，建议更新。")
        else:
            label.setText(f"本机 {installed}；已是最新稳定版。")

    def _request_cancel_litellm_worker(
        self,
        worker: object | None,
        *,
        button_name: str,
        status_message: str,
    ) -> bool:
        """If *worker* is running, request cancel and update the toggle button."""
        if worker is None or not getattr(worker, "isRunning", lambda: False)():
            return False
        request_cancel = getattr(worker, "request_cancel", None)
        if callable(request_cancel):
            request_cancel()
        button = getattr(self, button_name, None)
        if button is not None:
            button.setEnabled(True)
            button.setText("正在取消…")
        self._show_status(status_message, 4000)
        return True

    def _start_litellm_module_warmup(self) -> None:
        """Preload the heavy litellm package in a background thread.

        ``import litellm`` can take ~10s cold; running it off the main thread
        keeps the first LiteLLM settings visit and the custom-provider dialogs
        responsive.  When the import finishes, the installed provider table is
        merged into the reserved-id set and any open LiteLLM page is re-read.
        Idempotent and cheap when litellm is not installed (probe only).
        """
        if self._is_shutdown_requested():
            return
        if self._environ().get("RTL_DISABLE_LITELLM_WARMUP"):
            return
        if getattr(self, "_litellm_module_warmup_worker", None) is not None:
            return
        if not litellm_install_probe():
            return
        worker = self._make_warmup_worker()
        worker.setPriority(QThread.Priority.LowPriority)
        worker.completed.connect(self._on_litellm_module_warmed)
        self._litellm_module_warmup_worker = worker
        worker.start()

    def _on_litellm_module_warmed(self, _module: object) -> None:
        worker = getattr(self, "_litellm_module_warmup_worker", None)
        if worker is not None:
            self._litellm_module_warmup_worker = None
            worker.deleteLater()
        # A worker detached during shutdown finishes in the background; drop
        # it once the thread is no longer running so teardown stays safe.
        for retired_worker in tuple(RETIRED_LITELLM_WARMUP_WORKERS):
            if not getattr(retired_worker, "isRunning", lambda: False)():
                RETIRED_LITELLM_WARMUP_WORKERS.discard(retired_worker)
                retired_worker.deleteLater()
        if self._is_shutdown_requested():
            return
        # Skip when the user already edited providers unsaved, or when the
        # page was never opened.
        if getattr(self, "_custom_litellm_providers_modified", False):
            return
        # Re-validate the registry with the now-complete reserved set, then
        # refresh table/dropdowns/status while preserving the current page
        # selection.  A full host reload would silently reset any unsaved
        # edits the user made during the ~10s warmup window.
        try:
            config = self._load_translator_config()
            sync_raw = config.get("sync") if isinstance(config, Mapping) else None
            sync_config = sync_raw if isinstance(sync_raw, Mapping) else {}
            self._custom_litellm_providers = custom_provider_registry(
                sync_config.get("custom_litellm_providers"),
                allow_import=False,
            )
            self._custom_litellm_providers_load_error = ""
        except Exception:
            self._custom_litellm_providers = {}
            self._custom_litellm_providers_load_error = (
                "后台预热 LiteLLM 后重新校验自定义 Provider 失败，"
                "请重新加载设置页。"
            )
            self._log("后台预热 LiteLLM 后重新校验自定义 Provider 失败。")
        self._after_custom_providers_changed()

    def _detach_litellm_module_warmup(self) -> None:
        """Detach a still-running warmup thread from window teardown.

        A QThread destroyed while its import is still running aborts the
        process, so on shutdown the worker is released from the window's
        ownership but kept alive in the application-level retired set until
        the import finishes; the completed signal still fires and cleans it up
        afterwards.  The set lives at module scope so window destruction can
        never drop the last reference to a running thread.
        """
        worker = getattr(self, "_litellm_module_warmup_worker", None)
        if worker is None:
            return
        self._litellm_module_warmup_worker = None
        if not getattr(worker, "isRunning", lambda: False)():
            worker.deleteLater()
            return
        try:
            worker.setParent(None)
        except RuntimeError:
            pass
        RETIRED_LITELLM_WARMUP_WORKERS.add(worker)

    def _on_litellm_network_progress(
        self,
        message: str,
        *,
        worker_attr: str,
        status_label_name: str = "",
        worker: object | None = None,
    ) -> None:
        """Show mid-flight status for catalog/version/connection workers."""
        current = getattr(self, worker_attr, None)
        if current is None:
            return
        if worker is not None and worker is not current:
            return
        text = str(message or "").strip()
        if not text:
            return
        self._show_status(f"{text}（可再次点击停止）", 0)
        if status_label_name:
            label = getattr(self, status_label_name, None)
            if label is not None:
                label.setText(text)

    def _on_check_litellm_version(self) -> None:
        worker = getattr(self, "_litellm_version_worker", None)
        if self._request_cancel_litellm_worker(
            worker,
            button_name="litellm_check_version_btn",
            status_message="正在取消版本检查…",
        ):
            return
        if worker is not None:
            return
        button = getattr(self, "litellm_check_version_btn", None)
        if button is not None:
            button.setEnabled(True)
            button.setText("停止检查")
        worker = self._make_version_worker()
        worker.progress.connect(
            lambda message, owned=worker: self._on_litellm_network_progress(
                message,
                worker_attr="_litellm_version_worker",
                status_label_name="litellm_version_label",
                worker=owned,
            )
        )
        worker.completed.connect(self._on_litellm_version_checked)
        worker.finished.connect(worker.deleteLater)
        self._litellm_version_worker = worker
        worker.start()
        self._show_status("正在检查 LiteLLM 版本…（可再次点击停止）", 0)

    def _on_litellm_version_checked(
        self,
        installed: str,
        latest: str,
        compatible: str,
        requires_python: str,
        error: object,
    ) -> None:
        self._litellm_version_worker = None
        if self._is_shutdown_requested():
            return
        button = getattr(self, "litellm_check_version_btn", None)
        if button is not None:
            button.setText("检查更新")
        if is_cancelled_message(error):
            self._show_status("已取消版本检查。", 4000)
            self._on_sync_backend_changed(-1)
            return
        self._litellm_latest_version = latest
        self._litellm_latest_compatible_version = compatible
        self._litellm_latest_requires_python = requires_python
        self._refresh_litellm_version_label()
        if error:
            label = getattr(self, "litellm_version_label", None)
            if label is not None:
                current = f"本机 {installed}" if installed else "尚未安装"
                label.setText(f"{current}；检查更新失败，请稍后重试。")
        self._on_sync_backend_changed(-1)

    def _on_refresh_litellm_providers(self) -> None:
        worker = self._litellm_provider_catalog_worker
        if self._request_cancel_litellm_worker(
            worker,
            button_name="litellm_refresh_providers_btn",
            status_message="正在取消供应商列表加载…",
        ):
            return
        if worker is not None:
            return
        button = getattr(self, "litellm_refresh_providers_btn", None)
        if button is not None:
            button.setEnabled(True)
            button.setText("停止加载")
        worker = self._make_provider_catalog_worker()
        worker.progress.connect(
            lambda message, owned=worker: self._on_litellm_network_progress(
                message,
                worker_attr="_litellm_provider_catalog_worker",
                status_label_name="litellm_provider_catalog_status_label",
                worker=owned,
            )
        )
        worker.completed.connect(self._on_litellm_providers_loaded)
        worker.finished.connect(worker.deleteLater)
        self._litellm_provider_catalog_worker = worker
        worker.start()
        self._show_status("正在加载供应商…（可再次点击停止）", 0)

    def _on_litellm_providers_loaded(
        self,
        providers: object,
        source: str,
        error: object,
    ) -> None:
        self._litellm_provider_catalog_worker = None
        if self._is_shutdown_requested():
            return
        button = getattr(self, "litellm_refresh_providers_btn", None)
        if button is not None:
            button.setText("联网加载供应商")
        self._on_sync_backend_changed(-1)
        if is_cancelled_message(error):
            self._show_status("已取消供应商列表加载。", 4000)
            return
        if error and not providers:
            self._message_warning("供应商列表加载失败", str(error))
            return
        values = tuple(str(provider).strip().lower() for provider in providers)
        current = self._litellm_provider_combo_value()
        self._save_litellm_cache(
            lambda: self._litellm_cache.update_providers(
                values,
                source=source,
                litellm_version=installed_litellm_version(),
            )
        )
        self._populate_litellm_providers(values, selected=current)
        self._notify_providers_changed(current)
        self._refresh_litellm_catalog_status()
        if current:
            status = f"已加载 {len(values)} 个 LiteLLM 供应商；已保留当前选择。"
        else:
            status = f"已加载 {len(values)} 个 LiteLLM 供应商；未自动选择。"
        self._show_status(status, 8000)

    def _on_refresh_litellm_models(self) -> None:
        worker = self._litellm_catalog_worker
        if self._request_cancel_litellm_worker(
            worker,
            button_name="litellm_refresh_models_btn",
            status_message="正在取消模型列表加载…",
        ):
            return
        if worker is not None:
            return
        provider = self._current_litellm_provider()
        if not provider:
            return
        api_key = ""
        if provider != "ollama":
            try:
                api_key = self._load_api_key(provider)
            except ProviderCredentialStoreError:
                api_key = ""
        endpoint = native_catalog_endpoint(provider, self._custom_litellm_providers)
        allow_subset_only = False
        if endpoint is not None and endpoint.require_key:
            custom = self._custom_litellm_providers.get(provider)
            if custom is not None and not api_key and custom.api_key_env:
                # Same explicit env fallback as the request/connection paths;
                # never fall back to OPENAI_API_KEY for a third-party endpoint.
                api_key = str(self._environ().get(custom.api_key_env) or "").strip()
            if not api_key:
                if custom is not None:
                    # LiteLLM's online subset has no entry for user-defined ids,
                    # so the "subset only" fallback would always fail after a
                    # confusing prompt. Require a key (keyring or api_key_env).
                    self._message_information(CUSTOM_LITELLM_PROVIDER_COPY["missing_key_title"],
                        (
                            CUSTOM_LITELLM_PROVIDER_COPY["missing_key_body"].format(
                                label=endpoint.label
                            )
                            + (
                                CUSTOM_LITELLM_PROVIDER_COPY[
                                    "missing_key_env_hint"
                                ].format(env=custom.api_key_env)
                                if custom.api_key_env
                                else ""
                            )
                        ),
                    )
                    return
                reply = self._message_question("建议先保存 API Key",
                    (
                        f"{endpoint.label} 的官方模型列表需要已保存的 API Key。\n\n"
                        "请先在下方「Provider 凭据」中粘贴并保存密钥，再加载官方列表。\n\n"
                        "若仍继续，将只尝试 LiteLLM 在线子集目录（可能不完整，"
                        "且通常依赖 GitHub 网络，关代理时可能很慢或失败）。\n\n"
                        "是否仍使用 LiteLLM 子集目录？"
                    ),
                    yes_text="继续",
                    no_text="取消",
                    default="no",
                )
                if reply != "yes":
                    self._show_status(
                        f"已取消：请先保存 {endpoint.label} API Key 再加载官方模型列表。",
                        6000,
                    )
                    return
                allow_subset_only = True
        button = getattr(self, "litellm_refresh_models_btn", None)
        if button is not None:
            button.setEnabled(True)
            button.setText("停止加载")
        worker = self._make_model_catalog_worker(provider, api_key)
        worker.progress.connect(
            lambda message, owned=worker: self._on_litellm_network_progress(
                message,
                worker_attr="_litellm_catalog_worker",
                status_label_name="litellm_catalog_status_label",
                worker=owned,
            )
        )
        worker.completed.connect(
            lambda models, source, error, selected=provider: self._on_litellm_models_loaded(
                selected, models, error, source
            )
        )
        worker.finished.connect(worker.deleteLater)
        self._litellm_catalog_worker = worker
        worker.start()
        if allow_subset_only:
            self._show_status(
                f"未保存密钥：正在尝试 {provider} 的 LiteLLM 子集目录…"
                "（可再次点击停止）",
                0,
            )
        else:
            self._show_status(
                f"正在加载 {provider} 模型…（可再次点击停止）",
                0,
            )

    def _on_litellm_models_loaded(
        self, provider: str, models: object, error: object, source: str = ""
    ) -> None:
        self._litellm_catalog_worker = None
        if self._is_shutdown_requested():
            return
        button = getattr(self, "litellm_refresh_models_btn", None)
        if button is not None:
            button.setText("联网加载模型")
        self._on_sync_backend_changed(-1)
        if is_cancelled_message(error):
            self._show_status("已取消模型列表加载。", 4000)
            return
        if error and not models:
            self._message_warning("模型列表加载失败", str(error))
            return
        values = tuple(str(model) for model in models)
        self._save_litellm_cache(
            lambda: self._litellm_cache.update_models(
                provider,
                values,
                source=source,
                litellm_version=installed_litellm_version(),
            )
        )
        self._refresh_litellm_catalog_status()
        if self._current_litellm_provider() == provider:
            self._set_litellm_models(provider, values, preserve_current=True)
        message = f"已加载 {len(values)} 个 {provider} 模型。"
        if error:
            message = f"{message} {error}"
        self._show_status(message, 8000)

    def _on_test_litellm_connection(self) -> None:
        worker = self._litellm_connection_worker
        if self._request_cancel_litellm_worker(
            worker,
            button_name="litellm_test_connection_btn",
            status_message="正在取消连接测试…",
        ):
            status = getattr(self, "litellm_connection_status_label", None)
            if status is not None:
                status.setText("正在取消连接测试…")
            return
        if worker is not None:
            return
        model = self._litellm_model_text()
        if not model:
            self._message_information("缺少模型", "请先选择或填写模型。")
            return
        # Empty → backend loads the active key from the OS credential store.
        provider = self._current_litellm_provider()
        custom = self._custom_litellm_providers.get(provider)
        api_key = ""
        try:
            api_key = self._load_api_key(provider)
        except ProviderCredentialStoreError:
            api_key = ""
        if not api_key and custom is not None and custom.requires_key:
            env_key = (
                str(self._environ().get(custom.api_key_env) or "").strip()
                if custom.api_key_env
                else ""
            )
            if env_key:
                api_key = env_key
            else:
                self._message_information(CUSTOM_LITELLM_PROVIDER_COPY["missing_key_title"],
                    (
                        CUSTOM_LITELLM_PROVIDER_COPY[
                            "missing_connection_key"
                        ].format(label=custom.label)
                        + (
                            CUSTOM_LITELLM_PROVIDER_COPY[
                                "missing_connection_env_hint"
                            ].format(env=custom.api_key_env)
                            if custom.api_key_env
                            else CUSTOM_LITELLM_PROVIDER_COPY[
                                "missing_connection_env_suffix"
                            ]
                        )
                    ),
                )
                return
        self.litellm_test_connection_btn.setEnabled(True)
        self.litellm_test_connection_btn.setText("停止测试")
        self.litellm_connection_status_label.setText(
            "正在后台发起最小请求…（可再次点击停止）"
        )
        worker = self._make_connection_worker(model, api_key)
        worker.progress.connect(
            lambda message, owned=worker: self._on_litellm_network_progress(
                message,
                worker_attr="_litellm_connection_worker",
                status_label_name="litellm_connection_status_label",
                worker=owned,
            )
        )
        worker.completed.connect(self._on_litellm_connection_tested)
        worker.finished.connect(worker.deleteLater)
        self._litellm_connection_worker = worker
        worker.start()
        self._show_status("正在测试 LiteLLM 连接…（可再次点击停止）", 0)

    def _litellm_connection_operation_identity(self) -> str:
        """Digest the provider/model the LiteLLM settings page currently shows."""
        return litellm_connection_identity(
            provider=self._current_litellm_provider(),
            model=self._litellm_model_text(),
            custom_providers=getattr(self, "_custom_litellm_providers", None),
        )

    def _on_litellm_connection_tested(
        self,
        success: bool,
        message: str,
        operation_identity: str = "",
        sender: object | None = None,
    ) -> None:
        worker = self.sender() if sender is None else sender
        current = getattr(self, "_litellm_connection_worker", None)
        if worker is not None and current is not None and worker is not current:
            return
        if current is None or worker is current or worker is None:
            self._litellm_connection_worker = None
        if self._is_shutdown_requested():
            return
        self.litellm_test_connection_btn.setText("测试连接")
        if not is_current_identity(
            operation_identity,
            self._litellm_connection_operation_identity(),
        ):
            stale = LITELLM_CONNECTION_TEST_COPY["stale_result"]
            self.litellm_connection_status_label.setText(stale)
            self._show_status(stale, 6000)
            return
        self.litellm_connection_status_label.setText(message)
        self._on_sync_backend_changed(-1)
        if is_cancelled_message(message):
            self._show_status("已取消连接测试。", 4000)
            return
        if not success:
            self._show_status("LiteLLM 连接测试失败。", 5000)

    def _on_sync_backend_changed(self, _index: int, *, notify_host: bool = True) -> None:
        backend = self._selected_sync_backend()
        hint = getattr(self, "sync_backend_hint", None)
        if hint is None:
            return
        install_btn = getattr(self, "install_litellm_btn", None)
        install_progress = getattr(self, "litellm_install_progress", None)
        installing = self._is_install_running()
        idle = not self._is_global_task_running()
        installed_version = installed_litellm_version()
        installed = bool(installed_version) and importlib.util.find_spec("litellm") is not None
        if backend == "litellm":
            installed_version = installed_litellm_version()
            installed = bool(installed_version) and importlib.util.find_spec("litellm") is not None
            keyring_installed = importlib.util.find_spec("keyring") is not None
            state = "正在后台安装" if installing else ("已安装" if installed else "尚未安装")
            credential_state = "可用" if keyring_installed else "尚未安装"
            hint.setText(
                "同步替代模式；不使用 Gemini API Key，也没有远程 Batch 恢复。"
                f"LiteLLM：{state}；安全凭据支持：{credential_state}。"
            )
            if install_btn is not None:
                latest = str(getattr(self, "_litellm_latest_version", "") or "")
                compatible = str(
                    getattr(self, "_litellm_latest_compatible_version", "") or ""
                )
                target = compatible if latest else ""
                up_to_date = bool(
                    installed
                    and target
                    and version_key(installed_version) >= version_key(target)
                )
                compatibility_limited = bool(
                    latest and compatible and version_key(compatible) < version_key(latest)
                )
                no_compatible_release = bool(latest and not compatible)
                install_btn.setVisible(True)
                install_btn.setEnabled(
                    idle
                    and not installing
                    and not no_compatible_release
                    and not (up_to_date and keyring_installed)
                )
                if installing:
                    install_btn.setText("正在更新…" if installed else "正在安装…")
                elif not installed:
                    install_btn.setText("安装 LiteLLM")
                elif no_compatible_release:
                    install_btn.setText("当前 Python 无兼容版本")
                elif up_to_date and keyring_installed:
                    install_btn.setText(
                        "当前 Python 可用最新版"
                        if compatibility_limited
                        else "已是最新版"
                    )
                else:
                    install_btn.setText("更新 LiteLLM")
        else:
            hint.setText(
                "推荐路径仍为 Gemini；同步配置位于「模型」与「密钥」页，批量离线翻译仍使用 Gemini Batch。"
            )
            if install_btn is not None:
                install_btn.setVisible(False)
        if install_progress is not None:
            install_progress.setVisible(installing)
            if installing:
                # pip does not expose a trustworthy total; busy mode gives honest
                # visual feedback without inventing a completion percentage.
                install_progress.setRange(0, 0)
                install_progress.setFormat("正在后台更新 LiteLLM…" if installed else "正在后台安装 LiteLLM…")

        model_combo = getattr(self, "litellm_model_combo", None)
        litellm_active = backend == "litellm" and not installing
        can_edit = litellm_active and idle
        provider = self._current_litellm_provider()
        model = self._litellm_model_text()
        backend_combo = getattr(self, "sync_backend_combo", None)
        if backend_combo is not None:
            backend_combo.setEnabled(idle)
        provider_combo = getattr(self, "litellm_provider_combo", None)
        if provider_combo is not None:
            provider_combo.setEnabled(can_edit)
        provider_worker = getattr(self, "_litellm_provider_catalog_worker", None)
        provider_button = getattr(self, "litellm_refresh_providers_btn", None)
        if provider_button is not None:
            # Keep enabled while a catalog load is in flight so the user can stop.
            provider_button.setEnabled(can_edit or provider_worker is not None)
            if provider_worker is not None:
                provider_button.setText(
                    "正在取消…"
                    if getattr(provider_worker, "is_cancelled", lambda: False)()
                    else "停止加载"
                )
            else:
                provider_button.setText("联网加载供应商")
        clear_provider = getattr(self, "litellm_clear_provider_btn", None)
        if clear_provider is not None:
            clear_provider.setEnabled(can_edit and bool(provider))
        if model_combo is not None:
            model_combo.setEnabled(can_edit and bool(provider))
        model_worker = getattr(self, "_litellm_catalog_worker", None)
        model_button = getattr(self, "litellm_refresh_models_btn", None)
        if model_button is not None:
            model_button.setEnabled(
                (can_edit and bool(provider)) or model_worker is not None
            )
            if model_worker is not None:
                model_button.setText(
                    "正在取消…"
                    if getattr(model_worker, "is_cancelled", lambda: False)()
                    else "停止加载"
                )
            else:
                model_button.setText("联网加载模型")
        credential_enabled = can_edit and bool(provider) and provider != "ollama"
        manage_keys_btn = getattr(self, "litellm_manage_keys_btn", None)
        if manage_keys_btn is not None:
            manage_keys_btn.setEnabled(credential_enabled)
            manage_keys_btn.setToolTip(
                ""
                if credential_enabled
                else (
                    "该 Provider 不需要 API Key"
                    if provider == "ollama"
                    else "请先选择 Provider"
                )
            )
        connection_worker = getattr(self, "_litellm_connection_worker", None)
        test_button = getattr(self, "litellm_test_connection_btn", None)
        if test_button is not None:
            test_button.setEnabled(
                (can_edit and bool(provider) and bool(model))
                or connection_worker is not None
            )
            if connection_worker is not None:
                test_button.setText(
                    "正在取消…"
                    if getattr(connection_worker, "is_cancelled", lambda: False)()
                    else "停止测试"
                )
            else:
                test_button.setText("测试连接")
        version_worker = getattr(self, "_litellm_version_worker", None)
        version_button = getattr(self, "litellm_check_version_btn", None)
        if version_button is not None:
            # Stay clickable while a check is in flight so the user can stop it.
            version_button.setEnabled(
                (idle and not installing) or version_worker is not None
            )
            if version_worker is not None:
                version_button.setText(
                    "正在取消…"
                    if getattr(version_worker, "is_cancelled", lambda: False)()
                    else "停止检查"
                )
            else:
                version_button.setText("检查更新")
        self._refresh_custom_provider_actions()
        self._refresh_litellm_credential_status()
        if notify_host and not self._is_loading_config():
            self._notify_backend_gating()

