"""Dialog for adding / editing user-defined OpenAI-compatible LiteLLM providers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from .litellm_settings import validate_custom_provider_form
from .user_copy import CUSTOM_LITELLM_PROVIDER_COPY


class CustomLiteLLMProviderDialog(QDialog):
    """Collect one ``sync.custom_litellm_providers`` entry.

    The id is locked while editing an existing provider: it doubles as the
    model prefix and keyring username, so renaming it would orphan credentials
    and cached catalogs.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        provider: Mapping[str, Any] | None = None,
        reserved: frozenset[str] | None = None,
        title: str = "添加自定义 Provider",
    ):
        super().__init__(parent)
        self.setObjectName("custom_provider_dialog")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle(str(title or "自定义 Provider"))
        self.setModal(True)
        self.resize(520, 340)
        self._reserved = frozenset(reserved) if reserved is not None else None
        self._editing = bool(provider is not None)
        values = dict(provider) if isinstance(provider, Mapping) else {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        intro = QLabel(CUSTOM_LITELLM_PROVIDER_COPY["dialog_intro"])
        intro.setWordWrap(True)
        intro.setObjectName("config_hint_label")
        layout.addWidget(intro)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.id_edit = QLineEdit(str(values.get("id") or ""))
        self.id_edit.setPlaceholderText("例如 opencode-go")
        self.id_edit.setReadOnly(self._editing)
        self.id_edit.setToolTip(CUSTOM_LITELLM_PROVIDER_COPY["id_tooltip"])
        form.addRow("Provider id：", self.id_edit)

        self.label_edit = QLineEdit(str(values.get("label") or ""))
        self.label_edit.setPlaceholderText("留空则使用 id")
        form.addRow("显示名称：", self.label_edit)

        self.base_url_edit = QLineEdit(str(values.get("base_url") or ""))
        self.base_url_edit.setPlaceholderText("https://opencode.ai/zen/go/v1")
        form.addRow("API Base URL：", self.base_url_edit)

        self.models_url_edit = QLineEdit(str(values.get("models_url") or ""))
        self.models_url_edit.setPlaceholderText("留空则使用 API Base + /models")
        form.addRow("模型列表 URL：", self.models_url_edit)

        self.api_key_env_edit = QLineEdit(str(values.get("api_key_env") or ""))
        self.api_key_env_edit.setPlaceholderText("可选，例如 OPENCODE_GO_API_KEY")
        self.api_key_env_edit.setToolTip(CUSTOM_LITELLM_PROVIDER_COPY["env_tooltip"])
        form.addRow("密钥环境变量：", self.api_key_env_edit)

        self.requires_key_cb = QCheckBox("需要 API Key")
        self.requires_key_cb.setChecked(
            bool(values.get("requires_key", True))
        )
        self.requires_key_cb.setToolTip(
            CUSTOM_LITELLM_PROVIDER_COPY["requires_key_tooltip"]
        )
        form.addRow("认证：", self.requires_key_cb)
        layout.addLayout(form)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setObjectName("settings_error_label")
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        self._buttons = buttons
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        provider_id = self.id_edit.text().strip()
        if self._editing:
            # The id field is locked and the id is already registered; only the
            # mutable fields are re-validated on accept.
            error = validate_custom_provider_form(
                provider_id,
                self.base_url_edit.text().strip(),
                label=self.label_edit.text().strip(),
                models_url=self.models_url_edit.text().strip(),
                api_key_env=self.api_key_env_edit.text().strip(),
                # The id is locked and already registered; skip conflict
                # checking (empty reserved set) while still validating the
                # character set and the mutable fields.
                reserved=frozenset(),
            )
        else:
            error = validate_custom_provider_form(
                provider_id,
                self.base_url_edit.text().strip(),
                label=self.label_edit.text().strip(),
                models_url=self.models_url_edit.text().strip(),
                api_key_env=self.api_key_env_edit.text().strip(),
                reserved=self._reserved,
            )
        if error:
            self.error_label.setText(error)
            self.error_label.setVisible(True)
            return
        self.accept()

    def result_provider(self) -> dict[str, str]:
        """Return the validated entry for ``sync.custom_litellm_providers``."""
        provider_id = self.id_edit.text().strip().lower()
        base_url = self.base_url_edit.text().strip()
        models_url = self.models_url_edit.text().strip()
        entry: dict[str, str] = {
            "id": provider_id,
            "label": self.label_edit.text().strip() or provider_id,
            "base_url": base_url,
        }
        if models_url:
            entry["models_url"] = models_url
        api_key_env = self.api_key_env_edit.text().strip()
        if api_key_env:
            entry["api_key_env"] = api_key_env
        if not self.requires_key_cb.isChecked():
            entry["requires_key"] = False
        return entry
