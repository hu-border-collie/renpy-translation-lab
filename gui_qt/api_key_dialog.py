"""Small dialog for viewing and editing local API keys."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .api_key_helpers import commit_pending_key, mask_api_key


class ApiKeyDialog(QDialog):
    """Manage keys stored in api_keys.json without using QInputDialog."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        keys: list[str],
        env_key_count: int = 0,
    ):
        super().__init__(parent)
        self.setObjectName("api_key_dialog")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle("管理 API Key")
        self.setModal(True)
        self.resize(480, 420)
        self._keys = [key for key in keys if isinstance(key, str)]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        intro = QLabel(
            "密钥仅保存在本地配置文件中，不会上传或代理。"
            "可添加多个 Key；删除选中项后点击「保存」生效。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("config_hint_label")
        layout.addWidget(intro)

        layout.addWidget(QLabel("已保存的 Key："))

        self.key_list = QListWidget()
        self.key_list.setObjectName("api_key_list")
        self.key_list.setMinimumHeight(120)
        layout.addWidget(self.key_list)

        list_actions = QHBoxLayout()
        self.remove_btn = QPushButton("删除选中")
        self.remove_btn.setObjectName("secondary_btn")
        self.remove_btn.clicked.connect(self._on_remove_selected)
        list_actions.addWidget(self.remove_btn)
        list_actions.addStretch()
        layout.addLayout(list_actions)

        layout.addWidget(QLabel("添加新 Key："))

        input_hint = QLabel(
            "输入框默认以圆点隐藏 Key。点「显示明文」可核对粘贴内容是否正确，"
            "再点「隐藏明文」恢复隐藏。"
        )
        input_hint.setWordWrap(True)
        input_hint.setObjectName("config_hint_label")
        layout.addWidget(input_hint)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.new_key_edit = QLineEdit()
        self.new_key_edit.setPlaceholderText("在此粘贴新的 API Key")
        self.new_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.new_key_edit.setObjectName("api_key_input")
        input_row.addWidget(self.new_key_edit, 1)

        self.toggle_visible_btn = QPushButton("显示明文")
        self.toggle_visible_btn.setObjectName("secondary_btn")
        self.toggle_visible_btn.setCheckable(True)
        self.toggle_visible_btn.setToolTip("切换输入框中 Key 的显示/隐藏，方便核对是否粘贴正确")
        self.toggle_visible_btn.toggled.connect(self._on_toggle_visibility)
        input_row.addWidget(self.toggle_visible_btn)

        self.add_btn = QPushButton("添加 Key")
        self.add_btn.setObjectName("api_btn")
        self.add_btn.clicked.connect(self._on_add_key)
        input_row.addWidget(self.add_btn)
        layout.addLayout(input_row)

        self.new_key_edit.returnPressed.connect(self._on_add_key)

        self.env_label = QLabel()
        self.env_label.setWordWrap(True)
        self.env_label.setObjectName("config_hint_label")
        layout.addWidget(self.env_label)

        if env_key_count > 0:
            self.env_label.setText(
                f"另检测到 {env_key_count} 个通过环境变量配置的有效 Key（只读，不会写入文件）。"
            )
        else:
            self.env_label.hide()

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save_btn = button_box.button(QDialogButtonBox.StandardButton.Save)
        if save_btn is not None:
            save_btn.setText("保存")
        cancel_btn = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_btn is not None:
            cancel_btn.setText("取消")
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._refresh_key_list()

    def result_keys(self) -> list[str]:
        return list(self._keys)

    def _on_accept(self) -> None:
        pending = self.new_key_edit.text().strip()
        updated_keys, error = commit_pending_key(self._keys, pending)
        if error == "duplicate":
            QMessageBox.information(
                self,
                "Key 已存在",
                "输入框中的 Key 已在列表中。请清空输入框，或先删除列表中的重复项后再保存。",
            )
            return

        if pending:
            self._keys = updated_keys
            self.new_key_edit.clear()
            self._refresh_key_list()

        self.accept()

    def _refresh_key_list(self) -> None:
        self.key_list.clear()
        if not self._keys:
            self.key_list.addItem("（尚未保存任何 Key）")
            self.remove_btn.setEnabled(False)
            return

        self.remove_btn.setEnabled(True)
        for index, key in enumerate(self._keys, start=1):
            self.key_list.addItem(f"{index}. {mask_api_key(key)}")

    def _on_toggle_visibility(self, checked: bool) -> None:
        self.new_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self.toggle_visible_btn.setText("隐藏明文" if checked else "显示明文")

    def _on_add_key(self) -> None:
        value = self.new_key_edit.text().strip()
        if not value:
            QMessageBox.information(self, "请输入 Key", "请先粘贴要添加的 API Key。")
            return

        updated_keys, error = commit_pending_key(self._keys, value)
        if error == "duplicate":
            QMessageBox.information(self, "Key 已存在", "这个 Key 已经在列表中。")
            return

        self._keys = updated_keys
        self.new_key_edit.clear()
        self._refresh_key_list()
        self.key_list.setCurrentRow(self.key_list.count() - 1)

    def _on_remove_selected(self) -> None:
        if not self._keys:
            return

        row = self.key_list.currentRow()
        if row < 0 or row >= len(self._keys):
            QMessageBox.information(self, "请选择 Key", "请先在列表中选中要删除的 Key。")
            return

        self._keys.pop(row)
        self._refresh_key_list()
        if self._keys:
            self.key_list.setCurrentRow(min(row, self.key_list.count() - 1))