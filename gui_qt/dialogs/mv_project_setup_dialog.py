"""Focused MV material setup; saves settings, then delegates checks to doctor."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from engine_adapters.project_config import resolve_project_configuration
from engine_adapters.project_setup import build_mv_setup_config, mv_project_setup_status
from gui_qt.user_copy import MV_PROJECT_COPY


class MvProjectSetupDialog(QDialog):
    """Collect explicit source/window choices without changing text scope or game files."""

    def __init__(self, parent=None, *, config: dict, config_file: str | Path, target_root: str):
        super().__init__(parent)
        self.config_file = Path(config_file)
        self.config = resolve_project_configuration(
            {**config, "engine": "rpgmaker_mv"}, self.config_file
        )
        self.config["game_root"] = target_root
        self.config = resolve_project_configuration(self.config, self.config_file)
        self.setup = mv_project_setup_status(self.config, self.config_file)
        self._result_config = None
        self.setWindowTitle(MV_PROJECT_COPY["setup_title"])
        self.setObjectName("mv_project_setup_dialog")
        self.setMinimumWidth(650)
        layout = QVBoxLayout(self)
        hint = QLabel(MV_PROJECT_COPY["setup_message"])
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.target_edit = QLineEdit(self.setup["target_root"])
        self.target_edit.setObjectName("mv_setup_target")
        self.target_edit.setReadOnly(True)
        form.addRow(MV_PROJECT_COPY["setup_target"], self.target_edit)
        self.source_edit = QLineEdit(self.setup["source_root"] or self.setup["source_suggestion"])
        self.source_edit.setObjectName("mv_setup_source")
        self.source_edit.setToolTip(MV_PROJECT_COPY["setup_source_hint"])
        form.addRow(
            MV_PROJECT_COPY["setup_source"], self._browse_row(self.source_edit, self._browse_source)
        )
        self.window_edit = QLineEdit(self.setup["support_window_file"])
        self.window_edit.setObjectName("mv_setup_window")
        self.window_edit.setToolTip(MV_PROJECT_COPY["setup_window_hint"])
        form.addRow(
            MV_PROJECT_COPY["setup_window"], self._browse_row(self.window_edit, self._browse_window)
        )
        layout.addLayout(form)
        for text in (
            MV_PROJECT_COPY["setup_window_hint"],
            MV_PROJECT_COPY[
                "setup_selected_scope" if self.setup["scope"] == "selected" else "setup_full_scope"
            ],
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            layout.addWidget(label)
        self.error_label = QLabel()
        self.error_label.setObjectName("mv_setup_error")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText(
            MV_PROJECT_COPY["setup_save"]
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self._on_save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def result_config(self) -> dict | None:
        return self._result_config

    def _browse_row(self, edit, callback):
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        button = QPushButton("浏览…")
        button.clicked.connect(callback)
        row.addWidget(button)
        return host

    def _browse_source(self):
        path = QFileDialog.getExistingDirectory(
            self, MV_PROJECT_COPY["setup_source"], self.source_edit.text()
        )
        if path:
            self.source_edit.setText(path)

    def _browse_window(self):
        path, _ = QFileDialog.getOpenFileName(
            self, MV_PROJECT_COPY["setup_window"], self.window_edit.text(), "JSON (*.json)"
        )
        if path:
            self.window_edit.setText(path)

    def _on_save(self):
        try:
            self._result_config = build_mv_setup_config(
                self.config,
                self.config_file,
                target_root=self.target_edit.text().strip(),
                source_root=self.source_edit.text().strip(),
                support_window_file=self.window_edit.text().strip(),
            )
        except (ValueError, OSError) as exc:
            self.error_label.setText(str(exc))
            return
        self.accept()
