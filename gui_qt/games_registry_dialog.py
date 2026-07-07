"""Modal wrapper around the embeddable workspace registry panel."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget

from .games_registry_panel import GamesRegistryPanel


class GamesRegistryDialog(QDialog):
    """Thin modal wrapper kept for backward-compatible tests and callers."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        workspace_root: Path,
        current_game_root: Path | None = None,
        current_doctor_report: dict | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("games_registry_dialog")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle("工作区项目总览")
        self.setModal(True)
        self.resize(1040, 760)

        self._selected_project_root = ""

        def on_switch_project(target: str) -> bool:
            self._selected_project_root = target
            self.accept()
            return True

        self._panel = GamesRegistryPanel(
            self,
            workspace_root=workspace_root,
            current_game_root=current_game_root,
            get_doctor_report=lambda: current_doctor_report,
            on_switch_project=on_switch_project,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self._panel, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.setText("关闭")
        layout.addWidget(buttons)

        self._panel.activate_section()

    def selected_project_root(self) -> str:
        return self._selected_project_root

    def __getattr__(self, name: str):
        return getattr(self._panel, name)