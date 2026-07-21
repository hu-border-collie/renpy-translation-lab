"""Dialog: pick a game directory or zip, name it, preview Game_* folder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from game_ingest import (
    folder_conflict_message,
    game_name_to_folder,
    suggest_game_name,
    validate_game_name,
)


@dataclass(frozen=True)
class GameIngestDialogResult:
    source: Path
    game_name: str
    folder_name: str


def _display_path(path: Path) -> str:
    """Absolute path string for UI (native separators, e.g. backslash on Windows)."""
    try:
        path = path.expanduser().resolve()
    except OSError:
        path = path.expanduser()
    return str(path)


class GameIngestDialog(QDialog):
    """Collect source path + game name with live Game_* folder preview."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        workspace_root: Path,
        start_dir: Path | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("导入游戏")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setObjectName("game_ingest_dialog")

        self._workspace_root = Path(workspace_root).expanduser().resolve()
        self._start_dir = Path(start_dir or workspace_root).expanduser()
        self._source: Path | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        hint = QLabel(
            "从游戏目录或 .zip 复制到工作区，整理为 Game_*/original/work/build 并加入总表。\n"
            "不会移动源文件，也不会自动准备 work。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        # Source path on its own full-width row; browse buttons on the next row
        # so the path is not squeezed between two wide buttons.
        self._source_edit = QLineEdit()
        self._source_edit.setReadOnly(True)
        self._source_edit.setPlaceholderText("选择游戏目录或 .zip…")
        self._source_edit.setObjectName("game_ingest_source_edit")
        self._source_edit.setMinimumWidth(360)
        form.addRow("源路径", self._source_edit)

        browse_row = QHBoxLayout()
        browse_row.addStretch(1)
        browse_dir_btn = QPushButton("浏览文件夹…")
        browse_dir_btn.setObjectName("secondary_btn")
        browse_dir_btn.clicked.connect(self._browse_directory)
        browse_row.addWidget(browse_dir_btn)
        browse_zip_btn = QPushButton("浏览 zip…")
        browse_zip_btn.setObjectName("secondary_btn")
        browse_zip_btn.clicked.connect(self._browse_zip)
        browse_row.addWidget(browse_zip_btn)
        form.addRow("", browse_row)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("游戏名称（总表显示名）")
        self._name_edit.setObjectName("game_ingest_name_edit")
        self._name_edit.textChanged.connect(self._refresh_preview)
        form.addRow("游戏名称", self._name_edit)

        # Destination preview only — not the selected source path.
        self._folder_preview = QLabel("—")
        self._folder_preview.setObjectName("game_ingest_folder_preview")
        self._folder_preview.setTextFormat(Qt.TextFormat.PlainText)
        self._folder_preview.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("最终目录", self._folder_preview)

        # Read-only line edit: no QLabel wrap-after-"C:" clipping on Windows paths.
        self._path_preview = QLineEdit()
        self._path_preview.setReadOnly(True)
        self._path_preview.setObjectName("game_ingest_path_preview")
        self._path_preview.setPlaceholderText("—")
        self._path_preview.setMinimumWidth(360)
        self._path_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        form.addRow("将创建到", self._path_preview)

        layout.addLayout(form)

        self._error_label = QLabel("")
        self._error_label.setObjectName("game_ingest_error_label")
        self._error_label.setWordWrap(True)
        self._error_label.setTextFormat(Qt.TextFormat.PlainText)
        self._error_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._error_label.setStyleSheet("color: #b91c1c;")
        layout.addWidget(self._error_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setText("导入")
        # Keep Cancel in Chinese to match the rest of the workbench.
        cancel_btn = self._buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_btn is not None:
            cancel_btn.setText("取消")
        layout.addWidget(self._buttons)

        self._refresh_preview()

    def result_payload(self) -> GameIngestDialogResult | None:
        if self._source is None:
            return None
        game_name = self._name_edit.text().strip()
        folder = game_name_to_folder(game_name)
        if not folder:
            return None
        return GameIngestDialogResult(
            source=self._source,
            game_name=game_name,
            folder_name=folder,
        )

    def _set_source(self, path: Path) -> None:
        resolved = Path(path).expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:
            pass
        self._source = resolved
        display = _display_path(resolved)
        self._source_edit.setText(display)
        self._source_edit.setToolTip(str(resolved))
        self._source_edit.setCursorPosition(0)
        # Prefill game name from source; user can edit.
        self._name_edit.setText(suggest_game_name(resolved))
        self._refresh_preview()

    def _browse_directory(self) -> None:
        start = self._start_dir if self._start_dir.is_dir() else Path.home()
        try:
            start = start.resolve()
        except OSError:
            pass
        picked = QFileDialog.getExistingDirectory(self, "选择游戏目录", str(start))
        if picked:
            self._set_source(Path(picked))

    def _browse_zip(self) -> None:
        start = self._start_dir if self._start_dir.is_dir() else Path.home()
        try:
            start = start.resolve()
        except OSError:
            pass
        picked, _filter = QFileDialog.getOpenFileName(
            self,
            "选择游戏压缩包",
            str(start),
            "Zip archives (*.zip);;All files (*.*)",
        )
        if picked:
            self._set_source(Path(picked))

    def _destination_path(self, folder: str) -> Path:
        """Absolute path of the Game_* folder that will be created."""
        return (self._workspace_root / folder).resolve()

    def _refresh_preview(self) -> None:
        game_name = self._name_edit.text()
        folder = game_name_to_folder(game_name)
        error = ""
        if self._source is None:
            # Keep as one short phrase; non-breaking space avoids "或 / zip" split.
            error = "请先选择游戏目录或\u00a0.zip。"
        else:
            error = validate_game_name(game_name)
            if not error and folder:
                error = folder_conflict_message(self._workspace_root, folder)

        if folder:
            # Value only — form label already says 最终目录 / 将创建到.
            self._folder_preview.setText(folder)
            dest = self._destination_path(folder)
            display = _display_path(dest)
            self._path_preview.setText(display)
            self._path_preview.setToolTip(str(dest))
            self._path_preview.setCursorPosition(0)
        else:
            self._folder_preview.setText("（请输入游戏名称）")
            self._path_preview.clear()
            self._path_preview.setToolTip("")

        self._error_label.setText(error)
        self._ok_button.setEnabled(
            not bool(error) and self._source is not None and bool(folder)
        )
