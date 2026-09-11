"""Dialog: choose export directory and run export-only or apply+export.

The dialog never authorizes writes by itself.  It validates path shape and
surfaces the current check gate, then hands the same CLI options to the normal
apply executor so stale checks, source snapshots, structure blockers, plan
bindings and destination conflicts are revalidated there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import batch_export

EXPORT_ONLY_MODE = "export-only"
APPLY_EXPORT_MODE = "apply-export"


@dataclass(frozen=True)
class ExportPreviewFacts:
    """Read-only facts shown before the user chooses a destination."""

    game_root: str
    tl_dir: str
    package_dir: str
    pending_files: int
    pending_lines: int
    gate_label: str
    files: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class ExportWritebackChoice:
    mode: str
    export_root: str


def _mode_label(mode: str) -> str:
    return "仅导出（不修改游戏）" if mode == EXPORT_ONLY_MODE else "写回并导出（修改游戏）"


class ExportWritebackDialog(QDialog):
    """Collect export destination + mode with live gate/path preview."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        facts: ExportPreviewFacts,
        start_dir: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出翻译文件")
        self.setModal(True)
        self.setMinimumWidth(640)
        self.setObjectName("export_writeback_dialog")

        self._facts = facts
        self._start_dir = Path(start_dir or facts.game_root or facts.package_dir)
        self._choice: ExportWritebackChoice | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        hint = QLabel(
            "选择导出目录后执行「仅导出」或「写回并导出」。"
            "窗口只做预检，运行时仍会重新校验 check、源快照、结构闸门与目录冲突。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        form = QFormLayout()
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._mode_combo = QComboBox()
        self._mode_combo.setObjectName("export_writeback_mode_combo")
        self._mode_combo.addItem(_mode_label(EXPORT_ONLY_MODE), EXPORT_ONLY_MODE)
        self._mode_combo.addItem(_mode_label(APPLY_EXPORT_MODE), APPLY_EXPORT_MODE)
        self._mode_combo.currentIndexChanged.connect(self._refresh)
        form.addRow("操作", self._mode_combo)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setObjectName("export_writeback_path_edit")
        self._path_edit.setPlaceholderText("选择不存在或为空的导出目录…")
        self._path_edit.textChanged.connect(self._refresh)
        path_row.addWidget(self._path_edit, 1)
        browse_btn = QPushButton("浏览…")
        browse_btn.setObjectName("secondary_btn")
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        form.addRow("导出目录", path_row)

        self._source_root_label = QLabel(facts.game_root or "—")
        self._source_root_label.setObjectName("export_writeback_source_root")
        self._source_root_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("源根（游戏根）", self._source_root_label)

        self._target_root_label = QLabel("—")
        self._target_root_label.setObjectName("export_writeback_target_root")
        self._target_root_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("目标根", self._target_root_label)

        gate_text = (
            f"{facts.gate_label}；待处理 {facts.pending_files} 个文件 / "
            f"{facts.pending_lines} 行"
        )
        gate_label = QLabel(gate_text)
        gate_label.setWordWrap(True)
        gate_label.setObjectName("export_writeback_gate_label")
        form.addRow("当前门禁", gate_label)

        layout.addLayout(form)

        preview_label = QLabel("受管文件（执行时以最新 check 和源快照为准）：")
        preview_label.setObjectName("config_hint_label")
        layout.addWidget(preview_label)
        self._tree = QTreeWidget()
        self._tree.setObjectName("export_writeback_preview_tree")
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["文件", "任务数", "说明"])
        self._tree.setRootIsDecorated(False)
        self._tree.setUniformRowHeights(True)
        self._tree.setMinimumHeight(180)
        for relative_path, task_count in facts.files:
            item = QTreeWidgetItem(
                [relative_path, str(max(0, int(task_count))), "待执行时复核"]
            )
            self._tree.addTopLevelItem(item)
        if not facts.files:
            self._tree.addTopLevelItem(
                QTreeWidgetItem(["（无可显示的 manifest 文件列表）", "", ""])
            )
        layout.addWidget(self._tree, 1)

        self._notice_label = QLabel("")
        self._notice_label.setWordWrap(True)
        self._notice_label.setObjectName("export_writeback_notice_label")
        layout.addWidget(self._notice_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setText("继续")
        self._buttons.rejected.connect(self.reject)
        self._buttons.accepted.connect(self._accept)
        layout.addWidget(self._buttons)

        self._refresh()

    def _selected_mode(self) -> str:
        data = self._mode_combo.currentData()
        return str(data or EXPORT_ONLY_MODE)

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择导出目录",
            str(self._start_dir) if self._start_dir else "",
        )
        if selected:
            self._path_edit.setText(selected)

    def _validate_path(self):
        raw = self._path_edit.text().strip()
        if not raw:
            return None, "请选择导出目录。"
        mode = self._selected_mode()
        record_filename = (
            batch_export.APPLY_EXPORT_RECORD_FILE
            if mode == APPLY_EXPORT_MODE
            else batch_export.EXPORT_ONLY_RECORD_FILE
        )
        try:
            root = batch_export.validate_export_root(
                raw,
                game_root=self._facts.game_root,
                package_dir=self._facts.package_dir,
                record_filename=record_filename,
            )
        except batch_export.ExportOnlyError as exc:
            return None, str(exc)
        return root, ""

    def _refresh(self) -> None:
        root, error = self._validate_path()
        self._target_root_label.setText(root.canonical_path if root else "—")
        notice = error
        if root is not None and not error:
            requested = Path(root.requested_path)
            has_entries = False
            if requested.is_dir():
                try:
                    with os.scandir(requested) as iterator:
                        has_entries = next(iterator, None) is not None
                except OSError as exc:
                    error = f"导出目录无法读取：{exc}"
                    notice = error
            if has_entries and not error:
                receipt = Path(root.record_path)
                if receipt.is_file():
                    notice = (
                        "目标目录已有导出记录；执行时会复核回执、受管树与源快照。"
                        "不会被合并、清空或自动清理。"
                    )
                else:
                    error = (
                        "目标目录非空且没有匹配的导出记录；请选择新目录。"
                        "工具不会合并或清空已有文件。"
                    )
                    notice = error
        self._notice_label.setText(notice)
        self._ok_button.setEnabled(root is not None and not error)

    def _accept(self) -> None:
        root, error = self._validate_path()
        if root is None or error:
            self._refresh()
            return
        self._choice = ExportWritebackChoice(
            mode=self._selected_mode(),
            export_root=root.canonical_path,
        )
        self.accept()

    def choice(self) -> ExportWritebackChoice | None:
        return self._choice

    def selected_mode(self) -> str:
        return self._selected_mode()

    def selected_export_root(self) -> str:
        return self._path_edit.text().strip()
