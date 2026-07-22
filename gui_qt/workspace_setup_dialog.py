"""Dialog: create or attach a workspace with read-only plan preview."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from games_registry import WorkspaceScene, plan_workspace_setup

from .games_registry_actions import apply_workspace_setup_action
from .path_utils import canonical_abs_path


@dataclass(frozen=True)
class WorkspaceSetupDialogResult:
    workspace: Path
    message: str
    project_count: int
    created_registry: bool = False


def _display_path(path: Path) -> str:
    try:
        path = path.expanduser().resolve()
    except (OSError, RuntimeError):
        # RuntimeError: symlink loops on some platforms (match normalize_workspace_path).
        path = path.expanduser()
    return str(path)


def _scene_label(scene: WorkspaceScene) -> str:
    return {
        WorkspaceScene.MISSING_PATH: "目录不存在（可创建）",
        WorkspaceScene.NOT_DIRECTORY: "不是目录",
        WorkspaceScene.NOT_WRITABLE: "目录不可写",
        WorkspaceScene.EMPTY: "空目录",
        WorkspaceScene.REGISTRY_OK: "已有合法总表",
        WorkspaceScene.REGISTRY_CORRUPT: "总表损坏",
        WorkspaceScene.GAMES_MD_ONLY: "仅有 GAMES.md",
        WorkspaceScene.GAME_DIRS_ONLY: "仅有 Game_* 目录",
        WorkspaceScene.MIXED: "混合内容",
    }.get(scene, scene.value)


class WorkspaceSetupDialog(QDialog):
    """Pick a directory, preview setup plan, then apply shared core actions."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        start_dir: Path | None = None,
        initial_path: Path | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("创建 / 接入工作区")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setObjectName("workspace_setup_dialog")

        self._start_dir = Path(start_dir or Path.home()).expanduser()
        self._selected: Path | None = None
        self._result: WorkspaceSetupDialogResult | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        hint = QLabel(
            "选择或新建工作区根目录，预览后初始化/接入 games_registry.json。\n"
            "不会自动准备项目 work，也不会下载 Ren'Py SDK。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        self._path_edit.setPlaceholderText("选择存放 Game_* 与 games_registry.json 的文件夹…")
        self._path_edit.setObjectName("workspace_setup_path_edit")
        self._path_edit.setMinimumWidth(360)
        form.addRow("工作区", self._path_edit)

        browse_row = QHBoxLayout()
        browse_row.addStretch(1)
        browse_btn = QPushButton("浏览…")
        browse_btn.setObjectName("workspace_setup_browse_btn")
        browse_btn.clicked.connect(self._browse)
        browse_row.addWidget(browse_btn)
        form.addRow("", browse_row)

        self._scene_label = QLabel("—")
        self._scene_label.setObjectName("workspace_setup_scene_label")
        form.addRow("状态", self._scene_label)

        layout.addLayout(form)

        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setObjectName("workspace_setup_preview")
        self._preview.setMinimumHeight(140)
        self._preview.setPlaceholderText("选择目录后显示只读预览…")
        layout.addWidget(self._preview)

        self._import_md_cb = QCheckBox("从 GAMES.md 导入（已有总表时按路径合并）")
        self._import_md_cb.setObjectName("workspace_setup_import_md")
        self._import_md_cb.toggled.connect(self._sync_confirm_enabled)
        layout.addWidget(self._import_md_cb)

        self._discover_cb = QCheckBox("扫描并登记尚未出现在总表中的 Game_*")
        self._discover_cb.setObjectName("workspace_setup_discover")
        self._discover_cb.toggled.connect(self._sync_confirm_enabled)
        layout.addWidget(self._discover_cb)

        self._render_md_cb = QCheckBox("完成后同步生成 GAMES.md")
        self._render_md_cb.setObjectName("workspace_setup_render_md")
        layout.addWidget(self._render_md_cb)

        self._error_label = QLabel("")
        self._error_label.setObjectName("workspace_setup_error_label")
        self._error_label.setWordWrap(True)
        self._error_label.setTextFormat(Qt.TextFormat.PlainText)
        self._error_label.setStyleSheet("color: #b91c1c;")
        layout.addWidget(self._error_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        self._ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setText("接入工作区")
        self._ok_button.setObjectName("workspace_setup_ok_btn")
        cancel_btn = self._buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_btn is not None:
            cancel_btn.setText("取消")
        layout.addWidget(self._buttons)

        if initial_path is not None:
            self._set_path(Path(initial_path))
        else:
            self._refresh_plan()

    def result_payload(self) -> WorkspaceSetupDialogResult | None:
        return self._result

    def _browse(self) -> None:
        start = self._selected or self._start_dir
        if not start.is_dir():
            start = Path.home()
        try:
            start = start.resolve()
        except (OSError, RuntimeError):
            pass
        picked = QFileDialog.getExistingDirectory(
            self,
            "选择工作区根目录（存放 Game_* 与 games_registry.json）",
            str(start),
        )
        if picked:
            self._set_path(Path(canonical_abs_path(picked)))

    def _set_path(self, path: Path) -> None:
        self._selected = path
        display = _display_path(path)
        self._path_edit.setText(display)
        self._path_edit.setToolTip(str(path))
        self._path_edit.setCursorPosition(0)
        self._refresh_plan()

    def _refresh_plan(self) -> None:
        if self._selected is None:
            self._scene_label.setText("—")
            self._preview.clear()
            self._error_label.setText("请先选择工作区目录。")
            self._import_md_cb.setChecked(False)
            self._import_md_cb.setEnabled(False)
            self._discover_cb.setChecked(False)
            self._discover_cb.setEnabled(False)
            self._render_md_cb.setChecked(False)
            self._ok_button.setEnabled(False)
            return

        plan = plan_workspace_setup(self._selected)
        self._plan = plan
        self._scene_label.setText(_scene_label(plan.scene))

        lines: list[str] = []
        for note in plan.notes:
            lines.append(f"• {note}")
        lines.append("")
        lines.append(f"总表：{'存在' if plan.registry_exists else '无'}"
                     f"{'（合法）' if plan.registry_valid else ''}"
                     f"，项目 {plan.registry_project_count} 个")
        lines.append(
            f"GAMES.md：{'存在' if plan.games_md_exists else '无'}"
            f"（约 {plan.games_md_row_count} 行）"
        )
        lines.append(
            f"Game_*：{len(plan.game_dir_paths)} 个"
            f"（未登记 {len(plan.undiscovered_paths)} 个）"
        )
        if plan.undiscovered_paths:
            preview_paths = list(plan.undiscovered_paths[:12])
            lines.append("将登记：")
            for rel in preview_paths:
                lines.append(f"  + {rel}")
            if len(plan.undiscovered_paths) > 12:
                lines.append(f"  … 共 {len(plan.undiscovered_paths)} 个")

        self._preview.setPlainText("\n".join(lines).strip())

        self._import_md_cb.setEnabled(plan.ok and plan.games_md_exists)
        self._import_md_cb.setChecked(bool(plan.suggest_import_md))
        self._discover_cb.setEnabled(plan.ok and bool(plan.undiscovered_paths or plan.game_dir_paths))
        self._discover_cb.setChecked(bool(plan.suggest_discover))
        self._render_md_cb.setEnabled(plan.ok)
        self._render_md_cb.setChecked(bool(plan.suggest_render_md))

        if plan.ok:
            self._error_label.setText("")
            if plan.scene == WorkspaceScene.EMPTY:
                self._ok_button.setText("创建工作区")
            else:
                self._ok_button.setText("接入工作区")
        else:
            self._error_label.setText(plan.error_message or "无法接入该目录。")
            self._ok_button.setText("接入工作区")

        self._sync_confirm_enabled()

    def _sync_confirm_enabled(self) -> None:
        plan = getattr(self, "_plan", None)
        self._ok_button.setEnabled(bool(plan is not None and plan.ok and self._selected is not None))

    def _on_accept(self) -> None:
        if self._selected is None:
            return
        plan = plan_workspace_setup(self._selected)
        if not plan.ok:
            self._error_label.setText(plan.error_message or "无法接入该目录。")
            self._ok_button.setEnabled(False)
            return

        create_directory = plan.scene == WorkspaceScene.MISSING_PATH
        action = apply_workspace_setup_action(
            plan,
            import_md=self._import_md_cb.isChecked() if self._import_md_cb.isEnabled() else False,
            discover=self._discover_cb.isChecked() if self._discover_cb.isEnabled() else False,
            render_md=self._render_md_cb.isChecked(),
            create_directory=create_directory,
            persist_workspace_root=False,
        )
        if not action.ok:
            self._error_label.setText(action.message)
            return

        workspace = Path(action.workspace_root) if action.workspace_root else self._selected
        self._result = WorkspaceSetupDialogResult(
            workspace=workspace,
            message=action.message,
            project_count=action.project_count,
            created_registry=action.created_registry,
        )
        self.accept()
