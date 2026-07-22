"""Dialog: create or attach a workspace, then optional Ren'Py SDK setup."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from games_registry import WorkspaceScene, plan_workspace_setup
from renpy_sdk_install import (
    default_sdk_target,
    format_size_mib,
    recommended_sdk,
    save_renpy_sdk_dir,
)
from translator_runtime import discover_renpy_sdk_candidates, is_renpy_sdk_dir

from .games_registry_actions import apply_workspace_setup_action
from .path_utils import canonical_abs_path
from .sdk_install_worker import SdkInstallWorker


@dataclass(frozen=True)
class WorkspaceSetupDialogResult:
    """Outcome of the workspace create/attach dialog (workspace + optional SDK).

    Workspace fields always describe a successful registry attach/init.

    SDK contract for callers (``games_registry_panel`` summary, host config):

    - **Not configured / skipped:** ``sdk_dir is None`` and ``sdk_message`` is
      empty or a skip notice such as ``已跳过 SDK 配置。``
    - **Attempted but failed/cancelled:** ``sdk_dir is None`` and
      ``sdk_message`` starts with ``SDK 未配置：`` (preserves the failure
      reason after the user finishes with「跳过」).
    - **Configured successfully:** ``sdk_dir`` is an absolute path that was
      written to ``prepare.renpy_sdk_dir`` (find / browse / download), and
      ``sdk_message`` describes that success.
    """

    workspace: Path
    message: str
    project_count: int
    created_registry: bool = False
    sdk_dir: Path | None = None
    sdk_message: str = ""


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
    """Pick a directory, preview setup plan, apply core, then optional SDK."""

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
        self.setMinimumWidth(580)
        self.setObjectName("workspace_setup_dialog")

        self._start_dir = Path(start_dir or Path.home()).expanduser()
        self._selected: Path | None = None
        self._result: WorkspaceSetupDialogResult | None = None
        self._workspace_result: WorkspaceSetupDialogResult | None = None
        self._sdk_worker: SdkInstallWorker | None = None
        self._found_sdk: Path | None = None
        # Last SDK attempt message (failure / cancel) for completion summary.
        self._sdk_status_message: str = ""

        root = QVBoxLayout(self)
        root.setSpacing(12)

        self._stack = QStackedWidget()
        self._stack.setObjectName("workspace_setup_stack")
        root.addWidget(self._stack, 1)

        self._page_workspace = QWidget()
        self._build_workspace_page(self._page_workspace)
        self._stack.addWidget(self._page_workspace)

        self._page_sdk = QWidget()
        self._build_sdk_page(self._page_sdk)
        self._stack.addWidget(self._page_sdk)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self._on_reject)
        self._ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setText("接入工作区")
        self._ok_button.setObjectName("workspace_setup_ok_btn")
        cancel_btn = self._buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_btn is not None:
            cancel_btn.setText("取消")
            cancel_btn.setObjectName("workspace_setup_cancel_btn")
        root.addWidget(self._buttons)

        if initial_path is not None:
            self._set_path(Path(initial_path))
        else:
            self._refresh_plan()

    def result_payload(self) -> WorkspaceSetupDialogResult | None:
        return self._result

    def _build_workspace_page(self, page: QWidget) -> None:
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        hint = QLabel(
            "选择或新建工作区根目录，预览后初始化/接入 games_registry.json。\n"
            "不会自动准备项目 work；SDK 可在下一步可选配置。"
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
        layout.addStretch(1)

    def _build_sdk_page(self, page: QWidget) -> None:
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        hint = QLabel(
            "工作区已就绪。可选择配置 Ren'Py SDK（可选，可稍后在设置中配置）。\n"
            "下载仅在你确认后联网，且只使用本工具维护的官方推荐版本。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("config_hint_label")
        layout.addWidget(hint)

        self._workspace_summary = QLabel("")
        self._workspace_summary.setObjectName("workspace_setup_workspace_summary")
        self._workspace_summary.setWordWrap(True)
        layout.addWidget(self._workspace_summary)

        self._sdk_group = QButtonGroup(self)
        self._sdk_skip = QRadioButton("暂时跳过（不配置 SDK）")
        self._sdk_skip.setObjectName("workspace_setup_sdk_skip")
        self._sdk_skip.setChecked(True)
        self._sdk_group.addButton(self._sdk_skip, 0)
        layout.addWidget(self._sdk_skip)

        self._sdk_found = QRadioButton("使用查找到的 SDK")
        self._sdk_found.setObjectName("workspace_setup_sdk_found")
        self._sdk_found.setEnabled(False)
        self._sdk_group.addButton(self._sdk_found, 1)
        layout.addWidget(self._sdk_found)

        found_row = QHBoxLayout()
        self._found_path_edit = QLineEdit()
        self._found_path_edit.setReadOnly(True)
        self._found_path_edit.setObjectName("workspace_setup_sdk_found_path")
        self._found_path_edit.setPlaceholderText("点击「查找」扫描工作区附近…")
        found_row.addWidget(self._found_path_edit, 1)
        find_btn = QPushButton("查找")
        find_btn.setObjectName("workspace_setup_sdk_find_btn")
        find_btn.clicked.connect(self._find_sdk)
        found_row.addWidget(find_btn)
        layout.addLayout(found_row)

        self._sdk_browse = QRadioButton("浏览并指定已有 SDK 目录")
        self._sdk_browse.setObjectName("workspace_setup_sdk_browse")
        self._sdk_group.addButton(self._sdk_browse, 2)
        layout.addWidget(self._sdk_browse)

        browse_row = QHBoxLayout()
        self._browse_path_edit = QLineEdit()
        self._browse_path_edit.setReadOnly(True)
        self._browse_path_edit.setObjectName("workspace_setup_sdk_browse_path")
        browse_row.addWidget(self._browse_path_edit, 1)
        browse_sdk_btn = QPushButton("浏览…")
        browse_sdk_btn.setObjectName("workspace_setup_sdk_browse_btn")
        browse_sdk_btn.clicked.connect(self._browse_sdk)
        browse_row.addWidget(browse_sdk_btn)
        layout.addLayout(browse_row)

        self._sdk_download = QRadioButton("下载推荐 SDK（官方来源，需确认）")
        self._sdk_download.setObjectName("workspace_setup_sdk_download")
        self._sdk_group.addButton(self._sdk_download, 3)
        layout.addWidget(self._sdk_download)

        spec = recommended_sdk()
        self._download_info = QLabel(
            f"版本 {spec.version} · {spec.source_label}\n"
            f"来源 {spec.url}\n"
            f"预计约 {format_size_mib(spec.size_bytes)} · SHA-256 {spec.sha256[:16]}…"
        )
        self._download_info.setObjectName("workspace_setup_sdk_download_info")
        self._download_info.setWordWrap(True)
        self._download_info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._download_info)

        target_row = QHBoxLayout()
        self._download_target_edit = QLineEdit()
        self._download_target_edit.setObjectName("workspace_setup_sdk_target")
        self._download_target_edit.setPlaceholderText("下载安装目标目录")
        target_row.addWidget(self._download_target_edit, 1)
        target_btn = QPushButton("更改…")
        target_btn.setObjectName("workspace_setup_sdk_target_btn")
        target_btn.clicked.connect(self._browse_download_target)
        target_row.addWidget(target_btn)
        layout.addLayout(target_row)

        self._sdk_progress = QProgressBar()
        self._sdk_progress.setObjectName("workspace_setup_sdk_progress")
        self._sdk_progress.setRange(0, 100)
        self._sdk_progress.setValue(0)
        self._sdk_progress.setVisible(False)
        layout.addWidget(self._sdk_progress)

        self._sdk_status = QLabel("")
        self._sdk_status.setObjectName("workspace_setup_sdk_status")
        self._sdk_status.setWordWrap(True)
        layout.addWidget(self._sdk_status)
        layout.addStretch(1)

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
        lines.append(
            f"总表：{'存在' if plan.registry_exists else '无'}"
            f"{'（合法）' if plan.registry_valid else ''}"
            f"，项目 {plan.registry_project_count} 个"
        )
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
        self._discover_cb.setEnabled(
            plan.ok and bool(plan.undiscovered_paths or plan.game_dir_paths)
        )
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
        if self._stack.currentIndex() != 0:
            return
        plan = getattr(self, "_plan", None)
        self._ok_button.setEnabled(
            bool(plan is not None and plan.ok and self._selected is not None)
        )

    def _show_sdk_page(self, workspace_result: WorkspaceSetupDialogResult) -> None:
        self._workspace_result = workspace_result
        self._workspace_summary.setText(
            f"工作区：{_display_path(workspace_result.workspace)}\n"
            f"{workspace_result.message}"
        )
        self._download_target_edit.setText(
            str(default_sdk_target(workspace_result.workspace))
        )
        self._sdk_skip.setChecked(True)
        self._set_sdk_status("")
        self._sdk_progress.setVisible(False)
        self._sdk_progress.setValue(0)
        self._ok_button.setText("完成")
        self._ok_button.setEnabled(True)
        self._stack.setCurrentIndex(1)
        # Do not scan until the user clicks「查找」(explicit only).

    def _set_sdk_status(self, text: str, *, error: bool = False) -> None:
        """Update SDK status text; clear red error style unless *error* is set."""
        self._sdk_status.setStyleSheet("color: #b91c1c;" if error else "")
        self._sdk_status.setText(text)

    def _find_sdk(self) -> None:
        workspace = (
            self._workspace_result.workspace
            if self._workspace_result is not None
            else self._selected
        )
        candidates = discover_renpy_sdk_candidates(
            game_root=None,
            tool_root=None,
            workspace_root=str(workspace) if workspace is not None else None,
            include_runtime_defaults=False,
        )
        if candidates:
            self._found_sdk = Path(candidates[0])
            self._found_path_edit.setText(_display_path(self._found_sdk))
            self._sdk_found.setEnabled(True)
            self._sdk_found.setChecked(True)
            self._set_sdk_status(f"找到 {len(candidates)} 个候选，已选最新。")
            self._sdk_status_message = ""
        else:
            self._found_sdk = None
            self._found_path_edit.clear()
            self._sdk_found.setEnabled(False)
            self._set_sdk_status("未在工作区附近找到 Ren'Py SDK。")

    def _browse_sdk(self) -> None:
        start = (
            self._workspace_result.workspace
            if self._workspace_result is not None
            else Path.home()
        )
        picked = QFileDialog.getExistingDirectory(
            self,
            "选择 Ren'Py SDK 目录（需包含 renpy.py）",
            str(start),
        )
        if not picked:
            return
        path = Path(canonical_abs_path(picked))
        self._browse_path_edit.setText(_display_path(path))
        if is_renpy_sdk_dir(str(path)):
            self._sdk_browse.setChecked(True)
            self._set_sdk_status("已选择有效 SDK 目录。")
        else:
            self._set_sdk_status(
                "所选目录不是有效 SDK（缺少 renpy.py）。",
                error=True,
            )

    def _browse_download_target(self) -> None:
        start = self._download_target_edit.text().strip() or str(Path.home())
        parent = Path(start).parent if Path(start).name else Path(start)
        picked = QFileDialog.getExistingDirectory(
            self,
            "选择 SDK 安装的父目录（将创建 renpy-*-sdk 子目录时可手改路径）",
            str(parent if parent.is_dir() else Path.home()),
        )
        if picked:
            # Keep full target path editable; default to recommended folder name under pick.
            target = Path(picked) / recommended_sdk().folder_name
            self._download_target_edit.setText(str(target))
            self._sdk_download.setChecked(True)

    def _on_accept(self) -> None:
        if self._stack.currentIndex() == 0:
            self._apply_workspace()
            return
        self._finish_sdk()

    def _sdk_worker_active(self) -> bool:
        return self._sdk_worker is not None and self._sdk_worker.isRunning()

    def _stop_sdk_worker(self, *, wait_ms: int = 5000) -> bool:
        """Cancel and wait for the SDK worker.

        Returns True only when no worker is attached or it has fully stopped.
        On timeout the reference is **kept** so callers continue to block SDK
        actions and must not destroy the dialog while the thread runs.
        """
        worker = self._sdk_worker
        if worker is None:
            return True
        if worker.isRunning():
            worker.request_cancel()
            worker.requestInterruption()
            worker.wait(wait_ms)
        if worker.isRunning():
            return False
        self._sdk_worker = None
        return True

    def _on_reject(self) -> None:
        if self._sdk_worker_active():
            self._ok_button.setEnabled(False)
            self._set_sdk_status("正在取消 SDK 下载…")
            if not self._stop_sdk_worker():
                self._set_sdk_status(
                    "SDK 任务仍在结束，请稍候再关闭或重试。",
                    error=True,
                )
                # Keep OK disabled until completed signal clears the worker.
                return
            self._sdk_status_message = "SDK 安装已取消。"
            self._set_sdk_status(self._sdk_status_message)
            self._sdk_progress.setVisible(False)
            self._ok_button.setEnabled(True)
            # Stay on the SDK page so the user can skip or retry.
            return
        # If workspace already applied, still accept with skip SDK so host can persist root.
        if self._workspace_result is not None and self._stack.currentIndex() == 1:
            self._result = WorkspaceSetupDialogResult(
                workspace=self._workspace_result.workspace,
                message=self._workspace_result.message,
                project_count=self._workspace_result.project_count,
                created_registry=self._workspace_result.created_registry,
                sdk_message=self._sdk_summary_for_skip(),
            )
            self.accept()
            return
        self.reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._stop_sdk_worker(wait_ms=5000):
            event.ignore()
            self._set_sdk_status(
                "SDK 任务仍在结束，请稍候再关闭。",
                error=True,
            )
            return
        super().closeEvent(event)

    def _sdk_summary_for_skip(self) -> str:
        if self._sdk_status_message:
            return f"SDK 未配置：{self._sdk_status_message}"
        return "已跳过 SDK 配置。"

    def _apply_workspace(self) -> None:
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
        workspace_result = WorkspaceSetupDialogResult(
            workspace=workspace,
            message=action.message,
            project_count=action.project_count,
            created_registry=action.created_registry,
        )
        self._show_sdk_page(workspace_result)

    def _finish_sdk(self) -> None:
        if self._workspace_result is None:
            return
        if self._sdk_worker_active():
            self._set_sdk_status("SDK 任务进行中，请等待完成或取消后再操作。")
            return

        base = self._workspace_result
        sdk_dir: Path | None = None
        sdk_message = self._sdk_summary_for_skip()

        if self._sdk_found.isChecked() and self._found_sdk is not None:
            if not is_renpy_sdk_dir(str(self._found_sdk)):
                self._set_sdk_status(
                    "查找到的路径已失效，请重新查找或改选其它选项。",
                    error=True,
                )
                return
            try:
                save_renpy_sdk_dir(self._found_sdk)
            except Exception as exc:
                self._set_sdk_status(f"写入 SDK 配置失败：{exc}", error=True)
                return
            sdk_dir = self._found_sdk
            sdk_message = f"已配置查找到的 SDK：{self._found_sdk}"

        elif self._sdk_browse.isChecked():
            text = self._browse_path_edit.text().strip()
            if not text or not is_renpy_sdk_dir(text):
                self._set_sdk_status(
                    "请浏览选择包含 renpy.py 的有效 SDK 目录。",
                    error=True,
                )
                return
            path = Path(text)
            try:
                save_renpy_sdk_dir(path)
            except Exception as exc:
                self._set_sdk_status(f"写入 SDK 配置失败：{exc}", error=True)
                return
            sdk_dir = path
            sdk_message = f"已配置指定 SDK：{path}"

        elif self._sdk_download.isChecked():
            target_text = self._download_target_edit.text().strip()
            if not target_text:
                self._set_sdk_status("请指定下载目标目录。", error=True)
                return
            target = Path(target_text)
            spec = recommended_sdk()
            self._set_sdk_status(
                f"确认下载：{spec.version}（约 {format_size_mib(spec.size_bytes)}）→ {target}"
            )
            self._start_download(target)
            return

        self._result = WorkspaceSetupDialogResult(
            workspace=base.workspace,
            message=base.message,
            project_count=base.project_count,
            created_registry=base.created_registry,
            sdk_dir=sdk_dir,
            sdk_message=sdk_message,
        )
        self.accept()

    def _start_download(self, target: Path) -> None:
        if self._sdk_worker_active():
            self._set_sdk_status("已有 SDK 任务在运行，请稍候。", error=True)
            return
        self._ok_button.setEnabled(False)
        self._sdk_progress.setVisible(True)
        self._sdk_progress.setRange(0, 100)
        self._sdk_progress.setValue(0)
        self._set_sdk_status("正在下载并安装推荐 SDK…")

        workspace = (
            self._workspace_result.workspace
            if self._workspace_result is not None
            else None
        )
        worker = SdkInstallWorker(
            target,
            workspace_root=workspace,
            persist_config=True,
            parent=self,
        )
        self._sdk_worker = worker
        worker.progress.connect(self._on_sdk_progress)
        worker.completed.connect(self._on_sdk_completed)
        worker.start()

    def _on_sdk_progress(self, phase: str, current: int, total: int) -> None:
        if phase == "download" and total > 0:
            self._sdk_progress.setRange(0, 100)
            self._sdk_progress.setValue(min(100, int(100 * current / total)))
            self._sdk_status.setText(f"下载中… {self._sdk_progress.value()}%")
        elif phase == "extract":
            self._sdk_progress.setRange(0, max(1, total))
            self._sdk_progress.setValue(current)
            self._sdk_status.setText(f"解压中… {current}/{total}")

    def _on_sdk_completed(self, result: object) -> None:
        # Thread has finished — safe to drop the reference now.
        self._sdk_worker = None
        self._ok_button.setEnabled(True)
        from renpy_sdk_install import SdkInstallResult

        if not isinstance(result, SdkInstallResult):
            self._set_sdk_status("SDK 安装返回了未知结果。", error=True)
            return
        if result.cancelled:
            self._sdk_status_message = result.message
            self._set_sdk_status(result.message)
            self._sdk_progress.setVisible(False)
            return
        if not result.ok:
            # Keep workspace success; preserve failure for completion summary.
            self._sdk_status_message = result.message
            self._set_sdk_status(
                f"{result.message}\n工作区已保留；可改选「暂时跳过」完成（摘要会保留失败原因），或重试下载。",
                error=True,
            )
            self._sdk_progress.setVisible(False)
            return

        base = self._workspace_result
        if base is None:
            return
        self._sdk_status_message = ""
        self._result = WorkspaceSetupDialogResult(
            workspace=base.workspace,
            message=base.message,
            project_count=base.project_count,
            created_registry=base.created_registry,
            sdk_dir=result.sdk_dir,
            sdk_message=result.message,
        )
        self.accept()
