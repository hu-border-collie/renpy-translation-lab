"""Main GUI application for the optional workbench.

This is the first version shell (per #42):
- Pure PySide6
- Delegates everything to the existing CLI via QProcess
- Starts with project selection + doctor execution + live logs
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QInputDialog,
    QLineEdit,
)

from .cli_runner import CliRunner
from .project_state import ProjectState


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ren'Py Translation Lab - 图形工作台（实验版）")
        self.resize(1100, 720)

        self.state = ProjectState()
        self.runner = CliRunner()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        header = QLabel("实验性图形工作台 - 复用现有命令行流程")
        header.setStyleSheet("color: #1f2937; font-size: 15px; font-weight: 600;")
        layout.addWidget(header)

        # Project row
        proj_box = QGroupBox("当前项目")
        proj_layout = QHBoxLayout(proj_box)

        self.project_label = QLabel("尚未选择项目")
        self.project_label.setStyleSheet("color: #334155;")
        proj_layout.addWidget(self.project_label, 1)

        self.select_btn = QPushButton("选择游戏 work 目录...")
        self.select_btn.clicked.connect(self._on_select_project)
        proj_layout.addWidget(self.select_btn)

        layout.addWidget(proj_box)

        # Actions row
        action_layout = QHBoxLayout()
        self.doctor_btn = QPushButton("检查项目（doctor）")
        self.doctor_btn.clicked.connect(self._on_run_doctor)
        action_layout.addWidget(self.doctor_btn)

        self.api_btn = QPushButton("管理 API Key（基础）")
        self.api_btn.clicked.connect(self._on_manage_api_keys)
        action_layout.addWidget(self.api_btn)

        self.kill_btn = QPushButton("停止当前任务")
        self.kill_btn.clicked.connect(self._on_kill)
        self.kill_btn.setEnabled(False)
        action_layout.addWidget(self.kill_btn)

        action_layout.addStretch()
        layout.addLayout(action_layout)

        # Output
        out_label = QLabel("诊断输出（来自命令行）")
        layout.addWidget(out_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.log_view.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; "
            "font-size: 12px; background: #1e1e1e; color: #ddd;"
        )
        layout.addWidget(self.log_view, 1)

        # Connect runner
        self.runner.line_ready.connect(self._append_log)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._append_log)

        self._refresh_project_label()

        # Status
        self.statusBar().showMessage(
            "图形界面是可选组件；核心命令行不受影响。"
        )

    # --- UI actions ---

    def _on_select_project(self):
        start_dir = str(self.state.get_game_root() or Path.home())
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择游戏的 work 目录（通常包含 game/tl/schinese）",
            start_dir,
        )
        if directory:
            try:
                self.state.set_game_root(directory)
            except ValueError as exc:
                QMessageBox.warning(self, "无法更新配置", str(exc))
                self._append_log(f"更新 translator_config.json 失败：{exc}")
                return
            self._refresh_project_label()
            self._append_log(f"项目目录已设置为：{directory}")

    def _masked_key(self, key: str) -> str:
        stripped = key.strip()
        if not stripped:
            return "(none)"
        suffix = stripped[-4:] if len(stripped) > 4 else "****"
        return f"********{suffix}"

    def _on_manage_api_keys(self):
        keys = self.state.load_api_keys()
        current = keys[0] if keys else ""

        text, ok = QInputDialog.getText(
            self,
            "API Key",
            "当前第一个 Key："
            f"{self._masked_key(current)}\n"
            f"已配置 Key 数量：{len(keys)}\n\n"
            "输入新的 Key 会替换第一个 Key。\n"
            "留空则保持现有 Key 不变。",
            QLineEdit.EchoMode.Password,
            "",
        )
        if ok:
            replacement = text.strip()
            if not replacement:
                self._append_log("API Key 未修改。")
                return

            new_keys = list(keys)
            if new_keys:
                new_keys[0] = replacement
            else:
                new_keys = [replacement]
            try:
                self.state.save_api_keys(new_keys)
            except ValueError as exc:
                QMessageBox.warning(self, "无法更新 API Key", str(exc))
                self._append_log(f"更新 api_keys.json 失败：{exc}")
                return
            self._append_log(
                f"API Key 已更新（当前数量：{len(new_keys)}）。"
                "其他已配置 Key 已保留。"
            )

    def _refresh_project_label(self):
        root = self.state.get_game_root()
        if root:
            self.project_label.setText(str(root))
        else:
            self.project_label.setText("（尚未选择项目）")

    def _on_run_doctor(self):
        if not self.state.get_game_root():
            QMessageBox.information(
                self, "请先选择项目",
                "请先选择游戏的 work 目录。\n"
                "项目检查（doctor）会读取 translator_config.json 中的 game_root。"
            )
            return

        self.log_view.clear()
        self._append_log("=== 正在运行：gemini_translate_batch.py doctor ===\n")
        self.doctor_btn.setEnabled(False)
        self.kill_btn.setEnabled(True)

        script = self.state.get_batch_script_path()
        # Run with no extra args — it will pick up translator_config.json
        self.runner.run(script, ["doctor"])

    def _on_kill(self):
        self.runner.kill()

    # --- Runner callbacks ---

    def _append_log(self, text: str):
        self.log_view.append(text.rstrip("\n"))
        # scroll to bottom
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_finished(self, exit_code: int):
        self.doctor_btn.setEnabled(True)
        self.kill_btn.setEnabled(False)
        self._append_log(f"\n[进程已结束，退出码：{exit_code}]")
        self.statusBar().showMessage(f"项目检查完成（退出码：{exit_code}）", 6000)


def run_app(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    app = QApplication(argv)
    qss_path = Path(__file__).resolve().parent / "resources" / "app.qss"
    if qss_path.exists():
        try:
            app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"警告：无法加载 GUI 样式表：{exc}")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
