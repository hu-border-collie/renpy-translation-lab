"""Main GUI application for the optional workbench.

This is the first version shell (per #42):
- Pure PySide6
- Delegates everything to the existing CLI via QProcess
- Starts with project selection + doctor execution + live logs
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
    QComboBox,
    QGridLayout,
    QFrame,
    QFormLayout,
)

from .cli_runner import CliRunner
from .doctor_report import (
    DoctorSummary,
    idle_summary,
    running_summary,
    stale_summary,
    summarize_doctor_output,
)
from .project_state import ProjectState


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ren'Py Translation Lab - 图形工作台（实验版）")
        self.resize(1100, 720)

        self.state = ProjectState()
        self.runner = CliRunner()
        self._loading_config_to_ui = False
        self._updating_batch_thinking_combo = False
        self._batch_thinking_config_has_key = False
        self._batch_thinking_user_changed = False
        self._active_command = ""
        self._doctor_output_lines: list[str] = []

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header = QLabel("实验性图形工作台 - 复用现有命令行流程")
        header.setObjectName("header_label")
        layout.addWidget(header)

        # Project row
        project_frame = QFrame()
        project_frame.setObjectName("project_frame")
        proj_layout = QHBoxLayout(project_frame)
        proj_layout.setContentsMargins(12, 10, 12, 10)
        proj_layout.setSpacing(10)

        proj_layout.addWidget(QLabel("当前游戏 work 目录："))

        self.project_path_edit = QLineEdit("尚未选择项目")
        self.project_path_edit.setReadOnly(True)
        self.project_path_edit.setObjectName("project_path_edit")
        proj_layout.addWidget(self.project_path_edit, 1)

        self.select_btn = QPushButton("选择游戏目录...")
        self.select_btn.clicked.connect(self._on_select_project)
        proj_layout.addWidget(self.select_btn)

        layout.addWidget(project_frame)

        # Configuration Row (Sync + Batch side-by-side)
        config_row = QHBoxLayout()
        config_row.setSpacing(16)

        # Sync Box
        sync_box = QGroupBox("同步翻译配置 (Sync API)")
        sync_layout = QFormLayout(sync_box)
        sync_layout.setSpacing(8)
        sync_layout.setContentsMargins(12, 16, 12, 12)

        self.sync_model_combo = QComboBox()
        self.sync_model_combo.addItems([
            "gemini-3.5-flash",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ])
        sync_layout.addRow("翻译模型：", self.sync_model_combo)

        self.sync_embedding_combo = QComboBox()
        self.sync_embedding_combo.addItems([
            "gemini-embedding-2",
            "gemini-embedding-001",
        ])
        sync_layout.addRow("RAG 向量模型：", self.sync_embedding_combo)

        config_row.addWidget(sync_box, 1)

        # Batch Box
        batch_box = QGroupBox("批量离线配置 (Batch API)")
        batch_layout = QFormLayout(batch_box)
        batch_layout.setSpacing(8)
        batch_layout.setContentsMargins(12, 16, 12, 12)

        self.batch_model_combo = QComboBox()
        self.batch_model_combo.addItems([
            "gemini-3.5-flash",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ])
        batch_layout.addRow("翻译模型：", self.batch_model_combo)

        self.batch_embedding_combo = QComboBox()
        self.batch_embedding_combo.addItems([
            "gemini-embedding-2",
            "gemini-embedding-001",
        ])
        batch_layout.addRow("RAG 向量模型：", self.batch_embedding_combo)

        self.batch_thinking_combo = QComboBox()
        self.batch_thinking_combo.addItem("（不启用）", "")
        self.batch_thinking_combo.addItem("最小 (minimal)", "minimal")
        self.batch_thinking_combo.addItem("低 (low)", "low")
        self.batch_thinking_combo.addItem("中 (medium)", "medium")
        self.batch_thinking_combo.addItem("高 (high)", "high")
        batch_layout.addRow("Batch 思考程度：", self.batch_thinking_combo)

        config_row.addWidget(batch_box, 1)

        layout.addLayout(config_row)

        # Save config row
        save_layout = QHBoxLayout()
        save_layout.addStretch()
        self.save_config_btn = QPushButton("保存参数配置")
        self.save_config_btn.setObjectName("save_config_btn")
        self.save_config_btn.clicked.connect(self._on_save_config)
        save_layout.addWidget(self.save_config_btn)
        layout.addLayout(save_layout)

        # Connect model change signal to dynamically toggle thinking level
        self.batch_model_combo.currentTextChanged.connect(self._on_batch_model_changed)
        self.batch_thinking_combo.currentIndexChanged.connect(self._on_batch_thinking_changed)

        # Actions row
        action_layout = QHBoxLayout()
        self.doctor_btn = QPushButton("检查项目（doctor）")
        self.doctor_btn.setObjectName("doctor_btn")
        self.doctor_btn.clicked.connect(self._on_run_doctor)
        action_layout.addWidget(self.doctor_btn)

        self.api_btn = QPushButton("管理 API Key")
        self.api_btn.setObjectName("api_btn")
        self.api_btn.clicked.connect(self._on_manage_api_keys)
        action_layout.addWidget(self.api_btn)

        self.kill_btn = QPushButton("停止当前任务")
        self.kill_btn.setObjectName("kill_btn")
        self.kill_btn.clicked.connect(self._on_kill)
        self.kill_btn.setEnabled(False)
        action_layout.addWidget(self.kill_btn)

        action_layout.addStretch()
        layout.addLayout(action_layout)

        # Doctor summary
        doctor_summary_box = QGroupBox("项目检查摘要")
        doctor_summary_layout = QVBoxLayout(doctor_summary_box)
        doctor_summary_layout.setSpacing(6)
        doctor_summary_layout.setContentsMargins(12, 16, 12, 12)

        self.doctor_status_label = QLabel()
        self.doctor_status_label.setObjectName("doctor_status_label")
        doctor_summary_layout.addWidget(self.doctor_status_label)

        self.doctor_message_label = QLabel()
        self.doctor_message_label.setWordWrap(True)
        doctor_summary_layout.addWidget(self.doctor_message_label)

        self.doctor_facts_label = QLabel()
        self.doctor_facts_label.setWordWrap(True)
        self.doctor_facts_label.setObjectName("doctor_facts_label")
        doctor_summary_layout.addWidget(self.doctor_facts_label)

        self.doctor_findings_view = QTextEdit()
        self.doctor_findings_view.setReadOnly(True)
        self.doctor_findings_view.setMaximumHeight(92)
        self.doctor_findings_view.setObjectName("doctor_findings_view")
        doctor_summary_layout.addWidget(self.doctor_findings_view)

        layout.addWidget(doctor_summary_box)

        # Output
        out_label = QLabel("诊断输出（来自命令行）")
        layout.addWidget(out_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.log_view.setObjectName("log_view")
        layout.addWidget(self.log_view, 1)

        # Connect runner
        self.runner.line_ready.connect(self._on_cli_line_ready)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self._on_runner_error)

        self._refresh_project_label()
        self._load_config_to_ui()
        self._set_doctor_summary(idle_summary())

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
            self._load_config_to_ui()
            self._active_command = ""
            self._doctor_output_lines = []
            self._set_doctor_summary(stale_summary())
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
            self.project_path_edit.setText(str(root))
        else:
            self.project_path_edit.setText("（尚未选择项目）")

    def _on_run_doctor(self):
        if not self.state.get_game_root():
            QMessageBox.information(
                self, "请先选择项目",
                "请先选择游戏的 work 目录。\n"
                "项目检查（doctor）会读取 translator_config.json 中的 game_root。"
            )
            return

        self.log_view.clear()
        self._active_command = "doctor"
        self._doctor_output_lines = []
        self._set_doctor_summary(running_summary())
        self._append_log("=== 正在运行：gemini_translate_batch.py doctor ===\n")
        self.doctor_btn.setEnabled(False)
        self.kill_btn.setEnabled(True)

        script = self.state.get_batch_script_path()
        # Run with no extra args — it will pick up translator_config.json
        self.runner.run(script, ["doctor"])

    def _on_kill(self):
        self.runner.kill()

    # --- Runner callbacks ---

    def _on_cli_line_ready(self, text: str):
        if self._active_command == "doctor":
            self._doctor_output_lines.append(text)
        self._append_log(text)

    def _append_log(self, text: str):
        self.log_view.append(text.rstrip("\n"))
        # scroll to bottom
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_doctor_summary(self, summary: DoctorSummary):
        self.doctor_status_label.setText(summary.heading)
        self.doctor_status_label.setProperty("status", summary.status)
        self.doctor_status_label.style().unpolish(self.doctor_status_label)
        self.doctor_status_label.style().polish(self.doctor_status_label)
        self.doctor_message_label.setText(summary.message)
        self.doctor_facts_label.setText("\n".join(summary.facts))
        if summary.findings:
            self.doctor_findings_view.setPlainText(
                "\n".join(f"- {finding}" for finding in summary.findings)
            )
        elif summary.status in {"idle", "running"}:
            self.doctor_findings_view.setPlainText("等待项目检查结果。")
        elif summary.status == "stale":
            self.doctor_findings_view.setPlainText("请重新运行项目检查。")
        else:
            self.doctor_findings_view.setPlainText("未发现需要处理的事项。")

    def _on_runner_error(self, message: str):
        self._append_log(message)
        if self._active_command == "doctor":
            self._doctor_output_lines.append(message)
        self.doctor_btn.setEnabled(True)
        self.kill_btn.setEnabled(False)
        self.statusBar().showMessage("项目检查失败，请查看诊断输出。", 6000)

    def _on_finished(self, exit_code: int):
        self.doctor_btn.setEnabled(True)
        self.kill_btn.setEnabled(False)
        self._append_log(f"\n[进程已结束，退出码：{exit_code}]")
        if self._active_command == "doctor":
            api_key_count, api_key_source = self.state.get_api_key_status()
            summary = summarize_doctor_output(
                "\n".join(self._doctor_output_lines),
                exit_code,
                api_key_count=api_key_count,
                api_key_source=api_key_source,
            )
            self._set_doctor_summary(summary)
            self._active_command = ""
        if exit_code == 0:
            self.statusBar().showMessage("项目检查完成。", 6000)
        else:
            self.statusBar().showMessage(f"项目检查失败（退出码：{exit_code}）", 6000)

    # --- Config loading/saving helpers ---

    def _config_section(self, config: dict[str, Any], key: str) -> dict[str, Any]:
        section = config.get(key)
        return section if isinstance(section, dict) else {}

    def _ensure_config_section(self, config: dict[str, Any], key: str) -> dict[str, Any]:
        section = config.get(key)
        if not isinstance(section, dict):
            section = {}
            config[key] = section
        return section

    def _config_string(self, value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _supports_batch_thinking(self, model_name: Any) -> bool:
        return self._config_string(model_name).startswith("gemini-3")

    def _sync_models_for_save(
        self,
        existing_models: Any,
        selected_model: str,
    ) -> list[str] | None:
        existing: list[str] = []
        if isinstance(existing_models, list):
            for model in existing_models:
                cleaned = self._config_string(model)
                if cleaned and cleaned not in existing:
                    existing.append(cleaned)
        else:
            cleaned = self._config_string(existing_models)
            if cleaned:
                existing.append(cleaned)

        if not selected_model:
            return existing or None

        return [
            selected_model,
            *[model for model in existing if model != selected_model],
        ]

    def _batch_thinking_value_for_load(
        self,
        batch_config: dict[str, Any],
        batch_model: Any,
    ) -> str:
        if "thinking_level" in batch_config:
            return self._config_string(batch_config.get("thinking_level", ""))
        return "minimal" if self._supports_batch_thinking(batch_model) else ""

    def _batch_thinking_value_for_model_change(
        self,
        batch_model: Any,
        current_thinking_level: Any,
        config_has_key: bool,
        user_changed: bool,
    ) -> str | None:
        if (
            self._supports_batch_thinking(batch_model)
            and not self._config_string(current_thinking_level)
            and not config_has_key
            and not user_changed
        ):
            return "minimal"
        return None

    def _should_save_batch_thinking_level(
        self,
        batch_config: dict[str, Any],
        batch_model: str,
        thinking_level: str,
        user_changed: bool,
    ) -> bool:
        return (
            bool(thinking_level)
            or (self._supports_batch_thinking(batch_model) and user_changed)
            or "thinking_level" in batch_config
        )

    def _set_combo_value(self, combo: QComboBox, value: Any):
        value = self._config_string(value)
        if not value:
            combo.setCurrentIndex(-1)
            return
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.addItem(value)
            combo.setCurrentIndex(combo.count() - 1)

    def _set_batch_thinking_value(self, value: str):
        idx = self.batch_thinking_combo.findData(value)
        self._updating_batch_thinking_combo = True
        try:
            if idx >= 0:
                self.batch_thinking_combo.setCurrentIndex(idx)
            elif value:
                self.batch_thinking_combo.addItem(f"{value} (自定义)", value)
                self.batch_thinking_combo.setCurrentIndex(self.batch_thinking_combo.count() - 1)
            else:
                self.batch_thinking_combo.setCurrentIndex(0)
        finally:
            self._updating_batch_thinking_combo = False

    def _load_config_to_ui(self):
        self._loading_config_to_ui = True
        try:
            config = self.state.load_translator_config()
            sync_config = self._config_section(config, "sync")
            batch_config = self._config_section(config, "batch")
            sync_rag_config = self._config_section(sync_config, "rag")
            batch_rag_config = self._config_section(batch_config, "rag")
            self._batch_thinking_config_has_key = "thinking_level" in batch_config

            # Populate sync model
            sync_models = sync_config.get("models")
            sync_val = ""
            if isinstance(sync_models, list):
                for model in sync_models:
                    sync_val = self._config_string(model)
                    if sync_val:
                        break
            elif isinstance(sync_models, str):
                sync_val = self._config_string(sync_models)
            if not sync_val:
                sync_val = sync_config.get("model", "")
            self._set_combo_value(self.sync_model_combo, sync_val)

            # Populate batch model
            batch_val = batch_config.get("model", "")
            self._set_combo_value(self.batch_model_combo, batch_val)

            # Populate sync embedding
            sync_emb_val = sync_rag_config.get("embedding_model", "")
            self._set_combo_value(self.sync_embedding_combo, sync_emb_val)

            # Populate batch embedding
            batch_emb_val = batch_rag_config.get("embedding_model", "")
            self._set_combo_value(self.batch_embedding_combo, batch_emb_val)

            self._on_batch_model_changed(batch_val)
            # Populate thinking level. Missing config keeps the CLI's supported-model
            # default visible; choosing "not enabled" then saves an explicit empty value.
            thinking_val = self._batch_thinking_value_for_load(batch_config, batch_val)
            self._set_batch_thinking_value(thinking_val)
        finally:
            self._batch_thinking_user_changed = False
            self._loading_config_to_ui = False

    def _on_batch_model_changed(self, text: str):
        is_thinking_supported = self._supports_batch_thinking(text)
        self.batch_thinking_combo.setEnabled(is_thinking_supported)
        if not is_thinking_supported:
            self._set_batch_thinking_value("")
            return

        default_value = self._batch_thinking_value_for_model_change(
            text,
            self.batch_thinking_combo.currentData(),
            self._batch_thinking_config_has_key,
            self._batch_thinking_user_changed,
        )
        if default_value is not None and not self._loading_config_to_ui:
            self._set_batch_thinking_value(default_value)

    def _on_batch_thinking_changed(self, _index: int):
        if not self._loading_config_to_ui and not self._updating_batch_thinking_combo:
            self._batch_thinking_user_changed = True

    def _on_save_config(self):
        if not self.state.get_game_root():
            QMessageBox.information(self, "未选择项目", "请先选择游戏的 work 目录。")
            return

        try:
            config = self.state.load_translator_config()
            sync_config = self._ensure_config_section(config, "sync")
            batch_config = self._ensure_config_section(config, "batch")
            sync_rag_config = self._ensure_config_section(sync_config, "rag")
            batch_rag_config = self._ensure_config_section(batch_config, "rag")

            sync_model = self.sync_model_combo.currentText().strip()
            sync_config["model"] = sync_model
            if "models" in sync_config:
                sync_models = self._sync_models_for_save(sync_config.get("models"), sync_model)
                if sync_models:
                    sync_config["models"] = sync_models
                else:
                    sync_config.pop("models", None)
            batch_model = self.batch_model_combo.currentText().strip()
            batch_config["model"] = batch_model
            sync_rag_config["embedding_model"] = self.sync_embedding_combo.currentText().strip()
            batch_rag_config["embedding_model"] = self.batch_embedding_combo.currentText().strip()
            thinking_val = self.batch_thinking_combo.currentData()
            thinking_level = thinking_val if isinstance(thinking_val, str) else ""
            if self._should_save_batch_thinking_level(
                batch_config,
                batch_model,
                thinking_level,
                self._batch_thinking_user_changed,
            ):
                batch_config["thinking_level"] = thinking_level

            self.state.save_translator_config(config)
            self._append_log("配置已成功保存至 translator_config.json。")
            self.statusBar().showMessage("配置已成功保存", 3000)
        except Exception as exc:
            QMessageBox.warning(self, "保存配置失败", str(exc))
            self._append_log(f"保存配置失败：{exc}")


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
