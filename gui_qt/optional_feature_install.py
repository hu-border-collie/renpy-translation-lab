"""Reusable optional-feature install controller for the GUI.

Installs into the active interpreter via background QProcess. Concurrent
optional-feature installs are serialized; unrelated translation workflows are
not blocked by this controller.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from optional_feature import (
    FeatureInstallState,
    FeatureStatus,
    OptionalFeatureSpec,
    hash_checked_install_command,
    probe_feature,
    relation_analyzer_feature,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class OptionalFeatureInstallController(QObject):
    """Background pip install for one optional feature at a time (process-wide)."""

    output_received = Signal(str)
    state_changed = Signal(str, object)  # feature_id, FeatureStatus
    finished = Signal(str, bool, str)  # feature_id, success, message

    _active_feature_id: str | None = None
    _active_controller: "OptionalFeatureInstallController | None" = None
    # Set by non-controller install paths (e.g. LiteLLM page) sharing the mutex.
    _external_install_active: bool = False

    def __init__(
        self,
        feature: OptionalFeatureSpec,
        *,
        parent: QObject | None = None,
        repo_root: Path = REPO_ROOT,
        python_executable: str | None = None,
        prefer_hash_lock: bool = True,
    ) -> None:
        super().__init__(parent)
        self.feature = feature
        self.repo_root = Path(repo_root)
        self.python_executable = python_executable or sys.executable
        self.prefer_hash_lock = prefer_hash_lock
        self._process: QProcess | None = None
        self._active = False
        self._last_failed = False
        self._is_update = False

    @classmethod
    def any_install_running(cls) -> bool:
        if cls._external_install_active:
            return True
        controller = cls._active_controller
        if controller is None:
            return False
        return controller.is_running()

    @classmethod
    def active_feature_id(cls) -> str | None:
        if not cls.any_install_running():
            return None
        return cls._active_feature_id

    def is_running(self) -> bool:
        if self._active:
            return True
        process = self._process
        return (
            process is not None
            and process.state() != QProcess.ProcessState.NotRunning
        )

    def current_status(self) -> FeatureStatus:
        return probe_feature(
            self.feature,
            installing=self.is_running(),
            last_failed=self._last_failed and not self.is_running(),
        )

    def emit_status(self) -> FeatureStatus:
        status = self.current_status()
        self.state_changed.emit(self.feature.feature_id, status)
        return status

    def start_install(self, *, upgrade: bool | None = None) -> tuple[bool, str]:
        """Start a background install. Returns (started, message)."""
        if self.is_running():
            return False, f"{self.feature.display_name} 已在安装中。"
        if self.any_install_running():
            other = self.active_feature_id() or "另一可选功能"
            return False, f"已有可选功能正在安装（{other}），请完成后再试。"

        pip_args, error = self._build_pip_args(upgrade=upgrade)
        if error:
            return False, error

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUTF8", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)

        self._process = process
        self._active = True
        self._last_failed = False
        status_before = probe_feature(self.feature)
        self._is_update = status_before.state in {
            FeatureInstallState.INSTALLED,
            FeatureInstallState.UPDATE_AVAILABLE,
            FeatureInstallState.PARTIALLY_INSTALLED,
        }
        OptionalFeatureInstallController._active_feature_id = self.feature.feature_id
        OptionalFeatureInstallController._active_controller = self

        action = "更新" if self._is_update else "安装"
        header = (
            f"=== 正在{action} {self.feature.display_name} ===\n"
            f"{self.python_executable} {' '.join(pip_args)}\n"
        )
        self.output_received.emit(header)
        self.emit_status()
        process.start(self.python_executable, pip_args)
        return True, header

    def _build_pip_args(
        self,
        *,
        upgrade: bool | None,
    ) -> tuple[list[str] | None, str]:
        lock_path = self.feature.lock_path(self.repo_root)
        requirements_path = self.feature.requirements_path(self.repo_root)
        use_lock = self.prefer_hash_lock and lock_path.is_file()
        target = lock_path if use_lock else requirements_path
        if not target.is_file():
            return None, f"找不到依赖文件：{target}"

        pip_args = ["-m", "pip", "install"]
        if upgrade is True or (upgrade is None and self._should_upgrade()):
            pip_args.append("--upgrade")
        if use_lock:
            pip_args.extend(["--require-hashes", "-r", str(target)])
        else:
            pip_args.extend(["-r", str(target)])
        return pip_args, ""

    def _should_upgrade(self) -> bool:
        status = probe_feature(self.feature)
        return status.state in {
            FeatureInstallState.INSTALLED,
            FeatureInstallState.UPDATE_AVAILABLE,
            FeatureInstallState.PARTIALLY_INSTALLED,
        }

    def _on_output(self) -> None:
        process = self._process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.output_received.emit(text)

    def _on_finished(self, exit_code: int, _exit_status: object) -> None:
        process = self._process
        if process is not None:
            self._on_output()
        self._process = None
        self._active = False
        if OptionalFeatureInstallController._active_controller is self:
            OptionalFeatureInstallController._active_controller = None
            OptionalFeatureInstallController._active_feature_id = None

        importlib.invalidate_caches()
        action = "更新" if self._is_update else "安装"
        status = probe_feature(self.feature)
        succeeded = exit_code == 0 and status.state in {
            FeatureInstallState.INSTALLED,
            FeatureInstallState.UPDATE_AVAILABLE,
        }
        self._last_failed = not succeeded
        if succeeded:
            message = f"{self.feature.display_name} {action}完成。"
            self.output_received.emit(f"\n[{self.feature.display_name} {action}完成]\n")
        else:
            message = f"{self.feature.display_name} {action}失败，退出码：{exit_code}。"
            self.output_received.emit(
                f"\n[{self.feature.display_name} {action}失败，退出码：{exit_code}]\n"
            )
        self.emit_status()
        self.finished.emit(self.feature.feature_id, succeeded, message)

    def _on_error(self, error: object) -> None:
        process = self._process
        message = process.errorString() if process is not None else "未知进程错误"
        self.output_received.emit(
            f"\n[{self.feature.display_name} 安装进程错误] {message}\n"
        )
        if error != QProcess.ProcessError.FailedToStart:
            return
        self._process = None
        self._active = False
        self._last_failed = True
        if OptionalFeatureInstallController._active_controller is self:
            OptionalFeatureInstallController._active_controller = None
            OptionalFeatureInstallController._active_feature_id = None
        self.emit_status()
        self.finished.emit(
            self.feature.feature_id,
            False,
            f"无法启动安装进程：{message}",
        )


def build_relation_analyzer_controller(
    parent: QObject | None = None,
    *,
    repo_root: Path = REPO_ROOT,
) -> OptionalFeatureInstallController:
    return OptionalFeatureInstallController(
        relation_analyzer_feature(repo_root),
        parent=parent,
        repo_root=repo_root,
        prefer_hash_lock=True,
    )


def action_enabled_for_status(status: FeatureStatus) -> bool:
    return status.state not in {
        FeatureInstallState.INSTALLING,
        FeatureInstallState.INSTALLED,
    }


def bind_log_callbacks(
    controller: OptionalFeatureInstallController,
    *,
    append_log: Callable[[str], None],
    show_log_drawer: Callable[[], None] | None = None,
) -> None:
    def _on_output(text: str) -> None:
        if show_log_drawer is not None and text.startswith("==="):
            show_log_drawer()
        append_log(text)

    controller.output_received.connect(_on_output)


def recommended_install_command(feature: OptionalFeatureSpec) -> str:
    return hash_checked_install_command(feature, python_executable=sys.executable)
