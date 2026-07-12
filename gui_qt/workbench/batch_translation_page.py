"""Persistent batch-translation page for the workbench stack (#176 P5)."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..responsive_layout import FlowButtonBar, ResponsiveActionPanel
from ..work_modes import WorkMode
from ..workbench_session import WorkbenchModeSession
from .page_contract import WorkbenchPageActions


ControlState = tuple[bool, bool, str]


class BatchTranslationPage(QWidget):
    """Page-local batch actions; the coordinator remains the state authority."""

    content_height_changed = Signal()
    supported_modes = (WorkMode.BATCH_TRANSLATION,)
    _main_actions = ("start", "resume", "stop", "split_submit")
    _writeback_actions = ("apply",)
    _issue_actions = (
        "issues",
        "recheck",
        "retry",
        "retry_followup",
        "repair",
        "failure",
        "remediation",
    )
    _advanced_actions = ("probe", "split")
    _labels = {
        "start": "开始翻译",
        "resume": "继续翻译",
        "stop": "停止",
        "split_submit": "提交剩余包",
        "apply": "写回翻译",
        "recheck": "重新检查",
        "issues": "查看问题清单",
        "retry": "生成补译包",
        "retry_followup": "继续补译",
        "repair": "同步修补",
        "failure": "查看写回失败报告",
        "remediation": "补救命令",
        "probe": "试跑样本请求",
        "split": "拆分翻译包",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("batch_translation_page")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._actions = WorkbenchPageActions()
        self._running = False
        self._issues_expanded = False
        self.buttons: dict[str, QPushButton] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # Preserve the legacy workbench hierarchy while moving ownership on-page.
        self.main_frame = QFrame()
        self.main_frame.setObjectName("action_frame")
        main_layout = QVBoxLayout(self.main_frame)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(8)
        self.main_bar = ResponsiveActionPanel(compact_width=640)
        self.main_bar.setObjectName("batch_main_actions")
        for key in self._main_actions:
            button = self._make_button(key)
            if key == "stop":
                self.main_bar.add_translate_trailing(button)
            else:
                self.main_bar.add_translate_button(button)
        self.main_bar.finish_setup()
        main_layout.addWidget(self.main_bar)
        outer.addWidget(self.main_frame)

        # Keep writeback and recovery disclosure on the same row as before P5.
        self.writeback_row = QWidget()
        writeback_layout = QHBoxLayout(self.writeback_row)
        writeback_layout.setContentsMargins(0, 0, 0, 0)
        writeback_layout.setSpacing(8)
        self.writeback_bar = self._build_bar(
            "batch_writeback_actions", self._writeback_actions
        )
        writeback_layout.addWidget(self.writeback_bar, 1)
        self.issues_toggle_btn = QPushButton("问题处理 ▸")
        self.issues_toggle_btn.setObjectName("secondary_btn")
        self.issues_toggle_btn.setToolTip("展开补译、修补、问题清单和重新检查等恢复操作。")
        self.issues_toggle_btn.clicked.connect(self._toggle_issues)
        writeback_layout.addWidget(self.issues_toggle_btn)
        outer.addWidget(self.writeback_row)

        self.issues_bar = self._build_bar("batch_issue_actions", self._issue_actions)
        self.issues_bar.setVisible(False)
        outer.addWidget(self.issues_bar)

        self.advanced_frame = QFrame()
        self.advanced_frame.setObjectName("batch_advanced_frame")
        advanced_layout = QVBoxLayout(self.advanced_frame)
        advanced_layout.setContentsMargins(12, 10, 12, 10)
        advanced_layout.setSpacing(6)
        advanced_title = QLabel("高级工具")
        advanced_title.setObjectName("action_group_label")
        advanced_layout.addWidget(advanced_title)
        self.advanced_bar = self._build_bar(
            "batch_advanced_actions", self._advanced_actions
        )
        advanced_layout.addWidget(self.advanced_bar)
        outer.addWidget(self.advanced_frame)

    def _make_button(self, key: str) -> QPushButton:
        button = QPushButton(self._labels[key])
        button.setObjectName(
            {"start": "translate_btn", "stop": "kill_btn", "apply": "apply_btn"}.get(
                key, "secondary_btn"
            )
        )
        button.setEnabled(False)
        button.setVisible(False)
        button.clicked.connect(
            lambda _checked=False, name=key: self._trigger(name)
        )
        self.buttons[key] = button
        return button

    def _build_bar(self, role: str, keys: tuple[str, ...]) -> FlowButtonBar:
        bar = FlowButtonBar(spacing=8, row_spacing=8)
        bar.setObjectName("action_frame")
        bar.setProperty("batch_role", role)
        for key in keys:
            button = self._make_button(key)
            bar.add_widget(button, min_width=88)
        bar.finish_setup()
        return bar

    def preferred_height(self, width: int) -> int:
        """Return content height after responsive action rows have reflowed."""
        for bar in (
            self.main_bar,
            self.writeback_bar,
            self.issues_bar,
            self.advanced_bar,
        ):
            bar.reflow(force=True)
        layout = self.layout()
        if layout is None:
            return self.sizeHint().height()
        return max(
            self.minimumSizeHint().height(), layout.heightForWidth(max(width, 260))
        )

    def set_action_callbacks(self, actions: WorkbenchPageActions) -> None:
        self._actions = actions

    def activate(self, mode: WorkMode, session: WorkbenchModeSession) -> None:
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported batch mode: {mode.value}")
        del session

    def set_task_running(self, running: bool) -> None:
        self._running = running
        for key, button in self.buttons.items():
            if key == "stop":
                button.setEnabled(running)
            elif running:
                button.setEnabled(False)

    def set_controls(
        self,
        controls: dict[str, ControlState],
        *,
        issues_expanded: bool | None = None,
    ) -> None:
        """Render coordinator-derived readiness without owning workflow state."""
        for key, button in self.buttons.items():
            visible, enabled, label = controls.get(
                key, (False, False, self._labels[key])
            )
            button.setVisible(visible)
            button.setText(label)
            button.setEnabled(enabled and not self._running)
        self.buttons["stop"].setVisible(True)
        self.buttons["stop"].setEnabled(self._running)
        self.main_frame.setVisible(True)
        self.writeback_row.setVisible(
            any(not self.buttons[key].isHidden() for key in self._writeback_actions)
        )
        self.issues_toggle_btn.setVisible(
            any(not self.buttons[key].isHidden() for key in self._issue_actions)
        )
        if issues_expanded is not None and issues_expanded != self._issues_expanded:
            self._issues_expanded = issues_expanded
        self.issues_bar.setVisible(
            self._issues_expanded and not self.issues_toggle_btn.isHidden()
        )
        self.advanced_frame.setVisible(True)
        for bar in (
            self.main_bar,
            self.writeback_bar,
            self.issues_bar,
            self.advanced_bar,
        ):
            bar.reflow(force=True)
        self.updateGeometry()

    def reset_project(self) -> None:
        self._issues_expanded = False
        self.set_task_running(False)
        self.set_controls(
            {
                "probe": (True, False, self._labels["probe"]),
                "split": (True, False, self._labels["split"]),
            }
        )

    def _toggle_issues(self) -> None:
        self._issues_expanded = not self._issues_expanded
        self.issues_toggle_btn.setText(
            "问题处理 ▾" if self._issues_expanded else "问题处理 ▸"
        )
        self.issues_bar.setVisible(self._issues_expanded)
        self.issues_bar.reflow(force=True)
        self.updateGeometry()
        self.content_height_changed.emit()

    def _trigger(self, action: str) -> None:
        if action == "stop":
            if self._running and self._actions.action is not None:
                self._actions.action(action)
            return
        if (
            not self._running
            and self.buttons[action].isEnabled()
            and self._actions.action is not None
        ):
            self._actions.action(action)
