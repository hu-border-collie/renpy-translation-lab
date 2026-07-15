"""Persistent batch-translation page for the workbench stack (#176 P5)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..responsive_layout import FlowButtonBar
from ..work_modes import WorkMode
from ..workbench_session import WorkbenchModeSession
from .page_contract import WorkbenchPageActions


ControlState = tuple[bool, bool, str]


@dataclass(frozen=True)
class BatchActionState:
    """Render-only batch action state supplied by the coordinator."""

    controls: Mapping[str, ControlState] = field(default_factory=dict)
    running: bool = False


class BatchTranslationPage(QWidget):
    """Batch task controls; writeback and recovery stay on the writeback tab."""

    content_height_changed = Signal()
    supported_modes = (WorkMode.BATCH_TRANSLATION,)
    _main_actions = ("start", "resume", "stop", "probe", "split")
    _split_actions = ("split_submit",)
    _labels = {
        "start": "开始翻译",
        "resume": "继续翻译",
        "stop": "停止",
        "split_submit": "提交剩余包",
        "probe": "试跑样本请求",
        "split": "拆分翻译包",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("batch_translation_page")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._actions = WorkbenchPageActions()
        self._state = BatchActionState()
        self.buttons: dict[str, QPushButton] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        self.main_frame = QFrame()
        self.main_frame.setObjectName("action_frame")
        main_layout = QVBoxLayout(self.main_frame)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(8)
        main_title = QLabel("翻译任务")
        main_title.setObjectName("action_group_label")
        main_layout.addWidget(main_title)
        self.main_bar = self._build_bar(
            "batch_main_actions",
            self._main_actions,
            min_widths={
                "start": 108,
                "resume": 108,
                "stop": 80,
                "probe": 108,
                "split": 108,
            },
        )
        main_layout.addWidget(self.main_bar)
        outer.addWidget(self.main_frame)

        self.split_frame = QFrame()
        self.split_frame.setObjectName("batch_split_frame")
        split_layout = QVBoxLayout(self.split_frame)
        split_layout.setContentsMargins(12, 10, 12, 10)
        split_layout.setSpacing(6)
        split_title = QLabel("拆分任务")
        split_title.setObjectName("action_group_label")
        split_layout.addWidget(split_title)
        self.split_bar = self._build_bar(
            "batch_split_actions",
            self._split_actions,
            min_widths={"split_submit": 108},
        )
        split_layout.addWidget(self.split_bar)
        self.split_frame.setVisible(False)
        outer.addWidget(self.split_frame)


    def _make_button(self, key: str) -> QPushButton:
        button = QPushButton(self._labels[key])
        button.setObjectName(
            {"start": "translate_btn", "stop": "kill_btn"}.get(
                key, "secondary_btn"
            )
        )
        button.setEnabled(False)
        button.setVisible(False)
        button.clicked.connect(lambda _checked=False, name=key: self._trigger(name))
        self.buttons[key] = button
        return button

    def _build_bar(
        self,
        role: str,
        keys: tuple[str, ...],
        *,
        min_widths: Mapping[str, int] | None = None,
    ) -> FlowButtonBar:
        bar = FlowButtonBar(spacing=8, row_spacing=8)
        bar.setObjectName(role)
        bar.setProperty("batch_role", role)
        widths = min_widths or {}
        for key in keys:
            button = self._make_button(key)
            bar.add_widget(button, min_width=widths.get(key, 88))
        bar.finish_setup()
        return bar

    def preferred_height(self, width: int) -> int:
        """Return content height after responsive action rows have reflowed."""
        for bar in (self.main_bar, self.split_bar):
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
        self.set_action_state(
            BatchActionState(controls=self._state.controls, running=running)
        )

    def set_action_state(self, state: BatchActionState) -> None:
        self._state = state
        self._render_controls()

    def set_controls(
        self,
        controls: Mapping[str, ControlState],
        *,
        issues_expanded: bool | None = None,
    ) -> None:
        """Compatibility wrapper for callers moving to :class:`BatchActionState`."""
        del issues_expanded
        self.set_action_state(
            BatchActionState(controls=dict(controls), running=self._state.running)
        )

    def _render_controls(self) -> None:
        controls = self._state.controls
        running = self._state.running
        for key, button in self.buttons.items():
            visible, enabled, label = controls.get(
                key, (False, False, self._labels[key])
            )
            button.setText(label)
            show = bool(visible)
            if running:
                show = key == "stop"
            elif key == "resume":
                # A disabled resume action is dead chrome; reveal it when actionable.
                show = show and bool(enabled)
            button.setVisible(show)
            button.setEnabled(bool(enabled) and not running)

        stop = self.buttons["stop"]
        stop.setVisible(True)
        stop.setEnabled(running)

        split_visible = not self.buttons["split_submit"].isHidden() and not running
        self.split_frame.setVisible(split_visible)

        for bar in (self.main_bar, self.split_bar):
            bar.reflow(force=True)
        self.updateGeometry()

    def reset_project(self) -> None:
        self.set_action_state(
            BatchActionState(
                controls={
                    "probe": (True, False, self._labels["probe"]),
                    "split": (True, False, self._labels["split"]),
                }
            )
        )


    def _trigger(self, action: str) -> None:
        if action == "stop":
            if self._state.running and self._actions.action is not None:
                self._actions.action(action)
            return
        if (
            not self._state.running
            and self.buttons[action].isEnabled()
            and self._actions.action is not None
        ):
            self._actions.action(action)
