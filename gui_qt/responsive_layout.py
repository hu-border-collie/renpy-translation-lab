"""Responsive layouts for workbench action button groups."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _configure_action_button(widget: QWidget, *, min_width: int = 88) -> QWidget:
    # Preferred (not Fixed) so rows can shrink instead of painting over neighbors.
    widget.setMinimumWidth(min_width)
    widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    return widget


class ResponsiveActionPanel(QFrame):
    """Merge prep + translate rows on wide screens; stack them when narrow."""

    def __init__(
        self,
        *,
        prep_label: str = "项目准备",
        translate_label: str = "翻译任务",
        compact_width: int = 720,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._compact_width = compact_width
        self._is_wide: bool | None = None
        self._prep_buttons: list[QWidget] = []
        self._translate_buttons: list[QWidget] = []
        self._translate_trailing: list[QWidget] = []

        self.prep_label = QLabel(prep_label)
        self.prep_label.setObjectName("action_group_label")
        self.prep_label.setMinimumWidth(64)

        self.translate_label = QLabel(translate_label)
        self.translate_label.setObjectName("action_group_label")
        self.translate_label.setMinimumWidth(64)

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(8)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def add_prep_button(self, widget: QWidget, *, min_width: int = 88) -> QWidget:
        self._prep_buttons.append(_configure_action_button(widget, min_width=min_width))
        return widget

    def add_translate_button(self, widget: QWidget, *, min_width: int = 88) -> QWidget:
        self._translate_buttons.append(_configure_action_button(widget, min_width=min_width))
        return widget

    def add_translate_trailing(self, widget: QWidget, *, min_width: int = 72) -> QWidget:
        self._translate_trailing.append(_configure_action_button(widget, min_width=min_width))
        return widget

    def finish_setup(self) -> None:
        self._is_wide = None
        self._apply_layout_mode(self._should_use_wide_layout(), force=True)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_layout_mode(self._should_use_wide_layout())

    def _effective_width(self) -> int:
        width = self.width()
        if width > 0:
            return width
        parent = self.parentWidget()
        while parent is not None:
            if parent.width() > 0:
                return parent.width()
            parent = parent.parentWidget()
        return self._compact_width

    def _estimated_wide_min_width(self) -> int:
        """Minimum width needed for a single-row layout without clipping."""
        spacing = 12
        width = max(self.prep_label.minimumWidth(), self.prep_label.sizeHint().width())
        width += max(
            self.translate_label.minimumWidth(),
            self.translate_label.sizeHint().width(),
        )
        widgets = [
            *self._prep_buttons,
            *self._translate_buttons,
            *self._translate_trailing,
        ]
        for widget in widgets:
            hint = widget.sizeHint().width()
            width += max(widget.minimumWidth(), hint if hint > 0 else 88)
        # labels + vertical separator + inter-item spacing + padding slack
        # (Chinese captions are wider than min_width alone suggests.)
        gaps = max(0, len(widgets) + 2)
        width += spacing * gaps + 96
        return width

    def _should_use_wide_layout(self) -> bool:
        available = self._effective_width()
        # Stack early enough that min-width buttons cannot paint over each other.
        needed = max(self._compact_width, self._estimated_wide_min_width())
        return available >= needed

    def _apply_layout_mode(self, wide: bool, *, force: bool = False) -> None:
        if not force and wide == self._is_wide:
            return
        self._is_wide = wide
        self._rebuild_layout()

    def _detach_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            child_layout = item.layout()
            if child_layout is not None:
                self._detach_layout(child_layout)
                child_layout.deleteLater()

    def _rebuild_layout(self) -> None:
        self._detach_layout(self._root)

        prep = self._prep_buttons
        translate = self._translate_buttons
        trailing = self._translate_trailing

        if self._is_wide:
            row = QHBoxLayout()
            row.setSpacing(12)
            row.addWidget(self.prep_label)
            for widget in prep:
                row.addWidget(widget)
            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.VLine)
            separator.setObjectName("action_separator")
            row.addWidget(separator)
            row.addWidget(self.translate_label)
            for widget in translate:
                row.addWidget(widget)
            row.addStretch()
            for widget in trailing:
                row.addWidget(widget)
            self._root.addLayout(row)
            return

        prep_row = QHBoxLayout()
        prep_row.setSpacing(12)
        prep_row.addWidget(self.prep_label)
        for widget in prep:
            prep_row.addWidget(widget)
        prep_row.addStretch()

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("action_separator")

        translate_row = QHBoxLayout()
        translate_row.setSpacing(12)
        translate_row.addWidget(self.translate_label)
        for widget in translate:
            translate_row.addWidget(widget)
        translate_row.addStretch()
        for widget in trailing:
            translate_row.addWidget(widget)

        self._root.addLayout(prep_row)
        self._root.addWidget(separator)
        self._root.addLayout(translate_row)
