"""Responsive layouts for workbench / settings / diagnostics button groups."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Qt often keeps hidden tab pages at this default until first real layout pass.
_UNLAID_OUT_DEFAULT_WIDTH = 640

# Prefer these ancestors when a flow bar is still at the default width (off-stage tab).
_CONTENT_WIDTH_OBJECT_NAMES = frozenset(
    {
        "workbench_writeback_page",
        "workbench_status_tabs",
        "workbench_page",
        "workbench_stack",
    }
)


def configure_action_button(widget: QWidget, *, min_width: int = 88) -> QWidget:
    """Prefer Preferred sizing so rows can shrink instead of painting over neighbors."""
    widget.setMinimumWidth(min_width)
    widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    return widget


# Back-compat alias used by older call sites / tests.
_configure_action_button = configure_action_button


def widget_layout_width(widget: QWidget, *, fallback: int = 88) -> int:
    """Best-effort horizontal space a control needs in a flow row."""
    if not widget.isVisibleTo(widget.parentWidget() or widget):
        # Still measure hidden widgets that will be shown later via reflow.
        pass
    hint = widget.sizeHint()
    min_w = widget.minimumWidth()
    width = max(min_w, hint.width() if hint.isValid() and hint.width() > 0 else 0)
    if width <= 0:
        width = fallback
    return width


def effective_widget_width(widget: QWidget, *, fallback: int = 720) -> int:
    """Width available for laying out children of a flow / action strip.

    Hidden ``QTabWidget`` pages often keep Qt's default ~640px size until shown.
    When the bar itself still looks un-laid-out, prefer a named content ancestor
    (writeback page / status tabs) so off-stage reflow targets the real shell width.
    """
    own = int(widget.width() or 0)
    ancestor_w = 0
    parent = widget.parentWidget()
    depth = 0
    while parent is not None and depth < 16:
        pw = int(parent.width() or 0)
        name = parent.objectName() or ""
        if name in _CONTENT_WIDTH_OBJECT_NAMES and pw > ancestor_w:
            ancestor_w = pw
        elif parent.isVisible() and pw > ancestor_w and name:
            # Any visibly laid-out named chrome is better than the 640 default.
            if pw > _UNLAID_OUT_DEFAULT_WIDTH:
                ancestor_w = max(ancestor_w, pw)
        parent = parent.parentWidget()
        depth += 1

    if own > _UNLAID_OUT_DEFAULT_WIDTH:
        return own
    if widget.isVisible() and own > 0 and ancestor_w <= _UNLAID_OUT_DEFAULT_WIDTH:
        return own
    if ancestor_w > 0:
        # Account for typical page margins when borrowing ancestor width.
        return max(own, ancestor_w - 24) if own > 0 else max(120, ancestor_w - 24)
    if own > 0:
        return own
    return fallback


def clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        child = item.layout()
        if child is not None:
            clear_layout(child)
            # Drop nested layouts immediately so rebuild can re-parent widgets cleanly.
            child.setParent(None)


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
        # Never let the parent VBox crush stacked action rows into each other.
        self._root.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # Preferred allows vertical crush under tight shells; Minimum keeps row gaps.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def add_prep_button(self, widget: QWidget, *, min_width: int = 88) -> QWidget:
        self._prep_buttons.append(configure_action_button(widget, min_width=min_width))
        return widget

    def add_translate_button(self, widget: QWidget, *, min_width: int = 88) -> QWidget:
        self._translate_buttons.append(configure_action_button(widget, min_width=min_width))
        return widget

    def add_translate_trailing(self, widget: QWidget, *, min_width: int = 72) -> QWidget:
        self._translate_trailing.append(configure_action_button(widget, min_width=min_width))
        return widget

    def finish_setup(self) -> None:
        self._is_wide = None
        self._apply_layout_mode(self._should_use_wide_layout(), force=True)

    def reflow(self, *, force: bool = True) -> None:
        """Re-evaluate wide/stacked mode (e.g. after button visibility changes)."""
        self._apply_layout_mode(self._should_use_wide_layout(), force=force)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_layout_mode(self._should_use_wide_layout())

    def _effective_width(self) -> int:
        return effective_widget_width(self, fallback=self._compact_width)

    def _estimated_wide_min_width(self) -> int:
        """Minimum width needed for a single-row layout without clipping."""
        spacing = 12
        width = widget_layout_width(self.prep_label, fallback=64)
        width += widget_layout_width(self.translate_label, fallback=64)
        widgets = [
            *self._prep_buttons,
            *self._translate_buttons,
            *self._translate_trailing,
        ]
        # Only count currently visible action buttons for the estimate.
        visible = [w for w in widgets if w.isVisible() or not w.testAttribute(Qt.WidgetAttribute.WA_WState_Hidden)]
        # Before first show, isVisible() is false; treat non-explicitly-hidden as visible.
        measured: list[QWidget] = []
        for widget in widgets:
            if widget.isHidden():
                continue
            measured.append(widget)
        if not measured:
            measured = list(widgets)
        for widget in measured:
            width += widget_layout_width(widget)
        gaps = max(0, len(measured) + 2)
        width += spacing * gaps + 96
        return width

    def _should_use_wide_layout(self) -> bool:
        available = self._effective_width()
        needed = max(self._compact_width, self._estimated_wide_min_width())
        return available >= needed

    def _apply_layout_mode(self, wide: bool, *, force: bool = False) -> None:
        if not force and wide == self._is_wide:
            return
        self._is_wide = wide
        self._rebuild_layout()

    def _rebuild_layout(self) -> None:
        clear_layout(self._root)

        prep = [w for w in self._prep_buttons if not w.isHidden()]
        translate = [w for w in self._translate_buttons if not w.isHidden()]
        trailing = [w for w in self._translate_trailing if not w.isHidden()]
        # On first paint before show(), nothing is "shown" yet — keep all members.
        if not prep and not translate and not trailing:
            prep = list(self._prep_buttons)
            translate = list(self._translate_buttons)
            trailing = list(self._translate_trailing)

        if self._is_wide:
            self._root.setSpacing(8)
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
            row.addStretch(1)
            for widget in trailing:
                row.addWidget(widget)
            self._root.addLayout(row)
            self._sync_minimum_height()
            return

        # Stacked: keep a clear vertical gap so themed 38px buttons never paint over.
        self._root.setSpacing(12)
        prep_row = QHBoxLayout()
        prep_row.setSpacing(12)
        prep_row.addWidget(self.prep_label)
        for widget in prep:
            prep_row.addWidget(widget)
        prep_row.addStretch(1)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("action_separator")
        separator.setFixedHeight(1)

        translate_row = QHBoxLayout()
        translate_row.setSpacing(12)
        translate_row.addWidget(self.translate_label)
        for widget in translate:
            translate_row.addWidget(widget)
        translate_row.addStretch(1)
        for widget in trailing:
            translate_row.addWidget(widget)

        self._root.addLayout(prep_row)
        self._root.addWidget(separator)
        self._root.addLayout(translate_row)
        self._sync_minimum_height()

    def _sync_minimum_height(self) -> None:
        """Lock height to layout sizeHint so parent shells cannot crush stacked rows."""
        hint = self._root.sizeHint()
        height = max(36, hint.height() if hint.isValid() else 0)
        if not self._is_wide:
            # Two button rows + separator + spacings under QSS min-height.
            height = max(height, 96)
        self.setMinimumHeight(height)
        self.updateGeometry()
        # Bubble size change so siblings (e.g. batch advanced tools) reflow below us.
        parent = self.parentWidget()
        depth = 0
        while parent is not None and depth < 4:
            parent.updateGeometry()
            layout = parent.layout()
            if layout is not None:
                layout.activate()
            parent = parent.parentWidget()
            depth += 1


class FlowButtonBar(QFrame):
    """Button/control row that wraps into multiple HBox rows when width is tight.

    Intended for primary writeback actions, recovery tools, global project
    actions, and other flat button strips that previously overflowed.
    """

    def __init__(
        self,
        *,
        spacing: int = 8,
        row_spacing: int = 8,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._spacing = spacing
        self._row_spacing = row_spacing
        self._items: list[QWidget] = []
        self._trailing_stretch = True
        self._signature: tuple[object, ...] | None = None
        self._row_count = 1

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(self._row_spacing)
        self._root.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        # Preferred allows parent shells to crush multi-row wrap into ~12px.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def add_widget(self, widget: QWidget, *, min_width: int | None = 80) -> QWidget:
        if min_width is not None:
            configure_action_button(widget, min_width=min_width)
        else:
            widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._items.append(widget)
        return widget

    def add_stretch_end(self, enabled: bool = True) -> None:
        self._trailing_stretch = enabled

    def finish_setup(self) -> None:
        self.reflow(force=True)

    def reflow(self, *, force: bool = False) -> None:
        signature = self._layout_signature()
        if not force and signature == self._signature:
            return
        self._signature = signature
        self._rebuild()

    def _schedule_reflow(self) -> None:
        """Reflow after the current event so parent/tab geometry is final."""
        QTimer.singleShot(0, lambda: self.reflow(force=True))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.reflow()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.reflow(force=True)
        self._schedule_reflow()

    def _layout_signature(self) -> tuple[object, ...]:
        available = effective_widget_width(self, fallback=800)
        visible = tuple(
            (id(w), w.isHidden(), widget_layout_width(w)) for w in self._items
        )
        return (available, visible)

    def _visible_items(self) -> list[QWidget]:
        # Prefer explicitly non-hidden widgets; before first show treat all as visible.
        shown = [w for w in self._items if not w.isHidden()]
        return shown if shown else list(self._items)

    def _rebuild(self) -> None:
        clear_layout(self._root)
        items = self._visible_items()
        if not items:
            self._row_count = 0
            self.setMinimumHeight(0)
            return

        available = max(120, effective_widget_width(self, fallback=800) - 8)
        row = QHBoxLayout()
        row.setSpacing(self._spacing)
        used = 0
        rows = 0

        def flush_row(current: QHBoxLayout) -> QHBoxLayout:
            nonlocal rows
            if self._trailing_stretch:
                current.addStretch(1)
            self._root.addLayout(current)
            rows += 1
            nxt = QHBoxLayout()
            nxt.setSpacing(self._spacing)
            return nxt

        for widget in items:
            need = widget_layout_width(widget) + (self._spacing if used else 0)
            if used > 0 and used + need > available:
                row = flush_row(row)
                used = 0
                need = widget_layout_width(widget)
            row.addWidget(widget)
            used += need if used == 0 else need

        flush_row(row)
        self._row_count = max(1, rows)
        self._sync_minimum_height()

    def _sync_minimum_height(self) -> None:
        """Reserve full height for wrapped rows so parents cannot crush them."""
        row_h = 38
        for widget in self._visible_items():
            hint = widget.sizeHint()
            if hint.isValid() and hint.height() > 0:
                row_h = max(row_h, hint.height())
            row_h = max(row_h, widget.minimumHeight())
        rows = max(1, self._row_count)
        height = rows * row_h + max(0, rows - 1) * self._row_spacing
        hint = self._root.sizeHint()
        if hint.isValid():
            height = max(height, hint.height())
        self.setMinimumHeight(height)
        self.updateGeometry()
        parent = self.parentWidget()
        depth = 0
        while parent is not None and depth < 6:
            parent.updateGeometry()
            layout = parent.layout()
            if layout is not None:
                layout.invalidate()
                layout.activate()
            parent = parent.parentWidget()
            depth += 1

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        min_h = self.minimumHeight()
        if min_h > 0:
            hint.setHeight(max(hint.height(), min_h))
        elif hint.height() < 28:
            hint.setHeight(28)
        return hint

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        hint = super().minimumSizeHint()
        min_h = self.minimumHeight()
        if min_h > 0:
            hint.setHeight(max(hint.height(), min_h))
        return hint


def collect_visible_button_rects(root: QWidget) -> list[tuple[str, object]]:
    """Return (objectName, QRect in root coords) for visible QPushButtons under root."""
    from PySide6.QtCore import QRect

    rects: list[tuple[str, QRect]] = []
    for btn in root.findChildren(QPushButton):
        if not btn.isVisible():
            continue
        # Skip zero-size or not yet laid out widgets.
        if btn.width() <= 1 or btn.height() <= 1:
            continue
        top_left = btn.mapTo(root, btn.rect().topLeft())
        rect = btn.rect()
        rect.moveTopLeft(top_left)
        name = btn.objectName() or btn.text() or btn.__class__.__name__
        rects.append((name, rect))
    return rects


def find_overlapping_buttons(
    root: QWidget,
    *,
    min_overlap_px: int = 4,
) -> list[tuple[str, str, int, int]]:
    """Return overlapping pairs (name_a, name_b, overlap_w, overlap_h)."""
    items = collect_visible_button_rects(root)
    hits: list[tuple[str, str, int, int]] = []
    for i, (name_a, rect_a) in enumerate(items):
        for name_b, rect_b in items[i + 1 :]:
            inter = rect_a.intersected(rect_b)
            if inter.width() >= min_overlap_px and inter.height() >= min_overlap_px:
                # Same-center full containment can be nested chrome; still report.
                hits.append((name_a, name_b, inter.width(), inter.height()))
    return hits
