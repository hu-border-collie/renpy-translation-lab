"""Shared Settings page chrome for migrated pages (#202 Phase D)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def style_themed_surface(widget: QWidget) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


def build_settings_scroll_page(
    object_name: str,
) -> tuple[QScrollArea, QWidget, QVBoxLayout]:
    """Build the standard Settings scroll surface used by MainWindow pages."""
    scroll = QScrollArea()
    scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    scroll.setObjectName(f"{object_name}_scroll")
    style_themed_surface(scroll)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    viewport = scroll.viewport()
    viewport.setObjectName(f"{object_name}_viewport")
    style_themed_surface(viewport)

    content = QWidget()
    content.setObjectName(f"{object_name}_content")
    style_themed_surface(content)
    content_layout = QHBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(0)

    body = QWidget()
    body.setObjectName("settings_page_body")
    body.setProperty("settingsPage", object_name)
    body.setMinimumWidth(0)
    body.setMaximumWidth(16777215)
    body.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.MinimumExpanding,
    )
    style_themed_surface(body)
    layout = QVBoxLayout(body)
    layout.setContentsMargins(20, 18, 20, 20)
    layout.setSpacing(14)
    content_layout.addWidget(body, 1)
    scroll.setWidget(content)
    return scroll, body, layout


def settings_group(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    group = QGroupBox(title)
    layout = QVBoxLayout(group)
    layout.setSpacing(10)
    layout.setContentsMargins(14, 18, 14, 14)
    return group, layout


def settings_form(group: QGroupBox) -> QFormLayout:
    form = QFormLayout(group)
    form.setObjectName("settings_form")
    form.setContentsMargins(14, 18, 14, 14)
    form.setHorizontalSpacing(16)
    form.setVerticalSpacing(10)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    return form
