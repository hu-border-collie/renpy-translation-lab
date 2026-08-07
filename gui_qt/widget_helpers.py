"""Small Qt widget subclasses for safer desktop UX."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from PySide6.QtCore import Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QComboBox, QLineEdit, QMessageBox, QStyle, QTabWidget

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

QuestionReply = Literal["yes", "no", "cancel"]


class NoWheelComboBox(QComboBox):
    """Ignore mouse-wheel selection changes unless the dropdown list is open."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        popup = self.view()
        if popup is not None and popup.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()


def add_editable_combo_popup_action(combo: QComboBox) -> None:
    """Give an editable combo an explicit, visible way to open its item list."""
    line_edit = combo.lineEdit()
    if not combo.isEditable() or line_edit is None:
        raise ValueError("popup actions require an editable QComboBox")
    if line_edit.property("popup_action_installed"):
        return
    action = line_edit.addAction(
        combo.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown),
        QLineEdit.ActionPosition.TrailingPosition,
    )
    action.setObjectName("combo_popup_action")
    action.setToolTip("选择模型")
    action.triggered.connect(combo.showPopup)
    line_edit.setProperty("popup_action_installed", True)
    combo._popup_action = action


def _build_ok_box(
    parent: "QWidget | None",
    title: str,
    text: str,
    icon: QMessageBox.Icon,
) -> QMessageBox:
    """Build a single-button box with a Chinese 确定 button (not English OK)."""
    box = QMessageBox(parent)
    box.setIcon(icon)
    box.setWindowTitle(title)
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setText(text)
    ok_btn = box.addButton("确定", QMessageBox.ButtonRole.AcceptRole)
    box.setDefaultButton(ok_btn)
    return box


def build_information_box(
    parent: "QWidget | None",
    title: str,
    text: str,
) -> QMessageBox:
    """Build an information box with a Chinese 确定 button (not English OK)."""
    return _build_ok_box(parent, title, text, QMessageBox.Icon.Information)


def build_warning_box(
    parent: "QWidget | None",
    title: str,
    text: str,
) -> QMessageBox:
    """Build a warning box with a Chinese 确定 button (not English OK)."""
    return _build_ok_box(parent, title, text, QMessageBox.Icon.Warning)


def build_question_box(
    parent: "QWidget | None",
    title: str,
    text: str,
    *,
    yes_text: str = "确定",
    no_text: str = "取消",
    cancel_text: str | None = None,
    default: QuestionReply = "yes",
    icon: QMessageBox.Icon = QMessageBox.Icon.Question,
) -> QMessageBox:
    """Build a question box with explicit Chinese button labels."""
    box = QMessageBox(parent)
    box.setIcon(icon)
    box.setWindowTitle(title)
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setText(text)
    yes_btn = box.addButton(yes_text, QMessageBox.ButtonRole.YesRole)
    no_btn = box.addButton(no_text, QMessageBox.ButtonRole.NoRole)
    cancel_btn = None
    if cancel_text is not None:
        cancel_btn = box.addButton(cancel_text, QMessageBox.ButtonRole.RejectRole)
    if default == "no":
        box.setDefaultButton(no_btn)
    elif default == "cancel" and cancel_btn is not None:
        box.setDefaultButton(cancel_btn)
    else:
        box.setDefaultButton(yes_btn)
    return box


def message_box_information(
    parent: "QWidget | None",
    title: str,
    text: str,
) -> None:
    """Information box with a Chinese 确定 button (not English OK)."""
    build_information_box(parent, title, text).exec()


def message_box_warning(
    parent: "QWidget | None",
    title: str,
    text: str,
) -> None:
    """Warning box with a Chinese 确定 button (not English OK)."""
    build_warning_box(parent, title, text).exec()


def message_box_question(
    parent: "QWidget | None",
    title: str,
    text: str,
    *,
    yes_text: str = "确定",
    no_text: str = "取消",
    cancel_text: str | None = None,
    default: QuestionReply = "yes",
    icon: QMessageBox.Icon = QMessageBox.Icon.Question,
) -> QuestionReply:
    """Question box with explicit Chinese button labels.

    Returns ``\"yes\"``, ``\"no\"``, or ``\"cancel\"`` (when cancel is shown).
    """
    box = build_question_box(
        parent,
        title,
        text,
        yes_text=yes_text,
        no_text=no_text,
        cancel_text=cancel_text,
        default=default,
        icon=icon,
    )
    box.exec()
    clicked = box.clickedButton()
    if clicked is not None and box.buttonRole(clicked) == QMessageBox.ButtonRole.YesRole:
        return "yes"
    if clicked is not None and box.buttonRole(clicked) == QMessageBox.ButtonRole.RejectRole:
        return "cancel"
    return "no"


class NoWheelTabWidget(QTabWidget):
    """Ignore mouse-wheel tab switching; use clicks to change tabs."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()
        return
