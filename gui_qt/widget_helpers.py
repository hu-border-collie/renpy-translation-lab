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


def message_box_information(
    parent: "QWidget | None",
    title: str,
    text: str,
) -> None:
    """Information box with a Chinese 确定 button (not English OK)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Information)
    box.setWindowTitle(title)
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setText(text)
    ok_btn = box.addButton("确定", QMessageBox.ButtonRole.AcceptRole)
    box.setDefaultButton(ok_btn)
    box.exec()


def message_box_warning(
    parent: "QWidget | None",
    title: str,
    text: str,
) -> None:
    """Warning box with a Chinese 确定 button (not English OK)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setText(text)
    ok_btn = box.addButton("确定", QMessageBox.ButtonRole.AcceptRole)
    box.setDefaultButton(ok_btn)
    box.exec()


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
    box.exec()
    clicked = box.clickedButton()
    if clicked is yes_btn:
        return "yes"
    if cancel_btn is not None and clicked is cancel_btn:
        return "cancel"
    return "no"


class NoWheelTabWidget(QTabWidget):
    """Ignore mouse-wheel tab switching; use clicks to change tabs."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()
        return
