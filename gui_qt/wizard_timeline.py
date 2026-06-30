"""Custom QWidget representing a step-by-step progress timeline for the GUI.

Features:
- Dynamic step lists.
- Dynamically detected dark/light mode color scheme.
- Antialiased drawing of connecting tracks, node circles, checkmarks, and labels.
- Smooth pulsing glow animation for active steps.
"""
from __future__ import annotations

import math
from PySide6.QtCore import Qt, QTimer, QPoint, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget


class WizardTimeline(QWidget):
    """A horizontal progress timeline widget showing step nodes."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.steps: list[tuple[str, str]] = []  # list of (key, label)
        self.current_step_key: str | None = None
        self.status: str = "idle"  # idle, running, ready, done, failed, waiting, stale

        # Pulse animation state
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_tick_count = 0

        self.setMinimumHeight(66)
        self.setMaximumHeight(80)

    def set_steps(self, steps: list[tuple[str, str]]) -> None:
        """Set the workflow steps. e.g. [('build', '准备'), ('submit', '提交')]"""
        self.steps = list(steps)
        self.update()

    def set_current_step(self, step_key: str | None, status: str = "running") -> None:
        """Set the currently active step key and its status."""
        self.current_step_key = step_key
        self.status = status

        if status in {"running", "waiting"}:
            if not self._anim_timer.isActive():
                self._anim_timer.start(50)  # 20 FPS
        else:
            self._anim_timer.stop()

        self.update()

    def _on_anim_tick(self) -> None:
        self._anim_tick_count += 1
        self.update()

    def is_dark_mode(self) -> bool:
        """Detect if the widget is rendered on a dark background."""
        bg = self.palette().window().color()
        return bg.lightness() < 128

    def paintEvent(self, event) -> None:
        if not self.steps:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        width = self.width()
        n = len(self.steps)

        # Layout dimensions
        y_center = 22
        radius = 11
        spacing = width / n

        # Find current step index
        current_idx = -1
        if self.current_step_key is not None:
            for idx, (key, _) in enumerate(self.steps):
                if key == self.current_step_key:
                    current_idx = idx
                    break

        # Theme-based color palettes
        is_dark = self.is_dark_mode()

        if is_dark:
            # Dark theme colors
            color_pending_bg = QColor("#1e293b")
            color_pending_border = QColor("#475569")
            color_pending_text = QColor("#64748b")

            color_active = QColor("#3b82f6")  # Blue-500
            color_active_glow = QColor(59, 130, 246, 75)
            color_active_text = QColor("#60a5fa")

            color_success = QColor("#10b981")  # Emerald-500
            color_success_text = QColor("#34d399")

            color_failed = QColor("#ef4444")  # Red-500
            color_failed_text = QColor("#f87171")

            color_waiting = QColor("#f59e0b")  # Amber-500
            color_waiting_glow = QColor(245, 158, 11, 75)
            color_waiting_text = QColor("#fbbf24")
        else:
            # Light theme colors
            color_pending_bg = QColor("#f1f5f9")
            color_pending_border = QColor("#cbd5e1")
            color_pending_text = QColor("#94a3b8")

            color_active = QColor("#2563eb")  # Blue-600
            color_active_glow = QColor(37, 99, 235, 60)
            color_active_text = QColor("#1d4ed8")

            color_success = QColor("#059669")  # Emerald-600
            color_success_text = QColor("#047857")

            color_failed = QColor("#dc2626")  # Red-600
            color_failed_text = QColor("#b91c1c")

            color_waiting = QColor("#d97706")  # Amber-600
            color_waiting_glow = QColor(217, 119, 6, 60)
            color_waiting_text = QColor("#b45309")

        # Helper to get state of a step index
        def get_step_state(idx: int) -> str:
            if self.status == "done":
                return "success"
            if current_idx == -1:
                return "pending"
            if idx < current_idx:
                return "success"
            if idx == current_idx:
                if self.status == "failed":
                    return "failed"
                if self.status in {"waiting", "ready", "stale"}:
                    return "waiting"
                return "running"
            return "pending"

        # 1. Draw connecting tracks (lines between steps)
        for i in range(n - 1):
            x1 = int(spacing * (i + 0.5))
            x2 = int(spacing * (i + 1.5))
            state_from = get_step_state(i)
            state_to = get_step_state(i + 1)

            # Determine track color
            if state_from == "success":
                if state_to == "running":
                    track_color = color_active
                elif state_to == "waiting":
                    track_color = color_waiting
                elif state_to == "failed":
                    track_color = color_failed
                else:
                    track_color = color_success
            elif state_from == "running":
                track_color = color_active
            elif state_from == "waiting":
                track_color = color_waiting
            elif state_from == "failed":
                track_color = color_failed
            else:
                track_color = color_pending_border

            pen = QPen(track_color, 3)
            painter.setPen(pen)
            painter.drawLine(x1 + radius + 1, y_center, x2 - radius - 1, y_center)

        # 2. Draw nodes and text
        # Use standalone pixel-sized fonts. Copying QWidget.font() after QSS px
        # rules leaves pointSize at -1 and can trigger Qt warnings when toggled.
        font_index = QFont()
        font_index.setPixelSize(13)
        font_index.setBold(True)

        font_label = QFont()
        font_label.setPixelSize(13)
        font_label_bold = QFont(font_label)
        font_label_bold.setBold(True)

        for i, (_, label) in enumerate(self.steps):
            x = int(spacing * (i + 0.5))
            state = get_step_state(i)

            # Choose colors based on state
            if state == "success":
                bg_color = color_success
                border_color = color_success
                text_color = color_success_text
                num_color = QColor(Qt.GlobalColor.white)
            elif state == "running":
                bg_color = color_active
                border_color = color_active
                text_color = color_active_text
                num_color = QColor(Qt.GlobalColor.white)
            elif state == "waiting":
                bg_color = color_waiting
                border_color = color_waiting
                text_color = color_waiting_text
                num_color = QColor(Qt.GlobalColor.white)
            elif state == "failed":
                bg_color = color_failed
                border_color = color_failed
                text_color = color_failed_text
                num_color = QColor(Qt.GlobalColor.white)
            else:
                bg_color = color_pending_bg
                border_color = color_pending_border
                text_color = color_pending_text
                num_color = color_pending_text

            # Draw glowing pulse rings for running/waiting
            if state in {"running", "waiting"}:
                glow_color = color_active_glow if state == "running" else color_waiting_glow
                pulse_val = abs(math.sin(self._anim_tick_count * 0.15))
                glow_radius = radius + 1 + int(5 * pulse_val)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(glow_color))
                painter.drawEllipse(QPoint(x, y_center), glow_radius, glow_radius)

            # Draw outer circle border
            painter.setPen(QPen(border_color, 2))
            painter.setBrush(QBrush(bg_color))
            painter.drawEllipse(QPoint(x, y_center), radius, radius)

            # Draw number inside node circle
            painter.setFont(font_index)
            painter.setPen(QPen(num_color))
            if state == "success":
                # Draw checkmark symbol
                painter.drawText(
                    QRect(x - radius, y_center - radius, radius * 2, radius * 2),
                    Qt.AlignmentFlag.AlignCenter,
                    "✔",
                )
            else:
                painter.drawText(
                    QRect(x - radius, y_center - radius, radius * 2, radius * 2),
                    Qt.AlignmentFlag.AlignCenter,
                    str(i + 1),
                )

            # Draw label below the circle
            painter.setPen(QPen(text_color))
            if state in {"running", "waiting", "failed"}:
                painter.setFont(font_label_bold)
            else:
                painter.setFont(font_label)

            label_rect = QRect(x - int(spacing / 2), y_center + radius + 4, int(spacing), 20)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

        painter.end()
