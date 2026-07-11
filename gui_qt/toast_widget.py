"""浮动提示通知组件。

提供简短的成功/警告/错误消息，自动消失。
支持从底部滑入动画和淡出动画。
"""

from __future__ import annotations

import re
from enum import Enum, auto

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRectF,
    QTimer,
    Qt,
)
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)


class ToastStyle(Enum):
    """提示通知的样式类型。"""

    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()


# ---------------------------------------------------------------------------
# 样式常量
# ---------------------------------------------------------------------------

# (dark_rgba, light_rgba) — solid colours for painted rounded pill
_BACKGROUND_COLORS: dict[ToastStyle, tuple[str, str]] = {
    ToastStyle.SUCCESS: ("#10b981", "#059669"),
    ToastStyle.WARNING: ("#d97706", "#b45309"),
    ToastStyle.ERROR: ("#ef4444", "#dc2626"),
}

_ICON_TEXT: dict[ToastStyle, str] = {
    ToastStyle.SUCCESS: "✓",
    ToastStyle.WARNING: "!",
    ToastStyle.ERROR: "×",
}

# Strip caller-supplied status glyphs so we never show icon + message glyph twice.
_LEADING_STATUS_GLYPHS = re.compile(
    r"^[\s✓✔✕✖❌⚠⚠️✘×!！]+",
)

_SLIDE_OFFSET_PX = 20
_FADE_DURATION_MS = 350
_SLIDE_DURATION_MS = 250
_CORNER_RADIUS = 12


class ToastNotification(QWidget):
    """浮动提示通知，在父组件底部居中显示后自动消失。

    使用类方法 :meth:`show_toast` 创建并显示通知。
    """

    # ------------------------------------------------------------------
    # 公共类方法
    # ------------------------------------------------------------------

    @classmethod
    def show_toast(
        cls,
        parent: QWidget,
        message: str,
        style: ToastStyle = ToastStyle.SUCCESS,
        duration_ms: int = 2500,
    ) -> ToastNotification:
        """创建并显示一个浮动提示通知。

        Args:
            parent: 父组件，通知将相对于此组件定位。
            message: 要显示的消息文本（不必再写 ✓/⚠ 等图标）。
            style: 通知样式（成功/警告/错误）。
            duration_ms: 通知显示时长（毫秒），之后自动淡出。

        Returns:
            创建的 ToastNotification 实例。
        """
        toast = cls(parent, message, style, duration_ms)
        toast.show()
        toast._start_animations()
        return toast

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(
        self,
        parent: QWidget,
        message: str,
        style: ToastStyle = ToastStyle.SUCCESS,
        duration_ms: int = 2500,
    ) -> None:
        super().__init__(parent)
        self._style = style
        self._duration_ms = duration_ms
        self._is_dark = self._detect_dark_mode()
        self._bg_color = QColor(
            _BACKGROUND_COLORS[style][0 if self._is_dark else 1]
        )

        # Frameless floating tip; translucent so rounded corners are not a hard box.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self._setup_ui(message)

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_message(message: str) -> str:
        text = (message or "").strip()
        text = _LEADING_STATUS_GLYPHS.sub("", text).strip()
        return text or message.strip()

    def _setup_ui(self, message: str) -> None:
        """构建通知的内部布局。"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 18, 10)
        layout.setSpacing(8)

        icon_label = QLabel(_ICON_TEXT[self._style])
        icon_label.setObjectName("toast_icon")
        icon_label.setStyleSheet(
            "QLabel#toast_icon {"
            "  font-size: 14px;"
            "  font-weight: 700;"
            "  color: white;"
            "  background: transparent;"
            "  border: none;"
            "}"
        )
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        msg_label = QLabel(self._normalize_message(message))
        msg_label.setObjectName("toast_message")
        msg_label.setStyleSheet(
            "QLabel#toast_message {"
            "  font-size: 13px;"
            "  font-weight: 600;"
            "  color: white;"
            "  background: transparent;"
            "  border: none;"
            "}"
        )
        msg_label.setWordWrap(False)
        msg_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        layout.addWidget(msg_label)

        self.adjustSize()
        # Keep short toasts compact; cap very long messages.
        width = min(max(self.sizeHint().width(), 160), 420)
        self.setFixedWidth(width)
        self.adjustSize()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt override
        """Paint a rounded pill; avoids the hard rectangular ToolTip chrome on Windows."""
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path = QPainterPath()
        path.addRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5),
            _CORNER_RADIUS,
            _CORNER_RADIUS,
        )
        painter.fillPath(path, self._bg_color)
        painter.end()

    # ------------------------------------------------------------------
    # 暗色模式检测
    # ------------------------------------------------------------------

    def _detect_dark_mode(self) -> bool:
        """通过父组件的调色板判断当前是否为暗色模式。

        使用窗口背景色的亮度 (lightness) 来判断。
        """
        parent = self.parentWidget()
        if parent is None:
            return True
        palette = parent.palette()
        bg_color = palette.color(palette.ColorRole.Window)
        # HSL lightness: 0.0 (黑) ~ 1.0 (白)
        return bg_color.lightnessF() < 0.5

    # ------------------------------------------------------------------
    # 定位
    # ------------------------------------------------------------------

    def _position_at_bottom_center(self) -> QPoint:
        """计算通知在父组件底部居中的位置。"""
        parent = self.parentWidget()
        if parent is None:
            return QPoint(0, 0)

        parent_rect = parent.rect()
        x = parent_rect.center().x() - self.width() // 2
        y = parent_rect.bottom() - self.height() - 30  # 距底部 30px
        # 将父组件坐标映射到全局坐标
        return parent.mapToGlobal(QPoint(x, y))

    # ------------------------------------------------------------------
    # 动画
    # ------------------------------------------------------------------

    def _start_animations(self) -> None:
        """启动滑入动画，并设置延迟淡出。"""
        # 最终位置
        final_pos = self._position_at_bottom_center()
        # 起始位置：向下偏移
        start_pos = QPoint(final_pos.x(), final_pos.y() + _SLIDE_OFFSET_PX)

        self.move(start_pos)

        # 滑入动画
        self._slide_anim = QPropertyAnimation(self, b"pos", self)
        self._slide_anim.setStartValue(start_pos)
        self._slide_anim.setEndValue(final_pos)
        self._slide_anim.setDuration(_SLIDE_DURATION_MS)
        self._slide_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._slide_anim.start()

        # 延迟后开始淡出
        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.setInterval(self._duration_ms)
        self._dismiss_timer.timeout.connect(self._start_fade_out)
        self._dismiss_timer.start()

    def _start_fade_out(self) -> None:
        """开始淡出动画，完成后销毁自身。"""
        # 透明度效果
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)

        # 淡出动画
        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setDuration(_FADE_DURATION_MS)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InQuad)
        self._fade_anim.finished.connect(self._on_fade_finished)
        self._fade_anim.start()

    def _on_fade_finished(self) -> None:
        """淡出完成后关闭并销毁。"""
        self.close()
        self.deleteLater()
