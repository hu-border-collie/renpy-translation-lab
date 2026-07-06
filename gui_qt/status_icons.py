"""Status icon mappings and StatusBadge widget.

Provides:
- ``STATUS_ICONS`` – mapping from status strings to Unicode icon characters.
- ``format_status_text`` – pure function (no PySide6 dependency) that
  prepends the matching icon to a label string.
- ``StatusBadge`` – thin ``QLabel`` subclass that couples *text* and a
  *dynamic property* so that QSS rules can restyle the badge automatically.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QWidget

# ── icon look-up table ──────────────────────────────────────────────
STATUS_ICONS: dict[str, str] = {
    "ready":   "✔",
    "done":    "✔",
    "safe":    "✔",
    "applied": "✔",
    "idle":    "●",
    "running": "🔄",
    "warning": "⚠",
    "stale":   "⚠",
    "waiting": "🕐",
    "warn":    "⚠",
    "blocked": "✖",
    "failed":  "✖",
    "block":   "✖",
    "unknown": "?",
}


# ── pure helper (works without PySide6) ─────────────────────────────
def format_status_text(status: str, text: str) -> str:
    """Return *text* prefixed with the status icon if one exists.

    The icon and text are separated by two spaces so that the glyph is
    visually distinct from the label.  If *status* has no registered icon
    the original *text* is returned unchanged.

    This function is intentionally free of any Qt imports so that it can
    be used in non-GUI contexts (CLI helpers, tests, etc.).
    """
    icon = STATUS_ICONS.get(status)
    if icon is None:
        return text
    return f"{icon}  {text}"


# ── badge widget ────────────────────────────────────────────────────
class StatusBadge(QLabel):
    """A ``QLabel`` that keeps its displayed text and QSS status in sync.

    Typical usage replaces the repetitive four-liner found throughout
    *app.py*::

        badge = StatusBadge("myBadge")
        badge.set_status("ready", "翻译就绪")

    which is equivalent to::

        label.setText("✔  翻译就绪")
        label.setProperty("status", "ready")
        label.style().unpolish(label)
        label.style().polish(label)
    """

    def __init__(self, object_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)

    # ── public API ──────────────────────────────────────────────────
    def set_status(self, status: str, text: str) -> None:
        """Update the badge label and refresh its stylesheet.

        Parameters
        ----------
        status:
            One of the keys in :data:`STATUS_ICONS` (e.g. ``"ready"``).
            The value is also written to the ``status`` dynamic property
            so that QSS selectors like ``QLabel[status="ready"]`` work.
        text:
            Human-readable text shown on the badge.  The matching icon
            character is automatically prepended.
        """
        self.setText(format_status_text(status, text))
        self.setProperty("status", status)
        self.style().unpolish(self)
        self.style().polish(self)
