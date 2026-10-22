"""Syntax highlighter for the log output panel.

Provides colour-coded formatting for log levels, separators, file paths,
and JSON keys.  Supports both the light and dark GUI themes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat, QTextDocument


# ---------------------------------------------------------------------------
# Theme colour palettes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ThemeColors:
    """Hex colour strings for one theme variant."""

    red: str
    amber: str
    green: str
    blue: str
    cyan: str


_LIGHT_COLORS = _ThemeColors(
    red="#dc2626",
    amber="#d97706",
    green="#059669",
    blue="#2563eb",
    cyan="#0891b2",
)

_DARK_COLORS = _ThemeColors(
    red="#f87171",
    amber="#fbbf24",
    green="#34d399",
    blue="#60a5fa",
    cyan="#22d3ee",
)


# ---------------------------------------------------------------------------
# Highlighting rule definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _HighlightRule:
    """A compiled regex pattern paired with a *format builder* key.

    *whole_line* – when ``True`` the format is applied to the entire block
    rather than to each individual match.
    """

    pattern: re.Pattern[str]
    format_key: str
    whole_line: bool = False


# Pre-compiled patterns (module-level for efficiency).
_ERROR_PATTERN = re.compile(
    r"\[ERROR\]|(?<!\w)(?:error|Error|Exception|Traceback|failed)(?!\w)",
)
_WARNING_PATTERN = re.compile(
    r"\[WARNING\]|(?<!\w)(?:warning|Warning)(?!\w)",
)
_SUCCESS_PATTERN = re.compile(
    r"\[INFO\]|(?<!\w)(?:done|Done|Completed)(?!\w)",
)
_SEPARATOR_PATTERN = re.compile(r"^={3,}.*", re.MULTILINE)
_FILE_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:\\[^\s:*?\"<>|]+)|(?:(?<!\d)/[\w.-]+(?:/[\w.-]+)+)",
)
_JSON_KEY_PATTERN = re.compile(r'"[^"]+"\s*:')


def _build_rules() -> list[_HighlightRule]:
    """Return the ordered list of highlighting rules.

    Rules are checked top-to-bottom; for *whole_line* rules the first match
    wins and no further whole-line rules are tested for that block.
    """
    return [
        # Whole-line colouring by log level / separator ────────────────
        _HighlightRule(_SEPARATOR_PATTERN, "separator", whole_line=True),
        _HighlightRule(_ERROR_PATTERN, "error", whole_line=True),
        _HighlightRule(_WARNING_PATTERN, "warning", whole_line=True),
        _HighlightRule(_SUCCESS_PATTERN, "success", whole_line=True),
        # Inline patterns (applied on top of the base line colour) ─────
        _HighlightRule(_FILE_PATH_PATTERN, "path"),
        _HighlightRule(_JSON_KEY_PATTERN, "json_key"),
    ]


# ---------------------------------------------------------------------------
# QSyntaxHighlighter subclass
# ---------------------------------------------------------------------------

class LogHighlighter(QSyntaxHighlighter):
    """Applies colour formatting to log output in a ``QTextEdit``.

    Parameters
    ----------
    document:
        The ``QTextDocument`` to highlight.
    dark:
        ``True`` when the dark theme is active; determines the initial
        colour palette.
    """

    def __init__(self, document: QTextDocument, *, dark: bool = False) -> None:
        super().__init__(document)
        self._rules: list[_HighlightRule] = _build_rules()
        self._formats: dict[str, QTextCharFormat] = {}
        self._build_formats(dark)

    # -- public API --------------------------------------------------------

    def update_theme(self, dark: bool) -> None:
        """Rebuild formats for *dark* or light palette and re-highlight."""
        self._build_formats(dark)
        self.rehighlight()

    # -- QSyntaxHighlighter override ---------------------------------------

    def highlightBlock(self, text: str) -> None:  # noqa: N802  (Qt naming)
        """Apply formatting rules to a single text block."""
        if not text:
            return

        whole_line_applied = False

        for rule in self._rules:
            fmt = self._formats[rule.format_key]

            if rule.whole_line:
                if whole_line_applied:
                    continue
                if rule.pattern.search(text):
                    self.setFormat(0, len(text), fmt)
                    whole_line_applied = True
            else:
                for match in rule.pattern.finditer(text):
                    self.setFormat(match.start(), match.end() - match.start(), fmt)

    # -- internal ----------------------------------------------------------

    def _build_formats(self, dark: bool) -> None:
        """Populate ``_formats`` from the chosen colour palette."""
        colors = _DARK_COLORS if dark else _LIGHT_COLORS

        self._formats = {
            "error": self._make_format(colors.red),
            "warning": self._make_format(colors.amber),
            "success": self._make_format(colors.green),
            "separator": self._make_format(colors.blue, bold=True),
            "path": self._make_format(underline=True),
            "json_key": self._make_format(colors.cyan),
        }

    @staticmethod
    def _make_format(
        color: str | None = None,
        *,
        bold: bool = False,
        underline: bool = False,
    ) -> QTextCharFormat:
        """Create a ``QTextCharFormat`` with the given visual properties."""
        fmt = QTextCharFormat()
        if color is not None:
            fmt.setForeground(QColor(color))
        if bold:
            fmt.setFontWeight(700)
        if underline:
            fmt.setFontUnderline(True)
        return fmt
