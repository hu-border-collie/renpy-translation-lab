"""Strict MV escape tokens and the opt-in GALV_MessageStyles 2.8 subset.

This is a mechanical preservation policy, not a JavaScript interpreter. Unknown
escapes are rejected, including extensions to otherwise recognized commands.
The core's backslash followed by one ASCII space is preserved as an exact
escape token; it is never normalized away or inferred from other whitespace.
Popup targets support the numeric form used by the frozen first window; actor
targets and screen coordinates require a separately evidenced extension.
"""

from __future__ import annotations

import re

from .rpgmaker_mv_fields import DISPLAY_ROLES


_ARGUMENT = re.compile(r"\\[VvNnPpIiCc]\[[0-9]+\]")
_POPUP = re.compile(r"\\pop\[-?[0-9]+\]")
_SINGLE = frozenset("Gg$.|!><^{}")
_FORMAT = re.compile(r"%[0-9]+")


def control_spans(text: str, *, message_styles: bool = False,
                  text_role: str = "") -> list[tuple[int, int, str]]:
    """Preserve core controls and numbered display parameters with exact spelling.

    Numbered parameters may move with language order. Display roles never use
    the narrative popup extension, even when that plugin is approved globally.
    An absent role retains the legacy narrative rules.
    """
    if text_role and text_role not in DISPLAY_ROLES:
        raise ValueError("rpgmaker_mv.control.role")
    result: list[tuple[int, int, str]] = []
    index = 0
    while index < len(text):
        start = index
        character = text[index]
        if character in "\r\n\t":
            index += 2 if text.startswith("\r\n", index) else 1
            result.append((start, index, "boundary"))
        elif character == "\\":
            if text.startswith("\\\\", index):
                index += 2
                kind = "escape"
            elif text.startswith("\\ ", index):
                index += 2
                kind = "escape"
            elif match := _ARGUMENT.match(text, index):
                index = match.end()
                kind = "control"
            elif message_styles and not text_role and (match := _POPUP.match(text, index)):
                index = match.end()
                kind = "popup"
            elif index + 1 < len(text) and text[index + 1] in _SINGLE:
                # Alphabetic escape names are a whole token in MV. Do not
                # mistake an unknown \\Gold extension for the currency code.
                end = index + 2
                if (text[index + 1] in "Gg" and end < len(text)
                        and text[end] in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"):
                    raise ValueError("rpgmaker_mv.control.unknown")
                index = end
                kind = "control"
            else:
                raise ValueError("rpgmaker_mv.control.unknown")
            result.append((start, index, kind))
        elif text_role and (match := _FORMAT.match(text, index)):
            index = match.end()
            result.append((start, index, "variable"))
        else:
            if ord(character) < 32 or ord(character) == 127:
                raise ValueError("rpgmaker_mv.control.character")
            index += 1
    return result


def validate_message_position(text: str, *, message_styles: bool, first_line: bool,
                              text_role: str = "") -> None:
    """Keep physical lines and popup ownership; display fields forbid popups."""
    spans = control_spans(text, message_styles=message_styles, text_role=text_role)
    popups = [(start, end) for start, end, kind in spans if kind == "popup"]
    if popups and (not first_line or len(popups) != 1 or popups[0][0] != 0):
        raise ValueError("rpgmaker_mv.control.popup_position")
    if any(kind == "boundary" for _, _, kind in spans):
        raise ValueError("rpgmaker_mv.control.physical_line")
