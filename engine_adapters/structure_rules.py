"""Versioned structural spans for the two supported catalog engines.

Offsets refer to decoded canonical strings, never script byte offsets. These
rules deliberately do not infer a third engine from punctuation.
"""

import re
from collections import Counter

PERCENT = re.compile(
    r"%(?:\([^)]+\))?[#0 +\-]*(?:\d+|\*)?(?:\.\d+|\.\*)?[hlL]?"
    r"[diouxXeEfFgGcrsa](?![A-Za-z])"
)
PAIRED = frozenset('a alpha b color cps font i k outlinecolor plain s size u'.split())


def spans(text, engine):
    """Return non-overlapping (start, end, kind) spans under explicit rules."""
    if engine not in ('renpy', 'tyrano'):
        raise ValueError('protection.unsupported_engine')
    result = []
    index = 0
    while index < len(text):
        start = index
        kind = ''
        if text[index] in '\r\n\t':
            index += 2 if text.startswith('\r\n', index) else 1
            kind = 'boundary'
        elif text[index:index + 2] in ('\\n', '\\r', '\\t', '\\\\'):
            index += 2
            kind = 'escape'
        elif engine == 'renpy' and text[index:index + 2] in ('[[', '{{', '%%'):
            index += 2
            kind = 'escape'
        elif text[index] == '[':
            # Ren'Py expressions may contain indexed expressions and strings;
            # Tyrano tag attributes may contain quoted closing brackets.
            depth, quote, escaped = 1, '', False
            index += 1
            while index < len(text) and depth:
                char = text[index]
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif quote:
                    if char == quote:
                        quote = ''
                elif char in ('"', "'"):
                    quote = char
                elif char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1
                index += 1
            if depth:
                raise ValueError('protection.invalid_structure')
            kind = 'variable' if engine == 'renpy' else 'control'
        elif engine == 'renpy' and text[index] == '{':
            end = text.find('}', index + 1)
            if end < 0:
                raise ValueError('protection.invalid_structure')
            index = end + 1
            kind = 'tag'
        elif engine == 'renpy' and (match := PERCENT.match(text, index)):
            index = match.end()
            kind = 'variable' if match.group().startswith('%(') else 'format'
        else:
            index += 1
        if kind:
            result.append((start, index, kind))
    return result


def validate(text, translated, engine):
    """Check exact token multiplicity, structural order and tag nesting.

    Named variables may move with language order. Positional formats, engine
    commands, escapes, tags and line boundaries retain their relative order.
    """
    original = [(text[a:b], kind) for a, b, kind in spans(text, engine)]
    current = [(translated[a:b], kind) for a, b, kind in spans(translated, engine)]
    if Counter(original) != Counter(current):
        raise ValueError('protection.structure_changed')
    if [x for x in original if x[1] != 'variable'] != [x for x in current if x[1] != 'variable']:
        raise ValueError('protection.structure_order')
    if engine == 'renpy':
        stack = []
        closing_names = {value[2:-1] for value, kind in current
                         if kind == 'tag' and value.startswith('{/')}
        for value, kind in current:
            if kind != 'tag':
                continue
            name = value[1:-1].split('=', 1)[0]
            if name.startswith('/'):
                if not stack or stack.pop() != name[1:]:
                    raise ValueError('protection.invalid_structure')
            elif name in PAIRED or name in closing_names:
                stack.append(name)
        if stack:
            raise ValueError('protection.invalid_structure')
