"""Lightweight batch manifest readers for the GUI shell.

Full batch manifests can embed very large ``chunks`` arrays (tens of MB).
The GUI only needs header / summary / writeback metadata, so we read the
manifest head and tail instead of parsing the entire JSON object.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_CHUNKS_MARKER_RE = re.compile(r'"chunks"\s*:\s*\[')
_CHUNKS_MARKER_BYTES = b'"chunks"'
_HEAD_READ_BYTES = 262_144
_TAIL_READ_BYTES = 524_288

_HEAD_SCALAR_KEYS = (
    "mode",
    "base_dir",
    "display_name",
    "job_name",
    "job_state",
    "job_error",
    "result_jsonl_path",
    "retry_of_manifest",
)
_HEAD_INT_KEYS = ("split_index", "split_total")

_TAIL_OBJECT_KEYS = (
    "last_check_summary",
    "apply_summary",
    "split_children",
    "split_from_manifest",
)
_TAIL_SCALAR_KEYS = (
    "applied_at",
    "next_split_manifest_path",
    "last_apply_failure_report_path",
    "last_check_report_path",
    "retry_of_manifest",
    "last_retry_manifest_path",
    "last_check_at",
    "last_check",
)

# Manifests larger than this are treated as "heavy" for GUI resume/summary reads.
HEAVY_MANIFEST_BYTE_THRESHOLD = 256_000


def manifest_should_use_lite_reader(path: str | Path) -> bool:
    try:
        return Path(path).stat().st_size >= HEAVY_MANIFEST_BYTE_THRESHOLD
    except OSError:
        return False


def read_manifest_index_fields(path: str | Path) -> dict[str, Any]:
    """Read only the fields needed to index manifests by mode and project."""
    try:
        with Path(path).open("rb") as handle:
            head = handle.read(8192).decode("utf-8-sig", errors="replace")
    except OSError:
        return {}

    result: dict[str, Any] = {}
    for key in ("mode", "base_dir"):
        value = _extract_quoted_field(head, key)
        if value is not None:
            result[key] = value
    return result


def read_manifest_lite(path: str | Path) -> dict[str, Any]:
    """Load manifest metadata without parsing the heavy ``chunks`` array."""
    manifest_path = Path(path)
    try:
        size = manifest_path.stat().st_size
    except OSError:
        return {}
    if size <= 0:
        return {}

    if size < HEAVY_MANIFEST_BYTE_THRESHOLD:
        try:
            raw = manifest_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return {}
        return _read_json_object(raw)

    marker_offset = _locate_chunks_marker(manifest_path)
    try:
        with manifest_path.open("rb") as handle:
            if marker_offset is not None:
                head_limit = marker_offset
                head = handle.read(head_limit).decode("utf-8-sig", errors="replace")
                # Tail metadata lives after the huge ``chunks`` array near EOF.
                # Never read from the marker through EOF; that would decode the
                # entire chunks payload and stall the GUI for multi-MB manifests.
                if size > _TAIL_READ_BYTES:
                    handle.seek(size - _TAIL_READ_BYTES)
                    tail = handle.read().decode("utf-8-sig", errors="replace")
                elif head_limit < size:
                    handle.seek(head_limit)
                    tail = handle.read().decode("utf-8-sig", errors="replace")
                else:
                    tail = head
            else:
                head_limit = min(_HEAD_READ_BYTES, size)
                head = handle.read(head_limit).decode("utf-8-sig", errors="replace")
                if size > _TAIL_READ_BYTES:
                    handle.seek(size - _TAIL_READ_BYTES)
                    tail = handle.read().decode("utf-8-sig", errors="replace")
                elif head_limit < size:
                    tail = handle.read().decode("utf-8-sig", errors="replace")
                else:
                    tail = head
    except (OSError, UnicodeDecodeError):
        return {}

    result: dict[str, Any] = {}
    if marker_offset is not None:
        marker = _CHUNKS_MARKER_RE.search(head)
        if marker is not None:
            head_part = head[: marker.start()].rstrip().rstrip(",")
            head_obj = _read_json_object(head_part + "}")
            if head_obj:
                result.update(head_obj)
        else:
            result.update(_extract_sparse_head_fields(head))
    elif '"chunks"' not in head and '"chunks"' not in tail:
        head_obj = _read_json_object(head.rstrip().rstrip(","))
        if head_obj:
            result.update(head_obj)
        else:
            result.update(_extract_sparse_head_fields(head))
    else:
        result.update(_extract_sparse_head_fields(head))

    result.update(_extract_tail_fields(tail))
    result.pop("chunks", None)
    return result


def _locate_chunks_marker(path: Path) -> int | None:
    with path.open("rb") as handle:
        offset = 0
        carry = b""
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return None
            haystack = carry + block
            idx = haystack.find(_CHUNKS_MARKER_BYTES)
            if idx >= 0:
                return offset - len(carry) + idx
            carry = haystack[-(len(_CHUNKS_MARKER_BYTES) - 1) :] if len(_CHUNKS_MARKER_BYTES) > 1 else b""
            offset += len(block)


def _extract_sparse_head_fields(head: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _HEAD_SCALAR_KEYS:
        value = _extract_quoted_field(head, key)
        if value is not None:
            result[key] = value
    for key in _HEAD_INT_KEYS:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(\d+)', head)
        if match:
            result[key] = int(match.group(1))
    summary = _extract_json_value_after_key(head, "summary")
    if isinstance(summary, dict):
        result["summary"] = summary
    return result


def _extract_tail_fields(tail: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _TAIL_OBJECT_KEYS:
        value = _extract_json_value_after_key(tail, key)
        if value is not None:
            result[key] = value
    for key in _TAIL_SCALAR_KEYS:
        if key in result:
            continue
        value = _extract_scalar_field(tail, key)
        if value is not None:
            result[key] = value
    return result


def _extract_scalar_field(text: str, key: str) -> Any | None:
    match = re.search(
        rf'"{re.escape(key)}"\s*:\s*("((?:\\.|[^"\\])*)"|(true|false|null))',
        text,
    )
    if not match:
        return None
    if match.group(2) is not None:
        return _decode_json_string(match.group(2))
    raw = match.group(3)
    if raw == "true":
        return True
    if raw == "false":
        return False
    return None


def _extract_quoted_field(text: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    if not match:
        return None
    return _decode_json_string(match.group(1))


def _extract_json_value_after_key(text: str, key: str) -> Any | None:
    marker = re.search(rf'"{re.escape(key)}"\s*:\s*', text)
    if not marker:
        return None
    index = marker.end()
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return None

    opener = text[index]
    if opener == "{":
        end = _skip_balanced(text, index, "{", "}")
    elif opener == "[":
        end = _skip_balanced(text, index, "[", "]")
    elif opener == '"':
        end = _skip_json_string(text, index) + 1
    else:
        end_match = re.match(r"[^,\n}]+", text[index:])
        if not end_match:
            return None
        end = index + end_match.end()

    snippet = text[index:end]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _decode_json_string(raw: str) -> str:
    try:
        decoded = json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw
    return decoded if isinstance(decoded, str) else raw


def _read_json_object(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _skip_balanced(text: str, start: int, open_char: str, close_char: str) -> int:
    depth = 0
    index = start
    length = len(text)
    while index < length:
        char = text[index]
        if char == '"':
            index = _skip_json_string(text, index)
            index += 1
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return length


def _skip_json_string(text: str, start: int) -> int:
    index = start + 1
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index
        index += 1
    return length