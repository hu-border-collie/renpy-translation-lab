"""Read-only Ren'Py font glyph coverage spike (#487).

This module parses static ``font`` references and TrueType/OpenType ``cmap``
tables without executing game Python, starting Ren'Py, or writing to the
project.  It is deliberately a bounded spike: dynamic expressions, FontGroup
fallback chains and runtime style overrides are reported as ``unknown``
instead of being guessed.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SPIKE_SCHEMA_VERSION = 1
MAX_FONT_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_MISSING_CHARS = 200

STATUS_CHECKED = "checked"
STATUS_MISSING = "missing"
STATUS_UNAVAILABLE = "unavailable"
STATUS_UNKNOWN = "unknown"

REASON_FONT_FILE_MISSING = "font.file_missing"
REASON_FONT_INVALID = "font.invalid"
REASON_FONT_CMAP_UNSUPPORTED = "font.cmap_unsupported"
REASON_FONT_GLYPH_MISSING = "font.glyph_missing"
REASON_FONT_DYNAMIC_EXPRESSION = "font.dynamic_expression"
REASON_FONT_GROUP_UNSUPPORTED = "font.group_unsupported"
REASON_FONT_NOT_DECLARED = "font.not_declared"
REASON_FONT_EXPRESSION_UNPARSED = "font.expression_unparsed"
REASON_FONT_PATH_OUTSIDE_GAME = "font.path_outside_game"
REASON_TEXT_NO_SAMPLES = "text.no_samples"
REASON_FONT_ENGINE_SEARCH_PATH = "font.engine_search_path_unknown"
REASON_FONT_TOO_LARGE = "font.too_large"

_REASON_TEXT = {
    REASON_FONT_FILE_MISSING: "字体文件不存在",
    REASON_FONT_INVALID: "字体文件无法解析",
    REASON_FONT_CMAP_UNSUPPORTED: "字体缺少受支持的 cmap 子表",
    REASON_FONT_GLYPH_MISSING: "字体缺少目标字符",
    REASON_FONT_DYNAMIC_EXPRESSION: "字体引用是动态表达式，运行时才能确定",
    REASON_FONT_GROUP_UNSUPPORTED: "FontGroup / fallback 链无法在只读扫描中确定",
    REASON_FONT_NOT_DECLARED: "未找到静态字体声明",
    REASON_FONT_EXPRESSION_UNPARSED: "字体表达式无法安全解析",
    REASON_FONT_PATH_OUTSIDE_GAME: "字体路径不在 game_root 内，已按输入边界拒绝读取",
    REASON_TEXT_NO_SAMPLES: "没有可检查的文本样本，无法判定字形覆盖",
    REASON_FONT_ENGINE_SEARCH_PATH: "字体名可能由 Ren'Py 引擎/搜索路径解析，只读扫描无法确定",
    REASON_FONT_TOO_LARGE: "字体文件超过只读检查体积上限",
}

_STYLE_HEADER_RE = re.compile(
    r"^style\s+([A-Za-z_][\w.]*)(?:\s+is\s+[A-Za-z_][\w.]*)?\s*:\s*$"
)
_TRANSLATE_STYLE_HEADER_RE = re.compile(
    r"^translate\s+[A-Za-z0-9_.-]+\s+style\s+"
    r"([A-Za-z_][\w.]*)(?:\s+is\s+[A-Za-z_][\w.]*)?\s*:\s*$"
)
_FONT_PROPERTY_RE = re.compile(r"^font\s+(.+?)\s*$")
_GUI_FONT_ASSIGN_RE = re.compile(
    r"^(?:define\s+)?(gui\.[A-Za-z_]\w*font)\s*=\s*(.+?)\s*$"
)
_RENPY_TAG_RE = re.compile(r"\{[^{}]*\}")
_RENPY_SUBST_RE = re.compile(r"(?<!\[)\[[^\[\]]*\]")


class FontCoverageError(ValueError):
    """Stable read-only failure for one font/input boundary."""

    def __init__(self, code: str, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        self.code = str(code)
        self.detail = str(detail or "")


@dataclass(frozen=True)
class CmapRange:
    """One contiguous codepoint mapping range."""

    start: int
    end: int
    delta: int = 0
    glyph_ids: tuple[int, ...] | None = None

    def glyph_id(self, codepoint: int) -> int | None:
        if codepoint < self.start or codepoint > self.end:
            return None
        if self.glyph_ids is not None:
            index = codepoint - self.start
            if index < 0 or index >= len(self.glyph_ids):
                return None
            glyph_id = self.glyph_ids[index]
            return glyph_id or None
        glyph_id = (codepoint + self.delta) & 0xFFFF
        return glyph_id or None


@dataclass(frozen=True)
class CmapIndex:
    """Parsed cmap coverage, preserving ranges for bounded lookup."""

    format: int
    ranges: tuple[CmapRange, ...]
    codepoint_count: int
    subtable_count: int = 1

    def glyph_id(self, codepoint: int) -> int | None:
        for item in self.ranges:
            glyph_id = item.glyph_id(codepoint)
            if glyph_id is not None:
                return glyph_id
        return None


@dataclass(frozen=True)
class FontFace:
    path: str
    cmap_format: int
    glyph_count: int
    family_name: str
    cmap: CmapIndex

    def covers(self, character: str) -> bool:
        if not character:
            return False
        return self.cmap.glyph_id(ord(character)) is not None


@dataclass(frozen=True)
class FontReference:
    script: str
    line: int
    style: str
    expression: str
    kind: str
    path: str = ""
    reason: str = ""


@dataclass(frozen=True)
class TextSamples:
    samples: tuple[str, ...]
    sources: tuple[str, ...]
    dynamic_sample_count: int


@dataclass(frozen=True)
class FontCheck:
    reference: FontReference
    status: str
    reason: str
    font_path: str = ""
    cmap_format: int = 0
    cmap_subtables: int = 0
    glyph_count: int = 0
    family_name: str = ""
    missing_chars: tuple[str, ...] = ()
    missing_char_count: int = 0
    evidence: str = ""


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.name


def _sfnt_offset(data: bytes) -> int:
    if len(data) < 12:
        raise FontCoverageError(REASON_FONT_INVALID, "字体文件过短")
    if data[:4] == b"ttcf":
        if len(data) < 16:
            raise FontCoverageError(REASON_FONT_INVALID, "TTC 字体头不完整")
        count = struct.unpack(">I", data[8:12])[0]
        if count < 1:
            raise FontCoverageError(REASON_FONT_INVALID, "TTC 字体集合为空")
        return struct.unpack(">I", data[12:16])[0]
    return 0


def _read_tables(data: bytes, offset: int) -> dict[str, tuple[int, int]]:
    if offset + 12 > len(data):
        raise FontCoverageError(REASON_FONT_INVALID, "字体表目录越界")
    sfnt_version = data[offset:offset + 4]
    if sfnt_version not in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1"):
        raise FontCoverageError(REASON_FONT_INVALID, "不支持的 sfnt 版本")
    num_tables = struct.unpack(">H", data[offset + 4:offset + 6])[0]
    if num_tables < 1 or offset + 12 + num_tables * 16 > len(data):
        raise FontCoverageError(REASON_FONT_INVALID, "字体表数量异常")
    tables: dict[str, tuple[int, int]] = {}
    for index in range(num_tables):
        record = offset + 12 + index * 16
        tag = data[record:record + 4].decode("latin-1", errors="replace")
        table_offset, table_length = struct.unpack(
            ">II", data[record + 8:record + 16]
        )
        if table_offset + table_length > len(data):
            raise FontCoverageError(REASON_FONT_INVALID, f"字体表 {tag} 越界")
        tables[tag] = (table_offset, table_length)
    return tables


def _parse_format_4(data: bytes, offset: int, length: int) -> CmapIndex:
    if length < 16:
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap format 4 过短")
    sub = data[offset:offset + length]
    seg_count = struct.unpack(">H", sub[6:8])[0] // 2
    if seg_count < 1:
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap format 4 无 segment")
    end_offset = 14
    start_offset = end_offset + 2 * seg_count + 2
    delta_offset = start_offset + 2 * seg_count
    range_offset = delta_offset + 2 * seg_count
    if range_offset + 2 * seg_count > len(sub):
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap format 4 数组越界")
    ends = struct.unpack(f">{seg_count}H", sub[end_offset:end_offset + 2 * seg_count])
    starts = struct.unpack(f">{seg_count}H", sub[start_offset:start_offset + 2 * seg_count])
    deltas = struct.unpack(f">{seg_count}h", sub[delta_offset:delta_offset + 2 * seg_count])
    range_offsets = struct.unpack(
        f">{seg_count}H", sub[range_offset:range_offset + 2 * seg_count]
    )
    ranges: list[CmapRange] = []
    codepoint_count = 0
    for index in range(seg_count):
        start, end = starts[index], ends[index]
        if start > end or (start == 0xFFFF and end == 0xFFFF):
            continue
        if range_offsets[index] == 0:
            delta = deltas[index]
            if (start + delta) & 0xFFFF == 0 and start == end:
                continue
            ranges.append(CmapRange(start=start, end=end, delta=delta))
            codepoint_count += end - start + 1
            continue
        glyph_offset = range_offset + 2 * index + range_offsets[index]
        if glyph_offset + 2 * (end - start + 1) > len(sub):
            raise FontCoverageError(
                REASON_FONT_CMAP_UNSUPPORTED, "cmap format 4 glyph 数组越界"
            )
        glyph_ids: list[int] = []
        any_glyph = False
        for codepoint in range(start, end + 1):
            glyph_id = struct.unpack(
                ">H", sub[glyph_offset + 2 * (codepoint - start):
                           glyph_offset + 2 * (codepoint - start) + 2]
            )[0]
            if glyph_id:
                glyph_id = (glyph_id + deltas[index]) & 0xFFFF
                any_glyph = True
            glyph_ids.append(glyph_id)
        if any_glyph:
            ranges.append(CmapRange(start=start, end=end, glyph_ids=tuple(glyph_ids)))
            codepoint_count += end - start + 1
    if not ranges:
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap format 4 无可映射字符")
    return CmapIndex(format=4, ranges=tuple(ranges), codepoint_count=codepoint_count)


def _parse_format_12(data: bytes, offset: int, length: int) -> CmapIndex:
    if length < 16:
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap format 12 过短")
    sub = data[offset:offset + length]
    group_count = struct.unpack(">I", sub[12:16])[0]
    if group_count < 1 or 16 + group_count * 12 > len(sub):
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap format 12 数组越界")
    ranges: list[CmapRange] = []
    codepoint_count = 0
    for index in range(group_count):
        record = 16 + index * 12
        start, end, start_gid = struct.unpack(">III", sub[record:record + 12])
        if start > end:
            continue
        # startGlyphID == 0 only means the first codepoint maps to .notdef;
        # following codepoints still map to glyphs 1..N.  Skip only the
        # single-codepoint group that can never produce a real glyph.
        if start_gid == 0 and start == end:
            continue
        ranges.append(
            CmapRange(start=start, end=end, delta=start_gid - start)
        )
        codepoint_count += end - start + 1
    if not ranges:
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap format 12 无可映射字符")
    return CmapIndex(format=12, ranges=tuple(ranges), codepoint_count=codepoint_count)


def _parse_format_6(data: bytes, offset: int, length: int) -> CmapIndex:
    if length < 10:
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap format 6 过短")
    sub = data[offset:offset + length]
    first_code, entry_count = struct.unpack(">HH", sub[6:10])
    if entry_count < 1 or 10 + entry_count * 2 > len(sub):
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap format 6 数组越界")
    glyph_ids = struct.unpack(
        f">{entry_count}H", sub[10:10 + entry_count * 2]
    )
    return CmapIndex(
        format=6,
        ranges=(
            CmapRange(
                start=first_code,
                end=first_code + entry_count - 1,
                glyph_ids=tuple(glyph_ids),
            ),
        ),
        codepoint_count=entry_count,
    )


def _parse_format_0(data: bytes, offset: int, length: int) -> CmapIndex:
    if length < 262:
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap format 0 过短")
    sub = data[offset:offset + length]
    if len(sub) < 262:
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap format 0 数据截断")
    glyph_ids = tuple(sub[6:262])
    if not any(glyph_ids):
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap format 0 无可映射字符")
    return CmapIndex(
        format=0,
        ranges=(CmapRange(start=0, end=255, glyph_ids=glyph_ids),),
        codepoint_count=sum(1 for item in glyph_ids if item),
    )


_CMAP_FORMAT_PARSERS = {
    0: _parse_format_0,
    4: _parse_format_4,
    6: _parse_format_6,
    12: _parse_format_12,
}


def _cmap_score(platform_id: int, encoding_id: int, fmt: int) -> tuple[int, int, int]:
    format_rank = {12: 4, 4: 3, 6: 2, 0: 1}.get(fmt, 0)
    platform_rank = 3 if platform_id == 3 and encoding_id in {1, 10} else (
        2 if platform_id == 0 else 1
    )
    encoding_rank = 2 if encoding_id == 10 else (1 if encoding_id in {0, 1} else 0)
    return format_rank, platform_rank, encoding_rank


def _parse_cmap(data: bytes, offset: int, length: int) -> CmapIndex:
    if length < 4:
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap 表过短")
    sub = data[offset:offset + length]
    num_tables = struct.unpack(">H", sub[2:4])[0]
    if num_tables < 1 or 4 + num_tables * 8 > len(sub):
        raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "cmap 子表数量异常")
    candidates: list[tuple[tuple[int, int, int], int, int, int, int]] = []
    for index in range(num_tables):
        record = 4 + index * 8
        platform_id, encoding_id, sub_offset = struct.unpack(
            ">HHI", sub[record:record + 8]
        )
        if sub_offset + 4 > len(sub):
            continue
        # Only Unicode cmaps participate in glyph coverage: platform 0
        # (Unicode) and Windows BMP/UCS-4. Macintosh/symbol encodings are not
        # Unicode and must not create false "covered" answers.
        if platform_id != 0 and not (
            platform_id == 3 and encoding_id in {1, 10}
        ):
            continue
        fmt = struct.unpack(">H", sub[sub_offset:sub_offset + 2])[0]
        if fmt not in _CMAP_FORMAT_PARSERS:
            continue
        candidates.append(
            (
                _cmap_score(platform_id, encoding_id, fmt),
                fmt,
                sub_offset,
                platform_id,
                encoding_id,
            )
        )
    if not candidates:
        raise FontCoverageError(
            REASON_FONT_CMAP_UNSUPPORTED, "字体没有受支持的 Unicode cmap 子表"
        )
    errors: list[Exception] = []
    parsed: list[CmapIndex] = []
    for _score, fmt, sub_offset, _platform, _encoding in sorted(
        candidates, reverse=True
    ):
        try:
            if fmt in {8, 10, 12, 13}:
                sub_length = struct.unpack(
                    ">I", sub[sub_offset + 4:sub_offset + 8]
                )[0]
            else:
                sub_length = struct.unpack(
                    ">H", sub[sub_offset + 2:sub_offset + 4]
                )[0]
            parsed.append(
                _CMAP_FORMAT_PARSERS[fmt](sub, sub_offset, sub_length)
            )
        except FontCoverageError as exc:
            errors.append(exc)
        except (struct.error, IndexError, ValueError) as exc:
            errors.append(
                FontCoverageError(
                    REASON_FONT_CMAP_UNSUPPORTED,
                    "cmap 子表数据截断或越界",
                )
            )
    if not parsed:
        if errors:
            raise errors[0]
        raise FontCoverageError(
            REASON_FONT_CMAP_UNSUPPORTED, "字体 cmap 子表解析失败"
        )
    # Keep every successfully parsed Unicode subtable: a sparse high-priority
    # subtable (for example format 12 for supplementary planes) must not hide
    # BMP coverage that only exists in a lower-priority subtable.
    ranges = tuple(
        item for parsed_index in parsed for item in parsed_index.ranges
    )
    return CmapIndex(
        format=parsed[0].format,
        ranges=ranges,
        codepoint_count=sum(item.codepoint_count for item in parsed),
        subtable_count=len(parsed),
    )


def _glyph_count(data: bytes, table: tuple[int, int] | None) -> int:
    if table is None:
        return 0
    offset, length = table
    if length < 6:
        return 0
    return struct.unpack(">H", data[offset + 4:offset + 6])[0]


def _family_name(data: bytes, table: tuple[int, int] | None) -> str:
    if table is None:
        return ""
    offset, length = table
    if length < 6:
        return ""
    sub = data[offset:offset + length]
    count, string_offset = struct.unpack(">HH", sub[2:6])
    if 6 + count * 12 > len(sub):
        return ""
    for index in range(count):
        record = 6 + index * 12
        platform_id, encoding_id, _language, name_id, name_length, name_offset = (
            struct.unpack(">HHHHHH", sub[record:record + 12])
        )
        if name_id not in {1, 4, 6}:
            continue
        start = string_offset + name_offset
        end = start + name_length
        if start < 0 or end > len(sub):
            continue
        raw = sub[start:end]
        if platform_id in {0, 3}:
            try:
                text = raw.decode("utf-16-be")
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("latin-1", errors="replace")
        text = text.strip()
        if text:
            return text
    return ""


def load_font_face(path: str | Path) -> FontFace:
    """Parse one font file and return its cmap coverage index."""

    font_path = Path(path)
    if not font_path.is_file():
        raise FontCoverageError(
            REASON_FONT_FILE_MISSING,
            f"字体文件不存在：{font_path.name}",
            detail=str(font_path),
        )
    try:
        size = font_path.stat().st_size
    except OSError as exc:
        raise FontCoverageError(REASON_FONT_INVALID, "无法读取字体文件") from exc
    if size <= 0:
        raise FontCoverageError(REASON_FONT_INVALID, "字体文件大小异常")
    if size > MAX_FONT_BYTES:
        raise FontCoverageError(
            REASON_FONT_TOO_LARGE,
            f"字体文件超过只读检查上限（{MAX_FONT_BYTES} bytes）",
        )
    try:
        data = font_path.read_bytes()
        offset = _sfnt_offset(data)
        tables = _read_tables(data, offset)
        cmap_table = tables.get("cmap")
        if cmap_table is None:
            raise FontCoverageError(REASON_FONT_CMAP_UNSUPPORTED, "字体缺少 cmap 表")
        cmap = _parse_cmap(data, cmap_table[0], cmap_table[1])
        glyph_count = _glyph_count(data, tables.get("maxp"))
        family_name = _family_name(data, tables.get("name"))
    except FontCoverageError:
        raise
    except (OSError, ValueError, struct.error, UnicodeDecodeError) as exc:
        raise FontCoverageError(REASON_FONT_INVALID, "字体解析失败") from exc
    return FontFace(
        path=str(font_path),
        cmap_format=cmap.format,
        glyph_count=glyph_count,
        family_name=family_name,
        cmap=cmap,
    )


def _strip_comment(line: str) -> str:
    quote = ""
    for index, character in enumerate(line):
        if quote:
            if character == quote:
                quote = ""
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == "#":
            return line[:index]
    return line


def _literal_expression_value(text: str) -> str | None:
    """Return the value only when *text* is exactly one string literal."""

    if len(text) < 2 or text[0] not in {'"', "'"}:
        return None
    quote = text[0]
    value: list[str] = []
    index = 1
    while index < len(text):
        character = text[index]
        if character == "\\":
            if index + 1 >= len(text):
                return None
            value.append(text[index + 1])
            index += 2
            continue
        if character == quote:
            if text[index + 1:].strip():
                return None
            return "".join(value)
        value.append(character)
        index += 1
    return None


def _classify_font_expression(expression: str) -> tuple[str, str, str]:
    text = str(expression or "").strip()
    if not text:
        return "unparsed", "", REASON_FONT_EXPRESSION_UNPARSED
    literal = _literal_expression_value(text)
    if literal is not None:
        if not literal.strip():
            return "none", "", REASON_FONT_NOT_DECLARED
        return "static", literal, ""
    if "FontGroup" in text:
        return "group", "", REASON_FONT_GROUP_UNSUPPORTED
    if text.lower() == "none":
        return "none", "", REASON_FONT_NOT_DECLARED
    if any(marker in text for marker in ('"', "'", "+", "%", "[", "(", ".")):
        return "dynamic", "", REASON_FONT_DYNAMIC_EXPRESSION
    if re.search(r"[A-Za-z_]", text):
        return "dynamic", "", REASON_FONT_DYNAMIC_EXPRESSION
    return "unparsed", "", REASON_FONT_EXPRESSION_UNPARSED


def _resolve_font_path(game_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return game_root / candidate


def scan_font_references(game_root: str | Path) -> tuple[FontReference, ...]:
    """Scan ``.rpy`` files for static/dynamic ``font`` references."""

    root = Path(game_root)
    if not root.is_dir():
        raise FontCoverageError("input.invalid", "game_root 不是目录")
    references: list[FontReference] = []
    seen: set[tuple[str, int, str, str]] = set()
    for script_path in sorted(root.rglob("*.rpy")):
        if not script_path.is_file():
            continue
        rel_path = _relative(script_path, root)
        try:
            lines = script_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            continue
        current_style = ""
        style_indent = -1
        for line_number, raw_line in enumerate(lines, start=1):
            code = _strip_comment(raw_line.rstrip())
            stripped = code.strip()
            if not stripped:
                continue
            indent = len(code) - len(code.lstrip())
            if current_style and indent <= style_indent:
                current_style = ""
            if current_style:
                property_match = _FONT_PROPERTY_RE.match(stripped)
                if property_match:
                    expression = property_match.group(1)
                    key = (rel_path, line_number, current_style, expression)
                    if key not in seen:
                        seen.add(key)
                        kind, path, reason = _classify_font_expression(expression)
                        references.append(
                            FontReference(
                                script=rel_path,
                                line=line_number,
                                style=current_style,
                                expression=expression,
                                kind=kind,
                                path=path,
                                reason=reason,
                            )
                        )
                    continue
            header = _STYLE_HEADER_RE.match(stripped)
            if header is None:
                header = _TRANSLATE_STYLE_HEADER_RE.match(stripped)
            if header:
                current_style = header.group(1)
                style_indent = indent
                continue
            assignment = _GUI_FONT_ASSIGN_RE.match(stripped)
            if assignment:
                name, expression = assignment.group(1), assignment.group(2)
                key = (rel_path, line_number, name, expression)
                if key not in seen:
                    seen.add(key)
                    kind, path, reason = _classify_font_expression(expression)
                    references.append(
                        FontReference(
                            script=rel_path,
                            line=line_number,
                            style=name,
                            expression=expression,
                            kind=kind,
                            path=path,
                            reason=reason,
                        )
                    )
    return tuple(references)


def _unescape_renpy_string(value: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character == "\\" and index + 1 < len(value):
            escaped = value[index + 1]
            if escaped == "n":
                result.append("\n")
            elif escaped == "t":
                result.append("\t")
            elif escaped in {'\\', '"', "'"}:
                result.append(escaped)
            else:
                result.append(escaped)
            index += 2
            continue
        result.append(character)
        index += 1
    return "".join(result)


def _literal_from_line(line: str) -> str:
    match = re.search(
        r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'',
        line,
    )
    if not match:
        return ""
    value = match.group(1) if match.group(1) is not None else match.group(2)
    return _unescape_renpy_string(value)


def extract_translation_strings(tl_dir: str | Path) -> tuple[str, ...]:
    """Best-effort target-language string extraction from Ren'Py ``tl`` files.

    ``new "..."`` rows and translated dialogue rows are collected; ``old``
    rows are source text and are ignored.  Ren'Py is never executed.
    """

    root = Path(tl_dir)
    if not root.is_dir():
        return ()
    values: list[str] = []
    for script_path in sorted(root.rglob("*.rpy")):
        if not script_path.is_file():
            continue
        try:
            lines = script_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            continue
        for raw_line in lines:
            stripped = _strip_comment(raw_line).strip()
            if not stripped:
                continue
            if stripped.startswith("old "):
                continue
            if stripped.startswith("new "):
                value = _literal_from_line(stripped)
            elif re.match(r"^[A-Za-z_][\w.]*\s+[\"']", stripped):
                value = _literal_from_line(stripped)
            elif stripped.startswith(("'", '"')):
                value = _literal_from_line(stripped)
            else:
                continue
            if value:
                values.append(value)
    return tuple(values)


def normalize_renpy_text(text: str) -> tuple[str, bool]:
    """Remove tags/interpolation for coverage; flag dynamic substitution."""

    source = str(text or "")
    dynamic = bool(_RENPY_SUBST_RE.search(source))
    normalized = source.replace("[[", "\x00")
    normalized = _RENPY_TAG_RE.sub("", normalized)
    normalized = _RENPY_SUBST_RE.sub("", normalized)
    normalized = normalized.replace("\x00", "[")
    return normalized, dynamic


def collect_text_samples(
    *,
    text_file: str | Path | None = None,
    tl_dir: str | Path | None = None,
    text_values: Sequence[str] = (),
) -> TextSamples:
    """Collect and normalize explicit, file-based and ``tl`` text samples."""

    raw_samples: list[str] = []
    sources: list[str] = []
    if text_file is not None:
        path = Path(text_file)
        if path.is_file():
            raw_samples.extend(
                line
                for line in path.read_text(
                    encoding="utf-8-sig", errors="replace"
                ).splitlines()
                if line.strip()
            )
            sources.append(path.name)
    if tl_dir is not None:
        values = extract_translation_strings(tl_dir)
        raw_samples.extend(values)
        if values:
            sources.append("tl")
    raw_samples.extend(str(item) for item in text_values if str(item))
    normalized: list[str] = []
    dynamic_count = 0
    for sample in raw_samples:
        value, dynamic = normalize_renpy_text(sample)
        if not value:
            continue
        normalized.append(value)
        if dynamic:
            dynamic_count += 1
    deduped = tuple(dict.fromkeys(normalized))
    return TextSamples(
        samples=deduped,
        sources=tuple(dict.fromkeys(sources)),
        dynamic_sample_count=dynamic_count,
    )


def required_characters(samples: Sequence[str]) -> tuple[str, ...]:
    characters: set[str] = set()
    for sample in samples:
        for character in str(sample):
            if character == "\n" or character == "\r" or character == "\t":
                continue
            if not character.isprintable():
                continue
            characters.add(character)
    return tuple(sorted(characters))


def analyze_font_coverage(
    game_root: str | Path,
    *,
    tl_dir: str | Path | None = None,
    text_file: str | Path | None = None,
    text_values: Sequence[str] = (),
    max_missing_chars: int = DEFAULT_MAX_MISSING_CHARS,
) -> dict[str, Any]:
    """Return a bounded read-only coverage report for one Ren'Py game root."""

    root = Path(game_root)
    if not root.is_dir():
        raise FontCoverageError("input.invalid", "game_root 不是目录")
    references = scan_font_references(root)
    samples = collect_text_samples(
        text_file=text_file,
        tl_dir=tl_dir,
        text_values=text_values,
    )
    required = required_characters(samples.samples)
    checks: list[FontCheck] = []
    if not references:
        references = (
            FontReference(
                script="",
                line=0,
                style="",
                expression="",
                kind="none",
                reason=REASON_FONT_NOT_DECLARED,
            ),
        )
    limit = max(1, int(max_missing_chars or DEFAULT_MAX_MISSING_CHARS))
    for reference in references:
        if not required:
            checks.append(
                FontCheck(
                    reference=reference,
                    status=STATUS_UNKNOWN,
                    reason=REASON_TEXT_NO_SAMPLES,
                    evidence=(
                        f"{reference.script}:{reference.line} "
                        f"reason={REASON_TEXT_NO_SAMPLES}"
                    ),
                )
            )
            continue
        if reference.kind != "static":
            checks.append(
                FontCheck(
                    reference=reference,
                    status=STATUS_UNKNOWN,
                    reason=reference.reason or REASON_FONT_DYNAMIC_EXPRESSION,
                    evidence=(
                        f"{reference.script}:{reference.line} style={reference.style} "
                        f"expression={reference.expression!r}"
                    ),
                )
            )
            continue
        font_path = _resolve_font_path(root, reference.path)
        font_rel = _relative(font_path, root)
        try:
            font_path.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            checks.append(
                FontCheck(
                    reference=reference,
                    status=STATUS_UNKNOWN,
                    reason=REASON_FONT_PATH_OUTSIDE_GAME,
                    font_path=font_rel,
                    evidence=(
                        f"{reference.script}:{reference.line} font={font_rel} "
                        f"reason={REASON_FONT_PATH_OUTSIDE_GAME}"
                    ),
                )
            )
            continue
        try:
            face = load_font_face(font_path)
        except FontCoverageError as exc:
            # A bare file name (for example DejaVuSans.ttf) is commonly
            # resolved by Ren'Py's engine/search path rather than by
            # game_root; do not report it as a definite user typo.
            engine_search_path = (
                exc.code == REASON_FONT_FILE_MISSING
                and Path(reference.path).name == reference.path
            )
            checks.append(
                FontCheck(
                    reference=reference,
                    status=(
                        STATUS_UNKNOWN
                        if engine_search_path
                        else STATUS_UNAVAILABLE
                    ),
                    reason=(
                        REASON_FONT_ENGINE_SEARCH_PATH
                        if engine_search_path
                        else exc.code
                    ),
                    font_path=font_rel,
                    evidence=(
                        f"{reference.script}:{reference.line} font={font_rel} "
                        f"reason={REASON_FONT_ENGINE_SEARCH_PATH if engine_search_path else exc.code}"
                    ),
                )
            )
            continue
        missing = tuple(character for character in required if not face.covers(character))
        missing_count = len(missing)
        if missing:
            checks.append(
                FontCheck(
                    reference=reference,
                    status=STATUS_MISSING,
                    reason=REASON_FONT_GLYPH_MISSING,
                    font_path=font_rel,
                    cmap_format=face.cmap_format,
                    cmap_subtables=face.cmap.subtable_count,
                    glyph_count=face.glyph_count,
                    family_name=face.family_name,
                    missing_chars=missing[:limit],
                    missing_char_count=missing_count,
                    evidence=(
                        f"{reference.script}:{reference.line} font={font_rel} "
                        f"missing={missing_count}"
                    ),
                )
            )
        else:
            checks.append(
                FontCheck(
                    reference=reference,
                    status=STATUS_CHECKED,
                    reason="",
                    font_path=font_rel,
                    cmap_format=face.cmap_format,
                    cmap_subtables=face.cmap.subtable_count,
                    glyph_count=face.glyph_count,
                    family_name=face.family_name,
                    evidence=(
                        f"{reference.script}:{reference.line} font={font_rel} "
                        f"cmap={face.cmap_format} subtables={face.cmap.subtable_count} "
                        f"glyphs={face.glyph_count}"
                    ),
                )
            )
    if any(check.status == STATUS_MISSING for check in checks):
        status = STATUS_MISSING
    elif any(check.status == STATUS_UNAVAILABLE for check in checks):
        status = STATUS_UNAVAILABLE
    elif any(check.status == STATUS_UNKNOWN for check in checks):
        status = STATUS_UNKNOWN
    else:
        status = STATUS_CHECKED
    report = {
        "schema_version": SPIKE_SCHEMA_VERSION,
        "spike": "font_coverage",
        "status": status,
        "game_root": ".",
        "text": {
            "sample_count": len(samples.samples),
            "unique_char_count": len(required),
            "dynamic_sample_count": samples.dynamic_sample_count,
            "sources": list(samples.sources),
        },
        "fonts": [
            {
                "script": check.reference.script,
                "line": check.reference.line,
                "style": check.reference.style,
                "expression": check.reference.expression,
                "kind": check.reference.kind,
                "status": check.status,
                "reason": check.reason,
                "reason_text": _REASON_TEXT.get(check.reason, ""),
                "font_path": check.font_path,
                "cmap_format": check.cmap_format,
                "cmap_subtables": check.cmap_subtables,
                "glyph_count": check.glyph_count,
                "family_name": check.family_name,
                "missing_chars": list(check.missing_chars),
                "missing_char_count": check.missing_char_count,
                "evidence": check.evidence,
            }
            for check in checks
        ],
        "checked": [
            check.evidence for check in checks if check.status == STATUS_CHECKED
        ],
        "missing": [
            check.evidence for check in checks if check.status == STATUS_MISSING
        ],
        "unavailable": [
            check.evidence
            for check in checks
            if check.status == STATUS_UNAVAILABLE
        ],
        "unknown": [
            check.evidence for check in checks if check.status == STATUS_UNKNOWN
        ],
        "limits": {
            "max_missing_chars": limit,
            "max_font_bytes": MAX_FONT_BYTES,
        },
        "limitations": [
            "只读扫描，不启动 Ren'Py、不执行游戏 Python、不修改项目文件",
            "只解析静态字符串字体路径；动态表达式与 FontGroup/fallback 标记 unknown",
            "TTC 只读取第一个 face；字形覆盖不等于排版、shaping 或运行时渲染验收",
            "tl 文本提取为 best-effort，动态插值与运行时替换可能仍需人工复核",
        ],
    }
    return report


def format_report_markdown(report: Mapping[str, Any]) -> str:
    """Render the JSON report as a human-readable Markdown summary."""

    text = report.get("text") if isinstance(report.get("text"), Mapping) else {}
    lines = [
        "# Ren'Py Font Coverage Spike",
        "",
        f"- 状态：`{report.get('status')}`",
        f"- 样本：{text.get('sample_count', 0)} 条；唯一字符：{text.get('unique_char_count', 0)}",
        f"- 动态文本样本：{text.get('dynamic_sample_count', 0)}",
        "",
        "## 字体引用",
        "",
    ]
    fonts = report.get("fonts") or []
    if not fonts:
        lines.append("- 未发现字体引用。")
    for item in fonts:
        if not isinstance(item, Mapping):
            continue
        detail = (
            f"- `{item.get('status')}` `{item.get('style') or '(global)'}` "
            f"@ `{item.get('script')}:{item.get('line')}`"
        )
        if item.get("reason"):
            detail += f" reason=`{item.get('reason')}`"
        lines.append(detail)
        if item.get("font_path"):
            lines.append(
                f"  - font: `{item.get('font_path')}` "
                f"cmap={item.get('cmap_format')} glyphs={item.get('glyph_count')}"
            )
        if item.get("missing_char_count"):
            lines.append(
                f"  - missing: {item.get('missing_char_count')} "
                f"{''.join(str(ch) for ch in (item.get('missing_chars') or []))}"
            )
        if item.get("evidence"):
            lines.append(f"  - evidence: `{item.get('evidence')}`")
    lines.extend(["", "## 局限", ""])
    for limitation in report.get("limitations") or []:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def report_to_json(report: Mapping[str, Any]) -> str:
    return json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
