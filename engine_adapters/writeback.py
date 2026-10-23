"""Common validation and rendering for declarative adapter writeback plans.

Adapters describe safe text-span, JSON catalog or source-copy operations. This
module re-checks the plan against the live source snapshot and renders changed
files in memory. It has no filesystem write authority; workflow code remains
responsible for path resolution, transaction journaling, and atomic writes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import ntpath
import re
from typing import Any, Mapping, NoReturn, Sequence, cast

from .contracts import (
    WRITEBACK_PLAN_SCHEMA_VERSION,
    SOURCE_JSON_WRITEBACK_PLAN_SCHEMA_VERSION,
    SourceDocument,
    WritebackOperation,
    WritebackPlan,
)
from .coverage import digest_json


class WritebackPlanError(ValueError):
    """Raised when a declarative writeback plan is not safe to consume."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


def source_snapshot_fingerprint(source_documents: Sequence[SourceDocument]) -> str:
    """Return the common source snapshot digest used by adapter plans."""

    return digest_json(
        [
            document.manifest_entry()
            for document in sorted(source_documents, key=lambda item: item.file_rel_path)
        ]
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normal_relative_path(value: str) -> str:
    raw = str(value or "")
    if not raw or ntpath.isabs(raw) or ntpath.splitdrive(raw)[0]:
        raise WritebackPlanError(
            "common.writeback.path_escape",
            f"Writeback target must be a relative path: {value!r}",
        )
    normalized = raw.replace("\\", "/")
    parts = normalized.split("/")
    if any(not part or part == "." or part == ".." for part in parts):
        raise WritebackPlanError(
            "common.writeback.path_escape",
            f"Writeback target contains an unsafe path component: {value!r}",
        )
    return "/".join(parts)


def _plan_payload(plan: WritebackPlan) -> dict:
    payload = plan.to_dict()
    payload.pop("plan_digest", None)
    return payload


def _operation_payload(operation: WritebackOperation) -> dict:
    payload = operation.to_dict()
    payload.pop("operation_id", None)
    return payload


def _text_fragment(
    lines: Sequence[str],
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
) -> str:
    if start_line == end_line:
        return lines[start_line][start_col:end_col]
    pieces = [lines[start_line][start_col:]]
    pieces.extend(lines[start_line + 1 : end_line])
    pieces.append(lines[end_line][:end_col])
    return "".join(pieces)


def _fail(reason_code: str, message: str) -> NoReturn:
    raise WritebackPlanError(reason_code, message)


def _load_source_json(document: SourceDocument) -> object:
    def unique_members(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object member")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    try:
        return json.loads(
            document.content.decode("utf-8-sig"),
            object_pairs_hook=unique_members,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        _fail(
            "common.writeback.source_json_invalid",
            f"Source-copy JSON is invalid: {document.file_rel_path} ({type(exc).__name__})",
        )


def _source_json_string(data: object, path: tuple[str | int, ...], rel_path: str) -> str:
    current = data
    for part in path:
        if isinstance(current, dict) and type(part) is str and part in current:
            current = current[part]
        elif isinstance(current, list) and type(part) is int and 0 <= part < len(current):
            current = current[part]
        else:
            _fail(
                "common.writeback.source_json_path_missing",
                f"Source-copy JSON path is missing or has a wrong index type: {rel_path}:{path!r}",
            )
    if not isinstance(current, str):
        _fail(
            "common.writeback.source_json_value_invalid",
            f"Source-copy JSON target is not a string: {rel_path}:{path!r}",
        )
    return current


def _validate_source_json_operation(
    operation: WritebackOperation,
    rel_path: str,
    target: SourceDocument,
    sources: Mapping[str, SourceDocument],
    source_json: dict[str, object],
    target_json: dict[str, object],
    json_targets: set[tuple[str, tuple[str | int, ...]]],
) -> None:
    coordinates = (operation.line, operation.start_col, operation.end_col)
    if any(type(value) is not int for value in coordinates) or coordinates != (-1, -1, -1) or operation.end_line is not None:
        _fail("common.writeback.plan_invalid", "Source-copy JSON operations cannot use text spans.")
    path = operation.target_json_path
    if not isinstance(path, tuple) or not path or any(
        not ((type(part) is str and bool(part)) or (type(part) is int and part >= 0))
        for part in path
    ):
        _fail("common.writeback.plan_invalid", "Source-copy JSON paths require string keys or nonnegative integer indexes.")
    if (rel_path, path) in json_targets:
        _fail("common.writeback.target_duplicate", f"Duplicate JSON target: {rel_path}:{path!r}")
    json_targets.add((rel_path, path))
    source = sources.get(rel_path)
    if source is None:
        _fail("common.writeback.source_missing", f"Frozen JSON source is missing: {rel_path}")
    if rel_path not in source_json:
        source_json[rel_path] = _load_source_json(source)
    if rel_path not in target_json:
        target_json[rel_path] = _load_source_json(target)
    source_value = _source_json_string(source_json[rel_path], path, rel_path)
    target_value = _source_json_string(target_json[rel_path], path, rel_path)
    if not source_value.strip():
        _fail("common.writeback.source_json_value_invalid", "Empty source strings remain structural data.")
    replacement = operation.replacement_fragment
    if not isinstance(replacement, str) or not replacement.strip():
        _fail("common.writeback.replacement_invalid", "Source-copy JSON replacements must be nonblank strings.")
    try:
        replacement.encode("utf-8")
        source_digest = _sha256_text(source_value)
        target_digest = _sha256_text(target_value)
    except UnicodeError:
        _fail("common.writeback.replacement_invalid", "Source-copy JSON strings must encode as UTF-8.")
    if source_digest != operation.expected_text_digest:
        _fail("common.writeback.source_value_mismatch", f"Frozen JSON source value changed: {rel_path}:{path!r}")
    if target_digest != operation.expected_fragment_sha256:
        _fail("common.writeback.target_value_mismatch", f"Current JSON target value changed: {rel_path}:{path!r}")


def _validate_operation(
    operation: WritebackOperation,
    documents: Mapping[str, SourceDocument],
    spans: list[tuple[str, int, int, int, int]],
    json_targets: set[tuple[str, tuple[str | int, ...]]],
    target_kinds: dict[str, str],
    json_documents: dict[str, object],
    *,
    source_documents: Mapping[str, SourceDocument] | None = None,
    source_json_documents: dict[str, object] | None = None,
) -> tuple[str, SourceDocument, Sequence[str]]:
    source_copy = source_documents is not None
    if source_copy and any(
        not isinstance(value, str)
        for value in (
            operation.kind, operation.target_root, operation.target_rel_path,
            operation.occurrence_id, operation.expected_file_sha256,
            operation.expected_fragment_sha256, operation.expected_text_digest,
            operation.validation_digest,
        )
    ):
        _fail("common.writeback.plan_invalid", "Source-copy operation identity fields must be strings.")
    allowed_kinds = {"json_source_string_set"} if source_copy else {
        "text_span_replace",
        "multiline_text_span_replace",
        "json_catalog_set",
    }
    if operation.kind not in allowed_kinds:
        _fail(
            "common.writeback.operation_unsupported",
            f"Unsupported writeback operation kind: {operation.kind!r}",
        )
    if operation.target_root != ("source_copy" if source_copy else "localization_catalog"):
        _fail(
            "common.writeback.target_root_unsupported",
            f"Unsupported writeback target root: {operation.target_root!r}",
        )
    if any(
        not str(value or "")
        for value in (
            operation.occurrence_id,
            operation.expected_file_sha256,
            operation.expected_fragment_sha256,
            operation.expected_text_digest,
            operation.validation_digest,
        )
    ):
        _fail(
            "common.writeback.plan_invalid",
            f"Writeback operation is missing required integrity fields: {operation.operation_id!r}",
        )
    prefix = "op2:" if source_copy else "op1:"
    if not isinstance(operation.operation_id, str) or not operation.operation_id.startswith(prefix):
        _fail(
            "common.writeback.plan_digest_mismatch",
            f"Invalid writeback operation id: {operation.operation_id!r}",
        )
    if digest_json(_operation_payload(operation)) != operation.operation_id[4:]:
        _fail(
            "common.writeback.plan_digest_mismatch",
            f"Writeback operation digest mismatch: {operation.operation_id!r}",
        )

    rel_path = _normal_relative_path(operation.target_rel_path)
    if rel_path != operation.target_rel_path:
        _fail(
            "common.writeback.path_escape",
            f"Writeback target path is not normalized: {operation.target_rel_path!r}",
        )
    document = documents.get(rel_path)
    if document is None:
        _fail(
            "common.writeback.target_missing",
            f"Writeback target is not in the live source snapshot: {rel_path}",
        )
    if operation.expected_file_sha256 != document.sha256:
        _fail(
            "common.writeback.source_snapshot_mismatch",
            f"Writeback target file changed: {rel_path}",
        )

    kind_family = "json" if operation.kind in {"json_catalog_set", "json_source_string_set"} else "text"
    existing_family = target_kinds.setdefault(rel_path, kind_family)
    if existing_family != kind_family:
        _fail(
            "common.writeback.plan_invalid",
            f"Writeback target mixes incompatible operation kinds: {rel_path}",
        )

    if source_copy:
        assert source_documents is not None
        _validate_source_json_operation(
            operation,
            rel_path,
            document,
            source_documents,
            source_json_documents if source_json_documents is not None else {},
            json_documents,
            json_targets,
        )
        return rel_path, document, list(document.lines())

    if operation.kind == "json_catalog_set":
        if (operation.line, operation.start_col, operation.end_col) != (-1, -1, -1):
            _fail(
                "common.writeback.plan_invalid",
                f"JSON catalog operation has unexpected span coordinates: {rel_path}",
            )
        json_path = tuple(operation.target_json_path)
        if not json_path or any(
            not isinstance(part, str) or not part for part in json_path
        ):
            _fail(
                "common.writeback.plan_invalid",
                f"JSON catalog operation has an invalid target path: {rel_path}",
            )
        json_path = cast(tuple[str, ...], json_path)
        json_target = (rel_path, json_path)
        if json_target in json_targets:
            _fail(
                "common.writeback.target_duplicate",
                f"Duplicate JSON catalog target: {rel_path}:{'/'.join(json_path)}",
            )
        json_targets.add(json_target)
        if rel_path not in json_documents:
            try:
                json_documents[rel_path] = json.loads(document.text())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                _fail(
                    "common.writeback.catalog_invalid_json",
                    f"Writeback catalog is not valid JSON: {rel_path} ({type(exc).__name__})",
                )
        data = json_documents[rel_path]
        current = data
        for part in json_path[:-1]:
            if not isinstance(current, dict) or part not in current:
                _fail(
                    "common.writeback.catalog_path_missing",
                    f"Writeback catalog path is missing: {rel_path}:{'/'.join(json_path)}",
                )
            current = current[part]
        leaf = json_path[-1]
        if not isinstance(current, dict) or leaf not in current:
            _fail(
                "common.writeback.catalog_path_missing",
                f"Writeback catalog row is missing: {rel_path}:{'/'.join(json_path)}",
            )
        current_value = current[leaf]
        if not isinstance(current_value, str):
            _fail(
                "common.writeback.catalog_value_invalid",
                f"Writeback catalog row is not a string: {rel_path}:{'/'.join(json_path)}",
            )
        if _sha256_text(current_value) != operation.expected_fragment_sha256:
            _fail(
                "common.writeback.catalog_value_mismatch",
                f"Writeback catalog row no longer matches the plan: {rel_path}:{'/'.join(json_path)}",
            )
        if operation.replacement_fragment == "":
            _fail(
                "common.writeback.replacement_invalid",
                f"JSON catalog replacement must not be empty: {rel_path}:{'/'.join(json_path)}",
            )
        return rel_path, document, list(document.lines())

    if operation.target_json_path:
        _fail(
            "common.writeback.plan_invalid",
            f"Text-span operation has an unexpected JSON target path: {rel_path}",
        )
    lines = document.lines()
    is_multiline = operation.kind == "multiline_text_span_replace"
    if is_multiline:
        if operation.end_line is None:
            _fail(
                "common.writeback.span_invalid",
                f"Multiline writeback operation is missing end_line: {rel_path}",
            )
        end_line = int(operation.end_line)
        if (
            operation.line < 0
            or operation.line >= len(lines)
            or end_line < operation.line
            or end_line >= len(lines)
        ):
            _fail(
                "common.writeback.span_invalid",
                f"Writeback lines are outside the live file: {rel_path}:{operation.line}-{end_line}",
            )
        if operation.start_col < 0 or operation.start_col > len(lines[operation.line]):
            _fail(
                "common.writeback.span_invalid",
                f"Writeback start column is outside the live line: {rel_path}:{operation.line}",
            )
        if operation.end_col < 0 or operation.end_col > len(lines[end_line]):
            _fail(
                "common.writeback.span_invalid",
                f"Writeback end column is outside the live line: {rel_path}:{end_line}",
            )
        if operation.line == end_line and operation.end_col <= operation.start_col:
            _fail(
                "common.writeback.span_invalid",
                f"Writeback span is empty: {rel_path}:{operation.line}",
            )
    else:
        if operation.end_line is not None:
            _fail(
                "common.writeback.plan_invalid",
                f"Single-line operation has an unexpected end_line: {rel_path}",
            )
        if operation.line < 0 or operation.line >= len(lines):
            _fail(
                "common.writeback.span_invalid",
                f"Writeback line is outside the live file: {rel_path}:{operation.line}",
            )
        line = lines[operation.line]
        if (
            operation.start_col < 0
            or operation.end_col <= operation.start_col
            or operation.end_col > len(line)
        ):
            _fail(
                "common.writeback.span_invalid",
                f"Writeback span is outside the live line: {rel_path}:{operation.line}",
            )
        end_line = operation.line
    if not is_multiline and (
        "\n" in operation.replacement_fragment
        or "\r" in operation.replacement_fragment
    ):
        _fail(
            "common.writeback.replacement_invalid",
            f"Writeback replacement must stay on one line: {rel_path}:{operation.line}",
        )
    raw_fragment = _text_fragment(
        lines,
        operation.line,
        operation.start_col,
        end_line,
        operation.end_col,
    )
    if _sha256_text(raw_fragment) != operation.expected_fragment_sha256:
        _fail(
            "common.writeback.span_mismatch",
            f"Writeback span no longer matches the planned fragment: {rel_path}:{operation.line}",
        )
    if not operation.validation_digest:
        _fail(
            "common.writeback.validation_missing",
            f"Writeback operation has no validation digest: {operation.occurrence_id}",
        )

    span_start = (operation.line, operation.start_col)
    span_end = (end_line, operation.end_col)
    for existing in spans:
        if existing[0] != rel_path:
            continue
        existing_start = (existing[1], existing[2])
        existing_end = (existing[3], existing[4])
        if span_start < existing_end and existing_start < span_end:
            _fail(
                "common.writeback.span_overlap",
                f"Overlapping writeback spans: {rel_path}:{operation.line}",
            )
    spans.append((rel_path, operation.line, operation.start_col, end_line, operation.end_col))
    return rel_path, document, lines


def _index_documents(
    documents: Sequence[SourceDocument], *, verify_content: bool = False
) -> dict[str, SourceDocument]:
    indexed = {}
    for document in documents:
        rel_path = _normal_relative_path(document.file_rel_path)
        if rel_path in indexed:
            _fail("common.writeback.plan_invalid", f"Duplicate live document: {rel_path}")
        if verify_content and (
            len(document.content) != document.size
            or hashlib.sha256(document.content).hexdigest() != document.sha256
        ):
            _fail("common.writeback.source_snapshot_mismatch", f"Live document bytes do not match its digest: {rel_path}")
        indexed[rel_path] = document
    return indexed


def validate_writeback_plan(
    plan: WritebackPlan,
    live_sources: Sequence[SourceDocument],
    *,
    live_targets: Sequence[SourceDocument] | None = None,
) -> tuple[tuple[str, WritebackOperation, SourceDocument], ...]:
    """Validate a plan against live documents and return normalized operations.

    The returned tuple is metadata only.  No file is opened for writing and no
    adapter-specific rendering is performed here. Schema 2 binds immutable
    originals through ``live_sources`` and current copies through the required
    ``live_targets``; each operation uses the same relative file/path in both.
    """

    schema = plan.writeback_plan_schema_version
    if type(schema) is not int or schema not in {
        WRITEBACK_PLAN_SCHEMA_VERSION, SOURCE_JSON_WRITEBACK_PLAN_SCHEMA_VERSION
    }:
        _fail(
            "common.writeback.schema_unsupported",
            f"Unsupported writeback plan schema: {plan.writeback_plan_schema_version}",
        )
    if not plan.engine or not plan.adapter_version:
        _fail(
            "common.writeback.plan_invalid",
            "Writeback plan must identify its engine and adapter version.",
        )
    source_copy = schema == SOURCE_JSON_WRITEBACK_PLAN_SCHEMA_VERSION
    if source_copy:
        if live_targets is None or not plan.target_snapshot_fingerprint:
            _fail("common.writeback.target_snapshot_required", "Source-copy plans require a separate live target snapshot.")
        if source_snapshot_fingerprint(live_targets) != plan.target_snapshot_fingerprint:
            _fail("common.writeback.target_snapshot_mismatch", "Source-copy target snapshot changed.")
    elif live_targets is not None or plan.target_snapshot_fingerprint:
        _fail("common.writeback.plan_invalid", "Schema 1 plans cannot use separate target snapshots.")
    if source_snapshot_fingerprint(live_sources) != plan.source_snapshot_fingerprint:
        _fail(
            "common.writeback.source_snapshot_mismatch",
            "Writeback plan source snapshot does not match live sources.",
        )
    if digest_json(_plan_payload(plan)) != plan.plan_digest:
        _fail(
            "common.writeback.plan_digest_mismatch",
            "Writeback plan digest does not match its payload.",
        )

    sources = _index_documents(live_sources, verify_content=source_copy)
    documents = (
        _index_documents(live_targets, verify_content=True)
        if live_targets is not None else sources
    )
    spans: list[tuple[str, int, int, int, int]] = []
    json_targets: set[tuple[str, tuple[str | int, ...]]] = set()
    target_kinds: dict[str, str] = {}
    json_documents: dict[str, object] = {}
    source_json_documents: dict[str, object] = {}
    validated: list[tuple[str, WritebackOperation, SourceDocument]] = []
    for operation in plan.operations:
        rel_path, document, _lines = _validate_operation(
            operation,
            documents,
            spans,
            json_targets,
            target_kinds,
            json_documents,
            source_documents=sources if source_copy else None,
            source_json_documents=source_json_documents,
        )
        validated.append((rel_path, operation, document))
    return tuple(validated)


def _json_string_spans(text: str, data: object) -> dict[tuple[str | int, ...], tuple[int, int]]:
    # JSON has already been parsed strictly, with duplicate members rejected.
    # Its quoted tokens therefore follow dictionary insertion / array order.
    # Keys consume tokens but are never replacement spans. Non-string values
    # never consume a quoted token, so their original numeric lexemes survive.
    tokens = iter(re.finditer(r'"(?:[^"\\]|\\.)*"', text, re.DOTALL))
    spans = {}

    def walk(value, path):
        if isinstance(value, str):
            token = next(tokens)
            spans[path] = (token.start(), token.end())
        elif isinstance(value, dict):
            for key, child in value.items():
                next(tokens)  # Object member name.
                walk(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*path, index))

    try:
        walk(data, ())
        if next(tokens, None) is not None:
            raise ValueError("Unexpected quoted token")
    except (StopIteration, ValueError, RecursionError):
        _fail("common.writeback.source_json_invalid", "Cannot locate JSON string tokens safely.")
    return spans


def _render_source_json(
    validated: Sequence[tuple[str, WritebackOperation, SourceDocument]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[WritebackOperation]] = {}
    documents = {}
    for rel_path, operation, document in validated:
        grouped.setdefault(rel_path, []).append(operation)
        documents[rel_path] = document
    rendered = {}
    for rel_path, operations in grouped.items():
        document = documents[rel_path]
        text = document.content.decode("utf-8-sig")
        data = _load_source_json(document)
        spans = _json_string_spans(text, data)
        edits = []
        for operation in operations:
            path = operation.target_json_path
            if _source_json_string(data, path, rel_path) == operation.replacement_fragment:
                continue
            start, end = spans[path]
            edits.append((start, end, json.dumps(operation.replacement_fragment, ensure_ascii=False)))
        pieces: list[str] = []
        cursor = 0
        for start, end, replacement in sorted(edits):
            pieces.extend((text[cursor:start], replacement))
            cursor = end
        pieces.append(text[cursor:])
        result = "".join(pieces)
        if document.content.startswith(b"\xef\xbb\xbf"):
            result = "\ufeff" + result
        rendered[rel_path] = result.splitlines(keepends=True)
    return rendered


def render_writeback_plan(
    plan: WritebackPlan,
    live_sources: Sequence[SourceDocument],
    *,
    live_targets: Sequence[SourceDocument] | None = None,
) -> dict[str, list[str]]:
    """Render a validated plan into changed in-memory line arrays.

    Callers must pass the result to their existing atomic writer.  This
    function deliberately never resolves paths or mutates source documents.
    Schema 2 replaces only JSON string-value tokens in ``live_targets``;
    untouched bytes, including whitespace, number spelling and BOM, survive.
    """

    validated = validate_writeback_plan(plan, live_sources, live_targets=live_targets)
    if plan.writeback_plan_schema_version == SOURCE_JSON_WRITEBACK_PLAN_SCHEMA_VERSION:
        return _render_source_json(validated)
    text_operations: dict[str, list[WritebackOperation]] = {}
    text_documents: dict[str, SourceDocument] = {}
    json_items: list[tuple[str, WritebackOperation, SourceDocument]] = []
    for rel_path, operation, document in validated:
        if operation.kind == "json_catalog_set":
            json_items.append((rel_path, operation, document))
            continue
        text_operations.setdefault(rel_path, []).append(operation)
        text_documents[rel_path] = document

    rendered: dict[str, list[str]] = {}
    for rel_path, operations in text_operations.items():
        document = text_documents[rel_path]
        lines = document.lines()
        line_starts = [0]
        for line in lines:
            line_starts.append(line_starts[-1] + len(line))
        positioned: list[tuple[int, int, WritebackOperation]] = []
        for operation in operations:
            end_line = operation.end_line if operation.end_line is not None else operation.line
            start_offset = line_starts[operation.line] + operation.start_col
            end_offset = line_starts[end_line] + operation.end_col
            positioned.append((start_offset, end_offset, operation))
        positioned.sort(key=lambda item: (item[0], item[1]))
        file_text = "".join(lines)
        pieces: list[str] = []
        cursor = 0
        for start_offset, end_offset, operation in positioned:
            pieces.append(file_text[cursor:start_offset])
            pieces.append(operation.replacement_fragment)
            cursor = end_offset
        pieces.append(file_text[cursor:])
        rendered[rel_path] = "".join(pieces).splitlines(keepends=True)

    json_documents: dict[str, object] = {}
    json_source_documents: dict[str, SourceDocument] = {}
    for rel_path, operation, document in json_items:
        data = json_documents.setdefault(
            rel_path,
            copy.deepcopy(json.loads(document.text())),
        )
        json_source_documents[rel_path] = document
        current: Any = data
        for part in operation.target_json_path[:-1]:
            current = current[part]
        current[operation.target_json_path[-1]] = operation.replacement_fragment
    for rel_path, data in json_documents.items():
        document = json_source_documents[rel_path]
        source_text = document.text()
        newline = (
            "\r\n"
            if "\r\n" in source_text
            else "\r"
            if "\r" in source_text
            else "\n"
        )
        serialized = json.dumps(data, ensure_ascii=False, indent=2)
        if newline != "\n":
            serialized = serialized.replace("\n", newline)
        if document.content.startswith(b"\xef\xbb\xbf"):
            serialized = "\ufeff" + serialized
        if source_text.endswith(("\n", "\r")):
            serialized += newline
        rendered[rel_path] = serialized.splitlines(keepends=True)
    return rendered
