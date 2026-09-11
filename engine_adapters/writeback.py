"""Common validation and rendering for declarative adapter writeback plans.

Adapters describe safe text-span or semantic JSON catalog operations. This
module re-checks the plan against the live source snapshot and renders changed
files in memory. It has no filesystem write authority; workflow code remains
responsible for path resolution, transaction journaling, and atomic writes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import ntpath
from typing import Mapping, NoReturn, Sequence

from .contracts import (
    WRITEBACK_PLAN_SCHEMA_VERSION,
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


def _validate_operation(
    operation: WritebackOperation,
    documents: Mapping[str, SourceDocument],
    spans: list[tuple[str, int, int, int, int]],
    json_targets: set[tuple[str, tuple[str, ...]]],
    target_kinds: dict[str, str],
    json_documents: dict[str, object],
) -> tuple[str, SourceDocument, list[str]]:
    if operation.kind not in {
        "text_span_replace",
        "multiline_text_span_replace",
        "json_catalog_set",
    }:
        _fail(
            "common.writeback.operation_unsupported",
            f"Unsupported writeback operation kind: {operation.kind!r}",
        )
    if operation.target_root != "localization_catalog":
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
    if not operation.operation_id.startswith("op1:"):
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

    kind_family = "json" if operation.kind == "json_catalog_set" else "text"
    existing_family = target_kinds.setdefault(rel_path, kind_family)
    if existing_family != kind_family:
        _fail(
            "common.writeback.plan_invalid",
            f"Writeback target mixes incompatible operation kinds: {rel_path}",
        )

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


def validate_writeback_plan(
    plan: WritebackPlan,
    live_sources: Sequence[SourceDocument],
) -> tuple[tuple[str, WritebackOperation, SourceDocument], ...]:
    """Validate a plan against live documents and return normalized operations.

    The returned tuple is metadata only.  No file is opened for writing and no
    adapter-specific rendering is performed here.
    """

    if plan.writeback_plan_schema_version != WRITEBACK_PLAN_SCHEMA_VERSION:
        _fail(
            "common.writeback.schema_unsupported",
            f"Unsupported writeback plan schema: {plan.writeback_plan_schema_version}",
        )
    if not plan.engine or not plan.adapter_version:
        _fail(
            "common.writeback.plan_invalid",
            "Writeback plan must identify its engine and adapter version.",
        )
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

    documents: dict[str, SourceDocument] = {}
    for document in live_sources:
        rel_path = _normal_relative_path(document.file_rel_path)
        if rel_path in documents:
            _fail(
                "common.writeback.plan_invalid",
                f"Duplicate live source document: {rel_path}",
            )
        documents[rel_path] = document
    spans: list[tuple[str, int, int, int, int]] = []
    json_targets: set[tuple[str, tuple[str, ...]]] = set()
    target_kinds: dict[str, str] = {}
    json_documents: dict[str, object] = {}
    validated: list[tuple[str, WritebackOperation, SourceDocument]] = []
    for operation in plan.operations:
        rel_path, document, _lines = _validate_operation(
            operation,
            documents,
            spans,
            json_targets,
            target_kinds,
            json_documents,
        )
        validated.append((rel_path, operation, document))
    return tuple(validated)


def render_writeback_plan(
    plan: WritebackPlan,
    live_sources: Sequence[SourceDocument],
) -> dict[str, list[str]]:
    """Render a validated plan into changed in-memory line arrays.

    Callers must pass the result to their existing atomic writer.  This
    function deliberately never resolves paths or mutates source documents.
    """

    validated = validate_writeback_plan(plan, live_sources)
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
        current = data
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
