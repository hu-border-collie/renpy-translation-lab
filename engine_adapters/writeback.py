"""Common validation and rendering for declarative adapter writeback plans.

Adapters describe safe text-span operations.  This module re-checks the plan
against the live source snapshot and renders changed lines in memory.  It has
no filesystem write authority; workflow code remains responsible for path
resolution, transaction journaling, and atomic writes.
"""

from __future__ import annotations

import hashlib
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


def _fail(reason_code: str, message: str) -> NoReturn:
    raise WritebackPlanError(reason_code, message)


def _validate_operation(
    operation: WritebackOperation,
    documents: Mapping[str, SourceDocument],
    spans: list[tuple[str, int, int, int]],
) -> tuple[str, SourceDocument, list[str]]:
    if operation.kind != "text_span_replace":
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
    lines = document.lines()
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
    if "\n" in operation.replacement_fragment or "\r" in operation.replacement_fragment:
        _fail(
            "common.writeback.replacement_invalid",
            f"Writeback replacement must stay on one line: {rel_path}:{operation.line}",
        )
    raw_fragment = line[operation.start_col : operation.end_col]
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

    span = (rel_path, operation.line, operation.start_col, operation.end_col)
    for existing in spans:
        if (
            existing[0] == span[0]
            and existing[1] == span[1]
            and max(existing[2], span[2]) < min(existing[3], span[3])
        ):
            _fail(
                "common.writeback.span_overlap",
                f"Overlapping writeback spans: {rel_path}:{operation.line}",
            )
    spans.append(span)
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
    spans: list[tuple[str, int, int, int]] = []
    validated: list[tuple[str, WritebackOperation, SourceDocument]] = []
    for operation in plan.operations:
        rel_path, document, _lines = _validate_operation(operation, documents, spans)
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
    validated = tuple(
        sorted(
            validated,
            key=lambda item: (item[0], item[1].line, item[1].start_col),
            reverse=True,
        )
    )
    rendered: dict[str, list[str]] = {}
    for rel_path, operation, document in validated:
        lines = rendered.setdefault(rel_path, list(document.lines()))
        lines[operation.line] = (
            lines[operation.line][: operation.start_col]
            + operation.replacement_fragment
            + lines[operation.line][operation.end_col :]
        )
    return rendered
