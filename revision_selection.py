"""Shared staged-selection contract for revision proposal candidates.

The GUI only edits a small, serializable selection request.  Candidate
validation, filtering semantics, digest binding, and replay checks live here
so the CLI and GUI cannot drift into separate proposal state machines.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from atomic_io import atomic_write_json

import revision_proposals


STAGED_SELECTION_SCHEMA_VERSION = 1
STAGED_SELECTION_KIND = "revision_proposal_staged_selection"
SELECTION_REQUEST_KIND = "revision_proposal_selection"

STATUS_VALID = "valid"
STATUS_NO_OP = "no_op"
STATUS_INVALID = "invalid"
STATUS_STALE = "stale"
STATUS_CONFLICT = "conflict"

SELECTION_STATE_SELECTED = "selected"
SELECTION_STATE_UNSELECTED = "unselected"

_KNOWN_UNSELECTED_DISPOSITIONS = frozenset(
    {"rejected", "reject", "skipped", "skip", "none", "unselected", "pending"}
)
_CONFLICT_CODES = frozenset(
    {
        "DUPLICATE_OCCURRENCE_ID",
        "CONFLICTING_PROPOSAL",
        "IDENTITY_MISMATCH",
        "PROJECT_IDENTITY_CONFLICT",
        "ITEM_SNAPSHOT_CONFLICT",
    }
)
_STALE_CODES = frozenset(
    {
        "CORPUS_SNAPSHOT_INCONSISTENT",
        "CORPUS_SNAPSHOT_STALE",
        "LIVE_SOURCE_CHANGED_DURING_IMPORT",
        "PROJECT_IDENTITY_STALE",
        "SOURCE_MISMATCH",
        "CURRENT_TRANSLATION_STALE",
        "ITEM_SNAPSHOT_STALE",
        "FILE_PATH_MISMATCH",
    }
)


def file_sha256(path: str) -> str:
    """Return the SHA-256 digest of one local artifact."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    """Digest a JSON-compatible value with deterministic key ordering."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def operation_identity(
    *,
    project_identity: Mapping[str, Any] | None,
    proposal_path: str,
    proposal_sha256: str,
    corpus_manifest_path: str = "",
    corpus_manifest_sha256: str = "",
) -> str:
    """Return the owner identity for one staged proposal operation.

    The identity intentionally includes the project and input artifact
    digests.  A late worker result therefore cannot be applied after a project
    switch or after the proposal/companion manifest has been replaced.
    """

    normalized_project = {
        str(key): str(value or "").strip()
        for key, value in (project_identity or {}).items()
    }
    return canonical_digest(
        {
            "operation": "revision-proposal-selection",
            "project_identity": normalized_project,
            "proposal_path": os.path.normcase(os.path.abspath(str(proposal_path or ""))),
            "proposal_sha256": str(proposal_sha256 or ""),
            "corpus_manifest_path": (
                os.path.normcase(os.path.abspath(str(corpus_manifest_path)))
                if str(corpus_manifest_path or "").strip()
                else ""
            ),
            "corpus_manifest_sha256": str(corpus_manifest_sha256 or ""),
        }
    )


def diagnostic_status(codes: Sequence[str]) -> str:
    """Map proposal diagnostics to the safest candidate display status."""

    normalized = {str(code or "").strip() for code in codes if str(code or "").strip()}
    if normalized & _CONFLICT_CODES:
        return STATUS_CONFLICT
    if normalized & _STALE_CODES or any(code.endswith("_STALE") for code in normalized):
        return STATUS_STALE
    if normalized:
        return STATUS_INVALID
    return STATUS_VALID


def _row_identity(row: Mapping[str, Any]) -> str:
    return str(row.get("occurrence_id") or row.get("identity_v2") or "").strip()


def _row_number(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("_row_number") or 0)
    except (TypeError, ValueError):
        return 0


def _compact(value: object) -> str:
    return "".join(str(value or "").split())


def _candidate_row(
    row: Mapping[str, Any],
    *,
    status: str,
    diagnostic_codes: Sequence[str],
    diagnostic_messages: Sequence[str],
    initially_selected: bool,
) -> dict[str, Any]:
    identity = _row_identity(row)
    proposed = row.get("proposed_translation")
    current = row.get("current_translation")
    no_op = (
        status == STATUS_VALID
        and isinstance(proposed, str)
        and isinstance(current, str)
        and _compact(proposed) == _compact(current)
    )
    if no_op:
        status = STATUS_NO_OP
    selectable = status == STATUS_VALID and bool(identity)
    selected = bool(initially_selected and selectable)
    normalized = dict(row)
    normalized.pop("_row_number", None)
    return {
        "candidate_key": f"{identity}#row:{_row_number(row)}",
        "identity_v2": identity,
        "occurrence_id": str(row.get("occurrence_id") or identity),
        "file_rel_path": str(row.get("file_rel_path") or ""),
        "source": str(row.get("source") or ""),
        "current_translation": str(row.get("current_translation") or ""),
        "proposed_translation": str(proposed or "") if isinstance(proposed, str) else proposed,
        "reason": str(row.get("reason") or "") if isinstance(row.get("reason"), str) else row.get("reason"),
        "row": _row_number(row),
        "status": status,
        "valid": selectable,
        "selectable": selectable,
        "selection_state": (
            SELECTION_STATE_SELECTED if selected else SELECTION_STATE_UNSELECTED
        ),
        "selected": selected,
        "initial_selected": bool(initially_selected),
        "diagnostic_codes": sorted(set(str(code) for code in diagnostic_codes if code)),
        "diagnostic_messages": [str(message) for message in diagnostic_messages if message],
        # The normalized row is replay input for the confirm command.  It is
        # not a second source of truth: confirm revalidates it against live
        # project state before constructing a revision manifest.
        "proposal": normalized,
    }


def build_candidates(
    rows: Sequence[Mapping[str, Any]],
    live_items: Mapping[str, Mapping[str, Any]],
    *,
    live_snapshot_digest: str,
    live_project_identity: Mapping[str, Any] | None = None,
    corpus_manifest: Mapping[str, Any] | None = None,
    extra_diagnostics: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate every proposal row and return stable, display-ready candidates.

    Existing proposal validation treats ``selected=false`` rows as audit-only
    input.  Staging probes those rows as explicitly selectable candidates while
    retaining their original selection hint; a later confirmation still has
    to name the exact identity.  Invalid, stale, conflict, and no-op rows are
    never marked selectable.
    """

    probe_rows: list[dict[str, Any]] = []
    pre_diagnostics: list[dict[str, Any]] = []
    for index, raw in enumerate(rows, start=1):
        row = dict(raw)
        row.setdefault("_row_number", index)
        original_selected = row.get("selected")
        disposition = str(row.get("disposition") or "").strip().lower()
        if not isinstance(original_selected, bool):
            pre_diagnostics.append(
                {
                    "code": "INVALID_SELECTED",
                    "message": "selected must be a boolean",
                    "row": _row_number(row),
                    "occurrence_id": _row_identity(row),
                }
            )
        if not disposition:
            pre_diagnostics.append(
                {
                    "code": "MISSING_DISPOSITION",
                    "message": "disposition is required",
                    "row": _row_number(row),
                    "occurrence_id": _row_identity(row),
                }
            )
        # An exported proposal may carry ``rejected`` as its initial hint.  A
        # GUI confirmation is a new explicit decision, so validate the safety
        # fields with an accepting disposition without mutating the displayed
        # source row.
        if (
            original_selected is False
            and disposition in _KNOWN_UNSELECTED_DISPOSITIONS
        ):
            row["disposition"] = "proposed"
        row["selected"] = True
        probe_rows.append(row)

    validation = revision_proposals.validate(
        probe_rows,
        live_items,
        live_snapshot_digest=live_snapshot_digest,
        live_project_identity=live_project_identity,
        corpus_manifest=corpus_manifest,
    )
    diagnostics = [dict(item) for item in validation.diagnostics]
    diagnostics.extend(dict(item) for item in extra_diagnostics)
    diagnostics.extend(pre_diagnostics)

    by_row: dict[int, list[dict[str, Any]]] = {}
    by_identity: dict[str, list[dict[str, Any]]] = {}
    for diagnostic in diagnostics:
        row_number = int(diagnostic.get("row") or 0)
        if row_number:
            by_row.setdefault(row_number, []).append(diagnostic)
        identity = str(diagnostic.get("occurrence_id") or "").strip()
        if identity:
            by_identity.setdefault(identity, []).append(diagnostic)

    valid_by_row = {
        _row_number(row): row
        for row in validation.proposals
        if _row_number(row)
    }
    global_diagnostics = [item for item in diagnostics if not int(item.get("row") or 0)]
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(rows, start=1):
        row = dict(raw)
        row.setdefault("_row_number", index)
        row_number = _row_number(row)
        identity = _row_identity(row)
        row_diagnostics = list(by_row.get(row_number, ()))
        row_diagnostics.extend(by_identity.get(identity, ()))
        # A corpus/project snapshot diagnostic with row=0 invalidates every
        # candidate in this staged session, not only the row that happened to
        # trigger it.
        row_diagnostics.extend(global_diagnostics)
        codes = [str(item.get("code") or "") for item in row_diagnostics]
        messages = [str(item.get("message") or "") for item in row_diagnostics]
        status = diagnostic_status(codes)
        if status == STATUS_VALID and row_number not in valid_by_row:
            status = STATUS_INVALID
        candidate = _candidate_row(
            row,
            status=status,
            diagnostic_codes=codes,
            diagnostic_messages=messages,
            initially_selected=row.get("selected") is True,
        )
        candidates.append(candidate)

    summary = {
        "total_count": len(candidates),
        "valid_count": sum(item["status"] == STATUS_VALID for item in candidates),
        "selectable_count": sum(bool(item.get("selectable")) for item in candidates),
        "selected_count": sum(bool(item.get("selected")) for item in candidates),
        "unselected_count": sum(
            item["status"] == STATUS_VALID and not item.get("selected")
            for item in candidates
        ),
        "no_op_count": sum(item["status"] == STATUS_NO_OP for item in candidates),
        "invalid_count": sum(item["status"] == STATUS_INVALID for item in candidates),
        "stale_count": sum(item["status"] == STATUS_STALE for item in candidates),
        "conflict_count": sum(item["status"] == STATUS_CONFLICT for item in candidates),
    }
    return candidates, summary


def _stage_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(document, ensure_ascii=False, default=str))
    payload.pop("staged_selection_digest", None)
    session = payload.get("session")
    if isinstance(session, dict):
        session.pop("staged_selection_digest", None)
    selection = payload.get("selection")
    if isinstance(selection, dict):
        selection.pop("staged_selection_digest", None)
        selection.pop("selection_digest", None)
    return payload


def build_staged_selection(
    *,
    rows: Sequence[Mapping[str, Any]],
    live_items: Mapping[str, Mapping[str, Any]],
    live_snapshot_digest: str,
    project_identity: Mapping[str, Any],
    proposal_path: str,
    proposal_sha256: str,
    corpus_manifest_path: str = "",
    corpus_manifest_sha256: str = "",
    corpus_manifest: Mapping[str, Any] | None = None,
    source_file_digests: Mapping[str, str] | None = None,
    operation_id: str = "",
    extra_diagnostics: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a replayable staged-selection document without previewing."""

    candidates, summary = build_candidates(
        rows,
        live_items,
        live_snapshot_digest=live_snapshot_digest,
        live_project_identity=project_identity,
        corpus_manifest=corpus_manifest,
        extra_diagnostics=extra_diagnostics,
    )
    operation_id = str(operation_id or "").strip() or operation_identity(
        project_identity=project_identity,
        proposal_path=proposal_path,
        proposal_sha256=proposal_sha256,
        corpus_manifest_path=corpus_manifest_path,
        corpus_manifest_sha256=corpus_manifest_sha256,
    )
    document: dict[str, Any] = {
        "schema_version": STAGED_SELECTION_SCHEMA_VERSION,
        "kind": STAGED_SELECTION_KIND,
        "session": {
            "operation_identity": operation_id,
            "project_identity": dict(project_identity),
            "proposal_path": os.path.abspath(str(proposal_path or "")),
            "proposal_sha256": str(proposal_sha256 or ""),
            "corpus_manifest_path": os.path.abspath(str(corpus_manifest_path))
            if str(corpus_manifest_path or "").strip()
            else "",
            "corpus_manifest_sha256": str(corpus_manifest_sha256 or ""),
            "live_snapshot_digest": str(live_snapshot_digest or ""),
            "source_file_digests": dict(sorted((source_file_digests or {}).items())),
        },
        "candidates": candidates,
        "summary": summary,
        "selection": {
            "confirmed": False,
            "selected_identity_v2": [
                item["identity_v2"] for item in candidates if item.get("selected")
            ],
        },
    }
    digest = canonical_digest(_stage_payload(document))
    document["staged_selection_digest"] = digest
    document["session"]["staged_selection_digest"] = digest
    document["selection"]["staged_selection_digest"] = digest
    document["selection"]["selection_digest"] = canonical_digest(
        {
            "staged_selection_digest": digest,
            "selected_identity_v2": document["selection"]["selected_identity_v2"],
            "confirmed": False,
        }
    )
    return document


def make_selection_request(
    stage: Mapping[str, Any],
    selected_identity_v2: Sequence[str],
    *,
    operation_id: str = "",
) -> dict[str, Any]:
    """Create an explicit confirmation request for a staged document."""

    validate_staged_selection(stage)
    session = stage.get("session") or {}
    stage_operation = str(session.get("operation_identity") or "")
    requested_operation = str(operation_id or "").strip() or stage_operation
    if requested_operation != stage_operation:
        raise ValueError("selection operation_identity does not match the staged session")
    candidate_map = {
        str(item.get("identity_v2") or ""): item
        for item in stage.get("candidates") or []
    }
    ids = [str(identity or "").strip() for identity in selected_identity_v2]
    if any(not identity for identity in ids):
        raise ValueError("selection identities must be non-empty")
    if len(set(ids)) != len(ids):
        raise ValueError("selection identities must be unique")
    unavailable = [
        identity
        for identity in ids
        if identity not in candidate_map or not candidate_map[identity].get("selectable")
    ]
    if unavailable:
        raise ValueError(
            "selection includes invalid, stale, conflict, or unknown candidates: "
            + ", ".join(unavailable)
        )
    # Preserve stage order so the request is deterministic and human-auditable.
    ordered_ids = [
        str(item.get("identity_v2") or "")
        for item in stage.get("candidates") or []
        if str(item.get("identity_v2") or "") in set(ids)
    ]
    request: dict[str, Any] = {
        "schema_version": STAGED_SELECTION_SCHEMA_VERSION,
        "kind": SELECTION_REQUEST_KIND,
        "confirmed": True,
        "operation_identity": stage_operation,
        "staged_selection_digest": str(stage.get("staged_selection_digest") or ""),
        "selected_identity_v2": ordered_ids,
    }
    request["selection_digest"] = canonical_digest(request)
    return request


def validate_staged_selection(stage: Mapping[str, Any]) -> None:
    """Validate the immutable envelope of a staged-selection artifact."""

    if stage.get("kind") != STAGED_SELECTION_KIND:
        raise ValueError("staged selection kind is unsupported")
    if stage.get("schema_version") != STAGED_SELECTION_SCHEMA_VERSION:
        raise ValueError("staged selection schema/version is unsupported")
    session = stage.get("session")
    if not isinstance(session, Mapping) or not str(session.get("operation_identity") or ""):
        raise ValueError("staged selection session operation_identity is required")
    digest = str(stage.get("staged_selection_digest") or "")
    if not digest or digest != canonical_digest(_stage_payload(stage)):
        raise ValueError("staged selection digest does not match its contents")
    candidates = stage.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("staged selection candidates must be an array")
    candidate_keys: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("staged selection candidate must be an object")
        identity = str(candidate.get("identity_v2") or "").strip()
        if not identity:
            raise ValueError("staged selection candidate identity_v2 is required")
        candidate_key = str(candidate.get("candidate_key") or "").strip()
        if not candidate_key:
            raise ValueError("staged selection candidate candidate_key is required")
        if candidate_key in candidate_keys:
            raise ValueError("staged selection candidate keys must be unique")
        candidate_keys.add(candidate_key)


def validate_selection_request(
    stage: Mapping[str, Any],
    request: Mapping[str, Any],
) -> list[str]:
    """Validate a confirmed request and return selected identities in order."""

    validate_staged_selection(stage)
    if request.get("kind") != SELECTION_REQUEST_KIND:
        raise ValueError("selection request kind is unsupported")
    if request.get("schema_version") != STAGED_SELECTION_SCHEMA_VERSION:
        raise ValueError("selection request schema/version is unsupported")
    if request.get("confirmed") is not True:
        raise ValueError("selection request must be explicitly confirmed")
    if str(request.get("operation_identity") or "") != str(
        (stage.get("session") or {}).get("operation_identity") or ""
    ):
        raise ValueError("selection request operation_identity is stale")
    if str(request.get("staged_selection_digest") or "") != str(
        stage.get("staged_selection_digest") or ""
    ):
        raise ValueError("selection request staged-selection digest is stale")
    expected_request_digest = dict(request)
    expected_request_digest.pop("selection_digest", None)
    if str(request.get("selection_digest") or "") != canonical_digest(expected_request_digest):
        raise ValueError("selection request digest does not match its contents")
    identities = request.get("selected_identity_v2")
    if not isinstance(identities, list):
        raise ValueError("selection request selected_identity_v2 must be an array")
    candidates = {
        str(candidate.get("identity_v2") or ""): candidate
        for candidate in stage.get("candidates") or []
    }
    normalized = [str(identity or "").strip() for identity in identities]
    if len(set(normalized)) != len(normalized):
        raise ValueError("selection request identities must be unique")
    for identity in normalized:
        candidate = candidates.get(identity)
        if candidate is None or not candidate.get("selectable"):
            raise ValueError(
                "selection request includes invalid, stale, conflict, or unknown candidate: "
                + identity
            )
    return normalized


def filter_candidates(
    stage: Mapping[str, Any],
    *,
    reason: str = "",
    file_rel_path: str = "",
    status: str = "",
    valid_only: bool = False,
) -> list[dict[str, Any]]:
    """Filter staged candidates without changing their selection state."""

    validate_staged_selection(stage)
    reason_text = str(reason or "").strip()
    file_text = str(file_rel_path or "").strip()
    status_text = str(status or "").strip()
    result: list[dict[str, Any]] = []
    for candidate in stage.get("candidates") or []:
        if reason_text and str(candidate.get("reason") or "") != reason_text:
            continue
        if file_text and str(candidate.get("file_rel_path") or "") != file_text:
            continue
        if status_text and str(candidate.get("status") or "") != status_text:
            continue
        if valid_only and not candidate.get("selectable"):
            continue
        result.append(dict(candidate))
    return result


def write_staged_selection(path: str, stage: Mapping[str, Any]) -> str:
    """Persist a staged-selection document through the shared atomic writer."""

    validate_staged_selection(stage)
    target = os.path.abspath(str(path or "").strip())
    if not target:
        raise ValueError("staged selection path is required")
    atomic_write_json(target, dict(stage), ensure_ascii=False, indent=2)
    return target


def load_staged_selection(path: str) -> dict[str, Any]:
    """Load and validate one staged-selection artifact."""

    target = os.path.abspath(str(path or "").strip())
    with open(target, "r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("staged selection must be a JSON object")
    document = dict(value)
    validate_staged_selection(document)
    return document


def write_selection_request(path: str, request: Mapping[str, Any]) -> str:
    """Persist an explicit selection request atomically."""

    target = os.path.abspath(str(path or "").strip())
    if not target:
        raise ValueError("selection request path is required")
    atomic_write_json(target, dict(request), ensure_ascii=False, indent=2)
    return target


def load_selection_request(path: str) -> dict[str, Any]:
    """Load one selection request; binding to a stage is checked separately."""

    target = os.path.abspath(str(path or "").strip())
    with open(target, "r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("selection request must be a JSON object")
    return dict(value)
