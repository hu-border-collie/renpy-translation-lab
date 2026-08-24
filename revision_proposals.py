"""Structured revision-proposal contract and live-project validation.

The importer deliberately stops at validated candidates.  It never edits Ren'Py
files; callers must hand candidates to the existing revision preview/apply gates.
"""
from __future__ import annotations

import json
import ntpath
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import revision_corpus

PROPOSAL_SCHEMA_VERSION = 1
IMPORT_REPORT_SCHEMA_VERSION = 1
SUPPORTED_PRODUCERS = frozenset({"human", "agent"})
SELECTED_DISPOSITIONS = frozenset({"accepted", "approve", "approved", "proposed", "selected"})
STALE_DIAGNOSTIC_CODES = frozenset(
    {
        "CORPUS_SNAPSHOT_INCONSISTENT",
        "LIVE_SOURCE_CHANGED_DURING_IMPORT",
    }
)


@dataclass(frozen=True)
class ProposalValidation:
    """Normalized selected proposals and deterministic diagnostics."""

    proposals: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]
    input_count: int
    requested_selected_count: int
    selected_count: int
    status: str


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load proposal JSONL and retain row numbers for actionable diagnostics."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"proposal JSONL row {line_number} is invalid: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"proposal JSONL row {line_number} must be an object")
            row = dict(value)
            row["_row_number"] = line_number
            rows.append(row)
    return rows


def find_corpus_manifest(proposal_path: str, explicit_path: str = "") -> str:
    """Resolve an explicit or same-directory revision corpus manifest."""
    if str(explicit_path or "").strip():
        return os.path.abspath(str(explicit_path).strip())
    candidate = os.path.join(
        os.path.dirname(os.path.abspath(proposal_path)),
        revision_corpus.CORPUS_MANIFEST_NAME,
    )
    return candidate if os.path.isfile(candidate) else ""


def load_corpus_manifest(path: str) -> dict[str, Any]:
    """Load and minimally validate a revision-corpus manifest."""
    with open(path, "r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("kind") != "revision_corpus":
        raise ValueError("corpus manifest must be a revision_corpus object")
    if value.get("schema_version") != revision_corpus.REVISION_CORPUS_SCHEMA_VERSION:
        raise ValueError("corpus manifest schema/version is unsupported")
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("corpus manifest source must be an object")
    if not isinstance(source.get("snapshot_digest"), str) or not source.get("snapshot_digest").strip():
        raise ValueError("corpus manifest source.snapshot_digest must be a non-empty string")
    if not isinstance(value.get("project"), Mapping):
        raise ValueError("corpus manifest project must be an object")
    return value


def _diag(code: str, row: Mapping[str, Any] | None, message: str, **details: Any) -> dict[str, Any]:
    result = {
        "code": code,
        "message": message,
        "row": int((row or {}).get("_row_number") or 0),
        "occurrence_id": str(
            (row or {}).get("occurrence_id") or (row or {}).get("identity_v2") or ""
        ),
    }
    result.update(details)
    return result


def diagnostics_are_stale(diagnostics: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether diagnostics require re-exporting/reloading source state."""
    return any(
        str(item.get("code") or "").endswith("STALE")
        or str(item.get("code") or "") in STALE_DIAGNOSTIC_CODES
        for item in diagnostics
    )


def _uses_windows_path_syntax(value: str) -> bool:
    return bool(
        ntpath.splitdrive(value)[0]
        or value.startswith("\\\\")
        or (os.name == "nt" and "\\" in value)
    )


def normalize_project_path(value: Any, *, base_dir: Any = "") -> str:
    """Normalize a project path against the shared game-root semantics.

    ``tl_dir`` may be serialized as either an absolute path or a path relative
    to ``game_root``.  The path flavour is selected from the supplied values so
    Windows exports remain deterministic when validated on a non-Windows host.
    """

    text = str(value or "").strip()
    base = str(base_dir or "").strip()
    if not text:
        return ""
    if _uses_windows_path_syntax(text) or _uses_windows_path_syntax(base):
        if base and not ntpath.isabs(text):
            text = ntpath.join(normalize_project_path(base), text)
        return ntpath.normcase(ntpath.abspath(text))
    text = text.replace("\\", "/")
    if base and not os.path.isabs(text):
        text = os.path.join(base.replace("\\", "/"), text)
    return os.path.normcase(os.path.abspath(text))


def normalize_project_identity(
    project_identity: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Return one canonical game-root/TL-directory project identity."""

    values = project_identity or {}
    game_root = normalize_project_path(
        values.get("game_root") or values.get("base_dir")
    )
    return {
        "game_root": game_root,
        "tl_dir": normalize_project_path(values.get("tl_dir"), base_dir=game_root),
    }


def project_identity_from_paths(
    *,
    game_root: Any = "",
    tl_dir: Any = "",
) -> dict[str, str]:
    """Build the canonical identity shared by CLI, GUI, and validation."""

    return normalize_project_identity({"game_root": game_root, "tl_dir": tl_dir})


def project_identities_match(
    expected: Mapping[str, Any] | None,
    actual: Mapping[str, Any] | None,
) -> bool:
    """Compare project identities after resolving TL paths from game roots."""

    return normalize_project_identity(expected) == normalize_project_identity(actual)


def _normalized_relative_path(value: Any) -> str:
    """Normalize proposal file separators without weakening identity checks."""
    return os.path.normpath(str(value or "").replace("\\", "/")).replace("\\", "/")


def validate(
    rows: Sequence[Mapping[str, Any]],
    live_items: Mapping[str, Mapping[str, Any]],
    *,
    live_snapshot_digest: str,
    live_project_identity: Mapping[str, Any] | None = None,
    corpus_manifest: Mapping[str, Any] | None = None,
) -> ProposalValidation:
    """Validate selected proposal rows against exact live identity-v2 items.

    Structural, identity, source/current-text, item-snapshot and corpus-snapshot
    failures are all hard blockers.  Unselected rows remain auditable input but
    are not candidates.
    """
    diagnostics: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    seen: dict[str, Mapping[str, Any]] = {}
    requested_selected_count = sum(row.get("selected") is True for row in rows)
    live_identity = normalize_project_identity(live_project_identity)
    live_tl_dir = live_identity["tl_dir"]
    manifest_project = (corpus_manifest or {}).get("project") or {}
    if not isinstance(manifest_project, Mapping):
        manifest_project = {}
    manifest_identity = normalize_project_identity(
        {
            "game_root": (
                manifest_project.get("game_root")
                or manifest_project.get("base_dir")
                or live_identity.get("game_root")
            ),
            "tl_dir": manifest_project.get("tl_dir"),
        }
    )
    manifest_tl_dir = manifest_identity["tl_dir"]
    manifest_source = (corpus_manifest or {}).get("source") or {}
    if not isinstance(manifest_source, Mapping):
        manifest_source = {}
    expected_corpus_digest = str(manifest_source.get("snapshot_digest") or "")
    source_changed = bool(manifest_source.get("source_changed_during_scan"))
    if source_changed:
        diagnostics.append(_diag("CORPUS_SNAPSHOT_INCONSISTENT", None, "corpus source changed during export"))
    if expected_corpus_digest and expected_corpus_digest != live_snapshot_digest:
        diagnostics.append(_diag("CORPUS_SNAPSHOT_STALE", None, "corpus snapshot does not match the live project"))
    if corpus_manifest and not manifest_tl_dir:
        diagnostics.append(_diag("MISSING_PROJECT_IDENTITY", None, "corpus manifest project.tl_dir is required"))
    elif manifest_tl_dir and live_tl_dir and manifest_tl_dir != live_tl_dir:
        diagnostics.append(_diag("PROJECT_IDENTITY_STALE", None, "corpus project identity does not match the live project"))

    for raw in rows:
        row = dict(raw)
        identity = str(row.get("occurrence_id") or row.get("identity_v2") or "").strip()
        occurrence_id = str(row.get("occurrence_id") or "").strip()
        identity_v2 = str(row.get("identity_v2") or "").strip()
        selected_value = row.get("selected")
        disposition = str(row.get("disposition") or "").strip().lower()
        if row.get("schema_version") != PROPOSAL_SCHEMA_VERSION:
            diagnostics.append(_diag("UNSUPPORTED_SCHEMA_VERSION", row, "proposal schema_version is unsupported"))
            continue
        if not isinstance(selected_value, bool):
            diagnostics.append(_diag("INVALID_SELECTED", row, "selected must be a boolean"))
            continue
        if not disposition:
            diagnostics.append(_diag("MISSING_DISPOSITION", row, "disposition is required"))
            continue
        if not selected_value:
            continue
        if disposition not in SELECTED_DISPOSITIONS:
            diagnostics.append(_diag("INVALID_SELECTED_DISPOSITION", row, "selected proposal has a non-accepting disposition"))
            continue
        row_project = row.get("project_identity")
        if not isinstance(row_project, Mapping):
            row_project = {}
        row_identity = normalize_project_identity(
            {
                "game_root": (
                    row_project.get("game_root")
                    or row_project.get("base_dir")
                    or live_identity.get("game_root")
                ),
                "tl_dir": row_project.get("tl_dir"),
            }
        )
        row_tl_dir = row_identity["tl_dir"]
        if not corpus_manifest and not row_tl_dir:
            diagnostics.append(_diag("MISSING_PROJECT_IDENTITY", row, "project_identity.tl_dir or a companion corpus manifest is required"))
            continue
        if manifest_tl_dir and row_tl_dir and row_tl_dir != manifest_tl_dir:
            diagnostics.append(_diag("PROJECT_IDENTITY_CONFLICT", row, "proposal project identity conflicts with the corpus manifest"))
            continue
        if not corpus_manifest and live_tl_dir and row_tl_dir != live_tl_dir:
            diagnostics.append(_diag("PROJECT_IDENTITY_STALE", row, "proposal project identity does not match the live project"))
            continue
        producer = row.get("producer")
        producer_type = str((producer or {}).get("type") or "").strip().lower() if isinstance(producer, Mapping) else ""
        row_invalid = False
        if producer_type not in SUPPORTED_PRODUCERS:
            diagnostics.append(_diag("INVALID_PRODUCER", row, "producer.type must be human or agent"))
            row_invalid = True
        if not identity:
            diagnostics.append(_diag("MISSING_OCCURRENCE_ID", row, "occurrence_id/identity_v2 is required"))
            continue
        if occurrence_id and identity_v2 and occurrence_id != identity_v2:
            diagnostics.append(_diag("IDENTITY_MISMATCH", row, "occurrence_id and identity_v2 must identify the same occurrence"))
            continue
        if identity in seen:
            previous = seen[identity]
            previous_text = str(previous.get("proposed_translation") or "").strip()
            current_text = str(row.get("proposed_translation") or "").strip()
            code = "CONFLICTING_PROPOSAL" if previous_text != current_text else "DUPLICATE_OCCURRENCE_ID"
            diagnostics.append(_diag(code, row, "occurrence appears more than once", first_row=int(previous.get("_row_number") or 0)))
            continue
        seen[identity] = row
        live = live_items.get(identity)
        if live is None:
            diagnostics.append(_diag("UNKNOWN_OCCURRENCE_ID", row, "occurrence is not present in the live project"))
            continue
        mismatch = row_invalid
        if _normalized_relative_path(row.get("file_rel_path")) != _normalized_relative_path(
            live.get("file_rel_path")
        ):
            diagnostics.append(_diag("FILE_PATH_MISMATCH", row, "file_rel_path does not match the live occurrence"))
            mismatch = True
        for field, value, expected, type_code, mismatch_code in (
            (
                "source",
                row.get("source"),
                str(live.get("source") or ""),
                "INVALID_SOURCE_TYPE",
                "SOURCE_MISMATCH",
            ),
            (
                "current_translation",
                row.get("current_translation"),
                str(live.get("current_translation") or ""),
                "INVALID_CURRENT_TRANSLATION_TYPE",
                "CURRENT_TRANSLATION_STALE",
            ),
        ):
            if not isinstance(value, str):
                diagnostics.append(_diag(type_code, row, f"{field} must be a string"))
                mismatch = True
            elif value != expected:
                diagnostics.append(_diag(mismatch_code, row, f"{field} does not match the live occurrence"))
                mismatch = True
        proposed_value = row.get("proposed_translation")
        proposed = proposed_value.strip() if isinstance(proposed_value, str) else ""
        if not isinstance(proposed_value, str):
            diagnostics.append(_diag("INVALID_PROPOSED_TRANSLATION_TYPE", row, "proposed_translation must be a string"))
            mismatch = True
        elif not proposed:
            diagnostics.append(_diag("EMPTY_PROPOSED_TRANSLATION", row, "proposed_translation must not be empty"))
            mismatch = True
        reason_value = row.get("reason")
        reason = reason_value.strip() if isinstance(reason_value, str) else ""
        if not isinstance(reason_value, str):
            diagnostics.append(_diag("INVALID_REASON_TYPE", row, "reason must be a string"))
            mismatch = True
        elif not reason:
            diagnostics.append(_diag("MISSING_REASON", row, "reason is required for every selected proposal"))
            mismatch = True
        snapshot_digest = row.get("snapshot_digest")
        item_snapshot_digest = row.get("item_snapshot_digest")
        if (
            snapshot_digest is not None
            and item_snapshot_digest is not None
            and str(snapshot_digest or "") != str(item_snapshot_digest or "")
        ):
            diagnostics.append(
                _diag(
                    "ITEM_SNAPSHOT_CONFLICT",
                    row,
                    "snapshot_digest and item_snapshot_digest must identify the same item snapshot",
                )
            )
            mismatch = True
        item_digest = str(snapshot_digest or item_snapshot_digest or "")
        expected_item_digest = revision_corpus.item_snapshot_digest(
            str(live.get("source") or ""), str(live.get("current_translation") or "")
        )
        if not item_digest:
            diagnostics.append(_diag("MISSING_ITEM_SNAPSHOT", row, "snapshot_digest/item_snapshot_digest is required"))
            mismatch = True
        elif item_digest != expected_item_digest:
            diagnostics.append(_diag("ITEM_SNAPSHOT_STALE", row, "proposal item snapshot is stale"))
            mismatch = True
        row_corpus_digest = str(row.get("corpus_snapshot_digest") or "")
        effective_digest = row_corpus_digest or expected_corpus_digest
        if not effective_digest:
            diagnostics.append(_diag("MISSING_CORPUS_SNAPSHOT", row, "corpus_snapshot_digest or companion corpus manifest is required"))
            mismatch = True
        elif effective_digest != live_snapshot_digest:
            diagnostics.append(_diag("CORPUS_SNAPSHOT_STALE", row, "proposal corpus snapshot is stale"))
            mismatch = True
        if not mismatch:
            row["occurrence_id"] = identity
            row["identity_v2"] = identity
            row["proposed_translation"] = proposed
            row["reason"] = reason
            selected.append(row)

    status = (
        "no_op"
        if not selected and not diagnostics
        else "stale"
        if diagnostics_are_stale(diagnostics)
        else "blocked"
        if diagnostics
        else "imported"
    )
    return ProposalValidation(
        tuple(selected),
        tuple(diagnostics),
        len(rows),
        requested_selected_count,
        len(selected),
        status,
    )
