"""Project Analysis Phase 1: artifact contract, fingerprints, and invalidation.

This module owns the shared schema, atomic I/O helpers, lineage fingerprints,
publish-state evaluation, and partial invalidation planner used by later
route-aware generation (#254) and final-review campaigns (#255).

Phase 1 intentionally does **not** call LLMs or inject analysis text into
translation prompts. Downstream code must reuse these contracts rather than
re-implement status or fingerprint logic in the GUI.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from atomic_io import atomic_write_json, atomic_write_jsonl, atomic_write_text

SCHEMA_VERSION = 1
STORE_NAME = "project_analysis"

STATUS_MISSING = "missing"
STATUS_DRAFT = "draft"
STATUS_REVIEW_REQUIRED = "review_required"
STATUS_PUBLISHED = "published"
STATUS_STALE = "stale"
STATUS_FAILED = "failed"

VALID_STATUSES = frozenset(
    {
        STATUS_MISSING,
        STATUS_DRAFT,
        STATUS_REVIEW_REQUIRED,
        STATUS_PUBLISHED,
        STATUS_STALE,
        STATUS_FAILED,
    }
)

# Stored artifact records use these (missing is derived when files are absent).
RECORD_STATUSES = frozenset(
    {
        STATUS_DRAFT,
        STATUS_REVIEW_REQUIRED,
        STATUS_PUBLISHED,
        STATUS_STALE,
        STATUS_FAILED,
    }
)

KIND_CHUNK = "chunk"
KIND_SCENE = "scene"
KIND_LABEL = "label"
KIND_ROUTE = "route"
KIND_PROJECT_BRIEF = "project_brief"

VALID_KINDS = frozenset(
    {
        KIND_CHUNK,
        KIND_SCENE,
        KIND_LABEL,
        KIND_ROUTE,
        KIND_PROJECT_BRIEF,
    }
)

MANIFEST_FILENAME = "manifest.json"
CHUNK_SUMMARIES_FILENAME = "chunk_summaries.jsonl"
SCENE_SUMMARIES_FILENAME = "scene_summaries.jsonl"
LABEL_SUMMARIES_FILENAME = "label_summaries.jsonl"
ROUTE_SUMMARIES_FILENAME = "route_summaries.json"
PROJECT_BRIEF_DRAFT_FILENAME = "project_brief.draft.md"
PROJECT_BRIEF_PUBLISHED_FILENAME = "project_brief.published.md"

JSONL_FILENAMES = {
    KIND_CHUNK: CHUNK_SUMMARIES_FILENAME,
    KIND_SCENE: SCENE_SUMMARIES_FILENAME,
    KIND_LABEL: LABEL_SUMMARIES_FILENAME,
}

INJECTABLE_STATUSES = frozenset({STATUS_PUBLISHED})


class ProjectAnalysisError(ValueError):
    """Base error for project-analysis contract violations."""


class ProjectAnalysisPathEscapeError(ProjectAnalysisError):
    """Raised when a path would escape the analysis store directory."""


class ProjectAnalysisSchemaError(ProjectAnalysisError):
    """Raised when schema/version or record shape is invalid."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _as_str_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProjectAnalysisSchemaError(f"{field_name} must be a list of strings")
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _as_optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_line_span(value: Any) -> list[int] | None:
    if value is None or value == []:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ProjectAnalysisSchemaError("line_span must be [start, end] or null")
    try:
        start = int(value[0])
        end = int(value[1])
    except (TypeError, ValueError) as exc:
        raise ProjectAnalysisSchemaError("line_span values must be integers") from exc
    if start < 0 or end < start:
        raise ProjectAnalysisSchemaError("line_span must satisfy 0 <= start <= end")
    return [start, end]


def normalize_status(value: Any, *, allow_missing: bool = False) -> str:
    status = _as_optional_str(value).lower() or STATUS_MISSING
    allowed = VALID_STATUSES if allow_missing else RECORD_STATUSES
    if status not in allowed:
        raise ProjectAnalysisSchemaError(
            f"unknown status {status!r}; expected one of {sorted(allowed)}"
        )
    return status


def normalize_kind(value: Any) -> str:
    kind = _as_optional_str(value).lower()
    if kind not in VALID_KINDS:
        raise ProjectAnalysisSchemaError(
            f"unknown kind {kind!r}; expected one of {sorted(VALID_KINDS)}"
        )
    return kind


def empty_lineage(
    *,
    source_fingerprint: str = "",
    prompt_schema_version: str = "",
    provider: str = "",
    model: str = "",
    thinking_level: str = "",
    upstream_dependency_digest: str = "",
    generated_at: str = "",
    reviewed_at: str = "",
    published_at: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": str(source_fingerprint or ""),
        "prompt_schema_version": str(prompt_schema_version or ""),
        "provider": str(provider or ""),
        "model": str(model or ""),
        "thinking_level": str(thinking_level or ""),
        "upstream_dependency_digest": str(upstream_dependency_digest or ""),
        "generated_at": str(generated_at or ""),
        "reviewed_at": str(reviewed_at or ""),
        "published_at": str(published_at or ""),
    }


def normalize_lineage(value: Any) -> dict[str, Any]:
    if value is None:
        return empty_lineage()
    if not isinstance(value, Mapping):
        raise ProjectAnalysisSchemaError("lineage must be an object")
    schema_version = value.get("schema_version", SCHEMA_VERSION)
    try:
        schema_version_int = int(schema_version)
    except (TypeError, ValueError) as exc:
        raise ProjectAnalysisSchemaError("lineage.schema_version must be an integer") from exc
    if schema_version_int != SCHEMA_VERSION:
        raise ProjectAnalysisSchemaError(
            f"unsupported lineage schema_version {schema_version_int}; "
            f"expected {SCHEMA_VERSION}"
        )
    return empty_lineage(
        source_fingerprint=_as_optional_str(value.get("source_fingerprint")),
        prompt_schema_version=_as_optional_str(value.get("prompt_schema_version")),
        provider=_as_optional_str(value.get("provider")),
        model=_as_optional_str(value.get("model")),
        thinking_level=_as_optional_str(value.get("thinking_level")),
        upstream_dependency_digest=_as_optional_str(value.get("upstream_dependency_digest")),
        generated_at=_as_optional_str(value.get("generated_at")),
        reviewed_at=_as_optional_str(value.get("reviewed_at")),
        published_at=_as_optional_str(value.get("published_at")),
    )


def lineage_digest(lineage: Mapping[str, Any]) -> str:
    """Stable digest of fields that affect freshness (not review timestamps alone)."""
    normalized = normalize_lineage(lineage)
    payload = {
        "schema_version": normalized["schema_version"],
        "source_fingerprint": normalized["source_fingerprint"],
        "prompt_schema_version": normalized["prompt_schema_version"],
        "provider": normalized["provider"],
        "model": normalized["model"],
        "thinking_level": normalized["thinking_level"],
        "upstream_dependency_digest": normalized["upstream_dependency_digest"],
    }
    return stable_json_sha256(payload)


def digest_source_items(items: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint a set of source translation units / evidence rows."""
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        item_id = _as_optional_str(item.get("id") or item.get("item_id") or item.get("unit_id"))
        checksum = _as_optional_str(
            item.get("source_checksum") or item.get("checksum") or item.get("source_text_checksum")
        )
        if not checksum and item.get("source_text") is not None:
            checksum = sha256_text(str(item.get("source_text")))
        file_rel = _as_optional_str(item.get("file_rel_path") or item.get("file"))
        rows.append(
            {
                "id": item_id,
                "file_rel_path": file_rel,
                "source_checksum": checksum,
            }
        )
    rows.sort(key=lambda row: (row["id"], row["file_rel_path"], row["source_checksum"]))
    return stable_json_sha256(rows)


def digest_upstream_artifacts(artifact_ids: Sequence[str]) -> str:
    cleaned = sorted({str(x).strip() for x in artifact_ids if str(x or "").strip()})
    return stable_json_sha256(cleaned)


def normalize_summary_record(raw: Any, *, default_kind: str | None = None) -> dict[str, Any]:
    """Validate and normalize one summary / brief artifact record."""
    if not isinstance(raw, Mapping):
        raise ProjectAnalysisSchemaError("summary record must be an object")

    kind = normalize_kind(raw.get("kind") or default_kind)
    artifact_id = _as_optional_str(raw.get("id") or raw.get("artifact_id"))
    if not artifact_id:
        raise ProjectAnalysisSchemaError("summary record requires a stable id")

    status = normalize_status(raw.get("status") or STATUS_DRAFT, allow_missing=False)
    lineage = normalize_lineage(raw.get("lineage") or raw.get("fingerprint"))

    record = {
        "id": artifact_id,
        "kind": kind,
        "status": status,
        "source_files": _as_str_list(raw.get("source_files") or raw.get("files"), field_name="source_files"),
        "label_id": _as_optional_str(raw.get("label_id")),
        "scene_id": _as_optional_str(raw.get("scene_id")),
        "route_id": _as_optional_str(raw.get("route_id")),
        "evidence_item_ids": _as_str_list(
            raw.get("evidence_item_ids") or raw.get("summary_evidence_item_ids"),
            field_name="evidence_item_ids",
        ),
        "line_span": _as_line_span(raw.get("line_span")),
        "source_checksum": _as_optional_str(raw.get("source_checksum")),
        "upstream_artifact_ids": _as_str_list(
            raw.get("upstream_artifact_ids"),
            field_name="upstream_artifact_ids",
        ),
        "lineage": lineage,
        "summary": _as_optional_str(raw.get("summary") or raw.get("body") or raw.get("text")),
        "error": _as_optional_str(raw.get("error")),
    }

    # Preserve optional extension fields without treating them as schema authority.
    for key in ("title", "notes", "metadata"):
        if key in raw and raw[key] is not None:
            record[key] = raw[key]
    return record


def evaluate_record_status(
    record: Mapping[str, Any],
    *,
    expected_source_fingerprint: str = "",
    expected_upstream_digest: str = "",
    expected_prompt_schema_version: str = "",
    expected_provider: str = "",
    expected_model: str = "",
    expected_thinking_level: str = "",
) -> str:
    """Return effective status; published becomes stale on fingerprint mismatch."""
    status = normalize_status(record.get("status") or STATUS_DRAFT, allow_missing=False)
    if status in {STATUS_FAILED, STATUS_STALE, STATUS_DRAFT, STATUS_REVIEW_REQUIRED}:
        return status
    if status != STATUS_PUBLISHED:
        return status

    lineage = normalize_lineage(record.get("lineage"))
    checks = (
        (expected_source_fingerprint, lineage["source_fingerprint"]),
        (expected_upstream_digest, lineage["upstream_dependency_digest"]),
        (expected_prompt_schema_version, lineage["prompt_schema_version"]),
        (expected_provider, lineage["provider"]),
        (expected_model, lineage["model"]),
        (expected_thinking_level, lineage["thinking_level"]),
    )
    for expected, actual in checks:
        if expected and actual and expected != actual:
            return STATUS_STALE
        if expected and not actual:
            return STATUS_STALE
    return STATUS_PUBLISHED


def is_injectable_record(
    record: Mapping[str, Any],
    *,
    expected_source_fingerprint: str = "",
    expected_upstream_digest: str = "",
    expected_prompt_schema_version: str = "",
    expected_provider: str = "",
    expected_model: str = "",
    expected_thinking_level: str = "",
) -> bool:
    """Only published artifacts with matching fingerprints may be injected later."""
    effective = evaluate_record_status(
        record,
        expected_source_fingerprint=expected_source_fingerprint,
        expected_upstream_digest=expected_upstream_digest,
        expected_prompt_schema_version=expected_prompt_schema_version,
        expected_provider=expected_provider,
        expected_model=expected_model,
        expected_thinking_level=expected_thinking_level,
    )
    return effective in INJECTABLE_STATUSES


@dataclass
class InvalidationPlan:
    """Artifacts that must be marked stale after an upstream change."""

    stale_artifact_ids: set[str] = field(default_factory=set)
    stale_by_kind: dict[str, set[str]] = field(default_factory=dict)
    brief_stale: bool = False
    reasons: dict[str, list[str]] = field(default_factory=dict)

    def add(self, artifact_id: str, kind: str, reason: str) -> None:
        artifact_id = str(artifact_id or "").strip()
        if not artifact_id:
            return
        self.stale_artifact_ids.add(artifact_id)
        self.stale_by_kind.setdefault(kind, set()).add(artifact_id)
        self.reasons.setdefault(artifact_id, []).append(reason)
        if kind == KIND_PROJECT_BRIEF:
            self.brief_stale = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "stale_artifact_ids": sorted(self.stale_artifact_ids),
            "stale_by_kind": {
                kind: sorted(ids) for kind, ids in sorted(self.stale_by_kind.items())
            },
            "brief_stale": self.brief_stale,
            "reasons": {key: list(vals) for key, vals in sorted(self.reasons.items())},
        }


def plan_invalidation(
    *,
    chunks: Sequence[Mapping[str, Any]] = (),
    scenes: Sequence[Mapping[str, Any]] = (),
    labels: Sequence[Mapping[str, Any]] = (),
    routes: Sequence[Mapping[str, Any]] = (),
    project_brief: Mapping[str, Any] | None = None,
    changed_item_ids: Iterable[str] | None = None,
    changed_artifact_ids: Iterable[str] | None = None,
    always_stale_brief_on_any: bool = True,
) -> InvalidationPlan:
    """Mark dependent artifacts stale when source items or upstream artifacts change.

    Rules (Phase 1 contract):
    - A changed evidence item invalidates chunks that cite it, then scenes/labels
      that depend on those chunks, then routes that depend on those labels/scenes,
      then the global project brief.
    - Unrelated routes stay valid when their dependency closure is untouched.
    - Prompt/schema/provider/model changes are expressed by feeding the affected
      artifact ids into ``changed_artifact_ids`` (caller computes digest mismatch).
    """
    plan = InvalidationPlan()
    changed_items = {str(x).strip() for x in (changed_item_ids or []) if str(x or "").strip()}
    seed_artifacts = {
        str(x).strip() for x in (changed_artifact_ids or []) if str(x or "").strip()
    }

    chunk_records = [normalize_summary_record(r, default_kind=KIND_CHUNK) for r in chunks]
    scene_records = [normalize_summary_record(r, default_kind=KIND_SCENE) for r in scenes]
    label_records = [normalize_summary_record(r, default_kind=KIND_LABEL) for r in labels]
    route_records = [normalize_summary_record(r, default_kind=KIND_ROUTE) for r in routes]

    # 1) Seed from evidence items → chunks.
    for chunk in chunk_records:
        if seed_artifacts and chunk["id"] in seed_artifacts:
            plan.add(chunk["id"], KIND_CHUNK, "artifact marked changed")
            continue
        if changed_items and changed_items.intersection(chunk["evidence_item_ids"]):
            hit = sorted(changed_items.intersection(chunk["evidence_item_ids"]))
            plan.add(chunk["id"], KIND_CHUNK, f"evidence items changed: {', '.join(hit)}")

    stale_ids = set(plan.stale_artifact_ids)

    def _depends_on_stale(record: Mapping[str, Any]) -> list[str]:
        deps = set(record.get("upstream_artifact_ids") or [])
        # Label/scene may also embed evidence items directly.
        if changed_items and changed_items.intersection(record.get("evidence_item_ids") or []):
            return sorted(changed_items.intersection(record.get("evidence_item_ids") or []))
        return sorted(deps.intersection(stale_ids | seed_artifacts))

    # 2) Scenes / labels depending on stale chunks or changed items.
    for scene in scene_records:
        if scene["id"] in seed_artifacts:
            plan.add(scene["id"], KIND_SCENE, "artifact marked changed")
            continue
        deps = _depends_on_stale(scene)
        if deps:
            plan.add(scene["id"], KIND_SCENE, f"upstream/evidence changed: {', '.join(deps)}")

    stale_ids = set(plan.stale_artifact_ids)
    for label in label_records:
        if label["id"] in seed_artifacts:
            plan.add(label["id"], KIND_LABEL, "artifact marked changed")
            continue
        deps = _depends_on_stale(label)
        if deps:
            plan.add(label["id"], KIND_LABEL, f"upstream/evidence changed: {', '.join(deps)}")

    # 3) Routes depending on stale labels/scenes/chunks.
    stale_ids = set(plan.stale_artifact_ids)
    for route in route_records:
        if route["id"] in seed_artifacts:
            plan.add(route["id"], KIND_ROUTE, "artifact marked changed")
            continue
        deps = _depends_on_stale(route)
        if deps:
            plan.add(route["id"], KIND_ROUTE, f"upstream/evidence changed: {', '.join(deps)}")

    # 4) Global brief depends on any affected analysis product.
    brief_id = "project_brief"
    if project_brief is not None:
        brief = normalize_summary_record(project_brief, default_kind=KIND_PROJECT_BRIEF)
        brief_id = brief["id"] or brief_id
        if brief_id in seed_artifacts:
            plan.add(brief_id, KIND_PROJECT_BRIEF, "artifact marked changed")
        elif always_stale_brief_on_any and plan.stale_artifact_ids:
            plan.add(
                brief_id,
                KIND_PROJECT_BRIEF,
                "downstream of stale analysis artifacts",
            )
        else:
            deps = _depends_on_stale(brief)
            if deps:
                plan.add(
                    brief_id,
                    KIND_PROJECT_BRIEF,
                    f"upstream/evidence changed: {', '.join(deps)}",
                )
    elif always_stale_brief_on_any and plan.stale_artifact_ids:
        plan.add(brief_id, KIND_PROJECT_BRIEF, "downstream of stale analysis artifacts")

    return plan


def apply_invalidation_to_records(
    records: Sequence[Mapping[str, Any]],
    plan: InvalidationPlan,
    *,
    default_kind: str,
) -> list[dict[str, Any]]:
    """Return copies of records with planned ids flipped to stale."""
    out: list[dict[str, Any]] = []
    for raw in records:
        record = normalize_summary_record(raw, default_kind=default_kind)
        if record["id"] in plan.stale_artifact_ids:
            record = dict(record)
            record["status"] = STATUS_STALE
        out.append(record)
    return out


def get_default_project_analysis_store_dir(base_dir: str | None = None) -> str:
    """Resolve default store path via context_storage settings."""
    from translator_runtime import get_default_context_store_dir

    return get_default_context_store_dir(STORE_NAME, base_dir)


def resolve_under_store(store_dir: str | os.PathLike[str], relative: str) -> str:
    """Resolve *relative* under *store_dir*; reject escapes."""
    root = os.path.realpath(os.path.abspath(os.fspath(store_dir)))
    rel = str(relative or "").replace("\\", "/").lstrip("/")
    if not rel or rel in {".", ".."} or ".." in rel.split("/"):
        raise ProjectAnalysisPathEscapeError(
            f"refusing path that escapes project analysis store: {relative!r}"
        )
    # Absolute or drive-relative inputs are never allowed as relative store paths.
    if os.path.isabs(rel) or re.match(r"^[A-Za-z]:", rel):
        raise ProjectAnalysisPathEscapeError(
            f"refusing absolute path outside project analysis store: {relative!r}"
        )
    candidate = os.path.realpath(os.path.join(root, *rel.split("/")))
    try:
        common = os.path.commonpath([root, candidate])
    except ValueError as exc:
        # Different drives on Windows.
        raise ProjectAnalysisPathEscapeError(
            f"refusing path that escapes project analysis store: {relative!r}"
        ) from exc
    if common != root:
        raise ProjectAnalysisPathEscapeError(
            f"refusing path that escapes project analysis store: {relative!r}"
        )
    return candidate


def empty_manifest(
    *,
    project_identity: Mapping[str, Any] | None = None,
    store_dir: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "store_name": STORE_NAME,
        "store_dir": store_dir,
        "project_identity": dict(project_identity or {}),
        "artifacts": {
            KIND_CHUNK: {"status": STATUS_MISSING, "count": 0, "lineage": empty_lineage()},
            KIND_SCENE: {"status": STATUS_MISSING, "count": 0, "lineage": empty_lineage()},
            KIND_LABEL: {"status": STATUS_MISSING, "count": 0, "lineage": empty_lineage()},
            KIND_ROUTE: {"status": STATUS_MISSING, "count": 0, "lineage": empty_lineage()},
            KIND_PROJECT_BRIEF: {
                "status": STATUS_MISSING,
                "draft_present": False,
                "published_present": False,
                "lineage": empty_lineage(),
            },
        },
        "updated_at": "",
    }


def normalize_manifest(raw: Any, *, store_dir: str = "") -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProjectAnalysisSchemaError("manifest must be an object")
    try:
        schema_version = int(raw.get("schema_version", SCHEMA_VERSION))
    except (TypeError, ValueError) as exc:
        raise ProjectAnalysisSchemaError("manifest.schema_version must be an integer") from exc
    if schema_version != SCHEMA_VERSION:
        raise ProjectAnalysisSchemaError(
            f"unsupported project analysis schema_version {schema_version}; "
            f"expected {SCHEMA_VERSION}"
        )
    base = empty_manifest(
        project_identity=raw.get("project_identity")
        if isinstance(raw.get("project_identity"), Mapping)
        else {},
        store_dir=store_dir or _as_optional_str(raw.get("store_dir")),
    )
    artifacts_in = raw.get("artifacts") if isinstance(raw.get("artifacts"), Mapping) else {}
    for kind in (
        KIND_CHUNK,
        KIND_SCENE,
        KIND_LABEL,
        KIND_ROUTE,
        KIND_PROJECT_BRIEF,
    ):
        entry = artifacts_in.get(kind) if isinstance(artifacts_in.get(kind), Mapping) else {}
        status = normalize_status(entry.get("status") or STATUS_MISSING, allow_missing=True)
        if kind == KIND_PROJECT_BRIEF:
            base["artifacts"][kind] = {
                "status": status,
                "draft_present": bool(entry.get("draft_present", False)),
                "published_present": bool(entry.get("published_present", False)),
                "lineage": normalize_lineage(entry.get("lineage")),
                "id": _as_optional_str(entry.get("id") or "project_brief"),
            }
        else:
            try:
                count = int(entry.get("count", 0) or 0)
            except (TypeError, ValueError) as exc:
                raise ProjectAnalysisSchemaError(f"artifacts.{kind}.count must be an integer") from exc
            base["artifacts"][kind] = {
                "status": status,
                "count": max(0, count),
                "lineage": normalize_lineage(entry.get("lineage")),
            }
    base["updated_at"] = _as_optional_str(raw.get("updated_at"))
    return base


def _aggregate_status(statuses: Sequence[str]) -> str:
    if not statuses:
        return STATUS_MISSING
    unique = set(statuses)
    if STATUS_FAILED in unique:
        return STATUS_FAILED
    if STATUS_STALE in unique:
        return STATUS_STALE
    if STATUS_REVIEW_REQUIRED in unique:
        return STATUS_REVIEW_REQUIRED
    if unique == {STATUS_PUBLISHED}:
        return STATUS_PUBLISHED
    if STATUS_DRAFT in unique or STATUS_PUBLISHED in unique:
        return STATUS_DRAFT
    return next(iter(unique))


def _read_json_object(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ProjectAnalysisSchemaError(f"expected JSON object in {path}")
    return data


def _read_jsonl_objects(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ProjectAnalysisSchemaError(
                    f"invalid JSONL at {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise ProjectAnalysisSchemaError(
                    f"JSONL row at {path}:{line_no} must be an object"
                )
            rows.append(data)
    return rows


class ProjectAnalysisStore:
    """Filesystem-backed project analysis artifact store (Phase 1 I/O)."""

    def __init__(self, store_dir: str | os.PathLike[str]):
        self.store_dir = os.path.abspath(os.fspath(store_dir))

    def ensure_dir(self) -> None:
        os.makedirs(self.store_dir, exist_ok=True)

    def path_for(self, relative: str) -> str:
        return resolve_under_store(self.store_dir, relative)

    @property
    def manifest_path(self) -> str:
        return self.path_for(MANIFEST_FILENAME)

    def artifact_path(self, filename: str) -> str:
        return self.path_for(filename)

    def exists(self) -> bool:
        return os.path.isfile(self.manifest_path) or any(
            os.path.isfile(self.artifact_path(name))
            for name in (
                CHUNK_SUMMARIES_FILENAME,
                SCENE_SUMMARIES_FILENAME,
                LABEL_SUMMARIES_FILENAME,
                ROUTE_SUMMARIES_FILENAME,
                PROJECT_BRIEF_DRAFT_FILENAME,
                PROJECT_BRIEF_PUBLISHED_FILENAME,
            )
        )

    def load_manifest(self) -> dict[str, Any] | None:
        if not os.path.isfile(self.manifest_path):
            return None
        try:
            raw = _read_json_object(self.manifest_path)
        except json.JSONDecodeError as exc:
            raise ProjectAnalysisSchemaError(
                f"corrupt project analysis manifest: {self.manifest_path}: {exc}"
            ) from exc
        return normalize_manifest(raw, store_dir=self.store_dir)

    def save_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        self.ensure_dir()
        normalized = normalize_manifest(manifest, store_dir=self.store_dir)
        normalized["updated_at"] = normalized.get("updated_at") or utc_now_iso()
        atomic_write_json(self.manifest_path, normalized)
        return normalized

    def load_summaries(self, kind: str) -> list[dict[str, Any]]:
        kind = normalize_kind(kind)
        if kind not in JSONL_FILENAMES:
            raise ProjectAnalysisSchemaError(f"{kind} is not a JSONL summary kind")
        path = self.artifact_path(JSONL_FILENAMES[kind])
        if not os.path.isfile(path):
            return []
        rows = _read_jsonl_objects(path)
        return [normalize_summary_record(row, default_kind=kind) for row in rows]

    def save_summaries(self, kind: str, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        kind = normalize_kind(kind)
        if kind not in JSONL_FILENAMES:
            raise ProjectAnalysisSchemaError(f"{kind} is not a JSONL summary kind")
        self.ensure_dir()
        normalized = [normalize_summary_record(row, default_kind=kind) for row in records]
        atomic_write_jsonl(self.artifact_path(JSONL_FILENAMES[kind]), normalized)
        return normalized

    def load_routes(self) -> list[dict[str, Any]]:
        path = self.artifact_path(ROUTE_SUMMARIES_FILENAME)
        if not os.path.isfile(path):
            return []
        try:
            raw = _read_json_object(path)
        except json.JSONDecodeError as exc:
            raise ProjectAnalysisSchemaError(
                f"corrupt route_summaries.json: {path}: {exc}"
            ) from exc
        routes = raw.get("routes")
        if routes is None and isinstance(raw.get("items"), list):
            routes = raw["items"]
        if not isinstance(routes, list):
            raise ProjectAnalysisSchemaError("route_summaries.json must contain a routes list")
        return [normalize_summary_record(row, default_kind=KIND_ROUTE) for row in routes]

    def save_routes(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        self.ensure_dir()
        normalized = [normalize_summary_record(row, default_kind=KIND_ROUTE) for row in records]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "routes": normalized,
            "updated_at": utc_now_iso(),
        }
        atomic_write_json(self.artifact_path(ROUTE_SUMMARIES_FILENAME), payload)
        return normalized

    def load_brief_text(self, *, published: bool) -> str:
        name = (
            PROJECT_BRIEF_PUBLISHED_FILENAME if published else PROJECT_BRIEF_DRAFT_FILENAME
        )
        path = self.artifact_path(name)
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8-sig") as handle:
            return handle.read()

    def save_brief_text(self, text: str, *, published: bool) -> None:
        self.ensure_dir()
        name = (
            PROJECT_BRIEF_PUBLISHED_FILENAME if published else PROJECT_BRIEF_DRAFT_FILENAME
        )
        atomic_write_text(self.artifact_path(name), text if text.endswith("\n") else text + "\n")

    def load_brief_record(self, *, published: bool = True) -> dict[str, Any] | None:
        """Load brief metadata from manifest artifacts entry when present."""
        manifest = self.load_manifest()
        if not manifest:
            return None
        entry = (manifest.get("artifacts") or {}).get(KIND_PROJECT_BRIEF) or {}
        status = normalize_status(entry.get("status") or STATUS_MISSING, allow_missing=True)
        if status == STATUS_MISSING:
            return None
        text = self.load_brief_text(published=published)
        if published and not text and not entry.get("published_present"):
            text = self.load_brief_text(published=False)
        return normalize_summary_record(
            {
                "id": entry.get("id") or "project_brief",
                "kind": KIND_PROJECT_BRIEF,
                "status": STATUS_DRAFT if status == STATUS_MISSING else status,
                "summary": text,
                "lineage": entry.get("lineage") or empty_lineage(),
                "source_files": entry.get("source_files") or [],
                "evidence_item_ids": entry.get("evidence_item_ids") or [],
                "upstream_artifact_ids": entry.get("upstream_artifact_ids") or [],
                "source_checksum": entry.get("source_checksum") or "",
            },
            default_kind=KIND_PROJECT_BRIEF,
        )

    def rebuild_manifest(
        self,
        *,
        project_identity: Mapping[str, Any] | None = None,
        expected_source_fingerprint: str = "",
    ) -> dict[str, Any]:
        """Derive manifest aggregate status from on-disk artifacts."""
        chunks = self.load_summaries(KIND_CHUNK)
        scenes = self.load_summaries(KIND_SCENE)
        labels = self.load_summaries(KIND_LABEL)
        routes = self.load_routes()

        def _kind_entry(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            statuses = [
                evaluate_record_status(
                    r, expected_source_fingerprint=expected_source_fingerprint
                )
                for r in records
            ]
            lineage = empty_lineage(source_fingerprint=expected_source_fingerprint)
            if records:
                lineage = normalize_lineage(records[0].get("lineage"))
            return {
                "status": _aggregate_status(statuses),
                "count": len(records),
                "lineage": lineage,
            }

        draft_present = os.path.isfile(self.artifact_path(PROJECT_BRIEF_DRAFT_FILENAME))
        published_present = os.path.isfile(
            self.artifact_path(PROJECT_BRIEF_PUBLISHED_FILENAME)
        )
        previous = self.load_manifest()
        prev_brief = ((previous or {}).get("artifacts") or {}).get(KIND_PROJECT_BRIEF) or {}
        brief_status = normalize_status(
            prev_brief.get("status") or STATUS_MISSING, allow_missing=True
        )
        if published_present and brief_status == STATUS_MISSING:
            brief_status = STATUS_PUBLISHED
        elif draft_present and brief_status == STATUS_MISSING:
            brief_status = STATUS_DRAFT
        if brief_status == STATUS_PUBLISHED and expected_source_fingerprint:
            brief_lineage = normalize_lineage(prev_brief.get("lineage"))
            if (
                brief_lineage["source_fingerprint"]
                and brief_lineage["source_fingerprint"] != expected_source_fingerprint
            ):
                brief_status = STATUS_STALE

        identity = dict(project_identity or {})
        if not identity and previous:
            identity = dict(previous.get("project_identity") or {})

        manifest = empty_manifest(project_identity=identity, store_dir=self.store_dir)
        manifest["artifacts"][KIND_CHUNK] = _kind_entry(chunks)
        manifest["artifacts"][KIND_SCENE] = _kind_entry(scenes)
        manifest["artifacts"][KIND_LABEL] = _kind_entry(labels)
        manifest["artifacts"][KIND_ROUTE] = _kind_entry(routes)
        manifest["artifacts"][KIND_PROJECT_BRIEF] = {
            "status": brief_status,
            "draft_present": draft_present,
            "published_present": published_present,
            "lineage": normalize_lineage(prev_brief.get("lineage")),
            "id": _as_optional_str(prev_brief.get("id") or "project_brief"),
        }
        manifest["updated_at"] = utc_now_iso()
        return self.save_manifest(manifest)

    def collect_status(
        self,
        *,
        expected_source_fingerprint: str = "",
        project_identity: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Readonly status snapshot for CLI / GUI / doctor (no side effects)."""
        error = ""
        manifest: dict[str, Any] | None = None
        try:
            if not self.exists():
                overall = STATUS_MISSING
            else:
                manifest = self.load_manifest()
                if manifest is None:
                    # Artifacts without manifest still report via rebuild-in-memory.
                    chunks = self.load_summaries(KIND_CHUNK)
                    scenes = self.load_summaries(KIND_SCENE)
                    labels = self.load_summaries(KIND_LABEL)
                    routes = self.load_routes()
                    statuses = []
                    for kind, records in (
                        (KIND_CHUNK, chunks),
                        (KIND_SCENE, scenes),
                        (KIND_LABEL, labels),
                        (KIND_ROUTE, routes),
                    ):
                        statuses.extend(
                            evaluate_record_status(
                                r, expected_source_fingerprint=expected_source_fingerprint
                            )
                            for r in records
                        )
                    draft_present = os.path.isfile(
                        self.artifact_path(PROJECT_BRIEF_DRAFT_FILENAME)
                    )
                    published_present = os.path.isfile(
                        self.artifact_path(PROJECT_BRIEF_PUBLISHED_FILENAME)
                    )
                    if published_present:
                        statuses.append(STATUS_PUBLISHED)
                    elif draft_present:
                        statuses.append(STATUS_DRAFT)
                    overall = _aggregate_status(statuses)
                    manifest = empty_manifest(
                        project_identity=project_identity or {},
                        store_dir=self.store_dir,
                    )
                    manifest["artifacts"][KIND_CHUNK]["count"] = len(chunks)
                    manifest["artifacts"][KIND_CHUNK]["status"] = _aggregate_status(
                        [
                            evaluate_record_status(
                                r, expected_source_fingerprint=expected_source_fingerprint
                            )
                            for r in chunks
                        ]
                    )
                    manifest["artifacts"][KIND_SCENE]["count"] = len(scenes)
                    manifest["artifacts"][KIND_SCENE]["status"] = _aggregate_status(
                        [
                            evaluate_record_status(
                                r, expected_source_fingerprint=expected_source_fingerprint
                            )
                            for r in scenes
                        ]
                    )
                    manifest["artifacts"][KIND_LABEL]["count"] = len(labels)
                    manifest["artifacts"][KIND_LABEL]["status"] = _aggregate_status(
                        [
                            evaluate_record_status(
                                r, expected_source_fingerprint=expected_source_fingerprint
                            )
                            for r in labels
                        ]
                    )
                    manifest["artifacts"][KIND_ROUTE]["count"] = len(routes)
                    manifest["artifacts"][KIND_ROUTE]["status"] = _aggregate_status(
                        [
                            evaluate_record_status(
                                r, expected_source_fingerprint=expected_source_fingerprint
                            )
                            for r in routes
                        ]
                    )
                    brief_status = (
                        STATUS_PUBLISHED
                        if published_present
                        else STATUS_DRAFT
                        if draft_present
                        else STATUS_MISSING
                    )
                    manifest["artifacts"][KIND_PROJECT_BRIEF]["status"] = brief_status
                    manifest["artifacts"][KIND_PROJECT_BRIEF]["draft_present"] = draft_present
                    manifest["artifacts"][KIND_PROJECT_BRIEF][
                        "published_present"
                    ] = published_present
                else:
                    # Re-evaluate published → stale when expected fingerprint provided.
                    overall_statuses = []
                    for kind in (KIND_CHUNK, KIND_SCENE, KIND_LABEL, KIND_ROUTE):
                        entry = manifest["artifacts"][kind]
                        status = entry.get("status") or STATUS_MISSING
                        if status == STATUS_PUBLISHED and expected_source_fingerprint:
                            lineage = normalize_lineage(entry.get("lineage"))
                            if (
                                lineage["source_fingerprint"]
                                and lineage["source_fingerprint"] != expected_source_fingerprint
                            ):
                                status = STATUS_STALE
                                entry = dict(entry)
                                entry["status"] = STATUS_STALE
                                manifest["artifacts"][kind] = entry
                        overall_statuses.append(status)
                    brief = manifest["artifacts"][KIND_PROJECT_BRIEF]
                    brief_status = brief.get("status") or STATUS_MISSING
                    if brief_status == STATUS_PUBLISHED and expected_source_fingerprint:
                        lineage = normalize_lineage(brief.get("lineage"))
                        if (
                            lineage["source_fingerprint"]
                            and lineage["source_fingerprint"] != expected_source_fingerprint
                        ):
                            brief_status = STATUS_STALE
                            brief = dict(brief)
                            brief["status"] = STATUS_STALE
                            manifest["artifacts"][KIND_PROJECT_BRIEF] = brief
                    overall_statuses.append(brief_status)
                    overall = _aggregate_status(
                        [s for s in overall_statuses if s != STATUS_MISSING]
                    )
                    if overall == STATUS_MISSING and any(
                        s != STATUS_MISSING for s in overall_statuses
                    ):
                        overall = _aggregate_status(overall_statuses)
        except ProjectAnalysisError as exc:
            error = str(exc)
            overall = STATUS_FAILED
            manifest = empty_manifest(
                project_identity=project_identity or {}, store_dir=self.store_dir
            )
        except OSError as exc:
            error = str(exc)
            overall = STATUS_FAILED
            manifest = empty_manifest(
                project_identity=project_identity or {}, store_dir=self.store_dir
            )

        artifacts = (manifest or empty_manifest(store_dir=self.store_dir)).get("artifacts") or {}
        brief = artifacts.get(KIND_PROJECT_BRIEF) or {}
        return {
            "store_dir": self.store_dir,
            "store_exists": self.exists(),
            "schema_version": SCHEMA_VERSION,
            "overall_status": overall,
            "injectable": overall == STATUS_PUBLISHED,
            "project_identity": (manifest or {}).get("project_identity") or dict(project_identity or {}),
            "artifacts": artifacts,
            "chunk_count": int((artifacts.get(KIND_CHUNK) or {}).get("count") or 0),
            "scene_count": int((artifacts.get(KIND_SCENE) or {}).get("count") or 0),
            "label_count": int((artifacts.get(KIND_LABEL) or {}).get("count") or 0),
            "route_count": int((artifacts.get(KIND_ROUTE) or {}).get("count") or 0),
            "brief_status": brief.get("status") or STATUS_MISSING,
            "brief_draft_present": bool(brief.get("draft_present")),
            "brief_published_present": bool(brief.get("published_present")),
            "updated_at": (manifest or {}).get("updated_at") or "",
            "error": error,
        }


def collect_project_analysis_status(
    store_dir: str | os.PathLike[str] | None = None,
    *,
    base_dir: str | None = None,
    expected_source_fingerprint: str = "",
    project_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = (
        os.path.abspath(os.fspath(store_dir))
        if store_dir
        else get_default_project_analysis_store_dir(base_dir)
    )
    store = ProjectAnalysisStore(resolved)
    return store.collect_status(
        expected_source_fingerprint=expected_source_fingerprint,
        project_identity=project_identity,
    )


def format_status_lines(status: Mapping[str, Any]) -> list[str]:
    lines = [
        "Project analysis status:",
        f"- Store dir: {status.get('store_dir') or ''}",
        f"- Store exists: {bool(status.get('store_exists'))}",
        f"- Schema version: {status.get('schema_version')}",
        f"- Overall: {status.get('overall_status') or STATUS_MISSING}",
        f"- Injectable (published+fresh): {bool(status.get('injectable'))}",
        f"- Chunks: {status.get('chunk_count', 0)} "
        f"({((status.get('artifacts') or {}).get(KIND_CHUNK) or {}).get('status') or STATUS_MISSING})",
        f"- Scenes: {status.get('scene_count', 0)} "
        f"({((status.get('artifacts') or {}).get(KIND_SCENE) or {}).get('status') or STATUS_MISSING})",
        f"- Labels: {status.get('label_count', 0)} "
        f"({((status.get('artifacts') or {}).get(KIND_LABEL) or {}).get('status') or STATUS_MISSING})",
        f"- Routes: {status.get('route_count', 0)} "
        f"({((status.get('artifacts') or {}).get(KIND_ROUTE) or {}).get('status') or STATUS_MISSING})",
        f"- Project brief: {status.get('brief_status') or STATUS_MISSING} "
        f"(draft={bool(status.get('brief_draft_present'))}, "
        f"published={bool(status.get('brief_published_present'))})",
        f"- Updated at: {status.get('updated_at') or ''}",
    ]
    if status.get("error"):
        lines.append(f"- Error: {status.get('error')}")
    return lines


def format_status_label(status: Mapping[str, Any] | None) -> str:
    """Single-line Chinese status for GUI context library."""
    if not status:
        return "未检测"
    overall = str(status.get("overall_status") or STATUS_MISSING)
    labels = {
        STATUS_MISSING: "未生成",
        STATUS_DRAFT: "草稿",
        STATUS_REVIEW_REQUIRED: "待审核",
        STATUS_PUBLISHED: "已发布",
        STATUS_STALE: "已过期",
        STATUS_FAILED: "失败",
    }
    human = labels.get(overall, overall)
    parts = [human]
    if status.get("store_exists"):
        parts.append(
            f"chunk {status.get('chunk_count', 0)} / "
            f"label {status.get('label_count', 0)} / "
            f"route {status.get('route_count', 0)}"
        )
        brief = status.get("brief_status") or STATUS_MISSING
        if brief != STATUS_MISSING:
            parts.append(f"brief {labels.get(brief, brief)}")
    if status.get("error"):
        parts.append("读取错误")
    if overall == STATUS_PUBLISHED and not status.get("injectable"):
        parts.append("不可注入")
    return " · ".join(parts)


def print_status(status: Mapping[str, Any]) -> None:
    for line in format_status_lines(status):
        print(line)
