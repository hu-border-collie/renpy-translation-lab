"""Read-only export of the revision (old/new) polishing corpus.

P1 of #318: a stable, deterministic, auditable export contract consumed by
human reviewers and Agent batching. This module never modifies ``.rpy`` files,
manifests, glossary, or RAG state; write-back remains exclusively behind the
existing ``preview-revisions`` / ``apply-revisions`` gates.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from atomic_io import atomic_write_json, atomic_write_jsonl, atomic_write_text

REVISION_CORPUS_SCHEMA_VERSION = 1
CORPUS_JSONL_NAME = "revision_corpus.jsonl"
CORPUS_MARKDOWN_NAME = "revision_corpus.md"
CORPUS_MANIFEST_NAME = "revision_corpus_manifest.json"
SCANNER_ENGINE = "renpy-legacy-revision-scan-v1"
IDENTITY_SCHEMA = "identity_v2"


def stable_text_sha256(value: str) -> str:
    """Return the stable UTF-8 SHA-256 of ``value``."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def item_snapshot_digest(source: str, current_translation: str) -> str:
    """Digest of one old/new pair; the safety anchor for future proposals."""
    return stable_text_sha256(f"source\0{source}\0current\0{current_translation}")


def collect_file_digests(file_paths: Mapping[str, str]) -> dict[str, str]:
    """Map rel path -> content SHA-256 over a stable rel-path ordering.

    Unreadable or missing files raise so a partial corpus can never be
    presented as a complete snapshot.
    """
    digests: dict[str, str] = {}
    for rel_path in sorted(file_paths):
        with open(file_paths[rel_path], "rb") as handle:
            digests[rel_path] = hashlib.sha256(handle.read()).hexdigest()
    return digests


def aggregate_digest(digests: Mapping[str, str]) -> str:
    """Stable aggregate digest over a sorted rel-path digest map."""
    payload = json.dumps(
        {rel_path: digests[rel_path] for rel_path in sorted(digests)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return stable_text_sha256(payload)


def _int_or_zero(value: Any) -> tuple[int, str | None]:
    """Coerce a locator value to int; report a diagnostic on failure."""
    try:
        return int(value or 0), None
    except (TypeError, ValueError):
        return 0, repr(value)


def _attach_adjacent_context(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach same-file previous/next old/new pairs as human-review context."""
    by_file: dict[str, list[dict[str, Any]]] = {}
    file_order: list[str] = []
    for row in items:
        rel_path = str(row.get("file_rel_path") or "")
        if rel_path not in by_file:
            by_file[rel_path] = []
            file_order.append(rel_path)
        by_file[rel_path].append(row)
    for rel_path in file_order:
        rows = by_file[rel_path]
        for index, row in enumerate(rows):
            previous = rows[index - 1] if index > 0 else None
            following = rows[index + 1] if index + 1 < len(rows) else None
            row["context"] = {
                "previous": (
                    {
                        "source": previous["source"],
                        "current_translation": previous["current_translation"],
                    }
                    if previous is not None
                    else None
                ),
                "next": (
                    {
                        "source": following["source"],
                        "current_translation": following["current_translation"],
                    }
                    if following is not None
                    else None
                ),
            }
    return items


def build_corpus_items(
    file_jobs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build deterministic corpus rows (and diagnostics) from revision scan jobs.

    Jobs are explicitly sorted by rel path so item order is stable regardless
    of caller ordering; items keep their in-file order and carry a per-file
    1-based ordinal. Duplicate source text keeps distinct occurrences through
    ``identity_v2`` instead of being deduplicated by text. Non-numeric locator
    values degrade to 0 and produce a ``LOCATOR_NON_NUMERIC`` diagnostic instead
    of aborting the export.
    """
    items: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    ordered_jobs = sorted(
        file_jobs,
        key=lambda job: str(job.get("file_rel_path") or ""),
    )
    for job in ordered_jobs:
        file_rel_path = str(job.get("file_rel_path") or "")
        for ordinal, item in enumerate(job.get("items") or [], start=1):
            identity_v2 = str(item.get("identity_v2") or item.get("id") or "")
            source = str(item.get("source") or "")
            current_translation = str(item.get("current_translation") or "")
            line, line_diag = _int_or_zero(item.get("line"))
            start, start_diag = _int_or_zero(item.get("start"))
            end, end_diag = _int_or_zero(item.get("end"))
            line_number, line_number_diag = _int_or_zero(item.get("line_number"))
            for field, diagnostic in (
                ("line", line_diag),
                ("start", start_diag),
                ("end", end_diag),
                ("line_number", line_number_diag),
            ):
                if diagnostic is not None:
                    diagnostics.append(
                        {
                            "code": "LOCATOR_NON_NUMERIC",
                            "identity_v2": identity_v2,
                            "field": field,
                            "value": diagnostic,
                        }
                    )
            row = {
                "schema_version": REVISION_CORPUS_SCHEMA_VERSION,
                "occurrence_id": identity_v2,
                "identity_v2": identity_v2,
                "file_rel_path": file_rel_path,
                "locator": {
                    "line": line,
                    "line_number": line_number,
                    "start": start,
                    "end": end,
                    "ordinal": ordinal,
                },
                "display_line": line_number,
                "speaker_id": str(item.get("speaker_id") or ""),
                "source": source,
                "current_translation": current_translation,
                "snapshot_digest": item_snapshot_digest(
                    source,
                    current_translation,
                ),
            }
            items.append(row)
    return _attach_adjacent_context(items), diagnostics


def render_corpus_markdown(
    items: Sequence[Mapping[str, Any]],
    *,
    project_slug: str,
) -> str:
    """Render a linear human-review report; one section per file."""
    lines = [
        f"# 润色语料：{project_slug}",
        "",
        f"- 条目数：{len(items)}",
        f"- schema：revision-corpus v{REVISION_CORPUS_SCHEMA_VERSION}",
        "",
    ]
    current_file: str | None = None
    for row in items:
        file_rel_path = str(row.get("file_rel_path") or "")
        if file_rel_path != current_file:
            if current_file is not None:
                lines.append("")
            current_file = file_rel_path
            lines.append(f"## {file_rel_path}")
            lines.append("")
        locator = row.get("locator") or {}
        try:
            line_number = int(locator.get("line_number") or 0)
        except (TypeError, ValueError):
            line_number = 0
        speaker = str(row.get("speaker_id") or "").strip()
        speaker_label = f" [{speaker}]" if speaker else ""
        lines.append(f"- L{line_number}{speaker_label}：{row.get('source') or ''}")
        lines.append(f"  → {row.get('current_translation') or ''}")
    return "\n".join(lines) + "\n"


def export_revision_corpus(
    output_dir: str,
    file_jobs: Sequence[Mapping[str, Any]],
    *,
    project_slug: str,
    game_root: str,
    tl_dir: str,
    tl_subdir: str,
    include_files: Sequence[str] = (),
    include_prefixes: Sequence[str] = (),
    source_digests_before: Mapping[str, str] | None = None,
    source_digests_after: Mapping[str, str] | None = None,
    source_digests_scanned: Mapping[str, str] | None = None,
    file_line_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Write JSONL + Markdown + manifest into ``output_dir``; return the manifest.

    The manifest carries project identity, scanner/schema identity, the source
    snapshot digest (per-file and aggregate), scope counts, and whether source
    files changed while the scan was running. ``source_digests_scanned`` must be
    the digests of the exact bytes the scanner consumed (recorded at read time);
    a mismatch against either boundary digest means a file changed mid-scan and
    was restored, so the corpus is flagged instead of silently mixed.
    ``_output_dir`` / ``_manifest_path`` are CLI-facing conveniences and are not
    persisted inside the manifest file.
    """
    items, diagnostics = build_corpus_items(file_jobs)
    item_count_by_file: dict[str, int] = {}
    for row in items:
        rel_path = str(row.get("file_rel_path") or "")
        item_count_by_file[rel_path] = item_count_by_file.get(rel_path, 0) + 1
    line_counts = {
        str(rel_path): int(count or 0)
        for rel_path, count in (file_line_counts or {}).items()
    }
    file_summaries = {
        rel_path: {
            "line_count": line_counts.get(rel_path, 0),
            "item_count": item_count_by_file.get(rel_path, 0),
        }
        for rel_path in sorted(set(line_counts) | set(item_count_by_file))
    }
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.abspath(os.path.join(output_dir, CORPUS_JSONL_NAME))
    markdown_path = os.path.abspath(os.path.join(output_dir, CORPUS_MARKDOWN_NAME))
    manifest_path = os.path.abspath(os.path.join(output_dir, CORPUS_MANIFEST_NAME))

    atomic_write_jsonl(jsonl_path, items, ensure_ascii=False)
    atomic_write_text(
        markdown_path,
        render_corpus_markdown(items, project_slug=project_slug),
    )

    source_digests = dict(source_digests_before or {})
    scanned_files = {str(row.get("file_rel_path") or "") for row in items}
    scanned_files_missing_digest = sorted(scanned_files - set(source_digests))
    scanned_digests = dict(source_digests_scanned or {})
    scanned_files_digest_mismatch = sorted(
        rel_path
        for rel_path, digest in scanned_digests.items()
        if digest != source_digests.get(rel_path)
        or (
            source_digests_after is not None
            and digest != source_digests_after.get(rel_path)
        )
    )
    source_changed = (
        source_digests_after is not None
        and source_digests_after != source_digests
    ) or bool(scanned_files_missing_digest)
    if scanned_files_digest_mismatch:
        source_changed = True
    manifest = {
        "schema_version": REVISION_CORPUS_SCHEMA_VERSION,
        "kind": "revision_corpus",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": {
            "slug": project_slug,
            "game_root": os.path.abspath(game_root) if game_root else "",
            "tl_dir": os.path.abspath(tl_dir) if tl_dir else "",
            "tl_subdir": tl_subdir,
            "include_files": sorted(include_files),
            "include_prefixes": sorted(include_prefixes),
        },
        "scanner": {
            "engine": SCANNER_ENGINE,
            "identity_schema": IDENTITY_SCHEMA,
            "description": (
                "Reuses the legacy revision scanner: translate-block old/new "
                "pairs and comment translations recognized by the revision "
                "entry collector. Other source content is outside the corpus scope."
            ),
        },
        "source": {
            "snapshot_digest": aggregate_digest(source_digests),
            "file_digests": dict(sorted(source_digests.items())),
            "source_changed_during_scan": bool(source_changed),
            "scanned_files_missing_digest": scanned_files_missing_digest,
            "scanned_files_digest_mismatch": scanned_files_digest_mismatch,
        },
        "scope": {
            "file_count": len({str(row.get("file_rel_path") or "") for row in items}),
            "item_count": len(items),
            "note": (
                "Only revision-scanner-recognized old/new and comment-translation "
                "entries are included; other source content is outside the corpus scope."
            ),
        },
        "coverage": {
            "mode": "revision_recognized_only",
            "scanned_file_count": len(file_summaries),
            "recognized_item_count": len(items),
            "note": (
                "Every in-scope TL file was scanned by the revision entry "
                "collector; per-file line/item counts are reported in "
                "file_summaries so unrecognized content is visible instead of "
                "silently dropped."
            ),
        },
        "file_summaries": file_summaries,
        "diagnostics": diagnostics,
        "paths": {
            "output_dir": os.path.abspath(output_dir),
            "jsonl": jsonl_path,
            "markdown": markdown_path,
            "manifest": manifest_path,
        },
    }
    atomic_write_json(manifest_path, manifest, ensure_ascii=False, indent=2)
    manifest["_output_dir"] = os.path.abspath(output_dir)
    manifest["_manifest_path"] = manifest_path
    return manifest
