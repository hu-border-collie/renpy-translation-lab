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


def build_corpus_items(file_jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build deterministic corpus rows from revision scan jobs.

    ``file_jobs`` must already be in a stable order (the CLI sorts files by rel
    path); items keep their in-file order and carry a per-file 1-based ordinal.
    Duplicate source text keeps distinct occurrences through ``identity_v2``
    instead of being deduplicated by text.
    """
    items: list[dict[str, Any]] = []
    for job in file_jobs:
        file_rel_path = str(job.get("file_rel_path") or "")
        for ordinal, item in enumerate(job.get("items") or [], start=1):
            identity_v2 = str(item.get("identity_v2") or item.get("id") or "")
            source = str(item.get("source") or "")
            current_translation = str(item.get("current_translation") or "")
            try:
                line_number = int(item.get("line_number") or 0)
            except (TypeError, ValueError):
                line_number = 0
            row = {
                "schema_version": REVISION_CORPUS_SCHEMA_VERSION,
                "occurrence_id": identity_v2,
                "identity_v2": identity_v2,
                "file_rel_path": file_rel_path,
                "locator": {
                    "line": int(item.get("line") or 0),
                    "line_number": line_number,
                    "start": int(item.get("start") or 0),
                    "end": int(item.get("end") or 0),
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
    return items


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
) -> dict[str, Any]:
    """Write JSONL + Markdown + manifest into ``output_dir``; return the manifest.

    The manifest carries project identity, scanner/schema identity, the source
    snapshot digest (per-file and aggregate), scope counts, and whether source
    files changed while the scan was running. ``_output_dir`` / ``_manifest_path``
    are CLI-facing conveniences and are not persisted inside the manifest file.
    """
    items = build_corpus_items(file_jobs)
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
    source_changed = source_digests_after is not None and (
        source_digests_after != source_digests
    )
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
        },
        "source": {
            "snapshot_digest": aggregate_digest(source_digests),
            "file_digests": dict(sorted(source_digests.items())),
            "source_changed_during_scan": bool(source_changed),
        },
        "scope": {
            "file_count": len({str(row.get("file_rel_path") or "") for row in items}),
            "item_count": len(items),
        },
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
