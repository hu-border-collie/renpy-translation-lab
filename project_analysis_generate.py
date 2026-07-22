"""Project Analysis draft generation without LLM (#254 PR A).

Ingest keyword chunk summaries, attach them to a static route graph, and write
draft label/route/brief artifacts into the project_analysis store.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

from project_analysis import (
    KIND_CHUNK,
    KIND_LABEL,
    KIND_PROJECT_BRIEF,
    KIND_ROUTE,
    STATUS_DRAFT,
    ProjectAnalysisError,
    ProjectAnalysisStore,
    digest_source_items,
    digest_upstream_artifacts,
    empty_lineage,
    normalize_summary_record,
    resolve_project_analysis_store,
    sha256_text,
    utc_now_iso,
)
from project_analysis_routes import (
    build_route_graph,
    digest_script_paths,
    discover_script_files,
    graph_to_label_records,
    graph_to_route_records,
)


def load_keyword_chunk_summaries(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load keyword_chunk_summaries.jsonl rows."""
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise ProjectAnalysisError(f"keyword summary JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ProjectAnalysisError(
                    f"invalid keyword summary JSONL at {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise ProjectAnalysisError(
                    f"keyword summary row at {path}:{line_no} must be an object"
                )
            rows.append(data)
    return rows


def keyword_rows_to_chunk_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_fingerprint: str = "",
) -> list[dict[str, Any]]:
    """Map keyword export rows to analysis chunk draft records."""
    records: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("key") or "").strip()
        file_rel = str(row.get("file_rel_path") or "").replace("\\", "/").strip()
        chunk_index = row.get("chunk_index")
        try:
            chunk_index_int = int(chunk_index) if chunk_index is not None else 0
        except (TypeError, ValueError):
            chunk_index_int = 0
        artifact_id = key or f"chunk:{file_rel}:{chunk_index_int}"
        summary = str(row.get("chunk_summary") or row.get("summary") or "").strip()
        if not summary:
            continue
        evidence = [
            str(x).strip()
            for x in (row.get("summary_evidence_item_ids") or row.get("evidence_item_ids") or [])
            if str(x or "").strip()
        ]
        line_numbers = row.get("line_numbers") or row.get("source_lines") or []
        line_span = None
        if isinstance(line_numbers, list) and line_numbers:
            try:
                nums = [int(x) for x in line_numbers]
                line_span = [min(nums), max(nums)]
            except (TypeError, ValueError):
                line_span = None
        # Best-effort label hint from translate block style ids later; keep metadata.
        records.append(
            {
                "id": artifact_id,
                "kind": KIND_CHUNK,
                "status": STATUS_DRAFT,
                "source_files": [file_rel] if file_rel else [],
                "evidence_item_ids": evidence,
                "line_span": line_span,
                "source_checksum": sha256_text(summary),
                "upstream_artifact_ids": [],
                "lineage": empty_lineage(
                    source_fingerprint=source_fingerprint
                    or digest_source_items(
                        [{"id": e, "source_checksum": ""} for e in evidence]
                    ),
                    generated_at=utc_now_iso(),
                ),
                "summary": summary,
                "metadata": {
                    "keyword_key": key,
                    "chunk_index": chunk_index_int,
                    "file_rel_path": file_rel,
                },
            }
        )
    return [normalize_summary_record(r, default_kind=KIND_CHUNK) for r in records]


def _normalize_file_rel(path: str) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def _guess_label_for_chunk(chunk: Mapping[str, Any], label_names: Sequence[str]) -> str:
    """Fallback heuristic when line spans are unavailable."""
    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    file_rel = str(meta.get("file_rel_path") or (chunk.get("source_files") or [""])[0])
    summary = str(chunk.get("summary") or "")
    for name in sorted(label_names, key=len, reverse=True):
        if name and (
            name in file_rel
            or f"`{name}`" in summary
            or f" {name} " in f" {summary} "
        ):
            return name
    return ""


def _label_match_for_chunk(
    chunk: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> str:
    """Prefer file_rel_path + line_span overlap with label bodies; else name guess."""
    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    file_rel = _normalize_file_rel(
        str(meta.get("file_rel_path") or (chunk.get("source_files") or [""])[0])
    )
    span = chunk.get("line_span")
    if file_rel and isinstance(span, (list, tuple)) and len(span) == 2:
        try:
            c_start, c_end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            c_start = c_end = -1
        if c_start >= 0:
            candidates: list[tuple[int, str]] = []
            for name, node in labels.items():
                node_file = _normalize_file_rel(getattr(node, "file_rel_path", "") or "")
                if node_file != file_rel:
                    continue
                l_start = int(getattr(node, "line_start", 0) or 0)
                l_end = int(getattr(node, "line_end", 0) or l_start)
                # Overlap between chunk lines and label body span.
                if c_start <= l_end and c_end >= l_start:
                    # Prefer the tightest (smallest) containing/overlapping label.
                    candidates.append((max(0, l_end - l_start), name))
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]))
                return candidates[0][1]
    return _guess_label_for_chunk(chunk, list(labels.keys()))


def assign_chunks_to_labels(
    chunks: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Any] | Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Map chunks onto labels using line spans when possible.

    *labels* may be a ``{name: LabelNode}`` map (preferred) or a sequence of names.
    """
    if isinstance(labels, Mapping):
        label_map = labels
        label_names = list(label_map.keys())
    else:
        label_names = list(labels)
        label_map = {name: type("N", (), {"file_rel_path": "", "line_start": 0, "line_end": 0})() for name in label_names}

    by_label: dict[str, list[dict[str, Any]]] = {name: [] for name in label_names}
    unassigned: list[dict[str, Any]] = []
    for chunk in chunks:
        name = _label_match_for_chunk(chunk, label_map)
        if name and name in by_label:
            by_label[name].append(dict(chunk))
        else:
            unassigned.append(dict(chunk))
    if unassigned and len(label_names) == 1:
        by_label[label_names[0]].extend(unassigned)
        unassigned = []
    if unassigned:
        by_label["_unassigned"] = unassigned
    return by_label


def build_project_brief_text(
    *,
    routes: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    unresolved_count: int = 0,
) -> str:
    lines = [
        "# Project Analysis Brief (draft)",
        "",
        "Generated without LLM from keyword chunk summaries and static Ren'Py structure.",
        "Review before publish. Only the published copy may be injected into prompts.",
        "",
        f"- Labels: {len(labels)}",
        f"- Routes: {len(routes)}",
        f"- Unresolved dynamic edges: {unresolved_count}",
        "",
    ]
    if routes:
        lines.append("## Routes")
        lines.append("")
        for route in routes:
            rid = route.get("id") or route.get("route_id") or ""
            meta = route.get("metadata") if isinstance(route.get("metadata"), dict) else {}
            entry = meta.get("entry_label") or ""
            path = meta.get("label_ids") or []
            flag = " (unresolved)" if meta.get("unresolved") else ""
            lines.append(f"### {rid}{flag}")
            lines.append(f"- entry: `{entry}`")
            if path:
                lines.append(f"- path: {' -> '.join(f'`{x}`' for x in path)}")
            summary = str(route.get("summary") or "").strip()
            if summary:
                # Keep brief shorter: first line of route summary.
                first = summary.splitlines()[0]
                lines.append(f"- summary: {first}")
            lines.append("")
    else:
        lines.append("## Routes")
        lines.append("")
        lines.append("(no routes discovered)")
        lines.append("")
    lines.append("## Labels")
    lines.append("")
    for label in labels[:50]:
        lid = label.get("label_id") or label.get("id")
        lines.append(f"- `{lid}`")
    if len(labels) > 50:
        lines.append(f"- … and {len(labels) - 50} more")
    lines.append("")
    return "\n".join(lines)


def ingest_keyword_summaries(
    summary_jsonl: str | os.PathLike[str],
    *,
    store_dir: str | os.PathLike[str] | None = None,
    base_dir: str | None = None,
) -> dict[str, Any]:
    """Import keyword summaries into analysis chunk_summaries.jsonl as drafts."""
    rows = load_keyword_chunk_summaries(summary_jsonl)
    source_fp = digest_source_items(
        [
            {
                "id": str(r.get("key") or ""),
                "source_checksum": sha256_text(str(r.get("chunk_summary") or "")),
                "file_rel_path": str(r.get("file_rel_path") or ""),
            }
            for r in rows
        ]
    )
    chunks = keyword_rows_to_chunk_records(rows, source_fingerprint=source_fp)
    store = resolve_project_analysis_store(store_dir, base_dir=base_dir)
    store.save_summaries(KIND_CHUNK, chunks)
    store.rebuild_manifest(
        project_identity={"base_dir": base_dir or ""},
        expected_source_fingerprint=source_fp,
    )
    return {
        "store_dir": store.store_dir,
        "chunks_written": len(chunks),
        "source_fingerprint": source_fp,
        "input_rows": len(rows),
    }


def build_structure_drafts(
    *,
    store_dir: str | os.PathLike[str] | None = None,
    base_dir: str | None = None,
    script_roots: Sequence[str] | None = None,
    entry_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Parse scripts, merge existing chunk drafts, write label/route/brief drafts."""
    store = resolve_project_analysis_store(store_dir, base_dir=base_dir)
    roots: list[str] = []
    if script_roots:
        roots.extend(script_roots)
    elif base_dir:
        # Prefer game scripts under work/game or original/game.
        for rel in ("game", os.path.join("work", "game"), os.path.join("original", "game")):
            candidate = os.path.join(base_dir, rel)
            if os.path.isdir(candidate):
                roots.append(candidate)
        if not roots:
            roots.append(base_dir)
    else:
        raise ProjectAnalysisError(
            "build_structure_drafts requires base_dir or script_roots"
        )

    script_paths = discover_script_files(roots)
    if not script_paths:
        raise ProjectAnalysisError(
            f"no .rpy scripts found under: {', '.join(roots)}"
        )

    graph_base = base_dir or (roots[0] if roots else None)
    graph = build_route_graph(
        script_paths,
        base_dir=graph_base,
        entry_labels=entry_labels,
    )
    chunks = store.load_summaries(KIND_CHUNK)
    # Fingerprint must include script *contents*, not just names / chunk hashes.
    source_fp = digest_script_paths(script_paths, base_dir=graph_base)

    chunk_by_label = assign_chunks_to_labels(chunks, graph.labels)
    # Drop synthetic bucket from label record generation.
    label_chunk_map = {
        k: v for k, v in chunk_by_label.items() if k != "_unassigned" and k in graph.labels
    }
    labels = graph_to_label_records(
        graph, chunk_by_label=label_chunk_map, source_fingerprint=source_fp
    )
    routes = graph_to_route_records(
        graph, label_records=labels, source_fingerprint=source_fp
    )
    store.save_summaries(KIND_LABEL, labels)
    store.save_routes(routes)

    brief_text = build_project_brief_text(
        routes=routes,
        labels=labels,
        unresolved_count=len(graph.unresolved_edges),
    )
    store.save_brief_text(brief_text, published=False)
    # Persist absolute script roots so runtime fingerprint reuse matches build-time roots
    # (custom --script-root must not fall back to default game/ layout only).
    abs_roots = [os.path.abspath(r) for r in roots]
    identity = {
        "base_dir": os.path.abspath(base_dir) if base_dir else "",
        "script_roots": abs_roots,
        "graph_base": os.path.abspath(graph_base) if graph_base else "",
    }
    # Update brief lineage in manifest.
    manifest = store.rebuild_manifest(
        project_identity=identity,
        expected_source_fingerprint=source_fp,
    )
    brief_entry = dict((manifest.get("artifacts") or {}).get(KIND_PROJECT_BRIEF) or {})
    upstream = [r["id"] for r in routes] + [r["id"] for r in labels]
    brief_entry.update(
        {
            "status": STATUS_DRAFT,
            "draft_present": True,
            "published_present": os.path.isfile(
                store.artifact_path("project_brief.published.md")
            ),
            "id": "project_brief",
            "lineage": empty_lineage(
                source_fingerprint=source_fp,
                upstream_dependency_digest=digest_upstream_artifacts(upstream),
                generated_at=utc_now_iso(),
            ),
            "metadata": {
                "script_roots": abs_roots,
                "graph_base": identity["graph_base"],
            },
        }
    )
    # rebuild_manifest may overwrite project_identity; restore full identity.
    manifest["project_identity"] = identity
    manifest.setdefault("artifacts", {})[KIND_PROJECT_BRIEF] = brief_entry
    store.save_manifest(manifest)

    return {
        "store_dir": store.store_dir,
        "scripts_scanned": len(script_paths),
        "labels": len(labels),
        "routes": len(routes),
        "unresolved_edges": len(graph.unresolved_edges),
        "chunks": len(chunks),
        "source_fingerprint": source_fp,
        "brief_draft_chars": len(brief_text),
        "script_roots": abs_roots,
        "graph": graph.to_dict(),
    }
