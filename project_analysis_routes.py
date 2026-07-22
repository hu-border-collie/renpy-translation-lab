"""Static Ren'Py route / label graph for Project Analysis Phase 2 (#254).

Parses label / jump / call / menu structure without executing Ren'Py.
Dynamic jumps are marked unresolved; no single linear route is invented.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from project_analysis import (
    KIND_LABEL,
    KIND_ROUTE,
    STATUS_DRAFT,
    digest_upstream_artifacts,
    empty_lineage,
    sha256_text,
    stable_json_sha256,
)

# Story labels (not translate blocks).
_LABEL_RE = re.compile(r"^\s*label\s+([A-Za-z_][\w.]*)\s*:", re.MULTILINE)
_JUMP_RE = re.compile(r"^\s*jump\s+([A-Za-z_][\w.]*)\s*$", re.MULTILINE)
_JUMP_EXPR_RE = re.compile(r"^\s*jump\s+expression\b", re.MULTILINE | re.IGNORECASE)
_CALL_RE = re.compile(r"^\s*call\s+([A-Za-z_][\w.]*)\b", re.MULTILINE)
_CALL_EXPR_RE = re.compile(r"^\s*call\s+expression\b", re.MULTILINE | re.IGNORECASE)
_MENU_RE = re.compile(r"^\s*menu\s*(?:[A-Za-z_][\w.]*)?\s*:", re.MULTILINE)


@dataclass
class RouteEdge:
    source_label: str
    target_label: str = ""
    kind: str = "jump"  # jump | call | menu | unresolved
    unresolved: bool = False
    file_rel_path: str = ""
    line: int = 0
    raw: str = ""


@dataclass
class LabelNode:
    name: str
    file_rel_path: str
    line_start: int
    line_end: int = 0
    outgoing: list[RouteEdge] = field(default_factory=list)


@dataclass
class RouteGraph:
    labels: dict[str, LabelNode] = field(default_factory=dict)
    edges: list[RouteEdge] = field(default_factory=list)
    routes: list[dict[str, Any]] = field(default_factory=list)
    unresolved_edges: list[RouteEdge] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": {
                name: {
                    "name": node.name,
                    "file_rel_path": node.file_rel_path,
                    "line_start": node.line_start,
                    "line_end": node.line_end,
                    "outgoing": [edge.__dict__ for edge in node.outgoing],
                }
                for name, node in sorted(self.labels.items())
            },
            "edges": [edge.__dict__ for edge in self.edges],
            "routes": list(self.routes),
            "unresolved_edges": [edge.__dict__ for edge in self.unresolved_edges],
            "source_files": list(self.source_files),
        }


def _line_number_at(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _normalize_rel(path: str) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def safe_relpath(path: str, base_dir: str | None = None) -> str:
    """Relative path under *base_dir*, or basename when cross-drive / outside base.

    ``os.path.relpath`` raises ValueError on Windows when path and base are on
    different drives; that must not break route builds or CI temp dirs.
    """
    path_abs = os.path.abspath(path)
    if not base_dir:
        return _normalize_rel(os.path.basename(path_abs))
    base_abs = os.path.abspath(base_dir)
    try:
        common = os.path.commonpath([path_abs, base_abs])
    except ValueError:
        return _normalize_rel(os.path.basename(path_abs))
    if os.path.normcase(common) != os.path.normcase(base_abs):
        return _normalize_rel(os.path.basename(path_abs))
    try:
        return _normalize_rel(os.path.relpath(path_abs, base_abs))
    except ValueError:
        return _normalize_rel(os.path.basename(path_abs))


def digest_script_paths(
    script_paths: Sequence[str],
    *,
    base_dir: str | None = None,
) -> str:
    """Stable fingerprint of script relative paths + file content checksums."""
    rows: list[dict[str, str]] = []
    for path in sorted({os.path.abspath(p) for p in script_paths if p}):
        if not os.path.isfile(path):
            continue
        rel = safe_relpath(path, base_dir)
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        rows.append({"path": rel, "sha256": digest})
    rows.sort(key=lambda row: row["path"])
    return stable_json_sha256(rows)


def parse_rpy_labels_and_edges(text: str, *, file_rel_path: str = "") -> tuple[list[LabelNode], list[RouteEdge]]:
    """Parse one .rpy file into labels and control-flow edges."""
    file_rel_path = _normalize_rel(file_rel_path)
    labels: list[LabelNode] = []
    edges: list[RouteEdge] = []

    label_matches = list(_LABEL_RE.finditer(text))
    for i, match in enumerate(label_matches):
        name = match.group(1)
        start = _line_number_at(text, match.start())
        end_index = label_matches[i + 1].start() if i + 1 < len(label_matches) else len(text)
        end = _line_number_at(text, max(end_index - 1, match.start()))
        body = text[match.end() : end_index]
        node = LabelNode(
            name=name,
            file_rel_path=file_rel_path,
            line_start=start,
            line_end=end,
        )
        # Jumps / calls inside this label body.
        for jm in _JUMP_RE.finditer(body):
            edge = RouteEdge(
                source_label=name,
                target_label=jm.group(1),
                kind="jump",
                file_rel_path=file_rel_path,
                line=start + _line_number_at(body, jm.start()) - 1,
                raw=jm.group(0).strip(),
            )
            node.outgoing.append(edge)
            edges.append(edge)
        for jm in _JUMP_EXPR_RE.finditer(body):
            edge = RouteEdge(
                source_label=name,
                target_label="",
                kind="unresolved",
                unresolved=True,
                file_rel_path=file_rel_path,
                line=start + _line_number_at(body, jm.start()) - 1,
                raw=jm.group(0).strip(),
            )
            node.outgoing.append(edge)
            edges.append(edge)
        for cm in _CALL_RE.finditer(body):
            edge = RouteEdge(
                source_label=name,
                target_label=cm.group(1),
                kind="call",
                file_rel_path=file_rel_path,
                line=start + _line_number_at(body, cm.start()) - 1,
                raw=cm.group(0).strip(),
            )
            node.outgoing.append(edge)
            edges.append(edge)
        for cm in _CALL_EXPR_RE.finditer(body):
            edge = RouteEdge(
                source_label=name,
                target_label="",
                kind="unresolved",
                unresolved=True,
                file_rel_path=file_rel_path,
                line=start + _line_number_at(body, cm.start()) - 1,
                raw=cm.group(0).strip(),
            )
            node.outgoing.append(edge)
            edges.append(edge)
        # Menu blocks: treat jump/call still found above; also flag bare menu presence.
        if _MENU_RE.search(body):
            # Menu choices already contribute via jump/call lines inside body.
            pass
        labels.append(node)
    return labels, edges


def discover_script_files(roots: Sequence[str]) -> list[str]:
    """Find .rpy script files under roots (skips tl/ and common dump dirs)."""
    found: list[str] = []
    skip_parts = {"tl", "__pycache__", ".git", "cache", "saves"}
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            if os.path.isfile(root) and root.lower().endswith(".rpy"):
                found.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in skip_parts]
            # Skip game/tl trees.
            parts = {p.lower() for p in dirpath.replace("\\", "/").split("/")}
            if "tl" in parts:
                continue
            for name in filenames:
                if name.lower().endswith(".rpy"):
                    found.append(os.path.join(dirpath, name))
    found.sort()
    return found


def build_route_graph(
    script_paths: Sequence[str],
    *,
    base_dir: str | None = None,
    entry_labels: Sequence[str] | None = None,
) -> RouteGraph:
    """Build a multi-file route graph and enumerate simple routes from entries."""
    graph = RouteGraph()
    base = os.path.abspath(base_dir) if base_dir else ""
    all_labels: dict[str, LabelNode] = {}
    all_edges: list[RouteEdge] = []

    for path in script_paths:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            continue
        rel = safe_relpath(path, base or None)
        with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
            text = handle.read()
        labels, edges = parse_rpy_labels_and_edges(text, file_rel_path=rel)
        graph.source_files.append(rel)
        for node in labels:
            # Last definition wins; still records file.
            all_labels[node.name] = node
        all_edges.extend(edges)

    graph.labels = all_labels
    graph.edges = all_edges
    graph.unresolved_edges = [e for e in all_edges if e.unresolved]

    # Default entries: labels with no inbound edges (excluding self-unresolved only).
    inbound: dict[str, int] = {name: 0 for name in all_labels}
    for edge in all_edges:
        if edge.target_label and edge.target_label in inbound:
            inbound[edge.target_label] += 1
    if entry_labels:
        entries = [n for n in entry_labels if n in all_labels]
    else:
        entries = sorted(name for name, count in inbound.items() if count == 0)
        if not entries:
            entries = sorted(all_labels.keys())[:1]

    routes: list[dict[str, Any]] = []
    for entry in entries:
        paths = _enumerate_routes(entry, all_labels, max_depth=32, max_routes=32)
        for path_labels in paths:
            route_id = _route_id(entry, path_labels)
            unresolved = any(
                e.unresolved and e.source_label in path_labels for e in all_edges
            )
            shared = [lab for lab in path_labels if _label_shared(lab, all_edges, path_labels)]
            routes.append(
                {
                    "id": route_id,
                    "entry_label": entry,
                    "label_ids": path_labels,
                    "unresolved": unresolved,
                    "shared_labels": shared,
                    "source_files": sorted(
                        {
                            all_labels[lab].file_rel_path
                            for lab in path_labels
                            if lab in all_labels
                        }
                    ),
                }
            )
    # De-dupe route ids.
    seen: set[str] = set()
    unique_routes = []
    for route in routes:
        if route["id"] in seen:
            continue
        seen.add(route["id"])
        unique_routes.append(route)
    graph.routes = unique_routes
    return graph


def _label_shared(label: str, edges: Sequence[RouteEdge], path_labels: Sequence[str]) -> bool:
    # Shared if inbound from labels outside this path as well — approximate:
    # label appears as target from more than one distinct source.
    sources = {e.source_label for e in edges if e.target_label == label}
    return len(sources) > 1


def _enumerate_routes(
    entry: str,
    labels: dict[str, LabelNode],
    *,
    max_depth: int,
    max_routes: int,
) -> list[list[str]]:
    """Enumerate jump/menu branches only.

    ``call`` is *not* treated as an exclusive branch. A call returns to the
    caller; modeling it like jump would invent false mutually-exclusive routes
    (e.g. ``start → helper`` vs ``start → ending`` when the real flow is
    ``start → call helper → ending``). Call targets are recorded on edges for
    dependency/metadata but do not fork the walk.
    """
    results: list[list[str]] = []

    def walk(current: str, path: list[str], depth: int) -> None:
        if len(results) >= max_routes:
            return
        if depth > max_depth:
            results.append(list(path))
            return
        node = labels.get(current)
        if node is None:
            results.append(list(path))
            return
        jump_targets: list[str] = []
        for edge in node.outgoing:
            if edge.unresolved:
                continue
            # Calls are side dependencies, not route forks.
            if edge.kind == "call":
                continue
            if edge.target_label and edge.target_label not in path:
                jump_targets.append(edge.target_label)
            elif edge.target_label:
                # cycle — close route
                results.append(list(path))
        if not jump_targets:
            results.append(list(path))
            return
        for target in jump_targets:
            walk(target, path + [target], depth + 1)

    if entry not in labels:
        return []
    walk(entry, [entry], 0)
    if not results:
        results.append([entry])
    return results


def _route_id(entry: str, path_labels: Sequence[str]) -> str:
    digest = stable_json_sha256({"entry": entry, "path": list(path_labels)})[:10]
    return f"route:{entry}:{digest}"


def graph_to_label_records(
    graph: RouteGraph,
    *,
    chunk_by_label: dict[str, list[dict[str, Any]]] | None = None,
    source_fingerprint: str = "",
) -> list[dict[str, Any]]:
    """Build draft label summary records from the graph (+ optional chunk texts)."""
    chunk_by_label = chunk_by_label or {}
    records: list[dict[str, Any]] = []
    for name, node in sorted(graph.labels.items()):
        chunks = chunk_by_label.get(name) or []
        evidence: list[str] = []
        upstream: list[str] = []
        summaries: list[str] = []
        for chunk in chunks:
            cid = str(chunk.get("id") or "").strip()
            if cid:
                upstream.append(cid)
            for eid in chunk.get("evidence_item_ids") or []:
                if eid and eid not in evidence:
                    evidence.append(str(eid))
            summary = str(chunk.get("summary") or "").strip()
            if summary:
                summaries.append(summary)
        body = "\n".join(summaries) if summaries else f"Label `{name}` ({node.file_rel_path}:{node.line_start})."
        unresolved = any(e.unresolved for e in node.outgoing)
        if unresolved:
            body += "\n[unresolved dynamic jump/call present]"
        records.append(
            {
                "id": f"label:{name}",
                "kind": KIND_LABEL,
                "status": STATUS_DRAFT,
                "label_id": name,
                "source_files": [node.file_rel_path] if node.file_rel_path else [],
                "evidence_item_ids": evidence,
                "line_span": [node.line_start, node.line_end or node.line_start],
                "source_checksum": sha256_text(body),
                "upstream_artifact_ids": upstream,
                "lineage": empty_lineage(
                    source_fingerprint=source_fingerprint
                    or sha256_text("|".join(graph.source_files)),
                    upstream_dependency_digest=digest_upstream_artifacts(upstream),
                    generated_at="",
                ),
                "summary": body,
                "metadata": {
                    "unresolved_outgoing": unresolved,
                    "outgoing_targets": [
                        e.target_label for e in node.outgoing if e.target_label
                    ],
                },
            }
        )
    return records


def graph_to_route_records(
    graph: RouteGraph,
    *,
    label_records: Sequence[dict[str, Any]] | None = None,
    source_fingerprint: str = "",
) -> list[dict[str, Any]]:
    """Build draft route summary records."""
    label_by_name = {
        str(r.get("label_id") or "").strip(): r
        for r in (label_records or [])
        if str(r.get("label_id") or "").strip()
    }
    records: list[dict[str, Any]] = []
    for route in graph.routes:
        path = list(route.get("label_ids") or [])
        upstream = [f"label:{name}" for name in path]
        evidence: list[str] = []
        parts: list[str] = []
        for name in path:
            rec = label_by_name.get(name)
            if rec:
                for eid in rec.get("evidence_item_ids") or []:
                    if eid not in evidence:
                        evidence.append(str(eid))
                summary = str(rec.get("summary") or "").strip()
                if summary:
                    parts.append(f"## {name}\n{summary}")
            else:
                parts.append(f"## {name}\n(no label summary)")
        if route.get("unresolved"):
            parts.append("[route contains unresolved dynamic jump/call]")
        body = "\n\n".join(parts) if parts else f"Route from {route.get('entry_label')}"
        records.append(
            {
                "id": route["id"],
                "kind": KIND_ROUTE,
                "status": STATUS_DRAFT,
                "route_id": route["id"],
                "source_files": list(route.get("source_files") or []),
                "evidence_item_ids": evidence,
                "line_span": None,
                "source_checksum": sha256_text(body),
                "upstream_artifact_ids": upstream,
                "lineage": empty_lineage(
                    source_fingerprint=source_fingerprint
                    or sha256_text("|".join(graph.source_files)),
                    upstream_dependency_digest=digest_upstream_artifacts(upstream),
                ),
                "summary": body,
                "metadata": {
                    "entry_label": route.get("entry_label"),
                    "label_ids": path,
                    "unresolved": bool(route.get("unresolved")),
                    "shared_labels": list(route.get("shared_labels") or []),
                },
            }
        )
    return records
