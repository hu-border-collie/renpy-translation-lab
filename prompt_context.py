# -*- coding: utf-8 -*-
from rag_memory import truncate_text
import story_memory


def format_glossary_hits_block(hits, empty_label="(none)"):
    if not hits:
        return empty_label
    lines = []
    for hit in hits:
        source = hit.get("source", "")
        target = hit.get("target", "")
        if not source:
            continue
        if source == target:
            lines.append(f"- Keep unchanged: {source}")
        else:
            lines.append(f"- {source} -> {target}")
    return "\n".join(lines) if lines else empty_label


def format_history_hits_block(
    hits,
    empty_label="(none)",
    char_limit=220,
    include_source_text=True,
):
    if not hits:
        return empty_label
    lines = []
    for hit in hits:
        file_rel_path = hit.get("file_rel_path", "")
        line_start = hit.get("line_start", "")
        line_end = hit.get("line_end", "")
        score = float(hit.get("score", 0.0))
        quality = hit.get("quality_state", "")
        raw_source_text = hit.get("source_text", "")
        raw_translated_text = hit.get("translated_text", "") or raw_source_text
        source_text = truncate_text(raw_source_text, char_limit)
        translated_text = truncate_text(raw_translated_text, char_limit)
        prefix = f"- [{file_rel_path}:{line_start}-{line_end} score={score:.3f} quality={quality}]"
        if include_source_text and source_text and translated_text and raw_source_text != raw_translated_text:
            lines.append(f"{prefix} Source: {source_text} -> Translation: {translated_text}")
        else:
            lines.append(f"{prefix} Translation: {translated_text}")
    return "\n".join(lines) if lines else empty_label


def format_source_hits_block(hits, empty_label="(none)"):
    if not hits:
        return empty_label
    lines = []
    for hit in hits:
        file_rel_path = hit.get("file_rel_path", "")
        line_start = hit.get("line_start", "")
        line_end = hit.get("line_end", "")
        score = float(hit.get("score", 0.0))
        source_text = hit.get("source_text", "")
        prefix = f"- [{file_rel_path}:{line_start}-{line_end} score={score:.3f}]"
        lines.append(f"{prefix} Source excerpt: {source_text}")
    return "\n".join(lines) if lines else empty_label


def format_project_brief_block(brief_text, *, diagnostics="", empty_label="(none)"):
    text = str(brief_text or "").strip()
    if not text:
        return empty_label
    if diagnostics:
        return f"{text}\n\n[{diagnostics}]"
    return text


def format_project_local_context_block(labels=None, routes=None, diagnostics=""):
    sections = []
    for label in labels or []:
        label_id = str((label or {}).get("label_id") or (label or {}).get("id") or "")
        summary = str((label or {}).get("summary") or "").strip()
        if summary:
            sections.append(f"### Label: {label_id}\n{summary}")
    for route in routes or []:
        route_id = str((route or {}).get("route_id") or (route or {}).get("id") or "")
        summary = str((route or {}).get("summary") or "").strip()
        if summary:
            sections.append(f"### Route: {route_id}\n{summary}")
    if diagnostics and sections:
        sections.append(f"[{diagnostics}]")
    return "\n\n".join(sections)


def build_reference_blocks(
    *,
    include_translation_memory=True,
    glossary_hits=None,
    history_hits=None,
    story_hits=None,
    source_hits=None,
    project_brief_text="",
    project_brief_diagnostics="",
    project_local_labels=None,
    project_local_routes=None,
    project_local_diagnostics="",
    history_char_limit=220,
    story_char_limit=1200,
    include_source_text=True,
    empty_label="(none)",
    story_block_suffix="\n\n",
):
    blocks = []
    if include_translation_memory:
        blocks.append(
            "LOCKED TERMS:\n"
            f"{format_glossary_hits_block(glossary_hits or [], empty_label)}\n\n"
            "RETRIEVED MEMORY:\n"
            f"{format_history_hits_block(history_hits or [], empty_label, history_char_limit, include_source_text)}\n\n"
        )
    brief = str(project_brief_text or "").strip()
    if brief:
        blocks.append(
            "PROJECT BRIEF:\n"
            f"{format_project_brief_block(brief, diagnostics=project_brief_diagnostics, empty_label=empty_label)}\n\n"
        )
    local_context = format_project_local_context_block(
        project_local_labels,
        project_local_routes,
        project_local_diagnostics,
    )
    if local_context:
        blocks.append(f"PROJECT LOCAL CONTEXT:\n{local_context}\n\n")
    if source_hits:
        blocks.append(
            "RELATED PROJECT CONTEXT:\n"
            f"{format_source_hits_block(source_hits, empty_label)}\n\n"
        )
    if story_memory.has_story_hits(story_hits):
        blocks.append(
            "STORY MEMORY:\n"
            f"{story_memory.format_story_hits_block(story_hits, story_char_limit)}"
            f"{story_block_suffix}"
        )
    return "".join(blocks)
