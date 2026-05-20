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


def build_reference_blocks(
    *,
    include_translation_memory=True,
    glossary_hits=None,
    history_hits=None,
    story_hits=None,
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
    if story_memory.has_story_hits(story_hits):
        blocks.append(
            "STORY MEMORY:\n"
            f"{story_memory.format_story_hits_block(story_hits, story_char_limit)}"
            f"{story_block_suffix}"
        )
    return "".join(blocks)
