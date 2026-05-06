# -*- coding: utf-8 -*-
import json
import os
import re


_EMPTY_GRAPH = {
    "characters": {},
    "relations": [],
    "terms": [],
    "scenes": [],
}


def _empty_graph():
    return {
        "characters": {},
        "relations": [],
        "terms": [],
        "scenes": [],
    }


def _normalize_rel_path(value):
    if not value:
        return ""
    text = str(value).replace("\\", "/").strip()
    text = text.lstrip("./")
    text = text.lstrip("/")
    return text


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_character_map(value):
    characters = {}
    if isinstance(value, dict):
        iterable = value.items()
    elif isinstance(value, list):
        iterable = []
        for item in value:
            if isinstance(item, dict):
                char_id = item.get("id") or item.get("key") or item.get("name")
                iterable.append((char_id, item))
    else:
        iterable = []

    for char_id, raw_data in iterable:
        char_id = _clean_text(char_id)
        if not char_id:
            continue
        data = dict(raw_data) if isinstance(raw_data, dict) else {}
        speaker_ids = [
            _clean_text(item)
            for item in _as_list(data.get("speaker_ids"))
            if _clean_text(item)
        ]
        data["speaker_ids"] = speaker_ids
        characters[char_id] = data
    return characters


def _normalize_terms(value):
    terms = []
    if isinstance(value, dict):
        iterable = [
            {"source": source, "target": target}
            for source, target in value.items()
        ]
    elif isinstance(value, list):
        iterable = value
    else:
        iterable = []

    for item in iterable:
        if isinstance(item, dict):
            source = _clean_text(item.get("source") or item.get("term"))
            target = _clean_text(item.get("target") or item.get("translation"))
            note = _clean_text(item.get("note"))
            aliases = [
                _clean_text(alias)
                for alias in _as_list(item.get("aliases"))
                if _clean_text(alias)
            ]
            if source or target or aliases:
                normalized = dict(item)
                normalized["source"] = source
                normalized["target"] = target
                normalized["note"] = note
                normalized["aliases"] = aliases
                terms.append(normalized)
        else:
            source = _clean_text(item)
            if source:
                terms.append({"source": source, "target": "", "note": "", "aliases": []})
    return terms


def _normalize_dict_list(value):
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def normalize_story_graph(raw_graph):
    if not isinstance(raw_graph, dict):
        return _empty_graph()
    return {
        "characters": _normalize_character_map(raw_graph.get("characters")),
        "relations": _normalize_dict_list(raw_graph.get("relations")),
        "terms": _normalize_terms(raw_graph.get("terms")),
        "scenes": _normalize_dict_list(raw_graph.get("scenes")),
    }


def load_story_graph(path):
    if not path or not os.path.isfile(path):
        return _empty_graph()
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return normalize_story_graph(json.load(handle) or {})
    except Exception as exc:
        print(f"Warning: Failed to load story graph {path}: {exc}")
        return _empty_graph()


def _item_texts(items):
    texts = []
    for item in items or []:
        if isinstance(item, dict):
            text = _clean_text(item.get("text"))
        else:
            text = _clean_text(item)
        if text:
            texts.append(text)
    return texts


def _context_texts(context):
    texts = []
    for item in context or []:
        if isinstance(item, dict):
            text = _clean_text(item.get("text"))
        else:
            text = _clean_text(item)
        if text:
            texts.append(text)
    return texts


def _combined_search_text(target_items, context_past, context_future):
    parts = []
    parts.extend(_context_texts(context_past))
    parts.extend(_item_texts(target_items))
    parts.extend(_context_texts(context_future))
    return "\n".join(parts)


def _speaker_ids_from_items(target_items):
    speaker_ids = set()
    for item in target_items or []:
        if not isinstance(item, dict):
            continue
        for key in ("speaker", "speaker_id", "speaker_name", "character", "who"):
            value = _clean_text(item.get(key))
            if value:
                speaker_ids.add(value)
    return speaker_ids


def _target_line_span(target_items):
    lines = []
    for item in target_items or []:
        if not isinstance(item, dict):
            continue
        line_start = _safe_int(item.get("line_start"))
        line_end = _safe_int(item.get("line_end"))
        line_number = _safe_int(item.get("line_number"))
        zero_based_line = _safe_int(item.get("line"))
        if line_start is not None:
            lines.append(line_start)
        if line_end is not None:
            lines.append(line_end)
        elif line_number is not None:
            lines.append(line_number)
        elif zero_based_line is not None:
            lines.append(zero_based_line + 1)
    if not lines:
        return None, None
    return min(lines), max(lines)


def _looks_word_like(text):
    return bool(re.fullmatch(r"[A-Za-z0-9_ -]+", text or ""))


def _text_contains_alias(search_text_lower, alias):
    alias = _clean_text(alias)
    if len(alias) <= 1:
        return False
    alias_lower = alias.lower()
    if _looks_word_like(alias):
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(alias_lower) + r"(?![A-Za-z0-9_])"
        return re.search(pattern, search_text_lower) is not None
    return alias_lower in search_text_lower


def _character_aliases(char_id, data):
    aliases = [char_id]
    if isinstance(data, dict):
        aliases.extend(_as_list(data.get("speaker_ids")))
        aliases.extend(_as_list(data.get("zh_name")))
        aliases.extend(_as_list(data.get("name")))
        aliases.extend(_as_list(data.get("aliases")))
    cleaned = []
    seen = set()
    for alias in aliases:
        text = _clean_text(alias)
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return cleaned


def _term_aliases(term):
    aliases = []
    if isinstance(term, dict):
        aliases.extend(_as_list(term.get("source")))
        aliases.extend(_as_list(term.get("target")))
        aliases.extend(_as_list(term.get("aliases")))
    cleaned = []
    seen = set()
    for alias in aliases:
        text = _clean_text(alias)
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return cleaned


def _scene_line_span(scene):
    line_start = _safe_int(scene.get("line_start"))
    line_end = _safe_int(scene.get("line_end"))
    if line_start is None and line_end is None:
        return None, None
    if line_start is None:
        line_start = line_end
    if line_end is None:
        line_end = line_start
    if line_start > line_end:
        line_start, line_end = line_end, line_start
    return line_start, line_end


def _line_overlap_score(scene_start, scene_end, target_start, target_end):
    if scene_start is None or target_start is None:
        return 0
    if scene_start <= target_end and target_start <= scene_end:
        overlap = min(scene_end, target_end) - max(scene_start, target_start) + 1
        return 100 + max(0, overlap)
    if scene_end < target_start:
        distance = target_start - scene_end
    else:
        distance = scene_start - target_end
    return max(0, 40 - min(distance, 40))


def _rank_matching_scenes(story_graph, file_rel_path, target_start, target_end, active_char_ids):
    normalized_file = _normalize_rel_path(file_rel_path)
    ranked = []
    for scene in story_graph.get("scenes", []):
        scene_file = _normalize_rel_path(scene.get("file_rel_path"))
        if normalized_file and scene_file and scene_file != normalized_file:
            continue
        score = 0
        if normalized_file and scene_file == normalized_file:
            score += 50
        scene_start, scene_end = _scene_line_span(scene)
        score += _line_overlap_score(scene_start, scene_end, target_start, target_end)
        scene_chars = {
            _clean_text(char)
            for char in _as_list(scene.get("characters"))
            if _clean_text(char)
        }
        score += 8 * len(scene_chars & active_char_ids)
        if score > 0:
            ranked.append((score, scene))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [scene for _, scene in ranked]


def _collect_active_characters(story_graph, search_text_lower, speaker_ids):
    active = set()
    characters = story_graph.get("characters", {})
    for char_id, data in characters.items():
        aliases = _character_aliases(char_id, data)
        if any(alias in speaker_ids for alias in aliases):
            active.add(char_id)
            continue
        if any(_text_contains_alias(search_text_lower, alias) for alias in aliases):
            active.add(char_id)
    return active


def retrieve_story_hits(
    story_graph,
    file_rel_path,
    target_items,
    context_past=None,
    context_future=None,
    top_k_relations=6,
    top_k_terms=12,
    include_scene_summary=True,
):
    story_graph = normalize_story_graph(story_graph)
    search_text = _combined_search_text(target_items, context_past, context_future)
    search_text_lower = search_text.lower()
    speaker_ids = _speaker_ids_from_items(target_items)
    target_start, target_end = _target_line_span(target_items)
    active_char_ids = _collect_active_characters(story_graph, search_text_lower, speaker_ids)

    matching_scenes = _rank_matching_scenes(
        story_graph,
        file_rel_path,
        target_start,
        target_end,
        active_char_ids,
    )
    for scene in matching_scenes[:3]:
        for char_id in _as_list(scene.get("characters")):
            char_id = _clean_text(char_id)
            if char_id:
                active_char_ids.add(char_id)

    characters = []
    for char_id in sorted(active_char_ids):
        data = story_graph.get("characters", {}).get(char_id, {})
        item = {"id": char_id}
        if isinstance(data, dict):
            for key in ("zh_name", "style", "note"):
                value = _clean_text(data.get(key))
                if value:
                    item[key] = value
        characters.append(item)

    term_hits = []
    max_terms = max(0, int(top_k_terms or 0))
    for term in story_graph.get("terms", []):
        if len(term_hits) >= max_terms:
            break
        aliases = _term_aliases(term)
        if any(_text_contains_alias(search_text_lower, alias) for alias in aliases):
            term_hits.append(term)

    relation_hits = []
    for relation in story_graph.get("relations", []):
        left = _clean_text(relation.get("left"))
        right = _clean_text(relation.get("right"))
        if not left or not right:
            continue
        left_active = left in active_char_ids
        right_active = right in active_char_ids
        if not (left_active and right_active):
            continue
        relation_hits.append(relation)

    def relation_confidence(item):
        try:
            return float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    relation_hits.sort(key=relation_confidence, reverse=True)

    scene_hits = matching_scenes[:3] if include_scene_summary else []
    return {
        "characters": characters,
        "relations": relation_hits[:max(0, int(top_k_relations or 0))],
        "terms": term_hits,
        "scenes": scene_hits,
    }


def _format_confidence(value):
    try:
        return f" confidence={float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _append_limited(lines, line, max_chars):
    line = _clean_text(line)
    if not line:
        return False
    candidate = "\n".join(lines + [line]) if lines else line
    if len(candidate) <= max_chars:
        lines.append(line)
        return True
    if not lines and max_chars > 3:
        lines.append(line[:max_chars - 3].rstrip() + "...")
    return False


def has_story_hits(story_hits):
    if not isinstance(story_hits, dict):
        return False
    for key in ("characters", "relations", "terms", "scenes"):
        hits = story_hits.get(key)
        if isinstance(hits, list) and hits:
            return True
    return False


def format_story_hits_block(story_hits, max_chars, empty_label="(none)"):
    if not isinstance(story_hits, dict):
        return empty_label
    max_chars = max(1, int(max_chars or 1))
    lines = []

    characters = story_hits.get("characters") or []
    if characters:
        _append_limited(lines, "Characters:", max_chars)
        for item in characters:
            char_id = _clean_text(item.get("id"))
            zh_name = _clean_text(item.get("zh_name"))
            style = _clean_text(item.get("style"))
            note = _clean_text(item.get("note"))
            label = char_id
            if zh_name:
                label = f"{label} ({zh_name})"
            detail = style or note
            line = f"- {label}: {detail}" if detail else f"- {label}"
            if not _append_limited(lines, line, max_chars):
                break

    relations = story_hits.get("relations") or []
    if relations:
        _append_limited(lines, "Relations:", max_chars)
        for item in relations:
            left = _clean_text(item.get("left"))
            right = _clean_text(item.get("right"))
            relation_type = _clean_text(item.get("type"))
            note = _clean_text(item.get("note"))
            detail = relation_type or "related"
            confidence = _format_confidence(item.get("confidence"))
            line = f"- {left} -> {right} [{detail}{confidence}]"
            if note:
                line += f": {note}"
            if not _append_limited(lines, line, max_chars):
                break

    terms = story_hits.get("terms") or []
    if terms:
        _append_limited(lines, "Terms:", max_chars)
        for item in terms:
            source = _clean_text(item.get("source"))
            target = _clean_text(item.get("target"))
            note = _clean_text(item.get("note"))
            if source and target and source != target:
                line = f"- {source} -> {target}"
            elif source:
                line = f"- Term: {source}"
            elif target:
                line = f"- Term target: {target}"
            else:
                continue
            if note:
                line += f": {note}"
            if not _append_limited(lines, line, max_chars):
                break

    scenes = story_hits.get("scenes") or []
    if scenes:
        _append_limited(lines, "Scenes:", max_chars)
        for item in scenes:
            file_rel_path = _normalize_rel_path(item.get("file_rel_path"))
            line_start, line_end = _scene_line_span(item)
            summary = _clean_text(item.get("summary"))
            chars = [
                _clean_text(char)
                for char in _as_list(item.get("characters"))
                if _clean_text(char)
            ]
            location = file_rel_path or "scene"
            if line_start is not None and line_end is not None:
                location = f"{location}:{line_start}-{line_end}"
            line = f"- {location}"
            details = []
            if summary:
                details.append(summary)
            if chars:
                details.append("characters: " + ", ".join(chars))
            if details:
                line += ": " + "; ".join(details)
            if not _append_limited(lines, line, max_chars):
                break

    if not lines:
        return empty_label[:max_chars]
    return "\n".join(lines)[:max_chars]
