# -*- coding: utf-8 -*-
"""Shared Sync/Batch retrieval and published-analysis prompt helpers (issue #341).

Embedding identity checks stay in :mod:`embedding_runtime`. This module shapes
query text, compatibility-gated hits, independent prompt partitions, and
Published Project Analysis skip diagnostics so Sync and Gemini Batch consume
the same contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

import prompt_context
from rag_memory import truncate_text
import story_memory


def compact_text(text: object) -> str:
    if not isinstance(text, str):
        return ''
    return re.sub(r'\s+', ' ', text).strip()


def item_text(item: object) -> str:
    if isinstance(item, Mapping):
        return str(item.get('text') or item.get('source') or '')
    return str(item or '')


def compact_item_texts(items: Sequence[object] | None) -> list[str]:
    compacted = []
    for item in items or []:
        text = compact_text(item_text(item))
        if text:
            compacted.append(text)
    return compacted


def embedding_provider_diagnostics(settings: object) -> dict[str, Any]:
    """Return the credential-free identity of the active embedding provider.

    Sync and Gemini Batch construct their runtime settings in different
    modules, but the provider contract must be observable in the same shape.
    Keeping this small adapter here prevents either executor from serializing
    endpoints, API-key names, or other transport-only details into plan
    diagnostics.
    """

    public_dict = getattr(settings, 'public_dict', None)
    if not callable(public_dict):
        return {}
    try:
        payload = public_dict()
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def build_source_only_query_text(target_items: Sequence[object] | None) -> str:
    """Build a TARGET-only query for Source Index retrieval."""

    target_lines = compact_item_texts(target_items)
    if not target_lines:
        return ''
    return 'Target:\n' + '\n'.join(f'- {text}' for text in target_lines)


def build_history_query_text(
    target_items: Sequence[object] | None,
    context_past: Sequence[object] | None = None,
) -> str:
    """Build a history-store query; optional local past is Batch-only context."""

    parts = []
    local_past = compact_item_texts(list(context_past or [])[-2:])
    target_lines = compact_item_texts(target_items)
    if local_past:
        parts.append('Context before:\n' + '\n'.join(f'- {text}' for text in local_past))
    if target_lines:
        parts.append('Target:\n' + '\n'.join(f'- {text}' for text in target_lines))
    return '\n\n'.join(parts)


def _compatibility_payload(diagnostics: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = (diagnostics or {}).get('embedding_compatibility') or {}
    if not isinstance(raw, Mapping):
        return {}
    return dict(raw)


def shape_history_hits(matches: Sequence[Mapping[str, Any]] | None, char_limit: int) -> list[dict[str, Any]]:
    hits = []
    for match in matches or []:
        hits.append(
            {
                'memory_id': match.get('memory_id', ''),
                'file_rel_path': match.get('file_rel_path', ''),
                'line_start': match.get('line_start', 0),
                'line_end': match.get('line_end', 0),
                'source_text': truncate_text(match.get('source_text', ''), char_limit),
                'translated_text': truncate_text(match.get('translated_text', ''), char_limit),
                'quality_state': match.get('quality_state', ''),
                'score': float(match.get('score', 0.0)),
            }
        )
    return hits


def shape_source_hits(
    matches: Sequence[Mapping[str, Any]] | None,
    char_limit: int,
    char_budget: int = 0,
) -> tuple[list[dict[str, Any]], int, int]:
    """Shape source hits under both per-hit and aggregate text budgets.

    ``char_limit`` protects each excerpt while ``char_budget`` protects the
    complete Source Index partition for one request. A zero aggregate budget
    preserves the historical per-hit-only behavior; positive budgets consume
    ranked hits in order and truncate the final excerpt when necessary.
    """

    hits = []
    truncated_count = 0
    source_context_chars = 0
    try:
        remaining_budget = max(0, int(char_budget or 0))
    except (TypeError, ValueError):
        remaining_budget = 0
    bounded = remaining_budget > 0
    try:
        per_hit_limit = int(char_limit or 0)
    except (TypeError, ValueError):
        per_hit_limit = 0
    match_list = list(matches or [])
    for match in match_list:
        if bounded and remaining_budget <= 0:
            break
        source_text = match.get('source_text', '')
        effective_limit = per_hit_limit
        if bounded:
            effective_limit = min(effective_limit, remaining_budget) if effective_limit > 0 else remaining_budget
        truncated_source_text = truncate_text(source_text, effective_limit)
        was_truncated = isinstance(source_text, str) and truncated_source_text != source_text
        if was_truncated:
            truncated_count += 1
        source_context_chars += len(truncated_source_text)
        if bounded:
            remaining_budget = max(0, remaining_budget - len(truncated_source_text))
        hits.append(
            {
                'source_id': match.get('source_id', ''),
                'file_rel_path': match.get('file_rel_path', ''),
                'line_start': match.get('line_start', 0),
                'line_end': match.get('line_end', 0),
                'source_text': truncated_source_text,
                'source_text_truncated': was_truncated,
                'score': float(match.get('score', 0.0)),
            }
        )
    return hits, truncated_count, source_context_chars


def retrieve_history_hits_compatible(
    store: object,
    query_vector: Sequence[float],
    query_identity: object,
    *,
    top_k: int,
    min_similarity: float,
    char_limit: int,
    query_text: str = '',
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search a RAG history store only when query/store identities match."""

    matches, diagnostics = store.search_history_compatible(
        query_vector,
        query_identity,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    compatibility = _compatibility_payload(diagnostics)
    stats: dict[str, Any] = {
        'enabled': True,
        'query_text': truncate_text(query_text, 400),
        'hit_count': 0,
        'embedding_compatibility': compatibility,
    }
    if not compatibility.get('compatible'):
        stats['reason'] = 'rebuild_store'
        stats['action'] = compatibility.get('action') or 'rebuild_store'
        return [], stats
    hits = shape_history_hits(matches, char_limit)
    stats['hit_count'] = len(hits)
    return hits, stats


def retrieve_source_hits_compatible(
    store: object,
    query_vector: Sequence[float],
    query_identity: object,
    *,
    top_k: int,
    min_similarity: float,
    char_limit: int,
    query_text: str = '',
    embedding_model: str | None = None,
    embedding_task_type: str | None = None,
    embedding_dim: int | None = None,
    char_budget: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search a Source Index store only when query/store identities match."""

    matches, diagnostics = store.search_segments_compatible(
        query_vector,
        query_identity,
        top_k=top_k,
        min_similarity=min_similarity,
        embedding_model=embedding_model,
        embedding_task_type=embedding_task_type,
        embedding_dim=embedding_dim,
    )
    compatibility = _compatibility_payload(diagnostics)
    stats: dict[str, Any] = {
        'enabled': True,
        'query_text': truncate_text(query_text, 400),
        'query_char_count': len(query_text or ''),
        'hit_count': 0,
        'truncated_count': 0,
        'source_context_chars': 0,
        'source_context_char_budget': 0,
        'store_dir': getattr(store, 'store_dir', ''),
        'store_schema_version': (getattr(store, 'metadata', {}) or {}).get('schema_version'),
        'embedding_compatibility': compatibility,
        'search_diagnostics': {
            key: value
            for key, value in dict(diagnostics or {}).items()
            if key != 'embedding_compatibility'
        },
    }
    try:
        normalized_budget = max(0, int(char_budget or 0))
    except (TypeError, ValueError):
        normalized_budget = 0
    stats['source_context_char_budget'] = normalized_budget
    if not compatibility.get('compatible'):
        stats['reason'] = 'rebuild_store'
        stats['action'] = compatibility.get('action') or 'rebuild_store'
        return [], stats
    hits, truncated_count, source_context_chars = shape_source_hits(
        matches,
        char_limit,
        char_budget=char_budget,
    )
    stats['hit_count'] = len(hits)
    stats['truncated_count'] = truncated_count
    stats['source_context_chars'] = source_context_chars
    stats['source_context_budget_dropped_count'] = max(
        0,
        len(matches or []) - len(hits),
    ) if normalized_budget else 0
    stats['source_context_budget_exhausted'] = bool(
        normalized_budget and source_context_chars >= normalized_budget
    )
    stats['matched_count'] = diagnostics.get('matched_before_top_k', len(matches))
    stats['filtered_count'] = diagnostics.get('metadata_filtered_count', 0)
    stats['stale_hits_skipped'] = diagnostics.get('metadata_filtered_count', 0)
    stats['below_similarity_count'] = diagnostics.get('below_similarity_count', 0)
    return hits, stats


def render_retrieval_reference_text(
    history_hits: Sequence[Mapping[str, Any]] | None,
    story_hits: object,
    source_hits: Sequence[Mapping[str, Any]] | None,
    *,
    history_char_limit: int,
    story_char_limit: int,
    include_source_text: bool = True,
) -> str:
    """Render RAG / Source Index / Story Memory partitions without glossary."""

    blocks = []
    if history_hits:
        blocks.append(
            'RETRIEVED MEMORY:\n'
            f'{prompt_context.format_history_hits_block(history_hits, char_limit=history_char_limit, include_source_text=include_source_text)}\n\n'
        )
    if source_hits:
        blocks.append(
            'RELATED PROJECT CONTEXT:\n'
            f'{prompt_context.format_source_hits_block(source_hits)}\n\n'
        )
    if story_memory.has_story_hits(story_hits):
        blocks.append(
            'STORY MEMORY:\n'
            f'{story_memory.format_story_hits_block(story_hits, story_char_limit)}\n\n'
        )
    return ''.join(blocks)


def render_analysis_reference_text(project_context: Mapping[str, Any] | None) -> str:
    """Render the published Project Analysis partition; empty when not injectable."""

    project_context = project_context or {}
    blocks = []
    brief = str(project_context.get('text') or '').strip()
    if brief:
        diagnostics = str(project_context.get('diagnostics') or '')
        blocks.append(
            'PROJECT BRIEF:\n'
            f'{prompt_context.format_project_brief_block(brief, diagnostics=diagnostics)}\n\n'
        )
    local_context = prompt_context.format_project_local_context_block(
        project_context.get('labels') or [],
        project_context.get('routes') or [],
        str(project_context.get('local_diagnostics') or ''),
    )
    if local_context.strip():
        blocks.append(f'PROJECT LOCAL CONTEXT:\n{local_context}\n\n')
    return ''.join(blocks)


def analysis_skip_diagnostics(project_context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Public identity/status for a published-brief injection attempt."""

    payload = dict(project_context or {})
    status = payload.get('status') if isinstance(payload.get('status'), Mapping) else {}
    brief_status = str(
        payload.get('brief_status')
        or status.get('brief_status')
        or ('published' if payload.get('injectable') else '')
        or 'missing'
    )
    reason = str(payload.get('reason') or '')
    if not payload.get('injectable') and not reason:
        reason = 'injection_disabled'
    lineage = {}
    artifacts = status.get('artifacts') if isinstance(status, Mapping) else {}
    if isinstance(artifacts, Mapping):
        brief_entry = artifacts.get('project_brief') or {}
        if isinstance(brief_entry, Mapping):
            lineage = dict(brief_entry.get('lineage') or {})
    fingerprint = str(lineage.get('source_fingerprint') or payload.get('source_fingerprint') or '')
    return {
        'injectable': bool(payload.get('injectable')),
        'reason': reason,
        'brief_status': brief_status,
        'source_fingerprint': fingerprint,
        'diagnostics': str(payload.get('diagnostics') or ''),
        'injected_chars': len(str(payload.get('text') or '')),
    }
