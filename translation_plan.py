# -*- coding: utf-8 -*-
"""Execution-strategy-neutral translation plan (issue #346).

A :class:`TranslationPlan` captures the semantic contract the model must see —
translation units, chunk grouping, prompts, response schema, and assembled
context. Executors (the sync runtime and the Gemini batch builder, wired in
#346 P2/P3) consume the plan instead of assembling their own prompts; the
transport layer may only add envelopes, job ids, and scheduling metadata on
top of :class:`TranslationRequest`.

Hard constraints (enforced by tests):

* no optional SDK imports, no network access, no credential values — profile
  snapshots pass through :func:`redact_sensitive` before entering a plan;
* identical inputs rebuild byte-identical plans: every id and fingerprint is
  content-derived via :func:`canonical_json`, never time- or randomness-based,
  and unordered inputs (sets) are canonicalized before hashing;
* ``run_id`` is an audit label only; retrieved content is captured by
  ``prompt_fingerprint``, not by ``plan_id``;
* retrieval and analysis layer texts are budgeted, newline-normalized, and
  joined on a fixed blank-line separator; their ``char_used`` counts exactly
  the bytes the canonical prompt embeds. The required/local/project layers
  are canonical renderings owned by the ``translation_core`` prompt builders
  and are accounted in the assembly for budget and diagnostics only — the
  request's ``user_prompt`` stays the authoritative record of what the model
  sees.
"""

from dataclasses import dataclass, field
import difflib
import hashlib
import json
import re
from collections.abc import Mapping

import model_profile
import translation_core

PLAN_SCHEMA_VERSION = 1

STRATEGY_SYNC = model_profile.ExecutionStrategy.SYNC.value
STRATEGY_GEMINI_BATCH = model_profile.ExecutionStrategy.GEMINI_BATCH.value
EXECUTION_STRATEGIES = (STRATEGY_SYNC, STRATEGY_GEMINI_BATCH)

CANONICAL_TEMPERATURE = 0.2  # D6

CONTEXT_LAYER_REQUIRED = 'required'
CONTEXT_LAYER_LOCAL = 'local'
CONTEXT_LAYER_PROJECT = 'project'
CONTEXT_LAYER_RETRIEVAL = 'retrieval'
CONTEXT_LAYER_ANALYSIS = 'analysis'
CONTEXT_LAYER_RANKS = {
    CONTEXT_LAYER_REQUIRED: 1,
    CONTEXT_LAYER_LOCAL: 2,
    CONTEXT_LAYER_PROJECT: 3,
    CONTEXT_LAYER_RETRIEVAL: 4,
    CONTEXT_LAYER_ANALYSIS: 5,
}

# Keys are normalized (lowercased, '-'/'_'/' ' stripped) before matching so
# header spellings like ``X-Api-Key`` are caught. ``token`` matches by suffix
# (``X-Token`` → ``xtoken``): a bare substring rule would redact legitimate
# token-count keys such as ``max_output_tokens`` / ``usage_tokens`` — plural
# ``tokens`` never matches the singular suffix.
_SENSITIVE_KEY_SUBSTRINGS = (
    'apikey',
    'authorization',
    'authsecret',
    'accesstoken',
    'authtoken',
    'refreshtoken',
    'idtoken',
    'secret',
    'password',
    'credentialvalue',
    'bearer',
    'accesskey',
    'privatekey',
    'clientkey',
    'signingkey',
)
_SENSITIVE_KEY_SUFFIXES = ('token',)
_REDACTED_KEY_ALLOWLIST = ('credential_ref', 'credential_refs')
REDACTED_VALUE = '[redacted]'


def _normalize_sensitive_key(key):
    return re.sub(r'[-_ ]', '', str(key).lower())


def _is_sensitive_key(key):
    if str(key) in _REDACTED_KEY_ALLOWLIST:
        return False
    normalized = _normalize_sensitive_key(key)
    if normalized.endswith(_SENSITIVE_KEY_SUFFIXES):
        return True
    return any(marker in normalized for marker in _SENSITIVE_KEY_SUBSTRINGS)


def redact_sensitive(value):
    """Recursively replace credential-shaped values with a redaction marker.

    Keys whose normalized name carries API key, authorization, secret,
    password, or bearer markers — or ends with a singular ``token`` suffix
    (``X-Token``, ``session_token``) — have their values replaced wholesale,
    including header spellings like ``X-Api-Key``. Plural count keys
    (``max_output_tokens``) stay intact. ``credential_ref`` objects pass
    through: they hold lookup references (slot ids, env var names), never
    credential values (see ``model_profile.CredentialRef``).
    """
    if isinstance(value, Mapping):
        redacted = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                redacted[str(key)] = REDACTED_VALUE
            else:
                redacted[str(key)] = redact_sensitive(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(item) for item in value]
    return value

CONTEXT_TOKEN_ESTIMATE_METHOD = 'char_upper_bound'


# --- Stable serialization and fingerprints ------------------------------------


def canonical_json(value):
    """Serialize ``value`` deterministically for hashing or golden fixtures.

    Keys are sorted, separators are compact, non-ASCII passes through, and
    NaN/Infinity are rejected so the output is a stable UTF-8 byte sequence.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    )


def sha256_hex(text):
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def short_fingerprint(text):
    """16-hex content fingerprint used for plan/request/prompt identity."""
    return sha256_hex(text)[:16]


def canonical_semantic_request(request):
    """Return the canonical bytes used to compare executor-neutral requests."""
    if isinstance(request, TranslationRequest):
        payload = request.semantic_payload()
    elif isinstance(request, Mapping):
        payload = {
            'system_instruction': str(request.get('system_instruction') or ''),
            'user_prompt': str(request.get('user_prompt') or ''),
            'response_schema': dict(request.get('response_schema') or {}),
            'expected_ids': list(request.get('expected_ids') or []),
            'context_assembly': dict(request.get('context_assembly') or {}),
        }
    else:
        raise TypeError('request must be a TranslationRequest or mapping')
    return canonical_json(payload)


def recompute_request_fingerprints(request):
    """Recompute semantic and audit fingerprints without trusting stored fields."""
    if not isinstance(request, TranslationRequest):
        request = TranslationRequest.from_dict(request)
    prompt_fingerprint = short_fingerprint(canonical_semantic_request(request))
    request_fingerprint = short_fingerprint(canonical_json({
        'prompt_fingerprint': prompt_fingerprint,
        'generation_config': dict(request.generation_config or {}),
        'transport_metadata': dict(request.transport_metadata or {}),
    }))
    return prompt_fingerprint, request_fingerprint


def plan_diff(left_requests, right_requests, *, left_label='sync', right_label='gemini_batch'):
    """Compare normalized model-visible requests and return a readable report.

    Transport and generation metadata are intentionally excluded.  Request
    pairs are matched by position because stable request ids include the
    execution strategy through ``plan_id`` while chunk order is shared.
    """
    left = list(left_requests or [])
    right = list(right_requests or [])
    entries = []
    for index in range(max(len(left), len(right))):
        left_request = left[index] if index < len(left) else None
        right_request = right[index] if index < len(right) else None
        left_text = (
            canonical_semantic_request(left_request)
            if left_request is not None
            else canonical_json({'missing_request': True})
        )
        right_text = (
            canonical_semantic_request(right_request)
            if right_request is not None
            else canonical_json({'missing_request': True})
        )
        if left_text == right_text:
            continue
        left_chunk = str(
            getattr(left_request, 'chunk_id', '')
            or ((left_request or {}).get('chunk_id') if isinstance(left_request, Mapping) else '')
            or '<missing>'
        )
        right_chunk = str(
            getattr(right_request, 'chunk_id', '')
            or ((right_request or {}).get('chunk_id') if isinstance(right_request, Mapping) else '')
            or '<missing>'
        )
        unified = '\n'.join(difflib.unified_diff(
            json.dumps(json.loads(left_text), ensure_ascii=False, indent=2, sort_keys=True).splitlines(),
            json.dumps(json.loads(right_text), ensure_ascii=False, indent=2, sort_keys=True).splitlines(),
            fromfile=f'{left_label}:{left_chunk}',
            tofile=f'{right_label}:{right_chunk}',
            lineterm='',
        ))
        entries.append({
            'index': index,
            'left_chunk_id': left_chunk,
            'right_chunk_id': right_chunk,
            'unified_diff': unified,
        })
    return {
        'equivalent': not entries,
        'left_label': str(left_label),
        'right_label': str(right_label),
        'left_request_count': len(left),
        'right_request_count': len(right),
        'differences': entries,
    }


def format_plan_diff(report):
    """Format :func:`plan_diff` output for logs, tests, and review artifacts."""
    payload = dict(report or {})
    if payload.get('equivalent'):
        return (
            'TranslationPlan semantic requests are equivalent '
            f"({int(payload.get('left_request_count') or 0)} requests)."
        )
    lines = [
        'TranslationPlan semantic request mismatch: '
        f"{payload.get('left_label') or 'left'}="
        f"{int(payload.get('left_request_count') or 0)}, "
        f"{payload.get('right_label') or 'right'}="
        f"{int(payload.get('right_request_count') or 0)}."
    ]
    for entry in payload.get('differences') or []:
        lines.append(
            f"request[{int(entry.get('index') or 0)}] "
            f"{entry.get('left_chunk_id') or '<missing>'} <> "
            f"{entry.get('right_chunk_id') or '<missing>'}"
        )
        if entry.get('unified_diff'):
            lines.append(str(entry['unified_diff']))
    return '\n'.join(lines)


def validate_plan_fingerprint(payload):
    """Validate a persisted plan fingerprint without rebuilding its prompts."""
    if not isinstance(payload, Mapping):
        raise ValueError('TranslationPlan must be an object.')
    if int(payload.get('schema_version') or 0) != PLAN_SCHEMA_VERSION:
        raise ValueError('Unsupported TranslationPlan schema version.')
    fingerprint_payload = dict(payload)
    recorded = str(fingerprint_payload.pop('plan_fingerprint', '') or '')
    fingerprint_payload.pop('run_id', None)
    expected = short_fingerprint(canonical_json(fingerprint_payload))
    if not recorded or recorded != expected:
        raise ValueError('TranslationPlan fingerprint is invalid or stale.')
    return recorded


def refresh_plan_fingerprint(payload):
    """Return a deep-copied plan payload with a freshly computed fingerprint."""
    refreshed = json.loads(canonical_json(dict(payload or {})))
    fingerprint_payload = dict(refreshed)
    fingerprint_payload.pop('plan_fingerprint', None)
    fingerprint_payload.pop('run_id', None)
    refreshed['plan_fingerprint'] = short_fingerprint(
        canonical_json(fingerprint_payload)
    )
    return refreshed


def is_material_context_drop(entry):
    """Return whether a drop record represents model-visible context loss."""
    entry = dict(entry or {})
    if (
        entry.get('reason') == 'duplicate_text'
        and int(entry.get('char_used') or 0) <= 0
    ):
        return False
    return True


def summarize_request_diagnostics(request_summaries):
    """Aggregate stable trim/drop/provider counters from request summaries.

    Provider payloads are intentionally summarized by status and reason only;
    the detailed, credential-free payload remains attached to each context
    layer for diagnostics and golden comparisons.
    """
    summaries = list(request_summaries or [])
    provider_status_counts = {}
    provider_downgrade_reasons = {}
    provider_diagnostic_requests = 0
    provider_downgrade_count = 0

    def provider_status(provider_name, payload):
        if not isinstance(payload, Mapping):
            return 'invalid'
        if payload.get('enabled') is False:
            return 'disabled'
        if payload.get('injectable') is False:
            reason = str(payload.get('reason') or 'not_injectable')
            return 'disabled' if reason == 'injection_disabled' else reason
        compatibility = payload.get('embedding_compatibility')
        if isinstance(compatibility, Mapping) and compatibility.get('compatible') is False:
            return 'incompatible'
        reason = payload.get('failure_reason') or payload.get('reason')
        if reason:
            return str(reason)
        if int(payload.get('source_context_budget_dropped_count') or 0) > 0:
            return 'budget_cropped'
        if int(payload.get('stale_hits_skipped') or 0) > 0:
            return 'stale_hits_skipped'
        if int(payload.get('truncated_count') or 0) > 0:
            return 'excerpt_cropped'
        return 'available'

    for summary in summaries:
        request_has_provider_diagnostic = False
        for layer in ((summary.get('context_diagnostics') or {}).get('layers') or []):
            provider = ((layer or {}).get('diagnostics') or {}).get('provider')
            if not isinstance(provider, Mapping):
                continue
            for provider_name, payload in provider.items():
                if provider_name == 'embedding_provider':
                    continue
                request_has_provider_diagnostic = True
                status = provider_status(provider_name, payload)
                key = f'{provider_name}:{status}'
                provider_status_counts[key] = provider_status_counts.get(key, 0) + 1
                if status not in {
                    'available',
                    'disabled',
                    'empty_query',
                    'excerpt_cropped',
                    'stale_hits_skipped',
                }:
                    provider_downgrade_count += 1
                    provider_downgrade_reasons[key] = (
                        provider_downgrade_reasons.get(key, 0) + 1
                    )
        if request_has_provider_diagnostic:
            provider_diagnostic_requests += 1

    return {
        'request_count': len(summaries),
        'context_truncated_requests': sum(
            1
            for summary in summaries
            if any(
                bool((layer or {}).get('truncated'))
                for layer in (
                    (summary.get('context_diagnostics') or {}).get('layers') or []
                )
            )
        ),
        'context_dropped_entries': sum(
            1
            for summary in summaries
            for entry in (
                (summary.get('context_diagnostics') or {}).get('dropped') or []
            )
            if is_material_context_drop(entry)
        ),
        'context_provider_diagnostic_requests': provider_diagnostic_requests,
        'context_provider_downgrade_count': provider_downgrade_count,
        'context_provider_status_counts': provider_status_counts,
        'context_provider_downgrade_reasons': provider_downgrade_reasons,
    }


def derive_translation_plan_payload(parent_payload, request_payloads, *, derivation_kind):
    """Create a signed child-plan view for split or D7-derived requests.

    The parent plan remains immutable.  The child keeps its semantic ``plan_id``
    and full-project source identity, but binds only the requests that the child
    package can dispatch. Retry request lineage remains in each request's
    transport metadata and is also recorded in the plan artifacts.
    """
    validate_plan_fingerprint(parent_payload)
    parent = json.loads(canonical_json(dict(parent_payload or {})))
    parent_chunks = {
        str((chunk or {}).get('chunk_id') or ''): dict(chunk or {})
        for chunk in parent.get('chunks') or []
    }
    parent_summaries_by_chunk = {
        str((summary or {}).get('chunk_id') or ''): dict(summary or {})
        for summary in parent.get('request_summaries') or []
    }
    parent_request_ids = {
        str((summary or {}).get('request_id') or '')
        for summary in parent.get('request_summaries') or []
    }
    summaries = []
    chunks = []
    for index, payload in enumerate(request_payloads or [], start=1):
        raw = (
            payload.to_dict()
            if isinstance(payload, TranslationRequest)
            else dict(payload or {})
        )
        request = TranslationRequest.from_dict(raw)
        prompt_fingerprint, request_fingerprint = recompute_request_fingerprints(
            request
        )
        if (
            request.plan_id != str(parent.get('plan_id') or '')
            or request.prompt_fingerprint != prompt_fingerprint
            or request.request_fingerprint != request_fingerprint
        ):
            raise ValueError('Child TranslationPlan request fingerprint is stale.')
        parent_summary = parent_summaries_by_chunk.get(request.chunk_id)
        exact_parent_request = bool(
            parent_summary
            and str(parent_summary.get('request_id') or '') == request.request_id
            and str(parent_summary.get('prompt_fingerprint') or '')
            == prompt_fingerprint
            and str(parent_summary.get('request_fingerprint') or '')
            == request_fingerprint
        )
        retry_parent_request_id = str(
            (request.transport_metadata or {}).get('retry_parent_request_id') or ''
        )
        if not exact_parent_request and (
            str(derivation_kind or '') != 'retry'
            or retry_parent_request_id not in parent_request_ids
        ):
            raise ValueError(
                'Child TranslationPlan request is not bound to its parent plan.'
            )
        summaries.append(request.summary())
        existing = parent_chunks.get(request.chunk_id)
        if existing is not None:
            chunks.append(existing)
            continue
        items = list(raw.get('items') or [])
        chunks.append({
            'chunk_id': request.chunk_id or str(raw.get('key') or ''),
            'chunk_index': int(raw.get('chunk_index') or index),
            'file_rel_path': str(raw.get('file_rel_path') or ''),
            'file_path': str(raw.get('file_path') or ''),
            'line_numbers': list(raw.get('line_numbers') or [
                item.get('line') for item in items if item.get('line') is not None
            ]),
            'unit_ids': list(request.expected_ids),
            'source_char_count': int(raw.get('source_char_count') or sum(
                len(str(item.get('source', item.get('text', '')) or ''))
                for item in items
            )),
            'context_window_spec': dict(raw.get('context_window_spec') or {}),
        })
    artifacts = dict(parent.get('artifacts') or {})
    artifacts['derivation'] = {
        'kind': str(derivation_kind or 'child'),
        'parent_plan_id': str(parent.get('plan_id') or ''),
        'parent_plan_fingerprint': str(parent.get('plan_fingerprint') or ''),
    }
    parent['chunks'] = chunks
    parent['request_summaries'] = summaries
    parent['artifacts'] = artifacts
    return refresh_plan_fingerprint(parent)


def source_identity_differences(expected, actual):
    """Return stable, non-sensitive source/adapter identity mismatch codes."""
    expected = dict(expected or {})
    actual = dict(actual or {})
    differences = []
    for field, code in (
        ('engine', 'engine_changed'),
        ('adapter_version', 'adapter_version_changed'),
        ('project_identity_digest', 'project_identity_changed'),
        ('source_snapshot_fingerprint', 'source_snapshot_changed'),
        ('file_digests', 'source_file_digests_changed'),
    ):
        if canonical_json(expected.get(field) or ({} if field == 'file_digests' else '')) != canonical_json(
            actual.get(field) or ({} if field == 'file_digests' else '')
        ):
            differences.append(code)
    return differences


def _canonical_term_sequence(terms):
    """Deterministic sequence for term iterables of any collection type.

    A bare ``str`` is treated as one single term, never split into
    characters. Lists and tuples keep their given order (deduplicated);
    ``set`` / ``frozenset`` are sorted, because set iteration order is
    ``PYTHONHASHSEED``-dependent and plan ids must be reproducible across
    processes.
    """
    if isinstance(terms, str):
        items = [terms]
    else:
        items = list(terms or [])
        if isinstance(terms, (set, frozenset)):
            items.sort()
    seen = set()
    ordered = []
    for item in items:
        marker = repr(item)
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(item)
    return ordered


# --- Chunking policy and stable ids (D4) --------------------------------------


@dataclass(frozen=True)
class ChunkPolicy:
    """D4-A: shared chunk grouping for both execution strategies."""

    max_items: int = translation_core.CANONICAL_CHUNK_MAX_ITEMS
    max_chars: int = translation_core.CANONICAL_CHUNK_MAX_CHARS

    def to_dict(self):
        return {'max_items': int(self.max_items), 'max_chars': int(self.max_chars)}


@dataclass(frozen=True)
class ContextPolicy:
    """D1/D5: local-window algorithm limits and reference-block budgets."""

    local_context_before: int = translation_core.CANONICAL_LOCAL_CONTEXT_BEFORE
    local_context_after: int = translation_core.CANONICAL_LOCAL_CONTEXT_AFTER
    history_char_limit: int = translation_core.CANONICAL_HISTORY_CHAR_LIMIT
    story_char_limit: int = translation_core.CANONICAL_STORY_CHAR_LIMIT
    source_index_char_limit: int = 0
    analysis_char_limit: int = translation_core.CANONICAL_ANALYSIS_CHAR_LIMIT
    include_source_text: bool = translation_core.CANONICAL_INCLUDE_SOURCE_TEXT
    include_translation_memory: bool = True
    story_block_suffix: str = translation_core.CANONICAL_STORY_BLOCK_SUFFIX
    total_char_limit: int = 0

    def to_dict(self):
        return {
            'local_context_before': int(self.local_context_before),
            'local_context_after': int(self.local_context_after),
            'history_char_limit': int(self.history_char_limit),
            'story_char_limit': int(self.story_char_limit),
            'source_index_char_limit': int(self.source_index_char_limit),
            'analysis_char_limit': int(self.analysis_char_limit),
            'include_source_text': bool(self.include_source_text),
            'include_translation_memory': bool(self.include_translation_memory),
            'story_block_suffix': self.story_block_suffix,
            'total_char_limit': int(self.total_char_limit),
        }


def build_chunk_id(file_rel_path, chunk_number):
    """Legacy batch chunk-key form: ``sha1(path)[:10]-NNNNN``."""
    return f"{translation_core.file_hash_key(file_rel_path)}-{int(chunk_number):05d}"


def build_request_id(plan_id, chunk_id, expected_ids):
    payload = {
        'plan_id': str(plan_id),
        'chunk_id': str(chunk_id),
        'expected_ids': [str(item) for item in expected_ids or []],
    }
    return short_fingerprint(canonical_json(payload))


# --- Local context window (D1: block-bounded, from issue #338) ----------------


def _task_text_len(item):
    text = (item or {}).get('text')
    return len(text) if isinstance(text, str) else 0


def build_local_context_window(tasks, start, end, before_limit, after_limit):
    """Build a file-bounded, block-bounded local context window (issue #338).

    The window never crosses translate-block boundaries when the first task of
    a side carries a different ``block_name``. Returns
    ``(translation_core.ContextWindow, diagnostics)``; the diagnostics record
    applied limits, item/character counts, block bounding, and whether the
    item budget truncated the window.
    """
    before = []
    after = []
    before_truncated = False
    after_truncated = False
    block_bounded_before = False
    block_bounded_after = False

    if before_limit > 0 and start > 0:
        batch_block = tasks[start].get('block_name')
        index = start - 1
        while index >= 0 and len(before) < before_limit:
            item = tasks[index]
            if item.get('block_name') != batch_block:
                block_bounded_before = True
                break
            before.append(item)
            index -= 1
        if index >= 0 and len(before) >= before_limit:
            before_truncated = True
        before.reverse()

    if after_limit > 0 and end < len(tasks):
        batch_block = tasks[end - 1].get('block_name')
        index = end
        while index < len(tasks) and len(after) < after_limit:
            item = tasks[index]
            if item.get('block_name') != batch_block:
                block_bounded_after = True
                break
            after.append(item)
            index += 1
        if index < len(tasks) and len(after) >= after_limit:
            after_truncated = True

    diagnostics = {
        'context_before_limit': before_limit,
        'context_after_limit': after_limit,
        'context_before_items': len(before),
        'context_after_items': len(after),
        'context_before_chars': sum(_task_text_len(item) for item in before),
        'context_after_chars': sum(_task_text_len(item) for item in after),
        'context_truncated': before_truncated or after_truncated,
        'block_bounded_before': block_bounded_before,
        'block_bounded_after': block_bounded_after,
    }
    return translation_core.ContextWindow(before, after), diagnostics


# --- Lexical glossary hits (D2: always injected, never RAG-gated) -------------


def retrieve_lexical_glossary_hits(target_items, normalize_map=None, preserve_terms=None, non_translatable_exact=None):
    """Return every lexical glossary hit for the current TARGET chunk.

    Matches ``normalize_map`` (existing mappings), ``preserve_terms``, and
    ``non_translatable_exact`` against the combined chunk text, deduplicated by
    source term and never truncated (issue #338): normalize hits must not evict
    non-translatable terms, otherwise names could be mistranslated. Unordered
    collections are canonicalized first so hit order is process-stable.
    """
    combined_text = '\n'.join(
        item.get('text', '') for item in target_items or [] if item.get('text')
    )
    if not combined_text:
        return []
    hits = []
    seen = set()
    # Sorted iteration: mappings with identical content but different
    # insertion order must yield identical hit order (and prompts).
    for source, target in sorted((normalize_map or {}).items()):
        if source and source in combined_text and source not in seen:
            hits.append({'source': source, 'target': target, 'kind': 'normalize'})
            seen.add(source)
    for term in _canonical_term_sequence(preserve_terms):
        if not isinstance(term, str) or not term.strip():
            continue
        if term in combined_text and term not in seen:
            hits.append({'source': term, 'target': term, 'kind': 'preserve'})
            seen.add(term)
    for term in _canonical_term_sequence(non_translatable_exact):
        if not isinstance(term, str) or not term.strip():
            continue
        if term in combined_text and term not in seen:
            hits.append({'source': term, 'target': '', 'kind': 'non_translatable'})
            seen.add(term)
    return hits


def render_lexical_glossary_text(hits):
    """Render lexical hits with the issue #338 wording (order-preserving)."""
    lines = []
    for hit in hits or []:
        source = str((hit or {}).get('source') or '')
        target = str((hit or {}).get('target') or '')
        kind = str((hit or {}).get('kind') or '')
        if not source:
            continue
        if kind == 'normalize':
            lines.append(f'- Existing mapping: {source} -> {target}')
        elif kind == 'preserve':
            lines.append(f'- Preserve: {source}')
        elif kind == 'non_translatable':
            lines.append(f'- Non-translatable: {source}')
        else:
            lines.append(f'- {source} -> {target}')
    return '\n'.join(lines)


# --- Context assembly (five content layers, deterministic order and budgets) --


@dataclass
class ChunkContextInput:
    """Deterministic per-chunk inputs the context layers account for."""

    file_rel_path: str = ''
    target_items: list = field(default_factory=list)
    target_units: list = field(default_factory=list)
    context_window: object = None
    local_context_diagnostics: dict = field(default_factory=dict)
    macro_setting: str = ''
    lexical_glossary_hits: list = field(default_factory=list)
    retrieval_blocks_text: str = ''
    analysis_blocks_text: str = ''
    retrieval_diagnostics: dict = field(default_factory=dict)
    analysis_diagnostics: dict = field(default_factory=dict)


@dataclass
class ContextLayerResult:
    layer: str
    rank: int
    text: str = ''
    char_used: int = 0
    char_limit: int = 0
    truncated: bool = False
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            'layer': self.layer,
            'rank': int(self.rank),
            'char_used': int(self.char_used),
            'char_limit': int(self.char_limit),
            'truncated': bool(self.truncated),
            'diagnostics': dict(self.diagnostics or {}),
        }


@dataclass
class ContextAssembly:
    """Per-request context accounting: layers, budget totals, drop log."""

    layers: list = field(default_factory=list)
    total_char_used: int = 0
    dropped: list = field(default_factory=list)

    def to_dict(self):
        return {
            'layers': [layer.to_dict() for layer in self.layers],
            'total_char_used': int(self.total_char_used),
            'dropped': [dict(item) for item in self.dropped],
        }


def _required_layer(chunk_input):
    target_payload = json.dumps(
        [translation_core.translation_target_payload_item(unit) for unit in chunk_input.target_units],
        ensure_ascii=False,
        separators=(',', ':'),
    )
    speakers = sorted({
        unit.speaker_id
        for unit in chunk_input.target_units
        if unit.speaker_id
    })
    return ContextLayerResult(
        layer=CONTEXT_LAYER_REQUIRED,
        rank=CONTEXT_LAYER_RANKS[CONTEXT_LAYER_REQUIRED],
        text=target_payload,
        char_used=len(target_payload),
        char_limit=0,
        diagnostics={
            'unit_count': len(chunk_input.target_units),
            'speakers': speakers,
            'file_rel_path': chunk_input.file_rel_path,
        },
    )


def _local_layer(chunk_input):
    window = chunk_input.context_window or translation_core.ContextWindow()
    before_text = translation_core.format_context_block(window.before, '(none)')
    after_text = translation_core.format_context_block(window.after, '(none)')
    text = f'CONTEXT BEFORE:\n{before_text}\n\nCONTEXT AFTER:\n{after_text}'
    diagnostics = dict(chunk_input.local_context_diagnostics or {})
    diagnostics['algorithm'] = 'block_bounded_window'
    return ContextLayerResult(
        layer=CONTEXT_LAYER_LOCAL,
        rank=CONTEXT_LAYER_RANKS[CONTEXT_LAYER_LOCAL],
        text=text,
        char_used=len(text),
        char_limit=0,
        truncated=bool(diagnostics.get('context_truncated')),
        diagnostics=diagnostics,
    )


def _project_layer(chunk_input):
    blocks = []
    macro = str(chunk_input.macro_setting or '').strip()
    if macro:
        blocks.append(f'Setting:\n{macro}')
    glossary_text = render_lexical_glossary_text(chunk_input.lexical_glossary_hits)
    if glossary_text:
        blocks.append(f'Existing glossary entries:\n{glossary_text}')
    text = '\n\n'.join(blocks)
    diagnostics = {
        'macro_injected': bool(macro),
        'lexical_glossary_hits': len(chunk_input.lexical_glossary_hits or []),
        'rag_independent': True,
    }
    return ContextLayerResult(
        layer=CONTEXT_LAYER_PROJECT,
        rank=CONTEXT_LAYER_RANKS[CONTEXT_LAYER_PROJECT],
        text=text,
        char_used=len(text),
        char_limit=0,
        diagnostics=diagnostics,
    )


def _retrieval_layer(chunk_input, policy):
    """Budget the retrieval layer as a D5 backstop, not a re-render.

    Providers normally render through :func:`render_reference_blocks`, which
    already applies the per-section history/story limits. Whatever text still
    exceeds the combined D5 envelope here is truncated deterministically,
    then trailing newlines are stripped so ``char_used`` counts exactly the
    bytes the canonical prompt embeds (layers join on a fixed blank-line
    separator).
    """
    text = str(chunk_input.retrieval_blocks_text or '')
    limit = (
        policy.history_char_limit
        + policy.story_char_limit
        + policy.source_index_char_limit
    )
    truncated = False
    if text and len(text) > limit:
        text = text[:limit]
        truncated = True
    text = text.rstrip('\n')
    diagnostics = {
        'budget_mode': 'd5_combined_backstop',
        'history_char_limit': policy.history_char_limit,
        'story_char_limit': policy.story_char_limit,
        'source_index_char_limit': policy.source_index_char_limit,
        'include_source_text': policy.include_source_text,
    }
    if chunk_input.retrieval_diagnostics:
        diagnostics['provider'] = redact_sensitive(
            dict(chunk_input.retrieval_diagnostics)
        )
    if truncated:
        diagnostics['char_discarded'] = max(0, len(chunk_input.retrieval_blocks_text or '') - len(text))
    return ContextLayerResult(
        layer=CONTEXT_LAYER_RETRIEVAL,
        rank=CONTEXT_LAYER_RANKS[CONTEXT_LAYER_RETRIEVAL],
        text=text,
        char_used=len(text),
        char_limit=limit,
        truncated=truncated,
        diagnostics=diagnostics,
    )


def _analysis_layer(chunk_input, policy):
    """Budget the analysis layer like the retrieval layer (D5 backstop).

    Published Project Analysis briefs have their own upstream budget (batch
    ``max_brief_chars``); whatever provider text still exceeds the policy
    limit here is truncated deterministically and trailing newlines are
    stripped, so no layer enters the prompt unbounded and ``char_used``
    counts the embedded bytes.
    """
    text = str(chunk_input.analysis_blocks_text or '')
    limit = policy.analysis_char_limit
    truncated = False
    if text and len(text) > limit:
        text = text[:limit]
        truncated = True
    text = text.rstrip('\n')
    diagnostics = {
        'source': 'published_project_analysis',
        'budget_mode': 'd5_backstop',
        'analysis_char_limit': limit,
    }
    if chunk_input.analysis_diagnostics:
        diagnostics['provider'] = redact_sensitive(
            dict(chunk_input.analysis_diagnostics)
        )
    if truncated:
        diagnostics['char_discarded'] = max(0, len(chunk_input.analysis_blocks_text or '') - len(text))
    return ContextLayerResult(
        layer=CONTEXT_LAYER_ANALYSIS,
        rank=CONTEXT_LAYER_RANKS[CONTEXT_LAYER_ANALYSIS],
        text=text,
        char_used=len(text),
        char_limit=limit,
        truncated=truncated,
        diagnostics=diagnostics,
    )


def assemble_context_layers(chunk_input, context_policy=None):
    """Assemble the five content layers in fixed rank order (issue #346).

    Identical layer texts are dropped deterministically (later rank loses) and
    recorded in the drop log; per-layer character budgets and truncation
    flags are accounted at the assembly level via ``total_char_used`` and
    ``dropped``.
    """
    policy = context_policy or ContextPolicy()
    results = [
        _required_layer(chunk_input),
        _local_layer(chunk_input),
        _project_layer(chunk_input),
        _retrieval_layer(chunk_input, policy),
        _analysis_layer(chunk_input, policy),
    ]
    layers = []
    dropped = []
    seen_texts = set()
    for result in sorted(results, key=lambda item: item.rank):
        # Retain diagnostic-only provider partitions so safe skips remain explainable.
        diagnostic_only_provider_layer = bool(
            not result.text
            and result.layer in {CONTEXT_LAYER_RETRIEVAL, CONTEXT_LAYER_ANALYSIS}
            and result.diagnostics.get('provider')
        )
        if result.text in seen_texts and not diagnostic_only_provider_layer:
            dropped.append({
                'layer': result.layer,
                'reason': 'duplicate_text',
                'char_used': result.char_used,
            })
            continue
        if not diagnostic_only_provider_layer:
            seen_texts.add(result.text)
        layers.append(result)
    total_limit = max(0, int(policy.total_char_limit or 0))
    if total_limit:
        mandatory_used = sum(
            layer.char_used
            for layer in layers
            if layer.layer in {
                CONTEXT_LAYER_REQUIRED,
                CONTEXT_LAYER_LOCAL,
                CONTEXT_LAYER_PROJECT,
            }
        )
        remaining = max(0, total_limit - mandatory_used)
        for layer in layers:
            if layer.layer not in {
                CONTEXT_LAYER_RETRIEVAL,
                CONTEXT_LAYER_ANALYSIS,
            }:
                continue
            original_used = layer.char_used
            if original_used <= remaining:
                remaining -= original_used
                continue
            layer.text = layer.text[:remaining]
            layer.char_used = len(layer.text)
            layer.truncated = True
            layer.diagnostics = {
                **dict(layer.diagnostics or {}),
                'aggregate_budget_reason': 'lower_priority_trimmed',
                'aggregate_char_limit': total_limit,
                'char_discarded': original_used - layer.char_used,
            }
            dropped.append({
                'layer': layer.layer,
                'rank': layer.rank,
                'reason': 'aggregate_budget_exceeded',
                'char_used': original_used - layer.char_used,
            })
            remaining = 0
        if mandatory_used > total_limit:
            dropped.append({
                'layer': 'budget',
                'rank': 0,
                'reason': 'mandatory_layers_exceed_budget',
                'char_used': mandatory_used - total_limit,
            })
    return ContextAssembly(
        layers=layers,
        total_char_used=sum(layer.char_used for layer in layers),
        dropped=dropped,
    )


# --- Plan schema v1 -----------------------------------------------------------


@dataclass
class SourceIdentity:
    engine: str = ''
    adapter_version: str = ''
    project_identity_digest: str = ''
    source_snapshot_fingerprint: str = ''
    file_digests: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            'engine': self.engine,
            'adapter_version': self.adapter_version,
            'project_identity_digest': self.project_identity_digest,
            'source_snapshot_fingerprint': self.source_snapshot_fingerprint,
            'file_digests': dict(self.file_digests or {}),
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        return cls(
            engine=str(payload.get('engine') or ''),
            adapter_version=str(payload.get('adapter_version') or ''),
            project_identity_digest=str(payload.get('project_identity_digest') or ''),
            source_snapshot_fingerprint=str(payload.get('source_snapshot_fingerprint') or ''),
            file_digests=dict(payload.get('file_digests') or {}),
        )


@dataclass
class PlanChunk:
    chunk_id: str
    chunk_index: int
    file_rel_path: str
    file_path: str
    line_numbers: list
    unit_ids: list
    source_char_count: int
    context_window_spec: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            'chunk_id': self.chunk_id,
            'chunk_index': int(self.chunk_index),
            'file_rel_path': self.file_rel_path,
            'file_path': self.file_path,
            'line_numbers': list(self.line_numbers),
            'unit_ids': list(self.unit_ids),
            'source_char_count': int(self.source_char_count),
            'context_window_spec': dict(self.context_window_spec or {}),
        }


def estimate_context_tokens(system_instruction, user_prompt):
    """Conservative token estimate; exact counting is a provider concern.

    The heuristic is a character-count upper bound (one token per character),
    which over-estimates Latin text and lands close to CJK reality. The method
    id is recorded in capability diagnostics so estimates stay attributable.
    """
    return len(str(system_instruction or '')) + len(str(user_prompt or ''))


def default_generation_config():
    """D6 baseline shared by both strategies; strategy-only extras are added
    by executors and land in ``request_fingerprint`` via transport metadata."""
    return {'temperature': CANONICAL_TEMPERATURE}


@dataclass
class TranslationRequest:
    request_id: str
    plan_id: str
    chunk_id: str
    system_instruction: str
    user_prompt: str
    response_schema: dict
    expected_ids: list
    capability_requirements: dict = field(default_factory=dict)
    generation_config: dict = field(default_factory=dict)
    transport_metadata: dict = field(default_factory=dict)
    context_assembly: dict = field(default_factory=dict)
    prompt_fingerprint: str = ''
    request_fingerprint: str = ''

    def semantic_payload(self):
        return {
            'system_instruction': self.system_instruction,
            'user_prompt': self.user_prompt,
            'response_schema': self.response_schema,
            'expected_ids': list(self.expected_ids),
            'context_assembly': dict(self.context_assembly or {}),
        }

    def audit_payload(self):
        return {
            'prompt_fingerprint': self.prompt_fingerprint,
            'generation_config': dict(self.generation_config or {}),
            'transport_metadata': dict(self.transport_metadata or {}),
        }

    def to_dict(self):
        return {
            'request_id': self.request_id,
            'plan_id': self.plan_id,
            'chunk_id': self.chunk_id,
            'system_instruction': self.system_instruction,
            'user_prompt': self.user_prompt,
            'response_schema': dict(self.response_schema or {}),
            'expected_ids': list(self.expected_ids),
            'capability_requirements': dict(self.capability_requirements or {}),
            'generation_config': dict(self.generation_config or {}),
            'transport_metadata': dict(self.transport_metadata or {}),
            'context_assembly': dict(self.context_assembly or {}),
            'prompt_fingerprint': self.prompt_fingerprint,
            'request_fingerprint': self.request_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        return cls(
            request_id=str(payload.get('request_id') or ''),
            plan_id=str(payload.get('plan_id') or ''),
            chunk_id=str(payload.get('chunk_id') or ''),
            system_instruction=str(payload.get('system_instruction') or ''),
            user_prompt=str(payload.get('user_prompt') or ''),
            response_schema=dict(payload.get('response_schema') or {}),
            expected_ids=list(payload.get('expected_ids') or []),
            capability_requirements=dict(payload.get('capability_requirements') or {}),
            generation_config=dict(payload.get('generation_config') or {}),
            transport_metadata=dict(payload.get('transport_metadata') or {}),
            context_assembly=dict(payload.get('context_assembly') or {}),
            prompt_fingerprint=str(payload.get('prompt_fingerprint') or ''),
            request_fingerprint=str(payload.get('request_fingerprint') or ''),
        )

    def summary(self):
        return {
            'request_id': self.request_id,
            'chunk_id': self.chunk_id,
            'prompt_fingerprint': self.prompt_fingerprint,
            'request_fingerprint': self.request_fingerprint,
            'capability_requirements': dict(self.capability_requirements or {}),
            'context_diagnostics': {
                'total_char_used': int(
                    (self.context_assembly or {}).get('total_char_used') or 0
                ),
                'layers': [
                    {
                        'layer': str((layer or {}).get('layer') or ''),
                        'rank': int((layer or {}).get('rank') or 0),
                        'char_used': int((layer or {}).get('char_used') or 0),
                        'char_limit': int((layer or {}).get('char_limit') or 0),
                        'truncated': bool((layer or {}).get('truncated')),
                        'diagnostics': dict((layer or {}).get('diagnostics') or {}),
                    }
                    for layer in (self.context_assembly or {}).get('layers') or []
                ],
                'dropped': [
                    dict(item)
                    for item in (self.context_assembly or {}).get('dropped') or []
                ],
            },
        }


@dataclass
class TranslationPlan:
    schema_version: int
    plan_id: str
    run_id: str
    source_identity: dict
    config_fingerprint: str
    model_profile_snapshot: dict
    execution_strategy: str
    chunk_policy: dict
    context_policy: dict
    chunks: list
    request_summaries: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    plan_fingerprint: str = ''

    def to_dict(self):
        return {
            'schema_version': int(self.schema_version),
            'plan_id': self.plan_id,
            'run_id': self.run_id,
            'source_identity': dict(self.source_identity or {}),
            'config_fingerprint': self.config_fingerprint,
            'model_profile_snapshot': dict(self.model_profile_snapshot or {}),
            'execution_strategy': self.execution_strategy,
            'chunk_policy': dict(self.chunk_policy or {}),
            'context_policy': dict(self.context_policy or {}),
            'chunks': [chunk.to_dict() if isinstance(chunk, PlanChunk) else dict(chunk) for chunk in self.chunks],
            'request_summaries': [dict(item) for item in self.request_summaries or []],
            'artifacts': dict(self.artifacts or {}),
            'plan_fingerprint': self.plan_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        return cls(
            schema_version=int(payload.get('schema_version') or 0),
            plan_id=str(payload.get('plan_id') or ''),
            run_id=str(payload.get('run_id') or ''),
            source_identity=dict(payload.get('source_identity') or {}),
            config_fingerprint=str(payload.get('config_fingerprint') or ''),
            model_profile_snapshot=dict(payload.get('model_profile_snapshot') or {}),
            execution_strategy=str(payload.get('execution_strategy') or ''),
            chunk_policy=dict(payload.get('chunk_policy') or {}),
            context_policy=dict(payload.get('context_policy') or {}),
            chunks=[PlanChunk(**chunk) if isinstance(chunk, dict) else chunk for chunk in payload.get('chunks') or []],
            request_summaries=[dict(item) for item in payload.get('request_summaries') or []],
            artifacts=dict(payload.get('artifacts') or {}),
            plan_fingerprint=str(payload.get('plan_fingerprint') or ''),
        )


@dataclass
class PlanBuild:
    plan: TranslationPlan
    requests: list


def _resolve_strategy(value):
    if isinstance(value, model_profile.ExecutionStrategy):
        return value.value
    strategy = str(value or '')
    if strategy not in EXECUTION_STRATEGIES:
        raise ValueError(
            f'unsupported execution strategy: {strategy!r} (expected one of {list(EXECUTION_STRATEGIES)})'
        )
    return strategy


def _resolve_profile_snapshot(value):
    if value is None:
        return {}
    if isinstance(value, model_profile.ModelProfile):
        payload = value.to_manifest_dict()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError('model_profile_snapshot must be a ModelProfile or a mapping')
    return redact_sensitive(payload)


def normalize_context_provider_text(value):
    """Normalize provider text exactly once before prompt/fingerprint use."""
    return str(value or '').replace('\r\n', '\n')


def _resolve_provider_source(value, chunk_input, label):
    """Accept a constant string or a per-chunk callable, LF-normalized.

    Provider text is normalized to LF line endings on entry so a CRLF
    checkout (or CRLF-producing provider) cannot move fingerprints: prompt
    text is byte-stable by contract, not by caller discipline.
    """
    if value is None:
        return '', {}
    if callable(value):
        resolved = value(chunk_input)
    elif isinstance(value, str):
        resolved = value
    else:
        raise TypeError(
            f'{label} must be a string or a callable(chunk_input) -> str or mapping'
        )
    diagnostics = {}
    if isinstance(resolved, Mapping):
        if 'text' not in resolved and 'diagnostics' not in resolved:
            raise TypeError(
                f'{label} mapping result must contain text and/or diagnostics'
            )
        text_value = resolved.get('text', '')
        raw_diagnostics = resolved.get('diagnostics') or {}
        if not isinstance(raw_diagnostics, Mapping):
            raise TypeError(f'{label}.diagnostics must be a mapping')
        diagnostics = redact_sensitive(dict(raw_diagnostics))
    else:
        text_value = resolved
    return normalize_context_provider_text(text_value), diagnostics


def _resolve_text_source(value, chunk_input, label):
    """Compatibility wrapper returning only provider text."""

    return _resolve_provider_source(value, chunk_input, label)[0]


def _unit_semantic_entry(unit):
    return {
        'id': unit.id,
        'text': unit.text,
        'speaker_id': unit.speaker_id,
        'speaker_name': unit.speaker_name,
        'file_rel_path': unit.file_rel_path,
        'line': unit.line,
        # block_name drives the D1 local-context window; it must participate
        # in plan identity or two jobs differing only in block layout would
        # share a plan_id while their prompts diverge.
        'block_name': str(unit.metadata.get('block_name', '') if isinstance(unit.metadata, Mapping) else ''),
    }


def render_reference_blocks(context_bundle, context_policy, *, include_translation_memory):
    """Render a reference section exactly as the D5 policy freezes it.

    Retrieval and analysis providers (P2/P3, then #341) call this so their
    pre-rendered text stays byte-identical to what the canonical prompt embeds.
    """
    return translation_core.build_reference_blocks(
        context_bundle,
        history_char_limit=context_policy.history_char_limit,
        story_char_limit=context_policy.story_char_limit,
        include_translation_memory=include_translation_memory,
        include_source_text=context_policy.include_source_text,
        story_block_suffix=context_policy.story_block_suffix,
    )


def build_translation_plan(
    file_jobs,
    *,
    execution_strategy,
    source_identity=None,
    config_snapshot=None,
    model_profile_snapshot=None,
    run_id='',
    artifacts=None,
    chunk_policy=None,
    context_policy=None,
    preserve_terms=None,
    normalize_map=None,
    non_translatable_exact=None,
    macro_setting='',
    retrieval_blocks_provider=None,
    analysis_blocks_provider=None,
    generation_config=None,
    transport_metadata=None,
):
    """Build the deterministic :class:`TranslationPlan` plus its requests.

    ``file_jobs`` mirrors the legacy batch shape: a list of
    ``{'file_rel_path', 'file_path', 'tasks': [...]}`` mappings. Retrieved
    context enters through ``retrieval_blocks_provider`` /
    ``analysis_blocks_provider`` — each may be a string, a
    ``callable(chunk_input) -> str``, or a callable returning
    ``{'text': str, 'diagnostics': mapping}`` of pre-rendered reference text
    and credential-free provider facts (P2/P3 adapt their stores to this
    seam; #341 attaches providers here). ``plan_id`` covers source identity,
    redacted config/profile snapshots, strategy, policies, and unit content —
    never retrieved content or ``run_id``; retrieved content and provider
    diagnostics are captured by each request's ``prompt_fingerprint`` and
    context diagnostics instead.
    """
    strategy = _resolve_strategy(execution_strategy)
    chunk_policy = chunk_policy or ChunkPolicy()
    context_policy = context_policy or ContextPolicy()
    identity = source_identity if isinstance(source_identity, SourceIdentity) else SourceIdentity.from_dict(source_identity)
    profile_snapshot = _resolve_profile_snapshot(model_profile_snapshot)
    config_fingerprint = short_fingerprint(
        canonical_json(redact_sensitive(dict(config_snapshot or {})))
    )
    preserve_terms = _canonical_term_sequence(preserve_terms)
    non_translatable_exact = _canonical_term_sequence(non_translatable_exact)

    # Pass 1: derive unit grouping and the plan identity payload.
    chunk_specs = []
    unit_entries = []
    for job in file_jobs or []:
        tasks = list((job or {}).get('tasks') or [])
        file_rel_path = str((job or {}).get('file_rel_path') or '')
        file_path = str((job or {}).get('file_path') or '')
        for chunk_index, (start, end) in enumerate(
            translation_core.iter_translation_chunk_ranges(
                tasks, max_items=chunk_policy.max_items, max_chars=chunk_policy.max_chars,
            ),
            start=1,
        ):
            target_items = tasks[start:end]
            target_units = translation_core.units_from_items(
                target_items,
                translation_core.MODE_TRANSLATION,
                file_rel_path=file_rel_path,
                file_path=file_path,
            )
            chunk_specs.append({
                'file_rel_path': file_rel_path,
                'file_path': file_path,
                'chunk_index': chunk_index,
                'target_items': target_items,
                'target_units': target_units,
                'start': start,
                'end': end,
                'tasks': tasks,
            })
            unit_entries.extend(_unit_semantic_entry(unit) for unit in target_units)

    identity_payload = {
        'schema_version': PLAN_SCHEMA_VERSION,
        'source_identity': identity.to_dict(),
        'config_fingerprint': config_fingerprint,
        'model_profile_snapshot': profile_snapshot,
        'execution_strategy': strategy,
        'chunk_policy': chunk_policy.to_dict(),
        'context_policy': context_policy.to_dict(),
        'prompt_inputs': {
            'preserve_terms': [str(term) for term in preserve_terms],
            'normalize_map': dict(normalize_map or {}),
            'non_translatable_exact': [str(term) for term in non_translatable_exact],
            'macro_setting': str(macro_setting or ''),
        },
        'units': unit_entries,
    }
    plan_id = short_fingerprint(canonical_json(identity_payload))

    # Pass 2: assemble context and requests per chunk.
    chunks = []
    requests = []
    request_summaries = []
    for spec in chunk_specs:
        file_rel_path = spec['file_rel_path']
        target_units = spec['target_units']
        target_items = spec['target_items']
        chunk_id = build_chunk_id(file_rel_path, spec['chunk_index'])
        expected_ids = [unit.id for unit in target_units]
        context_window, local_diagnostics = build_local_context_window(
            spec['tasks'],
            spec['start'],
            spec['end'],
            context_policy.local_context_before,
            context_policy.local_context_after,
        )
        lexical_hits = retrieve_lexical_glossary_hits(
            target_items,
            normalize_map=normalize_map,
            preserve_terms=preserve_terms,
            non_translatable_exact=non_translatable_exact,
        )
        chunk_input = ChunkContextInput(
            file_rel_path=file_rel_path,
            target_items=target_items,
            target_units=target_units,
            context_window=context_window,
            local_context_diagnostics=local_diagnostics,
            macro_setting=str(macro_setting or ''),
            lexical_glossary_hits=lexical_hits,
        )
        retrieval_text, retrieval_diagnostics = _resolve_provider_source(
            retrieval_blocks_provider,
            chunk_input,
            'retrieval_blocks_provider',
        )
        analysis_text, analysis_diagnostics = _resolve_provider_source(
            analysis_blocks_provider,
            chunk_input,
            'analysis_blocks_provider',
        )
        chunk_input.retrieval_blocks_text = retrieval_text
        chunk_input.analysis_blocks_text = analysis_text
        chunk_input.retrieval_diagnostics = retrieval_diagnostics
        chunk_input.analysis_diagnostics = analysis_diagnostics

        assembly = assemble_context_layers(chunk_input, context_policy)
        system_instruction = translation_core.build_canonical_translation_system_instruction(
            preserve_terms,
            macro_setting=str(macro_setting or ''),
        )
        # Retrieval/analysis layer texts are embedded verbatim (budgeted by
        # the assembly); required/local/project are canonical renderings the
        # translation_core builders own — user_prompt stays the authoritative
        # record of the model's input. Layers join on a fixed blank-line
        # separator after stripping trailing newlines, so a truncated
        # retrieval section can never glue onto the PROJECT BRIEF header.
        reference_blocks_text = '\n\n'.join(
            layer.text.rstrip('\n')
            for layer in assembly.layers
            if layer.layer in (CONTEXT_LAYER_RETRIEVAL, CONTEXT_LAYER_ANALYSIS)
            and layer.text
        )
        user_prompt = translation_core.build_canonical_translation_user_prompt(
            context_window,
            target_units,
            reference_blocks_text=reference_blocks_text,
            lexical_glossary_text=render_lexical_glossary_text(lexical_hits),
        )
        response_schema = translation_core.build_response_json_schema(
            target_units,
            mode=translation_core.MODE_TRANSLATION,
        )
        request_id = build_request_id(plan_id, chunk_id, expected_ids)
        # Credential-shaped values are redacted before they can enter the
        # request at all: serialized requests, logs, and fingerprints only
        # ever see the redaction marker.
        generation = redact_sensitive(
            dict(generation_config) if generation_config is not None else default_generation_config()
        )
        transport = redact_sensitive(dict(transport_metadata or {}))
        if strategy == STRATEGY_GEMINI_BATCH:
            transport.setdefault('batch_key', chunk_id)
        elif strategy == STRATEGY_SYNC:
            transport.setdefault('sync_stage', 'initial_translation')
        request = TranslationRequest(
            request_id=request_id,
            plan_id=plan_id,
            chunk_id=chunk_id,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            response_schema=response_schema,
            expected_ids=expected_ids,
            capability_requirements={
                'structured_output': True,
                'context_budget_tokens': estimate_context_tokens(system_instruction, user_prompt),
                'estimate_method': CONTEXT_TOKEN_ESTIMATE_METHOD,
            },
            generation_config=generation,
            transport_metadata=transport,
            context_assembly=assembly.to_dict(),
        )
        request.prompt_fingerprint = short_fingerprint(canonical_json(request.semantic_payload()))
        request.request_fingerprint = short_fingerprint(canonical_json(request.audit_payload()))
        requests.append(request)
        request_summaries.append(request.summary())
        chunks.append(PlanChunk(
            chunk_id=chunk_id,
            chunk_index=spec['chunk_index'],
            file_rel_path=file_rel_path,
            file_path=spec['file_path'],
            line_numbers=[unit.line for unit in target_units],
            unit_ids=expected_ids,
            source_char_count=sum(
                translation_core.translation_text_char_count(item) for item in target_items
            ),
            context_window_spec=dict(local_diagnostics),
        ))

    plan = TranslationPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        plan_id=plan_id,
        run_id=str(run_id or ''),
        source_identity=identity.to_dict(),
        config_fingerprint=config_fingerprint,
        model_profile_snapshot=profile_snapshot,
        execution_strategy=strategy,
        chunk_policy=chunk_policy.to_dict(),
        context_policy=context_policy.to_dict(),
        chunks=chunks,
        request_summaries=request_summaries,
        artifacts=dict(artifacts or {}),
    )
    fingerprint_payload = dict(plan.to_dict())
    fingerprint_payload.pop('run_id')
    fingerprint_payload.pop('plan_fingerprint')
    plan.plan_fingerprint = short_fingerprint(canonical_json(fingerprint_payload))
    return PlanBuild(plan=plan, requests=requests)


def derive_translation_request(
    parent_request,
    target_items,
    *,
    lineage_suffix,
    file_rel_path='',
    file_path='',
    context_window=None,
    local_context_diagnostics=None,
    context_policy=None,
    preserve_terms=None,
    normalize_map=None,
    non_translatable_exact=None,
    macro_setting='',
    retrieval_blocks_text='',
    analysis_blocks_text='',
    retrieval_diagnostics=None,
    analysis_diagnostics=None,
    lineage_kind='',
):
    """Create a deterministic retry request without changing its parent plan.

    D7 freezes the initial plan chunks.  A targeted retry or response-driven
    split therefore receives a child request whose identity is derived from
    the parent's request id plus a stable suffix (for example ``--L`` or
    ``--R``).  The child re-renders the canonical semantic contract for only
    its requested units while retaining the parent ``plan_id`` and recording
    explicit lineage in transport metadata.
    """
    if not isinstance(parent_request, TranslationRequest):
        raise TypeError('parent_request must be a TranslationRequest')
    suffix = str(lineage_suffix or '').strip()
    if not suffix or not suffix.startswith('--'):
        raise ValueError("lineage_suffix must start with '--'")
    items = list(target_items or [])
    units = translation_core.units_from_items(
        items,
        translation_core.MODE_TRANSLATION,
        file_rel_path=str(file_rel_path or ''),
        file_path=str(file_path or ''),
    )
    expected_ids = [unit.id for unit in units]
    policy = context_policy or ContextPolicy()
    lexical_hits = retrieve_lexical_glossary_hits(
        items,
        normalize_map=normalize_map,
        preserve_terms=preserve_terms,
        non_translatable_exact=non_translatable_exact,
    )

    def inherited_provider_diagnostics(layer_name):
        for layer in (parent_request.context_assembly or {}).get('layers') or []:
            if (layer or {}).get('layer') != layer_name:
                continue
            diagnostics = (layer or {}).get('diagnostics') or {}
            provider = diagnostics.get('provider')
            return dict(provider) if isinstance(provider, Mapping) else {}
        return {}

    if retrieval_diagnostics is None:
        retrieval_diagnostics = inherited_provider_diagnostics(
            CONTEXT_LAYER_RETRIEVAL
        )
    if analysis_diagnostics is None:
        analysis_diagnostics = inherited_provider_diagnostics(
            CONTEXT_LAYER_ANALYSIS
        )
    chunk_input = ChunkContextInput(
        file_rel_path=str(file_rel_path or ''),
        target_items=items,
        target_units=units,
        context_window=context_window or translation_core.ContextWindow(),
        local_context_diagnostics=dict(local_context_diagnostics or {}),
        macro_setting=str(macro_setting or ''),
        lexical_glossary_hits=lexical_hits,
        retrieval_blocks_text=normalize_context_provider_text(
            retrieval_blocks_text
        ),
        analysis_blocks_text=normalize_context_provider_text(
            analysis_blocks_text
        ),
        retrieval_diagnostics=dict(retrieval_diagnostics or {}),
        analysis_diagnostics=dict(analysis_diagnostics or {}),
    )
    assembly = assemble_context_layers(chunk_input, policy)
    reference_blocks_text = '\n\n'.join(
        layer.text.rstrip('\n')
        for layer in assembly.layers
        if layer.layer in (CONTEXT_LAYER_RETRIEVAL, CONTEXT_LAYER_ANALYSIS)
        and layer.text
    )
    user_prompt = translation_core.build_canonical_translation_user_prompt(
        chunk_input.context_window,
        units,
        reference_blocks_text=reference_blocks_text,
        lexical_glossary_text=render_lexical_glossary_text(lexical_hits),
    )
    response_schema = translation_core.build_response_json_schema(
        units,
        mode=translation_core.MODE_TRANSLATION,
    )
    request_id = f'{parent_request.request_id}{suffix}'
    chunk_id = f'{parent_request.chunk_id}{suffix}'
    transport = dict(parent_request.transport_metadata or {})
    transport.update({
        'retry_parent_request_id': parent_request.request_id,
        'retry_parent_chunk_id': parent_request.chunk_id,
        'retry_lineage_kind': str(lineage_kind or 'derived'),
        'retry_item_ids': expected_ids,
    })
    capability_requirements = dict(
        parent_request.capability_requirements or {}
    )
    capability_requirements.setdefault('structured_output', True)
    capability_requirements.update({
        'context_budget_tokens': estimate_context_tokens(
            parent_request.system_instruction,
            user_prompt,
        ),
        'estimate_method': CONTEXT_TOKEN_ESTIMATE_METHOD,
    })
    request = TranslationRequest(
        request_id=request_id,
        plan_id=parent_request.plan_id,
        chunk_id=chunk_id,
        system_instruction=parent_request.system_instruction,
        user_prompt=user_prompt,
        response_schema=response_schema,
        expected_ids=expected_ids,
        capability_requirements=capability_requirements,
        generation_config=dict(parent_request.generation_config or {}),
        transport_metadata=redact_sensitive(transport),
        context_assembly=assembly.to_dict(),
    )
    request.prompt_fingerprint = short_fingerprint(
        canonical_json(request.semantic_payload())
    )
    request.request_fingerprint = short_fingerprint(
        canonical_json(request.audit_payload())
    )
    return request
