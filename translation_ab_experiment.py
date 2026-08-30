# -*- coding: utf-8 -*-
"""Run synchronous translation A/B experiments without writing game files."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import model_usage_ledger
import story_memory
import translator_runtime as legacy
from sync_model_backend import sync_error_category, sync_error_summary

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(_TOOL_DIR, 'logs', 'experiments')


def _batch():
    import gemini_translate_batch as batch_mod

    return batch_mod


def deep_merge_dict(base: object, override: object) -> object:
    if not isinstance(override, dict):
        return copy.deepcopy(override)
    if not isinstance(base, dict):
        base = {}
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_variants_file(path: str) -> list[dict]:
    if not path or not os.path.isfile(path):
        raise SystemExit(f'Variants file not found: {path}')
    with open(path, 'r', encoding='utf-8-sig') as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        variants = payload.get('variants')
        if isinstance(variants, list):
            payload = variants
        else:
            raise SystemExit('Variants JSON must be a list or an object with a "variants" array.')
    if not isinstance(payload, list) or len(payload) < 2:
        raise SystemExit('Variants file must contain at least two variants.')
    normalized = []
    for index, entry in enumerate(payload, start=1):
        if not isinstance(entry, dict):
            raise SystemExit(f'Variant #{index} must be an object.')
        name = _compact_text(entry.get('name') or entry.get('id') or f'variant_{index}')
        if not name:
            raise SystemExit(f'Variant #{index} is missing a name.')
        overrides = entry.get('overrides')
        if overrides is None:
            overrides = {key: value for key, value in entry.items() if key not in {'name', 'id'}}
        if not isinstance(overrides, dict):
            raise SystemExit(f'Variant "{name}" overrides must be an object.')
        normalized.append({'name': name, 'overrides': overrides})
    names = [entry['name'] for entry in normalized]
    if len(set(names)) != len(names):
        raise SystemExit('Variant names must be unique.')
    return normalized


def select_manifest_chunks(manifest: dict, *, limit: int = 3, offset: int = 0) -> list[dict]:
    chunks = manifest.get('chunks')
    if not isinstance(chunks, list) or not chunks:
        raise SystemExit('Manifest does not contain translation chunks.')
    if offset < 0:
        offset = 0
    if limit <= 0:
        raise SystemExit('--limit must be greater than 0.')
    selected = chunks[offset:offset + limit]
    if not selected:
        raise SystemExit('No chunks available for the requested limit/offset.')
    return [copy.deepcopy(chunk) for chunk in selected]


def _compact_text(value: object) -> str:
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def _macro_preview(text: object, limit: int = 120) -> str:
    compact = _compact_text(text).replace('\n', ' ')
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + '...'


def summarize_variant_settings() -> dict:
    batch_mod = _batch()
    return {
        'model': batch_mod.BATCH_MODEL,
        'story_memory_enabled': batch_mod.STORY_MEMORY_ENABLED,
        'story_memory_graph_file': batch_mod.STORY_MEMORY_GRAPH_FILE if batch_mod.STORY_MEMORY_ENABLED else '',
        'rag_enabled': batch_mod.RAG_ENABLED,
        'source_index_enabled': batch_mod.SOURCE_INDEX_ENABLED,
        'macro_setting_preview': _macro_preview(batch_mod.BATCH_MACRO_SETTING),
        'temperature': batch_mod.BATCH_TEMPERATURE,
        'max_output_tokens': batch_mod.BATCH_MAX_OUTPUT_TOKENS,
        'timeout_seconds': batch_mod.SYNC_TIMEOUT_SECONDS,
        'thinking_level': batch_mod.BATCH_THINKING_LEVEL,
    }


_BATCH_GLOBAL_KEYS = (
    'BATCH_MODEL',
    'BATCH_TARGET_SIZE',
    'BATCH_CONTEXT_BEFORE',
    'BATCH_CONTEXT_AFTER',
    'BATCH_TARGET_CHARS',
    'BATCH_RETRY_TARGET_SIZE',
    'BATCH_RETRY_TARGET_CHARS',
    'BATCH_MAX_OUTPUT_TOKENS',
    'BATCH_TEMPERATURE',
    'BATCH_THINKING_LEVEL',
    'BATCH_SAFETY_SETTINGS',
    'BATCH_DISPLAY_NAME_PREFIX',
    'BATCH_MACRO_SETTING',
    'KEYWORD_DISPLAY_NAME_PREFIX',
    'KEYWORD_CHUNK_SIZE',
    'KEYWORD_MAX_CANDIDATES_PER_CHUNK',
    'REVISION_DISPLAY_NAME_PREFIX',
    'REVISION_CHUNK_SIZE',
    'RAG_ENABLED',
    'RAG_STORE_DIR',
    'RAG_EMBEDDING_MODEL',
    'RAG_QUERY_TASK_TYPE',
    'RAG_DOCUMENT_TASK_TYPE',
    'RAG_OUTPUT_DIMENSIONALITY',
    'RAG_TOP_K_HISTORY',
    'RAG_TOP_K_TERMS',
    'RAG_MIN_SIMILARITY',
    'RAG_SEGMENT_LINES',
    'RAG_BOOTSTRAP_ON_BUILD',
    'RAG_HISTORY_CHAR_LIMIT',
    'SOURCE_INDEX_ENABLED',
    'SOURCE_INDEX_STORE_DIR',
    'SOURCE_INDEX_TOP_K',
    'SOURCE_INDEX_MIN_SIMILARITY',
    'SOURCE_INDEX_CHAR_LIMIT',
    'STORY_MEMORY_ENABLED',
    'STORY_MEMORY_GRAPH_FILE',
    'STORY_MEMORY_MAX_CONTEXT_CHARS',
    'STORY_MEMORY_TOP_K_RELATIONS',
    'STORY_MEMORY_TOP_K_TERMS',
    'STORY_MEMORY_INCLUDE_SCENE_SUMMARY',
    'BATCH_NON_CHINESE_RULES',
    'SYNC_TIMEOUT_SECONDS',
    '_RAG_STORE',
    '_SOURCE_INDEX_STORE',
    '_STORY_GRAPH',
    '_STORY_GRAPH_PATH',
)


def _snapshot_batch_globals() -> dict:
    batch_mod = _batch()
    snapshot = {key: getattr(batch_mod, key) for key in _BATCH_GLOBAL_KEYS}
    snapshot['load_json_file'] = batch_mod.load_json_file
    return snapshot


def _restore_batch_globals(snapshot: dict) -> None:
    batch_mod = _batch()
    load_json_file = snapshot.pop('load_json_file')
    for key, value in snapshot.items():
        setattr(batch_mod, key, copy.deepcopy(value))
    batch_mod.load_json_file = load_json_file


def _reapply_explicit_context_overrides(overrides: dict) -> None:
    """Re-apply variant context flags after load_batch_settings.

    load_batch_settings() always folds project_context_settings.json into
    batch.rag / batch.source_index, which would otherwise clobber explicit
    experiment overrides. Keep BASE_DIR intact so macro paths and store dirs
    still resolve against the active project.
    """
    if not isinstance(overrides, dict):
        return
    batch = overrides.get('batch')
    if not isinstance(batch, dict):
        return
    batch_mod = _batch()
    rag = batch.get('rag')
    if isinstance(rag, dict):
        if 'enabled' in rag:
            batch_mod.RAG_ENABLED = bool(rag['enabled'])
        if 'bootstrap_on_build' in rag:
            batch_mod.RAG_BOOTSTRAP_ON_BUILD = bool(rag['bootstrap_on_build'])
    source_index = batch.get('source_index')
    if isinstance(source_index, dict) and 'enabled' in source_index:
        batch_mod.SOURCE_INDEX_ENABLED = bool(source_index['enabled'])
    story_memory = batch.get('story_memory')
    if isinstance(story_memory, dict) and 'enabled' in story_memory:
        batch_mod.STORY_MEMORY_ENABLED = bool(story_memory['enabled'])


@contextmanager
def variant_batch_settings(overrides: dict):
    batch_mod = _batch()
    baseline = _snapshot_batch_globals()
    base_translator = baseline['load_json_file'](legacy.TRANSLATOR_CONFIG)
    base_legacy = baseline['load_json_file'](legacy.CONFIG_FILE)
    merged_translator = deep_merge_dict(base_translator, overrides)
    original_load = baseline['load_json_file']

    def patched_load(path):
        if path == legacy.TRANSLATOR_CONFIG:
            return copy.deepcopy(merged_translator)
        if path == legacy.CONFIG_FILE:
            return copy.deepcopy(base_legacy)
        return original_load(path)

    batch_mod.load_json_file = patched_load
    batch_mod._STORY_GRAPH = None
    batch_mod._STORY_GRAPH_PATH = ''
    batch_mod._RAG_STORE = None
    batch_mod._SOURCE_INDEX_STORE = None
    try:
        # Preserve legacy.BASE_DIR so project-scoped paths (macro_setting_file,
        # RAG/source-index store dirs) resolve against the active game root.
        batch_mod.load_batch_settings()
        _reapply_explicit_context_overrides(overrides)
        yield summarize_variant_settings()
    finally:
        _restore_batch_globals(baseline)


def enrich_chunk_for_current_settings(chunk: dict, *, dry_run: bool = False) -> dict:
    batch_mod = _batch()
    enriched = copy.deepcopy(chunk)
    target_items = enriched.get('items') or []
    context_past = enriched.get('context_past') or []
    context_future = enriched.get('context_future') or []
    file_rel_path = enriched.get('file_rel_path') or ''

    if dry_run:
        enriched['glossary_hits'] = []
        enriched['history_hits'] = []
        enriched['source_hits'] = []
        enriched.pop('rag_stats', None)
        enriched.pop('source_index_stats', None)
        enriched.pop('story_hits', None)
        return enriched

    if batch_mod.RAG_ENABLED:
        enriched['glossary_hits'] = batch_mod.retrieve_glossary_hits(target_items)
        enriched['history_hits'], enriched['rag_stats'] = batch_mod.retrieve_history_hits(
            target_items,
            context_past,
        )
    else:
        enriched['glossary_hits'] = []
        enriched['history_hits'] = []
        enriched.pop('rag_stats', None)

    if batch_mod.SOURCE_INDEX_ENABLED:
        enriched['source_hits'], enriched['source_index_stats'] = batch_mod.retrieve_source_hits(
            target_items,
            context_past,
        )
    else:
        enriched['source_hits'] = []
        enriched.pop('source_index_stats', None)

    enriched.pop('story_hits', None)
    if batch_mod.STORY_MEMORY_ENABLED:
        story_hits = batch_mod.retrieve_batch_story_hits(
            file_rel_path,
            target_items,
            context_past,
            context_future,
        )
        if story_memory.has_story_hits(story_hits):
            enriched['story_hits'] = story_hits

    return enriched


def extract_translation_map(response_text: str, items: list[dict]) -> tuple[dict[str, str], str]:
    if not response_text:
        return {}, 'Missing response text.'
    try:
        batch_mod = _batch()
        payload = batch_mod.parse_json_payload(response_text)
        result_items = batch_mod.normalize_result_items(payload)
    except Exception as exc:
        return {}, str(exc)

    expected_ids = [str(item.get('id') or '') for item in items if item.get('id')]
    if not expected_ids:
        return {}, 'Chunk has no item ids to validate against.'

    expected_id_set = set(expected_ids)
    translations: dict[str, str] = {}
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for row in result_items:
        item_id = str(row.get('id') or '')
        translation = _compact_text(row.get('translation'))
        if not item_id:
            continue
        if item_id in seen_ids:
            duplicate_ids.append(item_id)
        seen_ids.add(item_id)
        if translation:
            translations[item_id] = translation

    errors: list[str] = []
    if duplicate_ids:
        errors.append(
            'Duplicate result ids: ' + ', '.join(sorted(set(duplicate_ids))),
        )
    missing_ids = [item_id for item_id in expected_ids if item_id not in translations]
    if missing_ids:
        errors.append('Missing translations for ids: ' + ', '.join(missing_ids))
    extra_ids = [item_id for item_id in translations if item_id not in expected_id_set]
    if extra_ids:
        errors.append('Unexpected result ids: ' + ', '.join(extra_ids))

    if not translations and not errors:
        return {}, 'No translation items parsed from response.'
    if errors:
        return translations, '; '.join(errors)
    return translations, ''


@dataclass
class VariantRunResult:
    """Store one variant result and its safe synchronization metadata.

    ``output_diagnostics`` holds provider-neutral response-budget facts
    normalized via ``model_usage_ledger.normalize_response_diagnostics`` and is
    empty when no response was produced. ``request_metadata`` holds normalized
    non-secret routing metadata (provider, masked credential identity) and is
    empty when unavailable. ``error_category`` is set only for categorized
    request or response failures and is empty on success; the JSONL result
    contract persists these fields with the same empty-value semantics.
    """

    variant_name: str
    settings: dict
    translations: dict[str, str] = field(default_factory=dict)
    usage_metadata: dict = field(default_factory=dict)
    output_diagnostics: dict = field(default_factory=dict)
    request_metadata: dict = field(default_factory=dict)
    finish_reason: str = ''
    error_category: str = ''
    error: str = ''
    dry_run: bool = False


@dataclass
class ChunkExperimentResult:
    chunk_key: str
    file_rel_path: str
    items: list[dict]
    variant_results: list[VariantRunResult] = field(default_factory=list)


def _ensure_experiments_dir() -> str:
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    return EXPERIMENTS_DIR


def create_experiment_output_dir(prefix: str = 'ab') -> str:
    _ensure_experiments_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    base = os.path.join(EXPERIMENTS_DIR, f'{timestamp}_{prefix}')
    suffix = 0
    candidate = base
    while os.path.exists(candidate):
        suffix += 1
        candidate = f'{base}_{suffix:02d}'
    os.makedirs(candidate, exist_ok=False)
    return candidate


def render_markdown_report(
    *,
    manifest_path: str,
    variants: list[dict],
    chunk_results: list[ChunkExperimentResult],
    experiment_settings: dict,
) -> str:
    variant_names = [entry['name'] for entry in variants]
    settings_by_variant: dict[str, dict] = {}
    for chunk in chunk_results:
        for result in chunk.variant_results:
            settings_by_variant.setdefault(result.variant_name, result.settings)

    lines = [
        '# Translation A/B Experiment',
        '',
        f'- Manifest: `{manifest_path}`',
        f'- Generated: {datetime.now().isoformat(timespec="seconds")}',
        f'- Chunks: {len(chunk_results)}',
        f'- Variants: {len(variants)}',
        '',
        '## Variant Settings',
        '',
        '| Variant | Model | Story Memory | RAG | Source Index | Macro (preview) |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for name in variant_names:
        settings = settings_by_variant.get(name, {})
        lines.append(
            '| {name} | {model} | {story} | {rag} | {source_index} | {macro} |'.format(
                name=_escape_table_cell(name),
                model=_escape_table_cell(settings.get('model', '')),
                story='yes' if settings.get('story_memory_enabled') else 'no',
                rag='yes' if settings.get('rag_enabled') else 'no',
                source_index='yes' if settings.get('source_index_enabled') else 'no',
                macro=_escape_table_cell(settings.get('macro_setting_preview', '')),
            )
        )
    lines.extend(['', '## Samples', ''])

    for sample_index, chunk in enumerate(chunk_results, start=1):
        lines.append(f'### Sample {sample_index}: `{chunk.chunk_key}` ({chunk.file_rel_path})')
        lines.append('')
        escaped_variant_names = [_escape_table_cell(name) for name in variant_names]
        lines.append('| Item ID | Source | ' + ' | '.join(escaped_variant_names) + ' |')
        lines.append('| --- | --- | ' + ' | '.join(['---'] * len(variant_names)) + ' |')
        translation_maps = {
            result.variant_name: result.translations for result in chunk.variant_results
        }
        for item in chunk.items:
            item_id = str(item.get('id') or '')
            source_text = _escape_table_cell(item.get('text') or '')
            row_cells = [
                _escape_table_cell(translation_maps.get(name, {}).get(item_id, ''))
                for name in variant_names
            ]
            lines.append(
                '| {id} | {source} | {variants} |'.format(
                    id=_escape_table_cell(item_id),
                    source=source_text,
                    variants=' | '.join(row_cells),
                )
            )
        lines.append('')
        lines.append('#### Metadata')
        for result in chunk.variant_results:
            meta_bits = [
                f'finish_reason={result.finish_reason or "(none)"}',
                f'usage={json.dumps(result.usage_metadata, ensure_ascii=False)}',
            ]
            if result.output_diagnostics:
                meta_bits.append(
                    'output_diagnostics='
                    + json.dumps(result.output_diagnostics, ensure_ascii=False)
                )
            if result.error_category:
                meta_bits.append(f'error_category={result.error_category}')
            if result.error:
                meta_bits.append(f'error={result.error}')
            if result.dry_run:
                meta_bits.append('dry_run=true')
            lines.append(
                f'- **{_escape_table_cell(result.variant_name)}**: ' + '; '.join(meta_bits),
            )
        lines.append('')

    if experiment_settings:
        lines.extend(['## Run Settings', '', '```json', json.dumps(experiment_settings, ensure_ascii=False, indent=2), '```', ''])
    return '\n'.join(lines).rstrip() + '\n'


def _escape_table_cell(value: object) -> str:
    text = _compact_text(value).replace('|', '\\|')
    return text.replace('\n', '<br>')


def write_experiment_outputs(
    output_dir: str,
    *,
    manifest_path: str,
    variants: list[dict],
    chunk_results: list[ChunkExperimentResult],
    experiment_settings: dict,
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'ab_report.md')
    results_path = os.path.join(output_dir, 'ab_results.jsonl')
    settings_path = os.path.join(output_dir, 'ab_settings.json')

    report_text = render_markdown_report(
        manifest_path=manifest_path,
        variants=variants,
        chunk_results=chunk_results,
        experiment_settings=experiment_settings,
    )
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write(report_text)

    with open(results_path, 'w', encoding='utf-8') as handle:
        for chunk in chunk_results:
            handle.write(
                json.dumps(
                    {
                        'chunk_key': chunk.chunk_key,
                        'file_rel_path': chunk.file_rel_path,
                        'items': chunk.items,
                        'variants': [
                            {
                                'name': result.variant_name,
                                'settings': result.settings,
                                'translations': result.translations,
                                'usage_metadata': result.usage_metadata,
                                'output_diagnostics': result.output_diagnostics,
                                'request_metadata': result.request_metadata,
                                'finish_reason': result.finish_reason,
                                'error_category': result.error_category,
                                'error': result.error,
                                'dry_run': result.dry_run,
                            }
                            for result in chunk.variant_results
                        ],
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )

    with open(settings_path, 'w', encoding='utf-8') as handle:
        json.dump(experiment_settings, handle, ensure_ascii=False, indent=2)

    return {
        'report_path': report_path,
        'results_path': results_path,
        'settings_path': settings_path,
    }


def _sync_runner_uses_task_route(sync_runner: Callable[..., object]) -> bool:
    """Return whether ``sync_runner`` accepts the TaskRoute calling convention.

    ``True`` means call with ``(payload, route, plan=..., api_key_index=...)``.
    ``False`` means the pre-#345 ``(payload, model_name, api_key_index=...)``
    convention. An uninspectable signature is treated as the current
    TaskRoute convention; never guess from a later TypeError message.
    """
    try:
        signature = inspect.signature(sync_runner)
    except (TypeError, ValueError):
        return True
    parameters = signature.parameters
    if 'route' in parameters or 'plan' in parameters:
        return True
    if 'model_name' in parameters:
        return False
    return True


def _invoke_ab_sync_runner(
    sync_runner: Callable[..., dict],
    request_payload: dict,
    *,
    route,
    plan,
    model_name: str,
    api_key_index: int | None,
) -> dict:
    """Call a sync runner using only the inspected calling convention."""
    if _sync_runner_uses_task_route(sync_runner):
        return sync_runner(
            request_payload,
            route,
            plan=plan,
            api_key_index=api_key_index,
        )
    return sync_runner(
        request_payload,
        model_name,
        api_key_index=api_key_index,
    )


def _legacy_ab_routing_plan(model_name: str):
    """Resolve old manifests without leaking the live sync backend into them."""
    batch_mod = _batch()
    model = str(model_name or '').strip()
    sync_backend = (
        batch_mod.model_profile.SYNC_BACKEND_LITELLM
        if '/' in model
        else batch_mod.model_profile.SYNC_BACKEND_GEMINI
    )
    plan = batch_mod.model_profile.resolve_routing_plan_from_runtime(
        sync_backend=sync_backend,
        sync_model=model,
        batch_model=str(batch_mod.BATCH_MODEL or model),
        custom_providers=batch_mod._runtime_custom_providers(),
        execution=batch_mod.model_profile.ExecutionStrategy.SYNC,
        stage_overrides={
            batch_mod.model_profile.STAGE_AB_EXPERIMENT: model,
        } if model else None,
    )
    batch_mod.require_valid_routing_plan(
        plan,
        {batch_mod.model_profile.STAGE_AB_EXPERIMENT},
    )
    return plan


def _has_frozen_sync_ab_route(plan) -> bool:
    """Return whether a manifest plan froze a usable synchronous A/B route."""
    if plan is None:
        return False
    batch_mod = _batch()
    route = plan.routes.get(batch_mod.model_profile.STAGE_AB_EXPERIMENT)
    return bool(
        route is not None
        and route.strategy is batch_mod.model_profile.ExecutionStrategy.SYNC
    )


def run_variant_for_chunk(
    chunk: dict,
    *,
    variant_name: str,
    settings: dict,
    model_override: str = '',
    api_key_index: int | None = None,
    dry_run: bool = False,
    sync_runner: Callable[..., dict] | None = None,
    usage_context: dict | None = None,
    route=None,
    plan=None,
) -> VariantRunResult:
    batch_mod = _batch()
    if plan is None or route is None:
        explicit = (
            model_override.strip()
            or str(settings.get('model') or '').strip()
            or str(batch_mod.BATCH_MODEL or '').strip()
        )
        plan = _legacy_ab_routing_plan(explicit)
        route = plan.routes[batch_mod.model_profile.STAGE_AB_EXPERIMENT]
    model_name = batch_mod.route_model(plan, route)
    try:
        enriched = enrich_chunk_for_current_settings(chunk, dry_run=dry_run)
        request_row = batch_mod.build_batch_request(enriched, model=model_name)
        request_payload = request_row.get('request') or {}
        if dry_run:
            return VariantRunResult(
                variant_name=variant_name,
                settings=settings,
                dry_run=True,
            )

        if sync_runner is None:
            response = batch_mod.run_sync_request(
                request_payload,
                route,
                plan=plan,
                api_key_index=api_key_index,
            )
        else:
            response = _invoke_ab_sync_runner(
                sync_runner,
                request_payload,
                route=route,
                plan=plan,
                model_name=model_name,
                api_key_index=api_key_index,
            )
        output_diagnostics = _batch().sync_output_diagnostics(
            response,
            request_payload,
        )
        request_metadata = model_usage_ledger.normalize_request_metadata(
            response.get('request_metadata') or {}
        )
        game_root = str((usage_context or {}).get('game_root') or legacy.BASE_DIR or '')
        if usage_context and game_root:
            try:
                model_usage_ledger.record_generation_usage(
                    game_root=game_root,
                    task_mode='analysis',
                    stage='compare_variants',
                    provider=str(response.get('provider') or _batch().SYNC_BACKEND or 'unknown'),
                    model=str(response.get('model') or model_name),
                    usage_metadata=response.get('usage_metadata') or {},
                    response_payload=response.get('response_payload') or {},
                    operation_id=str(usage_context.get('operation_id') or ''),
                    run_id=str(usage_context.get('run_id') or ''),
                    thinking_level=str(settings.get('thinking_level') or ''),
                    execution_mode=str(response.get('execution_mode') or 'sync'),
                    source_key=f"{chunk.get('key') or ''}:{variant_name}",
                    source={
                        'kind': 'translation_ab_response',
                        'manifest_path': str(usage_context.get('manifest_path') or ''),
                        'output_dir': str(usage_context.get('output_dir') or ''),
                        'chunk_key': str(chunk.get('key') or ''),
                        'variant': variant_name,
                    },
                    response_diagnostics=output_diagnostics,
                    request_metadata=request_metadata,
                )
            except Exception as exc:
                # Accounting must not turn a successful variant into a failure.
                usage_context.setdefault('errors', []).append(str(exc))
        translations, parse_error = extract_translation_map(
            response.get('response_text') or '',
            enriched.get('items') or [],
        )
        return VariantRunResult(
            variant_name=variant_name,
            settings=settings,
            translations=translations,
            usage_metadata=response.get('usage_metadata') or {},
            output_diagnostics=output_diagnostics,
            request_metadata=request_metadata,
            finish_reason=response.get('finish_reason') or '',
            error=parse_error,
            error_category='invalid_response' if parse_error else '',
        )
    except Exception as exc:
        return VariantRunResult(
            variant_name=variant_name,
            settings=settings,
            error=sync_error_summary(exc),
            error_category=sync_error_category(exc),
            request_metadata=model_usage_ledger.normalize_request_metadata(
                getattr(exc, 'request_metadata', None) or {}
            ),
        )


def run_translation_ab_experiment(
    manifest: dict,
    variants: list[dict],
    *,
    limit: int = 3,
    offset: int = 0,
    output_dir: str = '',
    model_override: str = '',
    api_key_index: int | None = None,
    dry_run: bool = False,
    sync_runner: Callable[..., dict] | None = None,
) -> dict:
    batch_mod = _batch()
    batch_mod.require_manifest_mode(manifest, batch_mod.MANIFEST_MODE_TRANSLATION, 'compare-variants')
    chunks = select_manifest_chunks(manifest, limit=limit, offset=offset)
    explicit_model = model_override.strip()
    routing_plan = batch_mod.routing_plan_from_manifest(manifest)
    if not _has_frozen_sync_ab_route(routing_plan):
        # Legacy or partial manifests did not freeze a synchronous A/B route.
        # Preserve their historical fallback to the recorded Batch model,
        # then the loaded Batch default.
        fallback_model = (
            explicit_model
            or str(manifest.get('batch_model') or '').strip()
            or str(batch_mod.BATCH_MODEL or '').strip()
        )
        routing_plan = _legacy_ab_routing_plan(fallback_model)
    else:
        if explicit_model:
            routing_plan = batch_mod.model_profile.override_sync_stage(
                routing_plan,
                batch_mod.model_profile.STAGE_AB_EXPERIMENT,
                explicit_model,
                custom_providers=batch_mod._runtime_custom_providers(),
            )
        batch_mod.require_valid_routing_plan(
            routing_plan,
            {batch_mod.model_profile.STAGE_AB_EXPERIMENT},
        )
    ab_route = routing_plan.routes[batch_mod.model_profile.STAGE_AB_EXPERIMENT]
    default_model = batch_mod.route_model(routing_plan, ab_route)
    ab_profile = batch_mod.model_profile.profile_for_route(routing_plan, ab_route)

    if not output_dir:
        slug = batch_mod.guess_project_slug()
        output_dir = create_experiment_output_dir(f'{slug}_ab')

    manifest_path = str(manifest.get('_manifest_path') or '')
    usage_context = {
        'operation_id': (
            'compare-variants-manifest-'
            + hashlib.sha256(manifest_path.encode('utf-8')).hexdigest()[:20]
        ),
        'run_id': f'compare-variants-{os.path.basename(output_dir)}',
        'manifest_path': manifest_path,
        'output_dir': os.path.abspath(output_dir),
        'game_root': str(manifest.get('base_dir') or legacy.BASE_DIR or ''),
        'errors': [],
    }
    chunk_results: list[ChunkExperimentResult] = [
        ChunkExperimentResult(
            chunk_key=str(chunk.get('key') or ''),
            file_rel_path=str(chunk.get('file_rel_path') or ''),
            items=copy.deepcopy(chunk.get('items') or []),
        )
        for chunk in chunks
    ]
    experiment_error = ''
    try:
        for variant in variants:
            with variant_batch_settings(variant.get('overrides') or {}) as settings:
                settings = {
                    **settings,
                    'model': default_model,
                    'provider': ab_profile.provider,
                }
                for chunk_index, chunk in enumerate(chunks):
                    chunk_results[chunk_index].variant_results.append(
                        run_variant_for_chunk(
                            chunk,
                            variant_name=variant['name'],
                            settings=settings,
                            model_override=model_override,
                            api_key_index=api_key_index,
                            dry_run=dry_run,
                            sync_runner=sync_runner,
                            usage_context=usage_context,
                            route=ab_route,
                            plan=routing_plan,
                        )
                    )
    except Exception as exc:
        experiment_error = str(exc)
    finally:
        experiment_settings = {
            'manifest_path': manifest.get('_manifest_path', ''),
            'model': default_model,
            'limit': limit,
            'offset': offset,
            'dry_run': dry_run,
            'variants': variants,
            'chunk_keys': [chunk.chunk_key for chunk in chunk_results],
        }
        if experiment_error:
            experiment_settings['experiment_error'] = experiment_error
        output_paths = write_experiment_outputs(
            output_dir,
            manifest_path=manifest.get('_manifest_path', ''),
            variants=variants,
            chunk_results=chunk_results,
            experiment_settings=experiment_settings,
        )

    result = {
        'output_dir': output_dir,
        'chunk_count': len(chunk_results),
        'variant_count': len(variants),
        'dry_run': dry_run,
        **output_paths,
    }
    output_summary = {
        'completion_tokens': 0,
        'completion_tokens_known_requests': 0,
        'reasoning_tokens': 0,
        'reasoning_tokens_known_requests': 0,
        'text_output_tokens': 0,
        'text_output_tokens_known_requests': 0,
        'reasoning_budget_pressure_count': 0,
        'truncated_output_count': 0,
        'output_reason_counts': {},
    }
    for chunk_result in chunk_results:
        for variant_result in chunk_result.variant_results:
            if variant_result.output_diagnostics:
                batch_mod.record_sync_output_summary(
                    output_summary,
                    variant_result.output_diagnostics,
                )
    result['output_summary'] = output_summary
    if experiment_error:
        result['experiment_error'] = experiment_error
    if usage_context.get('errors'):
        result['usage_ledger_errors'] = list(usage_context['errors'])
    return result
