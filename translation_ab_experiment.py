# -*- coding: utf-8 -*-
"""Run synchronous translation A/B experiments without writing game files."""
from __future__ import annotations

import copy
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import story_memory
import translator_runtime as legacy

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
    if not isinstance(payload, list) or not payload:
        raise SystemExit('Variants file must contain at least one variant.')
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
    }


@contextmanager
def variant_batch_settings(overrides: dict):
    batch_mod = _batch()
    base_translator = batch_mod.load_json_file(legacy.TRANSLATOR_CONFIG)
    base_legacy = batch_mod.load_json_file(legacy.CONFIG_FILE)
    merged_translator = deep_merge_dict(base_translator, overrides)
    original_load = batch_mod.load_json_file

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
        batch_mod.load_batch_settings()
        yield summarize_variant_settings()
    finally:
        batch_mod.load_json_file = original_load
        batch_mod._STORY_GRAPH = None
        batch_mod._STORY_GRAPH_PATH = ''
        batch_mod._RAG_STORE = None
        batch_mod._SOURCE_INDEX_STORE = None
        batch_mod.load_batch_settings()


def enrich_chunk_for_current_settings(chunk: dict) -> dict:
    batch_mod = _batch()
    enriched = copy.deepcopy(chunk)
    target_items = enriched.get('items') or []
    context_past = enriched.get('context_past') or []
    context_future = enriched.get('context_future') or []
    file_rel_path = enriched.get('file_rel_path') or ''

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
    translations = {}
    for row in result_items:
        item_id = str(row.get('id') or '')
        translation = _compact_text(row.get('translation'))
        if item_id and translation:
            translations[item_id] = translation
    if not translations:
        return {}, 'No translation items parsed from response.'
    return translations, ''


@dataclass
class VariantRunResult:
    variant_name: str
    settings: dict
    translations: dict[str, str] = field(default_factory=dict)
    usage_metadata: dict = field(default_factory=dict)
    finish_reason: str = ''
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
        lines.append('| Item ID | Source | ' + ' | '.join(variant_names) + ' |')
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
            if result.error:
                meta_bits.append(f'error={result.error}')
            if result.dry_run:
                meta_bits.append('dry_run=true')
            lines.append(f'- **{result.variant_name}**: ' + '; '.join(meta_bits))
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
                                'finish_reason': result.finish_reason,
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


def run_variant_for_chunk(
    chunk: dict,
    *,
    variant_name: str,
    overrides: dict,
    model_name: str,
    api_key_index: int | None = None,
    dry_run: bool = False,
    sync_runner: Callable[..., dict] | None = None,
) -> VariantRunResult:
    with variant_batch_settings(overrides) as settings:
        enriched = enrich_chunk_for_current_settings(chunk)
        request_row = _batch().build_batch_request(enriched)
        request_payload = request_row.get('request') or {}
        if dry_run:
            return VariantRunResult(
                variant_name=variant_name,
                settings=settings,
                dry_run=True,
            )

        runner = sync_runner or _batch().run_sync_request
        try:
            response = runner(request_payload, model_name, api_key_index=api_key_index)
        except Exception as exc:
            return VariantRunResult(
                variant_name=variant_name,
                settings=settings,
                error=str(exc),
            )

        translations, parse_error = extract_translation_map(
            response.get('response_text') or '',
            enriched.get('items') or [],
        )
        return VariantRunResult(
            variant_name=variant_name,
            settings=settings,
            translations=translations,
            usage_metadata=response.get('usage_metadata') or {},
            finish_reason=response.get('finish_reason') or '',
            error=parse_error,
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
    model_name = model_override.strip() or manifest.get('batch_model') or batch_mod.BATCH_MODEL

    if not output_dir:
        slug = batch_mod.guess_project_slug()
        output_dir = create_experiment_output_dir(f'{slug}_ab')

    chunk_results: list[ChunkExperimentResult] = []
    for chunk in chunks:
        chunk_result = ChunkExperimentResult(
            chunk_key=str(chunk.get('key') or ''),
            file_rel_path=str(chunk.get('file_rel_path') or ''),
            items=copy.deepcopy(chunk.get('items') or []),
        )
        for variant in variants:
            chunk_result.variant_results.append(
                run_variant_for_chunk(
                    chunk,
                    variant_name=variant['name'],
                    overrides=variant.get('overrides') or {},
                    model_name=model_name,
                    api_key_index=api_key_index,
                    dry_run=dry_run,
                    sync_runner=sync_runner,
                )
            )
        chunk_results.append(chunk_result)

    experiment_settings = {
        'manifest_path': manifest.get('_manifest_path', ''),
        'model': model_name,
        'limit': limit,
        'offset': offset,
        'dry_run': dry_run,
        'variants': variants,
        'chunk_keys': [chunk.chunk_key for chunk in chunk_results],
    }
    output_paths = write_experiment_outputs(
        output_dir,
        manifest_path=manifest.get('_manifest_path', ''),
        variants=variants,
        chunk_results=chunk_results,
        experiment_settings=experiment_settings,
    )

    return {
        'output_dir': output_dir,
        'chunk_count': len(chunk_results),
        'variant_count': len(variants),
        'dry_run': dry_run,
        **output_paths,
    }