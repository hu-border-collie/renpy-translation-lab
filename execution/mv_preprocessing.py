"""MV source occurrences and reviewed history for the shared context stores.

No Ren'Py scanner or target-translation inference is used here. Store formats,
embedding identities, locks, atomic writes, ranking and budgets remain owned by
the existing context engine. Bootstrap is explicit; query never rebuilds data.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from dataclasses import replace
from datetime import datetime
from typing import Any

from context_engine import advanced_context, embedding_runtime
from context_engine.embedding_backend import check_persisted_store_query_compatibility
from context_engine.embedding_checkpoint import EmbeddingCheckpoint
from context_engine.rag_memory import JsonRagStore, JsonSourceIndexStore
from engine_adapters.coverage import digest_json
from engine_adapters.project_config import read_mv_project_config, _load_json, _unique_object, _reject_constant
from execution.mv_sync import material_identity, live_material_identity, require_material_fresh
from quality_review import translation_core


CONTEXT_SCHEMA_VERSION = 1


def store_paths(config, config_file, *, execution='sync') -> dict[str, Path]:
    """Resolve MV stores with the existing context-storage and asset-path rules."""
    import translator_runtime as runtime
    if execution not in {'sync', 'gemini_batch'}:
        raise ValueError('Unsupported MV context execution strategy.')
    scope = 'batch' if execution == 'gemini_batch' else 'sync'
    settings = read_mv_project_config(config, config_file)
    storage = config.get('context_storage') or {}
    location = runtime._normalize_context_storage_location(storage.get('location'))
    game_dir = runtime._normalize_context_storage_dir_name(storage.get('game_dir_name'))
    target = str(settings.target_root)
    root = (Path(runtime.resolve_project_root(target)) / game_dir if location == 'game'
            else Path(runtime.LOG_DIR))
    paths = {}
    for kind, section, name in (('history', 'rag', 'rag_store'), ('source', 'source_index', 'source_index_store')):
        configured = ((config.get(scope) or {}).get(section) or {}).get('store_dir')
        if configured:
            path = Path(configured)
            paths[kind] = (path if path.is_absolute() else settings.target_root / path).resolve()
        else:
            paths[kind] = root / name
            if location != 'game':
                paths[kind] /= runtime._project_slug_from_base_dir(target)
    if paths['history'].is_relative_to(paths['source']) or paths['source'].is_relative_to(paths['history']):
        raise ValueError('MV source-only and history stores must use separate, non-nested directories.')
    for path in paths.values():
        if (path.is_relative_to(settings.source_root)
                or path.is_relative_to(settings.target_root / 'data')
                or path.is_relative_to(settings.target_root / 'js')):
            raise ValueError('MV context stores cannot write inside frozen source or target data/js.')
    return paths


def context_execution(config, override=''):
    """Use the saved MV execution default, preserving Sync for legacy configs."""
    section = config.get('model_routing') or {}
    defaults = section.get('defaults') if isinstance(section, dict) else None
    if not isinstance(section, dict) or (defaults is not None and not isinstance(defaults, dict)):
        raise ValueError('Invalid MV context routing defaults.')
    strategy = override or (defaults or {}).get('execution_strategy') or 'sync'
    if strategy not in {'sync', 'gemini_batch'}:
        raise ValueError('Unsupported MV context execution strategy.')
    return strategy


def context_status(config, config_file, *, snapshot=None, execution='') -> dict:
    """Report actual MV preparation from bytes without mutable runtime setup or I/O providers."""
    import copy
    from foundation.project_context_settings import apply_project_context_settings_to_config, _coerce_bool
    from providers.model_routing_reader import read_embedding_settings, resolve_runtime_plan
    from providers.litellm_provider_config import custom_provider_registry
    from execution.mv_sync import build_snapshot
    effective = copy.deepcopy(config)
    apply_project_context_settings_to_config(effective, config.get('game_root'))
    snapshot = snapshot or build_snapshot(effective, config_file, require_review=False)
    execution = context_execution(effective, execution)
    scope = 'batch' if execution == 'gemini_batch' else 'sync'
    sync = effective.get(scope) or {}
    enabled = {'source_index': _coerce_bool((sync.get('source_index') or {}).get('enabled'), False),
               'history': _coerce_bool((sync.get('rag') or {}).get('enabled'), False)}
    try:
        profile_id = None
        if 'model_routing' in effective:
            plan = resolve_runtime_plan(effective, execution=execution, strict_override=False)
            profile_id = plan.routes['translation'].profile_id
        settings = read_embedding_settings(effective, execution=execution, profile_id=profile_id)
    except ValueError:
        if any(enabled.values()):
            return {name: {'enabled': value, 'prepared': False, 'status': 'failed' if value else 'disabled',
                          'record_count': None, 'reason': 'embedding_configuration_invalid'} for name, value in enabled.items()}
        settings = None
    if settings is not None and settings.backend == 'openai_compatible' and not settings.endpoint and 'model_routing' not in effective:
        registry = custom_provider_registry((effective.get(scope) or {}).get('custom_litellm_providers'), allow_import=False)
        if settings.provider in registry:
            settings = replace(settings, endpoint=registry[settings.provider].base_url)
    paths = store_paths(effective, config_file, execution=execution)
    return {'source_index': inspect_store(snapshot, paths['source'], settings, 'source',
                enabled=enabled['source_index']),
            'history': inspect_store(snapshot, paths['history'], settings, 'history',
                enabled=enabled['history']),
            'execution_strategy': execution,
            'coverage_review': 'required' if snapshot.review_error else 'current',
            'story_memory': {'status': 'unsupported'}, 'project_analysis': {'status': 'unsupported'}}


def preparation_revision(config, config_file) -> tuple:
    """Cheap GUI invalidation for preparations completed during a status scan."""
    try:
        paths = store_paths(config, config_file, execution=context_execution(config))
        revisions = []
        for kind, names in (('source', ('source_metadata.json', 'source_segments.jsonl')),
                            ('history', ('metadata.json', 'history.jsonl'))):
            for name in names:
                path = paths[kind] / name
                try:
                    stat = path.stat()
                    revisions.append((str(path), stat.st_mtime_ns, stat.st_size))
                except OSError:
                    revisions.append((str(path), None, None))
        return tuple(revisions)
    except (ValueError, TypeError, OSError) as exc:
        return ('invalid_mv_store_path', str(exc))


def source_binding(snapshot) -> dict:
    """Bind context to the frozen source/window/selection, independently of target text."""
    project = snapshot.project
    return {
        'schema_version': CONTEXT_SCHEMA_VERSION, 'engine': project.engine,
        'adapter_version': project.adapter_version,
        'project': project.project_snapshot_fingerprint,
        'support_window_digest': project.support_window_digest,
        'source_files': {doc.file_rel_path: doc.sha256 for doc in project.source_documents},
        'display_fields': list(snapshot.adapter.display_fields),
        'include_files': list(snapshot.include_files), 'include_prefixes': list(snapshot.include_prefixes),
        'occurrence_scope_digest': snapshot.occurrence_scope.digest if snapshot.occurrence_scope else '',
    }


def source_records(snapshot) -> list[dict]:
    """Keep one source record per physical occurrence, including equal strings."""
    binding = digest_json(source_binding(snapshot))
    return [{
        'source_id': row.unit.id, 'occurrence_id': row.occurrence_id,
        'engine': 'rpgmaker_mv', 'file_rel_path': row.unit.file_rel_path,
        'json_path': list(row.locator.locator['json_path']), 'locator': row.locator.to_dict(),
        'source_text': row.unit.source_text,
        'source_checksum': hashlib.sha256(row.unit.source_text.encode('utf-8')).hexdigest(),
        'mv_source_binding': binding,
    } for row in snapshot.selected_occurrences]


def keyword_jobs(snapshot) -> list[dict]:
    """Project actual source occurrences into the existing keyword item contract."""
    jobs = {}
    for row in snapshot.selected_occurrences:
        item = translation_core.legacy_item_from_unit(row.unit)
        item.update(source_locator=row.locator.to_dict(), occurrence_id=row.occurrence_id)
        job = jobs.setdefault(row.unit.file_rel_path, {
            'file_rel_path': row.unit.file_rel_path, 'file_path': row.unit.file_path, 'items': []})
        job['items'].append(item)
    return [{**job, 'task_count': len(job['items'])} for job in jobs.values()]


def history_paths(runtime, extra=()) -> list[Path]:
    """Resolve only explicit reviewed corpus inputs; never scan the target copy."""
    from engine_adapters.project_config import read_project_configuration
    settings = read_mv_project_config(read_project_configuration(runtime.TRANSLATOR_CONFIG), runtime.TRANSLATOR_CONFIG)
    return list(dict.fromkeys([*([settings.reviewed_history_file] if settings.reviewed_history_file else []),
                              *(Path(path).resolve() for path in extra)]))


def read_reviewed_history(snapshot, paths) -> tuple[list[dict], dict]:
    """Accept reviewed MV JSONL records only at current typed source positions.

    Each row pins the complete source file, window and adapter plus exact source
    text. Pending/rejected rows are counted and excluded. Approved malformed or
    stale rows fail closed; approval is supplied by an identified human/agent,
    never manufactured from the current target or a successful structural check.
    """
    canonical = {(row.unit.file_rel_path, tuple(row.locator.locator['json_path'])): row
                 for row in snapshot.selected_occurrences}
    records, seen, inputs = [], set(), {}
    by_id = {record['source_id']: record for record in source_records(snapshot)}
    documents = snapshot.project.document_by_path()
    summary = {'reviewed_records': 0, 'unreviewed_records': 0, 'input_files': inputs}
    if not paths:
        return [], {**summary, 'status': 'missing', 'reason': 'reviewed_history_not_configured'}
    for path in paths:
        try:
            raw = Path(path).read_bytes()
        except OSError:
            return [], {**summary, 'status': 'missing', 'reason': 'reviewed_history_missing'}
        inputs[str(Path(path))] = hashlib.sha256(raw).hexdigest()
        for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            if not isinstance(payload, dict):
                raise ValueError(f'MV history row {number} must be an object.')
            review = payload.get('review')
            if not isinstance(review, dict) or review.get('status') != 'approved':
                summary['unreviewed_records'] += 1
                continue
            reviewer = review.get('reviewer') or {}
            try:
                reviewed_at = datetime.fromisoformat(str(review.get('reviewed_at', '')).replace('Z', '+00:00'))
                dated_review = reviewed_at.tzinfo is not None
            except ValueError:
                dated_review = False
            if (not isinstance(reviewer, dict) or reviewer.get('type') not in {'human', 'agent'}
                    or not isinstance(reviewer.get('id'), str) or not reviewer['id'].strip()
                    or not dated_review
                    or type(payload.get('schema_version')) is not int or payload['schema_version'] != 1
                    or payload.get('engine') != 'rpgmaker_mv'):
                raise ValueError(f'MV history row {number} has invalid review identity or schema.')
            path_parts = payload.get('json_path')
            if (not isinstance(payload.get('file_rel_path'), str)
                    or not isinstance(path_parts, list) or not path_parts
                    or any(not ((type(part) is int and part >= 0) or (type(part) is str and part)) for part in path_parts)):
                raise ValueError(f'MV history row {number} has an invalid typed JSON path.')
            row = canonical.get((payload.get('file_rel_path'), tuple(path_parts)))
            translated = payload.get('translation')
            if (row is None or payload.get('source') != row.unit.source_text
                    or payload.get('source_file_sha256') != documents[row.unit.file_rel_path].sha256
                    or payload.get('support_window_digest') != snapshot.project.support_window_digest
                    or payload.get('adapter_version') != snapshot.project.adapter_version
                    or not isinstance(translated, str) or not translated.strip()
                    or snapshot.adapter.validate_translation(row, translated).status != 'pass'):
                raise ValueError(f'MV history row {number} is stale, outside scope or structurally invalid.')
            if row.unit.id in seen:
                raise ValueError('Duplicate MV reviewed history occurrence; resolve it explicitly.')
            seen.add(row.unit.id)
            base = by_id[row.unit.id]
            records.append({**base, 'memory_id': row.unit.id,
                'translated_text': translated, 'quality_state': 'reviewed', 'record_scope': 'mv_reviewed_history',
                'review': review, 'history_input_sha256': inputs[str(Path(path))]})
    summary.update(reviewed_records=len(records), status='ready', reason='')
    if summary['unreviewed_records'] and not records:
        summary.update(status='missing', reason='no_reviewed_history')
    return records, summary


def require_keyword_material(runtime, manifest) -> None:
    """Reject drift before keyword dispatch/export without converting it to translation."""
    expected = manifest.get('mv_preprocessing_identity')
    if expected is None and runtime.current_project_engine() == 'rpgmaker_mv':
        raise ValueError('MV keyword reports require a matching MV preprocessing manifest.')
    if expected is not None:
        snapshot = runtime.current_mv_sync_snapshot()
        require_material_fresh(expected, material_identity(snapshot))


def keyword_material_checker(runtime, manifest, *, snapshot=None):
    """Audit once, then hash live material before dispatch without extracting events.

    The existing durable MV material check detects changed source/target bytes,
    file sets, support windows, review, adapter and configuration. It retains the
    full freshness contract without repeating occurrence extraction per request.
    """
    from engine_adapters.project_config import read_project_configuration
    from execution.mv_sync import live_material_identity
    expected = manifest.get('mv_preprocessing_identity')
    if expected is None:
        require_keyword_material(runtime, manifest)
        return lambda: None
    snapshot = snapshot or runtime.current_mv_sync_snapshot()
    require_material_fresh(expected, material_identity(snapshot))

    def check():
        config = read_project_configuration(runtime.TRANSLATOR_CONFIG)
        require_material_fresh(expected, live_material_identity(
            snapshot, config, runtime.TRANSLATOR_CONFIG))

    return check


def bootstrap(runtime, kind: str, *, seed_jsonl=(), prune=True, execution='sync', embedding_source=None) -> dict:
    """Resume durable MV batches, publishing only after all inputs pass validation."""
    batch = execution == 'gemini_batch'
    source = (embedding_source or runtime) if batch else runtime
    enabled = ((source.RAG_ENABLED if kind == 'history' else source.SOURCE_INDEX_ENABLED) if batch
               else (runtime.SYNC_RAG_ENABLED if kind == 'history' else runtime.SYNC_SOURCE_INDEX_ENABLED))
    if not enabled:
        return {'enabled': False, 'status': 'disabled', 'prepared': False}
    if batch:
        source._require_batch_embedding_binding_for_retrieval()
    else:
        runtime._require_sync_embedding_binding_for_retrieval()
    snapshot = runtime.current_mv_sync_snapshot()
    settings = source.current_batch_embedding_settings() if batch else runtime.current_sync_embedding_settings()
    summary: dict[str, Any] = {'enabled': True, 'prepared': False, 'status': 'building',
        'scanned': 0, 'embedded': 0, 'upserted': 0, 'reused_embeddings': 0, 'pruned': 0,
        'checkpointed_embeddings': 0, 'resumed_embeddings': 0,
        'files_scanned': len({row.unit.file_rel_path for row in snapshot.selected_occurrences}),
        'scan_scope': 'mv_source_occurrences' if kind == 'source' else 'mv_reviewed_history'}
    records = source_records(snapshot)
    if kind == 'history':
        records, history = read_reviewed_history(snapshot, history_paths(runtime, seed_jsonl))
        summary.update(history)
        if history['status'] != 'ready':
            return summary
    directory = store_paths(_load_json(Path(runtime.TRANSLATOR_CONFIG)), runtime.TRANSLATOR_CONFIG,
                            execution=execution)[kind]
    store = JsonRagStore(directory) if kind == 'history' else JsonSourceIndexStore(directory)
    with store.write_session('mv_bootstrap'):
        return _bootstrap_store(runtime, snapshot, settings, kind, records, summary, store, prune,
                                embed_texts=source.embed_texts)


def _bootstrap_store(runtime, snapshot, settings, kind, records, summary, store, prune, *, embed_texts=None):
    started = phase_started = time.monotonic()
    failure_stage = 'store_load'
    batch_info = {}

    def enter_stage(name):
        nonlocal failure_stage, phase_started
        failure_stage, phase_started = name, time.monotonic()

    summary.update(store_dir=store.store_dir, scanned=len(records))
    try:
        store.load()
        enter_stage('store_identity')
        embedding_runtime.ensure_store_document_identity(store, settings.document_identity(), rebuild=True)
        data_file = Path(store.history_path if kind == 'history' else store.segments_path)
        if not data_file.exists():
            store.rebuild_document_identity(settings.document_identity())
        previous_preparation = store.metadata.get('mv_preparation') or {}
        trusted_previous = (previous_preparation.get('status') == 'ready'
            and previous_preparation.get('data_sha256') == hashlib.sha256(data_file.read_bytes()).hexdigest())
        previous = store.history if kind == 'history' else store.segments
        upsert = store.upsert_history if kind == 'history' else store.upsert_segments
        remove = store.delete_history if kind == 'history' else store.delete_segments
        id_key = 'memory_id' if kind == 'history' else 'source_id'
        summary['history_records_before'] = len(previous)
        expected = material_identity(snapshot)
        enter_stage('checkpoint_load')
        checkpoint = EmbeddingCheckpoint(Path(store.store_dir) / 'mv_embedding_checkpoint.jsonl',
            digest_json({'kind': kind, 'binding': source_binding(snapshot),
                'source_root': snapshot.project.project_root, 'target_root': snapshot.project.localization_root,
                'embedding_identity': settings.document_identity().to_dict(),
                'history_inputs': summary.get('input_files', {}),
                'history_selection': snapshot.reviewed_history_selection if kind == 'history' else None}),
            settings.output_dimension)
        enter_stage('checkpoint_selection')
        ready, pending, preserved = [], [], []
        fingerprints = {record[id_key]: digest_json(record) for record in records}
        for record in records:
            existing = previous.get(record[id_key]) or {}
            cached = checkpoint.get(record[id_key], fingerprints[record[id_key]])
            if (trusted_previous and existing.get('source_checksum') == record['source_checksum']
                    and len(existing.get('embedding') or []) == settings.output_dimension
                    and existing.get('embedding_metadata') == {
                        'embedding_model': settings.model, 'embedding_task_type': settings.native_document_task_type,
                        'embedding_dim': settings.output_dimension}):
                ready.append({**record, 'embedding': existing['embedding'],
                              'embedding_metadata': existing.get('embedding_metadata', {})})
                summary['reused_embeddings'] += 1
                if cached is None:
                    preserved.append({'id': record[id_key], 'fingerprint': fingerprints[record[id_key]],
                                      'embedding': existing['embedding']})
            elif cached is not None:
                ready.append({**record, 'embedding': cached, 'embedding_metadata': {
                    'embedding_model': settings.model, 'embedding_task_type': settings.native_document_task_type,
                    'embedding_dim': settings.output_dimension}})
                summary['reused_embeddings'] += 1
                summary['resumed_embeddings'] += 1
            else:
                pending.append(record)
        enter_stage('checkpoint_save')
        checkpoint.save(preserved)
        summary['checkpointed_embeddings'] = len(ready)
        enter_stage('metadata_write')
        store.set_metadata(mv_preparation={'status': 'building'})
        summary['embedding_pending'] = len(pending)
        print(f"MV context preparation: {len(records)} segments, "
              f"{summary['reused_embeddings']} reused, {len(pending)} pending.", flush=True)
        print(f"MV context checkpoint progress: {summary['checkpointed_embeddings']}/{len(records)} saved, "
              f"{summary['resumed_embeddings']} resumed.", flush=True)
        for start in range(0, len(pending), 16):
            batch = pending[start:start + 16]
            batch_info = {'batch_start': start + 1, 'batch_end': start + len(batch),
                          'batch_total': len(pending)}
            enter_stage('material_validation')
            require_material_fresh(expected, live_material_identity(snapshot,
                _load_json(Path(runtime.TRANSLATOR_CONFIG)), runtime.TRANSLATOR_CONFIG))
            enter_stage('embedding_request')
            print(f"MV context embedding request: {start + 1}-{start + len(batch)}/{len(pending)}.", flush=True)
            vectors = (embed_texts or runtime.embed_texts)([row['source_text'] for row in batch], settings.native_document_task_type)
            enter_stage('embedding_response_validation')
            if len(vectors) != len(batch):
                raise ValueError('MV document embedding count mismatch.')
            completed = []
            for row, vector in zip(batch, vectors):
                if len(vector) != settings.output_dimension:
                    raise ValueError('MV document embedding dimension mismatch.')
                completed.append({**row, 'embedding': vector, 'embedding_metadata': {
                    'embedding_model': settings.model, 'embedding_task_type': settings.native_document_task_type,
                    'embedding_dim': settings.output_dimension}})
            summary['embedded'] += len(batch)
            enter_stage('checkpoint_save')
            checkpoint.save([{'id': row[id_key], 'fingerprint': fingerprints[row[id_key]],
                              'embedding': row['embedding']} for row in completed])
            ready.extend(completed)
            summary['checkpointed_embeddings'] += len(completed)
            print(f"MV context embedding progress: {summary['embedded']}/{len(pending)}", flush=True)
            print(f"MV context checkpoint progress: {summary['checkpointed_embeddings']}/{len(records)} saved, "
                  f"{summary['resumed_embeddings']} resumed.", flush=True)
        batch_info = {}
        enter_stage('material_validation')
        print('MV context validation started.', flush=True)
        require_material_fresh(expected, live_material_identity(snapshot,
            _load_json(Path(runtime.TRANSLATOR_CONFIG)), runtime.TRANSLATOR_CONFIG))
        for path, digest in summary.get('input_files', {}).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('MV reviewed history changed during bootstrap.')
        enter_stage('store_write')
        print('MV context persistence started.', flush=True)
        summary['upserted'] = upsert(ready)
        stale = set(previous) - {row[id_key] for row in ready}
        summary['stale_count'] = len(stale)
        if prune and stale:
            summary['pruned'] = remove(stale)
        binding = source_binding(snapshot)
        preparation = {'status': 'ready', 'kind': kind, 'binding': binding,
            'binding_digest': digest_json(binding), 'record_count': len(records),
            'data_sha256': hashlib.sha256(data_file.read_bytes()).hexdigest(),
            'history_inputs': summary.get('input_files', {}),
            **({'history_selection': snapshot.reviewed_history_selection} if kind == 'history' else {})}
        enter_stage('metadata_write')
        store.set_metadata(mv_preparation=preparation)
        summary.update(prepared=True, status='ready', current_records=len(records),
            history_records_after=len(store.history if kind == 'history' else store.segments))
        print(f'MV context preparation complete: {len(records)} segments.', flush=True)
    except Exception as exc:
        diagnostics = embedding_runtime.public_error_diagnostics(exc)
        now = time.monotonic()
        diagnostics.update(failure_stage=failure_stage, **batch_info,
            phase_elapsed_seconds=round(now - phase_started, 3), elapsed_seconds=round(now - started, 3))
        summary.update(status='failed', error=diagnostics.get('failure_reason') or 'context_prepare_failed', **diagnostics)
        try:
            store.set_metadata(mv_preparation={'status': 'failed',
                'checkpointed_embeddings': summary['checkpointed_embeddings'], **diagnostics})
        except Exception as write_exc:
            # A disk error must not hide the original failure from CLI/JSON/GUI.
            summary['diagnostic_persistence_failed'] = True
            summary['diagnostic_write_exception'] = embedding_runtime.public_error_diagnostics(
                write_exc).get('exception_type', 'unknown')
    else:
        try:
            checkpoint.clear()
        except OSError:
            print('Warning: completed MV embedding checkpoint could not be removed.', flush=True)
    return summary


def inspect_store(snapshot, directory, settings, kind: str, *, enabled=True) -> dict:
    """Read preparation status without writes or provider calls.

    History pins the configured corpus selection separately from its explicit
    seed inputs. A missing selection binding is stale, including seed-only
    stores, rather than inferring approval from the previous input file list.
    """
    result = {'enabled': bool(enabled), 'prepared': False, 'store_dir': str(directory), 'record_count': None}
    if not enabled:
        return {**result, 'status': 'disabled', 'reason': 'disabled'}
    meta = Path(directory) / ('metadata.json' if kind == 'history' else 'source_metadata.json')
    data = Path(directory) / ('history.jsonl' if kind == 'history' else 'source_segments.jsonl')
    if not meta.is_file() or not data.is_file():
        return {**result, 'status': 'missing', 'reason': 'missing_store'}
    try:
        metadata = _load_json(meta)
        state = metadata.get('mv_preparation') or {}
        if state.get('status') != 'ready':
            return {**result, 'status': 'failed' if state.get('status') == 'failed' else 'missing',
                    'reason': 'preparation_not_complete'}
        if state.get('binding_digest') != digest_json(state.get('binding')):
            return {**result, 'status': 'failed', 'reason': 'store_binding_changed'}
        if state.get('kind') != kind or state.get('binding') != source_binding(snapshot):
            return {**result, 'status': 'stale', 'reason': 'source_material_changed'}
        if kind == 'history' and state.get('history_selection') != snapshot.reviewed_history_selection:
            return {**result, 'status': 'stale', 'reason': 'reviewed_history_selection_changed'}
        for path, digest in state.get('history_inputs', {}).items():
            if not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                return {**result, 'status': 'stale', 'reason': 'reviewed_history_changed'}
        compatibility = check_persisted_store_query_compatibility(metadata.get('embedding_identity'), settings.query_identity())
        if not compatibility.compatible:
            return {**result, 'status': 'incompatible', 'reason': 'rebuild_store',
                    'embedding_compatibility': compatibility.to_dict()}
        raw = data.read_bytes()
        if hashlib.sha256(raw).hexdigest() != state.get('data_sha256'):
            return {**result, 'status': 'failed', 'reason': 'store_records_changed'}
        records = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
        count = sum(row.get('mv_source_binding') == state.get('binding_digest') for row in records)
        if count != state.get('record_count'):
            return {**result, 'status': 'failed', 'reason': 'store_records_incomplete'}
        return {**result, 'prepared': True, 'status': 'ready', 'reason': '', 'record_count': count,
                'data_sha256': state['data_sha256'],
                'binding_digest': state['binding_digest'], 'embedding_compatibility': compatibility.to_dict()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {**result, 'status': 'failed', 'reason': 'store_unreadable'}


def retrieve(runtime, snapshot, target_items, kind: str, *, execution='sync', config_file=None):
    """Gate material and embedding identity before using shared ranked retrieval."""
    batch = execution == 'gemini_batch'
    settings = runtime.current_batch_embedding_settings() if batch else runtime.current_sync_embedding_settings()
    enabled = ((runtime.RAG_ENABLED if kind == 'history' else runtime.SOURCE_INDEX_ENABLED) if batch
               else (runtime.SYNC_RAG_ENABLED if kind == 'history' else runtime.SYNC_SOURCE_INDEX_ENABLED))
    config_file = config_file or runtime.TRANSLATOR_CONFIG
    directory = store_paths(_load_json(Path(config_file)), config_file, execution=execution)[kind]
    stats = inspect_store(snapshot, directory, settings, kind, enabled=enabled)
    stats['embedding_provider'] = advanced_context.embedding_provider_diagnostics(settings)
    if not stats['prepared'] or not stats['record_count']:
        if stats['prepared']:
            stats['reason'] = 'prepared_empty_store'
        return [], stats
    try:
        store = JsonRagStore(directory) if kind == 'history' else JsonSourceIndexStore(directory)
        store.load()
        state = store.metadata.get('mv_preparation') or {}
        after_load = inspect_store(snapshot, directory, settings, kind, enabled=enabled)
        if (state.get('status') != 'ready' or state.get('binding_digest') != stats['binding_digest']
                or not after_load['prepared'] or after_load.get('data_sha256') != stats['data_sha256']):
            return [], {**stats, 'prepared': False, 'status': 'stale', 'reason': 'preparation_changed'}
        # A query-only view of the same store excludes retained stale records
        # before ranking/top-k; no second format, index or write path is created.
        view = copy.copy(store)
        attribute = 'history' if kind == 'history' else 'segments'
        setattr(view, attribute, {key: row for key, row in getattr(store, attribute).items()
            if row.get('mv_source_binding') == stats['binding_digest']})
        view._search_cache = {}
        query = advanced_context.build_source_only_query_text(target_items)
        if not query:
            return [], {**stats, 'reason': 'empty_query'}
        expected = material_identity(snapshot)
        require_material_fresh(expected, live_material_identity(snapshot, _load_json(Path(config_file)), config_file))
        vector = runtime.embed_query_text(query) if batch else runtime.embed_sync_query_text(query)
        require_material_fresh(expected, live_material_identity(snapshot, _load_json(Path(config_file)), config_file))
        if kind == 'history':
            hits, diagnostics = advanced_context.retrieve_history_hits_compatible(view, vector, settings.query_identity(),
                top_k=runtime.RAG_TOP_K_HISTORY if batch else runtime.SYNC_RAG_TOP_K_HISTORY,
                min_similarity=runtime.RAG_MIN_SIMILARITY if batch else runtime.SYNC_RAG_MIN_SIMILARITY,
                char_limit=runtime.RAG_HISTORY_CHAR_LIMIT if batch else runtime.SYNC_RAG_HISTORY_CHAR_LIMIT,
                query_text=query)
        else:
            hits, diagnostics = advanced_context.retrieve_source_hits_compatible(view, vector, settings.query_identity(),
                top_k=runtime.SOURCE_INDEX_TOP_K if batch else runtime.SYNC_SOURCE_INDEX_TOP_K,
                min_similarity=runtime.SOURCE_INDEX_MIN_SIMILARITY if batch else runtime.SYNC_SOURCE_INDEX_MIN_SIMILARITY,
                char_limit=runtime.SOURCE_INDEX_CHAR_LIMIT if batch else runtime.SYNC_SOURCE_INDEX_CHAR_LIMIT,
                query_text=query,
                embedding_model=settings.model, embedding_task_type=settings.native_document_task_type,
                embedding_dim=settings.output_dimension,
                char_budget=runtime.get_source_index_char_budget() if batch else runtime.get_sync_source_index_char_budget())
        return hits, {**stats, **diagnostics, 'reason': '' if hits else 'zero_hits'}
    except Exception as exc:
        return [], {**stats, 'status': 'failed', 'reason': 'retrieval_failed',
                    **embedding_runtime.public_error_diagnostics(exc)}
